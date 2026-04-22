"""Tests for smart_crawl_url tool."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from crawl4ai_mcp.tools.smart_crawl_url import smart_crawl_url


@pytest.fixture
def mock_context():
    context = Mock()
    context.request_context.lifespan_context = SimpleNamespace(
        crawler=Mock(),
        db_connection=Mock(),
        settings=SimpleNamespace(
            default_chunk_size=5000,
            use_contextual_embeddings=False,
            use_agentic_rag=False,
            mistral_api_key="test-key",
        ),
    )
    return context


@pytest.fixture
def mock_services():
    with (
        patch("crawl4ai_mcp.tools.smart_crawl_url.EmbeddingService") as MockEmbedding,
        patch("crawl4ai_mcp.tools.smart_crawl_url.DatabaseService") as MockDatabase,
        patch("crawl4ai_mcp.tools.smart_crawl_url.CrawlingService") as MockCrawling,
        patch("crawl4ai_mcp.tools.smart_crawl_url.TextProcessor") as MockTextProcessor,
    ):
        embedding_instance = Mock()
        embedding_instance.create_embedding = AsyncMock(return_value=[0.1] * 1024)
        MockEmbedding.return_value = embedding_instance

        database_instance = Mock()
        database_instance.update_source_info = AsyncMock(return_value={"success": True})
        database_instance.add_documents = AsyncMock(return_value={"success": True, "count": 3})
        database_instance.add_code_examples = AsyncMock(return_value={"success": True, "count": 2})
        MockDatabase.return_value = database_instance

        crawling_instance = Mock()
        crawling_instance.is_txt = Mock(return_value=False)
        crawling_instance.is_sitemap = Mock(return_value=False)
        crawling_instance.crawl_markdown_file = AsyncMock(return_value=[])
        crawling_instance.parse_sitemap = Mock(return_value=[])
        crawling_instance.crawl_batch = AsyncMock(return_value=[])
        crawling_instance.crawl_recursive_internal_links = AsyncMock(
            return_value=[
                {"url": "https://example.com", "markdown": "# Page 1"},
                {"url": "https://example.com/page2", "markdown": "# Page 2"},
            ]
        )
        crawling_instance.extract_source_summary = AsyncMock(return_value="Test source summary")
        crawling_instance.extract_code_blocks = Mock(return_value=[])
        crawling_instance.generate_code_example_summary = AsyncMock(return_value="Code summary")
        MockCrawling.return_value = crawling_instance

        text_processor_instance = Mock()
        text_processor_instance.smart_chunk_markdown = Mock(return_value=["Chunk 1", "Chunk 2"])
        text_processor_instance.extract_section_info = Mock(
            return_value={"headers": "# Test", "char_count": 100, "word_count": 20}
        )
        text_processor_instance.generate_contextual_embedding = AsyncMock(
            return_value=("Contextual chunk", True)
        )
        MockTextProcessor.return_value = text_processor_instance

        yield {
            "database": database_instance,
            "crawling": crawling_instance,
            "text_processor": text_processor_instance,
        }


@pytest.mark.asyncio
async def test_smart_crawl_recursive(mock_context, mock_services) -> None:
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com", max_depth=3, max_concurrent=5))
    assert result["success"] is True
    assert result["crawl_type"] == "recursive"
    assert result["urls_processed"] == 2
    assert result["total_chunks_created"] == 6


@pytest.mark.asyncio
async def test_smart_crawl_txt_file(mock_context, mock_services) -> None:
    mock_services["crawling"].is_txt.return_value = True
    mock_services["crawling"].crawl_markdown_file.return_value = [
        {"url": "https://example.com/file.txt", "markdown": "Text content"}
    ]
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com/file.txt"))
    assert result["success"] is True
    assert result["crawl_type"] == "txt file"


@pytest.mark.asyncio
async def test_smart_crawl_sitemap(mock_context, mock_services) -> None:
    mock_services["crawling"].is_sitemap.return_value = True
    mock_services["crawling"].parse_sitemap.return_value = [
        "https://example.com/page1",
        "https://example.com/page2",
    ]
    mock_services["crawling"].crawl_batch.return_value = [
        {"url": "https://example.com/page1", "markdown": "Page 1"},
        {"url": "https://example.com/page2", "markdown": "Page 2"},
    ]
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com/sitemap.xml"))
    assert result["success"] is True
    assert result["crawl_type"] == "sitemap"


@pytest.mark.asyncio
async def test_smart_crawl_custom_chunk_size(mock_context, mock_services) -> None:
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com", chunk_size=3000))
    assert result["success"] is True
    for call in mock_services["text_processor"].smart_chunk_markdown.call_args_list:
        assert call.kwargs.get("chunk_size") == 3000


@pytest.mark.asyncio
async def test_smart_crawl_with_code_extraction(mock_context, mock_services) -> None:
    mock_context.request_context.lifespan_context.settings.use_agentic_rag = True
    mock_services["crawling"].extract_code_blocks.return_value = [
        {
            "code": "def test(): pass",
            "language": "python",
            "context_before": "Before",
            "context_after": "After",
        }
    ]
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com"))
    assert result["success"] is True
    assert result["total_code_examples"] == 4


@pytest.mark.asyncio
async def test_smart_crawl_no_results(mock_context, mock_services) -> None:
    mock_services["crawling"].crawl_recursive_internal_links.return_value = []
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com"))
    assert result["success"] is False


@pytest.mark.asyncio
async def test_smart_crawl_exception_handling(mock_context, mock_services) -> None:
    mock_services["crawling"].crawl_recursive_internal_links.side_effect = Exception("Network error")
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com"))
    assert result["success"] is False
    assert "Network error" in result["error"]


@pytest.mark.asyncio
async def test_smart_crawl_partial_failure(mock_context, mock_services) -> None:
    call_count = 0

    def chunk_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ["Chunk 1", "Chunk 2"]
        raise Exception("Processing error")

    mock_services["text_processor"].smart_chunk_markdown.side_effect = chunk_side_effect
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com"))
    assert result["success"] is True
    assert result["urls_processed"] == 1


@pytest.mark.asyncio
async def test_metadata_timestamp_format(mock_context, mock_services) -> None:
    captured_metadata = []

    async def capture_add_documents(**kwargs):
        captured_metadata.extend(kwargs.get("metadatas", []))
        return {"success": True, "count": len(kwargs.get("contents", []))}

    mock_services["database"].add_documents = capture_add_documents
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com"))
    assert result["success"] is True
    assert captured_metadata
    for metadata in captured_metadata:
        assert "crawl_time" in metadata
        assert "crawl_type" in metadata


@pytest.mark.asyncio
async def test_sitemap_with_empty_url_list(mock_context, mock_services) -> None:
    mock_services["crawling"].is_sitemap.return_value = True
    mock_services["crawling"].parse_sitemap.return_value = []
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com/sitemap.xml"))
    assert result["success"] is False


@pytest.mark.asyncio
async def test_sitemap_crawl_batch_failure(mock_context, mock_services) -> None:
    mock_services["crawling"].is_sitemap.return_value = True
    mock_services["crawling"].parse_sitemap.return_value = [
        "https://example.com/page1",
        "https://example.com/page2",
    ]
    mock_services["crawling"].crawl_batch.return_value = []
    result = json.loads(await smart_crawl_url(mock_context, "https://example.com/sitemap.xml"))
    assert result["success"] is False
