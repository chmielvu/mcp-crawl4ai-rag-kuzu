"""Tests for crawl_single_page tool."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from crawl4ai_mcp.tools.crawl_single_page import crawl_single_page


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
        patch("crawl4ai_mcp.tools.crawl_single_page.EmbeddingService") as MockEmbedding,
        patch("crawl4ai_mcp.tools.crawl_single_page.DatabaseService") as MockDatabase,
        patch("crawl4ai_mcp.tools.crawl_single_page.CrawlingService") as MockCrawling,
        patch("crawl4ai_mcp.tools.crawl_single_page.TextProcessor") as MockTextProcessor,
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
        crawling_instance.crawl_batch = AsyncMock(
            return_value=[{"url": "https://example.com", "markdown": "# Test Content\n\nSome content here."}]
        )
        crawling_instance.extract_source_summary = AsyncMock(return_value="Test source summary")
        crawling_instance.extract_code_blocks = Mock(return_value=[])
        crawling_instance.generate_code_example_summary = AsyncMock(return_value="Code summary")
        MockCrawling.return_value = crawling_instance

        text_processor_instance = Mock()
        text_processor_instance.smart_chunk_markdown = Mock(return_value=["Chunk 1", "Chunk 2", "Chunk 3"])
        text_processor_instance.extract_section_info = Mock(
            return_value={"headers": "# Test", "char_count": 100, "word_count": 20}
        )
        text_processor_instance.generate_contextual_embedding = AsyncMock(
            return_value=("Contextual chunk", True)
        )
        MockTextProcessor.return_value = text_processor_instance

        yield {
            "embedding": embedding_instance,
            "database": database_instance,
            "crawling": crawling_instance,
            "text_processor": text_processor_instance,
        }


@pytest.mark.asyncio
async def test_crawl_single_page_success(mock_context, mock_services) -> None:
    result = json.loads(await crawl_single_page(mock_context, "https://example.com"))
    assert result["success"] is True
    assert result["chunks_created"] == 3
    assert result["code_examples_created"] == 0
    mock_services["database"].add_documents.assert_called_once()


@pytest.mark.asyncio
async def test_crawl_single_page_with_code_examples(mock_context, mock_services) -> None:
    mock_context.request_context.lifespan_context.settings.use_agentic_rag = True
    mock_services["crawling"].extract_code_blocks.return_value = [
        {
            "code": "def hello(): pass",
            "language": "python",
            "context_before": "Before",
            "context_after": "After",
        }
    ]
    result = json.loads(await crawl_single_page(mock_context, "https://example.com"))
    assert result["success"] is True
    assert result["code_examples_created"] == 2
    mock_services["database"].add_code_examples.assert_called_once()


@pytest.mark.asyncio
async def test_crawl_single_page_with_contextual_embeddings(
    mock_context, mock_services
) -> None:
    mock_context.request_context.lifespan_context.settings.use_contextual_embeddings = True
    result = json.loads(await crawl_single_page(mock_context, "https://example.com"))
    assert result["success"] is True
    assert mock_services["text_processor"].generate_contextual_embedding.call_count == 3


@pytest.mark.asyncio
async def test_crawl_single_page_crawl_failure(mock_context, mock_services) -> None:
    mock_services["crawling"].crawl_batch.return_value = []
    result = json.loads(await crawl_single_page(mock_context, "https://example.com"))
    assert result["success"] is False
    assert "Failed to crawl the URL" in result["error"]


@pytest.mark.asyncio
async def test_crawl_single_page_exception_handling(mock_context, mock_services) -> None:
    mock_services["crawling"].crawl_batch.side_effect = Exception("Network error")
    result = json.loads(await crawl_single_page(mock_context, "https://example.com"))
    assert result["success"] is False
    assert "Network error" in result["error"]


@pytest.mark.asyncio
async def test_metadata_timestamp_format(mock_context, mock_services) -> None:
    captured_metadata = []

    async def capture_add_documents(**kwargs):
        captured_metadata.extend(kwargs.get("metadatas", []))
        return {"success": True, "count": len(kwargs.get("contents", []))}

    mock_services["database"].add_documents = capture_add_documents
    result = json.loads(await crawl_single_page(mock_context, "https://example.com"))
    assert result["success"] is True
    assert captured_metadata
    for metadata in captured_metadata:
        assert metadata["crawl_type"] == "single_page"
        assert "crawl_time" in metadata
