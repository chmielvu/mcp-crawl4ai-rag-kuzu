"""Tests for text processing utilities."""

from unittest.mock import AsyncMock, patch

import pytest

from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.utilities.text_processing import TextProcessor


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = AsyncMock(spec=EmbeddingService)
    service.chat_complete = AsyncMock(return_value="This chunk discusses the main topic.")
    return service


@pytest.fixture
def text_processor(test_settings, mock_embedding_service):
    """Create a text processor with mocked dependencies."""
    return TextProcessor(test_settings, mock_embedding_service)


class TestTextChunking:
    """Test text chunking functionality."""

    def test_smart_chunk_markdown_simple(self, text_processor) -> None:
        text = "A" * 10000
        chunks = text_processor.smart_chunk_markdown(text, chunk_size=5000)
        assert len(chunks) == 2
        assert len(chunks[0]) == 5000
        assert len(chunks[1]) == 5000

    def test_smart_chunk_markdown_respects_code_blocks(self, text_processor) -> None:
        text = (
            "Some text before\n\n"
            + "A" * 4000
            + "\n\n```python\ncode block\n```\n\n"
            + "B" * 2000
        )
        chunks = text_processor.smart_chunk_markdown(text, chunk_size=5000)
        assert len(chunks) >= 2
        full_text = " ".join(chunks)
        assert "```python" in full_text
        assert "code block" in full_text

    def test_smart_chunk_markdown_respects_paragraphs(self, text_processor) -> None:
        para1 = "First paragraph. " * 200
        para2 = "Second paragraph. " * 100
        para3 = "Third paragraph. " * 100
        text = f"{para1}\n\n{para2}\n\n{para3}"
        chunks = text_processor.smart_chunk_markdown(text, chunk_size=5000)
        assert len(chunks) >= 2
        full_text = " ".join(chunks)
        assert "First paragraph" in full_text
        assert "Second paragraph" in full_text
        assert "Third paragraph" in full_text

    def test_smart_chunk_markdown_respects_sentences(self, text_processor) -> None:
        sentence = "This is a sentence. "
        text = sentence * 300
        chunks = text_processor.smart_chunk_markdown(text, chunk_size=5000)
        assert len(chunks) == 2
        assert chunks[0].endswith(".")
        assert not chunks[0].endswith(". ")

    def test_smart_chunk_markdown_empty_text(self, text_processor) -> None:
        assert text_processor.smart_chunk_markdown("", chunk_size=5000) == []

    def test_smart_chunk_markdown_small_text(self, text_processor) -> None:
        text = "Small text"
        chunks = text_processor.smart_chunk_markdown(text, chunk_size=5000)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestSectionExtraction:
    """Test section information extraction."""

    def test_extract_section_info_with_headers(self, text_processor) -> None:
        chunk = """# Main Header

Some content here.

## Sub Header

More content.

### Sub-sub Header

Even more content."""
        info = text_processor.extract_section_info(chunk)
        assert "# Main Header" in info["headers"]
        assert "## Sub Header" in info["headers"]
        assert "### Sub-sub Header" in info["headers"]
        assert info["char_count"] == len(chunk)
        assert info["word_count"] == len(chunk.split())

    def test_extract_section_info_no_headers(self, text_processor) -> None:
        chunk = "Just some plain text without any headers."
        info = text_processor.extract_section_info(chunk)
        assert info["headers"] == ""
        assert info["char_count"] == len(chunk)
        assert info["word_count"] == len(chunk.split())

    def test_extract_section_info_empty_chunk(self, text_processor) -> None:
        info = text_processor.extract_section_info("")
        assert info["headers"] == ""
        assert info["char_count"] == 0
        assert info["word_count"] == 0


class TestContextualEmbedding:
    """Test contextual embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_contextual_embedding_success(
        self,
        text_processor,
        mock_embedding_service,
    ) -> None:
        mock_embedding_service.chat_complete = AsyncMock(
            return_value="This chunk discusses the main topic."
        )
        full_doc = "This is a document about AI. It covers many topics."
        chunk = "Machine learning is a subset of AI."
        contextual_text, success = await text_processor.generate_contextual_embedding(
            full_doc,
            chunk,
        )
        assert success is True
        assert "This chunk discusses the main topic." in contextual_text
        assert chunk in contextual_text
        assert "---" in contextual_text

    @pytest.mark.asyncio
    async def test_generate_contextual_embedding_error(
        self,
        text_processor,
        mock_embedding_service,
    ) -> None:
        mock_embedding_service.chat_complete = AsyncMock(side_effect=Exception("API error"))
        chunk = "Some chunk content"
        contextual_text, success = await text_processor.generate_contextual_embedding(
            "Full doc",
            chunk,
        )
        assert success is False
        assert contextual_text == chunk

    @pytest.mark.asyncio
    async def test_process_chunk_with_context(
        self,
        text_processor,
        mock_embedding_service,
    ) -> None:
        mock_embedding_service.chat_complete = AsyncMock(return_value="Context for chunk")
        contextual_text, success = await text_processor.process_chunk_with_context(
            url="https://example.com",
            content="Chunk content",
            full_document="Full document",
        )
        assert success is True
        assert "Context for chunk" in contextual_text


class TestCodeExampleProcessing:
    """Test code example processing."""

    @pytest.mark.asyncio
    async def test_process_code_example(self, text_processor) -> None:
        with patch("crawl4ai_mcp.services.crawling.CrawlingService") as mock_service:
            mock_service.return_value.generate_code_example_summary = AsyncMock(
                return_value="Code example summary"
            )
            summary = await text_processor.process_code_example(
                code="def hello(): pass",
                context_before="Here's a function:",
                context_after="That's it.",
            )
        assert summary == "Code example summary"
