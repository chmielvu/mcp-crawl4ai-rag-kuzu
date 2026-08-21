"""Tests for text processing utilities and TextProcessor chunking/context generation."""

import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.conftest import FakeChatGenerator
from crawl4ai_mcp.utilities.text_processing import TextProcessor


@pytest.fixture
def text_processor(test_settings: Settings, fake_chat: FakeChatGenerator) -> TextProcessor:
    """Create a text processor with fake chat generator."""
    return TextProcessor(test_settings, fake_chat)


class TestTextChunking:
    """Test text chunking functionality."""

    def test_smart_chunk_markdown_simple(self, text_processor: TextProcessor) -> None:
        text = "A" * 10000
        chunks = text_processor.smart_chunk_markdown(text, chunk_size=5000)
        assert len(chunks) == 2
        assert len(chunks[0]) == 5000
        assert len(chunks[1]) == 5000

    def test_smart_chunk_markdown_respects_code_blocks(self, text_processor: TextProcessor) -> None:
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

    def test_smart_chunk_markdown_respects_paragraphs(self, text_processor: TextProcessor) -> None:
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

    def test_smart_chunk_markdown_respects_sentences(self, text_processor: TextProcessor) -> None:
        sentence = "This is a sentence. "
        text = sentence * 300
        chunks = text_processor.smart_chunk_markdown(text, chunk_size=5000)
        assert len(chunks) == 2
        assert chunks[0].endswith(".")
        assert not chunks[0].endswith(". ")

    def test_smart_chunk_markdown_empty_text(self, text_processor: TextProcessor) -> None:
        assert text_processor.smart_chunk_markdown("", chunk_size=5000) == []

    def test_smart_chunk_markdown_small_text(self, text_processor: TextProcessor) -> None:
        text = "Small text"
        chunks = text_processor.smart_chunk_markdown(text, chunk_size=5000)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestSmartChunkWithOffsets:
    """Test chunking with character offsets and heading hierarchy."""

    def test_smart_chunk_with_offsets(self, text_processor: TextProcessor) -> None:
        markdown = (
            "# Main Heading\n\n"
            "This is the introductory paragraph.\n\n"
            "## Sub Heading\n\n"
            "This is subcontent under sub heading."
        )
        chunks = text_processor.smart_chunk_with_offsets(markdown, chunk_size=500)
        assert len(chunks) == 1
        c = chunks[0]
        assert c["index"] == 0
        assert c["start_char"] == 0
        assert c["end_char"] == len(markdown.strip())
        assert c["heading_path"] == "Main Heading > Sub Heading"
        assert "Main Heading" in c["section_info"]["headers"]

    def test_smart_chunk_with_offsets_empty(self, text_processor: TextProcessor) -> None:
        assert text_processor.smart_chunk_with_offsets("") == []


class TestSectionExtraction:
    """Test section information extraction."""

    def test_extract_section_info_with_headers(self, text_processor: TextProcessor) -> None:
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

    def test_extract_section_info_no_headers(self, text_processor: TextProcessor) -> None:
        chunk = "Just some plain text without any headers."
        info = text_processor.extract_section_info(chunk)
        assert info["headers"] == ""
        assert info["char_count"] == len(chunk)
        assert info["word_count"] == len(chunk.split())

    def test_extract_section_info_empty_chunk(self, text_processor: TextProcessor) -> None:
        info = text_processor.extract_section_info("")
        assert info["headers"] == ""
        assert info["char_count"] == 0
        assert info["word_count"] == 0


class TestContextualEmbedding:
    """Test contextual embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_contextual_embedding_success(
        self,
        text_processor: TextProcessor,
        fake_chat: FakeChatGenerator,
    ) -> None:
        fake_chat.chat_response = "Context summarizing the document."
        full_doc = "This is a document about AI. It covers many topics."
        chunk = "Machine learning is a subset of AI."
        contextual_text, success = await text_processor.generate_contextual_embedding(
            full_doc,
            chunk,
        )
        assert success is True
        assert "Context summarizing the document." in contextual_text
        assert chunk in contextual_text
        assert "---" in contextual_text

    @pytest.mark.asyncio
    async def test_generate_contextual_embedding_error(
        self,
        text_processor: TextProcessor,
        fake_chat: FakeChatGenerator,
    ) -> None:
        fake_chat.should_fail = True
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
        text_processor: TextProcessor,
        fake_chat: FakeChatGenerator,
    ) -> None:
        fake_chat.chat_response = "Context for chunk"
        contextual_text, success = await text_processor.process_chunk_with_context(
            url="https://example.com",
            content="Chunk content",
            full_document="Full document",
        )
        assert success is True
        assert "Context for chunk" in contextual_text

    @pytest.mark.asyncio
    async def test_process_code_example(
        self,
        text_processor: TextProcessor,
        fake_chat: FakeChatGenerator,
    ) -> None:
        fake_chat.code_summary_response = "Code example summary"
        summary = await text_processor.process_code_example(
            code="def hello(): pass",
            context_before="Here's a function:",
            context_after="That's it.",
        )
        assert summary == "Code example summary"
