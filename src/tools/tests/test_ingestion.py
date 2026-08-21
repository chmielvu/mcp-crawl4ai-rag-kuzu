"""Tests for ingestion helpers, deterministic offsets, batching, and GLiNER/LangExtract contracts."""

from datetime import datetime, timezone

import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.conftest import (
    FakeChatGenerator,
    FakeEmbedding,
    FakeGliner,
    FakeGraphStore,
    FakeLangExtract,
)
from crawl4ai_mcp.services.contracts import (
    ChunkPayload,
    CrawlDocument,
    PageExtraction,
    PagePayload,
)
from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.tools._ingestion import (
    build_crawl_ingestion,
    deterministic_chunk_id,
    deterministic_page_id,
    deterministic_site_id,
    embed_texts_in_batches,
    ingest_crawl_documents,
    map_extractions_to_chunks,
)
from crawl4ai_mcp.utilities.text_processing import TextProcessor


def test_deterministic_identifiers() -> None:
    assert deterministic_site_id("https://docs.example.com/sub/page") == "docs.example.com"
    assert deterministic_site_id("example.com") == "example.com"

    page_id1 = deterministic_page_id("https://example.com/docs")
    page_id2 = deterministic_page_id("https://example.com/docs")
    assert page_id1 == page_id2
    assert len(page_id1) == 16
    assert deterministic_page_id("https://EXAMPLE.com/docs/") == page_id1

    chunk_id = deterministic_chunk_id(page_id1, 0, content_type="text")
    assert chunk_id == f"{page_id1}::text::0"


@pytest.mark.asyncio
async def test_embed_texts_in_batches_enforces_limit_and_dimensions(fake_embedding: FakeEmbedding) -> None:
    texts = [f"Text item {i}" for i in range(70)]
    vectors = await embed_texts_in_batches(texts, fake_embedding, batch_size=32)

    assert len(vectors) == 70
    assert len(fake_embedding.embed_passages_calls) == 3
    assert len(fake_embedding.embed_passages_calls[0]) == 32
    assert len(fake_embedding.embed_passages_calls[1]) == 32
    assert len(fake_embedding.embed_passages_calls[2]) == 6
    for vec in vectors:
        assert len(vec) == 384


@pytest.mark.asyncio
async def test_embed_texts_in_batches_rejects_dimension_mismatch() -> None:
    wrong_dim_embedding = FakeEmbedding(dimension=128)
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await embed_texts_in_batches(["sample text"], wrong_dim_embedding)


def test_map_extractions_to_chunks_with_relative_offsets() -> None:
    # Page text: "01234567890123456789" (20 chars)
    # Chunk 1: span [0, 10]
    # Chunk 2: span [10, 20]
    chunk1 = ChunkPayload(
        chunk_id="c1",
        text="0123456789",
        index=0,
        start_char=0,
        end_char=10,
        content_type="text",
        embedding=[0.1] * 384,
        extractions=[],
    )
    chunk2 = ChunkPayload(
        chunk_id="c2",
        text="0123456789",
        index=1,
        start_char=10,
        end_char=20,
        content_type="text",
        embedding=[0.1] * 384,
        extractions=[],
    )

    ext1 = PageExtraction(
        extraction_class="technology",
        extraction_text="2345",
        start_char=2,
        end_char=6,
    )
    ext2 = PageExtraction(
        extraction_class="version",
        extraction_text="2345",
        start_char=12,
        end_char=16,
    )
    ext3 = PageExtraction(
        extraction_class="product",
        extraction_text="outside",
        start_char=50,
        end_char=60,
    )

    map_extractions_to_chunks([ext1, ext2, ext3], [chunk1, chunk2])

    assert len(chunk1.extractions) == 1
    assert chunk1.extractions[0].extraction_class == "technology"
    assert chunk1.extractions[0].start_char == 2
    assert chunk1.extractions[0].end_char == 6

    assert len(chunk2.extractions) == 1
    assert chunk2.extractions[0].extraction_class == "version"
    assert chunk2.extractions[0].start_char == 2
    assert chunk2.extractions[0].end_char == 6



@pytest.mark.asyncio
async def test_gliner_receives_prose_chunks_without_code_chunks(
    fake_embedding: FakeEmbedding,
    fake_chat: FakeChatGenerator,
    fake_gliner: FakeGliner,
    test_settings: Settings,
) -> None:
    now = datetime.now(timezone.utc)
    page = PagePayload(
        page_id="page-1",
        url="https://example.com/docs",
        canonical_url="https://example.com/docs",
        content_hash="hash",
        crawled_at=now,
        chunks=[
            ChunkPayload(
                chunk_id="page-1::text::0",
                text="Prose documentation.",
                index=0,
                start_char=0,
                end_char=20,
                content_type="text",
                embedding=[0.0] * 384,
            ),
            ChunkPayload(
                chunk_id="page-1::code::1",
                text="def hidden_code(): pass",
                index=1,
                start_char=20,
                end_char=44,
                content_type="code",
                embedding=[0.0] * 384,
            ),
        ],
    )

    await build_crawl_ingestion(
        root_url="https://example.com",
        pages=[page],
        gliner_port=fake_gliner,
        chat_generator=fake_chat,
        embedding_port=fake_embedding,
        settings=test_settings,
    )

    assert fake_gliner.extract_calls[0]["text"] == "Prose documentation."

@pytest.mark.asyncio
async def test_gliner_failure_blocks_graph_write(
    fake_embedding: FakeEmbedding,
    fake_chat: FakeChatGenerator,
    fake_gliner: FakeGliner,
    fake_graph_store: FakeGraphStore,
    test_settings: Settings,
) -> None:
    text_processor = TextProcessor(test_settings, fake_chat)
    crawling_service = CrawlingService(None, fake_chat, test_settings)  # type: ignore[arg-type]

    fake_gliner.should_fail = True

    doc = CrawlDocument(
        url="https://example.com/page",
        success=True,
        markdown="# Main Title\n\nSome important technical documentation text.",
    )

    with pytest.raises(RuntimeError, match="GLiNER extraction failure simulated"):
        await ingest_crawl_documents(
            documents=[doc],
            root_url="https://example.com",
            embedding_port=fake_embedding,
            text_processor=text_processor,
            graph_store=fake_graph_store,
            crawling_service=crawling_service,
            gliner_port=fake_gliner,
            chat_generator=fake_chat,
            settings=test_settings,
        )

    assert len(fake_graph_store.ingested_payloads) == 0


@pytest.mark.asyncio
async def test_successful_ingestion_with_gliner_and_langextract(
    fake_embedding: FakeEmbedding,
    fake_chat: FakeChatGenerator,
    fake_gliner: FakeGliner,
    fake_lang_extract: FakeLangExtract,
    fake_graph_store: FakeGraphStore,
    test_settings: Settings,
) -> None:
    test_settings.use_langextract_metadata = True
    text_processor = TextProcessor(test_settings, fake_chat)
    crawling_service = CrawlingService(None, fake_chat, test_settings)  # type: ignore[arg-type]

    doc = CrawlDocument(
        url="https://example.com/docs",
        success=True,
        markdown="# Title\n\nFastMCP is a python framework for building tools.\n\n```python\ndef test(): pass\n```",
    )

    result = await ingest_crawl_documents(
        documents=[doc],
        root_url="https://example.com",
        embedding_port=fake_embedding,
        text_processor=text_processor,
        graph_store=fake_graph_store,
        crawling_service=crawling_service,
        gliner_port=fake_gliner,
        chat_generator=fake_chat,
        lang_extract=fake_lang_extract,
        settings=test_settings,
    )

    assert result.success is True
    assert len(fake_graph_store.ingested_payloads) == 1
    payload = fake_graph_store.ingested_payloads[0]
    assert payload.site.site_id == "example.com"
    assert len(payload.site.entities) >= 1
    assert len(payload.site.relations) >= 1
    assert len(payload.pages) == 1
    assert len(payload.pages[0].chunks) >= 1
