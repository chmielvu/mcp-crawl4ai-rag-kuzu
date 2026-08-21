"""Tests for FalkorStore ontology ingestion, Cypher execution, and site listing."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.services.contracts import (
    ChunkPayload,
    CrawlIngestion,
    GlinerEntity,
    GlinerRelation,
    PageExtraction,
    PagePayload,
    RemoteLink,
    SitePayload,
)
from crawl4ai_mcp.services.falkor_store import (
    FalkorStore,
    canonicalize_url,
    deterministic_chunk_id,
    deterministic_content_hash,
    deterministic_entity_id,
    deterministic_page_id,
)


def test_canonicalize_url() -> None:
    assert canonicalize_url("") == ""
    assert canonicalize_url("https://example.com:443/docs/") == "https://example.com/docs"
    assert canonicalize_url("http://example.com:80/docs/page") == "http://example.com/docs/page"
    assert canonicalize_url("https://EXAMPLE.COM/docs") == "https://example.com/docs"
    assert canonicalize_url("https://example.com/") == "https://example.com/"


def test_deterministic_id_helpers() -> None:
    page_id1 = deterministic_page_id("https://example.com/page1")
    page_id2 = deterministic_page_id("https://example.com/page1")
    assert page_id1 == page_id2
    assert len(page_id1) == 16

    chunk_id = deterministic_chunk_id(page_id1, 0)
    assert chunk_id == f"{page_id1}::text::0"
    assert deterministic_chunk_id(page_id1, 1, "code") == f"{page_id1}::code::1"

    content_hash = deterministic_content_hash("some text content")
    assert len(content_hash) == 64

    entity_id1 = deterministic_entity_id("FastMCP", "technology")
    entity_id2 = deterministic_entity_id("fastmcp", "TECHNOLOGY")
    assert entity_id1 == entity_id2
    assert len(entity_id1) == 16


@pytest.mark.asyncio
async def test_ingest_crawl_executes_cypher_transactions(test_settings: Settings) -> None:
    mock_db = Mock()
    mock_graph = Mock()
    mock_graph.query = AsyncMock()
    mock_db.select_graph = Mock(return_value=mock_graph)

    now = datetime.now(timezone.utc)
    page_payload = PagePayload(
        page_id="page1",
        url="https://example.com/docs",
        canonical_url="https://example.com/docs",
        title="Docs",
        status_code=200,
        content_type="text/markdown",
        language="en",
        content_hash=deterministic_content_hash("doc text"),
        depth=0,
        crawled_at=now,
        chunks=[
            ChunkPayload(
                chunk_id="page1::text::0",
                text="Chunk text content",
                index=0,
                heading_path="# Title",
                start_char=0,
                end_char=18,
                content_type="text",
                language="en",
                embedding=[0.1] * 384,
                extractions=[
                    PageExtraction(
                        extraction_class="technology",
                        extraction_text="FastMCP",
                        start_char=0,
                        end_char=7,
                    )
                ],
            )
        ],
        links=[
            RemoteLink(href="https://example.com/other", text="Other", internal=True)
        ],
    )

    site_payload = SitePayload(
        site_id="example.com",
        domain="example.com",
        root_url="https://example.com",
        title="Example",
        summary="Example site summary",
        gliner_metadata={"enabled": True},
        entities=[
            GlinerEntity(
                text="FastMCP",
                label="technology",
                score=0.95,
                start=0,
                end=7,
                embedding=[0.2] * 384,
            )
        ],
        relations=[
            GlinerRelation(
                source="FastMCP",
                target="Python",
                relation="implements",
                score=0.9,
                embedding=[0.3] * 384,
            )
        ],
    )

    ingestion = CrawlIngestion(
        run_id="run-1",
        root_url="https://example.com",
        max_depth=3,
        started_at=now,
        finished_at=now,
        site=site_payload,
        pages=[page_payload],
    )

    store = FalkorStore(settings=test_settings, db=mock_db)
    result = await store.ingest_crawl(ingestion)

    assert result.success is True
    assert result.pages == 1
    assert result.chunks == 1
    assert result.entities == 2  # one grounded chunk entity plus one site entity
    assert result.relations == 1
    assert result.links == 1
    assert mock_graph.query.await_count >= 5


@pytest.mark.asyncio
async def test_get_available_sites_parses_query_results(test_settings: Settings) -> None:
    mock_db = Mock()
    mock_graph = Mock()

    # FalkorDB query result mock
    mock_result = Mock()
    mock_result.header = [
        ["s.site_id"],
        ["s.domain"],
        ["s.title"],
        ["s.summary"],
        ["page_count"],
        ["chunk_count"],
        ["last_crawled"],
        ["s.gliner_metadata"],
    ]
    mock_result.result_set = [
        [
            "example.com",
            "example.com",
            "Example Site",
            "Site summary",
            5,
            20,
            "2026-08-21T00:00:00+00:00",
            '{"enabled": true}',
        ]
    ]
    mock_graph.query = AsyncMock(return_value=mock_result)
    mock_db.select_graph = Mock(return_value=mock_graph)

    store = FalkorStore(settings=test_settings, db=mock_db)
    sites = await store.get_available_sites()

    assert len(sites) == 1
    site = sites[0]
    assert site.site_id == "example.com"
    assert site.domain == "example.com"
    assert site.title == "Example Site"
    assert site.page_count == 5
    assert site.chunk_count == 20
    assert site.gliner_metadata == {"enabled": True}


@pytest.mark.asyncio
async def test_get_available_sites_propagates_graph_errors(test_settings: Settings) -> None:
    mock_db = Mock()
    mock_graph = Mock()
    mock_graph.query = AsyncMock(side_effect=RuntimeError("connection lost"))
    mock_db.select_graph = Mock(return_value=mock_graph)

    store = FalkorStore(settings=test_settings, db=mock_db)

    with pytest.raises(RuntimeError, match="connection lost"):
        await store.get_available_sites()
