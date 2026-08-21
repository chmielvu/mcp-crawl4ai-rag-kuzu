"""Tests for FalkorSearchBackend vector, full-text, and code search retrieval."""

from unittest.mock import AsyncMock, Mock

import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.services.contracts import SearchHit
from crawl4ai_mcp.services.falkor_search_backend import (
    FalkorSearchBackend,
    _candidate_limit,
    _convert_vector_score,
    escape_redisearch_query,
)


def test_escape_redisearch_query() -> None:
    assert escape_redisearch_query("") == ""
    assert escape_redisearch_query("simple") == "simple"
    escaped = escape_redisearch_query("hello-world (test) [tag] @author")
    assert r"\-" in escaped
    assert r"\(" in escaped
    assert r"\)" in escaped
    assert r"\[" in escaped
    assert r"\]" in escaped
    assert r"\@" in escaped


def test_candidate_limit() -> None:
    assert _candidate_limit(5, has_filter=False) == 5
    assert _candidate_limit(5, has_filter=True) == 50
    assert _candidate_limit(10, has_filter=True) == 80


def test_convert_vector_score() -> None:
    assert _convert_vector_score(0.2) == pytest.approx(0.8)
    assert _convert_vector_score(0.0) == 1.0
    assert _convert_vector_score(1.5) == 0.0
    assert _convert_vector_score(-0.2) == 1.0
    assert _convert_vector_score("invalid") == 0.0


@pytest.mark.asyncio
async def test_search_chunks_by_vector_parses_results(test_settings: Settings) -> None:
    mock_graph = Mock()

    mock_result = Mock()
    mock_result.header = [
        ["chunk_id"],
        ["page_id"],
        ["site_id"],
        ["content"],
        ["url"],
        ["source"],
        ["chunk_number"],
        ["score"],
        ["content_type"],
        ["language"],
        ["metadata_json"],
        ["page_title"],
    ]
    mock_result.result_set = [
        [
            "chunk-1",
            "page-1",
            "example.com",
            "Chunk content text",
            "https://example.com/page",
            "example.com",
            0,
            0.15,  # distance -> similarity = 0.85
            "text",
            "en",
            '{"key": "val"}',
            "Page Title",
        ]
    ]
    mock_graph.ro_query = AsyncMock(return_value=mock_result)

    backend = FalkorSearchBackend(mock_graph, settings=test_settings)
    hits = await backend.search_chunks_by_vector(
        embedding=[0.1] * 384, limit=5, site_id="example.com"
    )

    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, SearchHit)
    assert hit.chunk_id == "chunk-1"
    assert hit.page_id == "page-1"
    assert hit.site_id == "example.com"
    assert hit.url == "https://example.com/page"
    assert hit.content == "Chunk content text"
    assert hit.similarity_score == pytest.approx(0.85)
    assert hit.content_type == "text"
    assert hit.chunk_number == 0



@pytest.mark.asyncio
async def test_search_chunks_by_vector_propagates_graph_errors(
    test_settings: Settings,
) -> None:
    mock_graph = Mock()
    mock_graph.ro_query = AsyncMock(side_effect=RuntimeError("connection lost"))
    backend = FalkorSearchBackend(mock_graph, settings=test_settings)

    with pytest.raises(RuntimeError, match="connection lost"):
        await backend.search_chunks_by_vector([0.1] * 384, limit=5)

@pytest.mark.asyncio
async def test_search_chunks_by_text_parses_results(test_settings: Settings) -> None:
    mock_graph = Mock()

    mock_result = Mock()
    mock_result.header = [
        ["chunk_id"],
        ["page_id"],
        ["site_id"],
        ["content"],
        ["url"],
        ["source"],
        ["chunk_number"],
        ["score"],
        ["content_type"],
        ["language"],
        ["metadata_json"],
        ["page_title"],
    ]
    mock_result.result_set = [
        [
            "chunk-text-1",
            "page-1",
            "example.com",
            "Full text query matching text",
            "https://example.com/page",
            "example.com",
            0,
            2.5,
            "text",
            "en",
            "{}",
            "Page Title",
        ]
    ]
    mock_graph.ro_query = AsyncMock(return_value=mock_result)

    backend = FalkorSearchBackend(mock_graph, settings=test_settings)
    hits = await backend.search_chunks_by_text("matching text", limit=3)

    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk-text-1"
    assert hits[0].similarity_score == 2.5


@pytest.mark.asyncio
async def test_search_code_chunks(test_settings: Settings) -> None:
    mock_graph = Mock()

    mock_result = Mock()
    mock_result.header = [
        ["chunk_id"],
        ["page_id"],
        ["site_id"],
        ["content"],
        ["url"],
        ["source"],
        ["chunk_number"],
        ["score"],
        ["content_type"],
        ["language"],
        ["metadata_json"],
        ["page_title"],
    ]
    mock_result.result_set = [
        [
            "chunk-code-1",
            "page-1",
            "example.com",
            "def foo():\n    return 42",
            "https://example.com/code",
            "example.com",
            1,
            0.1,  # similarity = 0.9
            "code",
            "python",
            '{"language": "python"}',
            "Page Title",
        ]
    ]
    mock_graph.ro_query = AsyncMock(return_value=mock_result)

    backend = FalkorSearchBackend(mock_graph, settings=test_settings)
    hits = await backend.search_code_chunks(
        embedding=[0.1] * 384, limit=5, site_id="example.com", language="python"
    )

    assert len(hits) == 1
    assert hits[0].content_type == "code"
    assert hits[0].language == "python"
    assert hits[0].similarity_score == pytest.approx(0.9)
