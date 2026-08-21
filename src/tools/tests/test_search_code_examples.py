"""Tests for search_code_examples tool returning typed CodeSearchResponse."""

from unittest.mock import Mock

import pytest

from crawl4ai_mcp.conftest import FakeSearchBackend
from crawl4ai_mcp.models import CodeSearchResponse, SearchHit
from crawl4ai_mcp.tools.search_code_examples import search_code_examples


@pytest.mark.asyncio
async def test_search_code_examples_success(
    mock_mcp_context: Mock,
    fake_search_backend: FakeSearchBackend,
) -> None:
    code_hit = SearchHit(
        chunk_id="code-chunk-1",
        page_id="page-1",
        site_id="example.com",
        content="```python\ndef get_user(id: int):\n    return db.query(id)\n```",
        url="https://example.com/docs/api",
        source="example.com",
        chunk_number=2,
        similarity_score=0.95,
        rerank_score=None,
        content_type="code",
        language="python",
        metadata={},
        provenance=[],
    )
    fake_search_backend.code_hits = [code_hit]

    response = await search_code_examples(
        mock_mcp_context,
        query="get user by id",
        source_id="example.com",
        language="python",
        match_count=5,
    )

    assert isinstance(response, CodeSearchResponse)
    assert response.success is True
    assert response.query == "get user by id"
    assert response.language == "python"
    assert response.total_results == 1
    assert len(response.results) == 1
    assert response.results[0].chunk_id == "code-chunk-1"
    assert response.results[0].content_type == "code"



@pytest.mark.asyncio
async def test_search_code_examples_reports_reranking_only_when_available(
    mock_mcp_context: Mock,
) -> None:
    mock_mcp_context.request_context.lifespan_context.reranker = None
    response = await search_code_examples(
        mock_mcp_context,
        query="get user by id",
        use_reranking=True,
    )

    assert response.success is True
    assert response.reranking_applied is False

@pytest.mark.asyncio
async def test_search_code_examples_uninitialized_context() -> None:
    empty_context = Mock()
    empty_context.request_context = None

    response = await search_code_examples(empty_context, query="test query")
    assert isinstance(response, CodeSearchResponse)
    assert response.success is False
    assert "FastMCP server lifespan is not initialized" in (response.message or "")
