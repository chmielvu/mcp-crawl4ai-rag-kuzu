"""Tests for perform_rag_query tool returning typed RagSearchResponse."""

from unittest.mock import Mock

import pytest

from crawl4ai_mcp.conftest import FakeSearchBackend
from crawl4ai_mcp.models import RagSearchResponse, SearchHit
from crawl4ai_mcp.tools.perform_rag_query import perform_rag_query


@pytest.mark.asyncio
async def test_perform_rag_query_success(
    mock_mcp_context: Mock,
    fake_search_backend: FakeSearchBackend,
) -> None:
    hit = SearchHit(
        chunk_id="chunk-rag-1",
        page_id="page-1",
        site_id="example.com",
        content="Content discussing Model Context Protocol.",
        url="https://example.com/docs",
        source="example.com",
        chunk_number=0,
        similarity_score=0.92,
        rerank_score=None,
        content_type="text",
        language="en",
        metadata={},
        provenance=[],
    )
    fake_search_backend.vector_hits = [hit]

    response = await perform_rag_query(
        mock_mcp_context,
        query="what is MCP?",
        source="example.com",
        match_count=5,
        use_hybrid=False,
    )

    assert isinstance(response, RagSearchResponse)
    assert response.success is True
    assert response.query == "what is MCP?"
    assert response.search_type == "semantic"
    assert response.total_results == 1
    assert len(response.results) == 1
    assert response.results[0].chunk_id == "chunk-rag-1"



@pytest.mark.asyncio
async def test_perform_rag_query_reports_reranking_only_when_available(
    mock_mcp_context: Mock,
) -> None:
    mock_mcp_context.request_context.lifespan_context.reranker = None
    response = await perform_rag_query(
        mock_mcp_context,
        query="what is MCP?",
        use_reranking=True,
    )

    assert response.success is True
    assert response.reranking_applied is False


@pytest.mark.asyncio
async def test_perform_rag_query_hybrid(
    mock_mcp_context: Mock,
    fake_search_backend: FakeSearchBackend,
) -> None:
    hit1 = SearchHit(
        chunk_id="chunk-1",
        page_id="page-1",
        site_id="example.com",
        content="Vector match text",
        url="https://example.com/docs",
        source="example.com",
        chunk_number=0,
        similarity_score=0.90,
        rerank_score=None,
        content_type="text",
        language="en",
        metadata={},
        provenance=[],
    )
    hit2 = SearchHit(
        chunk_id="chunk-2",
        page_id="page-1",
        site_id="example.com",
        content="Text match text",
        url="https://example.com/docs",
        source="example.com",
        chunk_number=1,
        similarity_score=3.0,
        rerank_score=None,
        content_type="text",
        language="en",
        metadata={},
        provenance=[],
    )
    fake_search_backend.vector_hits = [hit1]
    fake_search_backend.text_hits = [hit2]

    response = await perform_rag_query(
        mock_mcp_context,
        query="hybrid search query",
        use_hybrid=True,
    )

    assert isinstance(response, RagSearchResponse)
    assert response.success is True
    assert response.search_type == "hybrid"
    assert response.total_results == 2


@pytest.mark.asyncio
async def test_perform_rag_query_uninitialized_context() -> None:
    empty_context = Mock()
    empty_context.request_context = None

    response = await perform_rag_query(empty_context, query="test query")
    assert isinstance(response, RagSearchResponse)
    assert response.success is False
    assert "FastMCP server lifespan is not initialized" in (response.message or "")
