"""Tests for perform_rag_query tool."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from crawl4ai_mcp.models import SearchResponse, SearchResult, SearchType
from crawl4ai_mcp.tools.perform_rag_query import perform_rag_query


@pytest.fixture
def mock_context():
    context = Mock()
    context.request_context.lifespan_context = SimpleNamespace(
        db_connection=Mock(),
        reranking_model=None,
        settings=SimpleNamespace(
            use_hybrid_search=False,
            use_reranking=False,
            default_semantic_threshold=0.5,
            default_rerank_threshold=0.3,
        ),
    )
    return context


@pytest.fixture
def mock_search_service():
    with patch("crawl4ai_mcp.tools.perform_rag_query.SearchService") as MockSearch:
        search_instance = Mock()
        search_instance.perform_search = AsyncMock(
            return_value=SearchResponse(
                success=True,
                results=[
                    SearchResult(
                        content="First result about machine learning",
                        url="https://example.com/ml",
                        source="example.com",
                        chunk_number=1,
                        similarity_score=0.9,
                        metadata={"title": "ML Guide"},
                    ),
                    SearchResult(
                        content="Second result about deep learning",
                        url="https://example.com/dl",
                        source="example.com",
                        chunk_number=2,
                        similarity_score=0.8,
                        metadata={"title": "DL Guide"},
                    ),
                ],
                total_results=2,
                search_type=SearchType.SEMANTIC,
            )
        )
        MockSearch.return_value = search_instance
        yield search_instance


@pytest.mark.asyncio
async def test_perform_rag_query_success(mock_context, mock_search_service) -> None:
    result = json.loads(
        await perform_rag_query(
            mock_context,
            query="machine learning algorithms",
            source="example.com",
            match_count=3,
        )
    )
    assert result["success"] is True
    assert result["search_type"] == "semantic"
    assert result["total_results"] == 2


@pytest.mark.asyncio
async def test_perform_rag_query_hybrid_search(mock_context, mock_search_service) -> None:
    mock_context.request_context.lifespan_context.settings.use_hybrid_search = True
    result = json.loads(await perform_rag_query(mock_context, query="test query"))
    assert result["success"] is True
    assert result["search_type"] == "hybrid"


@pytest.mark.asyncio
async def test_perform_rag_query_with_reranking(mock_context, mock_search_service) -> None:
    mock_context.request_context.lifespan_context.settings.use_reranking = True
    mock_context.request_context.lifespan_context.reranking_model = Mock()
    with patch("crawl4ai_mcp.tools.perform_rag_query.Reranker") as MockReranker:
        reranker = Mock()
        reranked_results = [
            {
                "content": "Second result about deep learning",
                "url": "https://example.com/dl",
                "source": "example.com",
                "chunk_number": 2,
                "similarity_score": 0.8,
                "metadata": {"title": "DL Guide"},
                "rerank_score": 0.95,
            }
        ]
        reranker.rerank_results.return_value = reranked_results
        reranker.filter_by_threshold.return_value = reranked_results
        MockReranker.return_value = reranker
        result = json.loads(await perform_rag_query(mock_context, query="test query"))
    assert result["success"] is True
    assert result["reranking_applied"] is True
    assert result["results"][0]["rerank_score"] == 0.95


@pytest.mark.asyncio
async def test_perform_rag_query_search_failure(mock_context, mock_search_service) -> None:
    mock_search_service.perform_search.return_value = SearchResponse(
        success=False,
        results=[],
        total_results=0,
        search_type=SearchType.SEMANTIC,
        error="Search index unavailable",
    )
    result = json.loads(await perform_rag_query(mock_context, query="test query"))
    assert result["success"] is False
    assert "Search index unavailable" in result["error"]


@pytest.mark.asyncio
async def test_perform_rag_query_exception(mock_context, mock_search_service) -> None:
    mock_search_service.perform_search.side_effect = Exception("Database connection failed")
    result = json.loads(await perform_rag_query(mock_context, query="test query"))
    assert result["success"] is False
    assert "Database connection failed" in result["error"]
