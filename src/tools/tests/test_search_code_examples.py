"""Tests for search_code_examples tool."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from crawl4ai_mcp.tools.search_code_examples import search_code_examples


@pytest.fixture
def mock_context():
    context = Mock()
    context.request_context.lifespan_context = SimpleNamespace(
        db_connection=Mock(),
        reranking_model=None,
        settings=SimpleNamespace(
            use_agentic_rag=True,
            use_reranking=False,
            default_rerank_threshold=0.3,
        ),
    )
    return context


@pytest.fixture
def mock_search_service():
    with patch("crawl4ai_mcp.tools.search_code_examples.SearchService") as MockSearch:
        service = Mock()
        service.search_code_examples = AsyncMock(
            return_value=[
                {
                    "content": 'def example():\n    return "Hello"',
                    "url": "https://example.com/code1",
                    "source_id": "example.com",
                    "chunk_number": 1,
                    "similarity": 0.85,
                    "metadata": {"language": "python", "summary": "Example function"},
                }
            ]
        )
        MockSearch.return_value = service
        yield service


@pytest.mark.asyncio
async def test_search_code_examples_success(mock_context, mock_search_service) -> None:
    result = json.loads(await search_code_examples(mock_context, "example function"))
    assert result["success"] is True
    assert result["total_results"] == 1
    assert result["results"][0]["language"] == "python"


@pytest.mark.asyncio
async def test_search_code_examples_disabled(mock_context) -> None:
    mock_context.request_context.lifespan_context.settings.use_agentic_rag = False
    result = json.loads(await search_code_examples(mock_context, "example function"))
    assert result["success"] is False
    assert "not enabled" in result["error"]


@pytest.mark.asyncio
async def test_search_code_examples_with_reranking(mock_context, mock_search_service) -> None:
    mock_context.request_context.lifespan_context.settings.use_reranking = True
    mock_context.request_context.lifespan_context.reranking_model = Mock()
    with patch("crawl4ai_mcp.tools.search_code_examples.Reranker") as MockReranker:
        reranker = Mock()
        reranker.rerank_results.return_value = [
            {
                "content": 'def example():\n    return "Hello"',
                "url": "https://example.com/code1",
                "source_id": "example.com",
                "chunk_number": 1,
                "similarity": 0.85,
                "metadata": {"language": "python", "summary": "Example function"},
                "rerank_score": 0.95,
            }
        ]
        reranker.filter_by_threshold.return_value = reranker.rerank_results.return_value
        MockReranker.return_value = reranker
        result = json.loads(await search_code_examples(mock_context, "example function"))
    assert result["success"] is True
    assert result["results"][0]["rerank_score"] == 0.95


@pytest.mark.asyncio
async def test_search_code_examples_empty_results(mock_context, mock_search_service) -> None:
    mock_search_service.search_code_examples.return_value = []
    result = json.loads(await search_code_examples(mock_context, "obscure"))
    assert result["success"] is True
    assert result["total_results"] == 0


@pytest.mark.asyncio
async def test_search_code_examples_exception(mock_context, mock_search_service) -> None:
    mock_search_service.search_code_examples.side_effect = Exception("Search failed")
    result = json.loads(await search_code_examples(mock_context, "example function"))
    assert result["success"] is False
    assert "Search failed" in result["error"]
