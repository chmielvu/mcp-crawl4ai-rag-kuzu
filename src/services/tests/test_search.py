"""Tests for search service."""

from unittest.mock import AsyncMock, Mock

import pytest

from crawl4ai_mcp.models import SearchRequest, SearchResult
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.services.search import SearchService


@pytest.fixture
def mock_embedding_service():
    service = Mock(spec=EmbeddingService)
    service.create_embedding = AsyncMock(return_value=[0.1] * 1024)
    return service


@pytest.fixture
def search_service(mock_db_connection, test_settings, mock_embedding_service):
    return SearchService(mock_db_connection, test_settings, mock_embedding_service)


@pytest.fixture
def mock_search_rows():
    return [
        {
            "content": "First result content",
            "url": "https://example.com/1",
            "source_id": "example.com",
            "chunk_number": 1,
            "similarity": 0.9,
            "metadata": {"title": "First"},
        },
        {
            "content": "Second result content",
            "url": "https://example.com/2",
            "source_id": "example.com",
            "chunk_number": 2,
            "similarity": 0.8,
            "metadata": {"title": "Second", "category": "tutorial"},
        },
    ]


@pytest.fixture
def mock_code_rows():
    return [
        {
            "content": 'def example():\n    return "Hello"',
            "url": "https://example.com/code1",
            "source_id": "example.com",
            "chunk_number": 1,
            "similarity": 0.85,
            "metadata": {"language": "python", "summary": "Example function"},
        }
    ]


@pytest.mark.asyncio
async def test_search_documents_success(search_service, mock_search_rows) -> None:
    search_service.backend.search_documents_by_vector = Mock(return_value=mock_search_rows)
    results = await search_service.search_documents("test query", match_count=10)
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].content == "First result content"
    assert results[0].similarity_score == 0.9


@pytest.mark.asyncio
async def test_search_documents_with_filters(search_service, mock_search_rows) -> None:
    search_service.backend.search_documents_by_vector = Mock(return_value=mock_search_rows)
    results = await search_service.search_documents(
        "test query",
        match_count=10,
        filter_metadata={"category": "tutorial"},
    )
    assert len(results) == 1
    assert results[0].url == "https://example.com/2"


@pytest.mark.asyncio
async def test_search_documents_hybrid_search(search_service, mock_search_rows) -> None:
    text_rows = [
        {
            "content": "Keyword match",
            "url": "https://example.com/3",
            "source_id": "example.com",
            "chunk_number": 3,
            "similarity": 1.2,
            "metadata": {"title": "Keyword"},
        }
    ]
    search_service.backend.search_documents_by_vector = Mock(return_value=mock_search_rows)
    search_service.backend.search_documents_by_text = Mock(return_value=text_rows)
    results = await search_service.search_documents(
        "test query",
        match_count=2,
        use_hybrid_search=True,
    )
    assert len(results) == 2
    search_service.backend.search_documents_by_text.assert_called_once()


@pytest.mark.asyncio
async def test_search_code_examples_success(search_service, mock_code_rows) -> None:
    search_service.backend.search_code_by_vector = Mock(return_value=mock_code_rows)
    results = await search_service.search_code_examples("example function", match_count=5)
    assert len(results) == 1
    assert results[0]["content"].startswith("def example")


@pytest.mark.asyncio
async def test_search_code_examples_with_language_filter(
    search_service, mock_code_rows
) -> None:
    search_service.backend.search_code_by_vector = Mock(return_value=mock_code_rows)
    results = await search_service.search_code_examples(
        "example function",
        language="python",
        match_count=5,
    )
    assert len(results) == 1
    assert results[0]["metadata"]["language"] == "python"


@pytest.mark.asyncio
async def test_perform_search_documents_only(search_service) -> None:
    search_service.search_documents = AsyncMock(
        return_value=[
            SearchResult(
                content="First",
                url="url1",
                source="src",
                chunk_number=1,
                similarity_score=0.9,
            ),
            SearchResult(
                content="Second",
                url="url2",
                source="src",
                chunk_number=2,
                similarity_score=0.7,
            ),
        ]
    )
    response = await search_service.perform_search(
        SearchRequest(query="test query", num_results=5, semantic_threshold=0.75)
    )
    assert response.success is True
    assert len(response.results) == 1
    assert response.results[0].content == "First"


@pytest.mark.asyncio
async def test_perform_search_with_code_examples(search_service, test_settings) -> None:
    test_settings.use_agentic_rag = True
    search_service.search_documents = AsyncMock(
        return_value=[
            SearchResult(
                content="Doc",
                url="doc-url",
                source="src",
                chunk_number=1,
                similarity_score=0.9,
            )
        ]
    )
    search_service.search_code_examples = AsyncMock(
        return_value=[
            {
                "content": "def example(): pass",
                "url": "code-url",
                "source_id": "src",
                "chunk_number": 1,
                "similarity": 0.85,
                "metadata": {"language": "python", "summary": "Example"},
            }
        ]
    )
    response = await search_service.perform_search(
        SearchRequest(query="test query", num_results=5),
        include_code_examples=True,
    )
    assert response.success is True
    assert len(response.results) == 2
    assert any(result.metadata.get("type") == "code_example" for result in response.results)


@pytest.mark.asyncio
async def test_perform_search_error_handling(search_service) -> None:
    search_service.search_documents = AsyncMock(side_effect=Exception("Search failed"))
    response = await search_service.perform_search(SearchRequest(query="test query"))
    assert response.success is False
    assert response.total_results == 0


@pytest.mark.asyncio
async def test_rerank_results_flashrank(search_service) -> None:
    reranking_model = Mock()
    reranking_model.rerank.return_value = [{"id": 1, "score": 0.95}, {"id": 0, "score": 0.85}]
    results = [
        SearchResult(content="First", url="url1", source="src", chunk_number=1, similarity_score=0.7),
        SearchResult(content="Second", url="url2", source="src", chunk_number=2, similarity_score=0.8),
    ]
    reranked = await search_service.rerank_results(
        query="test query",
        results=results,
        reranking_model=reranking_model,
        threshold=0.5,
    )
    assert len(reranked) == 2
    assert reranked[0].content == "Second"
    assert reranked[0].rerank_score == 0.95


@pytest.mark.asyncio
async def test_rerank_results_legacy_predict(search_service) -> None:
    class LegacyPredictModel:
        def predict(self, pairs):
            assert len(pairs) == 2
            return [0.9, 0.2]

    reranking_model = LegacyPredictModel()
    results = [
        SearchResult(content="First", url="url1", source="src", chunk_number=1, similarity_score=0.7),
        SearchResult(content="Second", url="url2", source="src", chunk_number=2, similarity_score=0.8),
    ]
    reranked = await search_service.rerank_results(
        query="test query",
        results=results,
        reranking_model=reranking_model,
        threshold=0.5,
    )
    assert len(reranked) == 1
    assert reranked[0].content == "First"


@pytest.mark.asyncio
async def test_rerank_results_error_handling(search_service) -> None:
    reranking_model = Mock()
    reranking_model.rerank.side_effect = Exception("Reranking failed")
    original_results = [
        SearchResult(content="test", url="url", source="src", chunk_number=1, similarity_score=0.5)
    ]
    results = await search_service.rerank_results(
        query="test",
        results=original_results,
        reranking_model=reranking_model,
    )
    assert results == original_results
