"""Tests for SearchService vector, hybrid RRF, reranking, and code search."""

import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.conftest import FakeEmbedding, FakeReranker, FakeSearchBackend
from crawl4ai_mcp.services.contracts import RerankResult, SearchHit
from crawl4ai_mcp.services.search import SearchService, _rrf_fuse


@pytest.fixture
def search_service(
    fake_search_backend: FakeSearchBackend,
    fake_embedding: FakeEmbedding,
    fake_reranker: FakeReranker,
    test_settings: Settings,
) -> SearchService:
    return SearchService(
        backend=fake_search_backend,
        embeddings=fake_embedding,
        reranker=fake_reranker,
        settings=test_settings,
    )


def _make_hit(chunk_id: str, score: float, text: str = "hit text", content_type: str = "text") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        page_id="page1",
        site_id="example.com",
        content=text,
        url="https://example.com/p",
        source="example.com",
        chunk_number=0,
        similarity_score=score,
        rerank_score=None,
        content_type=content_type,
        language="en",
        metadata={},
        provenance=[],
    )


@pytest.mark.asyncio
async def test_search_documents_semantic(
    search_service: SearchService,
    fake_search_backend: FakeSearchBackend,
    fake_embedding: FakeEmbedding,
) -> None:
    hit1 = _make_hit("c1", 0.9)
    hit2 = _make_hit("c2", 0.7)
    fake_search_backend.vector_hits = [hit1, hit2]

    results = await search_service.search_documents("query test", match_count=5, use_hybrid=False)

    assert len(fake_embedding.embed_query_calls) == 1
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[1].chunk_id == "c2"


@pytest.mark.asyncio
async def test_search_documents_hybrid_rrf(
    search_service: SearchService,
    fake_search_backend: FakeSearchBackend,
) -> None:
    hit1_v = _make_hit("c1", 0.95, text="c1 text")
    hit2_v = _make_hit("c2", 0.80, text="c2 text")
    hit1_t = _make_hit("c1", 5.0, text="c1 text")
    hit3_t = _make_hit("c3", 4.0, text="c3 text")

    fake_search_backend.vector_hits = [hit1_v, hit2_v]
    fake_search_backend.text_hits = [hit1_t, hit3_t]

    results = await search_service.search_documents("hybrid query", match_count=5, use_hybrid=True)

    assert len(results) == 3
    # c1 was rank 0 in both vector and text -> highest RRF score
    assert results[0].chunk_id == "c1"


@pytest.mark.asyncio
async def test_search_documents_with_reranking(
    search_service: SearchService,
    fake_search_backend: FakeSearchBackend,
    fake_reranker: FakeReranker,
) -> None:
    hit1 = _make_hit("c1", 0.9, text="first hit")
    hit2 = _make_hit("c2", 0.8, text="second hit")
    fake_search_backend.vector_hits = [hit1, hit2]

    fake_reranker.custom_results = [
        RerankResult(id=1, text="second hit", score=0.99),
        RerankResult(id=0, text="first hit", score=0.45),
    ]

    # Test rerank_hits method directly
    reranked = await search_service.rerank_hits("query", [hit1, hit2])
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rerank_score == 0.99
    assert reranked[1].chunk_id == "c1"
    assert reranked[1].rerank_score == 0.45


@pytest.mark.asyncio
async def test_search_code_examples(
    search_service: SearchService,
    fake_search_backend: FakeSearchBackend,
) -> None:
    code_hit = _make_hit("code1", 0.92, text="def foo(): pass", content_type="code")
    code_hit.language = "python"
    fake_search_backend.code_hits = [code_hit]

    results = await search_service.search_code_examples(
        "foo function", site_id="example.com", language="python", match_count=3
    )

    assert len(results) == 1
    assert results[0].chunk_id == "code1"
    assert results[0].content_type == "code"
    assert len(fake_search_backend.code_calls) == 1
    assert fake_search_backend.code_calls[0]["language"] == "python"


@pytest.mark.asyncio
async def test_perform_rag_query(
    search_service: SearchService,
    fake_search_backend: FakeSearchBackend,
) -> None:
    hit1 = _make_hit("c1", 0.9, text="rag hit")
    fake_search_backend.vector_hits = [hit1]

    results = await search_service.perform_rag_query("test rag", match_count=5, use_hybrid=False)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert len(fake_search_backend.provenance_calls) == 1


def test_rrf_fuse_scoring() -> None:
    hit1 = _make_hit("a", 1.0)
    hit2 = _make_hit("b", 1.0)
    fused = _rrf_fuse([[hit1, hit2], [hit1]], limit=5)
    assert len(fused) == 2
    assert fused[0].chunk_id == "a"
    assert fused[1].chunk_id == "b"
