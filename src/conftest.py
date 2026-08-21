"""Shared pytest fixtures and typed fake port implementations for unit tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import Mock

import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.models import CrawlContext
from crawl4ai_mcp.services.contracts import (
    ChatGeneratorPort,
    CrawlDocument,
    CrawlIngestion,
    CrawlerPort,
    EmbeddingPort,
    GlinerEntity,
    GlinerExtraction,
    GlinerPort,
    GlinerRelation,
    GraphOperationResult,
    GraphStorePort,
    LangExtractPort,
    PageExtraction,
    RemoteLink,
    RerankResult,
    RerankerPort,
    SearchBackendPort,
    SearchHit,
    SiteInfo,
)


class FakeCrawler(CrawlerPort):
    """Typed in-memory fake implementing CrawlerPort."""

    def __init__(self, documents: Sequence[CrawlDocument] | None = None) -> None:
        self.documents: list[CrawlDocument] = list(documents) if documents is not None else []
        self.crawl_one_calls: list[str] = []
        self.crawl_many_calls: list[tuple[list[str], int]] = []
        self.closed: bool = False
        self.custom_handler: Any = None

    async def crawl_one(self, url: str) -> list[CrawlDocument]:
        self.crawl_one_calls.append(url)
        if self.custom_handler:
            return await self.custom_handler(url)
        matching = [d for d in self.documents if d.url == url]
        if matching:
            return matching
        return [
            CrawlDocument(
                url=url,
                success=True,
                markdown=f"# Content for {url}\n\nThis is synthetic test content.",
                raw_markdown=f"# Content for {url}\n\nThis is synthetic test content.",
                links=[
                    RemoteLink(href=f"{url}/page1", text="Page 1", internal=True),
                    RemoteLink(href="https://external.example.com", text="External", internal=False),
                ],
                title=f"Page {url}",
                status_code=200,
                content_type="text/markdown",
                language="en",
                metadata={"title": f"Page {url}"},
            )
        ]

    async def crawl_many(
        self, urls: Sequence[str], *, max_concurrent: int
    ) -> list[CrawlDocument]:
        url_list = list(urls)
        self.crawl_many_calls.append((url_list, max_concurrent))
        results: list[CrawlDocument] = []
        for url in url_list:
            results.extend(await self.crawl_one(url))
        return results

    async def aclose(self) -> None:
        self.closed = True


class FakeEmbedding(EmbeddingPort):
    """Typed in-memory fake implementing EmbeddingPort with 384 dimensions."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.embed_passages_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []
        self.should_fail: bool = False

    def _make_vector(self, text: str) -> list[float]:
        val = (abs(hash(text)) % 1000) / 1000.0
        vec = [val] * self.dimension
        vec[0] = 1.0
        return vec

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if self.should_fail:
            raise RuntimeError("Embedding failure simulated")
        text_list = list(texts)
        self.embed_passages_calls.append(text_list)
        return [self._make_vector(t) for t in text_list]

    async def embed_query(self, text: str) -> list[float]:
        if self.should_fail:
            raise RuntimeError("Embedding query failure simulated")
        self.embed_query_calls.append(text)
        return self._make_vector(text)


class FakeReranker(RerankerPort):
    """Typed in-memory fake implementing RerankerPort."""

    def __init__(self) -> None:
        self.rerank_calls: list[tuple[str, list[str]]] = []
        self.custom_results: list[RerankResult] | None = None
        self.should_fail: bool = False

    async def rerank(self, query: str, texts: Sequence[str]) -> list[RerankResult]:
        if self.should_fail:
            raise RuntimeError("Reranking failure simulated")
        text_list = list(texts)
        self.rerank_calls.append((query, text_list))
        if self.custom_results is not None:
            return self.custom_results
        results: list[RerankResult] = []
        for idx, t in enumerate(text_list):
            score = 1.0 / (idx + 1.0)
            results.append(RerankResult(id=idx, text=t, score=score))
        return results


class FakeGliner(GlinerPort):
    """Typed in-memory fake implementing GlinerPort."""

    def __init__(self) -> None:
        self.extract_calls: list[dict[str, Any]] = []
        self.custom_extraction: GlinerExtraction | None = None
        self.should_fail: bool = False

    async def extract(
        self,
        text: str,
        *,
        entities: Sequence[str],
        relations: Sequence[str],
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
    ) -> GlinerExtraction:
        if self.should_fail:
            raise RuntimeError("GLiNER extraction failure simulated")
        call_info = {
            "text": text,
            "entities": list(entities),
            "relations": list(relations),
            "threshold": threshold,
            "include_confidence": include_confidence,
            "include_spans": include_spans,
        }
        self.extract_calls.append(call_info)
        if self.custom_extraction is not None:
            return self.custom_extraction
        return GlinerExtraction(
            entities=[
                GlinerEntity(
                    text="FastMCP",
                    label="technology",
                    score=0.95,
                    start=0,
                    end=7,
                )
            ],
            relation_extraction=[
                GlinerRelation(
                    source="FastMCP",
                    target="Python",
                    relation="implements",
                    score=0.9,
                    fact="FastMCP implements Python framework",
                )
            ],
            text=text,
        )


class FakeChatGenerator(ChatGeneratorPort):
    """Typed in-memory fake implementing ChatGeneratorPort."""

    def __init__(self) -> None:
        self.chat_complete_calls: list[dict[str, Any]] = []
        self.code_summary_calls: list[tuple[str, str, str]] = []
        self.source_summary_calls: list[tuple[str, str, int]] = []
        self.chat_response: str = "This is a synthetic chat response."
        self.code_summary_response: str = "A Python utility function."
        self.source_summary_response: str = "Documentation for the synthetic test site."
        self.should_fail: bool = False

    async def chat_complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 150,
    ) -> str:
        if self.should_fail:
            raise RuntimeError("Chat completion failure simulated")
        self.chat_complete_calls.append(
            {"messages": list(messages), "temperature": temperature, "max_tokens": max_tokens}
        )
        return self.chat_response

    async def generate_code_example_summary(
        self, code: str, context_before: str, context_after: str
    ) -> str:
        if self.should_fail:
            raise RuntimeError("Code summary failure simulated")
        self.code_summary_calls.append((code, context_before, context_after))
        return self.code_summary_response

    async def extract_source_summary(
        self, source_id: str, content: str, max_length: int = 500
    ) -> str:
        if self.should_fail:
            raise RuntimeError("Source summary failure simulated")
        self.source_summary_calls.append((source_id, content, max_length))
        return self.source_summary_response


class FakeLangExtract(LangExtractPort):
    """Typed in-memory fake implementing LangExtractPort."""

    def __init__(self) -> None:
        self.extract_calls: list[str] = []
        self.custom_extractions: list[PageExtraction] | None = None
        self.should_fail: bool = False
        self.closed: bool = False

    async def extract_page(self, text: str) -> list[PageExtraction]:
        if self.should_fail:
            raise RuntimeError("LangExtract failure simulated")
        self.extract_calls.append(text)
        if self.custom_extractions is not None:
            return self.custom_extractions
        return [
            PageExtraction(
                extraction_class="technology",
                extraction_text="FastMCP",
                start_char=0,
                end_char=7,
                attributes={"type": "framework"},
                description="FastMCP framework",
            )
        ]

    async def aclose(self) -> None:
        self.closed = True


class FakeGraphStore(GraphStorePort):
    """Typed in-memory fake implementing GraphStorePort."""

    def __init__(self) -> None:
        self.ingested_payloads: list[CrawlIngestion] = []
        self.sites: list[SiteInfo] = []
        self.closed: bool = False
        self.should_fail: bool = False

    async def ingest_crawl(self, payload: CrawlIngestion) -> GraphOperationResult:
        if self.should_fail:
            raise RuntimeError("Graph store ingestion failure simulated")
        self.ingested_payloads.append(payload)
        chunk_count = sum(len(p.chunks) for p in payload.pages)
        return GraphOperationResult(
            success=True,
            run_id=payload.run_id,
            pages=len(payload.pages),
            chunks=chunk_count,
            entities=len(payload.site.entities),
            relations=len(payload.site.relations),
            links=sum(len(p.links) for p in payload.pages),
            details={"site_id": payload.site.site_id},
        )

    async def get_available_sites(self) -> list[SiteInfo]:
        if self.should_fail:
            raise RuntimeError("Graph store get_available_sites failure simulated")
        if self.sites:
            return self.sites
        result: list[SiteInfo] = []
        for p in self.ingested_payloads:
            result.append(
                SiteInfo(
                    site_id=p.site.site_id,
                    domain=p.site.domain,
                    root_url=p.site.root_url,
                    title=p.site.title,
                    summary=p.site.summary,
                    first_seen=p.started_at,
                    last_crawled=p.finished_at,
                    page_count=len(p.pages),
                    chunk_count=sum(len(page.chunks) for page in p.pages),
                    gliner_metadata=p.site.gliner_metadata,
                )
            )
        return result

    async def aclose(self) -> None:
        self.closed = True


class FakeSearchBackend(SearchBackendPort):
    """Typed in-memory fake implementing SearchBackendPort."""

    def __init__(self) -> None:
        self.vector_hits: list[SearchHit] = []
        self.text_hits: list[SearchHit] = []
        self.code_hits: list[SearchHit] = []
        self.vector_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []
        self.code_calls: list[dict[str, Any]] = []
        self.provenance_calls: list[str] = []
        self.should_fail: bool = False

    async def search_chunks_by_vector(
        self, embedding: Sequence[float], limit: int, site_id: str | None = None
    ) -> list[SearchHit]:
        if self.should_fail:
            raise RuntimeError("Search vector failure simulated")
        self.vector_calls.append(
            {"embedding": list(embedding), "limit": limit, "site_id": site_id}
        )
        hits = self.vector_hits
        if site_id:
            hits = [h for h in hits if h.site_id == site_id]
        return hits[:limit]

    async def search_chunks_by_text(
        self, query: str, limit: int, site_id: str | None = None
    ) -> list[SearchHit]:
        if self.should_fail:
            raise RuntimeError("Search text failure simulated")
        self.text_calls.append({"query": query, "limit": limit, "site_id": site_id})
        hits = self.text_hits
        if site_id:
            hits = [h for h in hits if h.site_id == site_id]
        return hits[:limit]

    async def search_code_chunks(
        self,
        embedding: Sequence[float],
        limit: int,
        site_id: str | None = None,
        language: str | None = None,
    ) -> list[SearchHit]:
        if self.should_fail:
            raise RuntimeError("Search code failure simulated")
        self.code_calls.append(
            {
                "embedding": list(embedding),
                "limit": limit,
                "site_id": site_id,
                "language": language,
            }
        )
        hits = self.code_hits
        if site_id:
            hits = [h for h in hits if h.site_id == site_id]
        if language:
            hits = [h for h in hits if (h.language or "").lower() == language.lower()]
        return hits[:limit]

    async def get_chunk_provenance(self, chunk_id: str) -> list[dict[str, Any]]:
        self.provenance_calls.append(chunk_id)
        return []


@pytest.fixture
def test_settings() -> Settings:
    """Return a valid Settings instance with testing defaults."""
    return Settings(
        mistral_api_key="test-mistral-api-key",
        model_choice="mistral-small-latest",
        crawl4ai_base_url="http://localhost:11235",
        crawl4ai_api_token="test-token",
        crawl4ai_timeout_seconds=60.0,
        crawl4ai_max_batch_size=100,
        falkordb_url="falkor://localhost:6380",
        falkordb_graph="crawl-graph",
        falkordb_query_timeout_ms=1000,
        falkordb_max_connections=16,
        unified_ml_base_url="http://localhost:8000",
        unified_ml_embed_model="intfloat/multilingual-e5-small",
        unified_ml_embedding_dimensions=384,
        unified_ml_timeout_seconds=30.0,
        unified_ml_batch_size=32,
        use_gliner_metadata=True,
        gliner_entity_labels="product,technology,library,organization,person",
        gliner_relation_labels="uses,depends_on,implements,stores",
        gliner_threshold=0.5,
        gliner_include_confidence=True,
        gliner_include_spans=True,
        use_langextract_metadata=False,
        use_contextual_embeddings=False,
        use_hybrid_search=False,
        use_reranking=False,
        use_agentic_rag=True,
    )


@pytest.fixture
def fake_crawler() -> FakeCrawler:
    return FakeCrawler()


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding(dimension=384)


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def fake_gliner() -> FakeGliner:
    return FakeGliner()


@pytest.fixture
def fake_chat() -> FakeChatGenerator:
    return FakeChatGenerator()


@pytest.fixture
def fake_graph_store() -> FakeGraphStore:
    return FakeGraphStore()


@pytest.fixture
def fake_search_backend() -> FakeSearchBackend:
    return FakeSearchBackend()


@pytest.fixture
def fake_lang_extract() -> FakeLangExtract:
    return FakeLangExtract()


@pytest.fixture
def crawl_context(
    fake_crawler: FakeCrawler,
    fake_embedding: FakeEmbedding,
    fake_reranker: FakeReranker,
    fake_gliner: FakeGliner,
    fake_chat: FakeChatGenerator,
    fake_graph_store: FakeGraphStore,
    fake_search_backend: FakeSearchBackend,
    fake_lang_extract: FakeLangExtract,
    test_settings: Settings,
) -> CrawlContext:
    """Create a fully typed CrawlContext populated with fake port implementations."""
    return CrawlContext(
        crawler=fake_crawler,
        embeddings=fake_embedding,
        reranker=fake_reranker,
        gliner=fake_gliner,
        chat=fake_chat,
        graph_store=fake_graph_store,
        lang_extract=fake_lang_extract,
        settings=test_settings,
        search_backend=fake_search_backend,
    )


@pytest.fixture
def mock_mcp_context(crawl_context: CrawlContext) -> Mock:
    """Create a mock FastMCP Context wrapping the typed CrawlContext."""
    ctx = Mock()
    request_ctx = Mock()
    request_ctx.lifespan_context = crawl_context
    ctx.request_context = request_ctx
    return ctx


@pytest.fixture
def sample_crawl_document() -> CrawlDocument:
    return CrawlDocument(
        url="https://example.com/test",
        success=True,
        markdown="# Test Document\n\nThis is a sample document for testing.",
        raw_markdown="# Test Document\n\nThis is a sample document for testing.",
        links=[
            RemoteLink(href="https://example.com/sub", text="Subpage", internal=True),
            RemoteLink(href="https://other.com", text="Other", internal=False),
        ],
        title="Test Document",
        status_code=200,
        content_type="text/markdown",
        language="en",
        metadata={"title": "Test Document"},
    )


@pytest.fixture
def sample_search_hit() -> SearchHit:
    return SearchHit(
        chunk_id="test-chunk-1",
        page_id="test-page-1",
        site_id="example.com",
        content="Sample search hit text content.",
        url="https://example.com/test",
        source="example.com",
        chunk_number=0,
        similarity_score=0.88,
        rerank_score=None,
        content_type="text",
        language="en",
        metadata={"title": "Test"},
        provenance=[],
    )
