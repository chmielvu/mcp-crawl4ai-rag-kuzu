"""Typed domain contracts shared by providers, ingestion, graph, and MCP tools."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Base model for remote payloads that may gain provider fields."""

    model_config = ConfigDict(extra="allow")


class RemoteLink(ContractModel):
    """Normalized link from a remote Crawl4AI Markdown response."""

    href: str
    text: str | None = None
    title: str | None = None
    rel: str | None = None
    internal: bool | None = None


class RemoteMarkdown(ContractModel):
    """The structured Markdown object returned by Crawl4AI REST."""

    fit_markdown: str | None = None
    raw_markdown: str | None = None


class CrawlFailure(ContractModel):
    """A structured per-URL crawl failure."""

    url: str
    error_message: str
    status_code: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class CrawlDocument(ContractModel):
    """Normalized crawl result consumed by all ingestion callers."""

    url: str
    success: bool = True
    markdown: str = ""
    raw_markdown: str | None = None
    links: list[RemoteLink] = Field(default_factory=list)
    title: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    failure: CrawlFailure | None = None


class RerankResult(ContractModel):
    """A result item returned by Unified-ML /rerank."""

    id: int
    text: str
    score: float


class GlinerEntity(ContractModel):
    """A GLiNER entity returned by Unified-ML."""

    text: str
    label: str
    score: float | None = None
    start: int | None = None
    end: int | None = None
    embedding: list[float] | None = None


class GlinerRelation(ContractModel):
    """A normalized GLiNER relation fact."""

    source: str
    target: str
    relation: str
    score: float | None = None
    fact: str | None = None
    description: str | None = None
    source_entity_type: str | None = None
    target_entity_type: str | None = None
    embedding: list[float] | None = None


class GlinerExtraction(ContractModel):
    """Complete Unified-ML extraction response."""

    entities: list[GlinerEntity] = Field(default_factory=list)
    relation_extraction: list[GlinerRelation] = Field(default_factory=list)
    text: str | None = None
    @model_validator(mode="before")
    @classmethod
    def normalize_unified_ml_shape(cls, value: Any) -> Any:
        """Normalize label/relation maps returned by the live GLiNER endpoint."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "entities" not in value or "relation_extraction" not in value:
            raise ValueError(
                "Unified-ML extraction results require entities and relation_extraction"
            )
        raw_entities = value.get("entities", [])
        entity_items: list[dict[str, Any]] = []
        if isinstance(raw_entities, dict):
            for label, items in raw_entities.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            entity_items.append(
                                {
                                    **item,
                                    "label": item.get("label", label),
                                    "score": item.get("score", item.get("confidence")),
                                }
                            )
        elif isinstance(raw_entities, list):
            for item in raw_entities:
                if isinstance(item, GlinerEntity):
                    entity_items.append(item.model_dump())
                elif isinstance(item, dict):
                    entity_items.append(item)
        normalized["entities"] = entity_items

        raw_relations = value.get("relation_extraction", [])
        relation_items: list[dict[str, Any]] = []
        relation_groups = (
            raw_relations.items()
            if isinstance(raw_relations, dict)
            else [(None, raw_relations)]
        )
        for relation_name, items in relation_groups:
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, GlinerRelation):
                    relation_items.append(item.model_dump())
                    continue
                if not isinstance(item, dict):
                    continue
                head = item.get("head", {})
                tail = item.get("tail", {})
                if not isinstance(head, dict) or not isinstance(tail, dict):
                    continue
                relation_items.append(
                    {
                        **item,
                        "source": head.get("text", ""),
                        "target": tail.get("text", ""),
                        "relation": item.get("relation", relation_name or ""),
                        "score": item.get(
                            "score",
                            min(
                                float(head.get("confidence", 1.0)),
                                float(tail.get("confidence", 1.0)),
                            ),
                        ),
                    }
                )
        normalized["relation_extraction"] = relation_items
        return normalized


class SearchHit(ContractModel):
    """A normalized graph search hit."""

    chunk_id: str
    page_id: str
    site_id: str
    content: str
    url: str
    source: str
    chunk_number: int
    similarity_score: float
    rerank_score: float | None = None
    content_type: str = "text"
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[dict[str, Any]] = Field(default_factory=list)


class PageExtraction(ContractModel):
    """Grounded page-level LangExtract item propagated to chunks."""

    extraction_class: str
    extraction_text: str
    start_char: int
    end_char: int
    attributes: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class ChunkPayload(ContractModel):
    """Chunk and embedding data ready for graph ingestion."""

    chunk_id: str
    text: str
    index: int
    heading_path: str = ""
    start_char: int = 0
    end_char: int = 0
    content_type: str = "text"
    language: str | None = None
    metadata_json: str = "{}"
    embedding: list[float]
    extractions: list[PageExtraction] = Field(default_factory=list)


class PagePayload(ContractModel):
    """Page and chunk data ready for one graph write."""

    page_id: str
    url: str
    canonical_url: str
    title: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    language: str | None = None
    content_hash: str
    depth: int = 0
    crawled_at: datetime
    metadata_json: str = "{}"
    chunks: list[ChunkPayload] = Field(default_factory=list)
    links: list[RemoteLink] = Field(default_factory=list)


class SitePayload(ContractModel):
    """Site-level graph metadata for a crawl run."""

    site_id: str
    domain: str
    root_url: str
    title: str | None = None
    summary: str | None = None
    gliner_metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[GlinerEntity] = Field(default_factory=list)
    relations: list[GlinerRelation] = Field(default_factory=list)


class CrawlIngestion(ContractModel):
    """Atomic site crawl payload written by GraphStorePort."""

    run_id: str
    root_url: str
    max_depth: int
    started_at: datetime
    finished_at: datetime
    site: SitePayload
    pages: list[PagePayload]


class GraphOperationResult(ContractModel):
    """Typed counts and errors from a graph operation."""

    success: bool
    run_id: str | None = None
    pages: int = 0
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    links: int = 0
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SiteInfo(ContractModel):
    """Typed site listing item."""

    site_id: str
    domain: str
    root_url: str
    title: str | None = None
    summary: str | None = None
    first_seen: datetime | None = None
    last_crawled: datetime | None = None
    page_count: int = 0
    chunk_count: int = 0
    gliner_metadata: dict[str, Any] = Field(default_factory=dict)


class SingleCrawlResponse(ContractModel):
    """Typed response for crawl_single_page."""

    success: bool
    url: str
    run_id: str | None = None
    pages_crawled: int = 0
    chunks_stored: int = 0
    failures: list[CrawlFailure] = Field(default_factory=list)
    error: GraphOperationResult | None = None
    message: str | None = None


class SmartCrawlResponse(ContractModel):
    """Typed response for smart_crawl_url."""

    success: bool
    url: str
    crawl_type: str
    run_id: str | None = None
    urls_processed: int = 0
    pages_crawled: int = 0
    chunks_stored: int = 0
    failures: list[CrawlFailure] = Field(default_factory=list)
    error: GraphOperationResult | None = None
    message: str | None = None


class RagSearchResponse(ContractModel):
    """Typed response for perform_rag_query."""

    success: bool
    query: str
    search_type: str
    results: list[SearchHit] = Field(default_factory=list)
    total_results: int = 0
    source_filter: str | None = None
    reranking_applied: bool = False
    error: GraphOperationResult | None = None
    message: str | None = None


class CodeSearchResponse(ContractModel):
    """Typed response for search_code_examples."""

    success: bool
    query: str
    results: list[SearchHit] = Field(default_factory=list)
    total_results: int = 0
    source_filter: str | None = None
    language: str | None = None
    reranking_applied: bool = False
    error: GraphOperationResult | None = None
    message: str | None = None


class AvailableSitesResponse(ContractModel):
    """Typed response for get_available_sites."""

    success: bool
    sites: list[SiteInfo] = Field(default_factory=list)
    total_sites: int = 0
    error: GraphOperationResult | None = None
    message: str | None = None


@runtime_checkable
class CrawlerPort(Protocol):
    """Remote crawler boundary."""

    async def crawl_one(self, url: str) -> list[CrawlDocument]:
        ...

    async def crawl_many(
        self, urls: Sequence[str], *, max_concurrent: int
    ) -> list[CrawlDocument]:
        ...

    async def aclose(self) -> None:
        ...


@runtime_checkable
class EmbeddingPort(Protocol):
    """Unified-ML embedding boundary."""

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    async def embed_query(self, text: str) -> list[float]:
        ...


@runtime_checkable
class RerankerPort(Protocol):
    """Unified-ML reranking boundary."""

    async def rerank(self, query: str, texts: Sequence[str]) -> list[RerankResult]:
        ...


@runtime_checkable
class GlinerPort(Protocol):
    """Unified-ML GLiNER extraction boundary."""

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
        ...


@runtime_checkable
class ChatGeneratorPort(Protocol):
    """Mistral chat-only boundary."""

    async def chat_complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 150,
    ) -> str:
        ...

    async def generate_code_example_summary(
        self, code: str, context_before: str, context_after: str
    ) -> str:
        ...

    async def extract_source_summary(
        self, source_id: str, content: str, max_length: int = 500
    ) -> str:
        ...


@runtime_checkable
class LangExtractPort(Protocol):
    """Optional grounded page-level metadata boundary."""

    async def extract_page(self, text: str) -> list[PageExtraction]:
        ...

    async def aclose(self) -> None:
        ...

@runtime_checkable
class GraphStorePort(Protocol):
    """Remote FalkorDB graph write/list boundary."""

    async def ingest_crawl(self, payload: CrawlIngestion) -> GraphOperationResult:
        ...

    async def get_available_sites(self) -> list[SiteInfo]:
        ...

    async def aclose(self) -> None:
        ...


@runtime_checkable
class SearchBackendPort(Protocol):
    """Async graph search boundary used by SearchService."""

    async def search_chunks_by_vector(
        self, embedding: Sequence[float], limit: int, site_id: str | None = None
    ) -> list[SearchHit]:
        ...

    async def search_chunks_by_text(
        self, query: str, limit: int, site_id: str | None = None
    ) -> list[SearchHit]:
        ...

    async def search_code_chunks(
        self,
        embedding: Sequence[float],
        limit: int,
        site_id: str | None = None,
        language: str | None = None,
    ) -> list[SearchHit]:
        ...
    async def get_chunk_provenance(self, chunk_id: str) -> list[dict[str, Any]]:
        ...
