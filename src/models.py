"""Request models, typed server context, and contract re-exports."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationInfo, field_validator

from crawl4ai_mcp.services.contracts import (
    AvailableSitesResponse,
    ChatGeneratorPort,
    CodeSearchResponse,
    CrawlDocument,
    CrawlFailure,
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
    PagePayload,
    RagSearchResponse,
    RemoteLink,
    RemoteMarkdown,
    RerankResult,
    RerankerPort,
    SearchBackendPort,
    SearchHit,
    SiteInfo,
    SitePayload,
    SingleCrawlResponse,
    SmartCrawlResponse,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


class CrawlType(str, Enum):
    """Types of crawl operations exposed by the tools."""

    SINGLE_PAGE = "single_page"
    SITEMAP = "sitemap"
    TXT_FILE = "txt_file"
    RECURSIVE = "recursive"


class SearchType(str, Enum):
    """Search modes supported by the graph search service."""

    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CODE = "code"


class CrawlRequest(BaseModel):
    """Validated single/recursive crawl request."""

    url: HttpUrl
    max_depth: int = Field(default=3, ge=1, le=10)
    max_concurrent: int = Field(default=10, ge=1, le=50)
    chunk_size: int = Field(default=5000, ge=100, le=10000)
    overlap: int = Field(default=200, ge=0, le=1000)
    extract_code_examples: bool | None = None

    @field_validator("overlap")
    @classmethod
    def validate_overlap(cls, value: int, info: ValidationInfo) -> int:
        """Ensure overlap is smaller than the configured chunk size."""

        chunk_size = info.data.get("chunk_size", 5000)
        if value >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")
        return value


class BatchCrawlRequest(BaseModel):
    """Validated batch crawl request."""

    urls: list[HttpUrl]
    max_concurrent: int = Field(default=10, ge=1, le=50)
    chunk_size: int = Field(default=5000, ge=100, le=10000)
    overlap: int = Field(default=200, ge=0, le=1000)
    extract_code_examples: bool | None = None

    @field_validator("overlap")
    @classmethod
    def validate_overlap(cls, value: int, info: ValidationInfo) -> int:
        """Ensure overlap is smaller than the configured chunk size."""

        chunk_size = info.data.get("chunk_size", 5000)
        if value >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")
        return value


class SearchRequest(BaseModel):
    """Validated document search request."""

    query: str = Field(min_length=1, max_length=1000)
    source: str | None = None
    num_results: int = Field(default=5, ge=1, le=20)
    semantic_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    use_reranking: bool | None = None
    use_hybrid_search: bool | None = None


class CodeSearchRequest(BaseModel):
    """Validated code-chunk search request."""

    query: str = Field(min_length=1, max_length=500)
    language: str | None = None
    source: str | None = None
    num_results: int = Field(default=5, ge=1, le=20)


class CrawlContext(BaseModel):
    """Typed resources initialized by the FastMCP lifespan."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crawler: CrawlerPort
    embeddings: EmbeddingPort
    reranker: RerankerPort | None = None
    gliner: GlinerPort | None = None
    chat: ChatGeneratorPort
    graph_store: GraphStorePort
    lang_extract: LangExtractPort | None = None
    settings: Any
    search_backend: SearchBackendPort | None = None


def get_server_context(ctx: Context) -> CrawlContext:
    """Return the initialized lifespan context or raise a clear server error."""

    request_context = getattr(ctx, "request_context", None)
    lifespan_context = getattr(request_context, "lifespan_context", None)
    if not isinstance(lifespan_context, CrawlContext):
        raise RuntimeError(
            "FastMCP server lifespan is not initialized; request context is unavailable"
        )
    return lifespan_context


__all__ = [
    "CrawlType",
    "SearchType",
    "CrawlRequest",
    "BatchCrawlRequest",
    "SearchRequest",
    "CodeSearchRequest",
    "CrawlContext",
    "get_server_context",
    "RemoteLink",
    "RemoteMarkdown",
    "CrawlDocument",
    "CrawlFailure",
    "RerankResult",
    "GlinerEntity",
    "GlinerRelation",
    "GlinerExtraction",
    "SearchHit",
    "SiteInfo",
    "PageExtraction",
    "PagePayload",
    "SitePayload",
    "CrawlIngestion",
    "GraphOperationResult",
    "SingleCrawlResponse",
    "SmartCrawlResponse",
    "RagSearchResponse",
    "LangExtractPort",
    "CodeSearchResponse",
    "AvailableSitesResponse",
]
