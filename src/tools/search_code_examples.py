"""Tool for searching code chunks in the indexed graph content."""

import logging

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import CodeSearchResponse, GraphOperationResult, get_server_context
from crawl4ai_mcp.services.falkor_search_backend import FalkorSearchBackend
from crawl4ai_mcp.services.search import SearchService

logger = logging.getLogger(__name__)


@mcp.tool()
async def search_code_examples(
    ctx: Context,
    query: str,
    source_id: str | None = None,
    language: str | None = None,
    match_count: int = 5,
    use_reranking: bool | None = None,
) -> CodeSearchResponse:
    """Search for code chunks in the stored graph content."""
    try:
        context = get_server_context(ctx)
        settings = context.settings

        backend = context.search_backend
        if backend is None and hasattr(context.graph_store, "graph"):
            backend = FalkorSearchBackend(
                graph=context.graph_store.graph,
                settings=settings,
            )

        if backend is None:
            raise RuntimeError("Search backend is not initialized")

        search_service = SearchService(
            backend=backend,
            embeddings=context.embeddings,
            reranker=context.reranker,
            settings=settings,
        )

        effective_reranking = bool(
            (
                use_reranking
                if use_reranking is not None
                else getattr(settings, "use_reranking", False)
            )
            and context.reranker is not None
        )

        fetch_count = max(match_count * 2, 10) if effective_reranking else match_count

        hits = await search_service.search_code_examples(
            query=query,
            language=language,
            match_count=fetch_count,
            site_id=source_id,
        )

        if effective_reranking and hits:
            hits = await search_service.rerank_hits(query, hits)

        final_hits = hits[:match_count]

        return CodeSearchResponse(
            success=True,
            query=query,
            results=final_hits,
            total_results=len(final_hits),
            source_filter=source_id,
            language=language,
            reranking_applied=effective_reranking,
            message=f"Found {len(final_hits)} code examples for query: {query}",
        )
    except Exception as error:
        logger.error("search_code_examples error: %s", error, exc_info=True)
        return CodeSearchResponse(
            success=False,
            query=query,
            results=[],
            total_results=0,
            source_filter=source_id,
            language=language,
            reranking_applied=False,
            error=GraphOperationResult(success=False, error=str(error)),
            message=f"Code search failed: {error}",
        )
