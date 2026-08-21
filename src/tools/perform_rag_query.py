"""Tool for performing semantic/hybrid RAG queries against indexed content."""

import logging

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import GraphOperationResult, RagSearchResponse, get_server_context
from crawl4ai_mcp.services.falkor_search_backend import FalkorSearchBackend
from crawl4ai_mcp.services.search import SearchService

logger = logging.getLogger(__name__)


@mcp.tool()
async def perform_rag_query(
    ctx: Context,
    query: str,
    source: str | None = None,
    match_count: int = 5,
    use_hybrid: bool | None = None,
    use_reranking: bool | None = None,
) -> RagSearchResponse:
    """Perform a RAG query on stored content in the graph."""
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

        effective_hybrid = (
            use_hybrid
            if use_hybrid is not None
            else getattr(settings, "use_hybrid_search", False)
        )
        effective_reranking = bool(
            (
                use_reranking
                if use_reranking is not None
                else getattr(settings, "use_reranking", False)
            )
            and context.reranker is not None
        )

        search_type = "hybrid" if effective_hybrid else "semantic"

        hits = await search_service.perform_rag_query(
            query=query,
            match_count=match_count,
            site_id=source,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
        )

        return RagSearchResponse(
            success=True,
            query=query,
            search_type=search_type,
            results=hits,
            total_results=len(hits),
            source_filter=source,
            reranking_applied=effective_reranking,
            message=f"Found {len(hits)} results for query: {query}",
        )
    except Exception as error:
        logger.error("perform_rag_query error: %s", error, exc_info=True)
        return RagSearchResponse(
            success=False,
            query=query,
            search_type="semantic",
            results=[],
            total_results=0,
            source_filter=source,
            reranking_applied=False,
            error=GraphOperationResult(success=False, error=str(error)),
            message=f"Search failed: {error}",
        )
