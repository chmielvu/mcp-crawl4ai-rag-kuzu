"""Tool for performing RAG queries."""

import json

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import CrawlContext, SearchRequest, SearchType
from crawl4ai_mcp.services.search import SearchService
from crawl4ai_mcp.utilities.reranking import Reranker


@mcp.tool()
async def perform_rag_query(
    ctx: Context,
    query: str,
    source: str | None = None,
    match_count: int = 5,
) -> str:
    """Perform a RAG query on stored content."""
    try:
        context: CrawlContext = ctx.request_context.lifespan_context
        settings = context.settings
        search_service = SearchService(context.db_connection, settings)

        search_response = await search_service.perform_search(
            SearchRequest(
                query=query,
                source=source,
                num_results=match_count,
                semantic_threshold=settings.default_semantic_threshold,
            ),
            search_type=(
                SearchType.HYBRID if settings.use_hybrid_search else SearchType.SEMANTIC
            ),
            include_code_examples=False,
        )
        if not search_response.success:
            return json.dumps(
                {
                    "success": False,
                    "error": search_response.error or "Search failed",
                    "results": [],
                }
            )

        results_dict = [
            {
                "content": result.content,
                "url": result.url,
                "source": result.source,
                "chunk_number": result.chunk_number,
                "similarity_score": result.similarity_score,
                "metadata": result.metadata,
            }
            for result in search_response.results
        ]
        if settings.use_reranking and results_dict and context.reranking_model:
            reranker = Reranker(model=context.reranking_model, settings=settings)
            results_dict = reranker.rerank_results(query, results_dict)
            results_dict = reranker.filter_by_threshold(
                results_dict,
                threshold=settings.default_rerank_threshold,
            )

        formatted_results = []
        for result in results_dict[:match_count]:
            formatted_result = {
                "content": result["content"],
                "url": result["url"],
                "source": result["source"],
                "chunk_number": result["chunk_number"],
                "similarity_score": result["similarity_score"],
                "metadata": result["metadata"],
            }
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)

        return json.dumps(
            {
                "success": True,
                "query": query,
                "search_type": (
                    SearchType.HYBRID if settings.use_hybrid_search else SearchType.SEMANTIC
                ).value,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "source_filter": source,
                "reranking_applied": settings.use_reranking,
                "message": f"Found {len(formatted_results)} relevant results",
            }
        )
    except Exception as error:
        return json.dumps(
            {"success": False, "error": str(error), "query": query, "results": []}
        )
