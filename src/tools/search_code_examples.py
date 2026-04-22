"""Tool for searching code examples."""

import json

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import CrawlContext
from crawl4ai_mcp.services.search import SearchService
from crawl4ai_mcp.utilities.reranking import Reranker


@mcp.tool()
async def search_code_examples(
    ctx: Context,
    query: str,
    source_id: str | None = None,
    match_count: int = 5,
) -> str:
    """Search for code examples in the stored content."""
    try:
        context: CrawlContext = ctx.request_context.lifespan_context
        settings = context.settings
        if not settings.use_agentic_rag:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "Code example extraction is not enabled. "
                        "Set USE_AGENTIC_RAG=true to enable it."
                    ),
                    "results": [],
                }
            )

        search_service = SearchService(context.db_connection, settings)
        results = await search_service.search_code_examples(
            query=query,
            language=None,
            match_count=match_count * 2,
            source_id=source_id,
        )
        if not results:
            return json.dumps(
                {
                    "success": True,
                    "query": query,
                    "results": [],
                    "total_results": 0,
                    "message": "No code examples found matching the query",
                }
            )

        if settings.use_reranking and context.reranking_model:
            for result in results:
                result["content_for_rerank"] = result.get("summary", result.get("content", ""))
            reranker = Reranker(model=context.reranking_model, settings=settings)
            results = reranker.rerank_results(query, results, content_key="content_for_rerank")
            results = reranker.filter_by_threshold(
                results,
                threshold=settings.default_rerank_threshold,
            )

        formatted_results = []
        for result in results[:match_count]:
            formatted_result = {
                "code": result.get("content", ""),
                "language": result.get("metadata", {}).get("language", "unknown"),
                "summary": result.get("metadata", {}).get("summary", ""),
                "url": result.get("url", ""),
                "source": result.get("source_id", ""),
                "chunk_number": result.get("chunk_number", 0),
                "similarity_score": result.get("similarity", 0.0),
                "metadata": result.get("metadata", {}),
            }
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)

        return json.dumps(
            {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "source_filter": source_id,
                "reranking_applied": settings.use_reranking,
                "message": f"Found {len(formatted_results)} code examples",
            }
        )
    except Exception as error:
        return json.dumps(
            {"success": False, "error": str(error), "query": query, "results": []}
        )
