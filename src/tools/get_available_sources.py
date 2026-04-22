"""Tool for retrieving available sources from the database."""

import json

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import CrawlContext
from crawl4ai_mcp.services.database import DatabaseService


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """Get a list of all available sources in Kuzu."""
    try:
        context: CrawlContext = ctx.request_context.lifespan_context
        database_service = DatabaseService(context.db_connection, context.settings)
        sources = await database_service.get_available_sources()
        return json.dumps(
            {
                "success": True,
                "sources": [
                    {
                        "source": source.source,
                        "summary": source.summary,
                        "total_documents": source.total_documents,
                        "total_chunks": source.total_chunks,
                        "total_code_examples": source.total_code_examples,
                        "word_count": source.word_count,
                        "last_updated": source.last_updated.isoformat(),
                    }
                    for source in sources
                ],
                "total_sources": len(sources),
                "message": f"Found {len(sources)} available sources",
            }
        )
    except Exception as error:
        return json.dumps(
            {
                "success": False,
                "error": str(error),
                "sources": [],
                "total_sources": 0,
            }
        )
