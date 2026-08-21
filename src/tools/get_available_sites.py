"""Tool for retrieving available indexed sites from the graph store."""

import logging

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import (
    AvailableSitesResponse,
    GraphOperationResult,
    get_server_context,
)

logger = logging.getLogger(__name__)


@mcp.tool()
async def get_available_sites(ctx: Context) -> AvailableSitesResponse:
    """Get a list of all indexed sites and their metadata from the graph store."""
    try:
        context = get_server_context(ctx)
        sites = await context.graph_store.get_available_sites()
        return AvailableSitesResponse(
            success=True,
            sites=sites,
            total_sites=len(sites),
            message=f"Found {len(sites)} available sites",
        )
    except Exception as error:
        logger.error("get_available_sites error: %s", error, exc_info=True)
        return AvailableSitesResponse(
            success=False,
            sites=[],
            total_sites=0,
            error=GraphOperationResult(success=False, error=str(error)),
            message=f"Failed to retrieve available sites: {error}",
        )
