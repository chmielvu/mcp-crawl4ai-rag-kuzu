"""MCP server setup and initialization."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from crawl4ai import AsyncWebCrawler, BrowserConfig
from flashrank import Ranker
from mcp.server.fastmcp import FastMCP

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import CrawlContext
from crawl4ai_mcp.services.kuzu_schema import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[CrawlContext]:
    """Manage the Crawl4AI and Kuzu lifecycle."""
    settings = get_settings()
    browser_config = BrowserConfig(headless=True, verbose=False)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()

    db_connection = init_db(settings.kuzu_db_path, settings.embedding_dimensions)
    reranking_model = None
    if settings.use_reranking:
        reranking_model = Ranker(
            model_name=settings.reranker_model,
            cache_dir=settings.reranker_cache_dir,
            max_length=settings.reranker_max_length,
        )

    context = CrawlContext(
        crawler=crawler,
        db_connection=db_connection,
        reranking_model=reranking_model,
        settings=settings,
    )

    try:
        yield context
    finally:
        db_connection.close()
        await crawler.__aexit__(None, None, None)


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server."""
    settings = get_settings()
    return FastMCP(
        "mcp-crawl4ai-rag",
        description="MCP server for RAG and web crawling with Crawl4AI",
        lifespan=crawl4ai_lifespan,
        host=settings.host,
        port=settings.port,
    )


mcp = create_mcp_server()


async def run_server() -> None:
    """Run the MCP server."""
    settings = get_settings()

    from crawl4ai_mcp.tools.crawl_single_page import crawl_single_page  # noqa: F401
    from crawl4ai_mcp.tools.get_available_sources import get_available_sources  # noqa: F401
    from crawl4ai_mcp.tools.perform_rag_query import perform_rag_query  # noqa: F401
    from crawl4ai_mcp.tools.search_code_examples import search_code_examples  # noqa: F401
    from crawl4ai_mcp.tools.smart_crawl_url import smart_crawl_url  # noqa: F401

    logger.info("Starting MCP server on %s:%s", settings.host, settings.port)
    logger.info("Transport: %s", settings.transport)

    if settings.transport == "stdio":
        await mcp.run_stdio_async()
    else:
        await mcp.run_sse_async()
