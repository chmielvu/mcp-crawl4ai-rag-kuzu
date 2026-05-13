"""MCP server setup and initialization."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

# Set Playwright browsers path before importing crawl4ai
# Find project root by looking for pyproject.toml or use MCP_PROJECT_ROOT
def _find_project_root() -> Path:
    """Find project root by searching for pyproject.toml."""
    # Try environment variable first
    env_root = os.environ.get("MCP_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Search upward from current working directory
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback to cwd
    return cwd

_project_root = _find_project_root()
_playwright_path = _project_root / ".venv" / "playwright"
if _playwright_path.exists():
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(_playwright_path))

# Import FastMCP only - defer crawl4ai and flashrank imports to lifespan
from mcp.server.fastmcp import FastMCP

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import CrawlContext
from crawl4ai_mcp.services.kuzu_schema import init_db
from crawl4ai_mcp.services.lazy_crawler import LazyCrawler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[CrawlContext]:
    """Manage the Crawl4AI and Kuzu lifecycle."""
    # Lazy imports to avoid module-level Playwright initialization
    from crawl4ai import BrowserConfig

    settings = get_settings()
    crawl4ai_base_directory = Path(settings.crawl4ai_base_directory)
    crawl4ai_base_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(crawl4ai_base_directory))
    browser_config = BrowserConfig(headless=True, verbose=False)

    crawler = LazyCrawler(browser_config)

    db_connection = init_db(settings.kuzu_db_path, settings.embedding_dimensions)
    reranking_model = None
    if settings.use_reranking:
        # Lazy import to avoid module-level model loading
        from flashrank import Ranker

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
        await crawler.aclose()


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server."""
    settings = get_settings()
    return FastMCP(
        name="mcp-crawl4ai-rag",
        instructions="MCP server for RAG and web crawling with Crawl4AI",
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
