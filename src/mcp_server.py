"""FastMCP server initialization and lifespan management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from crawl4ai_mcp.models import CrawlContext

logger = logging.getLogger(__name__)


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[CrawlContext]:
    """Manage lifecycle of remote Crawl4AI, Unified-ML, FalkorDB, and Chat clients."""
    from crawl4ai_mcp.config import get_settings
    from crawl4ai_mcp.services.chat import ChatGenerator
    from crawl4ai_mcp.services.crawl4ai_client import Crawl4AIClient
    from crawl4ai_mcp.services.falkor_schema import init_falkor_schema
    from crawl4ai_mcp.services.falkor_search_backend import FalkorSearchBackend
    from crawl4ai_mcp.services.falkor_store import FalkorStore
    from crawl4ai_mcp.services.langextract_metadata import LangExtractMetadata
    from crawl4ai_mcp.services.unified_ml_client import UnifiedMLClient
    from falkordb.asyncio import FalkorDB
    from redis.asyncio import BlockingConnectionPool

    settings = get_settings()
    settings.validate_required_fields()

    crawler: Any | None = None
    ml_client: Any | None = None
    chat: Any | None = None
    lang_extract: Any | None = None
    pool: Any | None = None
    db: Any | None = None
    graph_store: Any | None = None

    try:
        crawler = Crawl4AIClient(settings=settings)
        ml_client = UnifiedMLClient(settings=settings)
        chat = ChatGenerator(settings=settings)
        lang_extract = (
            LangExtractMetadata(settings=settings)
            if settings.use_langextract_metadata
            else None
        )
        assert crawler is not None
        assert ml_client is not None
        assert chat is not None

        await crawler.health_check()
        await ml_client.health_check()

        redis_url = settings.falkordb_url
        if redis_url.startswith("falkor://"):
            redis_url = "redis://" + redis_url[len("falkor://") :]
        elif redis_url.startswith("falkors://"):
            redis_url = "rediss://" + redis_url[len("falkors://") :]

        pool = BlockingConnectionPool.from_url(
            redis_url,
            max_connections=settings.falkordb_max_connections,
            decode_responses=True,
        )
        db = FalkorDB(connection_pool=pool)
        graph = db.select_graph(settings.falkordb_graph)

        # FalkorDB cannot run read-only/schema procedures until a named graph has
        # been materialized; this write is intentionally empty and creates it.
        await graph.query("RETURN 1 AS graph_initialized")
        await init_falkor_schema(graph, settings.unified_ml_embedding_dimensions)

        # The lifespan owns the Redis pool; FalkorStore closes only the client.
        graph_store = FalkorStore(
            graph=graph,
            db=db,
            settings=settings,
            graph_name=settings.falkordb_graph,
        )
        search_backend = FalkorSearchBackend(graph=graph, settings=settings)

        assert crawler is not None
        assert ml_client is not None
        assert chat is not None
        context = CrawlContext(
            crawler=crawler,
            embeddings=ml_client,
            reranker=ml_client,
            gliner=ml_client,
            chat=chat,
            graph_store=graph_store,
            lang_extract=lang_extract,
            search_backend=search_backend,
            settings=settings,
        )
        yield context
    finally:
        resources: list[tuple[str, Any]] = [
            ("crawler", crawler),
            ("unified_ml", ml_client),
            ("chat", chat),
            ("lang_extract", lang_extract),
            ("graph_store", graph_store),
        ]
        for name, instance in resources:
            if instance is not None:
                close_fn = getattr(instance, "aclose", None)
                if callable(close_fn):
                    try:
                        await close_fn()
                    except Exception as exc:
                        logger.warning("Error closing %s: %s", name, exc)

        # If store construction failed, it could not close the Falkor client.
        if graph_store is None and db is not None:
            try:
                close_fn = getattr(db, "aclose", None)
                if callable(close_fn):
                    await close_fn()
            except Exception as exc:
                logger.warning("Error closing FalkorDB client: %s", exc)

        if pool is not None:
            try:
                if hasattr(pool, "aclose"):
                    await pool.aclose()
                elif hasattr(pool, "disconnect"):
                    await pool.disconnect()
            except Exception as exc:
                logger.warning("Error closing Redis connection pool: %s", exc)


def create_mcp_server() -> FastMCP:
    """Create and configure the FastMCP server instance."""
    return FastMCP(
        name="mcp-crawl4ai-rag",
        instructions="MCP server for RAG and web crawling with Crawl4AI",
        lifespan=crawl4ai_lifespan,
    )


mcp = create_mcp_server()


async def run_server() -> None:
    """Run the MCP server, loading tools and settings at runtime."""
    from crawl4ai_mcp.config import get_settings

    settings = get_settings()
    settings.validate_required_fields()

    # Import tools to register @mcp.tool decorators
    import crawl4ai_mcp.tools.crawl_single_page  # noqa: F401
    import crawl4ai_mcp.tools.get_available_sites  # noqa: F401
    import crawl4ai_mcp.tools.perform_rag_query  # noqa: F401
    import crawl4ai_mcp.tools.search_code_examples  # noqa: F401
    import crawl4ai_mcp.tools.smart_crawl_url  # noqa: F401

    logger.info("Starting MCP server on %s:%s", settings.host, settings.port)
    logger.info("Transport: %s", settings.transport)

    if settings.transport == "stdio":
        await mcp.run_stdio_async()
    else:
        mcp.settings.host = settings.host
        mcp.settings.port = settings.port
        await mcp.run_sse_async()
