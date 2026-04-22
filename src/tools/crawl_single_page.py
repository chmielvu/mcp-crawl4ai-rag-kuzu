"""Tool for crawling a single web page."""

import json
from urllib.parse import urlparse

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import CrawlContext
from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.services.database import DatabaseService
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.tools._ingestion import ingest_markdown_result, update_source_records
from crawl4ai_mcp.utilities.text_processing import TextProcessor


@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """Crawl a single web page and store its content in Kuzu."""
    try:
        context: CrawlContext = ctx.request_context.lifespan_context
        settings = context.settings

        embedding_service = EmbeddingService(settings)
        database_service = DatabaseService(context.db_connection, settings)
        crawling_service = CrawlingService(context.crawler, settings, embedding_service)
        text_processor = TextProcessor(settings, embedding_service)

        results = await crawling_service.crawl_batch([url], max_concurrent=1)
        if not results:
            return json.dumps({"success": False, "error": "Failed to crawl the URL"})

        result = results[0]
        source_ids = await update_source_records(
            [result],
            database_service,
            crawling_service,
        )
        chunk_count, code_count = await ingest_markdown_result(
            result_url=result["url"],
            markdown_content=result["markdown"],
            crawl_type="single_page",
            settings=settings,
            embedding_service=embedding_service,
            database_service=database_service,
            crawling_service=crawling_service,
            text_processor=text_processor,
            chunk_size=settings.default_chunk_size,
        )

        return json.dumps(
            {
                "success": True,
                "url": url,
                "chunks_created": chunk_count,
                "code_examples_created": code_count,
                "total_word_count": len(result["markdown"].split()),
                "source_id": urlparse(url).netloc,
                "sources_updated": source_ids,
                "message": f"Successfully crawled and stored content from {url}",
            }
        )
    except Exception as error:
        return json.dumps({"success": False, "error": str(error), "url": url})
