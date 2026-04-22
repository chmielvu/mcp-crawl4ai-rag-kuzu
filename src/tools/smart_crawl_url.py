"""Tool for smart crawling with URL type detection."""

import json
import logging

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import CrawlContext
from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.services.database import DatabaseService
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.tools._ingestion import ingest_markdown_result, update_source_records
from crawl4ai_mcp.utilities.text_processing import TextProcessor

logger = logging.getLogger(__name__)


@mcp.tool()
async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
) -> str:
    """Intelligently crawl a URL and store content in Kuzu."""
    try:
        context: CrawlContext = ctx.request_context.lifespan_context
        settings = context.settings

        embedding_service = EmbeddingService(settings)
        database_service = DatabaseService(context.db_connection, settings)
        crawling_service = CrawlingService(context.crawler, settings, embedding_service)
        text_processor = TextProcessor(settings, embedding_service)

        actual_chunk_size = (
            chunk_size if chunk_size != 5000 else settings.default_chunk_size
        )
        crawl_type, results = await _crawl_by_url_type(
            crawling_service,
            url,
            max_depth=max_depth,
            max_concurrent=max_concurrent,
        )
        if not results:
            return json.dumps(
                {
                    "success": False,
                    "error": "No content was successfully crawled",
                    "crawl_type": crawl_type,
                }
            )

        source_ids = await update_source_records(
            results,
            database_service,
            crawling_service,
        )

        total_chunks_created = 0
        total_code_examples = 0
        processed_urls: list[str] = []
        for result in results:
            try:
                chunk_count, code_count = await ingest_markdown_result(
                    result_url=result["url"],
                    markdown_content=result["markdown"],
                    crawl_type=crawl_type,
                    settings=settings,
                    embedding_service=embedding_service,
                    database_service=database_service,
                    crawling_service=crawling_service,
                    text_processor=text_processor,
                    chunk_size=actual_chunk_size,
                )
                total_chunks_created += chunk_count
                total_code_examples += code_count
                processed_urls.append(result["url"])
            except Exception as error:
                logger.error("Error processing %s: %s", result.get("url"), error)

        return json.dumps(
            {
                "success": True,
                "crawl_type": crawl_type,
                "urls_processed": len(processed_urls),
                "total_chunks_created": total_chunks_created,
                "total_code_examples": total_code_examples,
                "sources_updated": source_ids,
                "message": (
                    f"Successfully crawled {len(processed_urls)} URLs using "
                    f"{crawl_type} strategy"
                ),
            }
        )
    except Exception as error:
        return json.dumps({"success": False, "error": str(error), "url": url})


async def _crawl_by_url_type(
    crawling_service: CrawlingService,
    url: str,
    max_depth: int,
    max_concurrent: int,
) -> tuple[str, list[dict[str, str]]]:
    if crawling_service.is_txt(url):
        return "txt file", await crawling_service.crawl_markdown_file(url)
    if crawling_service.is_sitemap(url):
        urls_from_sitemap = crawling_service.parse_sitemap(url)
        if not urls_from_sitemap:
            return "sitemap", []
        return "sitemap", await crawling_service.crawl_batch(
            urls_from_sitemap,
            max_concurrent=max_concurrent,
        )
    return "recursive", await crawling_service.crawl_recursive_internal_links(
        [url],
        max_depth=max_depth,
        max_concurrent=max_concurrent,
    )
