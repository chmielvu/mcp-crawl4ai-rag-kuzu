"""Tool for smart crawling with URL type detection, recursive BFS, and sitemap extraction."""

import asyncio
import logging

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import (
    CrawlDocument,
    CrawlFailure,
    GraphOperationResult,
    SmartCrawlResponse,
    get_server_context,
)
from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.tools._ingestion import ingest_crawl_documents
from crawl4ai_mcp.utilities.text_processing import TextProcessor

logger = logging.getLogger(__name__)


@mcp.tool()
async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
    timeout: int = 300,
) -> SmartCrawlResponse:
    """Intelligently crawl a URL based on type (sitemap, markdown, recursive web) and store in graph."""
    try:
        context = get_server_context(ctx)
        settings = context.settings

        crawling_service = CrawlingService(
            crawler=context.crawler,
            chat_generator=context.chat,
            settings=settings,
        )
        text_processor = TextProcessor(
            settings=settings,
            chat_generator=context.chat,
        )

        async def _execute_crawl() -> SmartCrawlResponse:
            crawl_type = "single_page"
            failures: list[CrawlFailure] = []
            urls_processed = 0
            valid_docs: list[CrawlDocument] = []

            if crawling_service.is_sitemap(url):
                crawl_type = "sitemap"
                sitemap_urls = await crawling_service.parse_sitemap(url)
                urls_processed = len(sitemap_urls)
                if not sitemap_urls:
                    failures.append(
                        CrawlFailure(
                            url=url,
                            error_message="Sitemap returned no URLs or could not be parsed",
                        )
                    )
                else:
                    raw_docs = await context.crawler.crawl_many(
                        sitemap_urls, max_concurrent=max_concurrent
                    )
                    for doc in raw_docs:
                        if doc.success and doc.markdown:
                            valid_docs.append(doc)
                        else:
                            failures.append(
                                doc.failure
                                or CrawlFailure(
                                    url=doc.url,
                                    error_message="Crawl failed or markdown empty",
                                    status_code=doc.status_code,
                                )
                            )

            elif crawling_service.is_txt(url):
                crawl_type = "txt_file"
                urls_processed = 1
                try:
                    raw_docs = await context.crawler.crawl_one(url)
                    for doc in raw_docs:
                        if doc.success and doc.markdown:
                            valid_docs.append(doc)
                        else:
                            failures.append(
                                doc.failure
                                or CrawlFailure(
                                    url=doc.url,
                                    error_message="Crawl failed or markdown empty",
                                    status_code=doc.status_code,
                                )
                            )
                except Exception as exc:
                    failures.append(CrawlFailure(url=url, error_message=str(exc)))

            elif max_depth > 1:
                crawl_type = "recursive"
                try:
                    valid_docs = await crawling_service.crawl_recursive_internal_links(
                        start_urls=[url],
                        max_depth=max_depth,
                        max_concurrent=max_concurrent,
                    )
                    urls_processed = len(valid_docs)
                    if not valid_docs:
                        failures.append(
                            CrawlFailure(
                                url=url,
                                error_message="No pages could be reached or scraped",
                            )
                        )
                except Exception as exc:
                    failures.append(CrawlFailure(url=url, error_message=str(exc)))
            else:
                crawl_type = "single_page"
                urls_processed = 1
                try:
                    raw_docs = await context.crawler.crawl_one(url)
                    for doc in raw_docs:
                        if doc.success and doc.markdown:
                            valid_docs.append(doc)
                        else:
                            failures.append(
                                doc.failure
                                or CrawlFailure(
                                    url=doc.url,
                                    error_message="Crawl failed or markdown empty",
                                    status_code=doc.status_code,
                                )
                            )
                except Exception as exc:
                    failures.append(CrawlFailure(url=url, error_message=str(exc)))

            if not valid_docs:
                return SmartCrawlResponse(
                    success=False,
                    url=url,
                    crawl_type=crawl_type,
                    urls_processed=urls_processed,
                    pages_crawled=0,
                    chunks_stored=0,
                    failures=failures or [
                        CrawlFailure(url=url, error_message="No valid content retrieved")
                    ],
                    message=f"Failed to crawl {url}: no valid pages retrieved",
                )

            ingest_result = await ingest_crawl_documents(
                documents=valid_docs,
                root_url=url,
                embedding_port=context.embeddings,
                text_processor=text_processor,
                graph_store=context.graph_store,
                crawling_service=crawling_service,
                gliner_port=context.gliner,
                chat_generator=context.chat,
                settings=settings,
                max_depth=max_depth,
                chunk_size=chunk_size,
                lang_extract=context.lang_extract,
            )

            if not ingest_result.success:
                return SmartCrawlResponse(
                    success=False,
                    url=url,
                    crawl_type=crawl_type,
                    run_id=ingest_result.run_id,
                    urls_processed=urls_processed,
                    pages_crawled=len(valid_docs),
                    chunks_stored=ingest_result.chunks,
                    failures=failures,
                    error=ingest_result,
                    message=f"Failed to store crawled content in graph: {ingest_result.error}",
                )

            return SmartCrawlResponse(
                success=True,
                url=url,
                crawl_type=crawl_type,
                run_id=ingest_result.run_id,
                urls_processed=urls_processed,
                pages_crawled=ingest_result.pages,
                chunks_stored=ingest_result.chunks,
                failures=failures,
                message=f"Successfully crawled and stored {ingest_result.pages} pages from {url}",
            )

        if timeout > 0:
            try:
                async with asyncio.timeout(timeout):
                    return await _execute_crawl()
            except TimeoutError:
                return SmartCrawlResponse(
                    success=False,
                    url=url,
                    crawl_type="unknown",
                    failures=[
                        CrawlFailure(
                            url=url,
                            error_message=f"Crawl operation timed out after {timeout} seconds",
                        )
                    ],
                    message=f"Crawl operation timed out after {timeout} seconds",
                )
        else:
            return await _execute_crawl()

    except Exception as error:
        logger.error("smart_crawl_url error: %s", error, exc_info=True)
        return SmartCrawlResponse(
            success=False,
            url=url,
            crawl_type="unknown",
            failures=[CrawlFailure(url=url, error_message=str(error))],
            error=GraphOperationResult(success=False, error=str(error)),
            message=f"Error during smart crawl for {url}: {error}",
        )
