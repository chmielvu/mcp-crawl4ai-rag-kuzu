"""Tool for crawling a single web page and ingesting into the graph store."""

import logging

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import (
    CrawlFailure,
    GraphOperationResult,
    SingleCrawlResponse,
    get_server_context,
)
from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.tools._ingestion import ingest_crawl_documents
from crawl4ai_mcp.utilities.text_processing import TextProcessor

logger = logging.getLogger(__name__)


@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> SingleCrawlResponse:
    """Crawl a single web page and store its content in the graph."""
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

        try:
            documents = await context.crawler.crawl_one(url)
        except Exception as crawl_err:
            return SingleCrawlResponse(
                success=False,
                url=url,
                failures=[
                    CrawlFailure(
                        url=url,
                        error_message=str(crawl_err),
                    )
                ],
                message=f"Failed to crawl {url}: {crawl_err}",
            )

        failures: list[CrawlFailure] = []
        for doc in documents:
            if not doc.success or not doc.markdown:
                if doc.failure is not None:
                    failures.append(doc.failure)
                else:
                    failures.append(
                        CrawlFailure(
                            url=doc.url,
                            error_message="No markdown content returned",
                            status_code=doc.status_code,
                        )
                    )

        valid_docs = [doc for doc in documents if doc.success and doc.markdown]
        if not valid_docs:
            return SingleCrawlResponse(
                success=False,
                url=url,
                failures=failures or [
                    CrawlFailure(url=url, error_message="No valid content retrieved")
                ],
                message=f"No valid markdown content retrieved from {url}",
            )

        chunk_size = getattr(settings, "default_chunk_size", 5000)
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
            max_depth=1,
            chunk_size=chunk_size,
            lang_extract=context.lang_extract,
        )

        if not ingest_result.success:
            return SingleCrawlResponse(
                success=False,
                url=url,
                run_id=ingest_result.run_id,
                pages_crawled=len(valid_docs),
                chunks_stored=ingest_result.chunks,
                failures=failures,
                error=ingest_result,
                message=f"Failed to store crawled content in graph: {ingest_result.error}",
            )

        return SingleCrawlResponse(
            success=True,
            url=url,
            run_id=ingest_result.run_id,
            pages_crawled=ingest_result.pages,
            chunks_stored=ingest_result.chunks,
            failures=failures,
            message=f"Successfully crawled and stored content from {url}",
        )
    except Exception as error:
        logger.error("crawl_single_page error: %s", error, exc_info=True)
        return SingleCrawlResponse(
            success=False,
            url=url,
            failures=[CrawlFailure(url=url, error_message=str(error))],
            error=GraphOperationResult(success=False, error=str(error)),
            message=f"Error crawling {url}: {error}",
        )
