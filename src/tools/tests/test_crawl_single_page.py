"""Tests for crawl_single_page tool returning typed SingleCrawlResponse."""

from unittest.mock import Mock

import pytest

from crawl4ai_mcp.conftest import FakeCrawler, FakeGraphStore
from crawl4ai_mcp.models import CrawlDocument, SingleCrawlResponse
from crawl4ai_mcp.tools.crawl_single_page import crawl_single_page


@pytest.mark.asyncio
async def test_crawl_single_page_success(
    mock_mcp_context: Mock,
    fake_crawler: FakeCrawler,
    fake_graph_store: FakeGraphStore,
) -> None:
    target_url = "https://example.com/single-page"
    fake_crawler.documents = [
        CrawlDocument(
            url=target_url,
            success=True,
            markdown="# Single Page\n\nThis is the content of the single page.",
            title="Single Page",
            status_code=200,
        )
    ]

    response = await crawl_single_page(mock_mcp_context, url=target_url)

    assert isinstance(response, SingleCrawlResponse)
    assert response.success is True
    assert response.url == target_url
    assert response.pages_crawled >= 1
    assert response.chunks_stored >= 1
    assert len(fake_graph_store.ingested_payloads) == 1


@pytest.mark.asyncio
async def test_crawl_single_page_crawl_failure(
    mock_mcp_context: Mock,
    fake_crawler: FakeCrawler,
) -> None:
    target_url = "https://example.com/not-found"
    fake_crawler.documents = [
        CrawlDocument(
            url=target_url,
            success=False,
            markdown="",
            status_code=404,
        )
    ]

    response = await crawl_single_page(mock_mcp_context, url=target_url)

    assert isinstance(response, SingleCrawlResponse)
    assert response.success is False
    assert response.url == target_url
    assert len(response.failures) >= 1


@pytest.mark.asyncio
async def test_crawl_single_page_uninitialized_context() -> None:
    empty_context = Mock()
    empty_context.request_context = None

    response = await crawl_single_page(empty_context, url="https://example.com")
    assert isinstance(response, SingleCrawlResponse)
    assert response.success is False
    assert "FastMCP server lifespan is not initialized" in (response.message or "")
