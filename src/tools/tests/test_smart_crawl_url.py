"""Tests for smart_crawl_url tool returning typed SmartCrawlResponse."""

from unittest.mock import Mock

import pytest

from crawl4ai_mcp.conftest import FakeCrawler, FakeGraphStore
from crawl4ai_mcp.models import CrawlDocument, SmartCrawlResponse
from crawl4ai_mcp.tools.smart_crawl_url import smart_crawl_url


@pytest.mark.asyncio
async def test_smart_crawl_url_success(
    mock_mcp_context: Mock,
    fake_crawler: FakeCrawler,
    fake_graph_store: FakeGraphStore,
) -> None:
    root_url = "https://example.com/docs"
    fake_crawler.documents = [
        CrawlDocument(
            url=root_url,
            success=True,
            markdown="# Root Documentation\n\nOverview of the system.",
            title="Root Docs",
            status_code=200,
        )
    ]

    response = await smart_crawl_url(
        mock_mcp_context,
        url=root_url,
        max_depth=2,
        max_concurrent=5,
        chunk_size=2000,
    )

    assert isinstance(response, SmartCrawlResponse)
    assert response.success is True
    assert response.url == root_url
    assert response.pages_crawled == 1
    assert response.chunks_stored >= 1
    assert len(fake_graph_store.ingested_payloads) == 1


@pytest.mark.asyncio
async def test_smart_crawl_url_no_valid_pages(
    mock_mcp_context: Mock,
    fake_crawler: FakeCrawler,
) -> None:
    root_url = "https://example.com/broken"
    fake_crawler.documents = [
        CrawlDocument(
            url=root_url,
            success=False,
            markdown="",
            status_code=500,
        )
    ]

    response = await smart_crawl_url(mock_mcp_context, url=root_url)

    assert isinstance(response, SmartCrawlResponse)
    assert response.success is False
    assert response.url == root_url
    assert "no valid pages" in (response.message or "").lower()


@pytest.mark.asyncio
async def test_smart_crawl_url_uninitialized_context() -> None:
    empty_context = Mock()
    empty_context.request_context = None

    response = await smart_crawl_url(empty_context, url="https://example.com")
    assert isinstance(response, SmartCrawlResponse)
    assert response.success is False
    assert "FastMCP server lifespan is not initialized" in (response.message or "")
