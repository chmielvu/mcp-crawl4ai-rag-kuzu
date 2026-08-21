"""Tests for CrawlingService web extraction, code extraction, and BFS recursion."""

import httpx
import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.conftest import FakeChatGenerator, FakeCrawler
from crawl4ai_mcp.services.contracts import CrawlDocument, RemoteLink
from crawl4ai_mcp.services.crawling import CrawlingService


@pytest.fixture
def crawling_service(
    fake_crawler: FakeCrawler,
    fake_chat: FakeChatGenerator,
    test_settings: Settings,
) -> CrawlingService:
    return CrawlingService(
        crawler=fake_crawler,
        chat_generator=fake_chat,
        settings=test_settings,
    )


def test_is_sitemap(crawling_service: CrawlingService) -> None:
    assert crawling_service.is_sitemap("https://example.com/sitemap.xml") is True
    assert crawling_service.is_sitemap("https://example.com/docs/sitemap") is True
    assert crawling_service.is_sitemap("https://example.com/page.html") is False


def test_is_txt(crawling_service: CrawlingService) -> None:
    assert crawling_service.is_txt("https://example.com/urls.txt") is True
    assert crawling_service.is_txt("https://example.com/page.html") is False


@pytest.mark.asyncio
async def test_parse_sitemap(crawling_service: CrawlingService) -> None:
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/page1</loc></url>
        <url><loc>https://example.com/page2</loc></url>
    </urlset>"""

    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, text=xml_content, headers={"Content-Type": "application/xml"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        urls = await crawling_service.parse_sitemap("https://example.com/sitemap.xml", client=client)

    assert len(urls) == 2
    assert "https://example.com/page1" in urls
    assert "https://example.com/page2" in urls


@pytest.mark.asyncio
async def test_crawl_markdown_file(
    crawling_service: CrawlingService, fake_crawler: FakeCrawler
) -> None:
    docs = await crawling_service.crawl_markdown_file(
        "https://example.com/links.txt"
    )
    assert len(docs) == 1
    assert docs[0].success is True
    assert docs[0].markdown


def test_extract_code_blocks(crawling_service: CrawlingService) -> None:
    markdown = (
        "Here is some introduction text.\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "And here is the conclusion."
    )
    blocks = crawling_service.extract_code_blocks(markdown, min_length=10)
    assert len(blocks) == 1
    b = blocks[0]
    assert b["language"] == "python"
    assert "def add(a, b):" in b["code"]
    assert "introduction text" in b["context_before"]
    assert "conclusion" in b["context_after"]
    assert b["start_char"] > 0
    assert b["end_char"] > b["start_char"]


@pytest.mark.asyncio
async def test_crawl_batch(crawling_service: CrawlingService, fake_crawler: FakeCrawler) -> None:
    urls = ["https://example.com/p1", "https://example.com/p2"]
    docs = await crawling_service.crawl_batch(urls, max_concurrent=5)
    assert len(docs) == 2
    assert len(fake_crawler.crawl_many_calls) == 1


@pytest.mark.asyncio
async def test_crawl_recursive_internal_links(
    crawling_service: CrawlingService,
    fake_crawler: FakeCrawler,
) -> None:
    root_doc = CrawlDocument(
        url="https://example.com/root",
        success=True,
        markdown="Root markdown",
        links=[RemoteLink(href="https://example.com/root/child", internal=True)],
    )
    child_doc = CrawlDocument(
        url="https://example.com/root/child",
        success=True,
        markdown="Child markdown",
        links=[],
    )
    fake_crawler.documents = [root_doc, child_doc]

    bfs_docs = await crawling_service.crawl_recursive_internal_links(["https://example.com/root"], max_depth=2)
    assert len(bfs_docs) == 2
    assert any(d.url == "https://example.com/root" for d in bfs_docs)
    assert any(d.url == "https://example.com/root/child" for d in bfs_docs)
