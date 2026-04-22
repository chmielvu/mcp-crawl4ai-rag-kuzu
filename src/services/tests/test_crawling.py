"""Tests for crawling service."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.services.embeddings import EmbeddingService


@pytest.fixture
def mock_crawler():
    crawler = Mock()
    crawler.arun = AsyncMock()
    crawler.arun_many = AsyncMock()
    return crawler


@pytest.fixture
def mock_embedding_service():
    service = Mock(spec=EmbeddingService)
    service.chat_complete = AsyncMock(return_value="This code demonstrates a hello world function.")
    return service


@pytest.fixture
def crawling_service(mock_crawler, test_settings, mock_embedding_service):
    return CrawlingService(mock_crawler, test_settings, mock_embedding_service)


class TestUrlChecking:
    def test_is_sitemap_true(self, crawling_service) -> None:
        assert crawling_service.is_sitemap("https://example.com/sitemap.xml") is True
        assert crawling_service.is_sitemap("https://example.com/sitemap/index.xml") is True

    def test_is_sitemap_false(self, crawling_service) -> None:
        assert crawling_service.is_sitemap("https://example.com/index.html") is False

    def test_is_txt_true(self, crawling_service) -> None:
        assert crawling_service.is_txt("https://example.com/file.txt") is True

    def test_is_txt_false(self, crawling_service) -> None:
        assert crawling_service.is_txt("https://example.com/file.pdf") is False


class TestSitemapParsing:
    @patch("requests.get")
    def test_parse_sitemap_success(self, mock_get, crawling_service) -> None:
        response = Mock()
        response.status_code = 200
        response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>"""
        mock_get.return_value = response
        urls = crawling_service.parse_sitemap("https://example.com/sitemap.xml")
        assert urls == ["https://example.com/page1", "https://example.com/page2"]

    @patch("requests.get")
    def test_parse_sitemap_failure(self, mock_get, crawling_service) -> None:
        response = Mock(status_code=404)
        mock_get.return_value = response
        assert crawling_service.parse_sitemap("https://example.com/sitemap.xml") == []

    @patch("requests.get")
    def test_parse_sitemap_invalid_xml(self, mock_get, crawling_service) -> None:
        response = Mock(status_code=200, content=b"invalid")
        mock_get.return_value = response
        assert crawling_service.parse_sitemap("https://example.com/sitemap.xml") == []


class TestCrawling:
    @pytest.mark.asyncio
    async def test_crawl_markdown_file_success(self, crawling_service, mock_crawler) -> None:
        mock_crawler.arun.return_value = Mock(success=True, markdown="# Test Content")
        results = await crawling_service.crawl_markdown_file("https://example.com/file.txt")
        assert results == [{"url": "https://example.com/file.txt", "markdown": "# Test Content"}]

    @pytest.mark.asyncio
    async def test_crawl_markdown_file_failure(self, crawling_service, mock_crawler) -> None:
        mock_crawler.arun.return_value = Mock(success=False, error_message="Failed")
        assert await crawling_service.crawl_markdown_file("https://example.com/file.txt") == []

    @pytest.mark.asyncio
    async def test_crawl_batch(self, crawling_service, mock_crawler) -> None:
        mock_crawler.arun_many.return_value = [
            Mock(success=True, url="https://example.com/1", markdown="Content 1"),
            Mock(success=True, url="https://example.com/2", markdown="Content 2"),
            Mock(success=False, url="https://example.com/3", markdown=None, error_message="boom"),
        ]
        results = await crawling_service.crawl_batch(["a", "b", "c"], max_concurrent=5)
        assert results == [
            {"url": "https://example.com/1", "markdown": "Content 1"},
            {"url": "https://example.com/2", "markdown": "Content 2"},
        ]

    @pytest.mark.asyncio
    async def test_crawl_recursive_internal_links(self, crawling_service, mock_crawler) -> None:
        mock_crawler.arun_many.side_effect = [
            [
                Mock(
                    success=True,
                    url="https://example.com/page1",
                    markdown="Page 1",
                    links={"internal": [{"href": "https://example.com/page2"}]},
                )
            ],
            [
                Mock(
                    success=True,
                    url="https://example.com/page2",
                    markdown="Page 2",
                    links={"internal": []},
                )
            ],
        ]
        results = await crawling_service.crawl_recursive_internal_links(
            ["https://example.com/page1"],
            max_depth=2,
        )
        assert results == [
            {"url": "https://example.com/page1", "markdown": "Page 1"},
            {"url": "https://example.com/page2", "markdown": "Page 2"},
        ]


class TestCodeExtraction:
    def test_extract_code_blocks_simple(self, crawling_service) -> None:
        markdown = """
Some text before

```python
def hello():
    print("Hello, world!")
    for i in range(100):
        print(f"Line {i}")
    return []
```

Some text after
"""
        code_blocks = crawling_service.extract_code_blocks(markdown, min_length=20)
        assert len(code_blocks) == 1
        assert code_blocks[0]["language"] == "python"
        assert "Some text before" in code_blocks[0]["context_before"]

    def test_extract_code_blocks_skip_short(self, crawling_service) -> None:
        markdown = """Intro text

```python
short
```

```python
long enough block with extra content here
```
"""
        code_blocks = crawling_service.extract_code_blocks(markdown, min_length=10)
        assert len(code_blocks) == 1

    @pytest.mark.asyncio
    async def test_generate_code_example_summary(
        self, crawling_service, mock_embedding_service
    ) -> None:
        mock_embedding_service.chat_complete.return_value = "This code demonstrates a hello world function."
        summary = await crawling_service.generate_code_example_summary(
            code="def hello(): print('hello')",
            context_before="Here's an example:",
            context_after="That's the basic function.",
        )
        assert summary == "This code demonstrates a hello world function."

    @pytest.mark.asyncio
    async def test_generate_code_example_summary_error(
        self, crawling_service, mock_embedding_service
    ) -> None:
        mock_embedding_service.chat_complete.side_effect = Exception("API error")
        summary = await crawling_service.generate_code_example_summary(
            code="def hello(): pass",
            context_before="",
            context_after="",
        )
        assert summary == "Code example for demonstration purposes."


class TestSourceSummary:
    @pytest.mark.asyncio
    async def test_extract_source_summary_success(
        self, crawling_service, mock_embedding_service
    ) -> None:
        mock_embedding_service.chat_complete.return_value = "This is a test library."
        summary = await crawling_service.extract_source_summary(
            source_id="test-lib",
            content="This is the documentation for test-lib...",
        )
        assert summary == "This is a test library."

    @pytest.mark.asyncio
    async def test_extract_source_summary_empty_content(self, crawling_service) -> None:
        summary = await crawling_service.extract_source_summary("test-lib", "")
        assert summary == "Content from test-lib"

    @pytest.mark.asyncio
    async def test_extract_source_summary_long_result(
        self, crawling_service, mock_embedding_service
    ) -> None:
        mock_embedding_service.chat_complete.return_value = "A" * 600
        summary = await crawling_service.extract_source_summary(
            source_id="test-lib",
            content="Documentation content",
            max_length=500,
        )
        assert len(summary) == 503
        assert summary.endswith("...")

    @pytest.mark.asyncio
    async def test_extract_source_summary_error(
        self, crawling_service, mock_embedding_service
    ) -> None:
        mock_embedding_service.chat_complete.side_effect = Exception("API error")
        summary = await crawling_service.extract_source_summary(
            source_id="test-lib",
            content="Some content",
        )
        assert summary == "Content from test-lib"
