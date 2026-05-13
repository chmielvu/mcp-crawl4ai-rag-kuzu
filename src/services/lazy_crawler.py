"""Lazy crawler proxy to avoid Playwright startup during MCP initialize."""

from __future__ import annotations

import asyncio
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig


class LazyCrawler:
    """Lazily initialize AsyncWebCrawler on first crawl call."""

    def __init__(self, browser_config: BrowserConfig) -> None:
        self._browser_config = browser_config
        self._crawler: AsyncWebCrawler | None = None
        self._lock = asyncio.Lock()

    async def _get_crawler(self) -> AsyncWebCrawler:
        if self._crawler is not None:
            return self._crawler
        async with self._lock:
            if self._crawler is None:
                crawler = AsyncWebCrawler(config=self._browser_config)
                await crawler.__aenter__()
                self._crawler = crawler
        return self._crawler

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        crawler = await self._get_crawler()
        return await crawler.arun(*args, **kwargs)

    async def arun_many(self, *args: Any, **kwargs: Any) -> Any:
        crawler = await self._get_crawler()
        return await crawler.arun_many(*args, **kwargs)

    async def aclose(self) -> None:
        if self._crawler is not None:
            await self._crawler.__aexit__(None, None, None)
            self._crawler = None
