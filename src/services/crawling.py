"""Crawling service for web content extraction and processing."""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urldefrag, urlparse
from xml.etree import ElementTree

import requests
from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
)

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

MAX_FAILURES_TO_LOG = 5


class CrawlingService:
    """Service for crawling web content and extracting information."""

    def __init__(
        self,
        crawler: AsyncWebCrawler,
        settings: Optional[Any] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.crawler = crawler
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(self.settings)

    def is_sitemap(self, url: str) -> bool:
        return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path

    def is_txt(self, url: str) -> bool:
        return url.endswith(".txt")

    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        response = requests.get(sitemap_url, timeout=30)
        if response.status_code != 200:
            return []
        try:
            tree = ElementTree.fromstring(response.content)
            return [loc.text for loc in tree.findall(".//{*}loc") if loc.text]
        except Exception as error:
            logger.error("Error parsing sitemap XML: %s", error)
            return []

    async def crawl_markdown_file(self, url: str) -> List[Dict[str, Any]]:
        result = await self.crawler.arun(url=url, config=CrawlerRunConfig())
        if result.success and result.markdown:
            return [{"url": url, "markdown": result.markdown}]
        logger.warning("Failed to crawl %s: %s", url, result.error_message)
        return []

    async def crawl_batch(
        self, urls: List[str], max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent,
        )
        try:
            results = await self.crawler.arun_many(
                urls=urls,
                config=crawl_config,
                dispatcher=dispatcher,
            )
        except Exception as error:
            logger.error("Error in crawl_batch: %s", error)
            return []

        successful_results = [
            {"url": result.url, "markdown": result.markdown}
            for result in results
            if result.success and result.markdown
        ]
        failed_results = [
            result for result in results if not result.success or not result.markdown
        ]
        for failed_result in failed_results[:MAX_FAILURES_TO_LOG]:
            logger.error(
                "Failed crawl: %s - %s",
                failed_result.url,
                getattr(failed_result, "error_message", "No markdown content"),
            )
        return successful_results

    async def crawl_recursive_internal_links(
        self,
        start_urls: List[str],
        max_depth: int = 3,
        max_concurrent: int = 10,
    ) -> List[Dict[str, Any]]:
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent,
        )
        visited: set[str] = set()
        current_urls = {_normalize_url(url) for url in start_urls}
        results_all: list[dict[str, Any]] = []

        for _ in range(max_depth):
            urls_to_crawl = [url for url in current_urls if url not in visited]
            if not urls_to_crawl:
                break

            results = await self.crawler.arun_many(
                urls=urls_to_crawl,
                config=run_config,
                dispatcher=dispatcher,
            )
            next_level_urls: set[str] = set()
            for result in results:
                normalized_url = _normalize_url(result.url)
                visited.add(normalized_url)
                if result.success and result.markdown:
                    results_all.append({"url": result.url, "markdown": result.markdown})
                    for link in result.links.get("internal", []):
                        href = _normalize_url(link["href"])
                        if href not in visited:
                            next_level_urls.add(href)
            current_urls = next_level_urls
        return results_all

    def extract_code_blocks(
        self, markdown_content: str, min_length: int = 1000
    ) -> List[Dict[str, Any]]:
        code_blocks: list[dict[str, Any]] = []
        backtick_positions = _backtick_positions(markdown_content)
        index = 0
        while index < len(backtick_positions) - 1:
            start_pos = backtick_positions[index]
            end_pos = backtick_positions[index + 1]
            language, code_content = _parse_code_section(
                markdown_content[start_pos + 3 : end_pos]
            )
            if len(code_content) >= min_length:
                code_blocks.append(
                    {
                        "code": code_content,
                        "language": language,
                        "context_before": markdown_content[max(0, start_pos - 1000) : start_pos].strip(),
                        "context_after": markdown_content[end_pos + 3 : end_pos + 1003].strip(),
                    }
                )
            index += 2
        return code_blocks

    async def generate_code_example_summary(
        self, code: str, context_before: str, context_after: str
    ) -> str:
        prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated."""
        messages = [
            {
                "role": "system",
                "content": "You provide concise code example summaries.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            return await self.embedding_service.chat_complete(messages, max_tokens=100)
        except Exception as error:
            logger.error("Error generating code example summary: %s", error)
            return "Code example for demonstration purposes."

    async def extract_source_summary(
        self, source_id: str, content: str, max_length: int = 500
    ) -> str:
        default_summary = f"Content from {source_id}"
        if not content.strip():
            return default_summary

        prompt = f"""<source_content>
{content[:25000]}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose."""
        messages = [
            {
                "role": "system",
                "content": "You provide concise library and framework summaries.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            summary = await self.embedding_service.chat_complete(messages, max_tokens=150)
            if len(summary) > max_length:
                return f"{summary[:max_length]}..."
            return summary
        except Exception as error:
            logger.error("Error generating summary for %s: %s", source_id, error)
            return default_summary


def _normalize_url(url: str) -> str:
    return urldefrag(url)[0]


def _backtick_positions(markdown_content: str) -> list[int]:
    positions: list[int] = []
    start_offset = 3 if markdown_content.strip().startswith("```") else 0
    position = start_offset
    while True:
        position = markdown_content.find("```", position)
        if position == -1:
            return positions
        positions.append(position)
        position += 3


def _parse_code_section(code_section: str) -> tuple[str, str]:
    lines = code_section.split("\n", 1)
    if len(lines) == 1:
        return "", code_section.strip()
    first_line = lines[0].strip()
    if first_line and " " not in first_line and len(first_line) < 20:
        return first_line, lines[1].strip()
    return "", code_section.strip()
