"""Crawling service for web content extraction and processing."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from urllib.parse import urldefrag, urlparse
import xml.etree.ElementTree as ET

import httpx

from crawl4ai_mcp.services.contracts import (
    ChatGeneratorPort,
    CrawlDocument,
    CrawlerPort,
)

logger = logging.getLogger(__name__)

MAX_FAILURES_TO_LOG = 5


class CrawlingService:
    """Service for crawling web content and extracting structured data."""

    def __init__(
        self,
        crawler: CrawlerPort,
        chat_generator: ChatGeneratorPort,
        settings: Any,
    ) -> None:
        self.crawler = crawler
        self.chat_generator = chat_generator
        self.settings = settings

    def is_sitemap(self, url: str) -> bool:
        """Check if URL points to an XML sitemap."""
        return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path

    def is_txt(self, url: str) -> bool:
        """Check if URL points to a text or markdown file."""
        return url.endswith(".txt")

    async def parse_sitemap(
        self, sitemap_url: str, client: httpx.AsyncClient | None = None
    ) -> list[str]:
        """Fetch and parse XML sitemap asynchronously using httpx."""
        try:
            if client is not None:
                response = await client.get(sitemap_url, timeout=30.0)
            else:
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    response = await http_client.get(sitemap_url)

            if response.status_code != 200:
                logger.warning(
                    "Failed to fetch sitemap %s: status %s",
                    sitemap_url,
                    response.status_code,
                )
                return []

            tree = ET.fromstring(response.content)
            urls = [
                loc.text.strip()
                for loc in tree.findall(".//{*}loc")
                if loc.text and loc.text.strip()
            ]
            return urls
        except Exception as error:
            logger.error("Error parsing sitemap XML from %s: %s", sitemap_url, error)
            return []

    async def crawl_markdown_file(self, url: str) -> list[CrawlDocument]:
        """Crawl a single markdown or text file."""
        if self.crawler is None:
            logger.error("CrawlerPort is not configured on CrawlingService")
            return []

        try:
            docs = await self.crawler.crawl_one(url)
        except Exception as error:
            logger.error("Error crawling markdown file %s: %s", url, error)
            return []

        return [doc for doc in docs if doc.success and doc.markdown]

    async def crawl_batch(
        self, urls: Sequence[str], max_concurrent: int = 10
    ) -> list[CrawlDocument]:
        """Crawl a batch of URLs using CrawlerPort."""
        if self.crawler is None:
            logger.error("CrawlerPort is not configured on CrawlingService")
            return []

        try:
            docs = await self.crawler.crawl_many(urls, max_concurrent=max_concurrent)
        except Exception as error:
            logger.error("Error in crawl_batch: %s", error)
            return []

        successful_docs = [doc for doc in docs if doc.success and doc.markdown]
        failed_docs = [doc for doc in docs if not doc.success or not doc.markdown]

        for failed_doc in failed_docs[:MAX_FAILURES_TO_LOG]:
            err_msg = (
                failed_doc.failure.error_message
                if failed_doc.failure
                else "No markdown content"
            )
            logger.error("Failed crawl: %s - %s", failed_doc.url, err_msg)

        return successful_docs

    async def crawl_recursive_internal_links(
        self,
        start_urls: Sequence[str],
        max_depth: int = 3,
        max_concurrent: int = 10,
    ) -> list[CrawlDocument]:
        """Recursively crawl internal links via client-side BFS using CrawlerPort."""
        if self.crawler is None:
            logger.error("CrawlerPort is not configured on CrawlingService")
            return []

        visited: set[str] = set()
        current_urls = {_normalize_url(url) for url in start_urls if url}
        results_all: list[CrawlDocument] = []

        for _ in range(max_depth):
            urls_to_crawl = [url for url in current_urls if url not in visited]
            if not urls_to_crawl:
                break

            for url in urls_to_crawl:
                visited.add(url)

            docs = await self.crawler.crawl_many(
                urls_to_crawl, max_concurrent=max_concurrent
            )
            next_level_urls: set[str] = set()

            for doc in docs:
                normalized_url = _normalize_url(doc.url)
                visited.add(normalized_url)

                if doc.success and doc.markdown:
                    results_all.append(doc)
                    for link in doc.links:
                        href = _normalize_url(link.href)
                        if not href:
                            continue
                        is_internal = (
                            link.internal
                            if link.internal is not None
                            else _is_same_domain(href, start_urls)
                        )
                        if is_internal and href not in visited:
                            next_level_urls.add(href)

            current_urls = next_level_urls

        return results_all

    def extract_code_blocks(
        self, markdown_content: str, min_length: int = 1000
    ) -> list[dict[str, Any]]:
        """Extract code blocks with surrounding context from markdown."""
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
                        "context_before": markdown_content[
                            max(0, start_pos - 1000) : start_pos
                        ].strip(),
                        "context_after": markdown_content[
                            end_pos + 3 : end_pos + 1003
                        ].strip(),
                        "start_char": start_pos,
                        "end_char": end_pos + 3,
                    }
                )
            index += 2
        return code_blocks

    async def generate_code_example_summary(
        self, code: str, context_before: str, context_after: str
    ) -> str:
        """Generate summary for a code example using ChatGeneratorPort."""
        if self.chat_generator is None:
            return "Code example for demonstration purposes."

        if hasattr(self.chat_generator, "generate_code_example_summary"):
            try:
                return await self.chat_generator.generate_code_example_summary(
                    code, context_before, context_after
                )
            except Exception as error:
                logger.error("Error generating code example summary via port: %s", error)

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
            return await self.chat_generator.chat_complete(messages, max_tokens=100)
        except Exception as error:
            logger.error("Error generating code example summary: %s", error)
            return "Code example for demonstration purposes."

    async def extract_source_summary(
        self, source_id: str, content: str, max_length: int = 500
    ) -> str:
        """Extract high-level summary for a documentation source."""
        default_summary = f"Content from {source_id}"
        if not content.strip():
            return default_summary

        if self.chat_generator is None:
            return default_summary

        if hasattr(self.chat_generator, "extract_source_summary"):
            try:
                return await self.chat_generator.extract_source_summary(
                    source_id, content, max_length=max_length
                )
            except Exception as error:
                logger.error("Error extracting source summary via port: %s", error)

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
            summary = await self.chat_generator.chat_complete(messages, max_tokens=150)
            if len(summary) > max_length:
                return f"{summary[:max_length]}..."
            return summary
        except Exception as error:
            logger.error("Error generating summary for %s: %s", source_id, error)
            return default_summary


def _normalize_url(url: str) -> str:
    return urldefrag(url)[0]


def _is_same_domain(url: str, start_urls: Sequence[str]) -> bool:
    target_netloc = urlparse(url).netloc
    if not target_netloc:
        return False
    return any(urlparse(start).netloc == target_netloc for start in start_urls)


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
