"""Typed async REST adapter for the deployed Crawl4AI 0.8.6 service."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from crawl4ai_mcp.config import Settings, get_settings
from crawl4ai_mcp.services.contracts import (
    CrawlDocument,
    CrawlFailure,
    CrawlerPort,
    RemoteLink,
)


class Crawl4AIProviderError(RuntimeError):
    """Structured failure from a Crawl4AI HTTP operation."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.status_code = status_code
        self.details = details or {}


class Crawl4AIClient(CrawlerPort):
    """Crawl4AI HTTP client with bounded client-side batching/concurrency."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.crawl4ai_base_url.rstrip("/")
        self.api_token = self.settings.crawl4ai_api_token
        self.timeout_seconds = self.settings.crawl4ai_timeout_seconds
        self.max_batch_size = min(100, self.settings.crawl4ai_max_batch_size)
        self._client = client
        self._owns_client = client is None

    async def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            timeout = httpx.Timeout(
                self.timeout_seconds,
                connect=min(10.0, self.timeout_seconds),
            )
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                headers=headers,
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close only the HTTP client owned by this adapter."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> dict[str, Any]:
        """Verify the remote Crawl4AI health endpoint."""

        client = await self._client_or_create()
        try:
            response = await client.get("/health")
        except httpx.TimeoutException as exc:
            raise Crawl4AIProviderError(
                "Crawl4AI health request timed out", operation="health"
            ) from exc
        except httpx.RequestError as exc:
            raise Crawl4AIProviderError(
                "Crawl4AI health request failed", operation="health"
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise Crawl4AIProviderError(
                f"Crawl4AI health returned HTTP {response.status_code}",
                operation="health",
                status_code=response.status_code,
                details={"response_excerpt": response.text[:500]},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise Crawl4AIProviderError(
                "Crawl4AI health returned malformed JSON", operation="health"
            ) from exc
        if not isinstance(payload, dict):
            raise Crawl4AIProviderError(
                "Crawl4AI health returned a non-object payload", operation="health"
            )
        return payload

    async def crawl_one(self, url: str) -> list[CrawlDocument]:
        """Crawl one URL through the remote /crawl endpoint."""

        return await self._crawl_batch([url])

    async def crawl_many(
        self, urls: Sequence[str], *, max_concurrent: int
    ) -> list[CrawlDocument]:
        """Crawl URLs in batches of at most 100 with bounded concurrency."""

        values = list(urls)
        if not values:
            return []
        batches = [
            values[offset : offset + self.max_batch_size]
            for offset in range(0, len(values), self.max_batch_size)
        ]
        semaphore = asyncio.Semaphore(max(1, max_concurrent))

        async def run_batch(batch: list[str]) -> list[CrawlDocument]:
            async with semaphore:
                return await self._crawl_batch(batch)

        results = await asyncio.gather(*(run_batch(batch) for batch in batches))
        return [document for batch in results for document in batch]

    async def _crawl_batch(self, urls: list[str]) -> list[CrawlDocument]:
        client = await self._client_or_create()
        payload = {
            "urls": urls,
            "browser_config": {
                "type": "BrowserConfig",
                "params": {"headless": True},
            },
            "crawler_config": {
                "type": "CrawlerRunConfig",
                "params": {"stream": False, "cache_mode": "bypass"},
            },
        }
        try:
            response = await client.post("/crawl", json=payload)
        except httpx.TimeoutException:
            return [_failure(url, "Crawl4AI crawl request timed out") for url in urls]
        except httpx.RequestError as exc:
            return [_failure(url, "Crawl4AI crawl request failed", details={"error_type": type(exc).__name__}) for url in urls]

        if response.status_code < 200 or response.status_code >= 300:
            return [
                _failure(
                    url,
                    f"Crawl4AI returned HTTP {response.status_code}",
                    status_code=response.status_code,
                    details={"response_excerpt": response.text[:500]},
                )
                for url in urls
            ]
        try:
            body = response.json()
        except ValueError:
            return [_failure(url, "Crawl4AI returned malformed JSON") for url in urls]

        raw_results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(raw_results, list) or not raw_results:
            return [_failure(url, "Crawl4AI returned empty results") for url in urls]

        documents: list[CrawlDocument] = []
        for index, requested_url in enumerate(urls):
            item = raw_results[index] if index < len(raw_results) else None
            if not isinstance(item, dict):
                documents.append(_failure(requested_url, "Crawl4AI result item is missing"))
                continue
            documents.append(self._normalize_result(item, requested_url))
        return documents

    def _normalize_result(self, item: dict[str, Any], requested_url: str) -> CrawlDocument:
        url = str(item.get("url") or requested_url)
        status_code = item.get("status_code")
        if not isinstance(status_code, int):
            status_code = None
        success = bool(item.get("success"))
        markdown_payload = item.get("markdown")
        fit_markdown: str | None = None
        raw_markdown: str | None = None
        if isinstance(markdown_payload, dict):
            fit_value = markdown_payload.get("fit_markdown")
            raw_value = markdown_payload.get("raw_markdown")
            fit_markdown = fit_value if isinstance(fit_value, str) else None
            raw_markdown = raw_value if isinstance(raw_value, str) else None
        elif isinstance(markdown_payload, str):
            raw_markdown = markdown_payload
        markdown = fit_markdown or raw_markdown or ""
        links_payload = item.get("links")
        if links_payload is None and isinstance(markdown_payload, dict):
            links_payload = markdown_payload.get("links")
        links = _normalize_links(links_payload)
        if not success or not markdown.strip():
            return _failure(
                url,
                str(item.get("error_message") or "Crawl4AI returned no Markdown"),
                status_code=status_code,
                details={"remote_success": success},
                markdown=markdown,
                raw_markdown=raw_markdown,
                links=links,
            )
        raw_metadata = item.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        return CrawlDocument(
            url=url,
            success=True,
            markdown=markdown,
            raw_markdown=raw_markdown,
            links=links,
            title=item.get("title") if isinstance(item.get("title"), str) else None,
            status_code=status_code,
            content_type=item.get("content_type") if isinstance(item.get("content_type"), str) else None,
            language=item.get("language") if isinstance(item.get("language"), str) else None,
            metadata=metadata,
        )


def _normalize_links(value: Any) -> list[RemoteLink]:
    if not isinstance(value, dict):
        return []
    links: list[RemoteLink] = []
    for key, internal in (("internal", True), ("external", False)):
        items = value.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("href"), str):
                continue
            links.append(
                RemoteLink(
                    href=item["href"],
                    text=item.get("text") if isinstance(item.get("text"), str) else None,
                    title=item.get("title") if isinstance(item.get("title"), str) else None,
                    rel=item.get("rel") if isinstance(item.get("rel"), str) else None,
                    internal=internal,
                )
            )
    return links


def _failure(
    url: str,
    message: str,
    *,
    status_code: int | None = None,
    details: dict[str, Any] | None = None,
    markdown: str = "",
    raw_markdown: str | None = None,
    links: list[RemoteLink] | None = None,
) -> CrawlDocument:
    return CrawlDocument(
        url=url,
        success=False,
        markdown=markdown,
        raw_markdown=raw_markdown,
        links=links or [],
        status_code=status_code,
        failure=CrawlFailure(
            url=url,
            error_message=message,
            status_code=status_code,
            details=details or {},
        ),
    )
