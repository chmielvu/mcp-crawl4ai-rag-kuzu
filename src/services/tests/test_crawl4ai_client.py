"""Tests for Crawl4AIClient REST adapter and payload normalization."""

import json
from typing import Any

import httpx
import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.services.crawl4ai_client import (
    Crawl4AIClient,
    Crawl4AIProviderError,
    _normalize_links,
)


def _make_mock_client(handler: Any) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="http://localhost:11235")


@pytest.mark.asyncio
async def test_health_check_success(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "healthy", "version": "0.8.6"})

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    res = await adapter.health_check()
    assert res["status"] == "healthy"
    assert res["version"] == "0.8.6"


@pytest.mark.asyncio
async def test_health_check_http_error(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    with pytest.raises(Crawl4AIProviderError) as exc_info:
        await adapter.health_check()
    assert exc_info.value.status_code == 500
    assert exc_info.value.operation == "health"


@pytest.mark.asyncio
async def test_health_check_timeout(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timeout connecting to Crawl4AI")

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    with pytest.raises(Crawl4AIProviderError, match="timed out"):
        await adapter.health_check()


@pytest.mark.asyncio
async def test_health_check_malformed_json(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not a json")

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    with pytest.raises(Crawl4AIProviderError, match="malformed JSON"):
        await adapter.health_check()


@pytest.mark.asyncio
async def test_health_check_non_object_payload(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["item1", "item2"])

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    with pytest.raises(Crawl4AIProviderError, match="non-object payload"):
        await adapter.health_check()


@pytest.mark.asyncio
async def test_crawl_one_normalizes_fit_markdown_and_links(test_settings: Settings) -> None:
    target_url = "https://example.com/docs/page"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/crawl"
        payload = json.loads(request.content)
        assert payload["urls"] == [target_url]
        assert payload["browser_config"]["type"] == "BrowserConfig"
        assert payload["crawler_config"]["params"]["cache_mode"] == "bypass"

        response_body = {
            "results": [
                {
                    "url": target_url,
                    "success": True,
                    "status_code": 200,
                    "markdown": {
                        "fit_markdown": "# Fit Markdown Header\n\nConcise content.",
                        "raw_markdown": "# Raw Markdown Header\n\nVerbose raw content.",
                        "links": {
                            "internal": [
                                {"href": "https://example.com/docs/sub", "text": "Sub", "title": "Subpage"},
                                {"invalid": "missing href"},
                            ],
                            "external": [
                                {"href": "https://github.com/org/repo", "text": "Repo", "rel": "nofollow"}
                            ],
                        },
                    },
                    "title": "Page Title",
                    "content_type": "text/html",
                    "language": "en",
                    "metadata": {"og:title": "OpenGraph Title"},
                }
            ]
        }
        return httpx.Response(200, json=response_body)

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    docs = await adapter.crawl_one(target_url)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.url == target_url
    assert doc.success is True
    assert doc.markdown == "# Fit Markdown Header\n\nConcise content."
    assert doc.raw_markdown == "# Raw Markdown Header\n\nVerbose raw content."
    assert doc.title == "Page Title"
    assert doc.status_code == 200
    assert doc.content_type == "text/html"
    assert doc.language == "en"
    assert doc.metadata == {"og:title": "OpenGraph Title"}
    assert len(doc.links) == 2
    assert doc.links[0].href == "https://example.com/docs/sub"
    assert doc.links[0].internal is True
    assert doc.links[1].href == "https://github.com/org/repo"
    assert doc.links[1].internal is False


@pytest.mark.asyncio
async def test_crawl_one_raw_markdown_fallback_and_string_markdown(test_settings: Settings) -> None:
    target_url = "https://example.com/page2"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": target_url,
                        "success": True,
                        "status_code": 200,
                        "markdown": "# String Markdown Content",
                        "links": {
                            "internal": [{"href": "https://example.com/other", "text": "Other"}]
                        },
                    }
                ]
            },
        )

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    docs = await adapter.crawl_one(target_url)

    assert len(docs) == 1
    assert docs[0].markdown == "# String Markdown Content"
    assert docs[0].raw_markdown == "# String Markdown Content"
    assert len(docs[0].links) == 1
    assert docs[0].links[0].href == "https://example.com/other"


@pytest.mark.asyncio
async def test_crawl_one_handles_failure_payload(test_settings: Settings) -> None:
    target_url = "https://example.com/fail"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": target_url,
                        "success": False,
                        "status_code": 404,
                        "error_message": "Page not found",
                        "markdown": "",
                    }
                ]
            },
        )

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    docs = await adapter.crawl_one(target_url)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.success is False
    assert doc.status_code == 404
    assert doc.failure is not None
    assert doc.failure.error_message == "Page not found"
    assert doc.failure.status_code == 404


@pytest.mark.asyncio
async def test_crawl_one_handles_http_500_response(test_settings: Settings) -> None:
    target_url = "https://example.com/server-error"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Service Crash")

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    docs = await adapter.crawl_one(target_url)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.success is False
    assert doc.status_code == 500
    assert doc.failure is not None
    assert "HTTP 500" in doc.failure.error_message


@pytest.mark.asyncio
async def test_crawl_one_handles_timeout_and_network_error(test_settings: Settings) -> None:
    target_url = "https://example.com/timeout"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timeout")

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    docs = await adapter.crawl_one(target_url)

    assert len(docs) == 1
    assert docs[0].success is False
    assert docs[0].failure is not None
    assert "timed out" in docs[0].failure.error_message


@pytest.mark.asyncio
async def test_crawl_one_handles_malformed_json_and_empty_results(test_settings: Settings) -> None:
    def handler_malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client1 = _make_mock_client(handler_malformed)
    adapter1 = Crawl4AIClient(settings=test_settings, client=client1)
    docs1 = await adapter1.crawl_one("https://example.com/1")
    assert len(docs1) == 1
    assert docs1[0].success is False
    assert "malformed JSON" in docs1[0].failure.error_message

    def handler_empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client2 = _make_mock_client(handler_empty)
    adapter2 = Crawl4AIClient(settings=test_settings, client=client2)
    docs2 = await adapter2.crawl_one("https://example.com/2")
    assert len(docs2) == 1
    assert docs2[0].success is False
    assert "empty results" in docs2[0].failure.error_message


@pytest.mark.asyncio
async def test_crawl_many_batching_and_concurrency(test_settings: Settings) -> None:
    urls = [f"https://example.com/item/{i}" for i in range(25)]
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        req_urls = payload["urls"]
        calls.append(req_urls)
        results = [
            {
                "url": u,
                "success": True,
                "status_code": 200,
                "markdown": f"# Content for {u}",
            }
            for u in req_urls
        ]
        return httpx.Response(200, json={"results": results})

    client = _make_mock_client(handler)
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    adapter.max_batch_size = 10

    docs = await adapter.crawl_many(urls, max_concurrent=2)
    assert len(docs) == 25
    assert len(calls) == 3
    assert len(calls[0]) == 10
    assert len(calls[1]) == 10
    assert len(calls[2]) == 5
    for i, doc in enumerate(docs):
        assert doc.url == f"https://example.com/item/{i}"
        assert doc.success is True


@pytest.mark.asyncio
async def test_crawl_many_empty_urls(test_settings: Settings) -> None:
    client = _make_mock_client(lambda r: httpx.Response(200, json={"results": []}))
    adapter = Crawl4AIClient(settings=test_settings, client=client)
    docs = await adapter.crawl_many([], max_concurrent=5)
    assert docs == []


def test_normalize_links_edge_cases() -> None:
    assert _normalize_links(None) == []
    assert _normalize_links("invalid") == []
    assert _normalize_links({"internal": "not a list"}) == []
    assert _normalize_links({"internal": [{"missing": "href"}]}) == []
