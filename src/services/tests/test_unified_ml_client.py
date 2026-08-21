"""Tests for UnifiedMLClient REST adapter and payload normalization."""

import json
from typing import Any

import httpx
import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.services.contracts import GlinerExtraction
from crawl4ai_mcp.services.unified_ml_client import (
    UnifiedMLClient,
    UnifiedMLProviderError,
)


def _make_mock_client(handler: Any) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="http://localhost:8000")


@pytest.mark.asyncio
async def test_health_check_success(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "loaded_models": [
                        "intfloat/multilingual-e5-small",
                        "ms-marco-MultiBERT-L-12",
                        "fastino/gliner2-multi-v1",
                    ],
                },
            )
        if request.url.path == "/info":
            return httpx.Response(
                200,
                json={
                    "embedding_model": "intfloat/multilingual-e5-small",
                    "embedding_dimensions": 384,
                },
            )
        return httpx.Response(404)

    client = _make_mock_client(handler)
    adapter = UnifiedMLClient(settings=test_settings, client=client)
    health, info = await adapter.health_check()
    assert health["status"] == "healthy"
    assert info["embedding_dimensions"] == 384


@pytest.mark.asyncio
async def test_health_check_model_mismatch(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={"loaded_models": ["other-model", "ms-marco-MultiBERT-L-12", "fastino/gliner2-multi-v1"]},
            )
        if request.url.path == "/info":
            return httpx.Response(
                200,
                json={"embedding_model": "other-model", "embedding_dimensions": 384},
            )
        return httpx.Response(404)

    client = _make_mock_client(handler)
    adapter = UnifiedMLClient(settings=test_settings, client=client)
    with pytest.raises(UnifiedMLProviderError, match="model mismatch"):
        await adapter.health_check()


@pytest.mark.asyncio
async def test_health_check_dimension_mismatch(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={"loaded_models": ["intfloat/multilingual-e5-small", "ms-marco-MultiBERT-L-12", "fastino/gliner2-multi-v1"]},
            )
        if request.url.path == "/info":
            return httpx.Response(
                200,
                json={"embedding_model": "intfloat/multilingual-e5-small", "embedding_dimensions": 512},
            )
        return httpx.Response(404)

    client = _make_mock_client(handler)
    adapter = UnifiedMLClient(settings=test_settings, client=client)
    with pytest.raises(UnifiedMLProviderError, match="dimension mismatch"):
        await adapter.health_check()


@pytest.mark.asyncio
async def test_embed_passages_prepends_prefix_and_batches(test_settings: Settings) -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embeddings"
        payload = json.loads(request.content)
        texts = payload["texts"]
        calls.append(texts)
        embeddings = [[0.01 * (i + 1)] * 384 for i in range(len(texts))]
        return httpx.Response(
            200,
            json={
                "embeddings": embeddings,
                "model": "intfloat/multilingual-e5-small",
                "dimensions": 384,
            },
        )

    client = _make_mock_client(handler)
    adapter = UnifiedMLClient(settings=test_settings, client=client)
    adapter.batch_size = 2

    input_texts = ["first passage", "passage: already prefixed", "third passage"]
    vectors = await adapter.embed_passages(input_texts)

    assert len(vectors) == 3
    assert len(calls) == 2
    assert calls[0] == ["passage: first passage", "passage: already prefixed"]
    assert calls[1] == ["passage: third passage"]
    for vec in vectors:
        assert len(vec) == 384


@pytest.mark.asyncio
async def test_embed_passages_dimension_validation(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embeddings": [[0.1] * 128],
                "model": "intfloat/multilingual-e5-small",
                "dimensions": 128,
            },
        )

    client = _make_mock_client(handler)
    adapter = UnifiedMLClient(settings=test_settings, client=client)
    with pytest.raises(UnifiedMLProviderError, match="dimension mismatch"):
        await adapter.embed_passages(["test"])


@pytest.mark.asyncio
async def test_embed_query_prepends_query_prefix(test_settings: Settings) -> None:
    received_texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embeddings"
        payload = json.loads(request.content)
        received_texts.extend(payload["texts"])
        return httpx.Response(
            200,
            json={
                "embeddings": [[0.5] * 384],
                "model": "intfloat/multilingual-e5-small",
                "dimensions": 384,
            },
        )

    client = _make_mock_client(handler)
    adapter = UnifiedMLClient(settings=test_settings, client=client)
    vec = await adapter.embed_query("what is fastmcp?")

    assert len(vec) == 384
    assert received_texts == ["query: what is fastmcp?"]

    await adapter.embed_query("query: already prefixed")
    assert received_texts[-1] == "query: already prefixed"


@pytest.mark.asyncio
async def test_rerank_normalizes_results(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        payload = json.loads(request.content)
        assert payload["query"] == "my search query"
        assert payload["texts"] == ["candidate A", "candidate B"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": 1, "text": "candidate B", "score": 0.95},
                    {"id": 0, "text": "candidate A", "score": 0.32},
                ]
            },
        )

    client = _make_mock_client(handler)
    adapter = UnifiedMLClient(settings=test_settings, client=client)
    results = await adapter.rerank("my search query", ["candidate A", "candidate B"])

    assert len(results) == 2
    assert results[0].id == 1
    assert results[0].score == 0.95
    assert results[1].id == 0
    assert results[1].score == 0.32

    empty_res = await adapter.rerank("query", [])
    assert empty_res == []


@pytest.mark.asyncio
async def test_extract_normalizes_gliner_entities_and_relations(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/extract"
        payload = json.loads(request.content)
        assert payload["text"] == "FastMCP is a Python framework."
        assert payload["entities"] == ["technology", "product"]
        assert payload["relations"] == ["implements"]
        assert payload["threshold"] == 0.5

        return httpx.Response(
            200,
            json={
                "results": {
                    "entities": [
                        {
                            "text": "FastMCP",
                            "label": "technology",
                            "score": 0.98,
                            "start": 0,
                            "end": 7,
                        }
                    ],
                    "relation_extraction": [
                        {
                            "source": "FastMCP",
                            "target": "Python",
                            "relation": "implements",
                            "score": 0.91,
                            "fact": "FastMCP implements Python framework",
                        }
                    ],
                    "text": "FastMCP is a Python framework.",
                }
            },
        )

    client = _make_mock_client(handler)
    adapter = UnifiedMLClient(settings=test_settings, client=client)
    extraction = await adapter.extract(
        "FastMCP is a Python framework.",
        entities=["technology", "product"],
        relations=["implements"],
        threshold=0.5,
        include_confidence=True,
        include_spans=True,
    )

    assert isinstance(extraction, GlinerExtraction)
    assert len(extraction.entities) == 1
    assert extraction.entities[0].text == "FastMCP"
    assert extraction.entities[0].label == "technology"
    assert len(extraction.relation_extraction) == 1
    assert extraction.relation_extraction[0].relation == "implements"


@pytest.mark.asyncio
async def test_extract_validation_failure(test_settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": {"invalid": "shape"}})

    client = _make_mock_client(handler)
    adapter = UnifiedMLClient(settings=test_settings, client=client)
    with pytest.raises(UnifiedMLProviderError, match="missing typed results"):
        await adapter.extract(
            "text",
            entities=["technology"],
            relations=["uses"],
            threshold=0.5,
            include_confidence=True,
            include_spans=True,
        )


@pytest.mark.asyncio
async def test_unified_ml_http_errors_and_timeouts(test_settings: Settings) -> None:
    def handler_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client1 = _make_mock_client(handler_500)
    adapter1 = UnifiedMLClient(settings=test_settings, client=client1)
    with pytest.raises(UnifiedMLProviderError) as exc:
        await adapter1.embed_query("test")
    assert exc.value.status_code == 500

    def handler_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timeout")

    client2 = _make_mock_client(handler_timeout)
    adapter2 = UnifiedMLClient(settings=test_settings, client=client2)
    with pytest.raises(UnifiedMLProviderError, match="timed out"):
        await adapter2.embed_query("test")
