"""Strict asynchronous client for the deployed Unified-ML service."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from crawl4ai_mcp.config import Settings, get_settings
from crawl4ai_mcp.services.contracts import (
    EmbeddingPort,
    GlinerExtraction,
    GlinerPort,
    RerankerPort,
    RerankResult,
)


class UnifiedMLProviderError(RuntimeError):
    """Structured failure from a Unified-ML HTTP operation."""

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


class EmbeddingsResponse(BaseModel):
    """Exact response contract for POST /embeddings."""

    model_config = ConfigDict(extra="allow")

    embeddings: list[list[float]]
    model: str
    dimensions: int


class RerankResponse(BaseModel):
    """Exact response contract for POST /rerank."""

    model_config = ConfigDict(extra="allow")

    results: list[RerankResult]


class ExtractResponse(BaseModel):
    """Exact response contract for POST /extract."""

    model_config = ConfigDict(extra="allow")

    results: GlinerExtraction


class UnifiedMLClient(EmbeddingPort, RerankerPort, GlinerPort):
    """Unified-ML embeddings, reranking, and GLiNER adapter."""

    _reranker_model = "ms-marco-MultiBERT-L-12"
    _gliner_model = "fastino/gliner2-multi-v1"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.unified_ml_base_url.rstrip("/")
        self.embedding_model = self.settings.unified_ml_embed_model
        self.embedding_dimensions = self.settings.unified_ml_embedding_dimensions
        self.batch_size = self.settings.unified_ml_batch_size
        self._client = client
        self._owns_client = client is None

    async def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            timeout = httpx.Timeout(
                self.settings.unified_ml_timeout_seconds,
                connect=min(10.0, self.settings.unified_ml_timeout_seconds),
            )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the client created by this adapter."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = await self._client_or_create()
        try:
            response = await client.request(method, path, json=json_body)
        except httpx.TimeoutException as exc:
            raise UnifiedMLProviderError(
                f"Unified-ML {operation} timed out",
                operation=operation,
                details={"error_type": type(exc).__name__},
            ) from exc
        except httpx.RequestError as exc:
            raise UnifiedMLProviderError(
                f"Unified-ML {operation} request failed",
                operation=operation,
                details={"error_type": type(exc).__name__},
            ) from exc

        if response.status_code < 200 or response.status_code >= 300:
            excerpt = response.text[:500]
            raise UnifiedMLProviderError(
                f"Unified-ML {operation} returned HTTP {response.status_code}: {excerpt}",
                operation=operation,
                status_code=response.status_code,
                details={"response_excerpt": excerpt},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UnifiedMLProviderError(
                f"Unified-ML {operation} returned malformed JSON",
                operation=operation,
            ) from exc
        if not isinstance(payload, dict):
            raise UnifiedMLProviderError(
                f"Unified-ML {operation} returned a non-object payload",
                operation=operation,
                details={"payload_type": type(payload).__name__},
            )
        return payload

    async def health_check(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Check health and enforce the configured embedding/model contract."""

        health = await self._request_json("GET", "/health", operation="health")
        info = await self._request_json("GET", "/info", operation="info")

        model = _find_value(info, {"embedding_model", "embed_model", "model"})
        dimensions = _find_value(
            info, {"embedding_dimensions", "embedding_dimension", "dimensions"}
        )
        if model != self.embedding_model:
            raise UnifiedMLProviderError(
                f"Unified-ML embedding model mismatch: expected {self.embedding_model!r}, got {model!r}",
                operation="info",
            )
        if dimensions is None:
            probe = await self._embed(["passage: startup dimension probe"])
            actual_dimensions = len(probe[0]) if probe else 0
        else:
            try:
                actual_dimensions = int(dimensions)
            except (TypeError, ValueError) as exc:
                raise UnifiedMLProviderError(
                    "Unified-ML /info reported invalid embedding dimensions",
                    operation="info",
                ) from exc
        if actual_dimensions != self.embedding_dimensions:
            raise UnifiedMLProviderError(
                f"Unified-ML embedding dimension mismatch: expected {self.embedding_dimensions}, got {actual_dimensions}",
                operation="info",
            )

        if health.get("status") not in (None, "ok", "healthy"):
            raise UnifiedMLProviderError(
                f"Unified-ML health status is {health.get('status')!r}",
                operation="health",
            )
        for loaded_key in ("embed_loaded", "rerank_loaded", "ner_loaded"):
            if loaded_key in health and health[loaded_key] is not True:
                raise UnifiedMLProviderError(
                    f"Unified-ML health reported {loaded_key}=false",
                    operation="health",
                )
        health_text = json.dumps({"health": health, "info": info}, sort_keys=True).lower()
        for required_model in (
            self.embedding_model,
            self._reranker_model,
            self._gliner_model,
        ):
            if required_model.lower() not in health_text:
                raise UnifiedMLProviderError(
                    f"Unified-ML /health did not report loaded model {required_model!r}",
                    operation="health",
                )
        return health, info

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = await self._request_json(
            "POST",
            "/embeddings",
            operation="embeddings",
            json_body={"texts": texts},
        )
        try:
            response = EmbeddingsResponse.model_validate(payload)
        except ValidationError as exc:
            raise UnifiedMLProviderError(
                "Unified-ML /embeddings response is missing required fields",
                operation="embeddings",
                details={"validation": str(exc)},
            ) from exc
        if response.model != self.embedding_model:
            raise UnifiedMLProviderError(
                f"Unified-ML embedding model mismatch: expected {self.embedding_model!r}, got {response.model!r}",
                operation="embeddings",
            )
        if response.dimensions != self.embedding_dimensions:
            raise UnifiedMLProviderError(
                f"Unified-ML embedding dimension mismatch: expected {self.embedding_dimensions}, got {response.dimensions}",
                operation="embeddings",
            )
        if len(response.embeddings) != len(texts):
            raise UnifiedMLProviderError(
                f"Unified-ML returned {len(response.embeddings)} vectors for {len(texts)} texts",
                operation="embeddings",
            )
        for index, vector in enumerate(response.embeddings):
            if len(vector) != self.embedding_dimensions:
                raise UnifiedMLProviderError(
                    f"Unified-ML vector {index} has {len(vector)} dimensions; expected {self.embedding_dimensions}",
                    operation="embeddings",
                )
        return response.embeddings

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages with the fixed E5 passage prefix in batches of 32."""

        values = [text if text.startswith("passage: ") else f"passage: {text}" for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            vectors.extend(await self._embed(values[start : start + self.batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query with the fixed E5 query prefix."""

        vectors = await self._embed([text if text.startswith("query: ") else f"query: {text}"])
        if not vectors:
            raise UnifiedMLProviderError("Unified-ML returned no query vector", operation="embeddings")
        return vectors[0]

    async def rerank(self, query: str, texts: Sequence[str]) -> list[RerankResult]:
        """Rerank candidate text without applying an absolute cutoff."""

        if not texts:
            return []
        payload = await self._request_json(
            "POST",
            "/rerank",
            operation="rerank",
            json_body={"query": query, "texts": list(texts)},
        )
        try:
            response = RerankResponse.model_validate(payload)
        except ValidationError as exc:
            raise UnifiedMLProviderError(
                "Unified-ML /rerank response is missing required fields",
                operation="rerank",
                details={"validation": str(exc)},
            ) from exc
        return response.results

    async def extract(
        self,
        text: str,
        *,
        entities: Sequence[str],
        relations: Sequence[str],
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
    ) -> GlinerExtraction:
        """Run the configured GLiNER extraction request."""

        payload = await self._request_json(
            "POST",
            "/extract",
            operation="extract",
            json_body={
                "text": text,
                "entities": list(entities),
                "relations": list(relations),
                "threshold": threshold,
                "include_confidence": include_confidence,
                "include_spans": include_spans,
            },
        )
        try:
            return ExtractResponse.model_validate(payload).results
        except ValidationError as exc:
            raise UnifiedMLProviderError(
                "Unified-ML /extract response is missing typed results.entities or results.relation_extraction",
                operation="extract",
                details={"validation": str(exc)},
            ) from exc


def _find_value(payload: Any, keys: set[str]) -> Any:
    """Find a named value in nested provider metadata."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in keys:
                return value
            found = _find_value(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_value(value, keys)
            if found is not None:
                return found
    return None
