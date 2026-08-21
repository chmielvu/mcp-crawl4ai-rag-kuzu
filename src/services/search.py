"""Search orchestration for FalkorDB chunks and Unified-ML providers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.services.contracts import (
    EmbeddingPort,
    RerankResult,
    RerankerPort,
    SearchBackendPort,
    SearchHit,
)

logger = logging.getLogger(__name__)
_RRF_K = 60


class SearchService:
    """Coordinate vector, full-text, RRF, and remote reranking operations."""

    def __init__(
        self,
        backend: SearchBackendPort,
        embeddings: EmbeddingPort,
        reranker: RerankerPort | None = None,
        settings: Any | None = None,
    ) -> None:
        self.backend = backend
        self.embeddings = embeddings
        self.reranker = reranker
        self.settings = settings or get_settings()

    async def search_documents(
        self,
        query: str,
        *,
        match_count: int = 5,
        site_id: str | None = None,
        use_hybrid: bool | None = None,
    ) -> list[SearchHit]:
        """Retrieve document chunks; no absolute score cutoff is applied."""

        query_embedding = await self.embeddings.embed_query(query)
        vector_hits = await self.backend.search_chunks_by_vector(
            query_embedding, max(match_count * 4, 20), site_id
        )
        hybrid = self.settings.use_hybrid_search if use_hybrid is None else use_hybrid
        if not hybrid:
            return vector_hits[:match_count]

        text_hits = await self.backend.search_chunks_by_text(
            query, max(match_count * 4, 20), site_id
        )
        return _rrf_fuse([vector_hits, text_hits], match_count)

    async def search_code_examples(
        self,
        query: str,
        *,
        language: str | None = None,
        match_count: int = 5,
        site_id: str | None = None,
    ) -> list[SearchHit]:
        """Retrieve code-content chunks from the shared Chunk graph label."""

        query_embedding = await self.embeddings.embed_query(query)
        return await self.backend.search_code_chunks(
            query_embedding, match_count, site_id, language
        )

    async def rerank_hits(
        self,
        query: str,
        hits: Sequence[SearchHit],
    ) -> list[SearchHit]:
        """Apply Unified-ML relative reranking while preserving all candidates."""

        if not hits or self.reranker is None:
            return list(hits)

        reranked: list[RerankResult] = await self.reranker.rerank(
            query, [hit.content for hit in hits]
        )
        by_id = {item.id: item.score for item in reranked}
        updated = [
            hit.model_copy(update={"rerank_score": by_id.get(index)})
            for index, hit in enumerate(hits)
        ]
        return sorted(
            updated,
            key=lambda hit: (
                hit.rerank_score if hit.rerank_score is not None else float("-inf")
            ),
            reverse=True,
        )

    async def _attach_provenance(self, hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Expand chunk hits with grounded entities and relation facts in parallel."""
        provenance_sets = await asyncio.gather(
            *(self.backend.get_chunk_provenance(hit.chunk_id) for hit in hits)
        )
        return [
            hit.model_copy(update={"provenance": provenance})
            for hit, provenance in zip(hits, provenance_sets, strict=True)
        ]

    async def perform_rag_query(
        self,
        query: str,
        *,
        match_count: int = 5,
        site_id: str | None = None,
        use_hybrid: bool | None = None,
        use_reranking: bool | None = None,
    ) -> list[SearchHit]:
        """Retrieve and optionally rerank chunks for a RAG request."""

        hits = await self.search_documents(
            query,
            match_count=max(match_count * 2, 10),
            site_id=site_id,
            use_hybrid=use_hybrid,
        )
        hits = await self._attach_provenance(hits)
        should_rerank = (
            self.settings.use_reranking if use_reranking is None else use_reranking
        )
        if should_rerank:
            hits = await self.rerank_hits(query, hits)
        return hits[:match_count]


def _rrf_fuse(result_sets: Sequence[Sequence[SearchHit]], limit: int) -> list[SearchHit]:
    """Fuse result lists by deterministic chunk ID using reciprocal rank."""

    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for result_set in result_sets:
        for rank, hit in enumerate(result_set):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + _rrf_score(rank)
            hits.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [hits[chunk_id] for chunk_id in ordered[:limit]]


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)
