"""Search service for document and code example retrieval."""

import logging
from typing import Any, Dict, List, Optional

from flashrank import RerankRequest

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import SearchRequest, SearchResponse, SearchResult, SearchType
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.services.kuzu_search_backend import KuzuSearchBackend

logger = logging.getLogger(__name__)

_RRF_K = 60


class SearchService:
    """Service for searching documents and code examples."""

    def __init__(
        self,
        client: Any,
        settings: Optional[Any] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.client = client
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(self.settings)
        self.backend = KuzuSearchBackend(client)

    async def search_documents(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
        use_hybrid_search: Optional[bool] = None,
    ) -> List[SearchResult]:
        query_embedding = await self.embedding_service.create_embedding(query)
        hybrid_search = (
            self.settings.use_hybrid_search
            if use_hybrid_search is None
            else use_hybrid_search
        )

        vector_rows = self.backend.search_documents_by_vector(
            query_embedding,
            match_count * 2 if hybrid_search else match_count,
            source_id,
        )
        if hybrid_search:
            text_rows = self.backend.search_documents_by_text(
                query,
                match_count * 2,
                source_id,
            )
            rows = _rrf_fuse(vector_rows, text_rows, match_count)
        else:
            rows = vector_rows

        rows = _filter_metadata(rows, filter_metadata)
        return [_to_search_result(row) for row in rows[:match_count]]

    async def search_code_examples(
        self,
        query: str,
        language: Optional[str] = None,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        enhanced_query = f"Code example for {query}"
        if language:
            enhanced_query += f" in {language}"
        enhanced_query += f"\n\nSummary: Example code showing {query}"

        query_embedding = await self.embedding_service.create_embedding(enhanced_query)
        rows = self.backend.search_code_by_vector(query_embedding, match_count, source_id)
        rows = _filter_metadata(rows, filter_metadata)
        if language:
            rows = [
                row for row in rows if row.get("metadata", {}).get("language") == language
            ]
        return rows[:match_count]

    async def perform_search(
        self,
        request: SearchRequest,
        search_type: SearchType = SearchType.SEMANTIC,
        include_code_examples: bool = False,
    ) -> SearchResponse:
        try:
            results = await self.search_documents(
                query=request.query,
                match_count=request.num_results,
                source_id=request.source,
                use_hybrid_search=(search_type == SearchType.HYBRID),
            )
            if request.semantic_threshold > 0:
                results = [
                    result
                    for result in results
                    if result.similarity_score >= request.semantic_threshold
                ]

            code_results: list[SearchResult] = []
            if include_code_examples and self.settings.use_agentic_rag:
                for code_row in await self.search_code_examples(
                    query=request.query,
                    match_count=request.num_results,
                    source_id=request.source,
                ):
                    metadata = code_row.get("metadata", {})
                    code_results.append(
                        SearchResult(
                            content=code_row.get("content", ""),
                            url=code_row.get("url", ""),
                            source=code_row.get("source_id", ""),
                            chunk_number=code_row.get("chunk_number", 0),
                            similarity_score=code_row.get("similarity", 0.0),
                            metadata={"type": "code_example", **metadata},
                        )
                    )

            all_results = sorted(
                results + code_results,
                key=lambda item: item.similarity_score,
                reverse=True,
            )
            return SearchResponse(
                success=True,
                results=all_results[: request.num_results],
                total_results=len(all_results[: request.num_results]),
                search_type=search_type,
            )
        except Exception as error:
            return SearchResponse(
                success=False,
                results=[],
                total_results=0,
                search_type=search_type,
                error=str(error),
            )

    async def rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        reranking_model: Any,
        threshold: float = 0.3,
    ) -> List[SearchResult]:
        if not results or not reranking_model:
            return results

        try:
            if hasattr(reranking_model, "rerank"):
                passages = [
                    {"id": index, "text": result.content}
                    for index, result in enumerate(results)
                ]
                reranked_passages = reranking_model.rerank(
                    RerankRequest(query=query, passages=passages)
                )
                score_map = {
                    passage["id"]: float(passage.get("score", 0.0))
                    for passage in reranked_passages
                }
            else:
                pairs = [[query, result.content] for result in results]
                scores = reranking_model.predict(pairs)
                score_map = {
                    index: float(score) for index, score in enumerate(scores)
                }

            reranked: list[SearchResult] = []
            for index, result in enumerate(results):
                result.rerank_score = score_map.get(index, 0.0)
                if result.rerank_score >= threshold:
                    reranked.append(result)

            reranked.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
            return reranked
        except Exception as error:
            logger.error("Error during reranking: %s", error)
            return results


def _to_search_result(row: Dict[str, Any]) -> SearchResult:
    return SearchResult(
        content=row.get("content", ""),
        url=row.get("url", ""),
        source=row.get("source_id", ""),
        chunk_number=row.get("chunk_number", 0),
        similarity_score=row.get("similarity", 0.0),
        metadata=row.get("metadata", {}),
    )


def _filter_metadata(
    rows: List[Dict[str, Any]], filter_metadata: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not filter_metadata:
        return rows
    return [
        row
        for row in rows
        if all(row.get("metadata", {}).get(key) == value for key, value in filter_metadata.items())
    ]


def _rrf_fuse(
    vector_rows: List[Dict[str, Any]],
    text_rows: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    scores: dict[str, float] = {}
    merged: dict[str, Dict[str, Any]] = {}

    for rank, row in enumerate(vector_rows):
        key = f"{row['url']}::{row['chunk_number']}"
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        merged[key] = row

    for rank, row in enumerate(text_rows):
        key = f"{row['url']}::{row['chunk_number']}"
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        merged.setdefault(key, row)

    ranked_keys = sorted(scores, key=lambda item: scores[item], reverse=True)[:limit]
    fused_rows: list[Dict[str, Any]] = []
    for key in ranked_keys:
        row = merged[key].copy()
        row["similarity"] = scores[key]
        fused_rows.append(row)
    return fused_rows


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)
