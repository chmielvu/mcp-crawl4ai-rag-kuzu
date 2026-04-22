"""Reranking utilities backed by FlashRank."""

import logging
from typing import Any, Dict, List, Optional

from flashrank import Ranker, RerankRequest

from crawl4ai_mcp.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Utility class for reranking search results."""

    def __init__(self, model: Optional[Ranker] = None, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.model = model
        if self.model is None and self.settings.use_reranking:
            self.model = Ranker(
                model_name=self.settings.reranker_model,
                cache_dir=self.settings.reranker_cache_dir,
                max_length=self.settings.reranker_max_length,
            )

    def rerank_results(
        self, query: str, results: List[Dict[str, Any]], content_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """Rerank search results using FlashRank."""
        if not self.model or not results:
            return results

        passages = [
            {"id": index, "text": result.get(content_key, "")}
            for index, result in enumerate(results)
        ]

        try:
            reranked_passages = self.model.rerank(
                RerankRequest(query=query, passages=passages)
            )
            score_map = {
                passage["id"]: float(passage.get("score", 0.0))
                for passage in reranked_passages
            }
            for index, result in enumerate(results):
                result["rerank_score"] = score_map.get(index, 0.0)
            return sorted(
                results,
                key=lambda item: item.get("rerank_score", 0.0),
                reverse=True,
            )
        except Exception as error:
            logger.error("FlashRank reranking failed: %s", error)
            return results

    def filter_by_threshold(
        self,
        results: List[Dict[str, Any]],
        threshold: float,
        score_key: str = "rerank_score",
    ) -> List[Dict[str, Any]]:
        """Filter results by minimum score threshold."""
        return [result for result in results if result.get(score_key, 0.0) >= threshold]
