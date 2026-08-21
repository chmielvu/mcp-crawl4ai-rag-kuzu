"""FalkorDB search backend for vector, full-text, and entity/relationship retrieval."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from crawl4ai_mcp.services.contracts import SearchBackendPort, SearchHit

logger = logging.getLogger(__name__)

# Characters with special syntactic meaning in RediSearch full-text queries
_REDISEARCH_SPECIAL_CHARS = re.compile(r'([,.<>{}[\]"\':;!@#$%^&*()\-+=~|/\\?])')


def escape_redisearch_query(query: str) -> str:
    """Escape special syntax characters for RediSearch full-text search."""
    if not query:
        return ""
    # Strip dangerous control chars and escape punctuation syntax
    cleaned = query.strip()
    return _REDISEARCH_SPECIAL_CHARS.sub(r"\\\1", cleaned)


def _candidate_limit(limit: int, has_filter: bool = False) -> int:
    """Calculate candidate limit to over-fetch for post-ANN property filtering."""
    if has_filter:
        return max(limit * 8, 50)
    return limit


def _convert_vector_score(score: Any) -> float:
    """Convert FalkorDB cosine distance score to similarity score (1 - distance)."""
    try:
        dist = float(score)
        return max(0.0, min(1.0, 1.0 - dist))
    except (ValueError, TypeError):
        return 0.0


def _parse_json_dict(value: Any) -> dict[str, Any]:
    """Safely parse JSON string into dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_rows(result: Any) -> list[list[Any]]:
    """Extract tabular rows from FalkorDB QueryResult or mock object."""
    if result is None:
        return []
    if hasattr(result, "result_set") and isinstance(result.result_set, list):
        return result.result_set
    if isinstance(result, list):
        return result
    return []


def _extract_headers(result: Any) -> list[str]:
    """Extract column header names from FalkorDB QueryResult or mock object."""
    if result is None:
        return []
    if hasattr(result, "header") and result.header:
        headers: list[str] = []
        for col in result.header:
            if isinstance(col, (list, tuple)):
                values = list(col)
                if values:
                    headers.append(str(values[1] if len(values) > 1 else values[0]))
            else:
                headers.append(str(col))
        return headers
    if hasattr(result, "columns") and result.columns:
        return [str(c) for c in result.columns]
    return []


def _query_result_to_dicts(result: Any) -> list[dict[str, Any]]:
    """Convert FalkorDB QueryResult rows to list of dictionaries."""
    rows = _extract_rows(result)
    headers = _extract_headers(result)
    if not headers or not rows:
        return []
    dicts: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            dicts.append(row)
        elif isinstance(row, (list, tuple)):
            dicts.append({headers[i]: row[i] for i in range(min(len(headers), len(row)))})
    return dicts


def _to_search_hit(row: dict[str, Any], similarity_score: float) -> SearchHit:
    """Convert normalized query row dict to typed SearchHit."""
    metadata_raw = row.get("metadata_json") or row.get("metadata") or "{}"
    metadata = _parse_json_dict(metadata_raw)
    page_title = row.get("page_title")
    if page_title and "title" not in metadata:
        metadata["title"] = page_title

    provenance_raw = row.get("provenance") or []
    provenance = provenance_raw if isinstance(provenance_raw, list) else []

    rerank_score = None
    if row.get("rerank_score") is not None:
        try:
            rerank_score = float(row["rerank_score"])
        except (ValueError, TypeError):
            rerank_score = None

    return SearchHit(
        chunk_id=str(row.get("chunk_id") or ""),
        page_id=str(row.get("page_id") or ""),
        site_id=str(row.get("site_id") or ""),
        content=str(row.get("content") or row.get("text") or ""),
        url=str(row.get("url") or ""),
        source=str(row.get("source") or row.get("domain") or row.get("site_id") or ""),
        chunk_number=int(row.get("chunk_number") or row.get("index") or 0),
        similarity_score=similarity_score,
        rerank_score=rerank_score,
        content_type=str(row.get("content_type") or "text"),
        language=row.get("language") or None,
        metadata=metadata,
        provenance=provenance,
    )


class FalkorSearchBackend(SearchBackendPort):
    """FalkorDB vector and full-text search backend implementing SearchBackendPort."""

    def __init__(self, graph: Any, settings: Any | None = None):
        self.graph = graph
        self.settings = settings

    async def _ro_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Execute a read-only Cypher query on the graph."""
        timeout = getattr(self.settings, "falkordb_query_timeout_ms", None)
        if hasattr(self.graph, "ro_query"):
            return await self.graph.ro_query(query, params=params, timeout=timeout)
        if hasattr(self.graph, "query"):
            return await self.graph.query(query, params=params, timeout=timeout)
        if hasattr(self.graph, "execute"):
            return await self.graph.execute(query, params=params)
        raise AttributeError(f"Graph object {type(self.graph)} has no query method")

    async def site_exists(self, site_id: str) -> bool:
        """Check if a site exists in the graph."""
        query = (
            "MATCH (s:Site {site_id: $site_id}) "
            "RETURN count(s) AS count"
        )
        try:
            result = await self._ro_query(query, params={"site_id": site_id})
            rows = _query_result_to_dicts(result)
            return bool(rows and int(rows[0].get("count") or 0) > 0)
        except Exception as exc:
            logger.error("Error checking site existence for %s: %s", site_id, exc)
            return False

    async def search_chunks_by_vector(
        self,
        embedding: Sequence[float],
        limit: int,
        site_id: str | None = None,
    ) -> list[SearchHit]:
        """Search document chunks using 384-dimensional cosine vector index."""
        candidate_limit = _candidate_limit(limit, has_filter=bool(site_id))
        query = (
            "CALL db.idx.vector.queryNodes('Chunk', 'embedding', $candidate_limit, vecf32($embedding)) "
            "YIELD node, score "
            "MATCH (p:Page)-[:HAS_CHUNK]->(node) "
            "MATCH (s:Site)-[:HAS_PAGE]->(p) "
            "WHERE ($site_id IS NULL OR s.site_id = $site_id) "
            "RETURN "
            "  node.chunk_id AS chunk_id, "
            "  p.page_id AS page_id, "
            "  s.site_id AS site_id, "
            "  node.text AS content, "
            "  p.url AS url, "
            "  s.domain AS source, "
            "  node.index AS chunk_number, "
            "  score, "
            "  node.content_type AS content_type, "
            "  node.language AS language, "
            "  node.metadata_json AS metadata_json, "
            "  p.title AS page_title "
            "ORDER BY score ASC "
            "LIMIT $limit"
        )
        params: dict[str, Any] = {
            "candidate_limit": candidate_limit,
            "embedding": list(embedding),
            "site_id": site_id,
            "limit": limit,
        }
        result = await self._ro_query(query, params=params)
        rows = _query_result_to_dicts(result)
        hits: list[SearchHit] = []
        for row in rows:
            score = _convert_vector_score(row.get("score"))
            hits.append(_to_search_hit(row, score))
        return hits

    async def search_chunks_by_text(
        self,
        query: str,
        limit: int,
        site_id: str | None = None,
    ) -> list[SearchHit]:
        """Search document chunks using Chunk.text full-text index with RediSearch escaping."""
        escaped_query = escape_redisearch_query(query)
        if not escaped_query:
            return []

        cypher = (
            "CALL db.idx.fulltext.queryNodes('Chunk', $escaped_query) "
            "YIELD node, score "
            "MATCH (p:Page)-[:HAS_CHUNK]->(node) "
            "MATCH (s:Site)-[:HAS_PAGE]->(p) "
            "WHERE ($site_id IS NULL OR s.site_id = $site_id) "
            "RETURN "
            "  node.chunk_id AS chunk_id, "
            "  p.page_id AS page_id, "
            "  s.site_id AS site_id, "
            "  node.text AS content, "
            "  p.url AS url, "
            "  s.domain AS source, "
            "  node.index AS chunk_number, "
            "  score, "
            "  node.content_type AS content_type, "
            "  node.language AS language, "
            "  node.metadata_json AS metadata_json, "
            "  p.title AS page_title "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )
        params: dict[str, Any] = {
            "escaped_query": escaped_query,
            "site_id": site_id,
            "limit": limit,
        }
        result = await self._ro_query(cypher, params=params)
        rows = _query_result_to_dicts(result)
        hits: list[SearchHit] = []
        for row in rows:
            try:
                text_score = float(row.get("score") or 1.0)
            except (ValueError, TypeError):
                text_score = 1.0
            hits.append(_to_search_hit(row, text_score))
        return hits

    async def search_code_chunks(
        self,
        embedding: Sequence[float],
        limit: int,
        site_id: str | None = None,
        language: str | None = None,
    ) -> list[SearchHit]:
        """Search code chunks with vector similarity and optional language/site filter."""
        has_filter = bool(site_id or language)
        candidate_limit = _candidate_limit(limit, has_filter=has_filter)
        query = (
            "CALL db.idx.vector.queryNodes('Chunk', 'embedding', $candidate_limit, vecf32($embedding)) "
            "YIELD node, score "
            "MATCH (p:Page)-[:HAS_CHUNK]->(node) "
            "MATCH (s:Site)-[:HAS_PAGE]->(p) "
            "WHERE node.content_type = 'code' "
            "  AND ($site_id IS NULL OR s.site_id = $site_id) "
            "  AND ($language IS NULL OR toLower(node.language) = toLower($language)) "
            "RETURN "
            "  node.chunk_id AS chunk_id, "
            "  p.page_id AS page_id, "
            "  s.site_id AS site_id, "
            "  node.text AS content, "
            "  p.url AS url, "
            "  s.domain AS source, "
            "  node.index AS chunk_number, "
            "  score, "
            "  node.content_type AS content_type, "
            "  node.language AS language, "
            "  node.metadata_json AS metadata_json, "
            "  p.title AS page_title "
            "ORDER BY score ASC "
            "LIMIT $limit"
        )
        params: dict[str, Any] = {
            "candidate_limit": candidate_limit,
            "embedding": list(embedding),
            "site_id": site_id,
            "language": language,
            "limit": limit,
        }
        result = await self._ro_query(query, params=params)
        rows = _query_result_to_dicts(result)
        hits: list[SearchHit] = []
        for row in rows:
            score = _convert_vector_score(row.get("score"))
            hits.append(_to_search_hit(row, score))
        return hits

    async def search_entities_by_vector(
        self,
        embedding: Sequence[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search __Entity__ nodes using vector similarity."""
        query = (
            "CALL db.idx.vector.queryNodes('__Entity__', 'embedding', $limit, vecf32($embedding)) "
            "YIELD node, score "
            "RETURN "
            "  node.name AS name, "
            "  node.entity_type AS entity_type, "
            "  node.description AS description, "
            "  score "
            "ORDER BY score ASC "
            "LIMIT $limit"
        )
        params = {"embedding": list(embedding), "limit": limit}
        try:
            result = await self._ro_query(query, params=params)
            rows = _query_result_to_dicts(result)
            for row in rows:
                row["similarity_score"] = _convert_vector_score(row.get("score"))
            return rows
        except Exception as exc:
            logger.error("Entity vector search failed: %s", exc)
            return []

    async def search_entities_by_text(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search __Entity__ nodes using full-text index on name and description."""
        escaped_query = escape_redisearch_query(query)
        if not escaped_query:
            return []

        cypher = (
            "CALL db.idx.fulltext.queryNodes('__Entity__', $escaped_query) "
            "YIELD node, score "
            "RETURN "
            "  node.name AS name, "
            "  node.entity_type AS entity_type, "
            "  node.description AS description, "
            "  score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )
        try:
            result = await self._ro_query(cypher, params={"escaped_query": escaped_query, "limit": limit})
            return _query_result_to_dicts(result)
        except Exception as exc:
            logger.error("Entity full-text search failed: %s", exc)
            return []

    async def search_relationships_by_vector(
        self,
        embedding: Sequence[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search RELATES relationships using vector similarity on relationship embeddings."""
        query = (
            "CALL db.idx.vector.queryRelationships('RELATES', 'embedding', $limit, vecf32($embedding)) "
            "YIELD relationship, score "
            "MATCH (src:__Entity__)-[relationship]->(tgt:__Entity__) "
            "RETURN "
            "  src.name AS source, "
            "  tgt.name AS target, "
            "  relationship.rel_type AS relation, "
            "  relationship.fact AS fact, "
            "  relationship.description AS description, "
            "  score "
            "ORDER BY score ASC "
            "LIMIT $limit"
        )
        params = {"embedding": list(embedding), "limit": limit}
        try:
            result = await self._ro_query(query, params=params)
            rows = _query_result_to_dicts(result)
            for row in rows:
                row["similarity_score"] = _convert_vector_score(row.get("score"))
            return rows
        except Exception as exc:
            logger.error("Relationship vector search failed: %s", exc)
            return []

    async def search_relationships_by_text(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search RELATES relationships using full-text index."""
        escaped_query = escape_redisearch_query(query)
        if not escaped_query:
            return []

        cypher = (
            "CALL db.idx.fulltext.queryRelationships('RELATES', $escaped_query) "
            "YIELD relationship, score "
            "MATCH (src:__Entity__)-[relationship]->(tgt:__Entity__) "
            "RETURN "
            "  src.name AS source, "
            "  tgt.name AS target, "
            "  relationship.rel_type AS relation, "
            "  relationship.fact AS fact, "
            "  relationship.description AS description, "
            "  score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )
        try:
            result = await self._ro_query(cypher, params={"escaped_query": escaped_query, "limit": limit})
            return _query_result_to_dicts(result)
        except Exception as exc:
            logger.error("Relationship full-text search failed: %s", exc)
            return []

    async def get_chunk_provenance(self, chunk_id: str) -> list[dict[str, Any]]:
        """Retrieve grounded entity extractions for a given chunk."""
        query = (
            "MATCH (e:__Entity__)-[m:MENTIONED_IN]->(c:Chunk {chunk_id: $chunk_id}) "
            "OPTIONAL MATCH (e)-[r:RELATES]-(other:__Entity__) "
            "RETURN "
            "  e.name AS entity_name, "
            "  e.entity_type AS entity_type, "
            "  m.extraction_class AS extraction_class, "
            "  m.extraction_text AS extraction_text, "
            "  m.start_char AS start_char, "
            "  m.end_char AS end_char, "
            "  m.confidence AS confidence, "
            "  m.extraction_source AS extraction_source, "
            "  m.attributes_json AS attributes_json, "
            "  r.rel_type AS relation, "
            "  other.name AS related_entity, "
            "  r.fact AS fact, "
            "  r.description AS relation_description"
        )
        result = await self._ro_query(query, params={"chunk_id": chunk_id})
        return _query_result_to_dicts(result)
