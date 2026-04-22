"""Low-level Kuzu queries for vector and text retrieval."""

import json
from typing import Any, Optional

import kuzu


class KuzuSearchBackend:
    """Encapsulate Kuzu vector and FTS queries."""

    def __init__(self, connection: kuzu.Connection):
        self.connection = connection

    def source_exists(self, source_id: str) -> bool:
        result = self.connection.execute(
            "MATCH (s:Source {source_id: $source_id}) RETURN count(s) AS count",
            {"source_id": source_id},
        )
        rows = _query_rows(result)
        return bool(rows and rows[0]["count"] > 0)

    def search_documents_by_vector(
        self, embedding: list[float], limit: int, source_id: Optional[str]
    ) -> list[dict[str, Any]]:
        return self._vector_query(
            table_name="Chunk",
            index_name="chunk_embedding_idx",
            relationship_name="CONTAINS",
            embedding=embedding,
            limit=limit,
            source_id=source_id,
        )

    def search_documents_by_text(
        self, query: str, limit: int, source_id: Optional[str]
    ) -> list[dict[str, Any]]:
        return self._fts_query(
            table_name="Chunk",
            index_name="chunk_fts_idx",
            relationship_name="CONTAINS",
            query=query,
            limit=limit,
            source_id=source_id,
        )

    def search_code_by_vector(
        self, embedding: list[float], limit: int, source_id: Optional[str]
    ) -> list[dict[str, Any]]:
        return self._vector_query(
            table_name="CodeExample",
            index_name="code_embedding_idx",
            relationship_name="HAS_EXAMPLE",
            embedding=embedding,
            limit=limit,
            source_id=source_id,
        )

    def search_code_by_text(
        self, query: str, limit: int, source_id: Optional[str]
    ) -> list[dict[str, Any]]:
        return self._fts_query(
            table_name="CodeExample",
            index_name="code_fts_idx",
            relationship_name="HAS_EXAMPLE",
            query=query,
            limit=limit,
            source_id=source_id,
        )

    def _vector_query(
        self,
        table_name: str,
        index_name: str,
        relationship_name: str,
        embedding: list[float],
        limit: int,
        source_id: Optional[str],
    ) -> list[dict[str, Any]]:
        candidate_limit = _candidate_limit(limit, source_id)
        if source_id:
            query = f"""
                CALL QUERY_VECTOR_INDEX('{table_name}', '{index_name}', $embedding, $limit)
                WITH node, distance
                MATCH (s:Source {{source_id: $source_id}})-[:{relationship_name}]->(node)
                RETURN
                    node.url AS url,
                    node.chunk_number AS chunk_number,
                    node.content AS content,
                    node.metadata AS metadata,
                    s.source_id AS source_id,
                    distance
                ORDER BY distance
                LIMIT $result_limit
            """
            params = {
                "embedding": embedding,
                "limit": candidate_limit,
                "source_id": source_id,
                "result_limit": limit,
            }
        else:
            query = f"""
                CALL QUERY_VECTOR_INDEX('{table_name}', '{index_name}', $embedding, $limit)
                WITH node, distance
                MATCH (s:Source)-[:{relationship_name}]->(node)
                RETURN
                    node.url AS url,
                    node.chunk_number AS chunk_number,
                    node.content AS content,
                    node.metadata AS metadata,
                    s.source_id AS source_id,
                    distance
                ORDER BY distance
                LIMIT $result_limit
            """
            params = {"embedding": embedding, "limit": limit, "result_limit": limit}
        rows = _query_rows(self.connection.execute(query, params))
        return [_normalize_row(row, score_key="distance", invert_distance=True) for row in rows]

    def _fts_query(
        self,
        table_name: str,
        index_name: str,
        relationship_name: str,
        query: str,
        limit: int,
        source_id: Optional[str],
    ) -> list[dict[str, Any]]:
        candidate_limit = _candidate_limit(limit, source_id)
        if source_id:
            statement = f"""
                CALL QUERY_FTS_INDEX('{table_name}', '{index_name}', $query, top := $top)
                WITH node, score
                MATCH (s:Source {{source_id: $source_id}})-[:{relationship_name}]->(node)
                RETURN
                    node.url AS url,
                    node.chunk_number AS chunk_number,
                    node.content AS content,
                    node.metadata AS metadata,
                    s.source_id AS source_id,
                    score
                ORDER BY score DESC
                LIMIT $result_limit
            """
            params = {
                "query": query,
                "top": candidate_limit,
                "source_id": source_id,
                "result_limit": limit,
            }
        else:
            statement = f"""
                CALL QUERY_FTS_INDEX('{table_name}', '{index_name}', $query, top := $top)
                WITH node, score
                MATCH (s:Source)-[:{relationship_name}]->(node)
                RETURN
                    node.url AS url,
                    node.chunk_number AS chunk_number,
                    node.content AS content,
                    node.metadata AS metadata,
                    s.source_id AS source_id,
                    score
                ORDER BY score DESC
                LIMIT $result_limit
            """
            params = {"query": query, "top": limit, "result_limit": limit}
        rows = _query_rows(self.connection.execute(statement, params))
        return [_normalize_row(row, score_key="score", invert_distance=False) for row in rows]


def _candidate_limit(limit: int, source_id: Optional[str]) -> int:
    if source_id:
        return max(limit * 8, 50)
    return limit


def _normalize_row(
    row: dict[str, Any], score_key: str, invert_distance: bool
) -> dict[str, Any]:
    metadata = row.get("metadata")
    parsed_metadata: dict[str, Any] = {}
    if isinstance(metadata, str) and metadata:
        parsed_metadata = json.loads(metadata)
    score_value = float(row.pop(score_key))
    similarity = 1.0 - score_value if invert_distance else score_value
    return {
        "url": row.get("url", ""),
        "chunk_number": row.get("chunk_number", 0),
        "content": row.get("content", ""),
        "metadata": parsed_metadata,
        "source_id": row.get("source_id", ""),
        "similarity": similarity,
    }


def _query_rows(
    result: kuzu.QueryResult | list[kuzu.QueryResult],
) -> list[dict[str, Any]]:
    if isinstance(result, list):
        if not result:
            return []
        result = result[-1]
    columns = result.get_column_names()
    return [dict(zip(columns, row)) for row in result.get_all()]
