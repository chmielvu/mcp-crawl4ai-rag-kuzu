"""Database service for Kuzu-backed storage operations."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import kuzu

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import SourceInfo

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing database operations with Kuzu."""

    def __init__(self, client: kuzu.Connection, settings: Optional[Any] = None):
        self.client = client
        self.settings = settings or get_settings()

    async def add_documents(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        url_to_full_document: Dict[str, str],
        batch_size: int = 20,
    ) -> Dict[str, Any]:
        del url_to_full_document
        if not urls:
            return {"success": True, "count": 0, "total": 0}

        for url in set(urls):
            self.client.execute(
                "MATCH (c:Chunk {url: $url}) DETACH DELETE c",
                {"url": url},
            )

        documents_added = 0
        for start in range(0, len(urls), batch_size):
            batch = zip(
                urls[start : start + batch_size],
                chunk_numbers[start : start + batch_size],
                contents[start : start + batch_size],
                embeddings[start : start + batch_size],
                metadatas[start : start + batch_size],
            )
            for url, chunk_number, content, embedding, metadata in batch:
                source_id = urlparse(url).netloc
                chunk_id = _chunk_id(url, chunk_number)
                self._merge_chunk(
                    chunk_id=chunk_id,
                    url=url,
                    chunk_number=chunk_number,
                    content=content,
                    metadata=metadata,
                    embedding=embedding,
                    source_id=source_id,
                )
                if chunk_number > 1:
                    self.client.execute(
                        """
                        MATCH (previous:Chunk {chunk_id: $previous_id}),
                              (current:Chunk {chunk_id: $current_id})
                        MERGE (previous)-[:NEXT_CHUNK]->(current)
                        """,
                        {
                            "previous_id": _chunk_id(url, chunk_number - 1),
                            "current_id": chunk_id,
                        },
                    )
                documents_added += 1
        return {"success": True, "count": documents_added, "total": len(urls)}

    async def add_code_examples(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        code_examples: List[str],
        summaries: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 20,
    ) -> Dict[str, Any]:
        if not urls:
            return {"success": True, "count": 0}

        for url in set(urls):
            self.client.execute(
                "MATCH (e:CodeExample {url: $url}) DETACH DELETE e",
                {"url": url},
            )

        examples_added = 0
        for start in range(0, len(urls), batch_size):
            batch = zip(
                urls[start : start + batch_size],
                chunk_numbers[start : start + batch_size],
                code_examples[start : start + batch_size],
                summaries[start : start + batch_size],
                embeddings[start : start + batch_size],
                metadatas[start : start + batch_size],
            )
            for url, chunk_number, code, summary, embedding, metadata in batch:
                source_id = urlparse(url).netloc
                example_id = _example_id(url, chunk_number)
                language = metadata.get("language") or _language_from_code(code)
                self.client.execute(
                    """
                    CREATE (e:CodeExample {
                        example_id: $example_id,
                        url: $url,
                        chunk_number: $chunk_number,
                        content: $content,
                        summary: $summary,
                        language: $language,
                        metadata: $metadata,
                        embedding: $embedding
                    })
                    """,
                    {
                        "example_id": example_id,
                        "url": url,
                        "chunk_number": chunk_number,
                        "content": code,
                        "summary": summary,
                        "language": language,
                        "metadata": json.dumps({**metadata, "language": language}),
                        "embedding": embedding,
                    },
                )
                self.client.execute(
                    """
                    MATCH (s:Source {source_id: $source_id}),
                          (e:CodeExample {example_id: $example_id})
                    MERGE (s)-[:HAS_EXAMPLE]->(e)
                    """,
                    {"source_id": source_id, "example_id": example_id},
                )
                examples_added += 1
        return {"success": True, "count": examples_added, "total": len(urls)}

    async def update_source_info(
        self, source_id: str, summary: str, word_count: int
    ) -> Dict[str, Any]:
        try:
            self.client.execute(
                """
                MERGE (s:Source {source_id: $source_id})
                SET
                    s.summary = $summary,
                    s.word_count = $word_count,
                    s.updated_at = $updated_at
                """,
                {
                    "source_id": source_id,
                    "summary": summary,
                    "word_count": word_count,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return {"success": True, "source_id": source_id}
        except Exception as error:
            logger.error("Error updating source info: %s", error)
            return {"success": False, "error": str(error)}

    async def get_available_sources(self) -> List[SourceInfo]:
        try:
            result = self.client.execute(
                """
                MATCH (s:Source)
                OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chunk)
                OPTIONAL MATCH (s)-[:HAS_EXAMPLE]->(e:CodeExample)
                RETURN
                    s.source_id AS source_id,
                    s.summary AS summary,
                    s.word_count AS word_count,
                    s.updated_at AS updated_at,
                    count(DISTINCT c.url) AS total_documents,
                    count(DISTINCT c.chunk_id) AS total_chunks,
                    count(DISTINCT e.example_id) AS total_code_examples
                ORDER BY s.source_id
                """
            )
            return [
                SourceInfo(
                    source=row["source_id"],
                    summary=row.get("summary"),
                    word_count=int(row.get("word_count") or 0),
                    last_updated=_parse_timestamp(row.get("updated_at")),
                    total_documents=int(row.get("total_documents") or 0),
                    total_chunks=int(row.get("total_chunks") or 0),
                    total_code_examples=int(row.get("total_code_examples") or 0),
                )
                for row in _query_rows(result)
            ]
        except Exception as error:
            logger.error("Error getting available sources: %s", error)
            return []

    def _merge_chunk(
        self,
        chunk_id: str,
        url: str,
        chunk_number: int,
        content: str,
        metadata: Dict[str, Any],
        embedding: List[float],
        source_id: str,
    ) -> None:
        self.client.execute(
            """
            CREATE (c:Chunk {
                chunk_id: $chunk_id,
                url: $url,
                chunk_number: $chunk_number,
                content: $content,
                metadata: $metadata,
                embedding: $embedding
            })
            """,
            {
                "chunk_id": chunk_id,
                "url": url,
                "chunk_number": chunk_number,
                "content": content,
                "metadata": json.dumps(metadata),
                "embedding": embedding,
            },
        )
        self.client.execute(
            """
            MERGE (s:Source {source_id: $source_id})
            ON CREATE SET s.updated_at = $updated_at
            """,
            {
                "source_id": source_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.client.execute(
            """
            MATCH (s:Source {source_id: $source_id}),
                  (c:Chunk {chunk_id: $chunk_id})
            MERGE (s)-[:CONTAINS]->(c)
            """,
            {"source_id": source_id, "chunk_id": chunk_id},
        )

    def _generate_contextual_content(
        self, chunk_content: str, full_document: str, chunk_number: int
    ) -> str:
        del full_document
        return f"Chunk {chunk_number} of document: {chunk_content}"


def _query_rows(
    result: kuzu.QueryResult | list[kuzu.QueryResult],
) -> list[dict[str, Any]]:
    if isinstance(result, list):
        if not result:
            return []
        result = result[-1]
    columns = result.get_column_names()
    return [dict(zip(columns, row)) for row in result.get_all()]


def _chunk_id(url: str, chunk_number: int) -> str:
    return f"{url}::chunk::{chunk_number}"


def _example_id(url: str, chunk_number: int) -> str:
    return f"{url}::code::{chunk_number}"


def _language_from_code(code: str) -> str:
    if code.startswith("```"):
        first_line = code.splitlines()[0].strip()
        if len(first_line) > 3:
            return first_line[3:].strip() or "unknown"
    return "unknown"


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)
