"""Idempotent schema initialization for the dedicated FalkorDB crawl graph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


NODE_LABELS = ("Site", "CrawlRun", "Page", "Chunk", "__Entity__")
RELATIONSHIP_TYPES = (
    "HAS_PAGE",
    "CRAWLED",
    "HAS_CHUNK",
    "LINKS_TO",
    "HAS_ENTITY",
    "MENTIONED_IN",
    "RELATES",
)

_UNIQUE_PROPERTIES = (
    ("Site", "site_id"),
    ("CrawlRun", "run_id"),
    ("Page", "page_id"),
    ("Chunk", "chunk_id"),
    ("__Entity__", "entity_id"),
)

_ALREADY_EXISTS = (
    "already exists",
    "already indexed",
    "already defined",
    "duplicate",
    "exists",
)


async def init_falkor_schema(graph: Any, embedding_dimension: int = 384) -> None:
    """Create the web-crawling graph constraints and indexes idempotently."""

    if embedding_dimension != 384:
        raise ValueError("crawl-graph vector indexes require exactly 384 dimensions")

    for label, property_name in _UNIQUE_PROPERTIES:
        await _create_unique_constraint(graph, label, property_name)

    await _create_node_vector_index(graph, "Chunk", "embedding", embedding_dimension)
    await _create_node_vector_index(graph, "__Entity__", "embedding", embedding_dimension)
    await _create_edge_vector_index(graph, "RELATES", "embedding", embedding_dimension)

    await _create_fulltext_index(graph, "Page", "title")
    await _create_fulltext_index(graph, "Chunk", "text")
    await _create_fulltext_index(graph, "__Entity__", "name", "description")


async def _create_unique_constraint(graph: Any, label: str, property_name: str) -> None:
    method = getattr(graph, "create_node_unique_constraint", None)
    if method is not None:
        await _idempotent_call(lambda: method(label, property_name))
        return
    query = f"CREATE CONSTRAINT FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
    await _idempotent_query(graph, query)


async def _create_node_vector_index(
    graph: Any, label: str, property_name: str, dimension: int
) -> None:
    method = getattr(graph, "create_node_vector_index", None)
    if method is not None:
        await _idempotent_call(
            lambda: method(
                label,
                property_name,
                dim=dimension,
                similarity_function="cosine",
            )
        )
        return
    query = (
        f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.{property_name}) "
        f"OPTIONS {{dimension:{dimension}, similarityFunction:'cosine'}}"
    )
    await _idempotent_query(graph, query)


async def _create_edge_vector_index(
    graph: Any, relation: str, property_name: str, dimension: int
) -> None:
    method = getattr(graph, "create_edge_vector_index", None)
    if method is not None:
        await _idempotent_call(
            lambda: method(
                relation,
                property_name,
                dim=dimension,
                similarity_function="cosine",
            )
        )
        return
    query = (
        f"CREATE VECTOR INDEX FOR ()-[r:{relation}]-() ON (r.{property_name}) "
        f"OPTIONS {{dimension:{dimension}, similarityFunction:'cosine'}}"
    )
    await _idempotent_query(graph, query)


async def _create_fulltext_index(graph: Any, label: str, *properties: str) -> None:
    method = getattr(graph, "create_node_fulltext_index", None)
    if method is not None:
        await _idempotent_call(lambda: method(label, *properties))
        return
    prop_expr = ", ".join(f"n.{prop}" for prop in properties)
    query = f"CREATE FULLTEXT INDEX FOR (n:{label}) ON ({prop_expr})"
    await _idempotent_query(graph, query)


async def _idempotent_call(operation: Callable[[], Awaitable[Any]]) -> None:
    try:
        await operation()
    except Exception as exc:
        if not _is_already_exists(exc):
            raise


async def _idempotent_query(graph: Any, query: str) -> None:
    query_method = getattr(graph, "query", None)
    if query_method is None:
        raise TypeError("Falkor graph does not expose async query()")
    await _idempotent_call(lambda: query_method(query))


def _is_already_exists(error: Exception) -> bool:
    message = str(error).lower()
    return any(fragment in message for fragment in _ALREADY_EXISTS)
