"""Tests for FalkorDB schema initialization, constraint helpers, and idempotency."""

from unittest.mock import AsyncMock, Mock

import pytest

from crawl4ai_mcp.services.falkor_schema import (
    NODE_LABELS,
    RELATIONSHIP_TYPES,
    _is_already_exists,
    init_falkor_schema,
)


@pytest.mark.asyncio
async def test_init_falkor_schema_dimension_validation() -> None:
    graph = Mock()
    with pytest.raises(ValueError, match="vector indexes require exactly 384 dimensions"):
        await init_falkor_schema(graph, embedding_dimension=512)


@pytest.mark.asyncio
async def test_init_falkor_schema_dedicated_methods() -> None:
    graph = Mock()
    graph.create_node_unique_constraint = AsyncMock()
    graph.create_node_vector_index = AsyncMock()
    graph.create_edge_vector_index = AsyncMock()
    graph.create_node_fulltext_index = AsyncMock()

    await init_falkor_schema(graph, embedding_dimension=384)

    # 5 unique constraints: Site, CrawlRun, Page, Chunk, __Entity__
    assert graph.create_node_unique_constraint.await_count == 5
    graph.create_node_unique_constraint.assert_any_await("Site", "site_id")
    graph.create_node_unique_constraint.assert_any_await("CrawlRun", "run_id")
    graph.create_node_unique_constraint.assert_any_await("Page", "page_id")
    graph.create_node_unique_constraint.assert_any_await("Chunk", "chunk_id")
    graph.create_node_unique_constraint.assert_any_await("__Entity__", "entity_id")

    # 2 node vector indexes: Chunk, __Entity__
    assert graph.create_node_vector_index.await_count == 2
    graph.create_node_vector_index.assert_any_await(
        "Chunk", "embedding", dim=384, similarity_function="cosine"
    )
    graph.create_node_vector_index.assert_any_await(
        "__Entity__", "embedding", dim=384, similarity_function="cosine"
    )

    # 1 edge vector index: RELATES
    assert graph.create_edge_vector_index.await_count == 1
    graph.create_edge_vector_index.assert_any_await(
        "RELATES", "embedding", dim=384, similarity_function="cosine"
    )

    # 3 fulltext indexes: Page(title), Chunk(text), __Entity__(name, description)
    assert graph.create_node_fulltext_index.await_count == 3
    graph.create_node_fulltext_index.assert_any_await("Page", "title")
    graph.create_node_fulltext_index.assert_any_await("Chunk", "text")
    graph.create_node_fulltext_index.assert_any_await("__Entity__", "name", "description")


@pytest.mark.asyncio
async def test_init_falkor_schema_query_fallback() -> None:
    graph = Mock(spec=["query"])
    executed_queries: list[str] = []

    async def mock_query(q: str) -> None:
        executed_queries.append(q)

    graph.query = mock_query

    await init_falkor_schema(graph, embedding_dimension=384)

    assert len(executed_queries) == 11
    # Check constraints
    assert any("CREATE CONSTRAINT FOR (n:Site) REQUIRE n.site_id IS UNIQUE" in q for q in executed_queries)
    # Check vector index
    assert any("CREATE VECTOR INDEX FOR (n:Chunk) ON (n.embedding)" in q for q in executed_queries)
    assert any("CREATE VECTOR INDEX FOR ()-[r:RELATES]-() ON (r.embedding)" in q for q in executed_queries)
    # Check fulltext index
    assert any("CREATE FULLTEXT INDEX FOR (n:Page) ON (n.title)" in q for q in executed_queries)


@pytest.mark.asyncio
async def test_init_falkor_schema_idempotency_ignores_already_exists() -> None:
    graph = Mock()
    graph.create_node_unique_constraint = AsyncMock(
        side_effect=Exception("Index on label Site already exists")
    )
    graph.create_node_vector_index = AsyncMock(
        side_effect=Exception("Vector index already indexed")
    )
    graph.create_edge_vector_index = AsyncMock(
        side_effect=Exception("Edge index duplicate")
    )
    graph.create_node_fulltext_index = AsyncMock(
        side_effect=Exception("Fulltext index exists")
    )

    # Should not raise exception
    await init_falkor_schema(graph, embedding_dimension=384)


@pytest.mark.asyncio
async def test_init_falkor_schema_bubbles_unexpected_errors() -> None:
    graph = Mock()
    graph.create_node_unique_constraint = AsyncMock(
        side_effect=RuntimeError("Fatal database connection failure")
    )

    with pytest.raises(RuntimeError, match="Fatal database connection failure"):
        await init_falkor_schema(graph, embedding_dimension=384)


def test_is_already_exists_helper() -> None:
    assert _is_already_exists(Exception("Index already exists")) is True
    assert _is_already_exists(Exception("Duplicate index name")) is True
    assert _is_already_exists(Exception("Already indexed field")) is True
    assert _is_already_exists(Exception("Random DB error")) is False


def test_schema_constants() -> None:
    assert "Site" in NODE_LABELS
    assert "Chunk" in NODE_LABELS
    assert "__Entity__" in NODE_LABELS
    assert "HAS_PAGE" in RELATIONSHIP_TYPES
    assert "RELATES" in RELATIONSHIP_TYPES
