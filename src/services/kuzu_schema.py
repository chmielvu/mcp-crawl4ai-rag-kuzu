"""Kuzu schema initialization and index management."""

from pathlib import Path
from typing import Any

import kuzu


def init_db(db_path: str, embedding_dimensions: int) -> kuzu.Connection:
    """Initialize the Kuzu database and ensure schema/indexes exist."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    database = kuzu.Database(str(database_path))
    connection = kuzu.Connection(database)

    _load_extensions(connection)
    _create_schema(connection, embedding_dimensions)
    _ensure_indexes(connection)

    return connection


def _load_extensions(connection: kuzu.Connection) -> None:
    for statement in ("INSTALL VECTOR;", "LOAD VECTOR;", "INSTALL FTS;", "LOAD FTS;"):
        connection.execute(statement)


def _create_schema(connection: kuzu.Connection, embedding_dimensions: int) -> None:
    connection.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS Source(
            source_id STRING,
            summary STRING,
            word_count INT64,
            updated_at STRING,
            PRIMARY KEY(source_id)
        )
        """
    )
    connection.execute(
        f"""
        CREATE NODE TABLE IF NOT EXISTS Chunk(
            chunk_id STRING,
            url STRING,
            chunk_number INT64,
            content STRING,
            metadata STRING,
            embedding FLOAT[{embedding_dimensions}],
            PRIMARY KEY(chunk_id)
        )
        """
    )
    connection.execute(
        f"""
        CREATE NODE TABLE IF NOT EXISTS CodeExample(
            example_id STRING,
            url STRING,
            chunk_number INT64,
            content STRING,
            summary STRING,
            language STRING,
            metadata STRING,
            embedding FLOAT[{embedding_dimensions}],
            PRIMARY KEY(example_id)
        )
        """
    )
    connection.execute("CREATE REL TABLE IF NOT EXISTS CONTAINS(FROM Source TO Chunk)")
    connection.execute(
        "CREATE REL TABLE IF NOT EXISTS HAS_EXAMPLE(FROM Source TO CodeExample)"
    )
    connection.execute("CREATE REL TABLE IF NOT EXISTS NEXT_CHUNK(FROM Chunk TO Chunk)")


def _ensure_indexes(connection: kuzu.Connection) -> None:
    existing_indexes = {
        row["index_name"] for row in _query_rows(connection.execute("CALL SHOW_INDEXES() RETURN *"))
    }

    index_statements = {
        "chunk_embedding_idx": (
            "CALL CREATE_VECTOR_INDEX("
            "'Chunk', 'chunk_embedding_idx', 'embedding', metric := 'cosine'"
            ")"
        ),
        "code_embedding_idx": (
            "CALL CREATE_VECTOR_INDEX("
            "'CodeExample', 'code_embedding_idx', 'embedding', metric := 'cosine'"
            ")"
        ),
        "chunk_fts_idx": (
            "CALL CREATE_FTS_INDEX('Chunk', 'chunk_fts_idx', ['content'])"
        ),
        "code_fts_idx": (
            "CALL CREATE_FTS_INDEX("
            "'CodeExample', 'code_fts_idx', ['content', 'summary']"
            ")"
        ),
    }

    for index_name, statement in index_statements.items():
        if index_name not in existing_indexes:
            connection.execute(statement)


def _query_rows(
    result: kuzu.QueryResult | list[kuzu.QueryResult],
) -> list[dict[str, Any]]:
    if isinstance(result, list):
        if not result:
            return []
        result = result[-1]
    columns = result.get_column_names()
    return [dict(zip(columns, row)) for row in result.get_all()]
