"""Minimal Kuzu smoke test for local verification."""

from pathlib import Path
from tempfile import TemporaryDirectory

from crawl4ai_mcp.services.kuzu_schema import init_db


def main() -> None:
    with TemporaryDirectory(prefix="verify-kuzu-") as temp_dir:
        db_path = Path(temp_dir) / "verify-kuzu"
        connection = init_db(str(db_path), embedding_dimensions=4)
        try:
            connection.execute(
                """
                MERGE (s:Source {source_id: 'test.com'})
                SET s.summary = 'Test source', s.word_count = 1, s.updated_at = '2026-04-22T00:00:00+00:00'
                """
            )
            connection.execute(
                """
                CREATE (c:Chunk {
                    chunk_id: 'test.com::chunk::1',
                    url: 'https://test.com',
                    chunk_number: 1,
                    content: 'hello world',
                    metadata: '{}',
                    embedding: [0.1, 0.2, 0.3, 0.4]
                })
                """
            )
            connection.execute(
                """
                MATCH (s:Source {source_id: 'test.com'}),
                      (c:Chunk {chunk_id: 'test.com::chunk::1'})
                MERGE (s)-[:CONTAINS]->(c)
                """
            )
            result = connection.execute(
                "MATCH (s:Source)-[:CONTAINS]->(c:Chunk) "
                "RETURN s.source_id AS source_id, c.content AS content"
            )
            rows = result.get_all()
            assert rows == [["test.com", "hello world"]], rows
            print("Kuzu verification passed:", rows)
        finally:
            connection.close()


if __name__ == "__main__":
    main()
