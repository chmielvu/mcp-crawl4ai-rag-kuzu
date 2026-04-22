"""Tests for database service."""

import pytest

from crawl4ai_mcp.models import SourceInfo
from crawl4ai_mcp.services.database import DatabaseService


@pytest.fixture
def database_service(mock_db_connection, test_settings):
    return DatabaseService(mock_db_connection, test_settings)


@pytest.mark.asyncio
async def test_add_documents_success(database_service, mock_db_connection) -> None:
    result = await database_service.add_documents(
        urls=["https://example.com/test"],
        chunk_numbers=[1],
        contents=["Test content"],
        embeddings=[[0.1] * 1024],
        metadatas=[{"title": "Test"}],
        url_to_full_document={"https://example.com/test": "Full document content"},
    )
    assert result == {"success": True, "count": 1, "total": 1}
    assert mock_db_connection.execute.call_count >= 4


@pytest.mark.asyncio
async def test_add_documents_empty_list(database_service) -> None:
    result = await database_service.add_documents([], [], [], [], [], {})
    assert result == {"success": True, "count": 0, "total": 0}


@pytest.mark.asyncio
async def test_add_documents_delete_error(database_service, mock_db_connection) -> None:
    mock_db_connection.execute.side_effect = Exception("Delete failed")
    with pytest.raises(Exception, match="Delete failed"):
        await database_service.add_documents(
            urls=["https://example.com/test"],
            chunk_numbers=[1],
            contents=["Test content"],
            embeddings=[[0.1] * 1024],
            metadatas=[{"title": "Test"}],
            url_to_full_document={},
        )


@pytest.mark.asyncio
async def test_add_code_examples_success(database_service, mock_db_connection) -> None:
    result = await database_service.add_code_examples(
        urls=["https://example.com/test"],
        chunk_numbers=[1],
        code_examples=["```python\ndef hello():\n    print('Hello')\n```"],
        summaries=["A hello function"],
        embeddings=[[0.1] * 1024],
        metadatas=[{"language": "python"}],
    )
    assert result == {"success": True, "count": 1, "total": 1}
    assert mock_db_connection.execute.call_count >= 3


@pytest.mark.asyncio
async def test_add_code_examples_empty_list(database_service) -> None:
    result = await database_service.add_code_examples([], [], [], [], [], [])
    assert result == {"success": True, "count": 0}


@pytest.mark.asyncio
async def test_update_source_info_success(database_service) -> None:
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Test source",
        word_count=100,
    )
    assert result == {"success": True, "source_id": "example.com"}


@pytest.mark.asyncio
async def test_update_source_info_error(database_service, mock_db_connection) -> None:
    mock_db_connection.execute.side_effect = Exception("Update failed")
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Test source",
        word_count=100,
    )
    assert result["success"] is False
    assert "Update failed" in result["error"]


@pytest.mark.asyncio
async def test_get_available_sources(database_service, mock_db_connection) -> None:
    class LocalQueryResult:
        def __init__(self, columns, rows):
            self.columns = columns
            self.rows = rows

        def get_column_names(self):
            return self.columns

        def get_all(self):
            return self.rows

    mock_db_connection.execute.return_value = LocalQueryResult(
        [
            "source_id",
            "summary",
            "word_count",
            "updated_at",
            "total_documents",
            "total_chunks",
            "total_code_examples",
        ],
        [
            [
                "example.com",
                "Example website",
                1000,
                "2024-01-02T00:00:00+00:00",
                4,
                10,
                2,
            ]
        ],
    )

    sources = await database_service.get_available_sources()
    assert len(sources) == 1
    assert isinstance(sources[0], SourceInfo)
    assert sources[0].source == "example.com"
    assert sources[0].total_documents == 4
    assert sources[0].total_chunks == 10


@pytest.mark.asyncio
async def test_get_available_sources_error(database_service, mock_db_connection) -> None:
    mock_db_connection.execute.side_effect = Exception("Query failed")
    assert await database_service.get_available_sources() == []


@pytest.mark.asyncio
async def test_generate_contextual_content(database_service) -> None:
    content = database_service._generate_contextual_content(
        chunk_content="This is chunk content",
        full_document="This is the full document with more content",
        chunk_number=2,
    )
    assert "Chunk 2" in content
    assert "This is chunk content" in content
