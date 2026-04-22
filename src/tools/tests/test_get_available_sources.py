"""Tests for get_available_sources tool."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from crawl4ai_mcp.models import SourceInfo
from crawl4ai_mcp.tools.get_available_sources import get_available_sources


@pytest.fixture
def mock_context():
    context = Mock()
    context.request_context.lifespan_context = SimpleNamespace(
        db_connection=Mock(),
        settings=SimpleNamespace(),
    )
    return context


@pytest.mark.asyncio
async def test_get_available_sources_success(mock_context) -> None:
    with patch("crawl4ai_mcp.tools.get_available_sources.DatabaseService") as MockDatabase:
        database = Mock()
        database.get_available_sources = AsyncMock(
            return_value=[
                SourceInfo(
                    source="example.com",
                    summary="Example source",
                    total_documents=2,
                    total_chunks=4,
                    total_code_examples=1,
                    word_count=100,
                    last_updated=datetime.now(timezone.utc),
                )
            ]
        )
        MockDatabase.return_value = database
        result = json.loads(await get_available_sources(mock_context))
    assert result["success"] is True
    assert result["total_sources"] == 1
    assert result["sources"][0]["source"] == "example.com"


@pytest.mark.asyncio
async def test_get_available_sources_failure(mock_context) -> None:
    with patch("crawl4ai_mcp.tools.get_available_sources.DatabaseService") as MockDatabase:
        database = Mock()
        database.get_available_sources = AsyncMock(side_effect=Exception("DB down"))
        MockDatabase.return_value = database
        result = json.loads(await get_available_sources(mock_context))
    assert result["success"] is False
    assert "DB down" in result["error"]
