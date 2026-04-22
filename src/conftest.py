"""Shared pytest fixtures for all tests."""

from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, Mock

import pytest


class MockQueryResult:
    """Minimal Kuzu query result test double."""

    def __init__(self, columns: list[str], rows: list[list[Any]]):
        self._columns = columns
        self._rows = rows

    def get_column_names(self) -> list[str]:
        return self._columns

    def get_all(self) -> list[list[Any]]:
        return self._rows


@pytest.fixture
def test_settings() -> Any:
    """Provide settings without requiring environment variables."""
    return SimpleNamespace(
        mistral_api_key="test-api-key",
        model_choice="mistral-small-latest",
        embedding_model="mistral-embed",
        embedding_dimensions=1024,
        kuzu_db_path="./data/test-kuzu",
        use_contextual_embeddings=False,
        use_hybrid_search=False,
        use_reranking=False,
        use_agentic_rag=False,
        host="0.0.0.0",
        port=8051,
        transport="sse",
        reranker_model="ms-marco-MiniLM-L-12-v2",
        reranker_cache_dir="./data/flashrank_cache",
        reranker_max_length=512,
        default_max_depth=3,
        default_max_concurrent=10,
        default_chunk_size=5000,
        default_overlap=200,
        default_num_results=5,
        default_semantic_threshold=0.5,
        default_rerank_threshold=0.3,
    )


@pytest.fixture
def mock_db_connection() -> Mock:
    """Mock Kuzu connection for testing."""
    client = Mock()
    client.execute = Mock(return_value=MockQueryResult([], []))
    client.close = Mock()
    return client


@pytest.fixture
def mock_mistral_client() -> Mock:
    """Mock Mistral client for testing."""
    client = Mock()
    client.embeddings = Mock()
    client.embeddings.create = Mock(
        return_value=Mock(data=[Mock(embedding=[0.1] * 1024)])
    )
    client.chat = Mock()
    client.chat.complete = Mock(
        return_value=Mock(choices=[Mock(message=Mock(content="Test response"))])
    )
    return client


@pytest.fixture
def mock_crawler() -> Mock:
    """Mock Crawl4AI crawler for testing."""
    crawler = Mock()
    crawler.arun = AsyncMock(
        return_value=Mock(
            success=True,
            markdown="# Test Content",
            cleaned_html="<h1>Test Content</h1>",
            extracted_content="Test Content",
            media={"images": [], "videos": [], "audios": []},
            links={"internal": [], "external": []},
            metadata={},
            screenshot=None,
            error_message=None,
        )
    )
    crawler.close = AsyncMock()
    return crawler


@pytest.fixture
def sample_document() -> dict[str, Any]:
    """Sample document for testing."""
    return {
        "url": "https://example.com/test",
        "content": "This is test content for unit testing.",
        "chunk_number": 1,
        "total_chunks": 1,
        "word_count": 7,
        "source": "example.com",
        "metadata": {
            "title": "Test Page",
            "description": "A test page for unit testing",
        },
    }


@pytest.fixture
def sample_code_example() -> dict[str, Any]:
    """Sample code example for testing."""
    return {
        "code": "def hello():\n    print('Hello, world!')",
        "language": "python",
        "context": "A simple hello world function",
        "summary": "Prints hello world",
        "url": "https://example.com/code",
        "source": "example.com",
    }


@pytest.fixture
async def async_iterator(items: list[Any]) -> AsyncIterator[Any]:
    """Create an async iterator from a list of items."""
    for item in items:
        yield item
