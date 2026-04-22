"""Tests for embeddings service."""

from unittest.mock import Mock, patch

import pytest

from crawl4ai_mcp.services.embeddings import EmbeddingService


@pytest.fixture
def embedding_service(test_settings):
    """Create EmbeddingService with mocked Mistral client."""
    with patch("crawl4ai_mcp.services.embeddings.Mistral") as MockMistral:
        client = Mock()
        MockMistral.return_value = client
        service = EmbeddingService(test_settings)
        yield service


@pytest.fixture
def mock_embedding_response():
    """Mock Mistral embeddings response."""
    response = Mock()
    response.data = [
        Mock(embedding=[0.1] * 1024),
        Mock(embedding=[0.2] * 1024),
        Mock(embedding=[0.3] * 1024),
    ]
    return response


@pytest.fixture
def mock_chat_response():
    """Mock Mistral chat completion response."""
    response = Mock()
    response.choices = [
        Mock(message=Mock(content="This chunk discusses the main topic of the document."))
    ]
    return response


@pytest.mark.asyncio
async def test_create_embeddings_batch_success(
    embedding_service, mock_embedding_response
) -> None:
    with patch.object(
        embedding_service.client.embeddings,
        "create",
        return_value=mock_embedding_response,
    ):
        embeddings = await embedding_service.create_embeddings_batch(
            ["text1", "text2", "text3"]
        )
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 1024
    assert embeddings[1][0] == 0.2


@pytest.mark.asyncio
async def test_create_embeddings_batch_empty_list(embedding_service) -> None:
    assert await embedding_service.create_embeddings_batch([]) == []


@pytest.mark.asyncio
async def test_create_embeddings_batch_rate_limit_retry(
    embedding_service, mock_embedding_response
) -> None:
    with patch.object(
        embedding_service.client.embeddings,
        "create",
        side_effect=[Exception("rate_limit_exceeded"), mock_embedding_response],
    ) as mock_create:
        embeddings = await embedding_service.create_embeddings_batch(["text1", "text2"])
    assert len(embeddings) == 3  # response fixture returns three embeddings
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_create_embeddings_batch_fallback_to_individual(embedding_service) -> None:
    responses = [
        Mock(data=[Mock(embedding=[0.1] * 1024)]),
        Mock(data=[Mock(embedding=[0.2] * 1024)]),
    ]
    with patch.object(
        embedding_service.client.embeddings,
        "create",
        side_effect=[
            Exception("API error"),
            Exception("API error"),
            Exception("API error"),
            *responses,
        ],
    ) as mock_create:
        embeddings = await embedding_service.create_embeddings_batch(["text1", "text2"])
    assert len(embeddings) == 2
    assert embeddings[0][0] == 0.1
    assert mock_create.call_count == 5


@pytest.mark.asyncio
async def test_create_embeddings_batch_partial_failure(embedding_service) -> None:
    with patch.object(
        embedding_service.client.embeddings,
        "create",
        side_effect=[
            Exception("API error"),
            Exception("API error"),
            Exception("API error"),
            Mock(data=[Mock(embedding=[0.1] * 1024)]),
            Exception("Individual call failed"),
            Mock(data=[Mock(embedding=[0.3] * 1024)]),
        ],
    ):
        embeddings = await embedding_service.create_embeddings_batch(
            ["text1", "text2", "text3"]
        )
    assert len(embeddings) == 3
    assert embeddings[1][0] == 0.0
    assert embeddings[2][0] == 0.3


@pytest.mark.asyncio
async def test_create_embedding_success(embedding_service) -> None:
    response = Mock(data=[Mock(embedding=[0.5] * 1024)])
    with patch.object(embedding_service.client.embeddings, "create", return_value=response):
        embedding = await embedding_service.create_embedding("test text")
    assert len(embedding) == 1024
    assert embedding[0] == 0.5


@pytest.mark.asyncio
async def test_create_embedding_failure(embedding_service) -> None:
    with patch.object(
        embedding_service.client.embeddings,
        "create",
        side_effect=Exception("API error"),
    ):
        embedding = await embedding_service.create_embedding("test text")
    assert len(embedding) == 1024
    assert all(value == 0.0 for value in embedding)


@pytest.mark.asyncio
async def test_generate_contextual_embedding_success(
    embedding_service, mock_chat_response
) -> None:
    with patch.object(
        embedding_service.client.chat,
        "complete",
        return_value=mock_chat_response,
    ):
        contextual_text, was_contextualized = await embedding_service.generate_contextual_embedding(
            "This is a document about Python programming...",
            "Functions in Python are defined using the def keyword.",
        )
    assert was_contextualized is True
    assert "This chunk discusses the main topic" in contextual_text


@pytest.mark.asyncio
async def test_generate_contextual_embedding_failure(embedding_service) -> None:
    with patch.object(
        embedding_service.client.chat,
        "complete",
        side_effect=Exception("API error"),
    ):
        contextual_text, was_contextualized = await embedding_service.generate_contextual_embedding(
            "This is a document about Python programming...",
            "Functions in Python are defined using the def keyword.",
        )
    assert was_contextualized is False
    assert contextual_text == "Functions in Python are defined using the def keyword."


@pytest.mark.asyncio
async def test_process_chunks_with_context_success(
    embedding_service, mock_chat_response
) -> None:
    with patch.object(
        embedding_service.client.chat,
        "complete",
        return_value=mock_chat_response,
    ):
        results = await embedding_service.process_chunks_with_context(
            [("url1", "chunk1", "full_doc1"), ("url2", "chunk2", "full_doc2")]
        )
    assert len(results) == 2
    assert all(flag is True for _, flag in results)


@pytest.mark.asyncio
async def test_process_chunks_with_context_mixed_results(
    embedding_service, mock_chat_response
) -> None:
    with patch.object(
        embedding_service.client.chat,
        "complete",
        side_effect=[mock_chat_response, Exception("API error"), mock_chat_response],
    ):
        results = await embedding_service.process_chunks_with_context(
            [("url1", "chunk1", "full_doc1"), ("url2", "chunk2", "full_doc2"), ("url3", "chunk3", "full_doc3")]
        )
    assert results[0][1] is True
    assert results[1] == ("chunk2", False)
    assert results[2][1] is True


def test_process_chunk_with_context_sync(embedding_service, mock_chat_response) -> None:
    with patch.object(
        embedding_service.client.chat,
        "complete",
        return_value=mock_chat_response,
    ):
        contextual_text, was_contextualized = embedding_service.process_chunk_with_context(
            ("url", "chunk content", "full document")
        )
    assert was_contextualized is True
    assert "chunk content" in contextual_text


@pytest.mark.asyncio
async def test_long_document_truncation(embedding_service, mock_chat_response) -> None:
    with patch.object(
        embedding_service.client.chat,
        "complete",
        return_value=mock_chat_response,
    ) as mock_complete:
        await embedding_service.generate_contextual_embedding("x" * 30000, "test chunk")
    messages = mock_complete.call_args.kwargs["messages"]
    assert len(messages[1]["content"]) < 26000
