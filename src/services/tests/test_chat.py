"""Tests for Mistral ChatGenerator service and ChatGeneratorPort implementation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.services.chat import (
    ChatGenerator,
    ChatProviderError,
    _extract_message_text,
)


def test_extract_message_text_formats() -> None:
    assert _extract_message_text(None) == ""

    # String content
    resp1 = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello world!"))]
    )
    assert _extract_message_text(resp1) == "Hello world!"

    # List of strings
    resp2 = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=["Part 1", "Part 2"]))]
    )
    assert _extract_message_text(resp2) == "Part 1Part 2"

    # List of dicts
    resp3 = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[{"text": "Chunk A"}, {"text": "Chunk B"}]
                )
            )
        ]
    )
    assert _extract_message_text(resp3) == "Chunk AChunk B"


@pytest.mark.asyncio
async def test_chat_complete_success(test_settings: Settings) -> None:
    mock_client = Mock()
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Assistant response."))]
    )
    mock_client.chat = Mock()
    mock_client.chat.complete = Mock(return_value=mock_response)

    service = ChatGenerator(settings=test_settings, client=mock_client)
    res = await service.chat_complete([{"role": "user", "content": "Hi"}])

    assert res == "Assistant response."
    mock_client.chat.complete.assert_called_once()


@pytest.mark.asyncio
async def test_chat_complete_client_error_raises_chat_provider_error(test_settings: Settings) -> None:
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.complete = Mock(side_effect=Exception("API key invalid"))

    service = ChatGenerator(settings=test_settings, client=mock_client)
    service.max_retries = 1
    with pytest.raises(ChatProviderError, match="API key invalid"):
        await service.chat_complete([{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_generate_code_example_summary(test_settings: Settings) -> None:
    mock_client = Mock()
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Calculates sum of two integers."))]
    )
    mock_client.chat = Mock()
    mock_client.chat.complete = Mock(return_value=mock_response)

    service = ChatGenerator(settings=test_settings, client=mock_client)
    summary = await service.generate_code_example_summary(
        code="def add(a, b): return a + b",
        context_before="Math helper function",
        context_after="End of module",
    )

    assert summary == "Calculates sum of two integers."


@pytest.mark.asyncio
async def test_extract_source_summary(test_settings: Settings) -> None:
    mock_client = Mock()
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Comprehensive developer documentation."))]
    )
    mock_client.chat = Mock()
    mock_client.chat.complete = Mock(return_value=mock_response)

    service = ChatGenerator(settings=test_settings, client=mock_client)
    summary = await service.extract_source_summary(
        source_id="example.com",
        content="Documentation content for the site.",
        max_length=500,
    )

    assert summary == "Comprehensive developer documentation."
