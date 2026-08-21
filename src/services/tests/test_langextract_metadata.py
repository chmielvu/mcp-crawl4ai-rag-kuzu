"""Tests for LangExtractMetadata extraction, grounding, and discarding rules."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.services.contracts import PageExtraction
from crawl4ai_mcp.services.langextract_metadata import (
    LangExtractError,
    LangExtractMetadata,
    _normalize_extractions,
)


@pytest.mark.asyncio
async def test_extract_page_empty_text_returns_empty(test_settings: Settings) -> None:
    runner_called = False

    def runner() -> Any:
        nonlocal runner_called
        runner_called = True
        return Mock()

    service = LangExtractMetadata(settings=test_settings, client_runner=runner)
    res = await service.extract_page("   ")
    assert res == []
    assert runner_called is False


@pytest.mark.asyncio
async def test_extract_page_normalizes_and_discards_invalid(test_settings: Settings) -> None:
    # 1 valid, 1 invalid class, 1 missing text, 1 invalid bounds (end < start), 1 negative start
    raw_extractions = [
        SimpleNamespace(
            extraction_class="technology",
            extraction_text="FastMCP",
            char_interval=SimpleNamespace(start_pos=0, end_pos=7),
            attributes={"framework": "mcp"},
            description="Framework description",
        ),
        SimpleNamespace(
            extraction_class="unsupported_class",
            extraction_text="Ignored",
            char_interval=SimpleNamespace(start_pos=10, end_pos=17),
        ),
        SimpleNamespace(
            extraction_class="product",
            extraction_text="",
            char_interval=SimpleNamespace(start_pos=20, end_pos=25),
        ),
        SimpleNamespace(
            extraction_class="version",
            extraction_text="1.0",
            char_interval=SimpleNamespace(start_pos=30, end_pos=25),  # end < start
        ),
        SimpleNamespace(
            extraction_class="organization",
            extraction_text="Org",
            char_interval=SimpleNamespace(start_pos=-5, end_pos=5),  # negative start
        ),
    ]

    mock_result = SimpleNamespace(extractions=raw_extractions)

    service = LangExtractMetadata(
        settings=test_settings,
        client_runner=lambda: mock_result,
    )
    extractions = await service.extract_page("FastMCP is a python framework.")

    assert len(extractions) == 1
    ext = extractions[0]
    assert isinstance(ext, PageExtraction)
    assert ext.extraction_class == "technology"
    assert ext.extraction_text == "FastMCP"
    assert ext.start_char == 0
    assert ext.end_char == 7
    assert ext.attributes == {"framework": "mcp"}
    assert ext.description == "Framework description"


@pytest.mark.asyncio
async def test_extract_page_dict_format(test_settings: Settings) -> None:
    raw_dict_extractions = {
        "extractions": [
            {
                "extraction_class": "person",
                "extraction_text": "Alice",
                "char_interval": {"start_pos": 5, "end_pos": 10},
                "attributes": {"role": "maintainer"},
                "description": "Lead maintainer",
            }
        ]
    }

    service = LangExtractMetadata(
        settings=test_settings,
        client_runner=lambda: raw_dict_extractions,
    )
    extractions = await service.extract_page("Authored by Alice in 2026.")

    assert len(extractions) == 1
    assert extractions[0].extraction_class == "person"
    assert extractions[0].extraction_text == "Alice"
    assert extractions[0].start_char == 5
    assert extractions[0].end_char == 10
    assert extractions[0].attributes == {"role": "maintainer"}


@pytest.mark.asyncio
async def test_extract_page_runner_error_raises_langextract_error(test_settings: Settings) -> None:
    def failing_runner() -> Any:
        raise ValueError("Remote model failed to generate response")

    service = LangExtractMetadata(
        settings=test_settings,
        client_runner=failing_runner,
    )
    with pytest.raises(LangExtractError, match="page extraction failed"):
        await service.extract_page("Some text")


def test_normalize_extractions_missing_extractions_field() -> None:
    with pytest.raises(LangExtractError, match="did not contain extractions"):
        _normalize_extractions({"invalid": "payload"})
