"""Optional page-level LangExtract enrichment with grounded spans."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import langextract as lx
from langextract.core.data import ExampleData, Extraction
from langextract.factory import ModelConfig

from crawl4ai_mcp.config import Settings, get_settings
from crawl4ai_mcp.services.contracts import PageExtraction


class LangExtractError(RuntimeError):
    """Structured page extraction failure."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


_EXTRACTION_CLASSES = (
    "page_topic",
    "page_type",
    "technology",
    "product",
    "organization",
    "person",
    "version",
    "relation",
)

_PROMPT = """Extract only grounded metadata from this technical web page.
For every extraction choose exactly one extraction_class from: page_topic, page_type,
technology, product, organization, person, version, relation. Preserve the exact
extraction_text from the page. Add attributes for relation source/target/type or
other useful structured details. Every extraction must be grounded in the source
text; do not invent values."""

_EXAMPLES = [
    ExampleData(
        text="FastMCP is a Python framework. Version 1.7.1 supports the Model Context Protocol.",
        extractions=[
            Extraction(
                extraction_class="technology",
                extraction_text="FastMCP",
                attributes={"language": "Python"},
            ),
            Extraction(extraction_class="version", extraction_text="1.7.1"),
            Extraction(
                extraction_class="page_topic",
                extraction_text="Model Context Protocol",
            ),
        ],
    )
]


class LangExtractMetadata:
    """One-call-per-page LangExtract adapter."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client_runner: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client_runner = client_runner
        self._model_config = ModelConfig(
            model_id=self.settings.langextract_model_id,
            provider="openai",
            provider_kwargs={
                "api_key": self.settings.mistral_api_key,
                "base_url": self.settings.langextract_base_url,
            },
        )

    async def aclose(self) -> None:
        """Release adapter state."""


    async def extract_page(self, text: str) -> list[PageExtraction]:
        """Extract fixed-schema grounded metadata once for one page."""

        if not text.strip():
            return []

        def run() -> Any:
            if self._client_runner is not None:
                return self._client_runner()
            return lx.extract(
                text_or_documents=text,
                prompt_description=_PROMPT,
                examples=_EXAMPLES,
                config=self._model_config,
                extraction_passes=self.settings.langextract_extraction_passes,
                max_workers=self.settings.langextract_max_workers,
                max_char_buffer=self.settings.langextract_max_char_buffer,
            )

        try:
            loop = asyncio.get_running_loop()
            raw_result = await loop.run_in_executor(None, run)
        except Exception as exc:
            raise LangExtractError(
                "LangExtract page extraction failed",
                details={"error_type": type(exc).__name__, "message": str(exc)},
            ) from exc
        return _normalize_extractions(raw_result)

    async def extract(self, text: str) -> list[PageExtraction]:
        """Extract one page; kept as the service's concise public method."""

        return await self.extract_page(text)


def _normalize_extractions(raw_result: Any) -> list[PageExtraction]:
    values = getattr(raw_result, "extractions", None)
    if values is None and isinstance(raw_result, dict):
        values = raw_result.get("extractions")
    if not isinstance(values, list):
        raise LangExtractError("LangExtract response did not contain extractions")

    normalized: list[PageExtraction] = []
    for extraction in values:
        extraction_class = getattr(extraction, "extraction_class", None)
        extraction_text = getattr(extraction, "extraction_text", None)
        interval = getattr(extraction, "char_interval", None)
        if isinstance(extraction, dict):
            extraction_class = extraction.get("extraction_class", extraction_class)
            extraction_text = extraction.get("extraction_text", extraction_text)
            interval = extraction.get("char_interval", interval)
        if extraction_class not in _EXTRACTION_CLASSES or not extraction_text:
            continue

        start, end = _interval_bounds(interval)
        if start is None or end is None or start < 0 or end <= start:
            continue
        attributes = getattr(extraction, "attributes", None)
        description = getattr(extraction, "description", None)
        if isinstance(extraction, dict):
            attributes = extraction.get("attributes", attributes)
            description = extraction.get("description", description)
        normalized.append(
            PageExtraction(
                extraction_class=str(extraction_class),
                extraction_text=str(extraction_text),
                start_char=start,
                end_char=end,
                attributes=dict(attributes) if isinstance(attributes, dict) else {},
                description=str(description) if description else None,
            )
        )
    return normalized


def _interval_bounds(interval: Any) -> tuple[int | None, int | None]:
    if interval is None:
        return None, None
    if isinstance(interval, dict):
        return _as_int(interval.get("start_pos")), _as_int(interval.get("end_pos"))
    return _as_int(getattr(interval, "start_pos", None)), _as_int(
        getattr(interval, "end_pos", None)
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
