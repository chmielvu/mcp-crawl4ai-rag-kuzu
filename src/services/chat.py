"""Mistral chat-only service implementing ChatGeneratorPort."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Callable, cast

from mistralai.client import Mistral

from crawl4ai_mcp.config import Settings, get_settings
from crawl4ai_mcp.services.contracts import ChatGeneratorPort

logger = logging.getLogger(__name__)


class ChatProviderError(Exception):
    """Structured provider error from Mistral chat completions."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def _extract_message_text(response: Any) -> str:
    """Extract string content from a Mistral completion response."""
    if not response or not hasattr(response, "choices") or not response.choices:
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts).strip()
    return str(content).strip()


class ChatGenerator(ChatGeneratorPort):
    """Chat-only Mistral service for text completions and summaries."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api_key: str | None = None,
        model_choice: str | None = None,
        client: Mistral | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.mistral_api_key
        self.model_choice = model_choice or self.settings.model_choice
        self.client = client or Mistral(api_key=self.api_key)
        self.max_retries = 3
        self.retry_delay = 1.0

    async def _run_in_executor(self, func: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

    async def aclose(self) -> None:
        """Close client resources if supported."""
        if hasattr(self.client, "aclose") and callable(self.client.aclose):
            try:
                await self.client.aclose()
            except Exception as exc:
                logger.debug("Mistral client aclose error (ignored): %s", exc)
        elif hasattr(self.client, "close") and callable(self.client.close):
            try:
                self.client.close()
            except Exception as exc:
                logger.debug("Mistral client close error (ignored): %s", exc)

    async def chat_complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 150,
    ) -> str:
        """Generate a text completion using the configured Mistral chat model."""
        if not messages:
            raise ChatProviderError("Cannot execute chat completion with empty messages")

        def _call_mistral() -> Any:
            return self.client.chat.complete(
                model=self.model_choice,
                messages=cast(Any, list(messages)),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self._run_in_executor(_call_mistral)
                return _extract_message_text(response)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Mistral chat_complete attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))

        raise ChatProviderError(
            f"Mistral chat_complete failed after {self.max_retries} attempts: {last_error}",
            details={"error": str(last_error)},
        ) from last_error

    async def generate_code_example_summary(
        self, code: str, context_before: str, context_after: str
    ) -> str:
        """Generate a 2-3 sentence summary of what a code example demonstrates."""
        before_slice = context_before[-500:] if len(context_before) > 500 else context_before
        code_slice = code[:1500] if len(code) > 1500 else code
        after_slice = context_after[:500] if len(context_after) > 500 else context_after

        prompt = f"""<context_before>
{before_slice}
</context_before>

<code_example>
{code_slice}
</code_example>

<context_after>
{after_slice}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated."""

        messages = [
            {
                "role": "system",
                "content": "You provide concise code example summaries.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            return await self.chat_complete(messages, max_tokens=100)
        except Exception as error:
            logger.error("Error generating code example summary: %s", error)
            return "Code example for demonstration purposes."

    async def extract_source_summary(
        self, source_id: str, content: str, max_length: int = 500
    ) -> str:
        """Generate a concise 3-5 sentence summary of source documentation."""
        default_summary = f"Content from {source_id}"
        if not content.strip():
            return default_summary

        prompt = f"""<source_content>
{content[:25000]}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose."""

        messages = [
            {
                "role": "system",
                "content": "You provide concise library and framework summaries.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            summary = await self.chat_complete(messages, max_tokens=150)
            if len(summary) > max_length:
                return f"{summary[:max_length]}..."
            return summary
        except Exception as error:
            logger.error("Error generating summary for %s: %s", source_id, error)
            return default_summary
