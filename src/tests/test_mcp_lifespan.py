"""Regression tests for MCP lifespan startup cleanup."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from crawl4ai_mcp import mcp_server


class _FakeResource:
    def __init__(self, *, health_error: Exception | None = None) -> None:
        self.health_error = health_error
        self.closed = False

    async def health_check(self) -> None:
        if self.health_error is not None:
            raise self.health_error

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_lifespan_closes_clients_when_health_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        use_langextract_metadata=False,
        validate_required_fields=lambda: None,
    )
    crawler = _FakeResource(health_error=RuntimeError("crawler unavailable"))
    unified_ml = _FakeResource()
    chat = _FakeResource()

    monkeypatch.setattr(
        "crawl4ai_mcp.config.get_settings", lambda: settings
    )
    monkeypatch.setattr(
        "crawl4ai_mcp.services.crawl4ai_client.Crawl4AIClient",
        lambda settings: crawler,
    )
    monkeypatch.setattr(
        "crawl4ai_mcp.services.unified_ml_client.UnifiedMLClient",
        lambda settings: unified_ml,
    )
    monkeypatch.setattr(
        "crawl4ai_mcp.services.chat.ChatGenerator",
        lambda settings: chat,
    )

    with pytest.raises(RuntimeError, match="crawler unavailable"):
        async with mcp_server.crawl4ai_lifespan(Mock()):
            pass

    assert crawler.closed is True
    assert unified_ml.closed is True
    assert chat.closed is True
