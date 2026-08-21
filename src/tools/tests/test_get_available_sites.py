"""Tests for get_available_sites tool returning typed AvailableSitesResponse."""

from unittest.mock import Mock

import pytest

from crawl4ai_mcp.conftest import FakeGraphStore
from crawl4ai_mcp.models import AvailableSitesResponse, SiteInfo
from crawl4ai_mcp.tools.get_available_sites import get_available_sites


@pytest.mark.asyncio
async def test_get_available_sites_success(
    mock_mcp_context: Mock,
    fake_graph_store: FakeGraphStore,
) -> None:
    fake_graph_store.sites = [
        SiteInfo(
            site_id="docs.example.com",
            domain="docs.example.com",
            root_url="https://docs.example.com",
            title="Example Documentation",
            summary="Documentation portal for example API.",
            page_count=12,
            chunk_count=48,
            gliner_metadata={"enabled": True},
        )
    ]

    response = await get_available_sites(mock_mcp_context)

    assert isinstance(response, AvailableSitesResponse)
    assert response.success is True
    assert response.total_sites == 1
    assert len(response.sites) == 1
    assert response.sites[0].site_id == "docs.example.com"
    assert response.sites[0].domain == "docs.example.com"
    assert response.sites[0].page_count == 12
    assert response.sites[0].chunk_count == 48
    assert response.sites[0].gliner_metadata == {"enabled": True}


@pytest.mark.asyncio
async def test_get_available_sites_graph_error(
    mock_mcp_context: Mock,
    fake_graph_store: FakeGraphStore,
) -> None:
    fake_graph_store.should_fail = True

    response = await get_available_sites(mock_mcp_context)

    assert isinstance(response, AvailableSitesResponse)
    assert response.success is False
    assert "Graph store get_available_sites failure simulated" in (response.message or "")


@pytest.mark.asyncio
async def test_get_available_sites_uninitialized_context() -> None:
    empty_context = Mock()
    empty_context.request_context = None

    response = await get_available_sites(empty_context)
    assert isinstance(response, AvailableSitesResponse)
    assert response.success is False
    assert "FastMCP server lifespan is not initialized" in (response.message or "")
