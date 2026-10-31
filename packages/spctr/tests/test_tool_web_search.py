"""Tests for WebSearchTool."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_web_search_class_attributes() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    assert isinstance(WebSearchTool.name, str) and WebSearchTool.name
    assert isinstance(WebSearchTool.description, str) and WebSearchTool.description
    assert isinstance(WebSearchTool.input_schema, dict)


def test_web_search_input_schema_requires_query() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    schema = WebSearchTool.input_schema
    assert "query" in schema.get("properties", {})
    assert "query" in schema.get("required", [])


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_web_search_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path(
        "sr2_spectre.tools.builtins.web_search.WebSearchTool",
        config={"base_url": "http://localhost:8080"},
    )
    assert "web_search" in reg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(results: list[dict], status: int = 200) -> MagicMock:
    """Build a mock aiohttp response."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value={"results": results})
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _mock_session(response: MagicMock) -> MagicMock:
    """Wrap a mock response in a mock aiohttp.ClientSession context manager."""
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


# ---------------------------------------------------------------------------
# Successful search — result formatting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_search_formats_single_result() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    results = [{"title": "Example", "url": "https://example.com", "content": "Some content"}]
    resp = _make_response(results)
    session = _mock_session(resp)

    with patch("aiohttp.ClientSession", return_value=session):
        tool = WebSearchTool(base_url="http://localhost:8080")
        result = await tool(query="test")

    assert "[1]" in result
    assert "Example" in result
    assert "https://example.com" in result
    assert "Some content" in result


@pytest.mark.asyncio
async def test_web_search_formats_multiple_results_with_indices() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    results = [
        {"title": "First", "url": "https://first.com", "content": "Content one"},
        {"title": "Second", "url": "https://second.com", "content": "Content two"},
        {"title": "Third", "url": "https://third.com", "content": "Content three"},
    ]
    resp = _make_response(results)
    session = _mock_session(resp)

    with patch("aiohttp.ClientSession", return_value=session):
        tool = WebSearchTool(base_url="http://localhost:8080")
        result = await tool(query="multi")

    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result


@pytest.mark.asyncio
async def test_web_search_respects_max_results() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    results = [
        {"title": f"Result {i}", "url": f"https://r{i}.com", "content": f"Content {i}"}
        for i in range(1, 11)  # 10 results from server
    ]
    resp = _make_response(results)
    session = _mock_session(resp)

    with patch("aiohttp.ClientSession", return_value=session):
        tool = WebSearchTool(base_url="http://localhost:8080", max_results=3)
        result = await tool(query="many")

    assert "[3]" in result
    assert "[4]" not in result


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_search_sends_correct_url() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    resp = _make_response([])
    session = _mock_session(resp)

    with patch("aiohttp.ClientSession", return_value=session):
        tool = WebSearchTool(base_url="http://localhost:8080")
        await tool(query="python async")

    call = session.get.call_args
    called_url = call.args[0] if call.args else call.kwargs.get("url", "")
    assert called_url.startswith("http://localhost:8080/search")
    assert "python" in called_url
    assert "format=json" in called_url


# ---------------------------------------------------------------------------
# No results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_search_no_results_returns_message() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    resp = _make_response([])
    session = _mock_session(resp)

    with patch("aiohttp.ClientSession", return_value=session):
        tool = WebSearchTool(base_url="http://localhost:8080")
        result = await tool(query="xyzzy nonce")

    assert result == "No results found."


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_search_http_error_raises_runtime_error() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    resp = _make_response([], status=500)
    session = _mock_session(resp)

    with patch("aiohttp.ClientSession", return_value=session):
        tool = WebSearchTool(base_url="http://localhost:8080")
        with pytest.raises(RuntimeError):
            await tool(query="broken")


@pytest.mark.asyncio
async def test_web_search_http_error_message_contains_status_code() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    resp = _make_response([], status=503)
    session = _mock_session(resp)

    with patch("aiohttp.ClientSession", return_value=session):
        tool = WebSearchTool(base_url="http://localhost:8080")
        with pytest.raises(RuntimeError) as exc_info:
            await tool(query="unavailable")

    assert "503" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Constructor args
# ---------------------------------------------------------------------------

def test_web_search_stores_base_url_and_max_results() -> None:
    from sr2_spectre.tools.builtins.web_search import WebSearchTool

    tool = WebSearchTool(base_url="http://searx.local", max_results=10)
    assert tool.base_url == "http://searx.local"
    assert tool.max_results == 10
