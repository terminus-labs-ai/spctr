"""Tests for MCPToolBridge.

Uses minimal mock objects in place of mcp.types.Tool and
mcp.types.CallToolResult — the mcp package is NOT imported.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from sr2_spectre.mcp.tool_bridge import MCPToolBridge


# ---------------------------------------------------------------------------
# Mock helpers (stand-ins for mcp.types.* without installing the package)
# ---------------------------------------------------------------------------

def _make_mcp_tool(
    name: str = "my_tool",
    description: str | None = "Does a thing",
    input_schema: dict | None = None,
) -> SimpleNamespace:
    """Minimal stand-in for mcp.types.Tool."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema or {"type": "object", "properties": {}},
    )


def _make_content_block(text: str) -> SimpleNamespace:
    """Minimal text content block (type='text')."""
    return SimpleNamespace(type="text", text=text)


def _make_non_text_block() -> SimpleNamespace:
    """Minimal non-text content block (e.g., type='image')."""
    return SimpleNamespace(type="image")


def _make_call_tool_result(content: list) -> SimpleNamespace:
    """Minimal stand-in for mcp.types.CallToolResult."""
    return SimpleNamespace(content=content)


def _make_call_tool(result: SimpleNamespace) -> AsyncMock:
    """Return an async callable that returns the given result."""
    mock = AsyncMock(return_value=result)
    return mock


# ---------------------------------------------------------------------------
# Attribute extraction
# ---------------------------------------------------------------------------

def test_name_is_extracted_from_mcp_tool() -> None:
    tool = _make_mcp_tool(name="search_files")
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=AsyncMock())
    assert bridge.name == "search_files"


def test_description_is_extracted_from_mcp_tool() -> None:
    tool = _make_mcp_tool(description="Searches files on disk")
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=AsyncMock())
    assert bridge.description == "Searches files on disk"


def test_description_defaults_to_empty_string_when_none() -> None:
    tool = _make_mcp_tool(description=None)
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=AsyncMock())
    assert bridge.description == ""


def test_input_schema_is_extracted_from_mcp_tool() -> None:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    tool = _make_mcp_tool(input_schema=schema)
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=AsyncMock())
    assert bridge.input_schema == schema


def test_input_schema_value_matches_mcp_tool_schema() -> None:
    """Schema value is preserved (not re-serialised or mutated)."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    tool = _make_mcp_tool(input_schema=schema)
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=AsyncMock())
    assert bridge.input_schema == schema


# ---------------------------------------------------------------------------
# __call__ — delegation to call_tool
# ---------------------------------------------------------------------------

async def test_call_invokes_call_tool_with_tool_name_and_kwargs() -> None:
    tool = _make_mcp_tool(name="list_dir")
    result = _make_call_tool_result([_make_content_block("file1\nfile2")])
    call_tool = _make_call_tool(result)

    bridge = MCPToolBridge(mcp_tool=tool, call_tool=call_tool)
    await bridge(path="/tmp")

    call_tool.assert_awaited_once_with("list_dir", {"path": "/tmp"})


async def test_call_passes_multiple_kwargs_to_call_tool() -> None:
    tool = _make_mcp_tool(name="search")
    result = _make_call_tool_result([_make_content_block("hit")])
    call_tool = _make_call_tool(result)

    bridge = MCPToolBridge(mcp_tool=tool, call_tool=call_tool)
    await bridge(query="foo", limit=10, recursive=True)

    call_tool.assert_awaited_once_with("search", {"query": "foo", "limit": 10, "recursive": True})


async def test_call_passes_empty_kwargs_to_call_tool() -> None:
    tool = _make_mcp_tool(name="ping")
    result = _make_call_tool_result([_make_content_block("pong")])
    call_tool = _make_call_tool(result)

    bridge = MCPToolBridge(mcp_tool=tool, call_tool=call_tool)
    await bridge()

    call_tool.assert_awaited_once_with("ping", {})


# ---------------------------------------------------------------------------
# __call__ — text extraction from CallToolResult
# ---------------------------------------------------------------------------

async def test_single_text_block_returns_its_text() -> None:
    tool = _make_mcp_tool()
    result = _make_call_tool_result([_make_content_block("hello world")])
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=_make_call_tool(result))

    output = await bridge()

    assert output == "hello world"


async def test_multiple_text_blocks_are_joined_with_newline() -> None:
    tool = _make_mcp_tool()
    result = _make_call_tool_result([
        _make_content_block("line one"),
        _make_content_block("line two"),
        _make_content_block("line three"),
    ])
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=_make_call_tool(result))

    output = await bridge()

    assert output == "line one\nline two\nline three"


async def test_non_text_blocks_are_ignored() -> None:
    tool = _make_mcp_tool()
    result = _make_call_tool_result([
        _make_non_text_block(),
        _make_content_block("useful text"),
        _make_non_text_block(),
    ])
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=_make_call_tool(result))

    output = await bridge()

    assert output == "useful text"


async def test_mixed_content_concatenates_only_text_blocks() -> None:
    tool = _make_mcp_tool()
    result = _make_call_tool_result([
        _make_content_block("first"),
        _make_non_text_block(),
        _make_content_block("second"),
        _make_non_text_block(),
    ])
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=_make_call_tool(result))

    output = await bridge()

    assert output == "first\nsecond"


async def test_empty_content_list_returns_empty_string() -> None:
    tool = _make_mcp_tool()
    result = _make_call_tool_result([])
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=_make_call_tool(result))

    output = await bridge()

    assert output == ""


async def test_all_non_text_blocks_returns_empty_string() -> None:
    tool = _make_mcp_tool()
    result = _make_call_tool_result([
        _make_non_text_block(),
        _make_non_text_block(),
    ])
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=_make_call_tool(result))

    output = await bridge()

    assert output == ""


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------

async def test_call_tool_exception_propagates() -> None:
    tool = _make_mcp_tool(name="exploder")
    call_tool = AsyncMock(side_effect=RuntimeError("MCP server died"))

    bridge = MCPToolBridge(mcp_tool=tool, call_tool=call_tool)

    with pytest.raises(RuntimeError, match="MCP server died"):
        await bridge(x=1)


async def test_call_tool_value_error_propagates() -> None:
    tool = _make_mcp_tool(name="validator")
    call_tool = AsyncMock(side_effect=ValueError("bad argument"))

    bridge = MCPToolBridge(mcp_tool=tool, call_tool=call_tool)

    with pytest.raises(ValueError, match="bad argument"):
        await bridge()


# ---------------------------------------------------------------------------
# Async detection — ToolRegistry compatibility
# ---------------------------------------------------------------------------

def test_bridge_call_is_detected_as_coroutine_function() -> None:
    """ToolRegistry sets is_async=True when iscoroutinefunction(fn) is True."""
    tool = _make_mcp_tool()
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=AsyncMock())
    assert asyncio.iscoroutinefunction(bridge.__call__)


def test_bridge_is_registered_as_async_in_tool_registry() -> None:
    """Smoke test: registering the bridge produces is_async=True in ToolSpec."""
    from sr2_spectre.tools.registry import ToolRegistry

    tool = _make_mcp_tool(
        name="registered_tool",
        description="A bridged MCP tool",
        input_schema={"type": "object", "properties": {}},
    )
    bridge = MCPToolBridge(mcp_tool=tool, call_tool=AsyncMock())

    reg = ToolRegistry()
    reg.register(
        name=bridge.name,
        description=bridge.description,
        input_schema=bridge.input_schema,
        fn=bridge,
    )

    spec = reg._tools["registered_tool"]
    assert spec.is_async is True
    assert spec.name == "registered_tool"
    assert spec.description == "A bridged MCP tool"
