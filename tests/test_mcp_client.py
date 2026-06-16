"""Tests for MCPClient.

Mocks stdio_client, sse_client, and ClientSession at the client module
import level — no real MCP server is required.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.mcp.tool_bridge import MCPToolBridge


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mcp_tool(
    name: str = "my_tool",
    description: str | None = "Does a thing",
    input_schema: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema or {"type": "object", "properties": {}},
    )


def _make_list_tools_result(tools: list) -> SimpleNamespace:
    return SimpleNamespace(tools=tools)


def _make_transport_ctx(read: Any = None, write: Any = None) -> AsyncMock:
    """Async context manager that yields (read, write) from __aenter__."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=(read or AsyncMock(), write or AsyncMock()))
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _make_streamable_transport_ctx(
    read: Any = None, write: Any = None, get_session_id: Any = None
) -> AsyncMock:
    """Async context manager that yields a 3-tuple (read, write, get_session_id).

    streamablehttp_client yields a third element (a get_session_id callable)
    that sse_client/stdio_client do not. The client must tolerate it.
    """
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(
        return_value=(
            read or AsyncMock(),
            write or AsyncMock(),
            get_session_id or MagicMock(return_value="sid-123"),
        )
    )
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _make_session_ctx(mock_session: AsyncMock) -> AsyncMock:
    """Async context manager that yields mock_session from __aenter__."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _make_mock_session(tools: list | None = None) -> AsyncMock:
    session = AsyncMock()
    session.initialize = AsyncMock(return_value=None)
    session.list_tools = AsyncMock(
        return_value=_make_list_tools_result(tools if tools is not None else [])
    )
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[])
    )
    return session


# ---------------------------------------------------------------------------
# Requirement 1: stdio connect — opens transport, returns MCPToolBridge list
# ---------------------------------------------------------------------------

async def test_stdio_connect_returns_bridge_per_tool() -> None:
    tools = [_make_mcp_tool("tool_a"), _make_mcp_tool("tool_b")]
    mock_session = _make_mock_session(tools)
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["my_server"], args=[], env={})
        bridges = await client.connect()

    assert len(bridges) == 2
    assert all(isinstance(b, MCPToolBridge) for b in bridges)


async def test_stdio_connect_bridge_names_match_tools() -> None:
    tools = [_make_mcp_tool("alpha"), _make_mcp_tool("beta")]
    mock_session = _make_mock_session(tools)
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        bridges = await client.connect()

    assert {b.name for b in bridges} == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Requirement 2: http connect — uses sse_client, same flow
# ---------------------------------------------------------------------------

async def test_http_connect_returns_bridge_per_tool() -> None:
    tools = [_make_mcp_tool("http_tool")]
    mock_session = _make_mock_session(tools)
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.sse_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("http", url="http://localhost:8080/sse")
        bridges = await client.connect()

    assert len(bridges) == 1
    assert bridges[0].name == "http_tool"


# ---------------------------------------------------------------------------
# Requirement 2b: streamable-http connect — uses streamablehttp_client,
# tolerates the 3-tuple (read, write, get_session_id) it yields
# ---------------------------------------------------------------------------

async def test_streamable_http_connect_returns_bridge_per_tool() -> None:
    tools = [_make_mcp_tool("glyph_search"), _make_mcp_tool("glyph_lookup")]
    mock_session = _make_mock_session(tools)
    transport_ctx = _make_streamable_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.streamablehttp_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("streamable-http", url="http://localhost:8080/mcp")
        bridges = await client.connect()

    assert {b.name for b in bridges} == {"glyph_search", "glyph_lookup"}
    # ClientSession must be built from the first two elements of the 3-tuple
    read, write = transport_ctx.__aenter__.return_value[:2]


async def test_streamable_http_connect_raises_mcp_connection_error_on_transport_failure() -> None:
    from sr2_spectre.mcp.client import MCPClient, MCPConnectionError

    failing_ctx = AsyncMock()
    failing_ctx.__aenter__ = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    failing_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("sr2_spectre.mcp.client.streamablehttp_client", return_value=failing_ctx):
        client = MCPClient("streamable-http", url="http://dead-host/mcp")
        with pytest.raises(MCPConnectionError):
            await client.connect()


# ---------------------------------------------------------------------------
# Requirement 3: bridges have correct name, description, input_schema
# ---------------------------------------------------------------------------

async def test_bridge_attributes_match_tool_definition() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    tool = _make_mcp_tool(name="search", description="Search docs", input_schema=schema)
    mock_session = _make_mock_session([tool])
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        bridges = await client.connect()

    bridge = bridges[0]
    assert bridge.name == "search"
    assert bridge.description == "Search docs"
    assert bridge.input_schema == schema


async def test_bridge_description_is_empty_string_when_tool_description_is_none() -> None:
    tool = _make_mcp_tool(name="nodesc", description=None)
    mock_session = _make_mock_session([tool])
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        bridges = await client.connect()

    assert bridges[0].description == ""


# ---------------------------------------------------------------------------
# Requirement 4: bridges are callable — call_tool is delegated to session
# ---------------------------------------------------------------------------

async def test_bridge_call_delegates_to_session_call_tool() -> None:
    tool = _make_mcp_tool(name="do_thing")
    mock_session = _make_mock_session([tool])
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        bridges = await client.connect()

    await bridges[0](key="value")
    mock_session.call_tool.assert_awaited_once_with("do_thing", {"key": "value"})


# ---------------------------------------------------------------------------
# Requirement 5: empty tools list → connect() returns []
# ---------------------------------------------------------------------------

async def test_connect_returns_empty_list_when_server_has_no_tools() -> None:
    mock_session = _make_mock_session([])
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        bridges = await client.connect()

    assert bridges == []


# ---------------------------------------------------------------------------
# Requirement 6: MCPConnectionError on stdio_client failure
# ---------------------------------------------------------------------------

async def test_stdio_connect_raises_mcp_connection_error_on_transport_failure() -> None:
    from sr2_spectre.mcp.client import MCPClient, MCPConnectionError

    failing_ctx = AsyncMock()
    failing_ctx.__aenter__ = AsyncMock(side_effect=OSError("spawn failed"))
    failing_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("sr2_spectre.mcp.client.stdio_client", return_value=failing_ctx):
        client = MCPClient("stdio", command=["no_such_binary"])
        with pytest.raises(MCPConnectionError):
            await client.connect()


# ---------------------------------------------------------------------------
# Requirement 7: MCPConnectionError on sse_client failure
# ---------------------------------------------------------------------------

async def test_http_connect_raises_mcp_connection_error_on_transport_failure() -> None:
    from sr2_spectre.mcp.client import MCPClient, MCPConnectionError

    failing_ctx = AsyncMock()
    failing_ctx.__aenter__ = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    failing_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("sr2_spectre.mcp.client.sse_client", return_value=failing_ctx):
        client = MCPClient("http", url="http://dead-host/sse")
        with pytest.raises(MCPConnectionError):
            await client.connect()


# ---------------------------------------------------------------------------
# Requirement 8: MCPConnectionError on initialize() failure
# ---------------------------------------------------------------------------

async def test_connect_raises_mcp_connection_error_on_initialize_failure() -> None:
    from sr2_spectre.mcp.client import MCPClient, MCPConnectionError

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock(side_effect=RuntimeError("protocol error"))
    mock_session.list_tools = AsyncMock()

    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        client = MCPClient("stdio", command=["srv"])
        with pytest.raises(MCPConnectionError):
            await client.connect()


# ---------------------------------------------------------------------------
# Requirement 9: MCPConnectionError on list_tools() failure
# ---------------------------------------------------------------------------

async def test_connect_raises_mcp_connection_error_on_list_tools_failure() -> None:
    from sr2_spectre.mcp.client import MCPClient, MCPConnectionError

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock(return_value=None)
    mock_session.list_tools = AsyncMock(side_effect=RuntimeError("server dropped"))

    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        client = MCPClient("stdio", command=["srv"])
        with pytest.raises(MCPConnectionError):
            await client.connect()


# ---------------------------------------------------------------------------
# Requirement 10: close() before connect() is safe (no-op)
# ---------------------------------------------------------------------------

async def test_close_before_connect_does_not_raise() -> None:
    from sr2_spectre.mcp.client import MCPClient

    client = MCPClient("stdio", command=["srv"])
    await client.close()  # must not raise


# ---------------------------------------------------------------------------
# Requirement 11: close() after connect() exits the session context
# ---------------------------------------------------------------------------

async def test_close_after_connect_exits_session_context() -> None:
    mock_session = _make_mock_session([_make_mcp_tool()])
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        await client.connect()
        await client.close()

    # The session context manager's __aexit__ must have been called
    session_ctx.__aexit__.assert_awaited()


# ---------------------------------------------------------------------------
# Requirement 12: close() suppresses RuntimeError (cancel scope mismatch)
# ---------------------------------------------------------------------------

async def test_close_suppresses_runtime_error_on_session_aexit() -> None:
    """close() must NOT propagate RuntimeError from session __aexit__.

    This is the cancel scope mismatch that anyio raises when the session
    is closed in a different task than it was entered — the classic
    'Attempted to exit cancel scope in a different task than it was entered in'.
    """
    mock_session = _make_mock_session([_make_mcp_tool()])
    transport_ctx = _make_transport_ctx()

    # Session __aexit__ raises RuntimeError (simulating cancel scope mismatch)
    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(
        side_effect=RuntimeError(
            "Attempted to exit cancel scope in a different task than it was entered in"
        )
    )

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        await client.connect()
        await client.close()  # must not raise

    session_ctx.__aexit__.assert_awaited()


async def test_close_suppresses_runtime_error_on_transport_aexit() -> None:
    """close() must NOT propagate RuntimeError from transport __aexit__."""
    mock_session = _make_mock_session([_make_mcp_tool()])

    # Transport __aexit__ raises RuntimeError
    transport_ctx = AsyncMock()
    transport_ctx.__aenter__ = AsyncMock(
        return_value=(AsyncMock(), AsyncMock())
    )
    transport_ctx.__aexit__ = AsyncMock(
        side_effect=RuntimeError("cancel scope error")
    )

    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        await client.connect()
        await client.close()  # must not raise

    transport_ctx.__aexit__.assert_awaited()


# ---------------------------------------------------------------------------
# Requirement 13: close() suppresses GeneratorExit
# ---------------------------------------------------------------------------

async def test_close_suppresses_generator_exit_on_session_aexit() -> None:
    """close() must NOT propagate GeneratorExit from session __aexit__."""
    mock_session = _make_mock_session([_make_mcp_tool()])
    transport_ctx = _make_transport_ctx()

    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(side_effect=GeneratorExit())

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        await client.connect()
        await client.close()  # must not raise

    session_ctx.__aexit__.assert_awaited()


# ---------------------------------------------------------------------------
# Requirement 14: close() clears internal state after teardown
# ---------------------------------------------------------------------------

async def test_close_clears_internal_state() -> None:
    """After close(), _session_ctx and _transport_ctx must be None."""
    mock_session = _make_mock_session([_make_mcp_tool()])
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        await client.connect()
        assert client._session_ctx is not None
        assert client._transport_ctx is not None

        await client.close()

        assert client._session_ctx is None
        assert client._transport_ctx is None


# ---------------------------------------------------------------------------
# Requirement 15: close() is idempotent — safe to call multiple times
# ---------------------------------------------------------------------------

async def test_close_is_idempotent() -> None:
    """Calling close() multiple times must not raise."""
    mock_session = _make_mock_session([_make_mcp_tool()])
    transport_ctx = _make_transport_ctx()
    session_ctx = _make_session_ctx(mock_session)

    with (
        patch("sr2_spectre.mcp.client.stdio_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    ):
        from sr2_spectre.mcp.client import MCPClient

        client = MCPClient("stdio", command=["srv"])
        await client.connect()
        await client.close()
        await client.close()  # second call must not raise
