"""Cage tests for obsidian-c573: MCP connect() cancellation handling.

Boot-race regression: an MCP streamable-http transport's anyio cancel scope
raised asyncio.CancelledError out of MCPClient.connect(). connect() only
caught Exception, so the error escaped to Runtime.initialize() (which never
logged 'MCP server failed to connect') and up through cli.main(), where
interpreter shutdown then hung on a leftover non-daemon thread.

Contract under test (bead obsidian-c573):
  AC1 — a spurious cancellation (the current task is NOT being cancelled, the
        same task.cancelling() test as MCPClient._suppress_spurious_cancel)
        escaping the transport/session/initialize sequence makes connect()
        raise MCPConnectionError and close the half-opened session/transport
        contexts; Runtime.initialize() logs a warning and continues with the
        remaining servers.
  AC2 — a genuine external cancellation of connect() still propagates.

All tests drive public APIs (MCPClient.connect, Runtime.initialize) with the
transport factory and ClientSession patched at the client module import level
— the convention established in tests/test_mcp_client.py. No real servers, no
bound ports.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.mcp.client import MCPConnectionError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _make_mcp_tool(name: str = "my_tool") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description="Does a thing",
        inputSchema={"type": "object", "properties": {}},
    )


def _read_write_get_session_id() -> tuple:
    """streamablehttp_client yields a 3-tuple (read, write, get_session_id)."""
    return (AsyncMock(), AsyncMock(), MagicMock(return_value="sid-123"))


class _RecordingCtx:
    """Async context manager counting successful entries and exits.

    ``enter_exc`` / ``exit_exc`` raise from the respective hook instead.
    ``exited`` counts every __aexit__ call that ran, including one whose body
    raised — cleanup must still reach the outer context afterwards.
    """

    def __init__(
        self,
        value: Any = None,
        enter_exc: BaseException | None = None,
        exit_exc: BaseException | None = None,
    ) -> None:
        self.value = value
        self.enter_exc = enter_exc
        self.exit_exc = exit_exc
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> Any:
        if self.enter_exc is not None:
            raise self.enter_exc
        self.entered += 1
        return self.value

    async def __aexit__(self, *exc_info: Any) -> None:
        self.exited += 1
        if self.exit_exc is not None:
            raise self.exit_exc
        return None


def _make_mock_session(
    tools: list | None = None,
    initialize_exc: BaseException | None = None,
) -> AsyncMock:
    session = AsyncMock()
    if initialize_exc is not None:
        session.initialize = AsyncMock(side_effect=initialize_exc)
    else:
        session.initialize = AsyncMock(return_value=None)
    session.list_tools = AsyncMock(
        return_value=SimpleNamespace(tools=tools if tools is not None else [])
    )
    session.call_tool = AsyncMock(return_value=SimpleNamespace(content=[]))
    return session


def _patch_boundaries(transport_ctx: Any, session_ctx: Any) -> tuple:
    """Patch the transport factory and ClientSession at the client module level."""
    return (
        patch("sr2_spectre.mcp.client.streamablehttp_client", return_value=transport_ctx),
        patch("sr2_spectre.mcp.client.ClientSession", return_value=session_ctx),
    )


def _make_client():
    from sr2_spectre.mcp.client import MCPClient

    return MCPClient("streamable-http", url="http://127.0.0.1:1/mcp")


# ---------------------------------------------------------------------------
# AC1 — spurious CancelledError from initialize() becomes MCPConnectionError
# ---------------------------------------------------------------------------

async def test_connect_spurious_cancel_at_initialize_raises_connection_error() -> None:
    """A spurious CancelledError (task.cancelling() == 0) escaping
    session.initialize() — the boot-race shape from the bead — must surface as
    MCPConnectionError, not CancelledError."""
    transport_ctx = _RecordingCtx(value=_read_write_get_session_id())
    session_ctx = _RecordingCtx(
        value=_make_mock_session(
            initialize_exc=asyncio.CancelledError("spurious anyio cancel scope")
        )
    )

    p_transport, p_session = _patch_boundaries(transport_ctx, session_ctx)
    with p_transport, p_session:
        client = _make_client()
        with pytest.raises(MCPConnectionError):
            await client.connect()

    # A bare `except Exception` does not catch CancelledError (a BaseException),
    # so this is red until connect() handles the cancellation explicitly.


async def test_connect_spurious_cancel_preserves_cause() -> None:
    """The converted MCPConnectionError carries the original cancellation as
    __cause__, matching the existing `raise MCPConnectionError(...) from exc`
    idiom in connect(), so the warning log and post-mortems show the real
    failure."""
    original = asyncio.CancelledError("spurious anyio cancel scope")
    transport_ctx = _RecordingCtx(value=_read_write_get_session_id())
    session_ctx = _RecordingCtx(value=_make_mock_session(initialize_exc=original))

    p_transport, p_session = _patch_boundaries(transport_ctx, session_ctx)
    with p_transport, p_session:
        client = _make_client()
        with pytest.raises(MCPConnectionError) as excinfo:
            await client.connect()

    assert excinfo.value.__cause__ is original


async def test_connect_spurious_cancel_closes_half_open_contexts() -> None:
    """On the spurious cancellation the already-entered session context and
    transport context must both be closed, and the client must not be left
    holding them — otherwise the transport's non-daemon reader threads are
    stranded (the 8-hour hang) and a later close() double-exits."""
    transport_ctx = _RecordingCtx(value=_read_write_get_session_id())
    session_ctx = _RecordingCtx(
        value=_make_mock_session(initialize_exc=asyncio.CancelledError("spurious"))
    )

    p_transport, p_session = _patch_boundaries(transport_ctx, session_ctx)
    with p_transport, p_session:
        client = _make_client()
        with pytest.raises(MCPConnectionError):
            await client.connect()

        assert session_ctx.exited == 1
        assert transport_ctx.exited == 1

        # Following a failed connect() with close() must stay safe and must not
        # exit either context a second time.
        await client.close()

    assert session_ctx.exited == 1
    assert transport_ctx.exited == 1


async def test_connect_spurious_cancel_survives_cancel_from_session_cleanup() -> None:
    """The half-open teardown may itself leak a spurious CancelledError out of
    the session context's __aexit__ — the shape close() already suppresses.
    connect() must still raise MCPConnectionError and must still close the
    transport context."""
    transport_ctx = _RecordingCtx(value=_read_write_get_session_id())
    session_ctx = _RecordingCtx(
        value=_make_mock_session(initialize_exc=asyncio.CancelledError("spurious")),
        exit_exc=asyncio.CancelledError("spurious leak from session aexit"),
    )

    p_transport, p_session = _patch_boundaries(transport_ctx, session_ctx)
    with p_transport, p_session:
        client = _make_client()
        with pytest.raises(MCPConnectionError):
            await client.connect()

        assert session_ctx.exited == 1
        assert transport_ctx.exited == 1
        await client.close()

    assert transport_ctx.exited == 1


async def test_connect_spurious_cancel_at_transport_enter_raises_connection_error() -> None:
    """The cancellation can also escape the transport __aenter__ itself (the
    endpoint is down mid-handshake during the boot race). That too must become
    MCPConnectionError, and no session may be created."""
    transport_ctx = _RecordingCtx(enter_exc=asyncio.CancelledError("spurious at enter"))
    session_ctx = _RecordingCtx(value=_make_mock_session())

    p_transport, p_session = _patch_boundaries(transport_ctx, session_ctx)
    with p_transport, p_session:
        client = _make_client()
        with pytest.raises(MCPConnectionError):
            await client.connect()

    assert session_ctx.entered == 0
    # A failed connect() must leave close() safe to call.
    await client.close()


# ---------------------------------------------------------------------------
# AC2 — a genuine external cancellation still propagates
# ---------------------------------------------------------------------------

async def test_connect_genuine_task_cancellation_propagates() -> None:
    """When the awaiting task IS genuinely being cancelled, connect() must let
    CancelledError propagate instead of converting it to MCPConnectionError.

    The cancellation is delivered for real (an outer task cancels the task
    awaiting connect()), so no asyncio internals are monkeypatched. The
    cancellation lands on a blocked await inside the session context's
    __aenter__, standing in for the transport's in-flight HTTP request when its
    cancel scope fires.
    """
    release = asyncio.Event()

    class _BlockedSessionCtx:
        def __init__(self) -> None:
            self.entered = 0
            self.exited = 0

        async def __aenter__(self) -> Any:
            self.entered += 1
            await release.wait()
            return _make_mock_session()

        async def __aexit__(self, *exc_info: Any) -> None:
            self.exited += 1
            return None

    transport_ctx = _RecordingCtx(value=_read_write_get_session_id())
    session_ctx = _BlockedSessionCtx()

    p_transport, p_session = _patch_boundaries(transport_ctx, session_ctx)

    async def _drive() -> tuple[str, BaseException | None]:
        with p_transport, p_session:
            client = _make_client()
            inner = asyncio.create_task(client.connect())
            await asyncio.sleep(0.05)  # let connect() reach the blocked await
            inner.cancel()
            try:
                async with asyncio.timeout(5):
                    await inner
            except asyncio.CancelledError:
                return "cancelled", None
            except BaseException as exc:  # noqa: BLE001 — surfaced as failure
                return "other", exc
            return "completed", None

    outcome, exc = await _drive()

    assert outcome == "cancelled", (
        "a genuine cancellation of connect() must propagate as CancelledError; "
        f"got {outcome}: {exc!r}"
    )


async def test_connect_genuine_cancel_via_cancelling_marker_propagates() -> None:
    """The same contract at the marker the implementation keys on: while
    asyncio.current_task().cancelling() > 0, a CancelledError escaping
    initialize() must propagate unchanged."""
    from sr2_spectre.mcp.client import MCPClient

    fake_task = MagicMock()
    fake_task.cancelling.return_value = 1

    transport_ctx = _RecordingCtx(value=_read_write_get_session_id())
    session_ctx = _RecordingCtx(
        value=_make_mock_session(initialize_exc=asyncio.CancelledError("genuine"))
    )

    p_transport, p_session = _patch_boundaries(transport_ctx, session_ctx)
    with (
        p_transport,
        p_session,
        patch("sr2_spectre.mcp.client.asyncio.current_task", return_value=fake_task),
    ):
        client = MCPClient("streamable-http", url="http://127.0.0.1:1/mcp")
        with pytest.raises(asyncio.CancelledError):
            await client.connect()


# ---------------------------------------------------------------------------
# Guard against over-conversion: the plain happy path still works
# ---------------------------------------------------------------------------

async def test_connect_without_cancellation_still_returns_bridges() -> None:
    """With no cancellation anywhere, connect() must still succeed and return
    one bridge per tool."""
    from sr2_spectre.mcp.tool_bridge import MCPToolBridge

    transport_ctx = _RecordingCtx(value=_read_write_get_session_id())
    session_ctx = _RecordingCtx(
        value=_make_mock_session(tools=[_make_mcp_tool("alpha"), _make_mcp_tool("beta")])
    )

    p_transport, p_session = _patch_boundaries(transport_ctx, session_ctx)
    with p_transport, p_session:
        client = _make_client()
        bridges = await client.connect()

    assert len(bridges) == 2
    assert all(isinstance(b, MCPToolBridge) for b in bridges)


# ---------------------------------------------------------------------------
# AC1 (Runtime half) — initialize() warns and continues with remaining servers
# ---------------------------------------------------------------------------

def _minimal_pipeline_dict() -> dict:
    return {
        "layers": [
            {
                "name": "system",
                "target": "system",
                "resolvers": [
                    {"type": "static", "config": {"text": "You are helpful."}}
                ],
            },
            {
                "name": "tools",
                "target": "tools",
                "resolvers": [],
                "tool_providers": [{"type": "spectre_tools"}],
            },
            {
                "name": "conversation",
                "target": "messages",
                "resolvers": [{"type": "session"}, {"type": "input"}],
            },
        ]
    }


def _make_runtime_config(**agent_kwargs: Any) -> Any:
    from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig

    return SpectreConfig(
        agent=AgentConfig(name="test", **agent_kwargs),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=_minimal_pipeline_dict(),
    )


def _stdio_server(name: str) -> Any:
    from sr2_spectre.config import McpServerConfig

    return McpServerConfig(name=name, type="stdio", command=[f"server_{name}"])


def _bridge(name: str) -> MagicMock:
    b = MagicMock()
    b.name = name
    b.description = "test"
    b.input_schema = {}
    return b


def _client_named(name: str, connect: Any) -> MagicMock:
    client = MagicMock(name=f"mcp_{name}")
    client.connect = connect
    client.close = AsyncMock(return_value=None)
    return client


def _runtime_with_clients(mcp_servers: list[Any], clients: list[MagicMock]) -> Any:
    """Build a Runtime whose per-server MCPClient constructions yield the mocks.

    Uses the same wiring seam as tests/test_runtime.py (patching
    runtime.MCPClient); clients are handed out in config order.
    """
    from sr2_spectre.runtime import Runtime

    cfg = _make_runtime_config(mcp_servers=mcp_servers)
    with (
        patch("sr2_spectre.live_llm.LiteLLMCallable"),
        patch("sr2_spectre.runtime.MCPClient", side_effect=list(clients)),
    ):
        runtime = Runtime(config=cfg)
    return runtime


async def test_runtime_initialize_logs_warning_on_connection_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Runtime.initialize() must catch MCPConnectionError and log the
    'MCP server failed to connect' warning, not let it escape."""
    failing = _client_named(
        "down",
        AsyncMock(side_effect=MCPConnectionError("spurious anyio cancel scope")),
    )
    runtime = _runtime_with_clients([_stdio_server("down")], [failing])

    with caplog.at_level(logging.WARNING, logger="sr2_spectre.runtime"):
        await runtime.initialize()  # must not raise

    assert any(
        "MCP server failed to connect" in rec.getMessage() and rec.levelno == logging.WARNING
        for rec in caplog.records
    ), f"expected warning, got: {[r.getMessage() for r in caplog.records]}"


async def test_runtime_initialize_continues_with_remaining_servers() -> None:
    """A server that fails to connect must not stop the ones after it from
    connecting and registering."""
    failing = _client_named(
        "down",
        AsyncMock(side_effect=MCPConnectionError("endpoint refused / cancelled")),
    )
    healthy = _client_named("up", AsyncMock(return_value=[_bridge("good_tool")]))
    runtime = _runtime_with_clients(
        [_stdio_server("down"), _stdio_server("up")], [failing, healthy]
    )

    await runtime.initialize()

    healthy.connect.assert_awaited_once()
    assert "good_tool" in runtime.registry


async def test_runtime_initialize_escapes_unexpected_cancellation() -> None:
    """A raw CancelledError escaping connect() must still abort initialize():
    only MCPConnectionError is tolerated. Pre-fix this is exactly what escaped
    all the way out of startup; post-fix connect() converts the spurious case,
    so a bare cancellation reaching initialize() is genuine and must propagate.
    """
    client = _client_named("down", AsyncMock(side_effect=asyncio.CancelledError()))
    runtime = _runtime_with_clients([_stdio_server("down")], [client])

    with pytest.raises(asyncio.CancelledError):
        await runtime.initialize()
