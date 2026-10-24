"""Cage tests for spc-5: boot-race tolerance for MCP connects.

Boot incident (spc-5): after a reboot the glyph docker container became
reachable ~35s after the agents started. Every MCP connect failed with
httpx.ReadError and the agents ran on without their glyph tools until a
manual restart. obsidian-c573 made connect() failures survivable; this round
makes them *recoverable*: Runtime.initialize() retries a failed connect with
bounded exponential backoff, so a server that comes up inside the window is
connected and its tools registered.

Also closes the symmetric cleanup gap in MCPClient.connect(): the generic
Exception path must close half-open session/transport contexts exactly like
the spurious-cancellation path does, so the retry loop (and the continue-on-
warning path) never strands the contexts of a failed attempt.

Contract under test (bead spc-5):
  AC1 — a connect that fails with MCPConnectionError and then succeeds inside
        the retry window ends with the client connected once more than it
        failed and its bridges registered.
  AC2 — a permanently failing connect is eventually tolerated: bounded
        attempts, the 'MCP server failed to connect' warning, no raise,
        remaining servers still connect.
  AC3 — a genuine external cancellation is not retried and propagates.
  AC4 — a generic Exception mid-connect closes the half-open contexts.

All tests drive public APIs (Runtime.initialize, MCPClient.connect) with the
same seams as tests/test_mcp_connect_cancellation.py. No real servers, no
bound ports; backoff constants are patched to keep the suite fast.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from sr2_spectre.mcp.client import MCPConnectionError

from .test_mcp_connect_cancellation import (
    _RecordingCtx,
    _bridge,
    _client_named,
    _make_client,
    _patch_boundaries,
    _runtime_with_clients,
    _stdio_server,
)


# ---------------------------------------------------------------------------
# AC1/AC2/AC3 — Runtime.initialize() retry policy
# ---------------------------------------------------------------------------

_FAST = {"connect_retry_backoff_seconds": 0.01}


async def test_initialize_retries_until_server_recovers() -> None:
    """A server that fails twice and answers on the third attempt must end up
    connected with its tools registered (glyph-boot-race shape)."""
    ok_bridge = _bridge("glyph_search")
    connect = AsyncMock(
        side_effect=[MCPConnectionError("connection refused"), MCPConnectionError("refused"), [ok_bridge]]
    )
    client = _client_named("glyph", connect)
    runtime = _runtime_with_clients(
        [_stdio_server("glyph", connect_retries=3, **_FAST)], [client]
    )

    await runtime.initialize()

    assert connect.await_count == 3
    assert "glyph_search" in runtime.registry


async def test_initialize_gives_up_bounded_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A permanently-down server: exactly 1 + connect_retries attempts, the
    existing warning still logged, no raise, and the next server still
    connects."""
    down = _client_named(
        "down", AsyncMock(side_effect=MCPConnectionError("still down"))
    )
    up = _client_named("up", AsyncMock(return_value=[_bridge("good_tool")]))
    runtime = _runtime_with_clients(
        [_stdio_server("down", connect_retries=3, **_FAST), _stdio_server("up")],
        [down, up],
    )

    with caplog.at_level(logging.WARNING, logger="sr2_spectre.runtime"):
        await runtime.initialize()  # must not raise

    assert down.connect.await_count == 4
    assert any(
        "MCP server failed to connect" in rec.getMessage()
        and rec.levelno == logging.WARNING
        for rec in caplog.records
    ), f"expected final warning, got: {[r.getMessage() for r in caplog.records]}"
    assert "good_tool" in runtime.registry


async def test_initialize_does_not_retry_genuine_cancellation() -> None:
    """A raw CancelledError reaching initialize() is genuine external
    cancellation: it must propagate immediately, untired."""
    client = _client_named("down", AsyncMock(side_effect=asyncio.CancelledError()))
    runtime = _runtime_with_clients(
        [_stdio_server("down", connect_retries=3, **_FAST)], [client]
    )

    with pytest.raises(asyncio.CancelledError):
        await runtime.initialize()

    assert client.connect.await_count == 1


# ---------------------------------------------------------------------------
# AC4 — MCPClient.connect() closes half-open contexts on generic failures
# ---------------------------------------------------------------------------


async def test_connect_generic_exception_closes_half_open_contexts() -> None:
    """An httpx.ReadError (the actual boot-race failure) raised after the
    transport was entered must close the entered contexts before raising
    MCPConnectionError — the same cleanup the spurious-cancel path does — so
    neither the retry loop nor the continue-on-warning path strands them."""
    transport_ctx = _RecordingCtx(value=(MagicMock(), MagicMock(), MagicMock()))
    session_ctx = _RecordingCtx(
        enter_exc=RuntimeError("boom")  # generic Exception from session __aenter__
    )

    p_transport, p_session = _patch_boundaries(transport_ctx, session_ctx)
    with p_transport, p_session:
        client = _make_client()
        with pytest.raises(MCPConnectionError):
            await client.connect()

    assert transport_ctx.entered == 1
    assert transport_ctx.exited == 1, (
        "the transport entered before the failure must be exited; "
        "half-open transports strand threads across retries"
    )
