"""Shared fixtures for TUI testing with Textual Pilot harness.

All TUI feature tests (spc-55 through spc-61) import from here.
Tests drive the app via Pilot, assert on widget content — never stdout.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Callable
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import (
    AgentDone,
    AgentEvent,
    AgentTextDelta,
    AgentThinkingDelta,
    AgentToolResult,
    AgentToolStart,
)
from sr2_spectre.interfaces.tui import SpectreTUI, TUIInterface


def _make_mock_agent(
    session_id: str = "test-session",
    history: list | None = None,
    tools: list[str] | None = None,
    stream_events: list[AgentEvent] | None = None,
) -> MagicMock:
    """Return a minimal mock agent for TUI tests.

    The mock satisfies the Agent API surface the TUI needs:
    - session_id, history, registry.list_names
    - set_run_context / run_context
    - stream_message (yields *stream_events* or empty by default)

    Args:
        stream_events: Events to yield from stream_message(). If None,
            yields an empty stream.
    """
    agent = MagicMock()
    agent.session_id = session_id
    agent.history = history or []
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=tools or ["tool_a"])

    # Return a real RunContext so we can inspect it
    rc = RunContext(interface="tui", mode=RunMode.INTERACTIVE, source="/tmp")
    type(agent).run_context = PropertyMock(return_value=rc)

    events = list(stream_events) if stream_events else []

    # Build a real async iterator factory so `async for` works,
    # wrapped in a MagicMock so we can assert calls.
    call_log: list[str] = []

    async def _stream(_text: str) -> AsyncIterator[AgentEvent]:
        for ev in events:
            yield ev

    def _stream_factory(text: str):
        call_log.append(text)
        return _stream(text)

    agent.stream_message = MagicMock(side_effect=_stream_factory)
    # Attach call_log so tests can inspect what was dispatched
    agent._stream_call_log = call_log
    return agent


@pytest.fixture
def mock_agent() -> MagicMock:
    """Minimal mock agent for TUI tests (empty stream)."""
    return _make_mock_agent()


@pytest.fixture
def tui_app(mock_agent: MagicMock) -> SpectreTUI:
    """SpectreTUI app instance wired to a mock agent."""
    return SpectreTUI(mock_agent)


@pytest.fixture
def tui_interface() -> TUIInterface:
    """TUIInterface instance for lifecycle tests."""
    return TUIInterface()


# ---------------------------------------------------------------------------
# Factory fixtures — downstream tasks use these to configure event streams
# ---------------------------------------------------------------------------

@pytest.fixture
def make_mock_agent() -> Callable[..., MagicMock]:
    """Factory to create mock agents with configurable event streams.

    Usage:
        agent = make_mock_agent(stream_events=[
            AgentTextDelta(text="Hello "),
            AgentTextDelta(text="world"),
            AgentDone(tool_calls_executed=0),
        ])
    """
    return _make_mock_agent


# ---------------------------------------------------------------------------
# Helper utilities for test authors
# ---------------------------------------------------------------------------

async def submit_input(pilot, app: SpectreTUI, text: str) -> None:
    """Type text into the prompt input and submit."""
    from textual.widgets import Input
    inp = app.query_one("#prompt", Input)
    inp.value = text
    await inp.run_action("submit")
    await pilot.pause()


def get_log_lines(app: SpectreTUI) -> list[str]:
    """Return the RichLog lines as a list of strings."""
    from textual.widgets import RichLog
    log = app.query_one("#output", RichLog)
    return [str(line.text) for line in log.lines]


def assert_log_contains(app: SpectreTUI, substring: str) -> None:
    """Assert the RichLog contains a line with the given substring."""
    lines = get_log_lines(app)
    assert any(substring in line for line in lines), (
        f"Log does not contain '{substring}'. Lines: {lines}"
    )
