"""Tests for REPL streaming rendering (mocked agent event stream).

Verifies that _stream_turn consumes the event stream and renders:
- text deltas accumulated into a live region, committed as markdown at end
- thinking deltas rendered dim/italic alongside text
- tool start/result status lines
- final status line with session/message/tool counts
- error path: stream exception reported, partial text still committed
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class _LineSink:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, s: str) -> int:
        self.chunks.append(s)
        return len(s)

    def flush(self) -> None:
        pass


def _make_repl_with_sink():
    """Build a REPLInterface with an in-memory Rich console; returns (iface, sink)."""
    from rich.console import Console
    from sr2_spectre.interfaces.repl import REPLInterface

    sink = _LineSink()
    iface = REPLInterface(console=Console(file=sink, force_terminal=True, width=100))
    return iface, sink


def _output(sink: _LineSink) -> str:
    return _strip_ansi("".join(sink.chunks))


@pytest.mark.asyncio
async def test_stream_turn_renders_text(make_mock_agent) -> None:
    from sr2_spectre.events import AgentDone, AgentTextDelta

    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="Hello "),
        AgentTextDelta(text="**world**"),
        AgentDone(tool_calls_executed=0),
    ])
    iface, sink = _make_repl_with_sink()

    await iface._stream_turn(agent, "hi")

    out = _output(sink)
    # Both deltas accumulated and committed (markdown-rendered: bold markers gone, text kept)
    assert "Hello" in out
    assert "world" in out
    # Status line at end of turn
    assert "session test-session" in out or "test-ses" in out


@pytest.mark.asyncio
async def test_stream_turn_renders_thinking(make_mock_agent) -> None:
    from sr2_spectre.events import AgentDone, AgentThinkingDelta, AgentTextDelta

    agent = make_mock_agent(stream_events=[
        AgentThinkingDelta(text="let me think"),
        AgentTextDelta(text="answer"),
        AgentDone(tool_calls_executed=0),
    ])
    iface, sink = _make_repl_with_sink()

    await iface._stream_turn(agent, "hi")

    out = _output(sink)
    assert "let me think" in out
    assert "answer" in out


@pytest.mark.asyncio
async def test_stream_turn_tool_start_and_result(make_mock_agent) -> None:
    from sr2_spectre.events import (
        AgentDone,
        AgentToolResult,
        AgentToolStart,
        AgentTextDelta,
    )

    agent = make_mock_agent(stream_events=[
        AgentToolStart(name="terminal", input={"command": "ls -la /some/path"}),
        AgentToolResult(name="terminal", is_error=False),
        AgentTextDelta(text="done"),
        AgentDone(tool_calls_executed=1),
    ])
    iface, sink = _make_repl_with_sink()

    await iface._stream_turn(agent, "ls")

    out = _output(sink)
    assert "⚙ terminal" in out
    # args preview truncated at 60 chars and shown
    assert "command" in out
    assert "✓ terminal done" in out
    # tool count reflected in status line
    assert "1 tools" in out


@pytest.mark.asyncio
async def test_stream_turn_tool_error(make_mock_agent) -> None:
    from sr2_spectre.events import (
        AgentDone,
        AgentToolResult,
        AgentToolStart,
        AgentTextDelta,
    )

    agent = make_mock_agent(stream_events=[
        AgentToolStart(name="grep", input={"pattern": "x"}),
        AgentToolResult(name="grep", is_error=True),
        AgentTextDelta(text="fallback"),
        AgentDone(tool_calls_executed=1),
    ])
    iface, sink = _make_repl_with_sink()

    await iface._stream_turn(agent, "find x")

    out = _output(sink)
    assert "✗ grep failed" in out


@pytest.mark.asyncio
async def test_stream_turn_error_path(make_mock_agent) -> None:
    """A stream exception is reported; partial text before it is still committed."""
    from sr2_spectre.events import AgentTextDelta

    agent = make_mock_agent()
    # Replace the factory with one that yields a delta then raises.
    async def _bad_stream(text: str):
        yield AgentTextDelta(text="partial ")
        raise RuntimeError("boom")

    agent.stream_message = MagicMock(side_effect=lambda text: _bad_stream(text))
    iface, sink = _make_repl_with_sink()

    await iface._stream_turn(agent, "hi")

    out = _output(sink)
    assert "Stream error: boom" in out or "Turn failed: boom" in out
    # Partial text still committed to output
    assert "partial" in out


@pytest.mark.asyncio
async def test_stream_turn_empty_response(make_mock_agent) -> None:
    from sr2_spectre.events import AgentDone

    agent = make_mock_agent(stream_events=[AgentDone(tool_calls_executed=0)])
    iface, sink = _make_repl_with_sink()

    await iface._stream_turn(agent, "hi")

    out = _output(sink)
    assert "(no response)" in out


@pytest.mark.asyncio
async def test_stream_turn_dispatches_text(make_mock_agent) -> None:
    """The user text is passed verbatim to agent.stream_message (quotes intact)."""
    from sr2_spectre.events import AgentDone

    agent = make_mock_agent(stream_events=[AgentDone(tool_calls_executed=0)])
    iface, sink = _make_repl_with_sink()

    quoted = 'say "hello" and \'world\''
    await iface._stream_turn(agent, quoted)

    # call_log records what was dispatched — quotes must survive untouched
    assert quoted in agent._stream_call_log


@pytest.mark.asyncio
async def test_run_loop_processes_commands_and_text(make_mock_agent, monkeypatch) -> None:
    """run() loops: dispatches a slash command, then a text turn, then /quit."""
    from sr2_spectre.events import AgentDone

    agent = make_mock_agent(stream_events=[AgentDone(tool_calls_executed=0)])
    iface, sink = _make_repl_with_sink()
    await iface.start(agent)

    # Feed the prompt loop scripted inputs via a fake session factory.
    class FakeSession:
        def __init__(self, it):
            self._it = it

        async def prompt_async(self):
            return next(self._it)

    def _fake_session(completer):
        return FakeSession(iter(["/tools", 'say "hi"', "/quit"]))

    monkeypatch.setattr(
        "sr2_spectre.interfaces.repl._make_prompt_session",
        _fake_session,
    )

    await iface.run(agent)

    out = _output(sink)
    assert "Available tools" in out          # /tools ran
    assert 'say "hi"' in agent._stream_call_log  # text turn dispatched with quotes
