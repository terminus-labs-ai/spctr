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
async def test_stream_turn_live_region_wipes_and_streams(make_mock_agent) -> None:
    """Regression guard for the double-render + no-streaming bug (spc-76).

    The Live region must be created with transient=True (so the plain-text
    streaming frame erases itself on exit — otherwise the final reply is
    committed to scrollback twice) and auto_refresh must NOT be forced off
    (auto_refresh=False is what silenced Rich's in-place redraws, making
    deltas appear as one blob instead of streaming).
    """
    import rich.live
    from sr2_spectre.events import AgentDone, AgentTextDelta

    captured: dict = {}
    real_live = rich.live.Live

    class _SpyLive(real_live):
        def __init__(self, renderable, console=None, **kwargs):
            captured["kwargs"] = kwargs
            super().__init__(renderable, console=console, **kwargs)

    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="Hello "),
        AgentTextDelta(text="world"),
        AgentDone(tool_calls_executed=0),
    ])
    iface, sink = _make_repl_with_sink()

    orig = rich.live.Live
    rich.live.Live = _SpyLive
    try:
        await iface._stream_turn(agent, "hi")
    finally:
        rich.live.Live = orig

    assert "kwargs" in captured, "Live was not created by _stream_turn"
    kwargs = captured["kwargs"]
    # transient=True => live frame erased on exit => single committed copy
    assert kwargs.get("transient") is True, (
        f"Live must be transient to avoid double-render; got {kwargs}"
    )
    # auto_refresh must not be disabled (that kills live streaming)
    assert kwargs.get("auto_refresh", True) is not False, (
        f"auto_refresh=False silences streaming; got {kwargs}"
    )


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
async def test_stream_turn_streams_thinking_in_the_live_frame(make_mock_agent) -> None:
    """Thinking is scratch: it streams in the transient frame, never committed.

    Asserting on the console sink would not distinguish the two — the sink
    records the live frames as well, because it cannot replay their erasure.
    So this inspects what _stream_turn actually hands to Live.update().
    """
    import rich.live
    from sr2_spectre.events import AgentDone, AgentThinkingDelta, AgentTextDelta

    frames: list[str] = []
    real_live = rich.live.Live

    class _SpyLive(real_live):
        def update(self, renderable, **kwargs):
            frames.append(getattr(renderable, "plain", str(renderable)))
            return super().update(renderable, **kwargs)

    agent = make_mock_agent(stream_events=[
        AgentThinkingDelta(text="let me think"),
        AgentTextDelta(text="answer"),
        AgentDone(tool_calls_executed=0),
    ])
    iface, sink = _make_repl_with_sink()

    rich.live.Live = _SpyLive
    try:
        await iface._stream_turn(agent, "hi")
    finally:
        rich.live.Live = real_live

    assert any("let me think" in f for f in frames), frames
    # The reply itself is committed to scrollback; the thinking is not.
    assert "answer" in _output(sink)


@pytest.mark.asyncio
async def test_stream_turn_commits_each_round_once(make_mock_agent, monkeypatch) -> None:
    """A multi-roundtrip turn commits each round's text exactly once, in order.

    A turn can span several LLM roundtrips: narration ("let me check that"),
    a tool call, then the real answer.  Every round is committed as Markdown
    at the moment it ends, so nothing the model said is dropped and nothing
    is printed twice.  The transient Live frame holds only the round that is
    still streaming.

    Note this is a *rendering* test, not a scrollback test — the in-memory
    console performs no erasure, so it cannot prove the live frame was wiped.
    tests/test_repl_pty.py covers that against a real terminal.
    """
    import sr2_spectre.interfaces.repl as repl_mod
    from sr2_spectre.events import (
        AgentDone,
        AgentToolResult,
        AgentToolStart,
        AgentTextDelta,
    )

    committed: list[str] = []
    real_render = repl_mod.render_markdown

    def _spy_render(text, console=None):
        committed.append(text)
        return real_render(text, console)

    monkeypatch.setattr(repl_mod, "render_markdown", _spy_render)

    agent = make_mock_agent(stream_events=[
        # round 1 — narration leading up to the tool call
        AgentTextDelta(text="First, let me "),
        AgentTextDelta(text="check that."),
        AgentToolStart(name="terminal", input={"command": "ls"}),
        AgentToolResult(name="terminal", is_error=False),
        # round 2 — the real answer
        AgentTextDelta(text="The answer is "),
        AgentTextDelta(text="**42**."),
        AgentDone(tool_calls_executed=1),
    ])
    iface, _sink = _make_repl_with_sink()

    await iface._stream_turn(agent, "what is the answer?")

    assert committed == ["First, let me check that.", "The answer is **42**."], committed


@pytest.mark.asyncio
async def test_stream_turn_commits_nothing_twice_within_a_round(make_mock_agent, monkeypatch) -> None:
    """Text committed when a round ends is not carried into the next commit."""
    import sr2_spectre.interfaces.repl as repl_mod
    from sr2_spectre.events import (
        AgentDone,
        AgentToolResult,
        AgentToolStart,
        AgentTextDelta,
    )

    committed: list[str] = []
    real_render = repl_mod.render_markdown
    monkeypatch.setattr(
        repl_mod,
        "render_markdown",
        lambda text, console=None: (committed.append(text), real_render(text, console))[1],
    )

    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="round one. "),
        AgentToolStart(name="a", input={}),
        AgentToolResult(name="a", is_error=False),
        AgentTextDelta(text="round two. "),
        AgentToolStart(name="b", input={}),
        AgentToolResult(name="b", is_error=False),
        AgentTextDelta(text="round three."),
        AgentDone(tool_calls_executed=2),
    ])
    iface, _sink = _make_repl_with_sink()

    await iface._stream_turn(agent, "go")

    assert committed == ["round one. ", "round two. ", "round three."], committed


@pytest.mark.asyncio
async def test_stream_turn_error_path_preserves_partial_text(make_mock_agent) -> None:
    """On a stream error the whole accumulated buffer is still committed (not
    just the final round, which never completed) so the user sees what streamed."""
    from sr2_spectre.events import (
        AgentTextDelta,
        AgentToolResult,
        AgentToolStart,
    )

    agent = make_mock_agent()

    async def _erroring_stream(text: str):
        # interim round text, then a tool result, then a crash mid-final-round
        yield AgentTextDelta(text="Interim scratch. ")
        yield AgentToolStart(name="terminal", input={"command": "ls"})
        yield AgentToolResult(name="terminal", is_error=False)
        yield AgentTextDelta(text="Partial final. ")
        raise RuntimeError("boom")

    agent.stream_message = MagicMock(side_effect=lambda text: _erroring_stream(text))
    iface, sink = _make_repl_with_sink()

    await iface._stream_turn(agent, "hi")

    out = _output(sink)
    assert "boom" in out
    # Both interim and partial-final are preserved on the error path.
    assert "Interim scratch." in out
    assert "Partial final." in out


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
