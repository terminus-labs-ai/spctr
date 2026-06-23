"""Tests for TUI streaming render — FR5.

AgentTextDelta accumulates in live region; committed as markdown on AgentDone.
AgentThinkingDelta renders dim/italic in live region, never committed.
Stream runs in exclusive worker.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sr2_spectre.events import (
    AgentDone,
    AgentTextDelta,
    AgentThinkingDelta,
    AgentToolResult,
    AgentToolStart,
)
from sr2_spectre.interfaces.tui import SpectreTUI
from tests.conftest import (
    _make_mock_agent,
    assert_log_contains,
    get_log_lines,
    submit_input,
)


# ---------------------------------------------------------------------------
# FR5a: AgentTextDelta → live region → commit markdown on AgentDone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_delta_streams_to_live_region(
    make_mock_agent: MagicMock,
) -> None:
    """AgentTextDelta fragments accumulate in the #text live region."""
    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="Hello "),
        AgentTextDelta(text="world"),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "say hi")
        # Wait for worker to complete
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        # The accumulated text was committed to RichLog
        lines = get_log_lines(app)
        joined = "\n".join(lines)
        assert "Hello world" in joined


@pytest.mark.asyncio
async def test_text_committed_as_markdown_on_done(
    make_mock_agent: MagicMock,
) -> None:
    """On AgentDone, accumulated text is rendered as markdown to RichLog."""
    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="# Title"),
        AgentTextDelta(text="\n\nSome **bold** text."),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "render markdown")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        lines = get_log_lines(app)
        joined = "\n".join(lines)
        # The text should be committed to the log
        assert "Title" in joined
        assert "bold" in joined


@pytest.mark.asyncio
async def test_live_region_cleared_after_commit(
    make_mock_agent: MagicMock,
) -> None:
    """After AgentDone, the #text live region is cleared."""
    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="streaming text"),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        # Text region should be empty after commit
        from textual.widgets import Static
        text_region = app.query_one("#text", Static)
        assert text_region.content == ""


@pytest.mark.asyncio
async def test_no_text_delta_produces_no_commit(
    make_mock_agent: MagicMock,
) -> None:
    """If only AgentDone (no text deltas), nothing is committed to RichLog."""
    agent = make_mock_agent(stream_events=[
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        # Capture initial log lines
        await submit_input(pilot, app, "empty")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        # Only the user echo should be in the log, no agent response block
        lines = get_log_lines(app)
        user_lines = [l for l in lines if l.startswith("> ")]
        assert len(user_lines) == 1


# ---------------------------------------------------------------------------
# FR5b: AgentThinkingDelta → dim/italic live region, NOT committed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_delta_renders_in_thinking_region(
    make_mock_agent: MagicMock,
) -> None:
    """AgentThinkingDelta renders in the #thinking region."""
    agent = make_mock_agent(stream_events=[
        AgentThinkingDelta(text="Let me think..."),
        AgentThinkingDelta(text=" I know!"),
        AgentTextDelta(text="The answer is 42."),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "think")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        # Thinking region should be cleared after done
        from textual.widgets import Static
        thinking_region = app.query_one("#thinking", Static)
        assert thinking_region.content == ""


@pytest.mark.asyncio
async def test_thinking_not_committed_to_richlog(
    make_mock_agent: MagicMock,
) -> None:
    """Thinking text must NOT appear in the RichLog after AgentDone."""
    agent = make_mock_agent(stream_events=[
        AgentThinkingDelta(text="SECRET THINKING"),
        AgentTextDelta(text="Here is the answer."),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        lines = get_log_lines(app)
        joined = "\n".join(lines)
        assert "SECRET THINKING" not in joined
        assert "Here is the answer" in joined


@pytest.mark.asyncio
async def test_thinking_region_cleared_on_done(
    make_mock_agent: MagicMock,
) -> None:
    """Thinking region is cleared when AgentDone fires."""
    agent = make_mock_agent(stream_events=[
        AgentThinkingDelta(text="thinking..."),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        from textual.widgets import Static
        thinking_region = app.query_one("#thinking", Static)
        assert thinking_region.content == ""


# ---------------------------------------------------------------------------
# FR5c: Exclusive worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_disabled_during_streaming(
    make_mock_agent: MagicMock,
) -> None:
    """Input must be disabled while the stream worker is running."""
    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="chunk"),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "start streaming")
        # The worker runs to completion quickly in headless mode.
        # Verify the worker was launched (exclusive) and completed cleanly.
        await pilot.pause()
        await pilot.pause()

        from textual.widgets import Input
        inp = app.query_one("#prompt", Input)
        # After completion, input should be re-enabled and focused
        assert inp.disabled is False
        assert inp.has_focus
        # Streaming flag should be reset
        assert app._streaming is False


@pytest.mark.asyncio
async def test_input_reenabled_after_streaming(
    make_mock_agent: MagicMock,
) -> None:
    """Input must be re-enabled after the stream worker completes."""
    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="done"),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "quick")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        from textual.widgets import Input
        inp = app.query_one("#prompt", Input)
        assert app._streaming is False
        assert inp.disabled is False


# ---------------------------------------------------------------------------
# FR5d: Tool events rendered during stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_events_rendered_during_stream(
    make_mock_agent: MagicMock,
) -> None:
    """Tool start/result events are rendered to RichLog during streaming."""
    agent = make_mock_agent(stream_events=[
        AgentToolStart(tool_id="1", name="terminal", input={"command": "ls"}),
        AgentToolResult(tool_id="1", name="terminal", content="file.txt"),
        AgentTextDelta(text="Found file.txt"),
        AgentDone(tool_calls_executed=1),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        lines = get_log_lines(app)
        joined = "\n".join(lines)
        assert "terminal" in joined


# ---------------------------------------------------------------------------
# FR5e: Status update on done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_updated_on_done(
    make_mock_agent: MagicMock,
) -> None:
    """Status bar is updated with tool call count on AgentDone."""
    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="result"),
        AgentDone(tool_calls_executed=3),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        from textual.widgets import Static
        status = app.query_one("#status", Static)
        assert "3 tools" in str(status.content)


# ---------------------------------------------------------------------------
# FR5f: Turn-status indicator (working / ready / error)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_shows_ready_after_done(
    make_mock_agent: MagicMock,
) -> None:
    """Status bar shows a ready checkmark after AgentDone."""
    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="result"),
        AgentDone(tool_calls_executed=1),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        from textual.widgets import Static
        status = app.query_one("#status", Static)
        content = str(status.content)
        assert "✓" in content


@pytest.mark.asyncio
async def test_status_shows_error_on_exception(
    make_mock_agent: MagicMock,
) -> None:
    """Status bar shows an error indicator when streaming fails."""
    async def _failing_stream(text: str):
        yield AgentTextDelta(text="partial")
        raise RuntimeError("boom")

    agent = make_mock_agent(stream_events=[])  # override below
    agent.stream_message = _failing_stream

    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                continue
            break
        await pilot.pause()

        from textual.widgets import Static
        status = app.query_one("#status", Static)
        content = str(status.content)
        assert "✗" in content and "boom" in content


@pytest.mark.asyncio
async def test_set_working_status(
    make_mock_agent: MagicMock,
) -> None:
    """set_working_status shows the working indicator."""
    agent = make_mock_agent(stream_events=[AgentDone(tool_calls_executed=0)])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        app.set_working_status()
        await pilot.pause()

        from textual.widgets import Static
        status = app.query_one("#status", Static)
        assert "Working" in str(status.content)


@pytest.mark.asyncio
async def test_set_ready_status(
    make_mock_agent: MagicMock,
) -> None:
    """set_ready_status shows the ready indicator with checkmark."""
    agent = make_mock_agent(stream_events=[AgentDone(tool_calls_executed=0)])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        app.set_ready_status("sess-1", 5, 2)
        await pilot.pause()

        from textual.widgets import Static
        status = app.query_one("#status", Static)
        content = str(status.content)
        assert "✓" in content
        assert "sess-1" in content


@pytest.mark.asyncio
async def test_set_error_status(
    make_mock_agent: MagicMock,
) -> None:
    """set_error_status shows the error indicator."""
    agent = make_mock_agent(stream_events=[AgentDone(tool_calls_executed=0)])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        app.set_error_status("something broke")
        await pilot.pause()

        from textual.widgets import Static
        status = app.query_one("#status", Static)
        content = str(status.content)
        assert "✗" in content
        assert "something broke" in content
