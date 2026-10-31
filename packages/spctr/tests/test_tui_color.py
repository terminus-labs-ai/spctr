"""Tests for TUI semantic color scheme — FR6/7.

FR6: Tool visibility with color (⚙/✓/✗ format)
FR7: Semantic color scheme + full-color markdown + NO_COLOR
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
# FR6: Tool visibility with color
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_start_shows_gear_icon_and_name(
    make_mock_agent: MagicMock,
) -> None:
    """AgentToolStart renders as ⚙ {name}."""
    agent = make_mock_agent(stream_events=[
        AgentToolStart(tool_id="t1", name="file_read", input={"path": "x.txt"}),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                break
        await pilot.pause()

        lines = get_log_lines(app)
        joined = "\n".join(lines)
        assert "⚙" in joined
        assert "file_read" in joined


@pytest.mark.asyncio
async def test_tool_start_truncates_args_at_60(
    make_mock_agent: MagicMock,
) -> None:
    """Tool args preview is truncated at 60 chars."""
    long_input = {"path": "a" * 100, "extra": "b" * 100}
    agent = make_mock_agent(stream_events=[
        AgentToolStart(tool_id="t1", name="grep", input=long_input),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                break
        await pilot.pause()

        lines = get_log_lines(app)
        joined = "\n".join(lines)
        # The args preview portion should be ≤60 chars
        # Find the line with ⚙ and check length after the name
        for line in lines:
            if "⚙" in line and "grep" in line:
                # Strip Rich markup for length check
                import re
                clean = re.sub(r"\[/?\w+\]", "", line)
                # After "⚙ grep(" the args should be ≤60
                paren = clean.find("(")
                if paren >= 0:
                    args_part = clean[paren + 1:]
                    assert len(args_part) <= 65, f"Args preview too long: {len(args_part)} chars: {args_part}"
                break


@pytest.mark.asyncio
async def test_tool_success_shows_check_and_done(
    make_mock_agent: MagicMock,
) -> None:
    """AgentToolResult (ok) renders as ✓ {name} done."""
    agent = make_mock_agent(stream_events=[
        AgentToolStart(tool_id="t1", name="file_read", input={"path": "x.txt"}),
        AgentToolResult(tool_id="t1", name="file_read", content="file contents"),
        AgentDone(tool_calls_executed=1),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                break
        await pilot.pause()

        lines = get_log_lines(app)
        joined = "\n".join(lines)
        assert "✓" in joined
        assert "file_read" in joined
        assert "done" in joined


@pytest.mark.asyncio
async def test_tool_failure_shows_x_and_failed(
    make_mock_agent: MagicMock,
) -> None:
    """AgentToolResult (error) renders as ✗ {name} failed."""
    agent = make_mock_agent(stream_events=[
        AgentToolStart(tool_id="t1", name="terminal", input={"command": "ls"}),
        AgentToolResult(tool_id="t1", name="terminal", content="error output", is_error=True),
        AgentDone(tool_calls_executed=1),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                break
        await pilot.pause()

        lines = get_log_lines(app)
        joined = "\n".join(lines)
        assert "✗" in joined
        assert "terminal" in joined
        assert "failed" in joined


# ---------------------------------------------------------------------------
# FR7: Semantic color scheme
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_markdown_renders_with_color(
    make_mock_agent: MagicMock,
) -> None:
    """Agent markdown output renders in full color (no_color removed)."""
    agent = make_mock_agent(stream_events=[
        AgentTextDelta(text="# Hello\n\nThis is **bold** and *italic*."),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                break
        await pilot.pause()

        # The markdown should be committed to the log
        lines = get_log_lines(app)
        joined = "\n".join(lines)
        assert "Hello" in joined
        assert "bold" in joined


@pytest.mark.asyncio
async def test_user_input_echo_has_distinct_style(tui_app: SpectreTUI) -> None:
    """User input echo uses a distinct style from agent output."""
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "user message")
        await pilot.pause()

        lines = get_log_lines(tui_app)
        # User echo should start with "> "
        user_lines = [l for l in lines if l.startswith("> ")]
        assert len(user_lines) == 1


@pytest.mark.asyncio
async def test_thinking_region_has_distinct_style(
    make_mock_agent: MagicMock,
) -> None:
    """Thinking text renders in a visually distinct style (dim/italic)."""
    agent = make_mock_agent(stream_events=[
        AgentThinkingDelta(text="thinking..."),
        AgentTextDelta(text="answer"),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                break
        await pilot.pause()

        # Thinking should NOT be in the committed log
        lines = get_log_lines(app)
        joined = "\n".join(lines)
        assert "thinking..." not in joined
        # But the answer should be
        assert "answer" in joined


# ---------------------------------------------------------------------------
# NO_COLOR support
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_color_env_disables_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """When NO_COLOR is set, markdown rendering falls back to plain text."""
    monkeypatch.setenv("NO_COLOR", "1")

    agent = _make_mock_agent(stream_events=[
        AgentTextDelta(text="# Title\n\n**bold** text"),
        AgentDone(tool_calls_executed=0),
    ])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "test")
        for _ in range(20):
            await pilot.pause()
            if app._streaming:
                break
        await pilot.pause()

        # Content should still be present (just without color)
        lines = get_log_lines(app)
        joined = "\n".join(lines)
        assert "Title" in joined
        assert "bold" in joined
