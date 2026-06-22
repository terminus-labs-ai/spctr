"""Tests for TUI input dispatch + slash commands — FR3/4.

Tests drive the app via Pilot, assert on widget content and agent calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import AgentDone, AgentEvent, AgentTextDelta
from sr2_spectre.interfaces.tui import SpectreTUI
from tests.conftest import (
    _make_mock_agent,
    assert_log_contains,
    get_log_lines,
    submit_input,
)


# ---------------------------------------------------------------------------
# FR3: Input dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_slash_input_dispatches_to_agent(tui_app: SpectreTUI) -> None:
    """Non-empty non-slash input must call agent.stream_message()."""
    agent = tui_app.agent
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "tell me a joke")
        await pilot.pause()
        assert agent._stream_call_log == ["tell me a joke"]


@pytest.mark.asyncio
async def test_slash_input_does_not_dispatch_to_agent(tui_app: SpectreTUI) -> None:
    """Slash-prefixed input must NOT call agent.stream_message()."""
    agent = tui_app.agent
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "/help")
        await pilot.pause()
        assert agent._stream_call_log == []


@pytest.mark.asyncio
async def test_empty_input_does_not_dispatch(tui_app: SpectreTUI) -> None:
    """Empty/whitespace input must not dispatch to agent."""
    agent = tui_app.agent
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "   ")
        await pilot.pause()
        assert agent._stream_call_log == []


# ---------------------------------------------------------------------------
# FR4: Slash commands
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quit_command_exits_app(tui_app: SpectreTUI) -> None:
    """/quit must exit the app."""
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "/quit")
        await pilot.pause()
        assert not tui_app.is_running


@pytest.mark.asyncio
async def test_exit_command_exits_app(tui_app: SpectreTUI) -> None:
    """/exit must exit the app."""
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "/exit")
        await pilot.pause()
        assert not tui_app.is_running


@pytest.mark.asyncio
async def test_reset_command_starts_new_session(tui_app: SpectreTUI) -> None:
    """/reset must call agent.new_session() and show confirmation."""
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "/reset")
        await pilot.pause()
        tui_app.agent.new_session.assert_called_once()
        assert_log_contains(tui_app, "New session started")


@pytest.mark.asyncio
async def test_help_command_shows_help(tui_app: SpectreTUI) -> None:
    """/help must display the help text in the log."""
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "/help")
        await pilot.pause()
        assert_log_contains(tui_app, "Commands:")
        assert_log_contains(tui_app, "/quit")


@pytest.mark.asyncio
async def test_tools_command_lists_tools(tui_app: SpectreTUI) -> None:
    """/tools must list available tool names in the log."""
    agent = tui_app.agent
    agent.registry.list_names = MagicMock(return_value=["edit", "grep", "terminal"])
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "/tools")
        await pilot.pause()
        lines = get_log_lines(tui_app)
        joined = "\n".join(lines)
        assert "edit" in joined
        assert "grep" in joined
        assert "terminal" in joined


@pytest.mark.asyncio
async def test_history_command_shows_summary(tui_app: SpectreTUI) -> None:
    """/history must show the conversation history summary."""
    # Create a mock agent with some history
    from sr2.models import Message, TextBlock

    history = [
        Message(role="user", content=[TextBlock(text="hello")]),
        Message(role="assistant", content=[TextBlock(text="hi there")]),
    ]
    agent = _make_mock_agent(history=history)
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "/history")
        await pilot.pause()
        assert_log_contains(app, "History (2 messages)")


@pytest.mark.asyncio
async def test_history_empty_shows_message(tui_app: SpectreTUI) -> None:
    """/history with no history must show 'No conversation history'."""
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "/history")
        await pilot.pause()
        assert_log_contains(tui_app, "No conversation history")


@pytest.mark.asyncio
async def test_save_command_saves_session(tmp_path: Path, tui_app: SpectreTUI) -> None:
    """/save [path] must serialize history to JSON at the given path."""
    from sr2.models import Message, TextBlock

    history = [
        Message(role="user", content=[TextBlock(text="test message")]),
    ]
    agent = _make_mock_agent(history=history)
    app = SpectreTUI(agent)

    save_path = tmp_path / "test_session.json"

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, f"/save {save_path}")
        await pilot.pause()
        assert_log_contains(app, "Session saved")
        # Verify file was written
        assert save_path.exists()
        data = json.loads(save_path.read_text())
        assert len(data) == 1
        assert data[0]["role"] == "user"


@pytest.mark.asyncio
async def test_save_default_path(tmp_path: Path, tui_app: SpectreTUI) -> None:
    """/save without path must use default location."""
    from sr2.models import Message, TextBlock

    history = [
        Message(role="user", content=[TextBlock(text="test")]),
    ]
    agent = _make_mock_agent(history=history)
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, "/save")
        await pilot.pause()
        # Should mention "Session saved"
        assert_log_contains(app, "Session saved")


@pytest.mark.asyncio
async def test_load_command_loads_session(tmp_path: Path) -> None:
    """/load [path] must deserialize history from JSON and restore it."""
    from sr2.models import Message, TextBlock

    # Create a save file
    save_path = tmp_path / "load_test.json"
    save_data = [
        {"role": "user", "content": [{"type": "text", "text": "loaded message"}]},
    ]
    save_path.write_text(json.dumps(save_data))

    agent = _make_mock_agent(history=[])
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        await submit_input(pilot, app, f"/load {save_path}")
        await pilot.pause()
        assert_log_contains(app, "Session loaded")
        # Verify history was restored
        assert len(app.agent.history) == 1


@pytest.mark.asyncio
async def test_load_nonexistent_file(tui_app: SpectreTUI) -> None:
    """/load with missing path must show error in log."""
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "/load /nonexistent/path.json")
        await pilot.pause()
        assert_log_contains(tui_app, "Error")


@pytest.mark.asyncio
async def test_unknown_command_shows_hint(tui_app: SpectreTUI) -> None:
    """/unknown must show 'Unknown command' hint in log."""
    async with tui_app.run_test(headless=True) as pilot:
        await submit_input(pilot, tui_app, "/foobar")
        await pilot.pause()
        assert_log_contains(tui_app, "Unknown command")
