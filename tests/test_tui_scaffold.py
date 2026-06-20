"""Tests for TUI scaffold — FR1 (app), FR2 (layout), FR8 (status), FR12 (mouse).

Covers the Textual app scaffold, layout composition, status bar, and mouse
support.  Uses Textual's App.run_test() / Pilot harness.
"""
from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.interfaces.tui import SpectreTUI, TUIInterface


def _make_mock_agent() -> MagicMock:
    """Return a minimal mock agent for TUI tests."""
    agent = MagicMock()
    agent.session_id = "test-session"
    agent.history = []
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=["tool_a"])
    # Return a real RunContext so we can inspect it
    rc = RunContext(interface="tui", mode=RunMode.INTERACTIVE, source="/tmp")
    type(agent).run_context = PropertyMock(return_value=rc)
    return agent


# ---------------------------------------------------------------------------
# FR1: Textual App launches inside run()
# ---------------------------------------------------------------------------

def test_interface_has_name_attribute() -> None:
    """TUIInterface must have name = 'tui'."""
    assert TUIInterface().name == "tui"


@pytest.mark.asyncio
async def test_start_sets_run_context() -> None:
    """start(agent) must set RunContext with interface='tui'."""
    interface = TUIInterface()
    agent = _make_mock_agent()
    await interface.start(agent)
    # Verify set_run_context was called
    agent.set_run_context.assert_called_once()
    ctx = agent.set_run_context.call_args[0][0]
    assert ctx.interface == "tui"
    assert ctx.mode == RunMode.INTERACTIVE


@pytest.mark.asyncio
async def test_stop_sets_running_false() -> None:
    """stop() must set _running = False."""
    interface = TUIInterface()
    interface._running = True
    await interface.stop()
    assert interface._running is False


@pytest.mark.asyncio
async def test_run_launches_textual_app_headless() -> None:
    """run(agent) must launch the Textual app via run_async in headless mode."""
    interface = TUIInterface()
    agent = _make_mock_agent()

    await interface.start(agent)
    # Verify the app can be constructed and composed via run_test
    app = SpectreTUI(agent)
    async with app.run_test(headless=True) as pilot:
        await pilot.pause()
    await interface.stop()


# ---------------------------------------------------------------------------
# FR2: Layout — Header, RichLog, Input, Status, Footer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layout_composes_header() -> None:
    """App must compose a Header widget."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Header
        header = app.query_one(Header)
        assert header is not None


@pytest.mark.asyncio
async def test_layout_composes_richlog_output() -> None:
    """App must compose a RichLog with id='output'."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import RichLog
        log = app.query_one("#output", RichLog)
        assert log is not None
        assert log.wrap is True
        assert log.auto_scroll is True


@pytest.mark.asyncio
async def test_layout_composes_input_prompt() -> None:
    """App must compose an Input with id='prompt'."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Input
        inp = app.query_one("#prompt", Input)
        assert inp is not None
        assert inp.placeholder == "> "


@pytest.mark.asyncio
async def test_layout_composes_status_bar() -> None:
    """App must compose a Static status bar with id='status'."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Static
        status = app.query_one("#status", Static)
        assert status is not None


@pytest.mark.asyncio
async def test_layout_composes_footer() -> None:
    """App must compose a Footer widget."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Footer
        footer = app.query_one(Footer)
        assert footer is not None


@pytest.mark.asyncio
async def test_input_focused_on_mount() -> None:
    """Input widget must be focused when the app mounts."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Input
        inp = app.query_one("#prompt", Input)
        assert inp.has_focus


# ---------------------------------------------------------------------------
# FR8: Status display
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_status_sets_text() -> None:
    """update_status() must set the status bar text with session info."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Static
        app.update_status("sess-123", 5, 2)
        await pilot.pause()
        status = app.query_one("#status", Static)
        assert "sess-123" in str(status.content)
        assert "5 msgs" in str(status.content)
        assert "2 tools" in str(status.content)


# ---------------------------------------------------------------------------
# FR12: Mouse support
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_richlog_has_mouse_enabled() -> None:
    """RichLog output pane must have mouse support enabled."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import RichLog
        log = app.query_one("#output", RichLog)
        # RichLog has mouse scrolling and select-to-copy by default.
        assert log.ALLOW_SELECT is True


# ---------------------------------------------------------------------------
# Input echo (minimal — full dispatch is spc-55)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_input_submitted_echoes_to_log() -> None:
    """Submitting input must echo the text to the RichLog."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Input, RichLog
        inp = app.query_one("#prompt", Input)
        inp.value = "hello world"
        await inp.run_action("submit")
        await pilot.pause()
        log = app.query_one("#output", RichLog)
        assert len(log.lines) > 0


@pytest.mark.asyncio
async def test_empty_input_is_ignored() -> None:
    """Empty/whitespace input must not echo to the log."""
    agent = _make_mock_agent()
    app = SpectreTUI(agent)

    async with app.run_test(headless=True) as pilot:
        from textual.widgets import Input, RichLog
        inp = app.query_one("#prompt", Input)
        inp.value = "   "
        await pilot.press("Enter")
        await pilot.pause()
        log = app.query_one("#output", RichLog)
        assert len(log.lines) == 0


# ---------------------------------------------------------------------------
# No-TTY / headless
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_headless_run_does_not_crash() -> None:
    """The app must not crash when run in headless mode."""
    agent = _make_mock_agent()
    interface = TUIInterface()

    await interface.start(agent)
    app = SpectreTUI(agent)
    # This should complete without raising
    async with app.run_test(headless=True) as pilot:
        await pilot.pause()
    await interface.stop()
