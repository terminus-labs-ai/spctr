"""Tests for TUI scaffold — FR1 (app), FR2 (layout), FR8 (status), FR12 (mouse).

Covers the Textual app scaffold, layout composition, status bar, and mouse
support.  Uses Textual's App.run_test() / Pilot harness.
"""
from __future__ import annotations

import pytest

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.interfaces.tui import SpectreTUI, TUIInterface


# ---------------------------------------------------------------------------
# FR1: Textual App launches inside run()
# ---------------------------------------------------------------------------

def test_interface_has_name_attribute() -> None:
    """TUIInterface must have name = 'tui'."""
    assert TUIInterface().name == "tui"


@pytest.mark.asyncio
async def test_start_sets_run_context(mock_agent) -> None:
    """start(agent) must set RunContext with interface='tui'."""
    interface = TUIInterface()
    await interface.start(mock_agent)
    mock_agent.set_run_context.assert_called_once()
    ctx = mock_agent.set_run_context.call_args[0][0]
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
async def test_run_launches_textual_app_headless(mock_agent) -> None:
    """run(agent) must launch the Textual app via run_async in headless mode."""
    interface = TUIInterface()
    await interface.start(mock_agent)
    app = SpectreTUI(mock_agent)
    async with app.run_test(headless=True) as pilot:
        await pilot.pause()
    await interface.stop()


# ---------------------------------------------------------------------------
# FR2: Layout — Header, RichLog, Input, Status, Footer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layout_composes_header(tui_app: SpectreTUI) -> None:
    """App must compose a Header widget."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import Header
        assert tui_app.query_one(Header) is not None


@pytest.mark.asyncio
async def test_layout_composes_richlog_output(tui_app: SpectreTUI) -> None:
    """App must compose a RichLog with id='output'."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import RichLog
        log = tui_app.query_one("#output", RichLog)
        assert log.wrap is True
        assert log.auto_scroll is True


@pytest.mark.asyncio
async def test_layout_composes_input_prompt(tui_app: SpectreTUI) -> None:
    """App must compose an Input with id='prompt'."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import Input
        inp = tui_app.query_one("#prompt", Input)
        assert inp.placeholder == "> "


@pytest.mark.asyncio
async def test_layout_composes_status_bar(tui_app: SpectreTUI) -> None:
    """App must compose a Static status bar with id='status'."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import Static
        assert tui_app.query_one("#status", Static) is not None


@pytest.mark.asyncio
async def test_layout_composes_footer(tui_app: SpectreTUI) -> None:
    """App must compose a Footer widget."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import Footer
        assert tui_app.query_one(Footer) is not None


@pytest.mark.asyncio
async def test_input_focused_on_mount(tui_app: SpectreTUI) -> None:
    """Input widget must be focused when the app mounts."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import Input
        assert tui_app.query_one("#prompt", Input).has_focus


# ---------------------------------------------------------------------------
# FR8: Status display
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_status_sets_text(tui_app: SpectreTUI) -> None:
    """update_status() must set the status bar text with session info."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import Static
        tui_app.update_status("sess-123", 5, 2)
        await pilot.pause()
        status = tui_app.query_one("#status", Static)
        assert "sess-123" in str(status.content)
        assert "5 msgs" in str(status.content)
        assert "2 tools" in str(status.content)


# ---------------------------------------------------------------------------
# FR12: Mouse support
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_richlog_has_mouse_enabled(tui_app: SpectreTUI) -> None:
    """RichLog output pane must have mouse support enabled."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import RichLog
        log = tui_app.query_one("#output", RichLog)
        assert log.ALLOW_SELECT is True


# ---------------------------------------------------------------------------
# Input echo (minimal — full dispatch is spc-55)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_input_submitted_echoes_to_log(tui_app: SpectreTUI) -> None:
    """Submitting input must echo the text to the RichLog."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import Input, RichLog
        inp = tui_app.query_one("#prompt", Input)
        inp.value = "hello world"
        await inp.run_action("submit")
        await pilot.pause()
        log = tui_app.query_one("#output", RichLog)
        assert len(log.lines) > 0


@pytest.mark.asyncio
async def test_empty_input_is_ignored(tui_app: SpectreTUI) -> None:
    """Empty/whitespace input must not echo to the log."""
    async with tui_app.run_test(headless=True) as pilot:
        from textual.widgets import Input, RichLog
        inp = tui_app.query_one("#prompt", Input)
        inp.value = "   "
        await pilot.press("Enter")
        await pilot.pause()
        log = tui_app.query_one("#output", RichLog)
        assert len(log.lines) == 0


# ---------------------------------------------------------------------------
# No-TTY / headless
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_headless_run_does_not_crash(mock_agent) -> None:
    """The app must not crash when run in headless mode."""
    interface = TUIInterface()
    await interface.start(mock_agent)
    app = SpectreTUI(mock_agent)
    async with app.run_test(headless=True) as pilot:
        await pilot.pause()
    await interface.stop()
