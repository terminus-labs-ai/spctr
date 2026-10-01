"""Replacement Sessions keep their session log (obsidian-gb5j, spec FR1/FR2).

Every Session created by Runtime owns a log. When an Agent replaces its
Session — REPL/TUI ``/reset`` via ``Agent.new_session()``, or Discord via the
``session_id`` setter — the replacement inherits the run context already in
force, so it opens its own log, records ``session.start`` with the interface,
and announces the new path on stderr exactly as the original Session did.

An Agent that has never received a run context still opens no log.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from sr2.protocols.llm import StreamEvent
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.run_log import SessionLogManager
from tests.test_session_run_log import ScriptedLLM, _agent, _events, _named


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def log_dir() -> Path:
    """The session-log directory under the conftest-isolated SR2_HOME."""
    return SessionLogManager().directory


def _files(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("*.jsonl")) if log_dir.exists() else []


def _new_files(log_dir: Path, before: list[Path]) -> list[Path]:
    return [p for p in _files(log_dir) if p not in before]


def _repl_ctx() -> RunContext:
    return RunContext(
        interface="repl", mode=RunMode.INTERACTIVE, source="/work/proj", area="proj",
    )


def _assert_fresh_log(path: Path, interface: str, session_id: str) -> None:
    starts = _named(_events(path), "session.start")
    assert len(starts) == 1, starts
    assert starts[0]["interface"] == interface
    assert starts[0]["session_id"] == session_id


async def _start_repl(agent):
    from sr2_spectre.interfaces.repl import REPLInterface

    repl = REPLInterface(console=Console(file=io.StringIO()))
    await repl.start(agent)
    return repl


async def _start_discord(agent):
    from sr2_spectre.interfaces.discord.config import DiscordConfig
    from sr2_spectre.interfaces.discord.interface import DiscordInterface

    adapter = AsyncMock()
    adapter.bot_id = 12345
    adapter.bot_mentions = []
    adapter.set_message_handler = MagicMock()
    adapter.set_slash_handler = MagicMock()
    iface = DiscordInterface(config=DiscordConfig())
    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter",
        return_value=adapter,
    ):
        await iface.start(agent)
    return iface


# ---------------------------------------------------------------------------
# AC 1 — Agent.new_session() on an agent with a run context
# ---------------------------------------------------------------------------

class TestNewSessionOpensLog:
    def test_new_session_opens_a_second_log_recording_the_interface(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)

        agent.new_session()

        new = _new_files(log_dir, [first])
        assert len(new) == 1, "reset Session opened no log of its own"
        assert new[0] != first
        _assert_fresh_log(new[0], "repl", agent.session_id)

    def test_new_session_announces_the_new_path_on_stderr(self, log_dir, capsys):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        capsys.readouterr()

        agent.new_session()

        new = _new_files(log_dir, [first])
        assert len(new) == 1, "reset Session opened no log of its own"
        out = capsys.readouterr()
        assert str(new[0]) in out.err
        assert str(new[0]) not in out.out
        assert str(first) not in out.err, "original log re-announced"

    def test_new_session_keeps_the_run_context(self):
        agent = _agent()
        ctx = _repl_ctx()
        agent.set_run_context(ctx)

        agent.new_session()

        assert agent.run_context == ctx

    def test_new_session_with_explicit_id_logs_under_that_id(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)

        agent.new_session("fresh-frame")

        new = _new_files(log_dir, [first])
        assert len(new) == 1
        _assert_fresh_log(new[0], "repl", "fresh-frame")

    def test_each_reset_opens_its_own_log(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        agent.new_session()
        agent.new_session()

        files = _files(log_dir)
        assert len(files) == 3
        for f in files:
            _assert_fresh_log(f, "repl", "edi-default")

    def test_original_log_gets_no_second_session_start(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)

        agent.new_session()

        _assert_fresh_log(first, "repl", "edi-default")

    async def test_turn_after_reset_is_logged_in_the_new_log(self, log_dir):
        agent = _agent(ScriptedLLM([StreamEvent(type="text", text="ok")]))
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        before = _events(first)

        agent.new_session()
        await agent.handle_user_message("after reset")

        new = _new_files(log_dir, [first])
        assert len(new) == 1, "reset Session opened no log of its own"
        names = [e["event"] for e in _events(new[0])]
        assert names[0] == "session.start"
        assert any(n.startswith("turn.") for n in names[1:]), names
        assert _events(first) == before, "post-reset turn written to the old log"


# ---------------------------------------------------------------------------
# AC 2 — the session_id setter replaces the Session too
# ---------------------------------------------------------------------------

class TestSessionIdSetterOpensLog:
    def test_setter_opens_a_log_without_reapplying_context(self, log_dir, capsys):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        capsys.readouterr()

        agent.session_id = "channel-7"

        new = _new_files(log_dir, [first])
        assert len(new) == 1, "Session built by the setter opened no log"
        _assert_fresh_log(new[0], "repl", "channel-7")
        assert str(new[0]) in capsys.readouterr().err

    def test_setter_keeps_the_run_context(self):
        agent = _agent()
        ctx = _repl_ctx()
        agent.set_run_context(ctx)

        agent.session_id = "channel-7"

        assert agent.run_context == ctx


# ---------------------------------------------------------------------------
# AC 3 — REPL and TUI /reset produce a logged new session
# ---------------------------------------------------------------------------

class TestInterfaceResetCommands:
    async def test_repl_reset_command_opens_a_repl_log(self, log_dir, capsys):
        agent = _agent()
        repl = await _start_repl(agent)
        (first,) = _files(log_dir)
        capsys.readouterr()

        await repl._handle_command(agent, "/reset", None)

        new = _new_files(log_dir, [first])
        assert len(new) == 1, "REPL /reset left the new session without a log"
        _assert_fresh_log(new[0], "repl", agent.session_id)
        assert str(new[0]) in capsys.readouterr().err

    async def test_tui_reset_command_opens_a_tui_log(self, log_dir, capsys):
        from sr2_spectre.interfaces.tui import SpectreTUI, TUIInterface

        agent = _agent()
        await TUIInterface().start(agent)
        (first,) = _files(log_dir)
        capsys.readouterr()

        await SpectreTUI(agent)._handle_command("/reset", None, MagicMock())

        new = _new_files(log_dir, [first])
        assert len(new) == 1, "TUI /reset left the new session without a log"
        _assert_fresh_log(new[0], "tui", agent.session_id)
        assert str(new[0]) in capsys.readouterr().err


# ---------------------------------------------------------------------------
# AC 4 — Discord session switch: exactly one log for the new session
# ---------------------------------------------------------------------------

class TestDiscordSessionSwitch:
    async def test_channel_switch_opens_exactly_one_discord_log(self, log_dir, capsys):
        from sr2_spectre.interfaces.discord.session_map import ChannelSession

        agent = _agent()
        iface = await _start_discord(agent)
        (first,) = _files(log_dir)
        capsys.readouterr()

        iface._restore_history(ChannelSession(channel_id=42, session_id="discord-42"))

        new = _new_files(log_dir, [first])
        assert len(new) == 1, new
        _assert_fresh_log(new[0], "discord", "discord-42")
        assert capsys.readouterr().err.count(str(new[0])) == 1
        _assert_fresh_log(first, "discord", "edi-default")
        assert agent.run_context is not None
        assert agent.run_context.interface == "discord"


# ---------------------------------------------------------------------------
# AC 5 — no run context yet: no log is opened
# ---------------------------------------------------------------------------

class TestNoRunContext:
    def test_new_session_without_run_context_opens_no_log(self, log_dir, capsys):
        agent = _agent()
        agent.new_session()

        assert _files(log_dir) == []
        assert agent.run_context is None
        assert str(log_dir) not in capsys.readouterr().err

    def test_setter_without_run_context_opens_no_log(self, log_dir):
        agent = _agent()
        agent.session_id = "channel-7"

        assert _files(log_dir) == []
        assert agent.run_context is None

    def test_first_context_after_reset_opens_exactly_one_log(self, log_dir):
        agent = _agent()
        agent.new_session()
        agent.set_run_context(_repl_ctx())

        (only,) = _files(log_dir)
        _assert_fresh_log(only, "repl", "edi-default")
