"""Tests for the REPL interface (prompt_toolkit + Rich).

Covers: protocol conformance, CLI wiring, lifecycle (start/stop), slash
command dispatch and handlers, session save/load round-trip through the
shared session_io helpers.  Streaming rendering is covered in test_repl_streaming.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.interfaces.repl import REPLInterface
from sr2_spectre.interfaces.session_io import (
    default_save_path,
    deserialize_history,
    format_history_summary,
    serialize_history,
)


# ---------------------------------------------------------------------------
# Protocol & CLI wiring
# ---------------------------------------------------------------------------

def test_repl_interface_has_name() -> None:
    assert REPLInterface().name == "repl"


# ---------------------------------------------------------------------------
# Slash-command completion gating (regression: FuzzyWordCompleter fired on
# ANY typed word, popping a menu of commands on every space/enter)
# ---------------------------------------------------------------------------

def _complete(text: str) -> list[str]:
    """Run the REPL's slash completer against a buffer and return texts."""
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from sr2_spectre.interfaces.repl import _COMMANDS, _SlashCompleter

    doc = Document(text=text, cursor_position=len(text))
    return [c.text for c in _SlashCompleter(_COMMANDS).get_completions(doc, CompleteEvent(""))]


def test_completion_suppressed_on_normal_text() -> None:
    """Typing a normal sentence must NOT offer slash-command completions.

    This is the regression that made the REPL unusable: hitting space/enter
    on ordinary text opened a menu of every slash command and swallowed the
    keystroke.
    """
    assert _complete("the ") == []
    assert _complete("the message") == []
    assert _complete("hello world ") == []
    assert _complete("line1\nplain") == []


def test_completion_offers_commands_only_after_slash() -> None:
    """Slash-prefixed input still completes; bare slash lists everything."""
    from sr2_spectre.interfaces.repl import _COMMANDS

    assert _complete("/") == _COMMANDS
    assert _complete("/q") == ["/quit"]
    assert _complete("/load") == ["/load"]


def test_repl_satisfies_interface_protocol() -> None:
    from sr2_spectre.interfaces import Interface

    instance = REPLInterface()
    assert isinstance(instance, Interface)


def test_cli_loads_repl_interface() -> None:
    """_load_interface('repl') must return the REPLInterface class."""
    from sr2_spectre.cli import _load_interface

    cls = _load_interface("repl")
    assert cls is REPLInterface


@pytest.mark.asyncio
async def test_start_sets_run_context(make_mock_agent) -> None:
    """start(agent) sets RunContext with interface='repl', mode=INTERACTIVE."""
    agent = make_mock_agent()
    interface = REPLInterface()
    await interface.start(agent)

    assert interface._running is True
    mock_ctx = agent.set_run_context.call_args[0][0]
    assert isinstance(mock_ctx, RunContext)
    assert mock_ctx.interface == "repl"
    assert mock_ctx.mode == RunMode.INTERACTIVE


# ---------------------------------------------------------------------------
# Area derivation — cwd-basename semantics
#
# The REPL's area is the basename of the exact directory the process starts
# in. No CLAUDE.md lookup, no .git walk, no ancestor discovery. A non-empty
# SR2_AREA is the only override.
# ---------------------------------------------------------------------------

async def _start_area(monkeypatch, cwd: Path) -> RunContext:
    """Start the REPL from ``cwd`` with SR2_AREA unset; return the RunContext."""
    monkeypatch.delenv("SR2_AREA", raising=False)
    monkeypatch.chdir(cwd)
    agent = MagicMock()
    await REPLInterface().start(agent)
    ctx = agent.set_run_context.call_args[0][0]
    assert isinstance(ctx, RunContext)
    return ctx


@pytest.mark.asyncio
async def test_area_is_cwd_basename_despite_ancestor_markers(tmp_path, monkeypatch) -> None:
    """AC1: ancestor CLAUDE.md and .git must not affect the derived area."""
    vault = tmp_path / "obsidian"
    vault.mkdir()
    (vault / "CLAUDE.md").write_text("vault doc")
    (vault / ".git").mkdir()
    harbinger = vault / "projects" / "harbinger"
    harbinger.mkdir(parents=True)

    ctx = await _start_area(monkeypatch, harbinger)
    assert ctx.area == "harbinger"


@pytest.mark.asyncio
async def test_area_is_cwd_basename_without_any_markers(tmp_path, monkeypatch) -> None:
    """AC2: no CLAUDE.md, NOW.md, or .git required anywhere."""
    launch = tmp_path / "pindamonhagaba"
    launch.mkdir()

    ctx = await _start_area(monkeypatch, launch)
    assert ctx.area == "pindamonhagaba"


@pytest.mark.asyncio
async def test_area_is_exact_cwd_not_git_root_name(tmp_path, monkeypatch) -> None:
    """AC3: a subdirectory of a repo resolves to the subdirectory, not the repo."""
    repo = tmp_path / "sr2-spectre"
    (repo / ".git").mkdir(parents=True)
    src = repo / "src"
    src.mkdir()

    ctx = await _start_area(monkeypatch, src)
    assert ctx.area == "src"


@pytest.mark.asyncio
async def test_sr2_area_overrides_cwd_basename(tmp_path, monkeypatch) -> None:
    """AC4: a non-empty SR2_AREA is the only override."""
    launch = tmp_path / "somewhere"
    launch.mkdir()
    monkeypatch.setenv("SR2_AREA", "fractured-roots")
    monkeypatch.chdir(launch)

    agent = MagicMock()
    await REPLInterface().start(agent)
    ctx = agent.set_run_context.call_args[0][0]
    assert ctx.area == "fractured-roots"


@pytest.mark.asyncio
async def test_sr2_area_empty_falls_back_to_basename(tmp_path, monkeypatch) -> None:
    """An empty or whitespace-only SR2_AREA is not an override."""
    launch = tmp_path / "somewhere"
    launch.mkdir()
    for value in ("", "   "):
        monkeypatch.setenv("SR2_AREA", value)
        monkeypatch.chdir(launch)
        agent = MagicMock()
        await REPLInterface().start(agent)
        ctx = agent.set_run_context.call_args[0][0]
        assert ctx.area == "somewhere"


@pytest.mark.asyncio
async def test_stop_sets_running_false() -> None:
    interface = REPLInterface()
    interface._running = True
    await interface.stop()
    assert interface._running is False


# ---------------------------------------------------------------------------
# Slash command dispatch & handlers
# ---------------------------------------------------------------------------

def _console_capturing(interface: REPLInterface) -> list[str]:
    """Point the interface console at an in-memory buffer; return it."""
    from rich.console import Console

    buf = []
    interface.console = Console(file=_LineSink(buf), force_terminal=False, width=100)
    return buf


class _LineSink:
    """Minimal file-like sink that records written strings line-wise."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def write(self, s: str) -> int:
        for part in s.split("\n"):
            if part:
                self._lines.append(part)
        return len(s)

    def flush(self) -> None:
        pass


def _plain_text(lines: list[str]) -> str:
    """Strip ANSI codes from captured output for assertions."""
    import re

    out = "\n".join(lines)
    return re.sub(r"\x1b\[[0-9;]*m", "", out)


@pytest.mark.asyncio
async def test_quit_command_stops_loop() -> None:
    interface = REPLInterface()
    _console_capturing(interface)
    interface._running = True

    await interface._handle_command(MagicMock(), "/quit", None)

    assert interface._running is False


@pytest.mark.asyncio
async def test_exit_command_stops_loop() -> None:
    interface = REPLInterface()
    _console_capturing(interface)
    interface._running = True

    await interface._handle_command(MagicMock(), "/exit", None)

    assert interface._running is False


@pytest.mark.asyncio
async def test_unknown_command_reports_error() -> None:
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(MagicMock(), "/nope", None)

    assert "Unknown command: /nope" in _plain_text(buf)


@pytest.mark.asyncio
async def test_reset_starts_new_session(make_mock_agent) -> None:
    agent = make_mock_agent()
    agent.new_session = MagicMock()
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/reset", None)

    agent.new_session.assert_called_once_with()
    assert "New session started" in _plain_text(buf)


@pytest.mark.asyncio
async def test_help_lists_commands(make_mock_agent) -> None:
    agent = make_mock_agent()
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/help", None)

    text = _plain_text(buf)
    for cmd in ("/quit", "/reset", "/tools", "/history", "/save", "/load"):
        assert cmd in text


@pytest.mark.asyncio
async def test_tools_lists_registry_names(make_mock_agent) -> None:
    agent = make_mock_agent(tools=["terminal", "grep"])
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/tools", None)

    assert "Available tools: terminal, grep" in _plain_text(buf)


@pytest.mark.asyncio
async def test_history_empty_summary(make_mock_agent) -> None:
    agent = make_mock_agent(history=[])
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/history", None)

    assert "No conversation history." in _plain_text(buf)


# ---------------------------------------------------------------------------
# Session save/load round-trip (shared helpers + command handlers)
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path: Path, make_mock_agent) -> None:
    agent = make_mock_agent()
    from sr2.models import Message, TextBlock

    agent.history = [
        Message(role="user", content=[TextBlock(text='say "hi" to the world')]),
        Message(role="assistant", content=[TextBlock(text="hi")]),
    ]

    path = tmp_path / "sess.json"
    interface = REPLInterface()
    _console_capturing(interface)

    # Run the async command handlers in a fresh event loop (plain sync test,
    # no pytest-asyncio coupling needed).
    import asyncio

    def run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    run(interface._cmd_save(agent, str(path)))

    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 2
    # Quotes survive serialization (the exact thing the old TUI broke on input)
    assert 'say "hi" to the world' in data[0]["content"][0]["text"]

    # /load into a fresh agent
    agent2 = make_mock_agent()
    run(interface._cmd_load(agent2, str(path)))

    assert len(agent2.history) == 2


@pytest.mark.asyncio
async def test_save_default_path(tmp_path: Path, make_mock_agent, monkeypatch) -> None:
    agent = make_mock_agent()
    from sr2.models import Message, TextBlock

    agent.history = [Message(role="user", content=[TextBlock(text="x")])]
    target = tmp_path / "default" / "session.json"
    monkeypatch.setattr(
        "sr2_spectre.interfaces.repl.default_save_path", lambda: target
    )
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._cmd_save(agent, None)

    assert target.exists()
    assert str(target) in _plain_text(buf)


@pytest.mark.asyncio
async def test_load_missing_file_reports_error(make_mock_agent) -> None:
    agent = make_mock_agent()
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._cmd_load(agent, "/nonexistent/nope.json")

    assert "file not found" in _plain_text(buf)


@pytest.mark.asyncio
async def test_load_without_arg_reports_usage(make_mock_agent) -> None:
    agent = make_mock_agent()
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._cmd_load(agent, None)

    assert "Usage: /load <path>" in _plain_text(buf)


# ---------------------------------------------------------------------------
# session_io unit coverage (shared with tui.py semantics)
# ---------------------------------------------------------------------------

def test_serialize_deserialize_roundtrip(make_mock_agent) -> None:
    from sr2.models import Message, TextBlock

    history = [
        Message(role="user", content=[TextBlock(text='a "quoted" string')]),
        Message(role="assistant", content=[TextBlock(text="b")]),
    ]
    data = serialize_history(history)
    assert isinstance(data, list)
    restored = deserialize_history(data)
    assert len(restored) == 2
    assert restored[0].role == "user"


def test_format_history_summary_truncates(make_mock_agent) -> None:
    from sr2.models import Message, TextBlock

    long_text = "x" * 500
    summary = format_history_summary([Message(role="user", content=[TextBlock(text=long_text)])])
    assert "..." in summary
    assert "(1 messages)" in summary


def test_default_save_path_under_home() -> None:
    p = default_save_path()
    assert p.name == "session.json"
    assert ".sr2-spectre" in str(p)


# ---------------------------------------------------------------------------
# Enter / Shift+Enter bindings (regression: prompt_toolkit's multiline default
# is inverted — Enter inserted a newline and Meta+Enter submitted)
# ---------------------------------------------------------------------------

async def _feed_keys(keys: str, timeout: float = 3.0) -> str:
    """Drive a real prompt session with raw key input and return the result.

    Returns the sentinel ``"<no submit>"`` when the keys never accept the
    buffer, so a binding that silently stops submitting is a visible failure
    rather than a hung test.
    """
    import asyncio

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from sr2_spectre.interfaces.repl import _COMMANDS, _SlashCompleter, _make_prompt_session

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            session = _make_prompt_session(_SlashCompleter(_COMMANDS))
            pipe_input.send_text(keys)
            try:
                return await asyncio.wait_for(session.prompt_async(), timeout=timeout)
            except asyncio.TimeoutError:
                return "<no submit>"


@pytest.mark.asyncio
async def test_enter_submits(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sr2_spectre.interfaces.repl._history_file", lambda: tmp_path / "h")
    assert await _feed_keys("hello\r") == "hello"


@pytest.mark.asyncio
async def test_enter_submits_multiline_buffer(tmp_path, monkeypatch) -> None:
    """Plain Enter submits even when the buffer already holds a newline.

    The embedded literal LF mirrors the newline Shift+Enter inserts. A trailing
    CR must accept the multiline buffer rather than insert another newline.
    """
    monkeypatch.setattr("sr2_spectre.interfaces.repl._history_file", lambda: tmp_path / "h")
    assert await _feed_keys("line1\nline2\r") == "line1\nline2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "sequence"),
    [
        ("xterm modifyOtherKeys", "\x1b[27;2;13~"),
        ("kitty CSI-u", "\x1b[13;2u"),
        ("meta/alt+enter", "\x1b\r"),
    ],
)
async def test_shift_enter_inserts_newline(name, sequence, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sr2_spectre.interfaces.repl._history_file", lambda: tmp_path / "h")
    assert await _feed_keys(f"a{sequence}b\r") == "a\nb"


def test_shift_enter_sequences_are_distinct_from_enter() -> None:
    """The shift-modified encodings must not resolve to plain Enter.

    prompt_toolkit ships them folded onto Keys.ControlM, which is what made
    Shift+Enter indistinguishable from Enter in the first place.
    """
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.keys import Keys

    from sr2_spectre.interfaces.repl import _SHIFT_ENTER_SEQUENCES

    for sequence in _SHIFT_ENTER_SEQUENCES:
        assert ANSI_SEQUENCES[sequence] is Keys.ControlJ
    assert ANSI_SEQUENCES["\x0d"] is Keys.ControlM
