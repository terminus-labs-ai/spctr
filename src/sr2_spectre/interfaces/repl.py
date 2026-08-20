"""REPL interface — prompt_toolkit + Rich terminal REPL.

A native-terminal replacement for the Textual TUI.  Because it never takes
over the screen (no alt-screen buffer), copy/select, paste and quotes all
work exactly like a normal shell session:

- prompt_toolkit PromptSession gives real readline shortcuts, multiline
  input, persistent history and slash-command completion.
- Rich renders streaming output to stdout: text deltas stream live, thinking
  is dimmed/italicized, tool calls show as status lines, the final reply is
  rendered as Markdown.

Slash commands (identical semantics to the TUI):
  /quit /exit /reset /help /tools /history /save [path] /load <path>

Usage: sr2-spectre config.yaml --interface repl
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.filters import IsMultiline
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import (
    AgentDone,
    AgentThinkingDelta,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
)
from sr2_spectre.interfaces.session_io import (
    default_save_path,
    format_history_summary,
    load_session,
    save_session,
)

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent

_HELP = """\
Commands:
  /quit     — exit the REPL
  /exit     — exit the REPL
  /reset    — start a new session
  /help     — show this help
  /tools    — list available tools
  /history  — show conversation history
  /save [path]  — save session to JSON (default: ~/.sr2-spectre/session.json)
  /load <path>  — load session from JSON

Multiline input: end a line with a backslash (\\) to continue on the next line.
Ctrl-D or /quit exits.  History is persisted between runs.
"""

_COMMANDS = ["/quit", "/exit", "/reset", "/help", "/tools", "/history", "/save", "/load"]


def render_markdown(text: str, console: Console | None = None) -> Text | Markdown:
    """Return a Rich renderable for markdown text.

    Falls back to plain Text if rich rendering is unavailable.  Honors NO_COLOR
    via the passed (or default) console — no string round-tripping needed in a
    native terminal, which also keeps copy/paste clean of ANSI artifacts.
    """
    try:
        return Markdown(text)
    except Exception:
        return Text(text)


def _history_file() -> Path:
    """Path to the persistent prompt history file."""
    path = Path.home() / ".sr2-spectre" / "history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _make_prompt_session(completer):
    """Build the PromptSession.  Factored out so tests can monkeypatch it
    without fighting parameterized generic syntax (PromptSession[str])."""
    return PromptSession(
        message="> ",
        history=FileHistory(str(_history_file())),
        completer=completer,
        # Only treat a trailing backslash as "continue on next line" when the
        # buffer is already multiline.  A plain single-line prompt with
        # multiline=True would swallow EVERY backslash (breaking normal text).
        multiline=IsMultiline(),
    )


class REPLInterface:
    """Interactive terminal REPL backed by prompt_toolkit + Rich."""

    name = "repl"

    def __init__(self, console: Console | None = None) -> None:
        self._running = False
        self.console: Console = console or Console()
        self._session: PromptSession | None = None
        self._stop_event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Interface protocol
    # ------------------------------------------------------------------

    async def start(self, agent: "Agent") -> None:
        """Initialize REPL state and set interactive run context."""
        self._running = True
        agent.set_run_context(RunContext(
            interface="repl",
            mode=RunMode.INTERACTIVE,
            source=os.getcwd(),
        ))

    async def stop(self) -> None:
        """Signal the run loop to exit (e.g. from another task)."""
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()

    async def run(self, agent: "Agent") -> None:
        """Run the interactive REPL loop until quit/EOF."""
        self._running = True
        self.console.print("[bold cyan]Spectre REPL[/bold cyan] — type [dim]/help[/dim] for commands, /quit to exit.")

        completer = FuzzyWordCompleter(_COMMANDS)
        session = _make_prompt_session(completer)
        self._session = session

        while self._running:
            # prompt_async is a real coroutine — await it directly so the event
            # loop stays free and stop() can interrupt between turns.  (Do NOT
            # wrap in asyncio.to_thread: that would freeze Ctrl-C handling.)
            try:
                text = await session.prompt_async()
            except (EOFError, KeyboardInterrupt):
                break

            if not text or not text.strip():
                continue
            stripped = text.strip()

            if stripped.startswith("/"):
                parts = stripped.split(None, 1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else None
                await self._handle_command(agent, cmd, arg)
                if not self._running:
                    break
            else:
                await self._stream_turn(agent, text)

        self.console.print("[dim]Bye.[/dim]")

    # ------------------------------------------------------------------
    # Streaming turn
    # ------------------------------------------------------------------

    async def _stream_turn(self, agent: "Agent", text: str) -> None:
        """Consume one agent.stream_message() and render events to the console.

        - AgentTextDelta: stream live via a Rich Live region (plain text).
        - AgentThinkingDelta: stream live in dim italic above/below text.
        - AgentToolStart/AgentToolResult: status lines before the reply body.
        - AgentDone: final; commit accumulated text as Markdown.
        """
        self._running = True  # keep loop alive during turn

        thinking_acc: list[str] = []
        text_acc: list[str] = []
        total_tool_calls = 0
        stream_exc: Exception | None = None

        from rich.live import Live

        def _frame() -> Text:
            out = Text()
            if thinking_acc:
                out.append("".join(thinking_acc), style="dim italic")
                out.append("\n")
            if text_acc:
                out.append("".join(text_acc))
            return out

        # auto_refresh=False: Rich's Live refresher re-renders the current frame
        # on its own timer AND on every update(), which double-writes each delta
        # into terminal scrollback (visible as duplicated text).  With it off,
        # only our explicit live.update() calls render — one write per event.
        with Live(_frame(), console=self.console, auto_refresh=False) as live:
            try:
                async for ev in agent.stream_message(text):
                    if isinstance(ev, AgentTextDelta):
                        text_acc.append(ev.text)
                        live.update(_frame())
                    elif isinstance(ev, AgentThinkingDelta):
                        thinking_acc.append(ev.text)
                        live.update(_frame())
                    elif isinstance(ev, AgentToolStart):
                        input_preview = ""
                        if ev.input:
                            input_preview = f"({str(ev.input)[:60]})"
                        self.console.print(f"[dim]⚙ {ev.name}{input_preview}[/dim]")
                    elif isinstance(ev, AgentToolResult):
                        if ev.is_error:
                            self.console.print(f"[red]✗ {ev.name} failed[/red]")
                        else:
                            self.console.print(f"[green]✓ {ev.name} done[/green]")
                    elif isinstance(ev, AgentDone):
                        total_tool_calls = ev.tool_calls_executed
            except Exception as exc:
                stream_exc = exc
                self.console.print(f"[red]Stream error: {exc}[/red]")

        # Commit final reply as Markdown (even on partial error)
        last_text = "".join(text_acc)
        if last_text:
            no_color = bool(int(os.environ.get("NO_COLOR", "0")))
            self.console.print(render_markdown(last_text), highlight=not no_color)
        elif not thinking_acc and stream_exc is None:
            self.console.print("[dim](no response)[/dim]")

        if stream_exc is not None:
            self.console.print(f"[red]Turn failed: {stream_exc}[/red]")
        else:
            session_id = getattr(agent, "session_id", "unknown")
            msg_count = len(getattr(agent, "history", []))
            self.console.print(
                f"[dim]— session {session_id[:8]} · {msg_count} msgs · {total_tool_calls} tools —[/dim]"
            )

    # ------------------------------------------------------------------
    # Slash command handlers
    # ------------------------------------------------------------------

    async def _handle_command(self, agent: "Agent", cmd: str, arg: str | None) -> None:
        """Route slash commands to their handlers."""
        handlers = {
            "/quit": self._cmd_quit,
            "/exit": self._cmd_exit,
            "/reset": lambda a: self._cmd_reset(agent),
            "/help": self._cmd_help,
            "/tools": lambda a: self._cmd_tools(agent),
            "/history": lambda a: self._cmd_history(agent),
            "/save": lambda a: self._cmd_save(agent, arg),
            "/load": lambda a: self._cmd_load(agent, arg),
        }
        handler = handlers.get(cmd)
        if handler is None:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            return
        await handler(arg)

    async def _cmd_quit(self, _arg: str | None) -> None:
        self._running = False

    async def _cmd_exit(self, _arg: str | None) -> None:
        self._running = False

    async def _cmd_reset(self, agent: "Agent") -> None:
        agent.new_session()
        self.console.print("[green]New session started[/green]")

    async def _cmd_help(self, _arg: str | None) -> None:
        for line in _HELP.splitlines():
            self.console.print(line)

    async def _cmd_tools(self, agent: "Agent") -> None:
        names = agent.registry.list_names()
        self.console.print(f"Available tools: {', '.join(names)}")

    async def _cmd_history(self, agent: "Agent") -> None:
        summary = format_history_summary(agent.history)
        for line in summary.splitlines():
            self.console.print(line)

    async def _cmd_save(self, agent: "Agent", arg: str | None) -> None:
        path = Path(arg) if arg else default_save_path()
        save_session(agent.history, path)
        self.console.print(f"[green]Session saved to {path}[/green]")

    async def _cmd_load(self, agent: "Agent", arg: str | None) -> None:
        if not arg:
            self.console.print("[red]Usage: /load <path>[/red]")
            return
        path = Path(arg)
        if not path.exists():
            self.console.print(f"[red]Error: file not found: {path}[/red]")
            return
        try:
            messages = load_session(path)
            agent.history = messages
            self.console.print(f"[green]Session loaded from {path} ({len(messages)} messages)[/green]")
        except Exception as e:
            self.console.print(f"[red]Error loading session: {e}[/red]")


__all__ = ["REPLInterface", "render_markdown"]
