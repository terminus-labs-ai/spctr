"""REPL interface — prompt_toolkit + Rich terminal REPL.

A native-terminal replacement for the Textual TUI.  Because it never takes
over the screen (no alt-screen buffer), copy/select, paste and quotes all
work exactly like a normal shell session:

- prompt_toolkit PromptSession gives real readline shortcuts, multiline
  input, persistent history and slash-command completion.
- Enter submits the message and Shift+Enter (or Alt+Enter) inserts a newline,
  which is the inverse of prompt_toolkit's multiline default.
- Rich renders streaming output to stdout: text deltas stream live, thinking
  is dimmed/italicized, tool calls show as status lines, the final reply is
  rendered as Markdown.

Slash commands (identical semantics to the TUI):
  /quit /exit /reset /help /tools /history /save [path] /load <path>

Terminal counterpart of Discord's /model (spc-956):
  /change-model [name]

Ctrl-C during a turn stops the turn (twice in quick succession force-quits);
Ctrl-C at the prompt abandons the half-typed input.

Usage: sr2-spectre config.yaml --interface repl
"""
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, FuzzyWordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
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
  /change-model [name]  — list models, or switch the active one

Multiline input: Enter sends the message, Shift+Enter (or Alt+Enter) adds a
newline.  Shift+Enter only reaches the REPL if your terminal is set to send a
distinct sequence for it; Alt+Enter always works.
Ctrl-C during a turn stops the turn (press twice to force quit).  Ctrl-C at
the prompt abandons the current input.  Ctrl-D or /quit exits.  History is
persisted between runs.
"""

_COMMANDS = [
    "/quit",
    "/exit",
    "/reset",
    "/help",
    "/tools",
    "/history",
    "/save",
    "/load",
    "/change-model",
]


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


def _rows(text: str, width: int) -> int:
    """Terminal rows a plain string occupies at *width* columns."""
    if width < 1:
        width = 1
    return sum(max(1, -(-len(line) // width)) for line in text.split("\n"))


def _tail_rows(text: str, width: int, max_rows: int) -> str:
    """Return the tail of *text* that fits in at most *max_rows* terminal rows.

    A Rich ``Live`` region can only erase itself while every one of its rows is
    still on screen: the escape sequence it emits walks the cursor up N rows and
    clears each one.  Rows that scrolled off the top are unreachable, so a live
    frame taller than the terminal leaves a permanent, unparsed copy of the
    reply in scrollback — which is then printed a second time as Markdown.
    Cropping the frame to a tail that always fits keeps the erase total.
    """
    if width < 1:
        width = 1
    if max_rows < 1:
        return ""
    kept: list[str] = []
    rows = 0
    for line in reversed(text.split("\n")):
        line_rows = max(1, -(-len(line) // width))
        if rows + line_rows > max_rows:
            room = max_rows - rows
            if room > 0:
                kept.append(line[-room * width:])
            break
        kept.append(line)
        rows += line_rows
    kept.reverse()
    return "\n".join(kept)


def _history_file() -> Path:
    """Path to the persistent prompt history file."""
    path = Path.home() / ".sr2-spectre" / "history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class _SlashCompleter(Completer):
    """Slash-command completer gated to slash-prefixed input.

    ``FuzzyWordCompleter`` alone would offer the command list against ANY
    typed word (it fuzzy-matches the current word against the list), so
    typing a normal sentence like ``the `` would pop up a menu of every
    slash command and swallow the keystroke.  This wrapper delegates to the
    fuzzy completer only when the line the cursor sits on begins with ``/``,
    and yields nothing otherwise.
    """

    def __init__(self, words) -> None:
        self._fuzzy = FuzzyWordCompleter(words)

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        # Only the first line matters: slash commands are top-level, and the
        # buffer only grows via backslash continuation, so a non-slash first
        # line means the user is typing a normal message.
        first_line = document.text_before_cursor.rsplit("\n", 1)[-1]
        if not first_line.startswith("/"):
            return
        yield from self._fuzzy.get_completions(document, complete_event)


# Shift+Enter has no encoding of its own in a default terminal: it sends the
# same "\r" as Enter.  The two escape encodings that DO carry the modifier are
# both folded onto Keys.ControlM by prompt_toolkit, which makes them
# indistinguishable from a plain Enter.  Re-point them at ControlJ so the
# newline binding below can claim them when the terminal emits them.
_SHIFT_ENTER_SEQUENCES = (
    "\x1b[27;2;13~",  # xterm modifyOtherKeys=2
    "\x1b[13;2u",  # kitty / CSI-u keyboard protocol
)


def _install_shift_enter_sequences() -> None:
    """Map the Shift+Enter escape sequences to ControlJ.  Idempotent."""
    for sequence in _SHIFT_ENTER_SEQUENCES:
        ANSI_SEQUENCES[sequence] = Keys.ControlJ


_install_shift_enter_sequences()


def _make_key_bindings() -> KeyBindings:
    """Enter submits; Shift+Enter (or Alt+Enter) inserts a newline.

    prompt_toolkit's multiline default is the inverse — Enter inserts and
    Meta+Enter accepts — so both keys are bound explicitly.  A session's own
    bindings are merged after the defaults, and the last match wins, so these
    take precedence.
    """
    kb = KeyBindings()

    @kb.add("c-m")  # Enter
    def _submit(event) -> None:
        buffer = event.current_buffer
        # A highlighted completion takes the keystroke first, otherwise
        # picking a slash command would also fire the message.
        completion = buffer.complete_state and buffer.complete_state.current_completion
        if completion is not None:
            buffer.apply_completion(completion)
            return
        buffer.validate_and_handle()

    @kb.add("c-j")  # Shift+Enter (remapped above), Ctrl+J
    @kb.add("escape", "c-m")  # Alt/Meta+Enter — what most terminals send
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    return kb


def _make_prompt_session(completer):
    """Build the PromptSession.  Factored out so tests can monkeypatch it
    without fighting parameterized generic syntax (PromptSession[str])."""
    return PromptSession(
        message="> ",
        history=FileHistory(str(_history_file())),
        completer=completer,
        # Newlines only reach the buffer through the explicit Shift+Enter
        # binding; multiline=True is what lets the buffer hold them at all.
        multiline=True,
        key_bindings=_make_key_bindings(),
    )


def _active_model_pointer() -> Path | None:
    """Path to the writable active-model pointer file, or None if unset.

    Same env var Discord's /model writes: the config itself is mounted
    read-only, so the pointer file is the only surface that survives a
    restart.
    """
    pointer = os.environ.get("SR2_ACTIVE_MODEL_FILE")
    return Path(pointer) if pointer else None


def render_change_model_command(
    arg: str | None,
    model_names: Iterable[str],
    active_model: str,
    pointer_configured: bool,
) -> tuple[list[tuple[str, str]], str | None]:
    """Pure decision logic for /change-model — no I/O.

    Mirrors ``interfaces.discord.handler.render_model_command``; the REPL
    renders Rich markup lines instead of Discord markdown, so the two stay
    deliberately separate (the duplication is a dozen lines).

    Args:
        arg: Model name after "/change-model"; None lists the models.
        model_names: Keys of the configured ``models`` map.
        active_model: The model name currently in force.
        pointer_configured: Whether ``SR2_ACTIVE_MODEL_FILE`` is set, i.e.
            whether a switch can be persisted across restarts.

    Returns:
        ``(lines, selection)`` — *lines* is a list of ``(markup, style)``
        pairs to print, *selection* is the model name the caller should
        persist and apply, or None when nothing should change (a listing,
        an error, or re-selecting the active model).
    """
    names = set(model_names)
    if arg is None:
        lines: list[tuple[str, str]] = [("[bold]Models:[/bold]", "")]
        for name in sorted(names):
            if name == active_model:
                lines.append((f"  [green]• {name}[/green]  ← active", ""))
            else:
                lines.append((f"  {name}", "dim"))
        return lines, None

    if arg not in names:
        available = ", ".join(sorted(names)) or "(none configured)"
        return (
            [(f"[red]Unknown model: {arg}[/red]", ""), (f"Available: {available}", "dim")],
            None,
        )

    if arg == active_model:
        return ([(f"Already using {arg}.", "dim")], None)

    note = (
        "Not persisted — SR2_ACTIVE_MODEL_FILE is unset, so a restart reverts."
        if not pointer_configured
        else "Takes effect on the next message."
    )
    return ([(f"[green]Switched to {arg}.[/green]", ""), (note, "dim")], arg)


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
        # The REPL's area is the basename of the exact directory the process
        # was launched from — no CLAUDE.md or .git discovery, no ancestor
        # walk. A REPL started in /data/obsidian/topics/harbinger is the
        # "harbinger" area even though /data/obsidian holds CLAUDE.md and a
        # git root. A non-empty SR2_AREA is the only override.
        env_area = os.environ.get("SR2_AREA", "").strip()
        area = env_area or Path.cwd().name
        agent.set_run_context(RunContext(
            interface="repl",
            mode=RunMode.INTERACTIVE,
            source=os.getcwd(),
            area=area,
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

        completer = _SlashCompleter(_COMMANDS)
        session = _make_prompt_session(completer)
        self._session = session

        while self._running:
            # prompt_async is a real coroutine — await it directly so the event
            # loop stays free and stop() can interrupt between turns.  (Do NOT
            # wrap in asyncio.to_thread: that would freeze Ctrl-C handling.)
            try:
                text = await session.prompt_async()
            except EOFError:
                break
            except KeyboardInterrupt:
                # Ctrl-C at the prompt only abandons the half-typed input;
                # /quit or Ctrl-D exits.  (prompt_toolkit consumes Ctrl-C as
                # a key while prompting, so no SIGINT reaches the loop here.)
                self.console.print("[dim]^C[/dim]")
                continue

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
                await self._run_turn(agent, text)

        self.console.print("[dim]Bye.[/dim]")

    # ------------------------------------------------------------------
    # Cancellable turn
    # ------------------------------------------------------------------

    async def _run_turn(self, agent: "Agent", text: str) -> None:
        """Run one streamed turn as a task cancellable with Ctrl-C.

        prompt_toolkit only owns the keyboard while prompting, so there is
        no way to type a /stop command mid-turn.  While the prompt is idle
        the terminal sends Ctrl-C as SIGINT instead, which is handled here:
        the first press cancels the agent turn and returns to the prompt; a
        second press restores Python's default Ctrl-C handling and re-raises,
        tearing the process down — the escape hatch for a wedged tool call.

        ``loop.add_signal_handler`` is POSIX-only.  Where it is unavailable
        (Windows), the handler install is skipped and a KeyboardInterrupt
        escaping the await still cancels the turn; if it escapes the loop
        entirely, Ctrl-C behaves as it always did there: hard exit.
        """
        turn = asyncio.create_task(self._stream_turn(agent, text))
        loop = asyncio.get_running_loop()
        previous_handler = signal.getsignal(signal.SIGINT)
        stop_requested = False

        def _on_sigint() -> None:
            nonlocal stop_requested
            if stop_requested:
                # Second press: force quit.  Restore the default interrupt
                # handler *before* re-raising so asyncio.run still cancels
                # and drains pending tasks instead of dying on SIG_DFL.
                signal.signal(signal.SIGINT, signal.default_int_handler)
                signal.raise_signal(signal.SIGINT)
                return
            stop_requested = True
            self.console.print(
                "[yellow]⏹ Stopping turn…[/yellow] "
                "[dim](Ctrl-C again to force quit)[/dim]"
            )
            turn.cancel()

        installed = False
        try:
            loop.add_signal_handler(signal.SIGINT, _on_sigint)
            installed = True
        except (NotImplementedError, RuntimeError, ValueError):
            pass

        try:
            await turn
        except asyncio.CancelledError:
            # _stream_turn already committed the partial text.
            self.console.print("[yellow]Turn stopped.[/yellow]")
        except KeyboardInterrupt:
            # Fallback for when the loop signal handler could not be
            # installed and the interrupt landed in this coroutine instead.
            if not turn.done():
                turn.cancel()
                try:
                    await turn
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
            self.console.print("[yellow]Turn stopped.[/yellow]")
        finally:
            if installed:
                loop.remove_signal_handler(signal.SIGINT)
                # remove_signal_handler restores SIG_DFL; put back whatever
                # disposition the process had before the turn (normally
                # Python's default KeyboardInterrupt handler).
                signal.signal(signal.SIGINT, previous_handler)

    # ------------------------------------------------------------------
    # Streaming turn
    # ------------------------------------------------------------------

    async def _stream_turn(self, agent: "Agent", text: str) -> None:
        """Consume one agent.stream_message() and render events to the console.

        Rendering model — every byte the model produces reaches scrollback
        exactly once, as Markdown:

        - Text and thinking deltas stream into a transient Rich ``Live``
          region, cropped to a tail that always fits on screen (see
          ``_tail_rows``).  That region is scratch: it is erased, never
          committed.
        - A turn can span several LLM roundtrips (narration -> tool -> answer).
          Each round's text is flushed to scrollback as Markdown at the moment
          the round ends — when its first tool call starts, or when the stream
          finishes — so interim narration is preserved in reading order
          instead of being discarded or re-printed.
        - Tool calls show as status lines between the committed rounds.
        """
        self._running = True  # keep loop alive during turn

        thinking_acc: list[str] = []
        text_acc: list[str] = []
        total_tool_calls = 0
        stream_exc: Exception | None = None
        committed_any = False

        from rich.live import Live

        def _frame() -> Text:
            """The live scratch frame: a tail of thinking + text that fits on screen.

            Two rows of headroom below the cap keep the frame strictly shorter
            than the terminal even as the prompt and status lines move around,
            so Rich's erase walk can always reach every row it drew.
            """
            size = self.console.size
            width = size.width
            max_rows = max(3, size.height - 4)

            body = _tail_rows("".join(text_acc), width, max_rows) if text_acc else ""
            remaining = max_rows - _rows(body, width) if body else max_rows

            out = Text()
            if thinking_acc and remaining > 1:
                head = _tail_rows("".join(thinking_acc), width, remaining - 1)
                if head:
                    out.append(head, style="dim italic")
                    out.append("\n")
            if body:
                out.append(body)
            return out

        def _commit(live: "Live") -> None:
            """Flush the round's accumulated text to scrollback as Markdown.

            The live region is emptied *first* so the console.print below draws
            into a cleared area — Rich erases the previous frame, prints the
            committed Markdown above, then redraws the (now empty) frame.
            """
            nonlocal committed_any
            body = "".join(text_acc)
            text_acc.clear()
            thinking_acc.clear()
            live.update(_frame())
            if body.strip():
                no_color = bool(int(os.environ.get("NO_COLOR", "0")))
                self.console.print(render_markdown(body), highlight=not no_color)
                committed_any = True

        # transient=True: the scratch frame erases itself on exit.  Combined
        # with the row cap in _frame() that erase is always complete, which is
        # what keeps the streamed (unparsed) copy out of scrollback.
        with Live(_frame(), console=self.console, transient=True) as live:
            try:
                async for ev in agent.stream_message(text):
                    if isinstance(ev, AgentTextDelta):
                        text_acc.append(ev.text)
                        live.update(_frame())
                    elif isinstance(ev, AgentThinkingDelta):
                        thinking_acc.append(ev.text)
                        live.update(_frame())
                    elif isinstance(ev, AgentToolStart):
                        # The round is over: its text is the narration that led
                        # to this call, so commit it above the status line.
                        _commit(live)
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
            except asyncio.CancelledError:
                # Ctrl-C stopped this turn (see _run_turn): commit whatever
                # already streamed so scrollback keeps the partial answer,
                # then unwind.
                _commit(live)
                raise
            except Exception as exc:
                stream_exc = exc
                self.console.print(f"[red]Stream error: {exc}[/red]")

            # Whatever the last round produced — including a partial round cut
            # short by the error above — is committed here, inside the Live so
            # the scratch frame is cleared before the Markdown lands.
            had_thinking = bool(thinking_acc)
            _commit(live)

        if not committed_any and not had_thinking and stream_exc is None:
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
            "/change-model": lambda a: self._cmd_change_model(agent, arg),
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

    async def _cmd_change_model(self, agent: "Agent", arg: str | None) -> None:
        """Handle /change-model — list models or switch the active one.

        Terminal counterpart of Discord's /model (spc-956).  A switch is
        applied in-process through ``agent.apply_config`` — the Runtime
        re-seats the LLM via ``LiveLLM.retarget`` — and persisted, when
        possible, by writing the pointer file *first*: a failed write aborts
        the switch so the pointer and the in-process model never diverge.
        """
        config = getattr(agent, "config", None)
        models = getattr(config, "models", None)
        if not models:
            self.console.print("[red]No models are configured.[/red]")
            return

        pointer = _active_model_pointer()
        lines, selection = render_change_model_command(
            arg, set(models), getattr(config, "active_model", ""), pointer is not None
        )
        if selection is None:
            # A listing, an error, or a no-op: print and stop.
            for markup, style in lines:
                self.console.print(markup, style=style or None)
            return

        if pointer is not None:
            # Persist first: a failed write must leave the in-process model
            # untouched, so the pointer and the running session never diverge.
            try:
                pointer.write_text(f"{selection}\n")
            except OSError as exc:
                self.console.print(
                    f"[red]Could not persist the switch: {exc}[/red] "
                    "[dim]Active model unchanged.[/dim]"
                )
                return

        new_config = config.model_copy(update={"active_model": selection})
        try:
            agent.apply_config(new_config)
        except Exception as exc:
            self.console.print(f"[red]Could not apply the switch: {exc}[/red]")
            return
        for markup, style in lines:
            self.console.print(markup, style=style or None)


__all__ = ["REPLInterface", "render_markdown"]
