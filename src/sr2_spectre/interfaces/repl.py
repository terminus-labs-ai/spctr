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
  /quit /exit /reset /help /tools /history /context /save [path] /load <path>
  /resume [| <idx> | <session-id>]

Terminal counterpart of Discord's /model (spc-956):
  /change-model [name]

Ctrl-C during a turn stops the turn (twice in quick succession force-quits);
Ctrl-C at the prompt abandons the half-typed input.

Usage: sr2-spectre config.yaml --interface repl
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
from datetime import datetime
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
from sr2_spectre.resume import load_resumable_history
from sr2_spectre.events import (
    AgentDone,
    AgentThinkingDelta,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
)
from sr2_spectre.interfaces.session_io import (
    default_save_path,
    estimate_history_stats,
    format_history_summary,
    load_session,
    save_session,
)

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent

logger = logging.getLogger(__name__)

# Rows a bare /resume shows (FR 12).
_RESUME_LIST_LIMIT = 15

_HELP = """\
Commands:
  /quit     — exit the REPL
  /exit     — exit the REPL
  /reset    — start a new session
  /help     — show this help
  /tools    — list available tools
  /history  — show conversation history
  /context  — show session id, message count and estimated context tokens
  /save [path]  — save session to JSON (default: ~/.sr2-spectre/session.json)
  /load <path>  — load session from JSON
  /resume [| <idx> | <session-id>]  — list archived sessions, or take one over
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
    "/context",
    "/save",
    "/load",
    "/resume",
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


def _mint_repl_key(name: str, now: datetime | None = None) -> str:
    """Fresh unique REPL session key: ``<name>-repl-<YYYYmmdd-HHMMSS>`` (FR 9).

    Local time, as the listing shows local time.  Uniqueness comes from the
    clock: one mint per REPL start (and one per ``/reset``) can only collide
    with another inside the same second.
    """
    moment = now if now is not None else datetime.now()
    return f"{name}-repl-{moment.strftime('%Y%m%d-%H%M%S')}"


def _needs_repl_key(session_id: str, name: str) -> bool:
    """True when ``session_id`` is still the constructed default identity.

    Only ``<name>-default`` — the identity ``Agent`` gives a session with no
    explicit ``--session-id`` — is replaced by a unique key.  An explicit
    ``--session-id`` survives start unchanged (FR 9).
    """
    return session_id == f"{name}-default"


def _is_repl_minted_key(session_id: str, name: str) -> bool:
    """True when ``session_id`` was minted by this interface (``/reset``).

    ``/reset`` replaces a key the REPL minted (``<name>-repl-<stamp>``) with a
    fresh stamp; a key the operator supplied keeps the shipped fallback.
    """
    return re.match(rf"^{re.escape(name)}-repl-\d{{8}}-\d{{6}}$", session_id) is not None


def render_session_listing(
    summaries: list, current_session_id: str, limit: int = 15
) -> tuple[list[tuple[str, str]], list[str]]:
    """Pure listing logic for bare ``/resume`` — no I/O (FR 12).

    Args:
        summaries: ``SessionSummary`` rows from ``list_sessions``, newest
            activity first (the store's own order).
        current_session_id: The agent's current key, marked on its row.
        limit: Row cap, applied here so the listing and the index the
            ``/resume <idx>`` form resolves share one row set.

    Returns:
        ``(lines, rows)`` — *lines* is a list of ``(markup, style)`` pairs
        (a header, one row per session as
        ``<idx>  <YYYY-MM-DD HH:MM>  <turns>  <session-id>``, and a footer
        naming the selection syntax); *rows* holds each session row as
        plain text for assertions.
    """
    lines: list[tuple[str, str]] = [("[bold]Sessions (newest first):[/bold]", "")]
    rows: list[str] = []
    for idx, summary in enumerate(summaries[:limit], 1):
        # A naive-local last_activity (the SQLite round-trip) is already this
        # machine's local time; a UTC-aware one (the InMemory store) converts.
        # Either way the row shows local time.
        last = summary.last_activity
        local = last if last.tzinfo is None else last.astimezone(tz=None)
        current = summary.session_id == current_session_id
        mark = "  → current" if current else ""
        style = "" if current else "dim"
        row = (
            f"{idx}  {local.strftime('%Y-%m-%d %H:%M')}  {summary.turn_count}"
            f"  {summary.session_id}{mark}"
        )
        rows.append(row)
        lines.append((f"  {row}", style))
    lines.append(("[dim]/resume <idx> or /resume <session-id> to take over[/dim]", ""))
    return lines, rows


def resolve_resume_target(
    summaries: list | None, arg: str, limit: int = 15
) -> tuple[str | None, str | None]:
    """Pure selection logic for ``/resume <idx|session-id>`` — no I/O (FR 11).

    An all-digits argument is a 1-based index into *summaries* (the rows the
    last listing fetched; None when nothing has been listed yet).  Anything
    else is an exact session id and resolves directly — no listing is needed
    and none is consulted, so an exact key works without a prior bare
    ``/resume`` and the target's archive is read exactly once.

    Returns:
        ``(session_id, error)`` — exactly one is non-None.
    """
    if arg.isdigit():
        listed = summaries or []
        idx = int(arg)
        if not 1 <= idx <= min(len(listed), limit):
            return None, (
                f"No session at index {arg} — {len(listed)} listed. Run /resume."
            )
        return listed[idx - 1].session_id, None
    return arg, None


def _max_archived_turn(history: list) -> int:
    """Highest turn number a reconstructed history covers (pair-count - 1).

    The reconstruction emits exactly two messages per archived turn, so the
    pair count IS the turn count; an empty history means no archived turn
    (turn_seq -1, the engine's pre-first-turn state).
    """
    return max(0, len(history) // 2) - 1


def render_takeover_line(
    session_id: str, message_count: int, interrupted: bool, restore_failed: bool
) -> tuple[str, str]:
    """One console line for a completed takeover: key, count, state (FR 11)."""
    state = "restore failed" if restore_failed else ("interrupted turn" if interrupted else "ok")
    return (
        f"[green]Resumed {session_id}[/green] — {message_count} messages restored ({state}).",
        "",
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
        # Rows fetched by the last bare /resume, so /resume <idx> refers to
        # the listing the operator just looked at; /reset clears it.
        self._listing: list = []

    # ------------------------------------------------------------------
    # Interface protocol
    # ------------------------------------------------------------------

    async def start(self, agent: "Agent") -> None:
        """Initialize REPL state, mint a unique session key, set run context."""
        self._running = True
        # A REPL run owns its archive key: on the constructed default identity
        # (<name>-default) the run switches to a fresh <name>-repl-<stamp>
        # before its first turn, so this run archives under a key written by
        # exactly one writer and past runs' archives stay untouched (FR 9).
        # An explicit --session-id is kept as-is.
        await self._collision_free_start_key(agent)
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
            est_tokens = estimate_history_stats(getattr(agent, "history", [])).tokens
            self.console.print(
                f"[dim]— session {session_id[:8]} · {msg_count} msgs · {total_tool_calls} tools"
                f" · ~{est_tokens} tok est. —[/dim]"
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
            "/context": lambda a: self._cmd_context(agent),
            "/save": lambda a: self._cmd_save(agent, arg),
            "/load": lambda a: self._cmd_load(agent, arg),
            "/resume": lambda a: self._cmd_resume(agent, arg),
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
        # A key matching <name>-repl-\d{8}-\d{6} is this interface's own
        # mint, so /reset replaces it with a fresh stamp (FR 9) — never the
        # shared <name>-default.  A stamp cannot be reused inside its own
        # second: the key in force may already have archived a turn under
        # exactly that stamp since the mint, so the fresh mint is checked
        # against the archive.  A key the operator supplied (--session-id)
        # is left to the shipped fallback.
        config = getattr(agent, "config", None)
        name = getattr(getattr(config, "agent", None), "name", None)
        previous = getattr(agent, "session_id", "")
        key = None
        if isinstance(name, str) and name and _is_repl_minted_key(previous, name):
            key = _mint_repl_key(name)
            store = getattr(agent, "provenance_store", None)
            if store is not None:
                try:
                    archived = {
                        s.session_id
                        for s in await self._list_sessions(store, _RESUME_LIST_LIMIT)
                    }
                    while key in archived:
                        key = _mint_repl_key(name)
                except Exception as exc:
                    logger.warning("Key-collision probe skipped, store unreadable: %s", exc)
        # No key minted (operator-supplied identity, or no name in config):
        # the shipped fallback — new_session with no argument resets to
        # <name>-default.
        if key is None:
            agent.new_session()
        else:
            agent.new_session(key)
        self._listing = []
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

    async def _cmd_context(self, agent: "Agent", _arg: str | None = None) -> None:
        """Handle /context — estimated context size for the current session.

        Read-only.  The token figure is CharacterTokenCounter over
        ``agent.history`` (see ``estimate_history_stats``), not LLM-reported
        usage; extra arguments are ignored.
        """
        session_id = getattr(agent, "session_id", "unknown")
        stats = estimate_history_stats(agent.history)
        self.console.print(
            f"Session {session_id[:8]} · {stats.messages} msgs · ~{stats.tokens} tokens est."
        )
        if stats.messages > 0:
            by_role = " · ".join(
                f"{role} ~{tokens}"
                for role, tokens in sorted(
                    stats.by_role.items(), key=lambda item: (-item[1], item[0])
                )
            )
            self.console.print(f"By role: {by_role}")

    # ------------------------------------------------------------------
    # Session identity and /resume
    # ------------------------------------------------------------------

    def _next_key(self, agent: "Agent") -> str | None:
        """Unique key this agent should move to, or None to keep its own.

        Only the constructed default identity (<name>-default) is replaced:
        an explicit --session-id is the operator's own choice and survives
        untouched (FR 9).
        """
        config = getattr(agent, "config", None)
        name = getattr(getattr(config, "agent", None), "name", None)
        if not name:
            return None
        session_id = getattr(agent, "session_id", "")
        if not _needs_repl_key(session_id, name):
            return None
        return _mint_repl_key(name)

    async def _collision_free_start_key(self, agent: "Agent") -> None:
        """Switch an agent still on its constructed identity to a unique key.

        Sub-second restarts collide on the stamp, so the fresh mint is checked
        against the archive (the contract is async; this runs inside
        ``start()``) and moves to the next second while the collision holds.
        """
        key = self._next_key(agent)
        if key is None:
            return
        store = getattr(agent, "provenance_store", None)
        if store is not None:
            try:
                archived = {
                    s.session_id
                    for s in await self._list_sessions(store, _RESUME_LIST_LIMIT)
                }
                while key in archived:
                    key = _mint_repl_key(agent.config.agent.name)
            except Exception as exc:
                logger.warning("Key-collision probe skipped, store unreadable: %s", exc)
        # The setter swaps in a fresh Session under the new identity.
        # Its archive (none: the key was just minted) stays untouched.
        agent.session_id = key

    async def _list_sessions(self, store, limit: int) -> list:
        """Fetch archived-session summaries through the store contract (FR 10)."""
        from sr2.pipeline.provenance import TURN_ARCHIVE_LAYER

        return await store.list_sessions(TURN_ARCHIVE_LAYER, limit)

    async def _cmd_resume(self, agent: "Agent", arg: str | None) -> None:
        """Handle /resume — list archived sessions, or take one over.

        Bare: list up to 15 archived sessions, newest first, current key
        marked.  ``<idx>``: select from that listing.  ``<session-id>``:
        select by exact key without a prior listing.  A takeover restores the
        target's resumable history (interrupted marker and rewinds included —
        the increment-1 reconstruction), adopts the key as this agent's
        identity, and reads archive continuation for the adopted Session off
        the archive, so the next turn continues the target's turn numbering
        and this run's previous key receives no further entries (FR 11).
        """
        store = getattr(agent, "provenance_store", None)
        if store is None:
            self.console.print(
                "[red]No provenance store configured — cannot list or resume sessions.[/red]"
            )
            return

        if arg:
            # The indexed form resolves against the shared listing cache —
            # the rows bare /resume last showed.  With nothing listed yet it
            # fetches once (an index implies the list the operator expects
            # to pick from) and stores it as the shared listing.  An exact
            # session id needs no listing at all; its archive read is the
            # single store access of the takeover.
            listing = self._listing
            if listing and arg.isdigit():
                summaries = listing
            else:
                try:
                    summaries = await self._list_sessions(store, _RESUME_LIST_LIMIT)
                except Exception as exc:
                    self.console.print(f"[red]Could not list archived sessions: {exc}[/red]")
                    return
                if arg.isdigit():
                    self._listing = list(summaries[:_RESUME_LIST_LIMIT])
            session_id, error = resolve_resume_target(summaries, arg, _RESUME_LIST_LIMIT)
            if error is not None:
                self.console.print(f"[red]{error}[/red]")
                return
            await self._takeover(agent, store, session_id)
            return

        try:
            summaries = await self._list_sessions(store, _RESUME_LIST_LIMIT)
        except Exception as exc:
            self.console.print(f"[red]Could not list archived sessions: {exc}[/red]")
            return
        if not summaries:
            self.console.print("[dim]No archived sessions found.[/dim]")
            return
        self._listing = list(summaries[:_RESUME_LIST_LIMIT])
        lines, _rows = render_session_listing(
            summaries, getattr(agent, "session_id", ""), _RESUME_LIST_LIMIT
        )
        for markup, style in lines:
            self.console.print(markup, style=style or None)

    async def _takeover(self, agent: "Agent", store, session_id: str) -> None:
        """Restore one archived session's history, then adopt its key.

        The restore reads the target's archive *before* the identity switch
        through the increment-1 reconstruction, so an unknown or empty key
        fails without touching the running conversation (FR 11).  Archive
        continuation deliberately stays deferred: the adopted Session runs
        its existing first-turn ``continue_archive``, which then reads the
        adopted key and numbers the next turn above the archived maximum.
        """
        from sr2_spectre.config import ResumeConfig

        resume_config = getattr(
            getattr(agent, "config", None), "agent", None
        )
        resume_config = getattr(resume_config, "resume", None) or ResumeConfig()
        try:
            result = await load_resumable_history(store, session_id, resume_config)
        except Exception:  # defensive: the loader already traps its reads
            self.console.print(f"[red]Could not restore {session_id}.[/red]")
            return
        if result.restore_failed:
            # Distinct from an empty archive so the line names the actual
            # condition (a read failure never claims "no conversation").
            self.console.print(
                f"[red]Could not read the archive for {session_id}[/red] "
                "[dim](restore failed, empty history) — no switch.[/dim]"
            )
            return
        if not result.history:
            self.console.print(f"[red]No archived conversation for {session_id}.[/red]")
            return

        agent.session_id = session_id
        agent.history[:] = result.history
        # Continuation state belongs to the identity that was just replaced.
        # The identity switch itself replaces the Session (fresh SR2, nothing
        # seeded); re-seat the adopted key's turn number at the archived max
        # (turns 0..N, so the next turn is N+1) — exactly what this key's
        # first-turn ``continue_archive`` would have seeded.
        agent.sr2._engine.seed_turn_seq(_max_archived_turn(result.history))
        self._listing = []
        markup, style = render_takeover_line(
            session_id,
            len(result.history),
            interrupted=result.interrupted,
            restore_failed=result.restore_failed,
        )
        self.console.print(markup, style=style or None)

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
