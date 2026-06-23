"""TUI interface — full-screen Textual app.

Phase 4 polish features:
- Streaming output (AgentTextDelta rendered in real time)
- Tool execution visibility (name/args preview, success/failure)
- Slash commands: /quit, /exit, /reset, /help, /tools, /history, /save, /load
- Session save/load (JSON serialization of agent.history)
- Status bar after each turn (session ID, message count, tool count)
- Markdown rendering via rich (falls back gracefully if rich is unavailable)

Usage: sr2-spectre config.yaml --interface tui

Note: positional arguments must come before options.  The TUI does not
require a prompt — it reads input interactively.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import (
    AgentDone,
    AgentEvent,
    AgentThinkingDelta,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
)

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent

_HELP = """\
Commands:
  /quit     — exit the TUI
  /exit     — exit the TUI
  /reset    — start a new session
  /help     — show this help
  /tools    — list available tools
  /history  — show conversation history
  /save [path]  — save session to JSON (default: ~/.sr2-spectre/session.json)
  /load [path]  — load session from JSON
"""


def _default_save_path() -> Path:
    """Return the default session save path, computed lazily."""
    return Path.home() / ".sr2-spectre" / "session.json"


def _render_markdown(text: str) -> str:
    """Render markdown text using rich, falling back to raw text.

    Rich's Markdown class can render to a string via its console rendering.
    We use it here to produce formatted output for the TUI.
    """
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        md = Markdown(text)
        # Capture rich output to a string
        import io
        output = io.StringIO()
        render_console = Console(file=output, width=80, force_terminal=True, no_color=True)
        render_console.print(md)
        return output.getvalue()
    except Exception:
        # Rich unavailable or rendering failed — return raw text
        return text


def _serialize_history(history: list) -> list[dict]:
    """Serialize agent.history (list[Message]) to plain dicts for JSON storage.

    Message objects from SR2 have role and content attributes where content
    is a list of blocks (TextBlock, ToolUseBlock, ToolResultBlock).
    """
    serialized = []
    for msg in history:
        msg_dict: dict = {"role": msg.role, "content": []}
        if hasattr(msg, "content") and msg.content:
            for block in msg.content:
                block_dict: dict = {}
                if hasattr(block, "type"):
                    block_dict["type"] = block.type
                if hasattr(block, "text"):
                    block_dict["text"] = block.text
                if hasattr(block, "name"):
                    block_dict["name"] = block.name
                if hasattr(block, "input") and block.input is not None:
                    block_dict["input"] = block.input
                if hasattr(block, "id"):
                    block_dict["id"] = block.id
                if block_dict:
                    msg_dict["content"].append(block_dict)
        serialized.append(msg_dict)
    return serialized


def _deserialize_history(data: list[dict]) -> list:
    """Deserialize JSON data back into Message objects.

    Reconstructs SR2 Message objects from the serialized format.
    """
    from sr2.models import Message, TextBlock

    messages = []
    for msg_dict in data:
        role = msg_dict.get("role", "user")
        content = []
        for block_dict in msg_dict.get("content", []):
            block_type = block_dict.get("type", "text")
            if block_type == "text" or "text" in block_dict:
                text = block_dict.get("text", "")
                if text:
                    content.append(TextBlock(text=text))
        messages.append(Message(role=role, content=content))
    return messages


def _format_history_summary(history: list) -> str:
    """Format conversation history as a readable summary string.

    Shows message count and last N messages truncated for readability.
    """
    if not history:
        return "No conversation history."

    lines = [f"History ({len(history)} messages):"]
    lines.append("-" * 40)

    for i, msg in enumerate(history):
        role = msg.role.upper()
        # Extract text content from message
        text_parts = []
        if hasattr(msg, "content") and msg.content:
            for block in msg.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
        text = " ".join(text_parts)
        # Truncate long messages
        if len(text) > 200:
            text = text[:200] + "..."
        prefix = f"[{role}]"
        if text:
            lines.append(f"  {prefix} {text}")
        else:
            lines.append(f"  {prefix} (non-text content)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Textual App
# ---------------------------------------------------------------------------

class SpectreTUI(App):
    """Full-screen Textual app for the SR2 Spectre TUI.

    Layout (top to bottom):
    - Header (fixed)
    - RichLog output pane (scrollable, fills remaining space)
    - Thinking region (live-updating, dim/italic, ephemeral)
    - Text region (live-updating, committed as markdown on done)
    - Input box (fixed height at bottom)
    - Status bar (fixed, shows session info)
    - Footer (fixed, shows bindings)
    """

    CSS = """
    Screen {
        layout: vertical;
    }

    #output {
        width: 100%;
        height: 1fr;
        overflow-y: scroll;
    }

    #thinking {
        width: 100%;
        height: auto;
        color: $accent;
        text-style: italic;
        opacity: 0.6;
        display: none;
    }

    #thinking.visible {
        display: block;
    }

    #text {
        width: 100%;
        height: auto;
        display: none;
    }

    #text.visible {
        display: block;
    }

    #prompt {
        width: 100%;
        dock: bottom;
    }

    #status {
        width: 100%;
        height: 1;
        dock: bottom;
        background: $boost;
        color: $text;
        content-align: center middle;
    }

    Footer {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent: "Agent") -> None:
        super().__init__()
        self.agent = agent
        self._tool_calls: int = 0
        self._streaming: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(
            id="output",
            wrap=True,
            markup=True,
            auto_scroll=True,
        )
        yield Static(id="thinking")
        yield Static(id="text")
        yield Input(id="prompt", placeholder="> ")
        yield Static(id="status", content="")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input on launch."""
        self.query_one("#prompt", Input).focus()

    def update_status(self, session_id: str, msg_count: int, tool_count: int) -> None:
        """Update the status bar text."""
        status = self.query_one("#status", Static)
        status.update(f"{session_id} | {msg_count} msgs | {tool_count} tools")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission — dispatch to agent or slash commands."""
        text = event.value.strip()
        if not text:
            event.input.value = ""
            return

        log = self.query_one("#output", RichLog)
        log.write(f"> {text}")
        event.input.value = ""

        # Slash command dispatch
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None
            await self._handle_command(cmd, arg, log)
            return

        # Non-slash input → agent (exclusive worker)
        self._run_stream_worker(text)

    def _run_stream_worker(self, text: str) -> None:
        """Launch the streaming worker with exclusive=True.

        The worker consumes agent.stream_message() and renders events
        to the live regions and RichLog.
        """
        self._stream_worker = self.run_worker(
            self._stream_and_render(text),
            name="stream",
            exclusive=True,
        )

    async def _stream_and_render(self, text: str) -> None:
        """Worker: stream agent events and render to live regions.

        - AgentTextDelta: accumulate in #text live region
        - AgentThinkingDelta: accumulate in #thinking live region (dim/italic)
        - AgentToolStart/AgentToolResult: render to RichLog
        - AgentDone: commit accumulated text as markdown to RichLog,
          clear both live regions, re-enable input
        """
        log = self.query_one("#output", RichLog)
        thinking_region = self.query_one("#thinking", Static)
        text_region = self.query_one("#text", Static)
        inp = self.query_one("#prompt", Input)

        # Disable input during streaming
        self._streaming = True
        inp.disabled = True

        text_acc: list[str] = []
        thinking_acc: list[str] = []
        total_tool_calls = 0

        try:
            async for ev in self.agent.stream_message(text):
                if isinstance(ev, AgentTextDelta):
                    text_acc.append(ev.text)
                    text_region.update("".join(text_acc))
                    text_region.add_class("visible")

                elif isinstance(ev, AgentThinkingDelta):
                    thinking_acc.append(ev.text)
                    thinking_region.update("".join(thinking_acc))
                    thinking_region.add_class("visible")

                elif isinstance(ev, AgentToolStart):
                    input_preview = ""
                    if ev.input:
                        # Show first 80 chars of input
                        inp_str = str(ev.input)[:80]
                        input_preview = f" ({inp_str})"
                    log.write(f"[dim]⏳ {ev.name}{input_preview}[/dim]")

                elif isinstance(ev, AgentToolResult):
                    if ev.is_error:
                        log.write(f"[red]✗ {ev.name}[/red]")
                    else:
                        log.write(f"[green]✓ {ev.name}[/green]")

                elif isinstance(ev, AgentDone):
                    total_tool_calls = ev.tool_calls_executed
        except Exception as exc:
            log.write(f"[red]Stream error: {exc}[/red]")
        finally:
            # Commit accumulated text as markdown to RichLog
            last_text = "".join(text_acc)
            if last_text:
                rendered = _render_markdown(last_text)
                log.write(rendered)

            # Clear live regions
            text_region.update("")
            text_region.remove_class("visible")
            thinking_region.update("")
            thinking_region.remove_class("visible")

            # Update status
            session_id = getattr(self.agent, "session_id", "unknown")
            msg_count = len(getattr(self.agent, "history", []))
            self.update_status(session_id, msg_count, total_tool_calls)

            # Re-enable input
            self._streaming = False
            inp.disabled = False
            inp.focus()

    async def _handle_command(
        self, cmd: str, arg: str | None, log: RichLog
    ) -> None:
        """Route slash commands to their handlers."""
        handlers = {
            "/quit": self._cmd_quit,
            "/exit": self._cmd_exit,
            "/reset": self._cmd_reset,
            "/help": self._cmd_help,
            "/tools": self._cmd_tools,
            "/history": self._cmd_history,
            "/save": self._cmd_save,
            "/load": self._cmd_load,
        }
        handler = handlers.get(cmd)
        if handler is None:
            log.write(f"[red]Unknown command: {cmd}[/red]")
            return
        await handler(arg, log)

    # ------------------------------------------------------------------
    # Slash command handlers
    # ------------------------------------------------------------------

    async def _cmd_quit(self, _arg: str | None, log: RichLog) -> None:
        self.exit()

    async def _cmd_exit(self, _arg: str | None, log: RichLog) -> None:
        self.exit()

    async def _cmd_reset(self, _arg: str | None, log: RichLog) -> None:
        self.agent.new_session()
        log.write("[green]New session started[/green]")

    async def _cmd_help(self, _arg: str | None, log: RichLog) -> None:
        log.write(_HELP)

    async def _cmd_tools(self, _arg: str | None, log: RichLog) -> None:
        names = self.agent.registry.list_names()
        log.write(f"Available tools: {', '.join(names)}")

    async def _cmd_history(self, _arg: str | None, log: RichLog) -> None:
        summary = _format_history_summary(self.agent.history)
        log.write(summary)

    async def _cmd_save(self, arg: str | None, log: RichLog) -> None:
        path = Path(arg) if arg else _default_save_path()
        data = _serialize_history(self.agent.history)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        log.write(f"[green]Session saved to {path}[/green]")

    async def _cmd_load(self, arg: str | None, log: RichLog) -> None:
        if not arg:
            log.write("[red]Usage: /load <path>[/red]")
            return
        path = Path(arg)
        if not path.exists():
            log.write(f"[red]Error: file not found: {path}[/red]")
            return
        try:
            data = json.loads(path.read_text())
            messages = _deserialize_history(data)
            self.agent.history = messages
            log.write(f"[green]Session loaded from {path} ({len(messages)} messages)[/green]")
        except Exception as e:
            log.write(f"[red]Error loading session: {e}[/red]")


# ---------------------------------------------------------------------------
# TUIInterface (unchanged protocol: name / start / stop / run)
# ---------------------------------------------------------------------------

class TUIInterface:
    name = "tui"

    def __init__(self) -> None:
        self._running = False
        self._app: SpectreTUI | None = None

    async def start(self, agent: "Agent") -> None:
        """Initialize TUI and set interactive run context."""
        self._running = True
        agent.set_run_context(RunContext(
            interface="tui",
            mode=RunMode.INTERACTIVE,
            source=os.getcwd(),
        ))

    async def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False
        if self._app is not None:
            self._app.exit()

    async def run(self, agent: "Agent") -> None:
        """Launch the Textual app.

        Uses run_async() for proper async integration.  When stdout is
        not a TTY (tests, pipes, CI) the app runs in headless mode so
        it does not crash.
        """
        self._running = True
        self._app = SpectreTUI(agent)

        headless = not sys.stdout.isatty()
        await self._app.run_async(headless=headless)
