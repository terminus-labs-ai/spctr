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

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(
            id="output",
            wrap=True,
            markup=True,
            auto_scroll=True,
        )
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
        """Handle input submission — echo to log for now."""
        text = event.value.strip()
        if not text:
            event.input.value = ""
            return

        # Echo user input to the log
        log = self.query_one("#output", RichLog)
        log.write(f"> {text}")
        event.input.value = ""


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
