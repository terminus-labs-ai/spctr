"""TUI interface — interactive loop with prompt-toolkit.

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

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import AgentDone, AgentTextDelta, AgentThinkingDelta, AgentToolResult, AgentToolStart

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


class TUIInterface:
    name = "tui"

    def __init__(self) -> None:
        self._running = False
        self._session: PromptSession | None = None

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

    async def run(self, agent: "Agent") -> None:
        """Interactive loop."""
        self._running = True
        session = PromptSession()

        # patch_stdout routes all output through prompt-toolkit's output
        # machinery, preventing keystrokes from echoing into streaming output.
        # Falls back to nullcontext when not connected to a real TTY (e.g. tests).
        stdout_ctx = patch_stdout() if sys.stdout.isatty() else nullcontext()
        with stdout_ctx:
            while self._running:
                # --- prompt ---
                try:
                    user_input = await session.prompt_async("> ")
                except KeyboardInterrupt:
                    print("\nInterrupted.")
                    break
                except EOFError:
                    print("\nEOF.")
                    break

                # --- empty / whitespace ---
                if not user_input or not user_input.strip():
                    continue

                # --- slash commands ---
                stripped = user_input.strip()
                if stripped in ("/quit", "/exit"):
                    print("Goodbye.")
                    break

                if stripped == "/reset":
                    agent.new_session()
                    print("Session reset.")
                    continue

                if stripped == "/help":
                    print(_HELP)
                    continue

                if stripped == "/tools":
                    print(str(agent.registry.list_names()))
                    continue

                if stripped == "/history":
                    print(_format_history_summary(agent.history))
                    continue

                if stripped.startswith("/save"):
                    parts = stripped.split(None, 1)
                    save_path = Path(parts[1]) if len(parts) > 1 else _default_save_path()
                    try:
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        session_data = {
                            "session_id": agent.session_id,
                            "history": _serialize_history(agent.history),
                        }
                        save_path.write_text(json.dumps(session_data, indent=2))
                        print(f"Session saved to {save_path}")
                    except OSError as exc:
                        print(f"Error saving session: {exc}")
                    continue

                if stripped.startswith("/load"):
                    parts = stripped.split(None, 1)
                    if len(parts) < 2:
                        print("Usage: /load <path>")
                        continue
                    load_path = Path(parts[1])
                    if not load_path.exists():
                        print(f"Error: session file not found: {load_path}")
                        continue
                    try:
                        data = json.loads(load_path.read_text())
                        agent.history = _deserialize_history(data.get("history", []))
                        if "session_id" in data:
                            agent.session_id = data["session_id"]
                        print(f"Session loaded from {load_path} ({len(agent.history)} messages)")
                    except (OSError, json.JSONDecodeError, KeyError) as exc:
                        print(f"Error loading session: {exc}")
                    continue

                # --- stream message ---
                try:
                    tool_calls = 0
                    thinking_active = False
                    async for ev in agent.stream_message(stripped):
                        if isinstance(ev, AgentTextDelta):
                            # Close thinking block if we were in one
                            if thinking_active:
                                sys.stdout.write("\n")
                                thinking_active = False
                            sys.stdout.write(ev.text)
                            sys.stdout.flush()
                        elif isinstance(ev, AgentThinkingDelta):
                            # Open thinking block on first chunk
                            if not thinking_active:
                                sys.stdout.write("\n> ")
                                thinking_active = True
                            sys.stdout.write(ev.text)
                            sys.stdout.flush()
                        elif isinstance(ev, AgentToolStart):
                            args_json = json.dumps(ev.input)
                            if len(args_json) > 60:
                                args_preview = args_json[:60] + "..."
                            else:
                                args_preview = args_json
                            print(f"\n⚙ {ev.name}({args_preview})...")
                        elif isinstance(ev, AgentToolResult):
                            if ev.is_error:
                                print(f"✗ {ev.name} failed")
                            else:
                                print(f"✓ {ev.name} done")
                        elif isinstance(ev, AgentDone):
                            print("\n\n", end="")
                            tool_calls = ev.tool_calls_executed

                    # --- status bar ---
                    msg_count = len(agent.history)
                    status = f"[{agent.session_id} | {msg_count} msgs | {tool_calls} tools]"
                    print(f"── {status}\n")

                except KeyboardInterrupt:
                    print("\n[Interrupted]")
                    continue
