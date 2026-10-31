"""Session persistence helpers shared by the TUI and REPL interfaces.

Provides JSON serialization/deserialization of ``agent.history`` (a list of
SR2 ``Message`` objects) plus a human-readable history summary.  Both
``tui.py`` and ``repl.py`` import from here so save/load semantics stay
identical across frontends.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from sr2.pipeline.token_counting import CharacterTokenCounter


def default_save_path() -> Path:
    """Return the default session save path, computed lazily."""
    return Path.home() / ".sr2-spectre" / "session.json"


def serialize_history(history: list) -> list[dict]:
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


def deserialize_history(data: list[dict]) -> list:
    """Deserialize JSON data back into Message objects.

    Reconstructs SR2 Message objects from the serialized format.  Only text
    blocks are restored (tool use/result blocks are dropped on load, matching
    the previous TUI behaviour).
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


def save_session(history: list, path: Path | None = None) -> Path:
    """Serialize ``history`` to JSON at ``path`` (default: default_save_path).

    Creates parent directories as needed.  Returns the path written.
    """
    target = path if path is not None else default_save_path()
    data = serialize_history(history)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2))
    return target


def load_session(path: Path) -> list:
    """Load and deserialize a session file.  Raises on missing/invalid file."""
    data = json.loads(Path(path).read_text())
    return deserialize_history(data)


def format_history_summary(history: list) -> str:
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


class HistoryStats(NamedTuple):
    """Estimated context footprint of an ``agent.history`` list.

    Attributes:
        messages: Number of messages in the history.
        tokens: Estimated token total over every message's content blocks.
        by_role: Token sums grouped by ``msg.role``.
    """

    messages: int
    tokens: int
    by_role: dict[str, int]


# One counter per module is fine: CharacterTokenCounter is stateless and is
# the same zero-dependency counter session.py/runtime.py hand to SR2.
_token_counter = CharacterTokenCounter()


def estimate_history_stats(history: list) -> HistoryStats:
    """Estimate context size over ``agent.history`` in tokens.

    Delegates per-message counting to SR2's ``CharacterTokenCounter``
    (chars // 4 over every content block: text, thinking, tool_use and
    tool_result) so the figure matches the counter the pipeline itself uses.
    This is deliberately NOT LLM-reported usage — real input/output usage
    needs shared-sr2 stream changes (spc-82).

    Messages with missing or empty content contribute 0 and never raise.
    """
    messages = 0
    tokens = 0
    by_role: dict[str, int] = {}
    for msg in history or []:
        messages += 1
        content = getattr(msg, "content", None)
        count = _token_counter.count(content) if content else 0
        tokens += count
        role = getattr(msg, "role", "") or ""
        by_role[role] = by_role.get(role, 0) + count
    return HistoryStats(messages=messages, tokens=tokens, by_role=by_role)


__all__ = [
    "default_save_path",
    "serialize_history",
    "deserialize_history",
    "save_session",
    "load_session",
    "format_history_summary",
    "HistoryStats",
    "estimate_history_stats",
]
