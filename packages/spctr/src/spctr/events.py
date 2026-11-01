"""Streaming event types for Agent.stream_message().

Each event is a dataclass inheriting from AgentEvent so that
isinstance(ev, AgentEvent) works for all concrete types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


class AgentEvent:
    """Base class for all streaming agent events.

    Not a dataclass itself — subclasses are dataclasses.  Kept as a plain
    class so isinstance(ev, AgentEvent) works without dataclass field
    inheritance complications.
    """


@dataclass
class AgentTextDelta(AgentEvent):
    """Incremental text fragment from the LLM."""
    type: str = field(default="text_delta", init=False)
    text: str = ""


@dataclass
class AgentThinkingDelta(AgentEvent):
    """Incremental thinking/reasoning fragment from the LLM.

    Emitted when the model uses extended thinking (Claude thinking blocks,
    OpenAI reasoning_content). Rendered visually distinct from regular text
    so the user can see the model's reasoning process.
    """
    type: str = field(default="thinking_delta", init=False)
    text: str = ""


@dataclass
class AgentToolStart(AgentEvent):
    """Emitted before a tool begins executing."""
    type: str = field(default="tool_start", init=False)
    tool_id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class AgentToolResult(AgentEvent):
    """Emitted after a tool finishes (success or error)."""
    type: str = field(default="tool_result", init=False)
    tool_id: str = ""
    name: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class AgentDone(AgentEvent):
    """Emitted once, as the final event in every stream_message() call."""
    type: str = field(default="done", init=False)
    tool_calls_executed: int = 0


__all__ = [
    "AgentEvent",
    "AgentTextDelta",
    "AgentThinkingDelta",
    "AgentToolStart",
    "AgentToolResult",
    "AgentDone",
]
