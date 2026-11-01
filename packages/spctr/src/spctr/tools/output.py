"""Tool output wrapper with optional post-execute bus events.

Tools return plain values (str, dict, etc.) by default.  When a tool wants
to emit events on the SR2 event bus *after* execution, it wraps its
result in a ``ToolOutput`` and attaches ``PostExecuteEvent`` objects.

The Session's tool executor checks for the wrapper and dispatches events
generically — no name-magic or JSON sniffing required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PostExecuteEvent:
    """A bus event to dispatch after a tool completes.

    Tools attach these to their ``ToolOutput`` to trigger downstream
    effects (e.g. plan_step_completed for the step compaction transformer).

    Attributes:
        event_name: SR2 event name (e.g. ``"plan_step_completed"``).
        data: Arbitrary dict passed as the event's ``data`` field.
        phase: SR2 ``EventPhase`` (default: ``COMPLETED``).
        source_layer: Source layer string for the event (default: ``"plan"``).
    """

    event_name: str
    data: dict[str, Any] = field(default_factory=dict)
    phase: str = "completed"
    source_layer: str = "plan"


@dataclass
class ToolOutput:
    """Wrapper that carries a tool's result alongside post-execute events.

    Use this when a tool needs to emit bus events after execution:

    .. code-block:: python

        return ToolOutput(
            result=json.dumps({"success": True, ...}),
            events=[
                PostExecuteEvent(
                    event_name="plan_step_completed",
                    data={"frame": "plan:x/y", ...},
                ),
            ],
        )

    The executor extracts ``result`` as the tool's string output and
    dispatches each ``PostExecuteEvent`` on the SR2 event bus.

    Attributes:
        result: The actual tool output (any type; converted to str by executor).
        events: Post-execute events to dispatch on the SR2 bus.
    """

    result: Any
    events: list[PostExecuteEvent] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow truthiness checks without unwrapping."""
        return True


__all__ = ["PostExecuteEvent", "ToolOutput"]
