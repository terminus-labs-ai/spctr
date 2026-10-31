"""Spectre core — value objects shared across layers."""

from __future__ import annotations

import enum
from dataclasses import dataclass


@dataclass
class TurnResult:
    """Result of a complete agent turn."""
    text: str
    tool_calls_executed: int = 0
    total_tokens: int = 0


class RunMode(enum.StrEnum):
    """Whether this run is interactive (can ask the user) or headless (must self-resolve)."""
    INTERACTIVE = "interactive"
    HEADLESS = "headless"


@dataclass(frozen=True)
class RunContext:
    """Execution context supplied by the Interface at startup.

    Attributes:
        interface: Which interface launched this run (e.g. "single_shot", "tui", "discord").
        mode: Whether the run is interactive or headless — controls agent proactivity.
        source: Where the run originated from (working directory, Discord channel name/id, etc.).
        area: Resolved area for this run, or None when the interface does not
            resolve areas. An empty string means "explicitly no area" and is
            distinct from None — consumers must not fall through on it.
    """
    interface: str
    mode: RunMode
    source: str | None
    area: str | None = None


__all__ = ["TurnResult", "RunContext", "RunMode"]
