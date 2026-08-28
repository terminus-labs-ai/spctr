"""Structured-action safety — bounded handling of malformed LLM tool calls.

When a model emits a tool call that overruns its configured response
allowance, the provider truncates the response and the accumulated tool-call
argument stream ends mid-structure. Parsing that stream then fails with an
opaque parser error (``json.JSONDecodeError: Unterminated string``) that
bubbles out of the turn and aborts the run before any repository change.

This module turns that class of failure into a *bounded, actionable* signal:

- :class:`MalformedActionError` names the failure and what to do about it
  (decompose the action; never "just raise the token budget").
- :func:`validate_structured_action` gives the session a single choke point to
  reject a malformed action BEFORE any tool executes, so a truncated call can
  never be half-applied and the user's turn state stays clean.

The error messages are deliberately bounded: they state the detected
condition, the tool name, and the fix — nothing more — so an automated
runner that surfaces them can act without parsing a traceback.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "MalformedActionError",
    "TruncatedActionError",
    "MAX_ACTION_DEPTH",
    "DEFAULT_MAX_DEPTH",
    "validate_structured_action",
]

# Structural depth ceiling for a structured action. Legitimate tool calls are
# shallow (a few nesting levels at most); far deeper payloads are a sign of
# malformed or adversarial construction and are rejected the same way.
DEFAULT_MAX_DEPTH = 32
MAX_ACTION_DEPTH = DEFAULT_MAX_DEPTH


class MalformedActionError(Exception):
    """A structured tool-call action failed to parse into a usable shape.

    Bounded by design: the message is safe to surface verbatim to a human or
    an automated runner. It never embeds raw model output beyond a short
    reason string.
    """

    def __init__(self, reason: str, *, tool_name: str = "") -> None:
        prefix = f"Tool call for '{tool_name}'" if tool_name else "Tool call"
        super().__init__(
            f"{prefix} could not be used as a structured action: {reason}. "
            "This usually means the response exceeded the configured output "
            "allowance and was truncated. Split the action into smaller "
            "tool calls (one per file, smaller hunks) so each fits within "
            "the response allowance; do not increase the token budget to "
            "work around this."
        )
        self.reason = reason
        self.tool_name = tool_name


class TruncatedActionError(MalformedActionError):
    """The structured action stream ended mid-structure (response truncated)."""

    def __init__(self, tool_name: str = "", *, detail: str = "") -> None:
        super().__init__(
            "the argument stream ended mid-structure (truncated response)",
            tool_name=tool_name,
        )
        self.detail = detail


def _measure_depth(node: Any) -> int:
    """Return the nesting depth of a decoded JSON value (list > scalar)."""
    if isinstance(node, list):
        return 1 + max((_measure_depth(item) for item in node), default=0)
    if isinstance(node, dict):
        return 1 + max(
            (_measure_depth(value) for value in node.values()), default=0
        )
    return 0


def validate_structured_action(
    arguments: str,
    *,
    tool_name: str = "",
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> dict[str, Any]:
    """Parse a structured action's argument stream and validate its shape.

    Returns the decoded dict on success. Raises :class:`MalformedActionError`
    (or :class:`TruncatedActionError` for the truncated-stream case) when the
    stream is not a valid JSON object within the depth ceiling — in every
    failure case the caller must treat the action as unusable and must not
    execute any tool with it.
    """
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError as exc:
        # Truncation mid-structure is the dominant real-world cause; the
        # parse-error kind names it precisely so a runner can react.
        raise TruncatedActionError(
            tool_name,
            detail=f"{exc.msg} (at char {exc.pos})",
        ) from exc
    if not isinstance(decoded, dict):
        raise MalformedActionError(
            f"decoded to {type(decoded).__name__}, expected an object",
            tool_name=tool_name,
        )
    depth = _measure_depth(decoded)
    if depth > max_depth:
        raise MalformedActionError(
            f"structure too deep (depth {depth} > {max_depth})",
            tool_name=tool_name,
        )
    return decoded
