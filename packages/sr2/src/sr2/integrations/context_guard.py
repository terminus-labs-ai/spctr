"""Context-window guard for OpenAI-compatible endpoints (obsidian-pt2k.1).

llm-relay admits a chat request only when its own estimate of the wire body
fits some node's context::

    ceil((len(json.dumps(messages)) + len(json.dumps(tools))) / 4)
        + requested_output_tokens <= max_context

A long run grows the conversation until that admission check fails and the
endpoint answers 400 ``context_length_exceeded`` ("no node has context large
enough for model ..."); unhandled, the error killed bead runs mid-flight with
uncommitted work. This module keeps requests inside the ceiling: the estimate
mirrors the relay's formula, ``shed_oldest_messages`` drops the oldest turn
while keeping tool-call/result pairs and the current turn intact, and
``is_context_overflow_error`` recognises the rejection when the estimate
under-shoots.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

# Provider phrasings for "the prompt is too long", matched case-insensitively
# against the exception text. The llm-relay admission message and the
# OpenAI-style code/message are the two shapes seen in production; the vllm
# and Anthropic wordings keep the guard portable across backends.
_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
  re.compile(p, re.IGNORECASE)
  for p in (
    r"no node has context",
    r"context[_ ]length[_ ]exceeded",
    r"maximum context length",
    r"context window",
    r"prompt is too long",
    r"input length.*exceed",
  )
]


def estimate_request_tokens(
  messages: list[dict[str, Any]],
  tools: list[dict[str, Any]] | None,
) -> int:
  """Estimate the wire size the way llm-relay does (router.estimate_tokens).

  ``ceil((len(json.dumps(messages)) + len(json.dumps(tools))) / 4)`` — the
  same formula the hub's admission check uses, so a request the guard accepts
  is a request the hub admits on a context-fitting node.
  """
  return math.ceil(
    (len(json.dumps(messages)) + len(json.dumps(tools or []))) / 4
  )


def requested_output_tokens(kwargs: Mapping[str, Any]) -> int:
  """Requested output budget: max_completion_tokens, else max_tokens, else 0.

  Mirrors llm-relay's ``output_tokens``: a null field counts as absent. The
  hub reserves this from the node's context alongside the input estimate.
  """
  for field in ("max_completion_tokens", "max_tokens"):
    value = kwargs.get(field)
    if value is not None:
      return int(value)
  return 0


def shed_oldest_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
  """Drop the oldest droppable turn, or None when nothing can be dropped.

  Rules, in order of what the wire format requires:

  * A leading ``system`` message is never dropped.
  * The last message (the current turn) is never dropped.
  * The run's user prompt, the earliest ``user`` message that is not a pure
    tool-result message, is never dropped; when it heads the conversation the
    oldest droppable turn is the one after it. A pure tool-result message is a
    ``user`` message whose content is a list of ``tool_result`` blocks only;
    it stays droppable.
  * An assistant message carrying ``tool_calls`` is dropped together with the
    ``tool`` messages that answer it; a ``tool`` message that would be
    orphaned at the head is dropped too. Upstream rejects either half of a
    tool-call/result pair, so shedding only ever removes whole pairs.

  Returns None when the only droppable group would swallow the current turn,
  so the caller fails with the over-limit error instead of sending a request
  without the user prompt.
  """
  if not messages:
    return None
  start = 1 if messages[0].get("role") == "system" else 0
  if len(messages) - start <= 1:
    return None

  prompt = next(
    (
      i for i in range(start, len(messages))
      if messages[i].get("role") == "user"
      and not _is_tool_result_only(messages[i].get("content"))
    ),
    None,
  )
  head_index = start + 1 if prompt == start else start

  end = head_index + 1
  head = messages[head_index]
  if head.get("role") == "assistant" and head.get("tool_calls"):
    while end < len(messages) and messages[end].get("role") == "tool":
      end += 1
  if end >= len(messages):
    # The group swallows the current turn; nothing droppable remains.
    return None

  kept = messages[:head_index] + messages[end:]
  while len(kept) - head_index > 1 and kept[head_index].get("role") == "tool":
    kept = kept[:head_index] + kept[head_index + 1:]
  return kept


def _is_tool_result_only(content: Any) -> bool:
  """True when *content* is a non-empty list made up only of tool_result blocks."""
  return (
    isinstance(content, list)
    and bool(content)
    and all(isinstance(b, Mapping) and b.get("type") == "tool_result" for b in content)
  )


def is_context_overflow_error(exc: BaseException | str) -> bool:
  """True when *exc* is a provider rejection of the request as too long."""
  message = exc if isinstance(exc, str) else str(exc)
  return any(pattern.search(message) for pattern in _OVERFLOW_PATTERNS)
