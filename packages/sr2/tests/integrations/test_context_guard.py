"""obsidian-pt2k.1 — context-window guard for OpenAI-compatible endpoints.

A long run grows the conversation until the serving endpoint's admission
check fails with a context-length 400 ("no node has context large enough
for model 'qwen3.8'" from llm-relay). The run must shed its oldest turns
instead of dying mid-flight.

Pure-function tests for ``sr2.integrations.context_guard``; the wire-in of
the guard into ``LiteLLMCallable`` lives in ``test_litellm_context_guard.py``.
"""

from __future__ import annotations

import json
import math

from sr2.integrations.context_guard import (
  estimate_request_tokens,
  is_context_overflow_error,
  requested_output_tokens,
  shed_oldest_messages,
)


def _msg(role: str, content: str = "x", **extra) -> dict:
  return {"role": role, "content": content, **extra}


def _assistant_tool_call(tool_id: str = "tu_1") -> dict:
  return {
    "role": "assistant",
    "content": None,
    "tool_calls": [
      {"id": tool_id, "type": "function", "function": {"name": "t", "arguments": "{}"}}
    ],
  }


def _tool_result(tool_id: str = "tu_1") -> dict:
  return {"role": "tool", "tool_call_id": tool_id, "content": "ok"}


# ---------------------------------------------------------------------------
# estimate_request_tokens — mirrors llm-relay router.estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimate:
  def test_mirrors_relay_formula(self):
    messages = [_msg("user", "hello")]
    tools = [{"type": "function", "function": {"name": "t"}}]
    expected = math.ceil(
      (len(json.dumps(messages)) + len(json.dumps(tools))) / 4
    )
    assert estimate_request_tokens(messages, tools) == expected

  def test_no_tools_counts_messages_only(self):
    messages = [_msg("user", "hello")]
    expected = math.ceil(len(json.dumps(messages)) / 4)
    assert estimate_request_tokens(messages, None) == expected


class TestRequestedOutputTokens:
  def test_max_completion_tokens_wins(self):
    assert requested_output_tokens({"max_tokens": 100, "max_completion_tokens": 50}) == 50

  def test_max_tokens_fallback(self):
    assert requested_output_tokens({"max_tokens": 100}) == 100

  def test_absent_is_zero(self):
    assert requested_output_tokens({}) == 0

  def test_null_counts_as_absent(self):
    assert requested_output_tokens({"max_tokens": None}) == 0


# ---------------------------------------------------------------------------
# shed_oldest_messages
# ---------------------------------------------------------------------------


class TestShedOldest:
  def test_keeps_system_and_drops_oldest_non_system(self):
    # spc-1046: the run's user prompt is protected, so the oldest droppable
    # non-system turn is the first one after it.
    messages = [
      _msg("system", "sys"),
      _msg("user", "prompt"),
      _msg("assistant", "reply"),
      _msg("user", "follow-up"),
      _msg("assistant", "reply 2"),
      _msg("user", "now"),
    ]
    assert shed_oldest_messages(messages) == [
      messages[0], messages[1], messages[3], messages[4], messages[5],
    ]

  def test_drops_tool_call_group_together(self):
    messages = [
      _msg("system", "sys"),
      _assistant_tool_call("tu_1"),
      _tool_result("tu_1"),
      _msg("user", "now"),
    ]
    assert shed_oldest_messages(messages) == [messages[0], messages[3]]

  def test_drops_orphaned_leading_tool_results(self):
    # A tool result whose parent assistant was already dropped must never
    # survive at the head of the request — upstream rejects the pairing.
    messages = [
      _msg("system", "sys"),
      _tool_result("tu_gone"),
      _msg("user", "now"),
    ]
    assert shed_oldest_messages(messages) == [messages[0], messages[2]]

  def test_never_drops_the_current_turn(self):
    messages = [_msg("system", "sys"), _msg("user", "only")]
    assert shed_oldest_messages(messages) is None

  def test_none_when_group_would_consume_current_turn(self):
    messages = [_msg("system", "sys"), _assistant_tool_call(), _tool_result()]
    assert shed_oldest_messages(messages) is None

  def test_no_system_message(self):
    # spc-1046: without a system message the protected prompt is first; the
    # oldest turn after it is dropped.
    messages = [_msg("user", "prompt"), _msg("user", "first"), _msg("user", "now")]
    assert shed_oldest_messages(messages) == [messages[0], messages[2]]

  def test_empty_returns_none(self):
    assert shed_oldest_messages([]) is None


# ---------------------------------------------------------------------------
# spc-1046 — the run's user prompt is never shed
# ---------------------------------------------------------------------------


def _tool_result_only_user(tool_id: str = "tu_0") -> dict:
  """Anthropic-style user message carrying only tool_result blocks."""
  return {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}],
  }


def _shed_until_none(messages: list[dict], limit: int = 20) -> list[list[dict]]:
  """Shed repeatedly; return every non-None intermediate list.

  Fails the test if shedding has not ended in None within *limit* steps.
  """
  steps: list[list[dict]] = []
  current = messages
  for _ in range(limit):
    shed = shed_oldest_messages(current)
    if shed is None:
      return steps
    steps.append(shed)
    current = shed
  raise AssertionError(f"shedding did not end in None within {limit} steps")


class TestProtectedUserPrompt:
  def test_sole_user_message_is_never_shed(self):
    # R1
    assert shed_oldest_messages([_msg("user", "prompt")]) is None

  def test_system_and_sole_user_returns_none(self):
    # R2
    assert shed_oldest_messages([_msg("system", "sys"), _msg("user", "prompt")]) is None

  def test_only_sheddable_group_holds_the_prompt_returns_none(self):
    # R3
    messages = [_msg("system", "sys"), _msg("user", "prompt"), _msg("assistant", "reply")]
    assert shed_oldest_messages(messages) is None

  def test_older_tool_group_shed_before_the_prompt(self):
    # R4
    messages = [
      _msg("system", "sys"),
      _msg("user", "prompt"),
      _assistant_tool_call("tu_1"),
      _tool_result("tu_1"),
      _assistant_tool_call("tu_2"),
      _tool_result("tu_2"),
      _msg("user", "now"),
    ]
    assert shed_oldest_messages(messages) == [
      messages[0], messages[1], messages[4], messages[5], messages[6],
    ]

  def test_repeated_shedding_never_touches_the_prompt(self):
    # R5
    prompt = _msg("user", "prompt")
    messages = [
      _msg("system", "sys"),
      prompt,
      _assistant_tool_call("tu_1"),
      _tool_result("tu_1"),
      _assistant_tool_call("tu_2"),
      _tool_result("tu_2"),
      _msg("user", "now"),
    ]
    steps = _shed_until_none(messages)
    assert steps, "the older tool groups must be sheddable"
    for step in steps:
      assert prompt in step

  def test_tool_result_only_user_message_is_sheddable_prompt_kept(self):
    # R6
    tool_only = _tool_result_only_user("tu_0")
    prompt = _msg("user", "prompt")
    messages = [
      _msg("system", "sys"),
      tool_only,
      prompt,
      _msg("assistant", "reply"),
      _msg("user", "now"),
    ]
    steps = _shed_until_none(messages)
    assert steps, "the tool-result-only user message must be sheddable"
    assert tool_only not in steps[-1]
    for step in steps:
      assert prompt in step


# ---------------------------------------------------------------------------
# is_context_overflow_error
# ---------------------------------------------------------------------------


class TestIsContextOverflowError:
  def test_matches_llm_relay_admission_message(self):
    msg = "no node has context large enough for model 'qwen3.8'"
    assert is_context_overflow_error(ValueError(msg))

  def test_matches_openai_style_message(self):
    msg = "This model's maximum context length is 131072 tokens"
    assert is_context_overflow_error(ValueError(msg))

  def test_matches_error_code_spelling(self):
    assert is_context_overflow_error("context_length_exceeded")

  def test_rejects_unrelated_bad_request(self):
    assert not is_context_overflow_error(ValueError("old_string not found in file"))

  def test_accepts_raw_string(self):
    assert is_context_overflow_error("no node has context large enough for model 'x'")


class _CodedError(Exception):
  """Exception carrying OpenAI-style ``code`` / ``body`` attributes."""

  def __init__(self, message: str, code=None, body=None):
    super().__init__(message)
    if code is not None:
      self.code = code
    if body is not None:
      self.body = body


class TestOverflowRecognitionSpc1117:
  """spc-1117 AC1: Gufo wording, litellm's typed error and the overflow code."""

  def test_ac1_r1_gufo_prompt_has_tokens_wording(self):
    assert is_context_overflow_error(
      "litellm.BadRequestError: OpenAIException - prompt has 215081 tokens but"
      " the context is 215040; increase --context or shorten the conversation"
    )

  def test_ac1_r1_gufo_token_wording_alone(self):
    assert is_context_overflow_error("prompt has 215081 tokens but the context is 215040")

  def test_ac1_r2_shorten_the_conversation(self):
    assert is_context_overflow_error("... please shorten the conversation")

  def test_ac1_r3_litellm_context_window_exceeded_type(self):
    import litellm

    exc = litellm.ContextWindowExceededError("x", model="m", llm_provider="openai")
    assert is_context_overflow_error(exc)

  def test_ac1_r4_code_attribute(self):
    exc = _CodedError("bad request", code="context_length_exceeded")
    assert not is_context_overflow_error(str(exc))
    assert is_context_overflow_error(exc)

  def test_ac1_r5_nested_body_error_code(self):
    exc = _CodedError("bad request", body={"error": {"code": "context_length_exceeded"}})
    assert is_context_overflow_error(exc)

  def test_ac1_r6_flat_body_code(self):
    exc = _CodedError("bad request", body={"code": "context_length_exceeded"})
    assert is_context_overflow_error(exc)

  def test_ac1_r7_other_errors_not_recognised(self):
    assert not is_context_overflow_error(RuntimeError("litellm.BadRequestError: bad tool schema"))
    assert not is_context_overflow_error(_CodedError("bad request", code="invalid_request_error"))
    assert not is_context_overflow_error(_CodedError("bad request", body="not a dict"))
    assert not is_context_overflow_error(
      _CodedError("bad request", body={"error": "context_length_exceeded"})
    )
