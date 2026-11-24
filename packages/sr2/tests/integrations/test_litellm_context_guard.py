"""obsidian-pt2k.1 — LiteLLMCallable sheds conversation instead of dying.

The endpoint (llm-relay) admits a request only when its estimate fits a
node; a long run used to crash on the resulting 400. With ``context_limit``
set, ``LiteLLMCallable`` must:

  1. shed the oldest turns before sending when the estimate is over limit;
  2. on a context-length rejection, shed and retry (bounded), then raise
     only when nothing droppable is left;
  3. leave non-context errors and unlimited callers untouched.

``litellm.acompletion`` is the only mocked boundary.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, TextBlock
from sr2.protocols.llm import CompletionRequest

from .test_litellm_callable import (
  _make_stream_chunk,
  _mock_litellm_response,
  _async_gen,
)


def _big_text(n: int) -> str:
  """Deterministic filler whose JSON-encoded length is roughly n chars."""
  return "x" * n


def _request(turns: int, size: int = 4000, first_size: int | None = None) -> CompletionRequest:
  """Alternating user/assistant turns of ``size`` chars; ``first_size``, when
  given, sizes only the first user turn (the protected prompt)."""
  messages = []
  for i in range(turns):
    role = "user" if i % 2 == 0 else "assistant"
    n = first_size if (i == 0 and first_size is not None) else size
    messages.append(Message(role=role, content=[TextBlock(text=_big_text(n) + f" turn{i}")]))
  return CompletionRequest(messages=messages)


def _has_prompt(sent: list[dict], prompt: str) -> bool:
  """True when a user-role wire message carries the first user prompt."""
  return any(m.get("role") == "user" and prompt in json.dumps(m.get("content")) for m in sent)


def _wire_tokens(messages: list[dict], tools=None) -> int:
  from sr2.integrations.context_guard import estimate_request_tokens

  return estimate_request_tokens(messages, tools)


def _capture_fake(responses):
  """Fake acompletion recording kwargs; yields the given responses/raises."""
  calls: list[dict] = []
  it = iter(responses)

  async def fake(**kwargs):
    calls.append(kwargs)
    outcome = next(it)
    if isinstance(outcome, Exception):
      raise outcome
    return outcome

  return fake, calls


class TestProactiveShed:
  async def test_request_over_limit_is_shed_before_send(self):
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([resp])
    callable_ = LiteLLMCallable(model="openai/test", context_limit=2000)
    # The protected first user prompt is small, so prompt + current turn fits.
    request = _request(turns=8, size=4000, first_size=20)
    prompt = request.messages[0].content[0].text
    with patch("litellm.acompletion", new=fake):
      await callable_.complete(request)
    sent = calls[0]["messages"]
    assert len(sent) < 8
    assert _wire_tokens(sent, calls[0].get("tools")) <= 2000
    assert _has_prompt(sent, prompt)

  async def test_request_under_limit_is_untouched(self):
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([resp])
    callable_ = LiteLLMCallable(model="openai/test", context_limit=200_000)
    request = _request(turns=2, size=50)
    with patch("litellm.acompletion", new=fake):
      await callable_.complete(request)
    assert len(calls[0]["messages"]) == 2

  async def test_no_context_limit_never_sheds(self):
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([resp])
    callable_ = LiteLLMCallable(model="openai/test")
    with patch("litellm.acompletion", new=fake):
      await callable_.complete(_request(turns=6, size=4000))
    assert len(calls[0]["messages"]) == 6

  async def test_output_tokens_count_toward_the_limit(self):
    # max_tokens is reserved: with a tight limit the input must shrink so
    # that estimate + max_tokens fits, mirroring the relay's admission rule.
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([resp])
    callable_ = LiteLLMCallable(model="openai/test", context_limit=6000, max_tokens=4000)
    # The protected first user prompt is small, so prompt + current turn fits.
    request = _request(turns=8, size=4000, first_size=20)
    prompt = request.messages[0].content[0].text
    with patch("litellm.acompletion", new=fake):
      await callable_.complete(request)
    sent = calls[0]["messages"]
    assert len(sent) < 8
    assert _wire_tokens(sent, calls[0].get("tools")) + 4000 <= 6000
    assert _has_prompt(sent, prompt)

  async def test_shed_applies_to_streams_too(self):
    chunk = _make_stream_chunk(content="ok")
    fake_calls: list[dict] = []

    async def fake(**kwargs):
      fake_calls.append(kwargs)

      async def gen():
        yield chunk

      return gen()

    callable_ = LiteLLMCallable(model="openai/test", context_limit=2000)
    with patch("litellm.acompletion", new=fake):
      events = [e async for e in callable_.stream(_request(turns=8, size=4000))]
    assert len(fake_calls[0]["messages"]) < 8
    assert any(e.type == "text" for e in events)


class TestReactiveShed:
  async def test_context_rejection_sheds_and_retries(self):
    overflow = RuntimeError(
      "litellm.BadRequestError: OpenAIException - no node has context large"
      " enough for model 'qwen3.8'"
    )
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([overflow, resp])
    callable_ = LiteLLMCallable(model="openai/test")
    with patch("litellm.acompletion", new=fake):
      out = await callable_.complete(_request(turns=4, size=100))
    assert len(calls) == 2
    assert len(calls[1]["messages"]) < len(calls[0]["messages"])
    assert out.content[0].text == "Hi there"

  async def test_non_context_error_propagates_after_one_attempt(self):
    boom = RuntimeError("litellm.BadRequestError: bad tool schema")
    fake, calls = _capture_fake([boom])
    callable_ = LiteLLMCallable(model="openai/test")
    with patch("litellm.acompletion", new=fake):
      with pytest.raises(RuntimeError, match="bad tool schema"):
        await callable_.complete(_request(turns=2, size=10))
    assert len(calls) == 1

  async def test_gives_up_when_nothing_left_to_shed(self):
    overflow = RuntimeError("no node has context large enough for model 'x'")
    fake, calls = _capture_fake([overflow] * 10)
    callable_ = LiteLLMCallable(model="openai/test")
    with patch("litellm.acompletion", new=fake):
      with pytest.raises(RuntimeError, match="no node has context"):
        await callable_.complete(CompletionRequest(
          messages=[Message(role="user", content=[TextBlock(text="huge single turn")])]
        ))
    # one initial attempt, no pointless retries on an undroppable request
    assert len(calls) == 1


class TestProtectedPromptOnOverflow:
  """spc-1046: an over-limit single-shot request whose only droppable turn
  would take the user prompt fails with the overflow error; it never sends a
  request stripped of the user query."""

  async def test_overflow_propagates_and_prompt_is_never_dropped(self):
    from sr2.integrations.litellm import MAX_CONTEXT_OVERFLOW_RETRIES
    from sr2.models import ToolResultBlock, ToolUseBlock

    overflow = RuntimeError("no node has context large enough for model 'x'")
    # The endpoint rejects every attempt as too long; the guard may retry at
    # most MAX_CONTEXT_OVERFLOW_RETRIES times, so the script covers exactly
    # the allowed attempts and a call past it fails the test loudly.
    allowed = MAX_CONTEXT_OVERFLOW_RETRIES + 1
    calls: list[dict] = []

    async def fake(**kwargs):
      calls.append(kwargs)
      if len(calls) > allowed:
        raise AssertionError("acompletion called past the scripted overflow replies")
      raise overflow

    prompt_text = "the run's user prompt"
    request = CompletionRequest(
      system=[TextBlock(text="sys")],
      messages=[
        Message(role="user", content=[TextBlock(text=prompt_text)]),
        Message(role="assistant", content=[ToolUseBlock(id="tu_1", name="t", input={})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="tu_1", content="ok")]),
      ],
    )
    callable_ = LiteLLMCallable(model="openai/test")
    with patch("litellm.acompletion", new=fake):
      with pytest.raises(RuntimeError, match="no node has context") as excinfo:
        await callable_.complete(request)

    assert "No user query" not in str(excinfo.value)
    assert calls, "the request must reach the endpoint at least once"
    for sent in calls:
      assert {"role": "user", "content": prompt_text} in sent["messages"]
