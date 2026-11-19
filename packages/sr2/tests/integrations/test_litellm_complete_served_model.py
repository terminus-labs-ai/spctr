"""spc-1007: non-streaming complete() carries the served model.

The pool name in the request can resolve to a different model id on the
backend; the provider response's ``model`` field carries what was actually
served. ``LiteLLMCallable.complete()`` must copy it onto
``CompletionResponse.served_model`` (the streaming path already reports it on
the end event's meta). Absence is not an error: a provider response without
``model`` yields ``served_model=None`` — "unknown" stays distinguishable from
"requested".
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, TextBlock
from sr2.protocols.llm import CompletionRequest, CompletionResponse


def _make_request() -> CompletionRequest:
  return CompletionRequest(messages=[Message(role="user", content=[TextBlock(text="Hi")])])


def _mock_litellm_response(*, model: str | None) -> MagicMock:
  """A litellm non-streaming response; pin ``model`` so a MagicMock does not
  auto-vivify it when the provider omits the field."""
  usage = MagicMock()
  usage.prompt_tokens = 10
  usage.completion_tokens = 5

  choice = MagicMock()
  choice.message.content = "Hi there"
  choice.message.tool_calls = None
  choice.finish_reason = "stop"

  resp = MagicMock()
  resp.id = "resp-1"
  resp.choices = [choice]
  resp.usage = usage
  resp.model = model
  return resp


class TestCompleteServedModel:
  @pytest.mark.asyncio
  async def test_served_model_from_provider_response_model(self):
    resp = _mock_litellm_response(model="qwen3.8-flash-next")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("openai/pool-name")
      result = await client.complete(_make_request())

    assert isinstance(result, CompletionResponse)
    assert result.served_model == "qwen3.8-flash-next"

  @pytest.mark.asyncio
  async def test_served_model_none_when_provider_omits_model(self):
    resp = _mock_litellm_response(model=None)
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("openai/pool-name")
      result = await client.complete(_make_request())

    # No fallback to the requested pool name: None keeps "unknown"
    # distinguishable from "requested", matching the streaming path.
    assert result.served_model is None
