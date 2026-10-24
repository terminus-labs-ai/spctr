"""obsidian-pt2k.1 — context_limit plumbs from ModelConfig to the callable.

The context guard in sr2's LiteLLMCallable only engages when a
``context_limit`` reaches it. ModelConfig carries the value; build_llm is
the single definition of how a ModelConfig becomes a callable, so the
passthrough lives there and is asserted here.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from sr2_spectre.config import ModelConfig
from sr2_spectre.live_llm import build_llm


def _cfg(**extra) -> ModelConfig:
    return ModelConfig(model="qwen3.8", base_url="http://relay:11434/v1", **extra)


class TestContextLimitPassthrough:
    def test_model_config_accepts_context_limit(self):
        assert _cfg(context_limit=229000).context_limit == 229000

    def test_default_is_none(self):
        assert _cfg().context_limit is None

    def test_nonpositive_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _cfg(context_limit=0)

    def test_build_llm_forwards_context_limit(self):
        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            build_llm(_cfg(context_limit=229000))
        assert MockLLM.call_args.kwargs["context_limit"] == 229000

    def test_build_llm_omits_context_limit_when_unset(self):
        # Callers that never set the knob must not see a new kwarg — the
        # guard stays opt-in and existing call assertions keep holding.
        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            build_llm(_cfg())
        assert "context_limit" not in MockLLM.call_args.kwargs
