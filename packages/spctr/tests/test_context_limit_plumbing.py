"""obsidian-pt2k.1 — context_limit plumbs from ModelConfig to the callable.

The context guard in sr2's LiteLLMCallable only engages when a
``context_limit`` reaches it. ModelConfig carries the value; build_llm is
the single definition of how a ModelConfig becomes a callable, so the
passthrough lives there and is asserted here.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from spctr.config import ModelConfig
from spctr.live_llm import build_llm


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
        with patch("spctr.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            build_llm(_cfg(context_limit=229000))
        assert MockLLM.call_args.kwargs["context_limit"] == 229000

    def test_build_llm_omits_context_limit_when_unset(self):
        # Callers that never set the knob must not see a new kwarg — the
        # guard stays opt-in and existing call assertions keep holding.
        with patch("spctr.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            build_llm(_cfg())
        assert "context_limit" not in MockLLM.call_args.kwargs


class TestContextSafetyRatioSpc1117:
    def test_ac2_r1_build_llm_forwards_limit_and_ratio(self):
        with patch("spctr.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            build_llm(ModelConfig(model="m", context_limit=229000, context_safety_ratio=0.85))
        assert MockLLM.call_args.kwargs["context_limit"] == 229000
        assert MockLLM.call_args.kwargs["context_safety_ratio"] == 0.85

    def test_ac2_r1_real_callable_applies_ratio(self):
        llm = build_llm(ModelConfig(model="m", context_limit=229000, context_safety_ratio=0.85))
        assert llm._context_limit == 194650
        assert "context_safety_ratio" not in llm._kwargs

    def test_ac2_r2_out_of_range_ratio_rejected(self):
        import pytest
        from pydantic import ValidationError

        for bad in (0, -0.1, 1.5):
            with pytest.raises(ValidationError):
                ModelConfig(model="m", context_safety_ratio=bad)

    def test_ac2_r3_default_ratio_is_one_and_not_forwarded(self):
        cfg = ModelConfig(model="m")
        assert cfg.context_safety_ratio == 1.0
        with patch("spctr.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            build_llm(cfg)
        assert "context_safety_ratio" not in MockLLM.call_args.kwargs
        assert "context_limit" not in MockLLM.call_args.kwargs
