"""Tests for per-agent model sampling params in ModelConfig (spc-24).

Covers:
  A. ModelConfig accepts params dict
  B. Runtime forwards params to LiteLLMCallable
  C. Absent params preserves current behavior (backward compat)
  D. YAML config with params loads correctly
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_pipeline_dict() -> dict:
    return {
        "layers": [
            {
                "name": "system",
                "target": "system",
                "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
            }
        ]
    }


def _make_config(params: dict | None = None) -> SpectreConfig:
    model_kwargs = {"model": "test-model", "base_url": "http://test:8000"}
    if params is not None:
        model_kwargs["params"] = params
    return SpectreConfig(
        agent=AgentConfig(name="test"),
        models={"default": ModelConfig(**model_kwargs)},
        pipeline=_minimal_pipeline_dict(),
    )


# ---------------------------------------------------------------------------
# A. ModelConfig accepts params
# ---------------------------------------------------------------------------

class TestModelConfigParams:
    def test_params_defaults_to_empty(self):
        cfg = ModelConfig(model="gpt-4o")
        assert cfg.params == {}

    def test_params_accepts_temperature(self):
        cfg = ModelConfig(model="gpt-4o", params={"temperature": 0.2})
        assert cfg.params["temperature"] == 0.2

    def test_params_accepts_multiple(self):
        cfg = ModelConfig(
            model="gpt-4o",
            params={"temperature": 0.15, "top_p": 0.9, "max_tokens": 4096},
        )
        assert cfg.params["temperature"] == 0.15
        assert cfg.params["top_p"] == 0.9
        assert cfg.params["max_tokens"] == 4096

    def test_params_arbitrary_keys(self):
        """Any key should be accepted — we're a pass-through."""
        cfg = ModelConfig(
            model="gpt-4o",
            params={"temperature": 0.5, "presence_penalty": 0.1, "frequency_penalty": 0.2},
        )
        assert len(cfg.params) == 3

    def test_backward_compat_no_params_key(self):
        """Config without params key should still work (defaults to {})."""
        cfg = ModelConfig.model_validate({"model": "gpt-4o"})
        assert cfg.params == {}


# ---------------------------------------------------------------------------
# B. Runtime forwards params to LiteLLMCallable
# ---------------------------------------------------------------------------

class TestRuntimeForwardsParams:
    def test_runtime_passes_params_to_llm(self):
        from sr2_spectre.runtime import Runtime

        cfg = _make_config(params={"temperature": 0.2, "top_p": 0.9})

        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            runtime = Runtime(config=cfg)

        MockLLM.assert_called_once_with(
            model="test-model",
            base_url="http://test:8000",
            temperature=0.2,
            top_p=0.9,
        )

    def test_runtime_empty_params_no_extra_kwargs(self):
        """When params is empty (default), Runtime passes only model+base_url."""
        from sr2_spectre.runtime import Runtime

        cfg = _make_config()  # params defaults to {}

        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            runtime = Runtime(config=cfg)

        MockLLM.assert_called_once_with(
            model="test-model",
            base_url="http://test:8000",
        )

    def test_runtime_forwards_all_param_types(self):
        """Params with various value types are forwarded correctly."""
        from sr2_spectre.runtime import Runtime

        cfg = _make_config(params={
            "temperature": 0.85,
            "top_p": 0.98,
            "max_tokens": 8192,
            "stream_options": {"include_usage": True},
        })

        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            runtime = Runtime(config=cfg)

        call_kwargs = MockLLM.call_args.kwargs
        assert call_kwargs["temperature"] == 0.85
        assert call_kwargs["top_p"] == 0.98
        assert call_kwargs["max_tokens"] == 8192
        assert call_kwargs["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# C. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_existing_config_without_params_works(self):
        """A config with only model+base_url (no params) still constructs fine."""
        cfg = SpectreConfig(
            agent=AgentConfig(name="test"),
            models={"default": ModelConfig(model="gpt-4o", base_url="http://x")},
            pipeline=_minimal_pipeline_dict(),
        )
        assert cfg.models["default"].params == {}

    def test_runtime_no_params_preserves_behavior(self):
        """Runtime with no params passes exactly model+base_url to LiteLLMCallable."""
        from sr2_spectre.runtime import Runtime

        cfg = SpectreConfig(
            agent=AgentConfig(name="test"),
            models={"default": ModelConfig(model="gpt-4o", base_url="http://x")},
            pipeline=_minimal_pipeline_dict(),
        )

        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            runtime = Runtime(config=cfg)

        assert MockLLM.call_args.kwargs == {"model": "gpt-4o", "base_url": "http://x"}


# ---------------------------------------------------------------------------
# D. YAML config with params
# ---------------------------------------------------------------------------

class TestYamlParams:
    def test_load_config_with_params(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "agent:\n"
            "  name: edi\n"
            "\n"
            "models:\n"
            "  default:\n"
            "    model: openai/qwen3:27b\n"
            "    base_url: http://localhost:11438/v1\n"
            "    params:\n"
            "      temperature: 0.15\n"
            "      top_p: 0.9\n"
            "\n"
            "pipeline:\n"
            "  layers:\n"
            "    - name: system\n"
            "      target: system\n"
            "      resolvers:\n"
            "        - type: static\n"
            "          config:\n"
            "            text: You are helpful.\n"
        )
        cfg = load_config(str(config_file))
        assert cfg.agent.name == "edi"
        assert cfg.models["default"].params["temperature"] == 0.15
        assert cfg.models["default"].params["top_p"] == 0.9

    def test_load_config_without_params(self, tmp_path):
        """YAML without params section should default to empty dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "agent:\n"
            "  name: miranda\n"
            "\n"
            "models:\n"
            "  default:\n"
            "    model: openai/gpt-4o\n"
            "\n"
            "pipeline:\n"
            "  layers:\n"
            "    - name: system\n"
            "      target: system\n"
            "      resolvers:\n"
            "        - type: static\n"
            "          config:\n"
            "            text: You are helpful.\n"
        )
        cfg = load_config(str(config_file))
        assert cfg.models["default"].params == {}
