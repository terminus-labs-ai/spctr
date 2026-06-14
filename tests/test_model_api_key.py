"""Tests for ModelConfig.api_key and its forwarding to the LLM (obsidian-87f).

The yaml carries `api_key` (e.g. "dummy" for a local OpenAI-compatible
llama.cpp endpoint). Before this fix ModelConfig had no api_key field, so the
value was silently dropped at validation and never reached litellm — the agent
then died with AuthenticationError ("set OPENAI_API_KEY"). These guard that the
configured key is parsed and forwarded.

Covers:
  A. ModelConfig accepts api_key (defaults to None)
  B. Runtime forwards api_key to LiteLLMCallable
  C. Absent api_key forwards nothing (backward compat)
  D. YAML config with api_key loads
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig, load_config


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


def _make_config(api_key: str | None = None) -> SpectreConfig:
    model_kwargs = {"model": "test-model", "base_url": "http://test:8000"}
    if api_key is not None:
        model_kwargs["api_key"] = api_key
    return SpectreConfig(
        agent=AgentConfig(name="test"),
        models={"default": ModelConfig(**model_kwargs)},
        pipeline=_minimal_pipeline_dict(),
    )


# A. ModelConfig accepts api_key
class TestModelConfigApiKey:
    def test_api_key_defaults_to_none(self):
        cfg = ModelConfig(model="gpt-4o")
        assert cfg.api_key is None

    def test_api_key_accepts_value(self):
        cfg = ModelConfig(model="gpt-4o", api_key="dummy")
        assert cfg.api_key == "dummy"


# B. Runtime forwards api_key
class TestRuntimeForwardsApiKey:
    def test_runtime_passes_api_key_to_llm(self):
        from sr2_spectre.runtime import Runtime

        cfg = _make_config(api_key="dummy")
        with patch("sr2_spectre.runtime.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            Runtime(config=cfg)

        MockLLM.assert_called_once_with(
            model="test-model",
            base_url="http://test:8000",
            api_key="dummy",
        )


# C. Backward compat — no api_key forwards nothing
class TestBackwardCompatibility:
    def test_runtime_no_api_key_omits_kwarg(self):
        from sr2_spectre.runtime import Runtime

        cfg = _make_config()  # no api_key
        with patch("sr2_spectre.runtime.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            Runtime(config=cfg)

        assert "api_key" not in MockLLM.call_args.kwargs
        assert MockLLM.call_args.kwargs == {
            "model": "test-model",
            "base_url": "http://test:8000",
        }


# D. YAML config with api_key
class TestYamlApiKey:
    def test_load_config_with_api_key(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "agent:\n"
            "  name: edi\n"
            "\n"
            "models:\n"
            "  default:\n"
            "    model: openai/qwen3.6:27b\n"
            "    base_url: http://localhost:11437/v1\n"
            "    api_key: dummy\n"
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
        assert cfg.models["default"].api_key == "dummy"
