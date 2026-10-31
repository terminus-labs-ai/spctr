"""Tests for config loading (Step 3: restructured config)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError
from sr2_spectre.config import (
    AgentConfig,
    ModelConfig,
    SpectreConfig,
    ToolConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_minimal(self):
        cfg = ModelConfig(model="openai/gpt-4o")
        assert cfg.model == "openai/gpt-4o"
        assert cfg.base_url is None

    def test_with_base_url(self):
        cfg = ModelConfig(model="openai/qwen3:27b", base_url="http://localhost:11438/v1")
        assert cfg.base_url == "http://localhost:11438/v1"


# ---------------------------------------------------------------------------
# AgentConfig — new shape (no model/base_url/system_prompt)
# ---------------------------------------------------------------------------

class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.name == "spectre"
        assert cfg.tools == []

    def test_no_model_field(self):
        """AgentConfig must NOT have model/base_url/system_prompt."""
        cfg = AgentConfig()
        assert not hasattr(cfg, "model"), "model field must be removed from AgentConfig"
        assert not hasattr(cfg, "base_url"), "base_url must be removed"
        assert not hasattr(cfg, "system_prompt"), "system_prompt must be removed"

    def test_with_tools(self):
        cfg = AgentConfig(tools=[
            ToolConfig(name="search", class_path="sr2_spectre.tools:Search")
        ])
        assert len(cfg.tools) == 1
        assert cfg.tools[0].name == "search"


# ---------------------------------------------------------------------------
# SpectreConfig — new shape with models + pipeline
# ---------------------------------------------------------------------------

class TestSpectreConfig:
    def _make_pipeline_dict(self):
        return {
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static", "config": {"text": "Hello"}}],
                }
            ]
        }

    def test_minimal_valid_config(self):
        cfg = SpectreConfig(
            agent=AgentConfig(),
            models={"default": ModelConfig(model="openai/gpt-4o")},
            pipeline=self._make_pipeline_dict(),
        )
        assert cfg.agent.name == "spectre"
        assert "default" in cfg.models
        assert len(cfg.pipeline.layers) == 1

    def test_models_required(self):
        with pytest.raises((ValidationError, TypeError)):
            SpectreConfig(agent=AgentConfig(), pipeline=self._make_pipeline_dict())

    def test_pipeline_required(self):
        with pytest.raises((ValidationError, TypeError)):
            SpectreConfig(
                agent=AgentConfig(),
                models={"default": ModelConfig(model="openai/gpt-4o")},
            )

 


# ---------------------------------------------------------------------------
# load_config from YAML — new structure
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_full_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "agent:\n"
            "  name: my-agent\n"
            "\n"
            "models:\n"
            "  default:\n"
            "    model: openai/qwen3:27b\n"
            "    base_url: http://localhost:11438/v1\n"
            "\n"
            "pipeline:\n"
            "  token_budget: 100000\n"
            "  layers:\n"
            "    - name: system\n"
            "      target: system\n"
            "      resolvers:\n"
            "        - type: static\n"
            "          config:\n"
            "            text: You are a helpful assistant.\n"
            "    - name: tools\n"
            "      target: tools\n"
            "      resolvers: []\n"
            "      tool_providers:\n"
            "        - type: spectre_tools\n"
            "    - name: conversation\n"
            "      target: messages\n"
            "      resolvers:\n"
            "        - type: session\n"
            "        - type: input\n"
        )
        cfg = load_config(str(config_file))
        assert cfg.agent.name == "my-agent"
        assert cfg.models["default"].model == "openai/qwen3:27b"
        assert cfg.models["default"].base_url == "http://localhost:11438/v1"
        assert cfg.pipeline.token_budget == 100000
        assert len(cfg.pipeline.layers) == 3

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_none_returns_error(self):
        """load_config(None) must raise — pipeline is now required."""
        with pytest.raises(Exception):
            load_config(None)

    def test_smoke_yaml_loads(self):
        """smoke.yaml (migrated) loads without errors."""
        import pathlib
        smoke = pathlib.Path(__file__).parent.parent / "smoke.yaml"
        if smoke.exists():
            cfg = load_config(str(smoke))
            assert cfg.agent is not None
            assert cfg.models is not None
            assert cfg.pipeline is not None
