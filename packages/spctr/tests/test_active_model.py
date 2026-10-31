"""active_model: the indirection that lets ``/model`` switch models at runtime.

The agent's LLM used to be hardcoded to ``models["default"]`` in three places
(``runtime`` build + reload, and the startup log). These cover the new
indirection that replaced it:

  A. ``SpectreConfig.active_model`` + ``active_model_config`` resolution,
     including the fallbacks that stop a stale name from leaving no LLM.
  B. The writable pointer-file overlay applied in ``cli.resolve_config`` —
     the one writable surface ``/model`` touches, deliberately kept out of
     the ``:ro`` ``config.yaml`` so a ``code_exec`` user cannot rewrite tool
     ``class_path`` entries.
  C. End to end: flipping ``active_model`` retargets a running Runtime's LLM.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

from sr2_spectre.cli import _ACTIVE_MODEL_ENV, _apply_active_model_pointer
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig


def _cfg(active_model: str = "default", *, model_names=("default",)) -> SpectreConfig:
    models = {
        n: ModelConfig(model=f"m-{n}", base_url=f"http://{n}/v1")
        for n in model_names
    }
    return SpectreConfig(
        agent=AgentConfig(name="t"),
        models=models,
        pipeline={"layers": []},
        active_model=active_model,
    )


# ---------------------------------------------------------------------------
# A. active_model resolution
# ---------------------------------------------------------------------------

class TestActiveModelResolution:
    def test_defaults_to_default(self):
        assert _cfg().active_model == "default"

    def test_resolves_the_named_model(self):
        cfg = _cfg("fast", model_names=("default", "fast"))
        assert cfg.active_model_config.model == "m-fast"

    def test_unknown_name_falls_back_to_default(self):
        cfg = _cfg("ghost", model_names=("default", "fast"))
        assert cfg.active_model_config.model == "m-default"

    def test_falls_back_to_first_when_no_default(self):
        cfg = _cfg("ghost", model_names=("fast",))
        assert cfg.active_model_config.model == "m-fast"


# ---------------------------------------------------------------------------
# B. Pointer-file overlay
# ---------------------------------------------------------------------------

class TestActiveModelPointer:
    def test_no_env_var_leaves_config_unchanged(self):
        cfg = _cfg(model_names=("default", "fast"))
        out = _apply_active_model_pointer(cfg, env={})
        assert out.active_model == "default"
        assert out is cfg

    def test_valid_pointer_switches_active_model(self, tmp_path):
        f = tmp_path / "active_model"
        f.write_text("fast")
        cfg = _cfg(model_names=("default", "fast"))
        out = _apply_active_model_pointer(cfg, env={_ACTIVE_MODEL_ENV: str(f)})
        assert out.active_model == "fast"
        assert out.active_model_config.model == "m-fast"

    def test_whitespace_is_stripped(self, tmp_path):
        f = tmp_path / "active_model"
        f.write_text("  fast \n")
        cfg = _cfg(model_names=("default", "fast"))
        out = _apply_active_model_pointer(cfg, env={_ACTIVE_MODEL_ENV: str(f)})
        assert out.active_model == "fast"

    def test_unknown_name_is_ignored_and_warns(self, tmp_path, caplog):
        f = tmp_path / "active_model"
        f.write_text("ghost")
        cfg = _cfg(model_names=("default", "fast"))
        with caplog.at_level(logging.WARNING):
            out = _apply_active_model_pointer(cfg, env={_ACTIVE_MODEL_ENV: str(f)})
        assert out.active_model == "default"
        assert "ghost" in caplog.text

    def test_empty_file_is_ignored(self, tmp_path):
        f = tmp_path / "active_model"
        f.write_text("   \n")
        cfg = _cfg(model_names=("default", "fast"))
        out = _apply_active_model_pointer(cfg, env={_ACTIVE_MODEL_ENV: str(f)})
        assert out.active_model == "default"

    def test_missing_file_is_ignored(self, tmp_path):
        missing = tmp_path / "nope"
        cfg = _cfg(model_names=("default", "fast"))
        out = _apply_active_model_pointer(cfg, env={_ACTIVE_MODEL_ENV: str(missing)})
        assert out.active_model == "default"


# ---------------------------------------------------------------------------
# C. End to end — flipping active_model retargets the running LLM
# ---------------------------------------------------------------------------

class TestActiveModelRetarget:
    def _build(self, cfg):
        from sr2_spectre.runtime import Runtime
        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            return Runtime(config=cfg)

    def test_switching_active_model_retargets_the_llm(self):
        start = _cfg("default", model_names=("default", "fast"))
        runtime = self._build(start)
        assert runtime.llm.model_config.base_url == "http://default/v1"

        switched = _cfg("fast", model_names=("default", "fast"))
        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            applied = runtime.apply_config(switched)

        assert "models" in applied
        assert MockLLM.call_args.kwargs["base_url"] == "http://fast/v1"
        assert runtime.llm.model_config.base_url == "http://fast/v1"
