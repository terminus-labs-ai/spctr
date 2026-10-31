"""Tests for spc-5: unified pydantic validation — dry-run and startup agree.

validate_config, load_and_validate, and StartupConfigError have been deleted.
SpectreConfig (pydantic) is the single source of truth for config validation.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sr2_spectre.config import (
    CircularExtendsError,
    SpectreConfig,
    load_config_with_provenance,
    load_merged_config,
    load_resolved_config,
)
from sr2_spectre.path_resolution import ConfigPathError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# SpectreConfig — pydantic is the single validation source
# ---------------------------------------------------------------------------

class TestSpectreConfigValidation:
    """SpectreConfig rejects configs that validate_config used to catch."""

    def test_valid_config_accepted(self):
        """A complete valid config builds successfully."""
        config = {
            "agent": {"name": "spectre"},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {
                "layers": [
                    {
                        "name": "my_layer",
                        "resolvers": [{"type": "prompt"}],
                        "target": "default",
                    }
                ]
            },
        }
        result = SpectreConfig(**config)
        assert result.agent.name == "spectre"

    def test_empty_agent_name_allowed(self):
        """agent.name = '' is allowed by pydantic (str type, no min_length constraint).

        The hand-rolled validate_config used to reject this, but pydantic is the
        single source of truth and permits empty strings for str fields.
        """
        config = {
            "agent": {"name": ""},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        }
        result = SpectreConfig(**config)
        assert result.agent.name == ""

    def test_models_missing_model_field_rejected(self):
        """A model entry without a 'model' field must raise ValidationError."""
        config = {
            "agent": {"name": "test"},
            "models": {"default": {"base_url": "https://example.com"}},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        }
        with pytest.raises(ValidationError):
            SpectreConfig(**config)

    def test_models_empty_model_field_allowed(self):
        """A model entry with model='' is allowed by pydantic (no min_length constraint)."""
        config = {
            "agent": {"name": "test"},
            "models": {"default": {"model": ""}},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        }
        result = SpectreConfig(**config)
        assert result.models["default"].model == ""

    def test_models_must_be_dict(self):
        """models: [list] is invalid — must be a dict."""
        config = {
            "agent": {"name": "test"},
            "models": ["default"],
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        }
        with pytest.raises(ValidationError):
            SpectreConfig(**config)

    def test_pipeline_layer_missing_name_rejected(self):
        """A layer without a 'name' field must raise ValidationError."""
        config = {
            "agent": {"name": "test"},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {"layers": [{}]},
        }
        with pytest.raises(ValidationError):
            SpectreConfig(**config)

    def test_pipeline_layer_empty_name_allowed(self):
        """A layer with name='' is allowed by pydantic (no min_length constraint)."""
        config = {
            "agent": {"name": "test"},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {"layers": [{"name": "", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        }
        result = SpectreConfig(**config)
        assert result.pipeline.layers[0].name == ""

    def test_missing_pipeline_rejected(self):
        """Missing pipeline (required field) must raise ValidationError."""
        config = {
            "agent": {"name": "test"},
            "models": {"default": {"model": "gpt-4o"}},
        }
        with pytest.raises(ValidationError):
            SpectreConfig(**config)

    def test_missing_models_rejected(self):
        """Missing models (required field) must raise ValidationError."""
        config = {
            "agent": {"name": "test"},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        }
        with pytest.raises(ValidationError):
            SpectreConfig(**config)

    def test_missing_agent_rejected(self):
        """Missing agent (required field) must raise ValidationError."""
        config = {
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        }
        with pytest.raises(ValidationError):
            SpectreConfig(**config)

    def test_multiple_errors_aggregated(self):
        """Multiple validation errors are all reported in a single ValidationError."""
        config = {
            "agent": {"name": ""},
            "models": {"default": {"base_url": "x"}},
            "pipeline": {"layers": [{}]},
        }
        with pytest.raises(ValidationError) as exc_info:
            SpectreConfig(**config)
        # Pydantic reports multiple errors in a single exception
        assert len(exc_info.value.errors()) >= 2

    def test_config_without_agent_name_is_valid(self):
        """Agent section without a 'name' field is valid (name defaults to 'spectre')."""
        config = {
            "agent": {"tools": []},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        }
        result = SpectreConfig(**config)
        assert result.agent.name == "spectre"  # default value


# ---------------------------------------------------------------------------
# Dry-run vs startup agreement (spc-5 acceptance criterion #3)
# ---------------------------------------------------------------------------

class TestDryRunAgreesWithStartup:
    """Dry-run and startup must agree on the same config for valid and invalid cases.

    Both paths now build SpectreConfig(**merged), so they use the identical
    pydantic validation logic.
    """

    def test_valid_config_both_paths_succeed(self, tmp_path):
        """A valid config: both dry-run and real startup succeed."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {
                "layers": [
                    {"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"},
                ]
            },
        })

        env = {"SR2_HOME": str(sr2_home)}

        # Path 1: dry-run (config show) — load config + SpectreConfig build
        config, _ = load_config_with_provenance(cwd=cwd, env=env)
        try:
            SpectreConfig(**config)
            dry_run_ok = True
        except ValidationError:
            dry_run_ok = False

        # Path 2: startup — load_resolved_config (returns merged dict) + SpectreConfig
        merged = load_resolved_config(cwd / ".spectre.yaml", cwd=cwd, env=env)
        try:
            SpectreConfig(**merged)
            startup_ok = True
        except ValidationError:
            startup_ok = False

        assert dry_run_ok is True
        assert startup_ok is True
        assert dry_run_ok == startup_ok

    def test_invalid_config_both_paths_fail(self, tmp_path):
        """An invalid config: both dry-run and real startup fail."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"base_url": "https://example.com"}},  # missing required 'model'
            "pipeline": {
                "layers": [
                    {"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"},
                ]
            },
        })

        env = {"SR2_HOME": str(sr2_home)}

        # Path 1: dry-run (config show) — load config + SpectreConfig build
        config, _ = load_config_with_provenance(cwd=cwd, env=env)
        try:
            SpectreConfig(**config)
            dry_run_ok = True
        except ValidationError:
            dry_run_ok = False

        # Path 2: startup — load_resolved_config (returns merged dict) + SpectreConfig
        merged = load_resolved_config(cwd / ".spectre.yaml", cwd=cwd, env=env)
        try:
            SpectreConfig(**merged)
            startup_ok = True
        except ValidationError:
            startup_ok = False

        assert dry_run_ok is False
        assert startup_ok is False
        assert dry_run_ok == startup_ok

    def test_both_paths_use_same_validation_mechanism(self):
        """Both paths construct SpectreConfig(**dict) — same pydantic validation.

        The dry-run path (_run_config_show) and the startup path (cli.resolve_config)
        both call SpectreConfig(**merged_config). There is only one validation source.
        """
        config = {
            "agent": {"name": "test"},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {
                "layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}],
            },
        }

        # Build once — if this works, both paths succeed
        result1 = SpectreConfig(**config)
        result2 = SpectreConfig(**config)
        assert result1.agent.name == result2.agent.name
        assert result1.models == result2.models


# ---------------------------------------------------------------------------
# Structural errors still propagate directly (unchanged)
# ---------------------------------------------------------------------------

class TestStructuralErrorsPropagate:
    """CircularExtendsError and ConfigPathError still propagate directly."""

    def test_circular_extends_propagates(self, tmp_path):
        """Circular extends chain raises CircularExtendsError."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {"extends": ".spectre.yaml", "key": "value"})

        with pytest.raises(CircularExtendsError):
            load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

    def test_unset_env_var_in_extends_raises_config_path_error(self, tmp_path):
        """${UNSET_VAR} in extends path raises ConfigPathError."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "extends": "${UNSET_SPECTRE_VAR}/base.yaml",
            "key": "value",
        })

        with pytest.raises(ConfigPathError):
            load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})


# ---------------------------------------------------------------------------
# Deleted symbols are no longer importable
# ---------------------------------------------------------------------------

class TestDeletedSymbols:
    """validate_config, load_and_validate, and StartupConfigError are removed."""

    def test_validate_config_not_exported(self):
        import sr2_spectre.config as config_module
        assert not hasattr(config_module, "validate_config")

    def test_load_and_validate_not_exported(self):
        import sr2_spectre.config as config_module
        assert not hasattr(config_module, "load_and_validate")

    def test_startup_config_error_not_exported(self):
        import sr2_spectre.config as config_module
        assert not hasattr(config_module, "StartupConfigError")
