"""Tests for FR9: dry-run / config inspection mode (obsidian-17w.6).

Covers:
- load_config_with_provenance: provenance tracking at single and multi-tier
- format_dry_run: output format (annotations, plain YAML, errors section)
- CLI: `spectre config show` subcommand (exit codes, --no-provenance)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


def _make_sr2_home(tmp_path: Path) -> Path:
    sr2_home = tmp_path / "sr2home"
    sr2_home.mkdir()
    return sr2_home


def _make_cwd(tmp_path: Path) -> Path:
    cwd = tmp_path / "project"
    cwd.mkdir()
    return cwd


# ---------------------------------------------------------------------------
# Import guard — these names must exist in config.py
# ---------------------------------------------------------------------------

class TestImports:
    def test_provenance_value_importable(self):
        from sr2_spectre.config import ProvenanceValue  # noqa: F401

    def test_load_config_with_provenance_importable(self):
        from sr2_spectre.config import load_config_with_provenance  # noqa: F401

    def test_format_dry_run_importable(self):
        from sr2_spectre.config import format_dry_run  # noqa: F401


# ---------------------------------------------------------------------------
# ProvenanceValue dataclass
# ---------------------------------------------------------------------------

class TestProvenanceValue:
    def test_has_value_attribute(self):
        from sr2_spectre.config import ProvenanceValue
        pv = ProvenanceValue(value="claude-3-opus", source="global/config.yaml (global)")
        assert pv.value == "claude-3-opus"

    def test_has_source_attribute(self):
        from sr2_spectre.config import ProvenanceValue
        pv = ProvenanceValue(value=42, source="project/.spectre.yaml (project)")
        assert pv.source == "project/.spectre.yaml (project)"

    def test_value_can_be_any_type(self):
        from sr2_spectre.config import ProvenanceValue
        pv_dict = ProvenanceValue(value={"nested": "dict"}, source="file.yaml (extends)")
        assert pv_dict.value == {"nested": "dict"}


# ---------------------------------------------------------------------------
# load_config_with_provenance — return shape
# ---------------------------------------------------------------------------

class TestLoadConfigWithProvenanceReturnShape:
    def test_returns_tuple_of_two(self, tmp_path):
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)
        result = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_dict(self, tmp_path):
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)
        config, _ = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert isinstance(config, dict)

    def test_second_element_is_dict(self, tmp_path):
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)
        _, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert isinstance(provenance, dict)

    def test_empty_config_returns_empty_dicts(self, tmp_path):
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)
        config, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert config == {}
        assert provenance == {}


# ---------------------------------------------------------------------------
# load_config_with_provenance — single tier provenance
# ---------------------------------------------------------------------------

class TestSingleTierProvenance:
    def test_single_tier_key_has_provenance_value(self, tmp_path):
        """Top-level keys from a single file should have ProvenanceValue provenance."""
        from sr2_spectre.config import ProvenanceValue, load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)
        _write_yaml(cwd / ".spectre.yaml", {"agent": {"name": "spectre"}})

        _, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        assert "agent" in provenance
        pv = provenance["agent"]
        assert isinstance(pv, ProvenanceValue)

    def test_single_tier_provenance_source_contains_filename(self, tmp_path):
        """The provenance source string should reference the contributing file."""
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)
        _write_yaml(cwd / ".spectre.yaml", {"agent": {"name": "spectre"}})

        _, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        source = provenance["agent"].source
        # Should reference the actual file name (not just say "unknown")
        assert ".spectre.yaml" in source or "spectre" in source.lower()

    def test_global_tier_key_provenance_source_references_global_file(self, tmp_path):
        """A key set only in the global config.yaml should have provenance from that file."""
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)
        _write_yaml(sr2_home / "config.yaml", {"agent": {"name": "global-agent"}})

        _, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        source = provenance["agent"].source
        assert "config.yaml" in source or "global" in source.lower()


# ---------------------------------------------------------------------------
# load_config_with_provenance — multi-tier provenance (child overrides)
# ---------------------------------------------------------------------------

class TestMultiTierProvenance:
    def test_child_overrides_scalar_wins_child_provenance(self, tmp_path):
        """When child overrides parent's scalar, provenance points to child's file."""
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)

        # Tier 1 (global): sets "agent"
        _write_yaml(sr2_home / "config.yaml", {"agent": {"name": "global-agent"}})
        # Tier 3 (project): overrides "agent"
        _write_yaml(cwd / ".spectre.yaml", {"agent": {"name": "project-agent"}})

        config, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        # The merged config value should be the child's value
        assert config["agent"]["name"] == "project-agent"
        # The provenance source should reference the project file, not global
        source = provenance["agent"].source
        # It should NOT be the global config; it should point to .spectre.yaml
        assert ".spectre.yaml" in source or "project" in source.lower()

    def test_parent_key_not_overridden_keeps_parent_provenance(self, tmp_path):
        """A key set in global but not overridden by project should have global provenance."""
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)

        # Tier 1 (global): sets "heartbeat"
        _write_yaml(sr2_home / "config.yaml", {"heartbeat": {"interval_seconds": 30}})
        # Tier 3 (project): sets only "agent", does NOT override heartbeat
        _write_yaml(cwd / ".spectre.yaml", {"agent": {"name": "my-agent"}})

        _, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        # heartbeat provenance should reference the global config, not the project file
        assert "heartbeat" in provenance
        source = provenance["heartbeat"].source
        assert "config.yaml" in source or "global" in source.lower()
        assert ".spectre.yaml" not in source

    def test_provenance_map_has_same_top_level_keys_as_config(self, tmp_path):
        """The provenance map should have the same top-level keys as the merged config."""
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)
        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"model": "gpt-4o"}},
        })

        config, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        # Every top-level key in config should appear in provenance
        for key in config:
            assert key in provenance, f"Missing provenance for key: {key}"


# ---------------------------------------------------------------------------
# load_config_with_provenance — extends chain provenance
# ---------------------------------------------------------------------------

class TestExtendsProvenance:
    def test_value_from_base_file_has_extends_provenance(self, tmp_path):
        """A key inherited from an extends base should have provenance referencing that base."""
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)

        # Create base file
        base = cwd / "agents" / "base.yaml"
        base.parent.mkdir()
        _write_yaml(base, {"heartbeat": {"interval_seconds": 60}})

        # Project file extends base, doesn't override heartbeat
        _write_yaml(cwd / ".spectre.yaml", {
            "extends": "agents/base.yaml",
            "agent": {"name": "derived-agent"},
        })

        config, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        # heartbeat should be present (inherited from base)
        assert "heartbeat" in config
        # Provenance should reference the base file (not .spectre.yaml)
        source = provenance["heartbeat"].source
        assert "base.yaml" in source or "extends" in source.lower()

    def test_child_override_wins_over_extends_base(self, tmp_path):
        """A key in the extending file overrides the base — provenance points to extending file."""
        from sr2_spectre.config import load_config_with_provenance
        sr2_home = _make_sr2_home(tmp_path)
        cwd = _make_cwd(tmp_path)

        base = cwd / "base.yaml"
        _write_yaml(base, {"agent": {"name": "base-agent"}})
        _write_yaml(cwd / ".spectre.yaml", {
            "extends": "base.yaml",
            "agent": {"name": "override-agent"},
        })

        config, provenance = load_config_with_provenance(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        assert config["agent"]["name"] == "override-agent"
        # Provenance should reference the overriding file (.spectre.yaml)
        source = provenance["agent"].source
        assert ".spectre.yaml" in source


# ---------------------------------------------------------------------------
# format_dry_run — provenance annotations
# ---------------------------------------------------------------------------

class TestFormatDryRunProvenance:
    def test_show_provenance_true_output_contains_source_annotation(self, tmp_path):
        """With show_provenance=True, the output should contain source annotations."""
        from sr2_spectre.config import ProvenanceValue, format_dry_run

        config = {"agent": {"name": "spectre"}}
        provenance = {
            "agent": ProvenanceValue(
                value={"name": "spectre"},
                source=".spectre.yaml (project)",
            )
        }
        output = format_dry_run(config, provenance, errors=[], show_provenance=True)

        # The source string should appear somewhere in the output
        assert ".spectre.yaml" in output or "project" in output

    def test_show_provenance_true_uses_comment_style(self, tmp_path):
        """With show_provenance=True, annotations appear as YAML comments (# ←)."""
        from sr2_spectre.config import ProvenanceValue, format_dry_run

        config = {"agent": {"name": "spectre"}}
        provenance = {
            "agent": ProvenanceValue(
                value={"name": "spectre"},
                source=".spectre.yaml (project)",
            )
        }
        output = format_dry_run(config, provenance, errors=[], show_provenance=True)

        # Should have a # comment in the output
        assert "#" in output

    def test_show_provenance_false_output_is_plain_yaml(self):
        """With show_provenance=False, output has no source annotations (← marker absent)."""
        from sr2_spectre.config import ProvenanceValue, format_dry_run

        config = {"agent": {"name": "spectre"}, "models": {"default": {"model": "gpt-4o"}}}
        provenance = {
            "agent": ProvenanceValue(value={"name": "spectre"}, source="file.yaml (project)"),
            "models": ProvenanceValue(value={"default": {"model": "gpt-4o"}}, source="file.yaml (project)"),
        }
        output = format_dry_run(config, provenance, errors=[], show_provenance=False)

        # The specific provenance arrow marker must not be present
        assert "# ←" not in output
        # The source label strings must not appear as annotations
        assert "(project)" not in output

    def test_show_provenance_false_output_parses_as_valid_yaml(self):
        """Plain YAML output (show_provenance=False) should be parseable by yaml.safe_load."""
        from sr2_spectre.config import ProvenanceValue, format_dry_run

        config = {"agent": {"name": "spectre"}, "timeout": 30}
        provenance = {
            "agent": ProvenanceValue(value={"name": "spectre"}, source="file.yaml (project)"),
            "timeout": ProvenanceValue(value=30, source="file.yaml (project)"),
        }
        output = format_dry_run(config, provenance, errors=[], show_provenance=False)

        # Extract the YAML part (before any errors section)
        parsed = yaml.safe_load(output)
        assert isinstance(parsed, dict)
        assert parsed["agent"]["name"] == "spectre"
        assert parsed["timeout"] == 30


# ---------------------------------------------------------------------------
# format_dry_run — errors section
# ---------------------------------------------------------------------------

class TestFormatDryRunErrors:
    def test_no_errors_produces_no_error_section_or_empty(self):
        """With errors=[], the output should not have any error content."""
        from sr2_spectre.config import ProvenanceValue, format_dry_run

        config = {"agent": {"name": "spectre"}}
        provenance = {
            "agent": ProvenanceValue(value={"name": "spectre"}, source="file.yaml (project)"),
        }
        output = format_dry_run(config, provenance, errors=[])

        # Either no errors section, or an empty one — but no actual error messages
        assert "agent.name must be" not in output
        # No error keyword from validate_config should appear
        assert "must be a non-empty string" not in output

    def test_errors_listed_in_output(self):
        """When errors are provided, they should appear in the output."""
        from sr2_spectre.config import ProvenanceValue, format_dry_run

        config = {"agent": {"name": ""}}
        provenance = {
            "agent": ProvenanceValue(value={"name": ""}, source="file.yaml (project)"),
        }
        errors = ["agent.name must be a non-empty string", "models required"]
        output = format_dry_run(config, provenance, errors=errors)

        assert "agent.name must be a non-empty string" in output
        assert "models required" in output

    def test_multiple_errors_all_present_in_output(self):
        """All errors in the list should appear in the output string."""
        from sr2_spectre.config import ProvenanceValue, format_dry_run

        config = {}
        provenance = {}
        errors = ["error one", "error two", "error three"]
        output = format_dry_run(config, provenance, errors=errors)

        for e in errors:
            assert e in output


# ---------------------------------------------------------------------------
# format_dry_run — return type
# ---------------------------------------------------------------------------

class TestFormatDryRunReturnType:
    def test_returns_string(self):
        from sr2_spectre.config import format_dry_run
        output = format_dry_run({}, {}, errors=[])
        assert isinstance(output, str)

    def test_non_empty_string_for_non_empty_config(self):
        from sr2_spectre.config import ProvenanceValue, format_dry_run
        config = {"agent": {"name": "spectre"}}
        provenance = {"agent": ProvenanceValue(value={"name": "spectre"}, source="f.yaml (project)")}
        output = format_dry_run(config, provenance, errors=[])
        assert len(output) > 0


# ---------------------------------------------------------------------------
# format_dry_run — include_content removed (spc-17)
# ---------------------------------------------------------------------------

class TestIncludeContentRemoved:
    def test_format_dry_run_no_include_content_parameter(self):
        """format_dry_run no longer accepts include_content (dead param removed)."""
        from sr2_spectre.config import format_dry_run
        import inspect

        sig = inspect.signature(format_dry_run)
        param_names = list(sig.parameters.keys())
        assert "include_content" not in param_names

    def test_format_dry_run_calling_with_include_content_raises(self):
        """Passing include_content to format_dry_run raises TypeError."""
        from sr2_spectre.config import format_dry_run
        with pytest.raises(TypeError):
            format_dry_run({}, {}, errors=[], include_content=True)


# ---------------------------------------------------------------------------
# CLI — --include-content flag removed (spc-17)
# ---------------------------------------------------------------------------

class TestIncludeContentFlagRemoved:
    def test_include_content_flag_not_in_parser(self, tmp_path):
        """The --include-content flag no longer exists on 'config show'."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {"agent": {"name": "spectre"}})

        result = _run_cli(
            ["config", "show", f"--sr2-home={sr2_home}", "--include-content"],
            cwd=str(cwd),
        )
        # Should error — unrecognized argument
        assert result.returncode != 0
        assert "include-content" in result.stderr or "unrecognized" in result.stderr.lower()


# ---------------------------------------------------------------------------
# CLI — spectre config show
# ---------------------------------------------------------------------------

def _run_cli(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run the sr2-spectre CLI as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "sr2_spectre.cli"] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


class TestCliConfigShow:
    def test_config_show_valid_exits_zero(self, tmp_path):
        """'config show' with a valid config exits 0."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        })

        result = _run_cli(
            ["config", "show", f"--sr2-home={sr2_home}"],
            cwd=str(cwd),
        )
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_config_show_invalid_exits_one(self, tmp_path):
        """'config show' with an invalid config exits 1."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"base_url": "https://example.com"}},  # missing required 'model' field
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        })

        result = _run_cli(
            ["config", "show", f"--sr2-home={sr2_home}"],
            cwd=str(cwd),
        )
        assert result.returncode == 1, f"Expected exit 1, got {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_config_show_prints_to_stdout(self, tmp_path):
        """'config show' produces output on stdout."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
        })

        result = _run_cli(
            ["config", "show", f"--sr2-home={sr2_home}"],
            cwd=str(cwd),
        )
        assert len(result.stdout) > 0

    def test_config_show_no_provenance_flag(self, tmp_path):
        """'--no-provenance' flag produces output without source annotations."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        })

        result = _run_cli(
            ["config", "show", f"--sr2-home={sr2_home}", "--no-provenance"],
            cwd=str(cwd),
        )
        assert result.returncode == 0
        # No provenance comment annotations in output
        assert "# ←" not in result.stdout

    def test_config_show_with_provenance_contains_source_annotation(self, tmp_path):
        """Default (provenance enabled) output contains source annotation comments."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        })

        result = _run_cli(
            ["config", "show", f"--sr2-home={sr2_home}"],
            cwd=str(cwd),
        )
        assert result.returncode == 0
        # Output should contain a # comment (provenance annotation)
        assert "#" in result.stdout

    def test_config_show_invalid_prints_errors_to_stdout(self, tmp_path):
        """'config show' with invalid config prints error messages to stdout."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"base_url": "https://example.com"}},  # missing required 'model' field
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        })

        result = _run_cli(
            ["config", "show", f"--sr2-home={sr2_home}"],
            cwd=str(cwd),
        )
        # Errors should be in stdout (the formatted report), not just stderr
        assert "model" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_config_show_does_not_launch_agent(self, tmp_path):
        """'config show' must not attempt to load any LLM or Agent."""
        # This is a negative test: we run with no model configured.
        # If an agent were launched, it would fail loudly on missing model config.
        # If it exits cleanly (0 or 1), no agent was launched.
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"model": "gpt-4o"}},
            "pipeline": {"layers": [{"name": "l1", "resolvers": [{"type": "prompt"}], "target": "default"}]},
        })

        result = _run_cli(
            ["config", "show", f"--sr2-home={sr2_home}"],
            cwd=str(cwd),
        )
        # Should exit cleanly (0 or 1) without crashing
        assert result.returncode in (0, 1)
        # Must not attempt LLM connection — no async connection errors expected
        assert "ConnectionError" not in result.stderr
        assert "asyncio" not in result.stderr or "event loop" not in result.stderr
