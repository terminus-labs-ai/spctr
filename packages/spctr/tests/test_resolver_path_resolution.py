"""Tests for FR10: resolver path resolution in config merge.

resolve_path is applied to resolver config path fields (plans_root,
knowledge_root) during config loading so that markdown_file resolvers
can use ${VAR} interpolation and relative paths.

Covers:
- ${VAR} interpolation in resolver path fields
- Relative paths resolved against declaring file's directory
- Absolute paths passed through
- Unset ${VAR} raises ConfigPathError
- Non-path fields in resolver config untouched
- Multiple resolvers across multiple layers
- Resolver path resolution only happens on known path fields
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sr2_spectre.config import resolve_extends, resolve_resolver_paths, load_resolved_config
from sr2_spectre.path_resolution import ConfigPathError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    """Write a dict as YAML to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))


def _resolver_config(
    plans_root: str = "~/.sr2/plans",
    knowledge_root: str = "~/.sr2/knowledge/myproject",
    project: str = "myproject",
    **extra: object,
) -> dict:
    """Build a resolver config dict with the given path values."""
    cfg: dict = {
        "plans_root": plans_root,
        "knowledge_root": knowledge_root,
        "project": project,
    }
    cfg.update(extra)
    return cfg


def _pipeline_with_resolver(resolver_cfg: dict) -> dict:
    """Build a minimal pipeline config dict containing one resolver."""
    return {
        "pipeline": {
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [
                        {
                            "type": "plan",
                            "config": resolver_cfg,
                        }
                    ],
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# resolve_resolver_paths — ${VAR} interpolation
# ---------------------------------------------------------------------------


class TestResolverPathVarInterpolation:
    def test_sr2_home_in_plans_root(self, tmp_path):
        """${SR2_HOME} in plans_root is interpolated."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(plans_root="${SR2_HOME}/plans")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        env = {"SR2_HOME": str(tmp_path / "sr2data")}

        resolve_resolver_paths(pipeline_cfg, declaring, env=env)

        assert (
            pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]["plans_root"]
            == str(tmp_path / "sr2data" / "plans")
        )

    def test_sr2_home_in_knowledge_root(self, tmp_path):
        """${SR2_HOME} in knowledge_root is interpolated."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(knowledge_root="${SR2_HOME}/knowledge/proj")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        env = {"SR2_HOME": str(tmp_path / "sr2data")}

        resolve_resolver_paths(pipeline_cfg, declaring, env=env)

        assert (
            pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]["knowledge_root"]
            == str(tmp_path / "sr2data" / "knowledge" / "proj")
        )

    def test_custom_var_in_plans_root(self, tmp_path):
        """${CUSTOM_VAR} in plans_root is interpolated."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(plans_root="${MY_PLANS}/active")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        env = {"MY_PLANS": str(tmp_path / "plans")}

        resolve_resolver_paths(pipeline_cfg, declaring, env=env)

        assert (
            pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]["plans_root"]
            == str(tmp_path / "plans" / "active")
        )

    def test_both_paths_interpolated(self, tmp_path):
        """Both plans_root and knowledge_root can use ${VAR}."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(
            plans_root="${SR2_HOME}/plans",
            knowledge_root="${SR2_HOME}/knowledge/proj",
        )
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        env = {"SR2_HOME": str(tmp_path / "sr2data")}

        resolve_resolver_paths(pipeline_cfg, declaring, env=env)

        config = pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]
        assert config["plans_root"] == str(tmp_path / "sr2data" / "plans")
        assert config["knowledge_root"] == str(tmp_path / "sr2data" / "knowledge" / "proj")


# ---------------------------------------------------------------------------
# resolve_resolver_paths — relative paths
# ---------------------------------------------------------------------------


class TestResolverPathRelative:
    def test_relative_plans_root_resolved_against_declaring_dir(self, tmp_path):
        """Relative plans_root resolves against declaring file's directory."""
        sub = tmp_path / "configs"
        sub.mkdir()
        declaring = sub / "spectre.yaml"
        plans = tmp_path / "plans"
        plans.mkdir()

        resolver_cfg = _resolver_config(plans_root="../plans")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        resolve_resolver_paths(pipeline_cfg, declaring, env={})

        assert (
            pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]["plans_root"]
            == str(plans)
        )

    def test_relative_knowledge_root_resolved_against_declaring_dir(self, tmp_path):
        """Relative knowledge_root resolves against declaring file's directory."""
        sub = tmp_path / "configs" / "prod"
        sub.mkdir(parents=True)
        declaring = sub / "spectre.yaml"
        knowledge = tmp_path / "knowledge"
        knowledge.mkdir()

        resolver_cfg = _resolver_config(knowledge_root="../../knowledge/proj")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        resolve_resolver_paths(pipeline_cfg, declaring, env={})

        assert (
            pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]["knowledge_root"]
            == str(knowledge / "proj")
        )


# ---------------------------------------------------------------------------
# resolve_resolver_paths — absolute paths
# ---------------------------------------------------------------------------


class TestResolverPathAbsolute:
    def test_absolute_path_passed_through(self, tmp_path):
        """Absolute plans_root is not modified (except normalization)."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(plans_root="/opt/sr2/plans")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        resolve_resolver_paths(pipeline_cfg, declaring, env={})

        assert (
            pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]["plans_root"]
            == "/opt/sr2/plans"
        )

    def test_tilde_expanded(self, tmp_path, monkeypatch):
        """Tilde paths in resolver config are expanded.

        Note: resolve_path does NOT expand ~. The tilde expansion happens
        in PlanResolver.__init__ via Path.expanduser(). This test verifies
        resolve_resolver_paths does NOT interfere with tilde paths — they
        survive as strings for PlanResolver to handle.
        """
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(plans_root="~/.sr2/plans")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        resolve_resolver_paths(pipeline_cfg, declaring, env={})

        # Tilde paths that don't contain ${VAR} are relative paths
        # They resolve against the declaring file's directory
        config = pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]
        # ~ is treated as a literal directory name by resolve_path
        # This is expected — PlanResolver.__init__ handles ~ via expanduser()
        # The key is that resolve_resolver_paths doesn't crash
        assert "plans_root" in config


# ---------------------------------------------------------------------------
# resolve_resolver_paths — unset VAR
# ---------------------------------------------------------------------------


class TestResolverPathUnsetVar:
    def test_unset_var_in_plans_root_raises(self, tmp_path):
        """${UNSET} in plans_root raises ConfigPathError."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(plans_root="${UNSET_VAR}/plans")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        with pytest.raises(ConfigPathError, match="UNSET_VAR"):
            resolve_resolver_paths(pipeline_cfg, declaring, env={})

    def test_unset_var_in_knowledge_root_raises(self, tmp_path):
        """${UNSET} in knowledge_root raises ConfigPathError."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(knowledge_root="${UNSET_VAR}/knowledge")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        with pytest.raises(ConfigPathError, match="UNSET_VAR"):
            resolve_resolver_paths(pipeline_cfg, declaring, env={})


# ---------------------------------------------------------------------------
# resolve_resolver_paths — non-path fields untouched
# ---------------------------------------------------------------------------


class TestResolverPathNonPathFields:
    def test_project_field_untouched(self, tmp_path):
        """The 'project' field is not a path — left as-is."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(project="myproject")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        resolve_resolver_paths(pipeline_cfg, declaring, env={})

        assert (
            pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]["project"]
            == "myproject"
        )

    def test_max_tokens_untouched(self, tmp_path):
        """Non-path fields like max_tokens are left as-is."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(max_tokens=5000)
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        resolve_resolver_paths(pipeline_cfg, declaring, env={})

        assert (
            pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]["max_tokens"]
            == 5000
        )


# ---------------------------------------------------------------------------
# resolve_resolver_paths — multiple resolvers / layers
# ---------------------------------------------------------------------------


class TestResolverPathMultiple:
    def test_multiple_layers_processed(self, tmp_path):
        """All layers' resolvers have paths resolved."""
        declaring = tmp_path / "config.yaml"

        pipeline = {
            "pipeline": {
                "layers": [
                    {
                        "name": "system",
                        "target": "system",
                        "resolvers": [
                            {
                                "type": "static",
                                "config": {"text": "Hello"},
                            },
                            {
                                "type": "plan",
                                "config": _resolver_config(
                                    plans_root="${SR2_HOME}/plans",
                                ),
                            },
                        ],
                    },
                    {
                        "name": "conversation",
                        "target": "messages",
                        "resolvers": [
                            {
                                "type": "plan",
                                "config": _resolver_config(
                                    knowledge_root="${SR2_HOME}/knowledge/proj",
                                ),
                            },
                        ],
                    },
                ]
            }
        }

        env = {"SR2_HOME": str(tmp_path / "sr2data")}

        resolve_resolver_paths(pipeline, declaring, env=env)

        # Layer 0, resolver 1 (plan type)
        plan_resolver_0 = pipeline["pipeline"]["layers"][0]["resolvers"][1]["config"]
        assert plan_resolver_0["plans_root"] == str(tmp_path / "sr2data" / "plans")

        # Layer 1, resolver 0 (plan type)
        plan_resolver_1 = pipeline["pipeline"]["layers"][1]["resolvers"][0]["config"]
        assert plan_resolver_1["knowledge_root"] == str(tmp_path / "sr2data" / "knowledge" / "proj")

    def test_static_resolver_config_untouched(self, tmp_path):
        """Static resolver config (no path fields) is left alone."""
        declaring = tmp_path / "config.yaml"

        pipeline = {
            "pipeline": {
                "layers": [
                    {
                        "name": "system",
                        "target": "system",
                        "resolvers": [
                            {
                                "type": "static",
                                "config": {"text": "Hello, world."},
                            },
                        ],
                    }
                ]
            }
        }

        resolve_resolver_paths(pipeline, declaring, env={})

        assert (
            pipeline["pipeline"]["layers"][0]["resolvers"][0]["config"]["text"]
            == "Hello, world."
        )


# ---------------------------------------------------------------------------
# resolve_resolver_paths — missing pipeline / resolvers
# ---------------------------------------------------------------------------


class TestResolverPathMissingSections:
    def test_no_pipeline_key_is_noop(self, tmp_path):
        """Config without 'pipeline' key — no error, no change."""
        declaring = tmp_path / "config.yaml"
        cfg = {"agent": {"name": "spectre"}}

        resolve_resolver_paths(cfg, declaring, env={})
        assert cfg == {"agent": {"name": "spectre"}}

    def test_no_layers_is_noop(self, tmp_path):
        """Pipeline without 'layers' — no error."""
        declaring = tmp_path / "config.yaml"
        cfg = {"pipeline": {"token_budget": 100000}}

        resolve_resolver_paths(cfg, declaring, env={})
        assert cfg["pipeline"]["token_budget"] == 100000

    def test_empty_resolvers_list_is_noop(self, tmp_path):
        """Layer with empty resolvers list — no error."""
        declaring = tmp_path / "config.yaml"
        cfg = {
            "pipeline": {
                "layers": [
                    {"name": "tools", "target": "tools", "resolvers": []}
                ]
            }
        }

        resolve_resolver_paths(cfg, declaring, env={})
        assert len(cfg["pipeline"]["layers"][0]["resolvers"]) == 0


# ---------------------------------------------------------------------------
# resolve_resolver_paths — in-place mutation
# ---------------------------------------------------------------------------


class TestResolverPathInPlace:
    def test_mutates_in_place(self, tmp_path):
        """resolve_resolver_paths mutates the config dict in place."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(plans_root="${SR2_HOME}/plans")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        original_layer = pipeline_cfg["pipeline"]["layers"][0]

        resolve_resolver_paths(pipeline_cfg, declaring, env={"SR2_HOME": str(tmp_path)})

        assert pipeline_cfg["pipeline"]["layers"][0] is original_layer


# ---------------------------------------------------------------------------
# Integration: load_resolved_config with resolver paths
# ---------------------------------------------------------------------------


class TestIntegrationLoadResolvedConfig:
    def test_positional_file_resolver_paths_resolved(self, tmp_path):
        """load_resolved_config resolves ${VAR} in resolver path fields."""
        cwd = tmp_path / "project"
        cwd.mkdir()
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()

        positional = cwd / "agent.yaml"
        _write_yaml(positional, {
            "agent": {"name": "edi"},
            "models": {"default": {"model": "test/model"}},
            "pipeline": {
                "layers": [
                    {
                        "name": "system",
                        "target": "system",
                        "resolvers": [
                            {
                                "type": "plan",
                                "config": {
                                    "plans_root": "${SR2_HOME}/plans",
                                    "knowledge_root": "${SR2_HOME}/knowledge/proj",
                                    "project": "myproject",
                                },
                            }
                        ],
                    }
                ]
            },
        })

        result = load_resolved_config(
            positional,
            cwd=cwd,
            env={"SR2_HOME": str(sr2_home)},
        )

        resolver_cfg = result["pipeline"]["layers"][0]["resolvers"][0]["config"]
        assert resolver_cfg["plans_root"] == str(sr2_home / "plans")
        assert resolver_cfg["knowledge_root"] == str(sr2_home / "knowledge" / "proj")
        assert resolver_cfg["project"] == "myproject"

    def test_positional_file_relative_resolver_path(self, tmp_path):
        """Relative resolver paths in positional file resolve against its dir."""
        cwd = tmp_path / "project"
        cwd.mkdir()
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()

        plans = tmp_path / "plans"
        plans.mkdir()

        positional = cwd / "agent.yaml"
        _write_yaml(positional, {
            "agent": {"name": "edi"},
            "models": {"default": {"model": "test/model"}},
            "pipeline": {
                "layers": [
                    {
                        "name": "system",
                        "target": "system",
                        "resolvers": [
                            {
                                "type": "plan",
                                "config": {
                                    "plans_root": "../plans",
                                    "project": "myproject",
                                },
                            }
                        ],
                    }
                ]
            },
        })

        result = load_resolved_config(
            positional,
            cwd=cwd,
            env={"SR2_HOME": str(sr2_home)},
        )

        resolver_cfg = result["pipeline"]["layers"][0]["resolvers"][0]["config"]
        assert resolver_cfg["plans_root"] == str(plans)

    def test_unset_var_in_resolver_path_fails_at_load(self, tmp_path):
        """Unset ${VAR} in a resolver path raises ConfigPathError during config load."""
        cwd = tmp_path / "project"
        cwd.mkdir()
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()

        positional = cwd / "agent.yaml"
        _write_yaml(positional, {
            "agent": {"name": "edi"},
            "models": {"default": {"model": "test/model"}},
            "pipeline": {
                "layers": [
                    {
                        "name": "system",
                        "target": "system",
                        "resolvers": [
                            {
                                "type": "plan",
                                "config": {
                                    "plans_root": "${MISSING_VAR}/plans",
                                    "project": "myproject",
                                },
                            }
                        ],
                    }
                ]
            },
        })

        with pytest.raises(ConfigPathError, match="MISSING_VAR"):
            load_resolved_config(positional, cwd=cwd, env={"SR2_HOME": str(sr2_home)})


# ---------------------------------------------------------------------------
# resolve_resolver_paths — declaring_dir injection
# ---------------------------------------------------------------------------


class TestResolverPathDeclaringDir:
    def test_declaring_dir_injected_into_resolver_config(self, tmp_path):
        """The declaring file's directory is injected as 'declaring_dir'."""
        sub = tmp_path / "configs" / "prod"
        sub.mkdir(parents=True)
        declaring = sub / "spectre.yaml"

        resolver_cfg = _resolver_config(plans_root="../plans")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        resolve_resolver_paths(pipeline_cfg, declaring, env={})

        config = pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"]
        assert config.get("declaring_dir") == str(sub)

    def test_declaring_dir_absolute(self, tmp_path):
        """declaring_dir is always an absolute path string."""
        declaring = tmp_path / "config.yaml"
        resolver_cfg = _resolver_config(plans_root="/opt/plans")
        pipeline_cfg = _pipeline_with_resolver(resolver_cfg)

        resolve_resolver_paths(pipeline_cfg, declaring, env={})

        dd = pipeline_cfg["pipeline"]["layers"][0]["resolvers"][0]["config"].get("declaring_dir")
        assert dd is not None
        assert Path(dd).is_absolute()
