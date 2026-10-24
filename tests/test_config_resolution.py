"""Tests for four-tier config resolution and SR2_HOME resolution (obsidian-17w.3)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sr2_spectre.config import (
    RemindersConfig,
    load_merged_config,
    resolve_sr2_home,
)


# ---------------------------------------------------------------------------
# resolve_sr2_home
# ---------------------------------------------------------------------------

class TestResolveSr2Home:
    def test_unset_defaults_to_tilde_sr2(self):
        """When SR2_HOME is not in env, return ~/.sr2 (expanduser + resolve)."""
        result = resolve_sr2_home(env={})
        expected = Path("~/.sr2").expanduser().resolve()
        assert result == expected

    def test_unset_is_absolute(self):
        """Default result must be an absolute path."""
        result = resolve_sr2_home(env={})
        assert result.is_absolute()

    def test_relative_path_resolved_to_absolute(self, tmp_path):
        """Relative SR2_HOME is resolved to absolute."""
        env = {"SR2_HOME": "relative/path"}
        result = resolve_sr2_home(env=env)
        assert result.is_absolute()
        # Should expand relative to the current working directory
        expected = Path("relative/path").expanduser().resolve()
        assert result == expected

    def test_absolute_path_returned_resolved(self, tmp_path):
        """Absolute SR2_HOME is returned as resolved Path."""
        env = {"SR2_HOME": str(tmp_path)}
        result = resolve_sr2_home(env=env)
        assert result == tmp_path.resolve()
        assert result.is_absolute()

    def test_tilde_path_expanded(self, tmp_path):
        """SR2_HOME with ~ is expanded."""
        env = {"SR2_HOME": "~/.sr2-custom"}
        result = resolve_sr2_home(env=env)
        expected = Path("~/.sr2-custom").expanduser().resolve()
        assert result == expected
        assert result.is_absolute()

    def test_none_env_uses_os_environ(self, monkeypatch):
        """When env=None, reads from os.environ."""
        monkeypatch.setenv("SR2_HOME", "/tmp/sr2-test-home")
        result = resolve_sr2_home(env=None)
        assert result == Path("/tmp/sr2-test-home").resolve()


# ---------------------------------------------------------------------------
# RemindersConfig.resolved_store_path — SR2_HOME-rooted default (spc-160)
# ---------------------------------------------------------------------------

class TestRemindersStorePathResolution:
    """AC1-AC4: default store path follows SR2_HOME (spc-160).

    The autouse conftest fixture already points SR2_HOME at a per-test
    temp dir; tests below set or delete it explicitly where required.
    """

    def test_default_follows_sr2_home_env(self, monkeypatch, tmp_path):
        """AC1: empty store_path resolves under the environment's SR2_HOME."""
        home = tmp_path / "home"
        monkeypatch.setenv("SR2_HOME", str(home))
        path = Path(RemindersConfig(enabled=True).resolved_store_path("miranda"))
        assert path == (
            resolve_sr2_home() / "state" / "miranda" / "reminders.sqlite"
        )

    def test_default_tracks_sr2_home_changes(self, monkeypatch, tmp_path):
        """AC1: resolution is dynamic — a later SR2_HOME value is followed."""
        first = tmp_path / "first"
        monkeypatch.setenv("SR2_HOME", str(first))
        assert (
            Path(RemindersConfig(enabled=True).resolved_store_path("miranda"))
            == resolve_sr2_home() / "state" / "miranda" / "reminders.sqlite"
        )
        second = tmp_path / "second"
        monkeypatch.setenv("SR2_HOME", str(second))
        path = Path(RemindersConfig(enabled=True).resolved_store_path("miranda"))
        assert path == resolve_sr2_home() / "state" / "miranda" / "reminders.sqlite"
        assert path.is_relative_to(second.resolve())

    def test_agent_name_interpolated_into_default(self, monkeypatch, tmp_path):
        """AC1: the agent name parameter lands in the default, not a fixed name."""
        monkeypatch.setenv("SR2_HOME", str(tmp_path / "home"))
        path = Path(RemindersConfig(enabled=True).resolved_store_path("other-bot"))
        assert path == (
            resolve_sr2_home() / "state" / "other-bot" / "reminders.sqlite"
        )

    def test_default_without_sr2_home_env_is_tilde_sr2_state(self, monkeypatch):
        """AC2: SR2_HOME unset — default is ~/.sr2/state/<agent>/reminders.sqlite."""
        monkeypatch.delenv("SR2_HOME", raising=False)
        path = Path(RemindersConfig(enabled=True).resolved_store_path("miranda"))
        expected = (
            Path("~/.sr2").expanduser().resolve()
            / "state"
            / "miranda"
            / "reminders.sqlite"
        )
        assert path == expected

    def test_explicit_store_path_used_verbatim(self, monkeypatch, tmp_path):
        """AC3: an explicit store_path is never re-rooted under SR2_HOME."""
        monkeypatch.setenv("SR2_HOME", str(tmp_path / "home"))
        explicit = tmp_path / "elsewhere" / "reminders.sqlite"
        path = RemindersConfig(
            enabled=True, store_path=str(explicit)
        ).resolved_store_path("miranda")
        assert path == str(explicit)

    def test_explicit_store_path_substitutes_agent(self, monkeypatch, tmp_path):
        """AC3: {agent} still substitutes in an explicit store_path."""
        monkeypatch.setenv("SR2_HOME", str(tmp_path / "home"))
        raw = str(tmp_path / "data" / "{agent}" / "reminders.sqlite")
        path = RemindersConfig(
            enabled=True, store_path=raw
        ).resolved_store_path("miranda")
        assert path == str(tmp_path / "data" / "miranda" / "reminders.sqlite")

    def test_explicit_store_path_expands_tilde(self, monkeypatch, tmp_path):
        """AC3: ~ still expands in an explicit store_path."""
        monkeypatch.setenv("SR2_HOME", str(tmp_path / "home"))
        path = RemindersConfig(
            enabled=True, store_path="~/custom-reminders/{agent}.sqlite"
        ).resolved_store_path("miranda")
        assert path == str(
            Path("~").expanduser() / "custom-reminders" / "miranda.sqlite"
        )

    def test_default_store_creation_stays_inside_sr2_home(
        self, monkeypatch, tmp_path
    ):
        """AC4: creating the default store writes only inside the temp SR2_HOME."""
        home = tmp_path / "sr2_home"
        monkeypatch.setenv("SR2_HOME", str(home))
        path = Path(RemindersConfig(enabled=True).resolved_store_path("miranda"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert path.resolve().is_relative_to(home.resolve())
        assert path.read_bytes() == b""

    def test_enabled_config_needs_no_store_path_for_wiring(self, monkeypatch, tmp_path):
        """AC1: enabled=True is constructible with empty store_path (spc-138 regression)."""
        monkeypatch.setenv("SR2_HOME", str(tmp_path / "home"))
        block = RemindersConfig(enabled=True, timezone="UTC")
        assert block.store_path == ""
        assert block.resolved_store_path("miranda") != ""

# ---------------------------------------------------------------------------
# load_merged_config — helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# load_merged_config — SR2_HOME and tier loading
# ---------------------------------------------------------------------------

class TestLoadMergedConfig:
    def test_no_files_returns_empty_dict(self, tmp_path):
        """When no config files exist anywhere, return {}."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert result == {}

    def test_only_tier1_exists(self, tmp_path):
        """Only $SR2_HOME/config.yaml exists — return its contents."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(sr2_home / "config.yaml", {"key": "tier1"})

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert result == {"key": "tier1"}

    def test_only_tier2_exists(self, tmp_path):
        """Only $SR2_HOME/spectre.yaml exists — return its contents."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(sr2_home / "spectre.yaml", {"key": "tier2"})

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert result == {"key": "tier2"}

    def test_only_tier3_exists(self, tmp_path):
        """Only <cwd>/.spectre.yaml exists — return its contents."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {"key": "tier3"})

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert result == {"key": "tier3"}

    def test_all_three_tiers_tier3_wins(self, tmp_path):
        """All three files exist — tier 3 overrides tier 2 overrides tier 1."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(sr2_home / "config.yaml", {"key": "tier1", "from_t1": True})
        _write_yaml(sr2_home / "spectre.yaml", {"key": "tier2", "from_t2": True})
        _write_yaml(cwd / ".spectre.yaml", {"key": "tier3", "from_t3": True})

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        # Tier 3 wins the scalar "key"
        assert result["key"] == "tier3"
        # Unique keys from lower tiers survive
        assert result["from_t1"] is True
        assert result["from_t2"] is True
        assert result["from_t3"] is True

    def test_tier2_overrides_tier1(self, tmp_path):
        """Tier 2 wins over tier 1 when tier 3 absent."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(sr2_home / "config.yaml", {"key": "tier1", "shared": "from_t1"})
        _write_yaml(sr2_home / "spectre.yaml", {"key": "tier2"})

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert result["key"] == "tier2"
        assert result["shared"] == "from_t1"

    def test_empty_yaml_file_treated_as_empty_dict(self, tmp_path):
        """An empty YAML file (None from safe_load) is treated as {}."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        # Write empty files for tier 1 and tier 3; tier 2 has real content
        (sr2_home / "config.yaml").write_text("")
        _write_yaml(sr2_home / "spectre.yaml", {"key": "tier2"})
        (cwd / ".spectre.yaml").write_text("")

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert result == {"key": "tier2"}

    def test_named_list_merge_across_tiers(self, tmp_path):
        """Named lists with matching 'name:' keys are merged, not replaced."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        tier1 = {"tools": [{"name": "search", "timeout": 30, "enabled": True}]}
        tier3 = {"tools": [{"name": "search", "timeout": 60}]}

        _write_yaml(sr2_home / "config.yaml", tier1)
        _write_yaml(cwd / ".spectre.yaml", tier3)

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        tools = result["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert tools[0]["timeout"] == 60     # tier3 wins
        assert tools[0]["enabled"] is True   # tier1 survives

    def test_named_list_new_entry_appended(self, tmp_path):
        """Child tier can add new named entries to a named list."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        tier1 = {"tools": [{"name": "search", "timeout": 30}]}
        tier3 = {"tools": [{"name": "exec", "timeout": 10}]}

        _write_yaml(sr2_home / "config.yaml", tier1)
        _write_yaml(cwd / ".spectre.yaml", tier3)

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        names = [t["name"] for t in result["tools"]]
        assert "search" in names
        assert "exec" in names

    def test_cwd_defaults_to_path_cwd(self, tmp_path, monkeypatch):
        """When cwd=None, uses Path.cwd()."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        _write_yaml(project_dir / ".spectre.yaml", {"from_cwd": True})

        monkeypatch.chdir(project_dir)
        result = load_merged_config(cwd=None, env={"SR2_HOME": str(sr2_home)})
        assert result.get("from_cwd") is True

    def test_sr2_home_points_to_dir_with_config(self, tmp_path):
        """SR2_HOME env var pointing to a dir loads its config.yaml as tier 1."""
        sr2_home = tmp_path / "custom_home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(sr2_home / "config.yaml", {"source": "custom_home"})

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert result["source"] == "custom_home"

    def test_deep_dict_merge_across_tiers(self, tmp_path):
        """Nested dicts are deep-merged, not replaced."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        tier1 = {"agent": {"name": "base", "tools": [{"name": "terminal", "class_path": "x.Y"}]}}
        tier3 = {"agent": {"name": "project"}}

        _write_yaml(sr2_home / "config.yaml", tier1)
        _write_yaml(cwd / ".spectre.yaml", tier3)

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert result["agent"]["name"] == "project"
        assert result["agent"]["tools"] == [{"name": "terminal", "class_path": "x.Y"}]
