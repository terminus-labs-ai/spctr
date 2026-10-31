"""Tests for FR3: extends: chain resolution and cycle detection (obsidian-17w.4)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sr2_spectre.config import CircularExtendsError, resolve_extends, load_merged_config
from sr2_spectre.path_resolution import ConfigPathError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# Basic extends: absent
# ---------------------------------------------------------------------------

class TestResolveExtendsNoExtends:
    def test_no_extends_key_returns_config_unchanged(self, tmp_path):
        """Config without 'extends:' is returned as-is."""
        declaring = tmp_path / "config.yaml"
        declaring.write_text("")
        cfg = {"agent": {"name": "spectre"}, "timeout": 30}

        result = resolve_extends(cfg, declaring_file=declaring)

        assert result == cfg

    def test_no_extends_key_empty_dict_unchanged(self, tmp_path):
        """Empty config dict without 'extends:' is returned as-is."""
        declaring = tmp_path / "config.yaml"
        declaring.write_text("")
        cfg = {}

        result = resolve_extends(cfg, declaring_file=declaring)

        assert result == {}


# ---------------------------------------------------------------------------
# Basic extends: one level
# ---------------------------------------------------------------------------

class TestResolveExtendsSingleLevel:
    def test_parent_values_appear_in_result(self, tmp_path):
        """Values from parent appear in result when child doesn't override."""
        parent = tmp_path / "parent.yaml"
        _write_yaml(parent, {"from_parent": True, "shared": "parent_value"})

        child_cfg = {"extends": "parent.yaml", "from_child": True}
        declaring = tmp_path / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert result["from_parent"] is True
        assert result["from_child"] is True

    def test_child_values_override_parent(self, tmp_path):
        """Child values win over parent when keys conflict."""
        parent = tmp_path / "parent.yaml"
        _write_yaml(parent, {"key": "parent_value", "other": "from_parent"})

        child_cfg = {"extends": "parent.yaml", "key": "child_value"}
        declaring = tmp_path / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert result["key"] == "child_value"
        assert result["other"] == "from_parent"

    def test_extends_key_not_in_result(self, tmp_path):
        """The 'extends:' key itself must not appear in the merged result."""
        parent = tmp_path / "parent.yaml"
        _write_yaml(parent, {"base": True})

        child_cfg = {"extends": "parent.yaml", "mine": True}
        declaring = tmp_path / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert "extends" not in result

    def test_extends_relative_path_resolved_to_parent_dir(self, tmp_path):
        """Relative extends path is resolved against the declaring file's directory."""
        subdir = tmp_path / "sub"
        subdir.mkdir()

        parent = subdir / "base.yaml"
        _write_yaml(parent, {"from_base": True})

        # declaring file is in subdir, extends uses just the filename
        child_cfg = {"extends": "base.yaml", "local": True}
        declaring = subdir / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert result["from_base"] is True
        assert result["local"] is True

    def test_extends_absolute_path(self, tmp_path):
        """Absolute extends path is used as-is."""
        parent = tmp_path / "absolute_parent.yaml"
        _write_yaml(parent, {"from_absolute": True})

        child_cfg = {"extends": str(parent), "child_key": "yes"}
        declaring = tmp_path / "other_dir" / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert result["from_absolute"] is True
        assert result["child_key"] == "yes"

    def test_extends_empty_parent(self, tmp_path):
        """Extending an empty parent returns the child's own values."""
        parent = tmp_path / "empty.yaml"
        parent.write_text("")

        child_cfg = {"extends": "empty.yaml", "key": "value"}
        declaring = tmp_path / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert result["key"] == "value"
        assert "extends" not in result


# ---------------------------------------------------------------------------
# Chain resolution (A extends B extends C)
# ---------------------------------------------------------------------------

class TestResolveExtendsChain:
    def test_three_level_chain_all_values_present(self, tmp_path):
        """Three-level chain: grandparent, parent, child — all unique keys present."""
        grandparent = tmp_path / "grandparent.yaml"
        _write_yaml(grandparent, {"from_grandparent": True, "level": "grandparent"})

        parent = tmp_path / "parent.yaml"
        _write_yaml(parent, {"extends": "grandparent.yaml", "from_parent": True, "level": "parent"})

        child_cfg = {"extends": "parent.yaml", "from_child": True, "level": "child"}
        declaring = tmp_path / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert result["from_grandparent"] is True
        assert result["from_parent"] is True
        assert result["from_child"] is True

    def test_three_level_chain_child_wins_over_all(self, tmp_path):
        """Child's value for a shared key wins over grandparent and parent."""
        grandparent = tmp_path / "grandparent.yaml"
        _write_yaml(grandparent, {"level": "grandparent"})

        parent = tmp_path / "parent.yaml"
        _write_yaml(parent, {"extends": "grandparent.yaml", "level": "parent"})

        child_cfg = {"extends": "parent.yaml", "level": "child"}
        declaring = tmp_path / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert result["level"] == "child"

    def test_three_level_chain_parent_wins_over_grandparent(self, tmp_path):
        """Parent's value wins over grandparent when child doesn't override."""
        grandparent = tmp_path / "grandparent.yaml"
        _write_yaml(grandparent, {"shared": "grandparent", "only_gp": True})

        parent = tmp_path / "parent.yaml"
        _write_yaml(parent, {"extends": "grandparent.yaml", "shared": "parent"})

        child_cfg = {"extends": "parent.yaml", "child_only": True}
        declaring = tmp_path / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert result["shared"] == "parent"
        assert result["only_gp"] is True
        assert result["child_only"] is True

    def test_three_level_chain_extends_not_in_result(self, tmp_path):
        """The 'extends:' key at any level must not appear in the final result."""
        grandparent = tmp_path / "grandparent.yaml"
        _write_yaml(grandparent, {"base": True})

        parent = tmp_path / "parent.yaml"
        _write_yaml(parent, {"extends": "grandparent.yaml", "mid": True})

        child_cfg = {"extends": "parent.yaml", "top": True}
        declaring = tmp_path / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring)

        assert "extends" not in result


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

class TestResolveExtendsCircularDetection:
    def test_self_extends_raises_circular_error(self, tmp_path):
        """A file that extends itself raises CircularExtendsError."""
        self_file = tmp_path / "self.yaml"
        _write_yaml(self_file, {"extends": "self.yaml", "key": "value"})

        cfg = {"extends": "self.yaml", "key": "value"}

        with pytest.raises(CircularExtendsError):
            resolve_extends(cfg, declaring_file=self_file)

    def test_two_file_cycle_raises_circular_error(self, tmp_path):
        """A extends B extends A raises CircularExtendsError."""
        file_a = tmp_path / "a.yaml"
        file_b = tmp_path / "b.yaml"

        _write_yaml(file_b, {"extends": "a.yaml", "from_b": True})
        _write_yaml(file_a, {"extends": "b.yaml", "from_a": True})

        cfg = {"extends": "b.yaml", "from_a": True}

        with pytest.raises(CircularExtendsError):
            resolve_extends(cfg, declaring_file=file_a)

    def test_circular_error_message_includes_chain(self, tmp_path):
        """CircularExtendsError message includes the file names involved."""
        file_a = tmp_path / "a.yaml"
        file_b = tmp_path / "b.yaml"

        _write_yaml(file_b, {"extends": "a.yaml"})
        _write_yaml(file_a, {"extends": "b.yaml"})

        cfg = {"extends": "b.yaml"}

        with pytest.raises(CircularExtendsError) as exc_info:
            resolve_extends(cfg, declaring_file=file_a)

        msg = str(exc_info.value)
        # Message should reference the files creating the cycle
        assert "a.yaml" in msg or str(file_a) in msg


# ---------------------------------------------------------------------------
# Missing extended file
# ---------------------------------------------------------------------------

class TestResolveExtendsMissingFile:
    def test_extends_missing_file_raises(self, tmp_path):
        """Extending a non-existent file raises FileNotFoundError or ConfigPathError."""
        child_cfg = {"extends": "nonexistent.yaml", "key": "value"}
        declaring = tmp_path / "child.yaml"

        with pytest.raises((FileNotFoundError, ConfigPathError)):
            resolve_extends(child_cfg, declaring_file=declaring)


# ---------------------------------------------------------------------------
# Environment variable interpolation in extends path
# ---------------------------------------------------------------------------

class TestResolveExtendsEnvInterpolation:
    def test_extends_with_env_var_interpolated(self, tmp_path):
        """${VAR} in extends path is interpolated from env."""
        parent = tmp_path / "base.yaml"
        _write_yaml(parent, {"from_env_parent": True})

        env = {"BASE_DIR": str(tmp_path)}
        child_cfg = {"extends": "${BASE_DIR}/base.yaml", "child": True}
        declaring = tmp_path / "child.yaml"

        result = resolve_extends(child_cfg, declaring_file=declaring, env=env)

        assert result["from_env_parent"] is True
        assert result["child"] is True


# ---------------------------------------------------------------------------
# Integration with load_merged_config
# ---------------------------------------------------------------------------

class TestLoadMergedConfigWithExtends:
    def test_tier3_with_extends_inherits_values(self, tmp_path):
        """A tier 3 file with 'extends:' brings inherited values into the merged result."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        base = cwd / "base.yaml"
        _write_yaml(base, {"inherited_key": "from_base", "shared": "base"})

        _write_yaml(cwd / ".spectre.yaml", {
            "extends": "base.yaml",
            "shared": "project",
            "project_key": True,
        })

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        assert result["inherited_key"] == "from_base"
        assert result["shared"] == "project"
        assert result["project_key"] is True
        assert "extends" not in result

    def test_tier1_with_extends_inherits_values(self, tmp_path):
        """A tier 1 file with 'extends:' brings inherited values into the merged result."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        base = sr2_home / "global_base.yaml"
        _write_yaml(base, {"global_default": "from_base"})

        _write_yaml(sr2_home / "config.yaml", {
            "extends": "global_base.yaml",
            "tier1_key": True,
        })

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        assert result["global_default"] == "from_base"
        assert result["tier1_key"] is True

    def test_tier3_extends_base_overrides_tier2(self, tmp_path):
        """Tier 3's fully-resolved config (including extends chain) overrides tier 2."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        # tier 2 sets key
        _write_yaml(sr2_home / "spectre.yaml", {"key": "tier2_value"})

        # tier 3 extends a base that also sets key, but tier 3 itself doesn't
        base = cwd / "base.yaml"
        _write_yaml(base, {"key": "base_value"})
        _write_yaml(cwd / ".spectre.yaml", {"extends": "base.yaml", "other": True})

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        # tier 3's fully resolved config = {key: base_value, other: True}
        # This is merged on top of tier 2's {key: tier2_value}
        # So key = base_value (tier 3 wins), other = True
        assert result["key"] == "base_value"
        assert result["other"] is True
        assert "extends" not in result

    def test_extends_key_absent_from_final_merged_config(self, tmp_path):
        """'extends:' never leaks into the final merged config dict."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        base = cwd / "base.yaml"
        _write_yaml(base, {"base": True})
        _write_yaml(cwd / ".spectre.yaml", {"extends": "base.yaml", "project": True})

        result = load_merged_config(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        assert "extends" not in result
