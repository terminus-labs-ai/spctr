"""Tests for the PR review approve path — FR6 (config apply + merge + close).

Covers:
- LIVE-CONFIG parsing from PR body
- Config-affecting change detection in PR diff
- Approve path orchestration types
"""

from __future__ import annotations

import pytest

from sr2_spectre.pr_review import (
    ConfigChange,
    ConfigChangeType,
    parse_live_config_section,
    scan_diff_for_config_changes,
)


# ---------------------------------------------------------------------------
# LIVE-CONFIG parsing
# ---------------------------------------------------------------------------

class TestParseLiveConfigSection:
    """Parse the LIVE-CONFIG section from a PR body."""

    def test_none_declared(self):
        body = """Bead: spc-39
Summary: Add PR review module

## Changes
- Added pr_review.py

LIVE-CONFIG:
none
"""
        result = parse_live_config_section(body)
        assert result is None

    def test_none_with_whitespace(self):
        body = """LIVE-CONFIG:
   none   """
        result = parse_live_config_section(body)
        assert result is None

    def test_single_tool_add(self):
        body = """Bead: spc-50
Summary: Add foo tool

## Changes
- New foo tool

LIVE-CONFIG:
- add tool `foo` to ~/.sr2/config.yaml agent.tools
"""
        result = parse_live_config_section(body)
        assert result is not None
        assert len(result) == 1
        assert "foo" in result[0]

    def test_multiple_edits(self):
        body = """LIVE-CONFIG:
- add tool `foo` to ~/.sr2/config.yaml agent.tools
- register skill `bar` in ~/.sr2/agents/edi.yaml skills[]
"""
        result = parse_live_config_section(body)
        assert result is not None
        assert len(result) == 2

    def test_no_live_config_section(self):
        body = """Bead: spc-50
Summary: A pure refactor

## Changes
- Refactored module
"""
        result = parse_live_config_section(body)
        assert result is None

    def test_live_config_with_comments(self):
        body = """LIVE-CONFIG:
- add tool `foo` to ~/.sr2/config.yaml agent.tools
# - register skill `bar` (deprecated)
"""
        result = parse_live_config_section(body)
        assert result is not None
        assert len(result) == 1
        assert "#" not in result[0]

    def test_case_insensitive_section_header(self):
        body = """live-config:
- add tool `foo` to ~/.sr2/config.yaml agent.tools
"""
        result = parse_live_config_section(body)
        assert result is not None
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Diff scanner for config-affecting changes
# ---------------------------------------------------------------------------

class TestScanDiffForConfigChanges:
    """Detect config-affecting changes in a PR diff."""

    def test_no_changes_in_empty_diff(self):
        diff = ""
        changes = scan_diff_for_config_changes(diff)
        assert changes == []

    def test_detect_new_tool_module_by_path(self):
        """New .py files under tools/ directories are flagged as new tools."""
        diff = """--- /dev/null
+++ b/src/sr2_spectre/tools/builtins/my_tool.py
@@ -0,0 +1,3 @@
+class MyTool:
+    name = "my_tool"
"""
        changes = scan_diff_for_config_changes(diff)
        assert len(changes) == 1
        assert changes[0].change_type == ConfigChangeType.TOOL

    def test_detect_new_resolver_entry_point(self):
        diff = """--- a/pyproject.toml
+++ b/pyproject.toml
@@ -5,0 +6 @@
+[project.entry-points."sr2.resolvers"]
+my_resolver = "mypackage:MyResolver"
"""
        changes = scan_diff_for_config_changes(diff)
        assert len(changes) == 1
        assert changes[0].change_type == ConfigChangeType.RESOLVER

    def test_detect_new_transformer_entry_point(self):
        diff = """--- a/pyproject.toml
+++ b/pyproject.toml
@@ -5,0 +6 @@
+[project.entry-points."sr2.transformers"]
+my_transformer = "mypackage:MyTransformer"
"""
        changes = scan_diff_for_config_changes(diff)
        assert len(changes) == 1
        assert changes[0].change_type == ConfigChangeType.TRANSFORMER

    def test_detect_new_skill_module(self):
        diff = """--- /dev/null
+++ b/skills/my-skill/SKILL.md
@@ -0,0 +1,3 @@
+# My Skill
+Some skill content.
"""
        changes = scan_diff_for_config_changes(diff)
        assert len(changes) == 1
        assert changes[0].change_type == ConfigChangeType.SKILL

    def test_detect_new_builtin_tool_module(self):
        diff = """--- /dev/null
+++ b/src/sr2_spectre/tools/builtins/my_tool.py
@@ -0,0 +1,5 @@
+class MyTool:
+    pass
"""
        changes = scan_diff_for_config_changes(diff)
        assert len(changes) >= 1
        tool_changes = [c for c in changes if c.change_type == ConfigChangeType.TOOL]
        assert len(tool_changes) >= 1

    def test_no_false_positives_on_regular_code(self):
        diff = """--- a/src/sr2_spectre/agent.py
+++ b/src/sr2_spectre/agent.py
@@ -10,0 +11 @@
+def some_function():
+    return True
"""
        changes = scan_diff_for_config_changes(diff)
        assert changes == []

    def test_multiple_change_types(self):
        diff = """--- /dev/null
+++ b/src/sr2_spectre/tools/builtins/foo.py
@@ -0,0 +1,3 @@
+class Foo:
+    pass

--- a/pyproject.toml
+++ b/pyproject.toml
@@ -5,0 +6 @@
+[project.entry-points."sr2.resolvers"]
+new_resolver = "pkg:Resolver"
"""
        changes = scan_diff_for_config_changes(diff)
        types_found = {c.change_type for c in changes}
        assert ConfigChangeType.TOOL in types_found
        assert ConfigChangeType.RESOLVER in types_found

    def test_removed_lines_not_flagged(self):
        """Only additions matter for detecting new config-affecting code."""
        diff = """--- a/src/sr2_spectre/tools/builtins/old_tool.py
+++ /dev/null
@@ -1,3 +0,0 @@
-class OldTool:
-    pass
"""
        changes = scan_diff_for_config_changes(diff)
        assert changes == []

    def test_config_yaml_edits_not_flagged_as_new_tool(self):
        """Edits to config files themselves aren't 'new tools' — they're config edits."""
        diff = """--- a/config.yaml
+++ b/config.yaml
@@ -1,0 +2 @@
+tools:
+  - name: foo
"""
        changes = scan_diff_for_config_changes(diff)
        # Config file edits are not new code that needs config wiring
        assert changes == []


class TestConfigChange:
    """ConfigChange dataclass properties."""

    def test_frozen(self):
        change = ConfigChange(
            change_type=ConfigChangeType.TOOL,
            description="new tool foo",
            file_path="src/foo.py",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            change.description = "modified"  # type: ignore

    def test_change_type_enum(self):
        assert ConfigChangeType.TOOL.value == "tool"
        assert ConfigChangeType.RESOLVER.value == "resolver"
        assert ConfigChangeType.TRANSFORMER.value == "transformer"
        assert ConfigChangeType.SKILL.value == "skill"
