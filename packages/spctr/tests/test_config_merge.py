"""Tests for config merge utility (FR4: deep + named-merge + ordering)."""
from __future__ import annotations

import pytest
from sr2_spectre.config_merge import merge_configs


# ---------------------------------------------------------------------------
# Scalar merging
# ---------------------------------------------------------------------------

class TestScalarMerge:
    def test_child_scalar_wins(self):
        parent = {"key": "parent_value"}
        child = {"key": "child_value"}
        result = merge_configs(parent, child)
        assert result["key"] == "child_value"

    def test_parent_scalar_kept_when_child_missing(self):
        parent = {"key": "parent_value", "other": "kept"}
        child = {"key": "child_value"}
        result = merge_configs(parent, child)
        assert result["other"] == "kept"

    def test_child_scalar_overrides_with_none(self):
        parent = {"key": "value"}
        child = {"key": None}
        result = merge_configs(parent, child)
        assert result["key"] is None

    def test_child_adds_new_scalar_key(self):
        parent = {"a": 1}
        child = {"b": 2}
        result = merge_configs(parent, child)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_child_integer_wins_over_parent(self):
        parent = {"count": 5}
        child = {"count": 10}
        result = merge_configs(parent, child)
        assert result["count"] == 10


# ---------------------------------------------------------------------------
# Dict (map) merging — deep merge
# ---------------------------------------------------------------------------

class TestDictMerge:
    def test_nested_dict_deep_merged(self):
        parent = {"config": {"a": 1, "b": 2}}
        child = {"config": {"b": 99, "c": 3}}
        result = merge_configs(parent, child)
        assert result["config"]["a"] == 1   # parent key kept
        assert result["config"]["b"] == 99  # child wins
        assert result["config"]["c"] == 3   # child-only key added

    def test_deeply_nested_dict_merged(self):
        parent = {"level1": {"level2": {"x": 1, "y": 2}}}
        child = {"level1": {"level2": {"y": 99}}}
        result = merge_configs(parent, child)
        assert result["level1"]["level2"]["x"] == 1
        assert result["level1"]["level2"]["y"] == 99

    def test_child_dict_overwrites_parent_scalar(self):
        parent = {"key": "scalar"}
        child = {"key": {"nested": "value"}}
        result = merge_configs(parent, child)
        assert result["key"] == {"nested": "value"}

    def test_parent_dict_overwritten_by_child_scalar(self):
        parent = {"key": {"nested": "value"}}
        child = {"key": "scalar"}
        result = merge_configs(parent, child)
        assert result["key"] == "scalar"


# ---------------------------------------------------------------------------
# Named list merging (list of dicts with 'name:' field)
# ---------------------------------------------------------------------------

class TestNamedListMerge:
    def test_matched_entry_deep_merged_in_parent_position(self):
        parent = {"layers": [
            {"name": "system", "weight": 1, "config": {"text": "hello"}},
            {"name": "tools", "weight": 2},
        ]}
        child = {"layers": [
            {"name": "system", "config": {"text": "world"}},
        ]}
        result = merge_configs(parent, child)
        layers = result["layers"]
        assert len(layers) == 2
        assert layers[0]["name"] == "system"
        assert layers[0]["config"]["text"] == "world"  # child wins
        assert layers[0]["weight"] == 1               # parent key kept
        assert layers[1]["name"] == "tools"            # unchanged

    def test_parent_only_entries_kept(self):
        parent = {"resolvers": [
            {"name": "static", "value": "A"},
            {"name": "session", "value": "B"},
        ]}
        child = {"resolvers": [
            {"name": "static", "value": "Z"},
        ]}
        result = merge_configs(parent, child)
        resolvers = result["resolvers"]
        assert len(resolvers) == 2
        session = next(r for r in resolvers if r["name"] == "session")
        assert session["value"] == "B"

    def test_child_only_entries_appended(self):
        parent = {"layers": [
            {"name": "system"},
        ]}
        child = {"layers": [
            {"name": "new_layer"},
        ]}
        result = merge_configs(parent, child)
        names = [l["name"] for l in result["layers"]]
        assert names == ["system", "new_layer"]

    def test_child_only_entries_appended_in_child_order(self):
        parent = {"layers": [
            {"name": "existing"},
        ]}
        child = {"layers": [
            {"name": "third"},
            {"name": "first"},
            {"name": "second"},
        ]}
        result = merge_configs(parent, child)
        names = [l["name"] for l in result["layers"]]
        assert names == ["existing", "third", "first", "second"]

    def test_parent_positions_preserved_matched_entry(self):
        parent = {"layers": [
            {"name": "first"},
            {"name": "second"},
            {"name": "third"},
        ]}
        child = {"layers": [
            {"name": "third", "extra": "x"},
            {"name": "first", "extra": "y"},
        ]}
        result = merge_configs(parent, child)
        names = [l["name"] for l in result["layers"]]
        # Parent order preserved; both matched in-place
        assert names == ["first", "second", "third"]
        assert result["layers"][0]["extra"] == "y"
        assert result["layers"][2]["extra"] == "x"

    def test_mixed_match_and_new_entries(self):
        parent = {"layers": [
            {"name": "a"},
            {"name": "b"},
        ]}
        child = {"layers": [
            {"name": "b", "updated": True},
            {"name": "c"},
            {"name": "d"},
        ]}
        result = merge_configs(parent, child)
        names = [l["name"] for l in result["layers"]]
        # Parent positions: a(0), b(1); new entries c,d appended in child order
        assert names == ["a", "b", "c", "d"]
        b_entry = result["layers"][1]
        assert b_entry["updated"] is True

    def test_new_key_inside_matched_entry_is_added(self):
        parent = {"layers": [
            {"name": "system", "config": {"text": "hello"}},
        ]}
        child = {"layers": [
            {"name": "system", "config": {"text": "world"}, "priority": 99},
        ]}
        result = merge_configs(parent, child)
        layer = result["layers"][0]
        assert layer["config"]["text"] == "world"
        assert layer["priority"] == 99


# ---------------------------------------------------------------------------
# Plain list merging (no 'name:' field) — child replaces
# ---------------------------------------------------------------------------

class TestPlainListMerge:
    def test_plain_list_child_replaces_entirely(self):
        parent = {"tags": ["a", "b", "c"]}
        child = {"tags": ["x", "y"]}
        result = merge_configs(parent, child)
        assert result["tags"] == ["x", "y"]

    def test_plain_list_of_scalars_replaced(self):
        parent = {"ids": [1, 2, 3]}
        child = {"ids": [4, 5]}
        result = merge_configs(parent, child)
        assert result["ids"] == [4, 5]

    def test_list_of_dicts_without_name_field_replaced(self):
        parent = {"items": [{"type": "a"}, {"type": "b"}]}
        child = {"items": [{"type": "c"}]}
        result = merge_configs(parent, child)
        assert result["items"] == [{"type": "c"}]

    def test_mixed_list_some_have_name_not_all(self):
        """If not all elements have 'name:', treat as plain list — child replaces."""
        parent = {"resolvers": [
            {"name": "static", "value": "A"},
            {"type": "dynamic"},   # no name field
        ]}
        child = {"resolvers": [
            {"name": "static", "value": "Z"},
        ]}
        result = merge_configs(parent, child)
        # Not all elements have name, so child replaces
        assert result["resolvers"] == [{"name": "static", "value": "Z"}]


# ---------------------------------------------------------------------------
# Empty parent / child edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_child_parent_unchanged(self):
        parent = {"a": 1, "b": {"x": 2}, "layers": [{"name": "sys"}]}
        child = {}
        result = merge_configs(parent, child)
        assert result == {"a": 1, "b": {"x": 2}, "layers": [{"name": "sys"}]}

    def test_empty_parent_child_becomes_result(self):
        parent = {}
        child = {"a": 1, "layers": [{"name": "sys"}]}
        result = merge_configs(parent, child)
        assert result == {"a": 1, "layers": [{"name": "sys"}]}

    def test_both_empty(self):
        result = merge_configs({}, {})
        assert result == {}

    def test_empty_named_list_in_parent(self):
        parent = {"layers": []}
        child = {"layers": [{"name": "new"}]}
        result = merge_configs(parent, child)
        assert result["layers"] == [{"name": "new"}]

    def test_empty_child_list_with_named_parent(self):
        """Empty child list: named-ness determined from parent; no child matches/appends → parent kept."""
        parent = {"layers": [{"name": "existing"}]}
        child = {"layers": []}
        result = merge_configs(parent, child)
        # Parent is a named list; child has no entries to match or append.
        # Result: all parent entries kept, no new entries added.
        assert result["layers"] == [{"name": "existing"}]

    def test_inputs_not_mutated(self):
        parent = {"key": "original", "layers": [{"name": "a", "v": 1}]}
        child = {"key": "override", "layers": [{"name": "a", "v": 2}]}
        parent_copy = {"key": "original", "layers": [{"name": "a", "v": 1}]}
        child_copy = {"key": "override", "layers": [{"name": "a", "v": 2}]}
        merge_configs(parent, child)
        assert parent == parent_copy
        assert child == child_copy


# ---------------------------------------------------------------------------
# Recursive named-merge (named-merge applies inside matched entries)
# ---------------------------------------------------------------------------

class TestRecursiveNamedMerge:
    def test_named_merge_inside_matched_named_entry(self):
        """A layer's resolvers list also gets named-merged."""
        parent = {"layers": [
            {
                "name": "system",
                "resolvers": [
                    {"name": "static", "config": {"text": "hello"}},
                    {"name": "session"},
                ],
            }
        ]}
        child = {"layers": [
            {
                "name": "system",
                "resolvers": [
                    {"name": "static", "config": {"text": "world"}},
                    {"name": "new_resolver"},
                ],
            }
        ]}
        result = merge_configs(parent, child)
        resolvers = result["layers"][0]["resolvers"]
        names = [r["name"] for r in resolvers]
        # static matched in-place, session kept, new_resolver appended
        assert names == ["static", "session", "new_resolver"]
        static = next(r for r in resolvers if r["name"] == "static")
        assert static["config"]["text"] == "world"

    def test_deep_nested_named_merge(self):
        """Three-level nesting: pipeline > layers > resolvers."""
        parent = {
            "pipeline": {
                "layers": [
                    {
                        "name": "tools",
                        "resolvers": [{"name": "tool_list", "max": 5}],
                    }
                ]
            }
        }
        child = {
            "pipeline": {
                "layers": [
                    {
                        "name": "tools",
                        "resolvers": [{"name": "tool_list", "max": 10}],
                    }
                ]
            }
        }
        result = merge_configs(parent, child)
        resolver = result["pipeline"]["layers"][0]["resolvers"][0]
        assert resolver["name"] == "tool_list"
        assert resolver["max"] == 10
