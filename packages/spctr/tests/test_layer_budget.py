"""Unit tests for LayerBudget (pure logic, no resolver I/O).

These test the extracted budget allocator in isolation — no file reads,
no PlanResolver.  Each test constructs (header, content, priority) tuples
and calls ``allocate()`` directly.
"""

from __future__ import annotations

import pytest

from sr2_spectre.planning.budget import LayerBudget


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Priority constants (lower = more protected)
PRI_L1 = 3   # project knowledge — dropped first
PRI_L2 = 2   # plan overview
PRI_L3 = 1   # current task — most protected
PRI_TRIGGER = 3  # planning trigger — same as L1


def small_layer(header: str, content: str, priority: int) -> tuple[str, str, int]:
    return (header, content, priority)


def big_layer(header: str, char: str, n: int, priority: int) -> tuple[str, str, int]:
    return (header, char * n, priority)


# ---------------------------------------------------------------------------
# 1. Basic — within budget
# ---------------------------------------------------------------------------


class TestWithinBudget:
    def test_all_layers_fit(self):
        layers = [
            small_layer("## Project Knowledge", "K", PRI_L1),
            small_layer("## Active Plan", "P", PRI_L2),
            small_layer("## Current Task", "T", PRI_L3),
        ]
        result = LayerBudget(max_tokens=1000).allocate(layers)
        assert len(result) == 3
        headers = [h for h, _ in result]
        assert "## Project Knowledge" in headers
        assert "## Active Plan" in headers
        assert "## Current Task" in headers

    def test_empty_layers(self):
        result = LayerBudget(max_tokens=100).allocate([])
        assert result == []

    def test_single_layer_fits(self):
        layers = [small_layer("## Current Task", "important work", PRI_L3)]
        result = LayerBudget(max_tokens=100).allocate(layers)
        assert result == [("## Current Task", "important work")]


# ---------------------------------------------------------------------------
# 2. L1 dropped first
# ---------------------------------------------------------------------------


class TestDropL1:
    def test_drop_l1_keeps_l2_l3(self):
        """When L1 pushes over budget, it's dropped first."""
        layers = [
            big_layer("## Project Knowledge", "K", 4000, PRI_L1),
            small_layer("## Active Plan", "P", PRI_L2),
            small_layer("## Current Task", "T", PRI_L3),
        ]
        # Budget fits L2+L3 (~20 chars each = ~40 chars + separators) but not L1
        result = LayerBudget(max_tokens=20).allocate(layers)
        headers = [h for h, _ in result]
        assert "## Project Knowledge" not in headers
        assert "## Active Plan" in headers
        assert "## Current Task" in headers

    def test_l3_survives_l1_drop(self):
        """L3 must never be lost when L1 is the culprit."""
        layers = [
            big_layer("## Project Knowledge", "K", 8000, PRI_L1),
            small_layer("## Active Plan", "P", PRI_L2),
            small_layer("## Current Task", "CRITICAL_TASK", PRI_L3),
        ]
        result = LayerBudget(max_tokens=20).allocate(layers)
        assert any("CRITICAL_TASK" in c for _, c in result)
        assert all("K" * 100 not in c for _, c in result)


# ---------------------------------------------------------------------------
# 3. L2 dropped second
# ---------------------------------------------------------------------------


class TestDropL2:
    def test_drop_l1_then_l2_keeps_l3(self):
        """When L1+L2 push over budget, L3 alone survives."""
        layers = [
            big_layer("## Project Knowledge", "K", 4000, PRI_L1),
            big_layer("## Active Plan", "P", 2000, PRI_L2),
            small_layer("## Current Task", "TASK_BODY", PRI_L3),
        ]
        # Budget fits only L3 (~50 chars = ~12 tokens)
        result = LayerBudget(max_tokens=14).allocate(layers)
        headers = [h for h, _ in result]
        assert "## Project Knowledge" not in headers
        assert "## Active Plan" not in headers
        assert "## Current Task" in headers
        assert any("TASK_BODY" in c for _, c in result)


# ---------------------------------------------------------------------------
# 4. L3 truncation (nuclear option)
# ---------------------------------------------------------------------------


class TestL3Truncation:
    def test_l3_truncated_when_alone_exceeds_budget(self):
        """When even L3 alone exceeds budget, truncate tail with notice."""
        layers = [
            big_layer("## Project Knowledge", "K", 4000, PRI_L1),
            big_layer("## Active Plan", "P", 2000, PRI_L2),
            big_layer("## Current Task", "T", 5000, PRI_L3),
        ]
        result = LayerBudget(max_tokens=20).allocate(layers)
        # Only L3 should remain, truncated
        assert len(result) == 1
        header, content = result[0]
        assert header == "## Current Task"
        assert "truncated" in content.lower()
        assert "T" * 100 not in content

    def test_truncation_preserves_beginning(self):
        """L3 tail-truncation keeps the start (most load-bearing)."""
        important = "IMPORTANT_FIRST_PART "
        less_important = "LESS_IMPORTANT " * 500
        layers = [
            big_layer("## Active Plan", "P", 2000, PRI_L2),
            small_layer(
                "## Current Task",
                important + less_important,
                PRI_L3,
            ),
        ]
        result = LayerBudget(max_tokens=20).allocate(layers)
        header, content = result[0]
        assert header == "## Current Task"
        assert "truncated" in content.lower()
        assert "IMPORTANT" in content
        assert "LESS_IMPORTANT" * 10 not in content


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_budget_raises(self):
        with pytest.raises(ValueError, match="positive"):
            LayerBudget(max_tokens=0)

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="positive"):
            LayerBudget(max_tokens=-5)

    def test_exact_boundary(self):
        """Content exactly at the budget boundary passes through."""
        body = "X" * 100
        layers = [small_layer("## Active Plan", body, PRI_L2)]
        result = LayerBudget(max_tokens=100).allocate(layers)
        assert len(result) == 1
        assert body in result[0][1]
        assert "truncated" not in result[0][1].lower()

    def test_single_layer_over_budget_truncates_not_drops(self):
        """A single layer that exceeds budget is truncated, not dropped."""
        layers = [big_layer("## Project Knowledge", "K", 4000, PRI_L1)]
        result = LayerBudget(max_tokens=10).allocate(layers)
        assert len(result) == 1
        assert "truncated" in result[0][1].lower()

    def test_equal_priority_drops_earliest_first(self):
        """When two layers share priority, the one appearing later is kept."""
        layers = [
            small_layer("## Layer A", "A" * 200, PRI_L1),
            small_layer("## Layer B", "B" * 200, PRI_L1),
        ]
        # Budget fits one but not both
        result = LayerBudget(max_tokens=50).allocate(layers)
        # Layer A (index 0) is dropped first in sort order
        # (both same priority, so A at index 0 comes first in sort = dropped)
        assert len(result) == 1
        # After dropping A, only B remains
        assert result[0][0] == "## Layer B"

    def test_planning_trigger_same_priority_as_l1(self):
        """Planning trigger should be dropped at the same time as L1."""
        layers = [
            small_layer("## Planning", "nudge", PRI_TRIGGER),
            big_layer("## Project Knowledge", "K", 4000, PRI_L1),
            small_layer("## Active Plan", "P", PRI_L2),
        ]
        result = LayerBudget(max_tokens=20).allocate(layers)
        headers = [h for h, _ in result]
        # Both trigger and L1 should be gone
        assert "## Planning" not in headers
        assert "## Project Knowledge" not in headers
        assert "## Active Plan" in headers


# ---------------------------------------------------------------------------
# 6. Order preservation
# ---------------------------------------------------------------------------


class TestOrderPreservation:
    def test_original_order_preserved(self):
        """Surviving layers maintain their original order."""
        layers = [
            small_layer("## Project Knowledge", "K", PRI_L1),
            small_layer("## Active Plan", "P", PRI_L2),
            small_layer("## Current Task", "T", PRI_L3),
        ]
        result = LayerBudget(max_tokens=1000).allocate(layers)
        headers = [h for h, _ in result]
        assert headers == [
            "## Project Knowledge",
            "## Active Plan",
            "## Current Task",
        ]

    def test_order_after_l1_drop(self):
        """After L1 eviction, L2 and L3 keep their relative order."""
        layers = [
            big_layer("## Project Knowledge", "K", 4000, PRI_L1),
            small_layer("## Active Plan", "P", PRI_L2),
            small_layer("## Current Task", "T", PRI_L3),
        ]
        result = LayerBudget(max_tokens=20).allocate(layers)
        headers = [h for h, _ in result]
        assert headers == ["## Active Plan", "## Current Task"]
