"""Tests for PlanResolver layer-priority-aware token budget enforcement (FR8).

Resolves `obsidian-2v2`: replaces naive tail-truncate with a strategy that
drops layers from lowest to highest priority (L1 → L2 → L3) before
resorting to tail-truncation of the most protected layer.

The old `_truncate_to_budget` was a tail-cutoff on the combined text, which
could lop off L3 (current task — most load-bearing) while keeping L1
(project knowledge — least urgent for the current step). This test suite
validates the new `_enforce_budget` strategy.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from sr2.config.models import ResolverConfig
from sr2.pipeline.events import Event, EventPhase
from sr2_spectre.planning.resolver import (
    PlanResolver,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT = "test-project"


def make_config(
    plans_root: str | None = None,
    knowledge_root: str | None = None,
    project: str = PROJECT,
    max_tokens: int | None = None,
) -> ResolverConfig:
    cfg: dict = {"project": project}
    if plans_root is not None:
        cfg["plans_root"] = plans_root
    if knowledge_root is not None:
        cfg["knowledge_root"] = knowledge_root
    if max_tokens is not None:
        cfg["max_tokens"] = max_tokens
    return ResolverConfig(type="plan", config=cfg)


def make_turn_start_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


def write_plan_file(
    plan_dir: Path,
    slug: str = "test-plan",
    status: str = "open",
    goal: str = "Test goal",
    body: str = "Plan body.",
) -> Path:
    plan_file = plan_dir / "_plan.md"
    plan_file.write_text(
        f"""---
kind: plan
slug: {slug}
status: {status}
goal: "{goal}"
---

{body}
"""
    )
    return plan_file


def write_task_file(
    plan_dir: Path,
    order: int,
    slug: str,
    status: str = "pending",
    verify: str = "echo verify",
    title: str = "Test task",
    body: str = "Task body content.",
) -> Path:
    task_file = plan_dir / f"{order:02d}-{slug}.md"
    task_file.write_text(
        f"""---
kind: task
plan: test-plan
order: {order}
status: {status}
verify: "{verify}"
title: "{title}"
---

{body}
"""
    )
    return task_file


def write_knowledge_file(
    knowledge_dir: Path,
    filename: str,
    project: str = PROJECT,
    body: str = "Knowledge content.",
) -> Path:
    kfile = knowledge_dir / filename
    kfile.write_text(
        f"""---
kind: project-knowledge
project: {project}
---

{body}
"""
    )
    return kfile


# ---------------------------------------------------------------------------
# 1. Within budget — no modification
# ---------------------------------------------------------------------------


class TestBudgetWithinLimits:
    @pytest.mark.asyncio
    async def test_all_layers_within_budget(self, tmp_path):
        """When combined content fits the budget, all layers are returned intact."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        write_knowledge_file(knowledge_dir, "arch.md", body="K")
        write_plan_file(plan_dir, status="open", body="P")
        write_task_file(plan_dir, order=1, slug="task1", body="T")

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                max_tokens=10000,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Project Knowledge" in text
        assert "## Active Plan" in text
        assert "## Current Task" in text
        assert "K" in text
        assert "P" in text
        assert "T" in text

    @pytest.mark.asyncio
    async def test_no_budget_means_no_truncation(self, tmp_path):
        """Without max_tokens set, large content is returned as-is."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "big"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="X" * 50000)

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        assert "X" * 50000 in result.content[0].text


# ---------------------------------------------------------------------------
# 2. L1 dropped first (lowest priority)
# ---------------------------------------------------------------------------


class TestDropL1First:
    @pytest.mark.asyncio
    async def test_over_budget_drops_l1_keeps_l2_l3(self, tmp_path):
        """When L1 pushes over budget, L1 is dropped first, preserving L2 and L3."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        # Large L1 (will push over budget)
        write_knowledge_file(knowledge_dir, "arch.md", body="K" * 4000)
        # Small L2 and L3
        write_plan_file(plan_dir, status="open", body="PLAN_BODY")
        write_task_file(plan_dir, order=1, slug="task1", body="TASK_BODY")

        # L2+L3 combined is ~80 chars. L1 adds 4000+. With 21-token budget (84 chars),
        # L2+L3 alone fits, L1+L2+L3 does not.
        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                max_tokens=21,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "## Project Knowledge" not in text
        assert "K" * 100 not in text  # L1 content gone
        assert "## Active Plan" in text
        assert "PLAN_BODY" in text
        assert "## Current Task" in text
        assert "TASK_BODY" in text

    @pytest.mark.asyncio
    async def test_l1_drop_does_not_affect_l3(self, tmp_path):
        """Critical: dropping L1 never removes L3 content."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        write_knowledge_file(knowledge_dir, "huge.md", body="K" * 8000)
        write_plan_file(plan_dir, status="open", body="P")
        write_task_file(plan_dir, order=1, slug="critical-task", body="CRITICAL_INSTRUCTIONS_DO_NOT_LOSE")

        # L2+L3 alone = ~80 chars. L1+L2+L3 = ~8000+ chars.
        # Budget of 20 tokens fits L2+L3 but not L1+L2+L3.
        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                max_tokens=20,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        # L1 should be dropped
        assert "## Project Knowledge" not in text
        assert "K" * 100 not in text
        # L2 and L3 must survive
        assert "## Active Plan" in text
        assert "## Current Task" in text
        assert "CRITICAL_INSTRUCTIONS_DO_NOT_LOSE" in text


# ---------------------------------------------------------------------------
# 3. L2 dropped second (medium priority)
# ---------------------------------------------------------------------------


class TestDropL2Second:
    @pytest.mark.asyncio
    async def test_over_budget_drops_l1_then_l2_keeps_l3(self, tmp_path):
        """When L1+L2 push over budget, drop L1 first, then L2, keeping L3."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        # Large L1
        write_knowledge_file(knowledge_dir, "arch.md", body="K" * 4000)
        # Large L2
        write_plan_file(plan_dir, status="open", body="P" * 2000)
        # Small L3
        write_task_file(plan_dir, order=1, slug="task1", body="TASK_BODY")

        # L3 alone = ~41 chars content + ~16 header = ~57 chars = ~14 tokens
        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                max_tokens=14,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "## Project Knowledge" not in text
        assert "K" * 100 not in text
        assert "## Active Plan" not in text
        assert "P" * 100 not in text
        assert "## Current Task" in text
        assert "TASK_BODY" in text


# ---------------------------------------------------------------------------
# 4. L3 tail-truncation (nuclear option)
# ---------------------------------------------------------------------------


class TestL3TruncationNuclearOption:
    @pytest.mark.asyncio
    async def test_l3_truncated_when_alone_exceeds_budget(self, tmp_path):
        """When even L3 alone exceeds budget, truncate L3 tail with notice."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        # Large L1 (will be dropped)
        write_knowledge_file(knowledge_dir, "arch.md", body="K" * 4000)
        # Large L2 (will be dropped)
        write_plan_file(plan_dir, status="open", body="P" * 2000)
        # Large L3 (will be truncated, not dropped)
        write_task_file(plan_dir, order=1, slug="task1", body="T" * 5000)

        # Budget of 20 tokens = 80 chars. L3 alone is way over → truncate L3 tail.
        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                max_tokens=20,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "## Current Task" in text
        assert "truncated" in text.lower()
        assert "T" * 100 not in text

    @pytest.mark.asyncio
    async def test_l3_truncation_preserves_beginning(self, tmp_path):
        """L3 tail-truncation preserves the beginning (most load-bearing)."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        # L2 is large so it gets dropped, leaving L3 alone which is still too big
        write_plan_file(plan_dir, status="open", body="P" * 2000)
        task_body = "IMPORTANT_FIRST_PART that must survive\n" + "LESS_IMPORTANT" * 500

        write_task_file(plan_dir, order=1, slug="task1", body=task_body)

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                max_tokens=20,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "## Current Task" in text
        assert "truncated" in text.lower()
        # The beginning of content survives tail-truncation (may be partially cut)
        assert "IMPORTANT" in text
        assert "LESS_IMPORTANT" * 10 not in text  # tail content is gone


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestBudgetEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_layers_no_crash(self, tmp_path):
        """Budget enforcement on zero layers returns empty list cleanly."""
        resolver = PlanResolver(
            make_config(plans_root=str(tmp_path), max_tokens=1)
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert text == ""

    @pytest.mark.asyncio
    async def test_single_layer_over_budget_gets_truncated(self, tmp_path):
        """When only one layer exists and it exceeds budget, truncate (not drop)."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        write_knowledge_file(knowledge_dir, "arch.md", body="K" * 4000)

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                max_tokens=10,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Project Knowledge" in text
        assert "truncated" in text.lower()
        assert "K" * 100 not in text

    @pytest.mark.asyncio
    async def test_exact_budget_boundary(self, tmp_path):
        """Content that exactly fits the budget is returned unmodified."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        body = "X" * 100
        write_plan_file(plan_dir, status="open", body=body)

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), max_tokens=100)
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert body in text
        assert "truncated" not in text.lower()

    @pytest.mark.asyncio
    async def test_only_l2_l3_no_l1_budget_works(self, tmp_path):
        """With only L2+L3 (no knowledge), budget enforcement preserves both when they fit."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        write_plan_file(plan_dir, status="open", body="PLAN")
        write_task_file(plan_dir, order=1, slug="task1", body="TASK")

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), max_tokens=100)
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Active Plan" in text
        assert "## Current Task" in text
        assert "PLAN" in text
        assert "TASK" in text


# ---------------------------------------------------------------------------
# 6. Layer priority ordering validation
# ---------------------------------------------------------------------------


class TestLayerPriorityOrdering:
    def test_layer_priority_values(self):
        """Verify layer priority constants are ordered correctly."""
        from sr2_spectre.planning.resolver import (
            _LAYER1_HEADER,
            _LAYER2_HEADER,
            _LAYER3_HEADER,
            _LAYER_PRIORITY,
            _PLANNING_HEADER,
        )

        # L3 (1) < L2 (2) < L1 (3) = L3 most protected, L1 dropped first
        assert _LAYER_PRIORITY[_LAYER3_HEADER] < _LAYER_PRIORITY[_LAYER2_HEADER]
        assert _LAYER_PRIORITY[_LAYER2_HEADER] < _LAYER_PRIORITY[_LAYER1_HEADER]
        # Planning trigger has same priority as L1 (both disposable)
        assert _LAYER_PRIORITY[_PLANNING_HEADER] == _LAYER_PRIORITY[_LAYER1_HEADER]

    @pytest.mark.asyncio
    async def test_dropping_l1_before_l3(self, tmp_path):
        """Verify the algorithm drops L1 before touching L3 content."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        write_knowledge_file(knowledge_dir, "arch.md", body="K" * 4000)
        write_plan_file(plan_dir, status="open", body="L2CONTENT")
        write_task_file(plan_dir, order=1, slug="task1", body="L3CONTENT")

        # Budget = 21 tokens (84 chars) — fits L2+L3 but not L1+L2+L3
        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                max_tokens=21,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "## Project Knowledge" not in text
        assert "K" * 100 not in text
        assert "L2CONTENT" in text
        assert "L3CONTENT" in text
