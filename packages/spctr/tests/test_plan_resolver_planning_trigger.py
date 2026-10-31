"""Tests for PlanResolver planning-trigger injection (obsidian-dwe).

The planning trigger is a state-aware nudge injected by the resolver:
  - When NO open plan exists: injects a short directive telling the agent
    to load the planning guide for multi-step work.
  - When an open plan exists: suppresses the nudge (L2/L3 already carry context).
  - When planning_guide_path is NOT configured: no nudge is injected.

This replaces the always-on squadron-rules pointer with a path-aware,
scoped, state-aware trigger that only fires for agents wiring the PlanResolver.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sr2.config.models import ResolverConfig
from sr2.pipeline.events import Event, EventPhase
from sr2_spectre.planning.resolver import PlanResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT = "test-project"
GUIDE_PATH = "/home/shepard/.sr2/plans/planning-guide.md"


def make_config(
    plans_root: str | None = None,
    knowledge_root: str | None = None,
    project: str = PROJECT,
    planning_guide_path: str | None = None,
    max_tokens: int | None = None,
) -> ResolverConfig:
    cfg: dict = {"project": project}
    if plans_root is not None:
        cfg["plans_root"] = plans_root
    if knowledge_root is not None:
        cfg["knowledge_root"] = knowledge_root
    if planning_guide_path is not None:
        cfg["planning_guide_path"] = planning_guide_path
    if max_tokens is not None:
        cfg["max_tokens"] = max_tokens
    return ResolverConfig(type="plan", config=cfg)


def make_turn_start_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


def write_plan_file(plan_dir: Path, status: str = "open") -> Path:
    plan_file = plan_dir / "_plan.md"
    plan_file.write_text(
        f"""---
kind: plan
slug: test-plan
status: {status}
goal: "Test goal"
---

Test contract.
"""
    )
    return plan_file


def write_task_file(plan_dir: Path, order: int = 1, status: str = "pending") -> None:
    task_file = plan_dir / f"{order:02d}-task.md"
    task_file.write_text(
        f"""---
kind: task
plan: test-plan
order: {order}
status: {status}
verify: "echo verify"
title: "Test task"
---

Task body.
"""
    )


# ---------------------------------------------------------------------------
# 1. No planning_guide_path configured → no trigger
# ---------------------------------------------------------------------------


class TestPlanningTriggerNoPath:
    @pytest.mark.asyncio
    async def test_no_trigger_without_guide_path(self, tmp_path):
        """When planning_guide_path is not set, no nudge is injected."""
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "multi-step" not in text.lower()
        assert "planning" not in text.lower()


# ---------------------------------------------------------------------------
# 2. No open plan + guide path configured → trigger injected
# ---------------------------------------------------------------------------


class TestPlanningTriggerNoPlan:
    @pytest.mark.asyncio
    async def test_trigger_injected_when_no_open_plan(self, tmp_path):
        """When there's no open plan and a guide path is configured,
        the nudge is injected as the first section."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                planning_guide_path=GUIDE_PATH,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "## Planning" in text
        assert GUIDE_PATH in text
        assert "multi-step" in text.lower()

    @pytest.mark.asyncio
    async def test_trigger_includes_correct_guide_path(self, tmp_path):
        """The nudge contains the exact guide path from config."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        custom_guide = str(tmp_path / "custom-guide.md")
        Path(custom_guide).write_text("# Custom Guide\n")

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                planning_guide_path=custom_guide,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert custom_guide in text

    @pytest.mark.asyncio
    async def test_trigger_shown_with_l1_only(self, tmp_path):
        """Trigger is shown even when L1 knowledge exists but no plan is open."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        # Write a knowledge file
        (knowledge_dir / "arch.md").write_text(
            "---\nkind: project-knowledge\nproject: test-project\n---\n\nArch stuff.\n"
        )

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                planning_guide_path=GUIDE_PATH,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        # Both knowledge AND trigger should be present
        assert "## Project Knowledge" in text
        assert "Arch stuff." in text
        assert "## Planning" in text
        assert GUIDE_PATH in text


# ---------------------------------------------------------------------------
# 3. Open plan exists → trigger suppressed
# ---------------------------------------------------------------------------


class TestPlanningTriggerWithPlan:
    @pytest.mark.asyncio
    async def test_trigger_suppressed_when_plan_open(self, tmp_path):
        """When a plan is open, the planning trigger is NOT injected."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open")
        write_task_file(plan_dir, order=1)

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                planning_guide_path=GUIDE_PATH,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        # L2/L3 should be present
        assert "## Active Plan" in text
        assert "## Current Task" in text
        # Trigger should NOT be present
        assert "## Planning" not in text
        assert GUIDE_PATH not in text

    @pytest.mark.asyncio
    async def test_trigger_suppressed_plan_no_pending_tasks(self, tmp_path):
        """Even when plan is open but all tasks are done, trigger stays suppressed."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open")
        write_task_file(plan_dir, order=1, status="done")

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                planning_guide_path=GUIDE_PATH,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        # L2 present (plan is open)
        assert "## Active Plan" in text
        # No L3 (all tasks done)
        assert "## Current Task" not in text
        # Trigger suppressed because plan is open
        assert "## Planning" not in text


# ---------------------------------------------------------------------------
# 4. State transition: plan created → trigger disappears
# ---------------------------------------------------------------------------


class TestPlanningTriggerStateTransition:
    @pytest.mark.asyncio
    async def test_trigger_disappears_when_plan_created(self, tmp_path):
        """Creating an open plan mid-run removes the trigger on next resolve."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                planning_guide_path=GUIDE_PATH,
            )
        )

        # Turn 1: no plan → trigger present
        result = await resolver.resolve([make_turn_start_event()])
        assert "## Planning" in result.content[0].text
        assert GUIDE_PATH in result.content[0].text

        # Create plan mid-run
        plan_dir = plans_dir / "new-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open")
        write_task_file(plan_dir, order=1)

        # Turn 2: plan exists → trigger suppressed
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Planning" not in text
        assert "## Active Plan" in text

    @pytest.mark.asyncio
    async def test_trigger_reappears_when_plan_closed(self, tmp_path):
        """Closing the last open plan brings the trigger back."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        plan_file = write_plan_file(plan_dir, status="open")
        write_task_file(plan_dir, order=1)

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                planning_guide_path=GUIDE_PATH,
            )
        )

        # Turn 1: plan open → no trigger
        result = await resolver.resolve([make_turn_start_event()])
        assert "## Planning" not in result.content[0].text
        assert "## Active Plan" in result.content[0].text

        # Close the plan
        plan_text = plan_file.read_text()
        plan_text = plan_text.replace("status: open", "status: done")
        plan_file.write_text(plan_text)

        # Turn 2: no open plans → trigger returns
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Planning" in text
        assert "## Active Plan" not in text


# ---------------------------------------------------------------------------
# 5. Trigger text content
# ---------------------------------------------------------------------------


class TestPlanningTriggerContent:
    @pytest.mark.asyncio
    async def test_trigger_text_is_concise(self, tmp_path):
        """The trigger nudge is a single short paragraph, not the full guide."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                planning_guide_path=GUIDE_PATH,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        # Should be brief — a nudge, not the full guide content
        lines = [l for l in text.split("\n") if l.strip()]
        # The trigger section should be short (header + 1-2 lines of content)
        # Find the Planning section
        planning_section = False
        planning_lines = []
        for line in lines:
            if "## Planning" in line:
                planning_section = True
            elif planning_section and line.startswith("## "):
                break
            if planning_section:
                planning_lines.append(line)

        # Should be a header + a couple lines max, not a wall of text
        assert len(planning_lines) <= 5

    @pytest.mark.asyncio
    async def test_trigger_mentions_file_read(self, tmp_path):
        """The nudge tells the agent to use file_read to load the guide."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                planning_guide_path=GUIDE_PATH,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "file_read" in text.lower()


# ---------------------------------------------------------------------------
# 6. Token budget includes trigger
# ---------------------------------------------------------------------------


class TestPlanningTriggerTokenBudget:
    @pytest.mark.asyncio
    async def test_trigger_included_in_token_budget(self, tmp_path):
        """The trigger text counts toward the token budget."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                planning_guide_path=GUIDE_PATH,
                max_tokens=10000,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.token_count > 0
        assert "## Planning" in result.content[0].text
