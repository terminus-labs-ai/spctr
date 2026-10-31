"""Tests for PlanResolver _findings.md injection and active-frame provider.

Covers acceptance criteria from obsidian-aal.7 (FR8/FR9):
  - _findings.md body injected when present and non-empty
  - Layer omitted when file absent or empty
  - Findings persist across simulated burns (resolver re-reads disk each turn)
  - current_frame_id returns plan:<plan-slug>/<task-slug> for lowest-order pending
  - current_frame_id returns None when no open plans or no pending tasks
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sr2.config.models import ResolverConfig
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2_spectre.planning.resolver import PlanResolver

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
    body: str = "Understanding & constraints contract.",
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
plan: {slug}
order: {order}
status: {status}
verify: "{verify}"
title: "{title}"
---

{body}
"""
    )
    return task_file


def write_findings_file(
    plan_dir: Path,
    content: str = "Finding: --plugin also referenced in tests/conftest.py",
) -> Path:
    findings_file = plan_dir / "_findings.md"
    findings_file.write_text(content)
    return findings_file


# ---------------------------------------------------------------------------
# 1. _findings.md injection — present and non-empty
# ---------------------------------------------------------------------------


class TestFindingsInjection:
    @pytest.mark.asyncio
    async def test_findings_injected_when_present(self, tmp_path):
        """_findings.md body is injected alongside L2 when present."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Plan contract.")
        write_task_file(plan_dir, order=1, slug="task1", body="Task details.")
        write_findings_file(plan_dir, content="Cross-step discovery here.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "## Active Plan" in text
        assert "Plan contract." in text
        assert "Cross-step discovery here." in text
        # Findings appear within the Active Plan layer, not as a separate header
        assert "## Current Task" in text

    @pytest.mark.asyncio
    async def test_findings_omitted_when_file_absent(self, tmp_path):
        """When _findings.md doesn't exist, no findings content is injected."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Plan contract.")
        write_task_file(plan_dir, order=1, slug="task1", body="Task details.")
        # No _findings.md created

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "## Active Plan" in text
        # No findings header or content
        assert "_findings" not in text.lower()

    @pytest.mark.asyncio
    async def test_findings_omitted_when_empty(self, tmp_path):
        """When _findings.md exists but is empty/whitespace, layer is omitted."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Plan contract.")
        write_task_file(plan_dir, order=1, slug="task1", body="Task details.")
        write_findings_file(plan_dir, content="   \n\n  ")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "## Active Plan" in text
        # Only whitespace = treated as empty
        assert "## Active Findings" not in text

    @pytest.mark.asyncio
    async def test_findings_survive_simulated_burn(self, tmp_path):
        """Findings persist across a simulated burn — resolver re-reads from disk each turn."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Plan contract.")
        write_task_file(plan_dir, order=1, slug="task1", body="Task one.")
        write_task_file(plan_dir, order=2, slug="task2", body="Task two.")
        write_findings_file(plan_dir, content="Important cross-step finding.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))

        # Turn 1: task1 pending, findings present
        result = await resolver.resolve([make_turn_start_event()])
        assert "Important cross-step finding." in result.content[0].text
        assert "Task one." in result.content[0].text

        # Simulate task1 completion: flip status to done
        task1_file = plan_dir / "01-task1.md"
        task1_text = task1_file.read_text()
        task1_text = task1_text.replace("status: pending", "status: done")
        task1_file.write_text(task1_text)

        # Simulate appending another finding
        findings_file = plan_dir / "_findings.md"
        findings_file.write_text(
            "Important cross-step finding.\nAdditional finding from task1."
        )

        # Turn 2: task2 pending, findings still present (burned task1 gone from prompt,
        # but findings re-injected from disk)
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Important cross-step finding." in text
        assert "Additional finding from task1." in text
        assert "Task two." in text
        assert "Task one." not in text  # done task not injected

    @pytest.mark.asyncio
    async def test_findings_no_frontmatter_required(self, tmp_path):
        """_findings.md has no required frontmatter — injected as raw body."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Plan.")
        write_task_file(plan_dir, order=1, slug="task1", body="Task.")
        # Raw text, no frontmatter
        (plan_dir / "_findings.md").write_text("Raw finding line one.\nRaw finding line two.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert "Raw finding line one." in text
        assert "Raw finding line two." in text

    @pytest.mark.asyncio
    async def test_findings_excluded_from_task_glob(self, tmp_path):
        """_findings.md is not picked up as a task by the L3 resolver."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Plan.")
        write_task_file(plan_dir, order=1, slug="task1", body="The real task.")
        write_findings_file(plan_dir, content="Finding content.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        # Findings should appear in Active Plan layer, NOT in Current Task
        assert "Finding content." in text
        assert "The real task." in text
        # Findings shouldn't duplicate into the Current Task section
        l3_start = text.index("## Current Task")
        l3_section = text[l3_start:]
        assert "Finding content." not in l3_section


# ---------------------------------------------------------------------------
# 2. Active-frame provider (current_frame_id)
# ---------------------------------------------------------------------------


class TestActiveFrameProvider:
    def test_current_frame_id_with_pending_task(self, tmp_path):
        """Returns plan:<slug>/<task-slug> for the lowest-order pending task."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "my-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, slug="my-plan", status="open", body="Plan.")
        write_task_file(plan_dir, order=1, slug="first-task", body="Task 1.")
        write_task_file(plan_dir, order=2, slug="second-task", body="Task 2.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        frame = resolver.current_frame_id()

        assert frame == "plan:my-plan/first-task"

    def test_current_frame_id_follows_pending_after_done(self, tmp_path):
        """Returns the frame of the next pending task when task1 is done."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "my-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, slug="my-plan", status="open", body="Plan.")
        task1 = write_task_file(plan_dir, order=1, slug="first-task", body="Task 1.")
        write_task_file(plan_dir, order=2, slug="second-task", body="Task 2.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))

        # Initially: first-task is pending
        assert resolver.current_frame_id() == "plan:my-plan/first-task"

        # Flip task1 to done
        task1_text = task1.read_text()
        task1_text = task1_text.replace("status: pending", "status: done")
        task1.write_text(task1_text)

        # Now: second-task should be the active frame (re-reads from disk)
        assert resolver.current_frame_id() == "plan:my-plan/second-task"

    def test_current_frame_id_none_when_no_open_plans(self, tmp_path):
        """Returns None when no open plans exist."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        assert resolver.current_frame_id() is None

    def test_current_frame_id_none_when_no_pending_tasks(self, tmp_path):
        """Returns None when plan is open but all tasks are done."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "my-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, slug="my-plan", status="open", body="Plan.")
        write_task_file(plan_dir, order=1, slug="task1", status="done", body="Done.")
        write_task_file(plan_dir, order=2, slug="task2", status="done", body="Done.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        assert resolver.current_frame_id() is None

    def test_current_frame_id_none_when_plans_dir_missing(self, tmp_path):
        """Returns None when plans_root doesn't exist."""
        plans_dir = tmp_path / "nonexistent"

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        assert resolver.current_frame_id() is None

    def test_frame_provider_callable_for_dependencies(self, tmp_path):
        """current_frame_id works as a Callable[[], str | None] for Dependencies."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test"
        plan_dir.mkdir()
        write_plan_file(plan_dir, slug="test", status="open", body="Plan.")
        write_task_file(plan_dir, order=1, slug="setup", body="Setup task.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))

        # Verify it can be used as an active_frame_provider on Dependencies
        deps = Dependencies(active_frame_provider=resolver.current_frame_id)
        assert deps.active_frame_provider is not None
        assert deps.active_frame_provider() == "plan:test/setup"


# ---------------------------------------------------------------------------
# 3. Layer positioning of findings
# ---------------------------------------------------------------------------


class TestFindingsLayerPositioning:
    @pytest.mark.asyncio
    async def test_findings_within_active_plan_layer(self, tmp_path):
        """Findings appear within the Active Plan layer, after _plan.md body."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Plan contract body.")
        write_task_file(plan_dir, order=1, slug="task1", body="Task body.")
        write_findings_file(plan_dir, content="Finding A.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        # Findings appear after plan body but before Current Task
        plan_pos = text.index("Plan contract body.")
        finding_pos = text.index("Finding A.")
        task_pos = text.index("## Current Task")

        assert plan_pos < finding_pos < task_pos

    @pytest.mark.asyncio
    async def test_findings_not_present_without_plan(self, tmp_path):
        """Findings are never injected when there's no open plan."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        assert text == ""
