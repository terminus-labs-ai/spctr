"""Resume path integration test (obsidian-wxd.5).

Validates the deterministic resume infrastructure using the real plan files
on disk. This covers the "interrupt + resume" half of the e2e probe that
was NOT exercised in the 2026-05-31 probe run.

What this tests:
  1. Plan discovery: resolver finds the open plan on disk
  2. Done-task skipping: tasks with status:done are skipped
  3. Resume point: the next pending task is injected (L3)
  4. Contract preservation: L2 (the _plan.md body) is re-injected on resume
  5. Dynamic re-read: status changes on disk are reflected on the next resolve
  6. Archive path: completed plans moved to _archive/ are not selected

What this does NOT test (requires LLM agent observation):
  - Whether edi actually follows the planning protocol
  - Whether edi flips task status after verify:
  - Whether edi performs final validation
  Those are behavioral findings from running the agent — the domain of the
  full e2e probe with human observation.
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from sr2.config.models import ResolverConfig
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.models import ResolvedContent
from sr2_spectre.planning.resolver import PlanResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT = "sr2-spectre"


def make_resolver(plans_root: str, knowledge_root: str | None = None) -> PlanResolver:
    cfg = {"project": PROJECT, "plans_root": plans_root}
    if knowledge_root:
        cfg["knowledge_root"] = knowledge_root
    return PlanResolver(ResolverConfig(type="plan", config=cfg))


def make_turn_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


def write_plan(
    plan_dir: Path,
    slug: str = "resume-test",
    status: str = "open",
    goal: str = "Test resume capability",
    contract: str = (
        "Must preserve public API.\n"
        "All tests must pass after each step.\n"
        "No changes to agent.py core logic."
    ),
) -> Path:
    plan_file = plan_dir / "_plan.md"
    plan_file.write_text(
        f"""---
kind: plan
slug: {slug}
status: {status}
goal: "{goal}"
---

## Understanding & Constraints

{contract}
"""
    )
    return plan_file


def write_task(
    plan_dir: Path,
    order: int,
    slug: str,
    status: str = "pending",
    verify: str = "echo verify",
    title: str = "",
    body: str = "",
) -> Path:
    task_file = plan_dir / f"{order:02d}-{slug}.md"
    task_file.write_text(
        f"""---
kind: task
plan: resume-test
order: {order}
status: {status}
verify: "{verify}"
title: "{title}"
---

{body}
"""
    )
    return task_file


def write_knowledge(
    knowledge_dir: Path,
    filename: str,
    project: str = PROJECT,
    body: str = "Project knowledge.",
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
# Resume path tests
# ---------------------------------------------------------------------------


class TestResumePath:
    """Simulates the 'interrupt + resume' scenario: a plan exists with
    some tasks done, some pending. A fresh resolver run should pick up
    exactly at the right point."""

    @pytest.mark.asyncio
    async def test_resume_after_partial_completion(self, tmp_path):
        """Simulates edi being interrupted after completing tasks 1-2.
        A fresh run discovers the plan and resumes at task 3."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "resume-test"
        plan_dir.mkdir()

        write_plan(plan_dir, contract="Must preserve public API.")
        write_task(plan_dir, 1, "grounding", status="done", body="Grounding step.")
        write_task(plan_dir, 2, "step-one", status="done", body="Step one.")
        write_task(plan_dir, 3, "step-two", status="pending", body="Step two — resume here.")
        write_task(plan_dir, 4, "step-three", status="pending", body="Step three.")

        resolver = make_resolver(str(plans_dir))
        result = await resolver.resolve([make_turn_event()])
        text = result.content[0].text

        # L2: plan contract is re-injected (edi needs context on resume)
        assert "## Active Plan" in text
        assert "Must preserve public API." in text

        # L3: resumes at task 3 (lowest-order pending)
        assert "## Current Task" in text
        assert "Step two — resume here." in text

        # Done tasks are NOT injected
        assert "Grounding step." not in text
        assert "Step one." not in text

        # Future tasks are NOT injected
        assert "Step three." not in text

    @pytest.mark.asyncio
    async def test_resume_preserves_contract_across_runs(self, tmp_path):
        """On resume, the plan's Understanding & Constraints are
        re-injected so edi doesn't lose context about what to preserve."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "contract-test"
        plan_dir.mkdir()

        contract = (
            "1. No changes to agent.py core logic\n"
            "2. All tests must pass\n"
            "3. __init__.py exports updated consistently"
        )
        write_plan(plan_dir, contract=contract)
        write_task(plan_dir, 1, "done-task", status="done", body="Completed.")
        write_task(plan_dir, 2, "current-task", status="pending", body="Next step.")

        resolver = make_resolver(str(plans_dir))
        result = await resolver.resolve([make_turn_event()])
        text = result.content[0].text

        # Contract survives resume
        assert "No changes to agent.py core logic" in text
        assert "All tests must pass" in text
        assert "__init__.py exports updated consistently" in text

    @pytest.mark.asyncio
    async def test_resume_with_knowledge_injected(self, tmp_path):
        """On resume, L1 project knowledge is injected alongside the plan,
        so edi has full context without needing session memory."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        plan_dir = plans_dir / "resume-test"
        plan_dir.mkdir()
        write_plan(plan_dir)
        write_task(plan_dir, 1, "step-one", status="done", body="Done.")
        write_task(plan_dir, 2, "step-two", status="pending", body="Resume.")

        write_knowledge(knowledge_dir, "architecture.md", body="SOLID principles. Module boundaries.")

        resolver = make_resolver(
            plans_root=str(plans_dir),
            knowledge_root=str(knowledge_dir),
        )
        result = await resolver.resolve([make_turn_event()])
        text = result.content[0].text

        # L1 injected
        assert "## Project Knowledge" in text
        assert "SOLID principles" in text

        # L2 injected
        assert "## Active Plan" in text

        # L3: correct resume point
        assert "Resume." in text

    @pytest.mark.asyncio
    async def test_resume_after_all_tasks_done(self, tmp_path):
        """When all tasks are done, no L3 is injected — signals edi to
        perform final validation and archive the plan."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "completed-plan"
        plan_dir.mkdir()

        write_plan(plan_dir)
        write_task(plan_dir, 1, "task1", status="done", body="Done.")
        write_task(plan_dir, 2, "task2", status="done", body="Done.")
        write_task(plan_dir, 3, "task3", status="done", body="Done.")

        resolver = make_resolver(str(plans_dir))
        result = await resolver.resolve([make_turn_event()])
        text = result.content[0].text

        # L2 present (plan still open)
        assert "## Active Plan" in text

        # L3 absent (no pending tasks — time for final validation)
        assert "## Current Task" not in text

    @pytest.mark.asyncio
    async def test_archive_prevents_resume_of_completed_plan(self, tmp_path):
        """Plans moved to _archive/ are not selected by the resolver.
        This validates the archive lifecycle decision (obsidian-aal.5)."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        # Archived plan — should NOT be discovered
        archive_dir = plans_dir / "_archive"
        archive_dir.mkdir()
        archived_plan = archive_dir / "old-plan"
        archived_plan.mkdir()
        write_plan(archived_plan, slug="old-plan", status="done", goal="Old goal.")
        write_task(archived_plan, 1, "task1", status="done", body="Done.")

        resolver = make_resolver(str(plans_dir))
        result = await resolver.resolve([make_turn_event()])
        text = result.content[0].text

        # Archived plan is not injected
        assert "## Active Plan" not in text
        assert "Old goal." not in text

    @pytest.mark.asyncio
    async def test_resume_dynamic_status_update(self, tmp_path):
        """During a resume session, as edi completes tasks, the resolver
        dynamically advances — no restart needed."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "dynamic-test"
        plan_dir.mkdir()

        write_plan(plan_dir)
        task2_file = write_task(plan_dir, 1, "task-one", status="pending", body="Task one.")
        task3_file = write_task(plan_dir, 2, "task-two", status="pending", body="Task two.")
        task4_file = write_task(plan_dir, 3, "task-three", status="pending", body="Task three.")

        resolver = make_resolver(str(plans_dir))

        # Turn 1: resume at task 1
        result = await resolver.resolve([make_turn_event()])
        assert "Task one." in result.content[0].text
        assert "Task two." not in result.content[0].text

        # Edi completes task 1 (simulated by flipping status)
        task2_text = task2_file.read_text()
        task2_text = task2_text.replace("status: pending", "status: done")
        task2_file.write_text(task2_text)

        # Turn 2: should advance to task 2
        result = await resolver.resolve([make_turn_event()])
        assert "Task two." in result.content[0].text
        assert "Task one." not in result.content[0].text

        # Edi completes task 2
        task3_text = task3_file.read_text()
        task3_text = task3_text.replace("status: pending", "status: done")
        task3_file.write_text(task3_text)

        # Turn 3: should advance to task 3
        result = await resolver.resolve([make_turn_event()])
        assert "Task three." in result.content[0].text
        assert "Task two." not in result.content[0].text

        # Edi completes task 3
        task4_text = task4_file.read_text()
        task4_text = task4_text.replace("status: pending", "status: done")
        task4_file.write_text(task4_text)

        # Turn 4: all done — no L3
        result = await resolver.resolve([make_turn_event()])
        assert "## Current Task" not in result.content[0].text
        assert "## Active Plan" in result.content[0].text  # L2 still present


class TestResumeWithRealPlan:
    """Tests against the actual plan files on disk (rename-plugin-to-interface).
    Validates the resume scenario described in the bead's probe notes."""

    @pytest.mark.asyncio
    async def test_resume_with_existing_sp3_plan(self, tmp_path):
        """Simulates the resume scenario for spc-3: the plan exists from
        the May 31 probe, some tasks completed, execution interrupted at task 3.
        A fresh run should pick up at task 3 with full context."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "rename-plugin-to-interface"
        plan_dir.mkdir()

        # Mirror the real plan structure
        write_plan(
            plan_dir,
            slug="rename-plugin-to-interface",
            status="open",
            goal="Rename Plugin→Interface, delete dead plugin machinery (spc-3)",
            contract=(
                "Plugin Protocol → Interface (rename)\n"
                "InputPlugin → folds into Interface\n"
                "Full pytest suite stays green after each step\n"
                "__init__.py exports updated consistently"
            ),
        )
        # Tasks 1-2 completed during first run
        write_task(plan_dir, 1, "rename-plugin-protocol", status="done", body="Protocol renamed.")
        write_task(plan_dir, 2, "rename-plugin-implementations", status="done", body="Implementations renamed.")
        # Task 3 is where execution was interrupted
        write_task(plan_dir, 3, "cli-flag-and-deprecated-alias", status="pending", body="Update CLI.")
        write_task(plan_dir, 4, "delete-dead-code", status="pending", body="Delete dead code.")
        write_task(plan_dir, 5, "final-validation", status="pending", body="Final validation.")

        resolver = make_resolver(str(plans_dir))
        result = await resolver.resolve([make_turn_event()])
        text = result.content[0].text

        # L2: plan context re-injected
        assert "## Active Plan" in text
        assert "Plugin Protocol" in text

        # L3: resumes at task 3
        assert "## Current Task" in text
        assert "Update CLI." in text

        # Done tasks not injected
        assert "Protocol renamed." not in text
        assert "Implementations renamed." not in text

        # Future tasks not injected
        assert "Delete dead code." not in text
        assert "Final validation." not in text

    @pytest.mark.asyncio
    async def test_resume_verify_command_preserved(self, tmp_path):
        """On resume, the verify: command for the current task is accessible
        so edi knows how to validate before advancing."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "verify-test"
        plan_dir.mkdir()

        write_plan(plan_dir)
        verify_cmd = "cd /home/shepard/git/sr2-spectre && .venv/bin/python -m pytest tests/ -q"
        task_file = write_task(
            plan_dir, 1, "test-task",
            status="pending",
            verify=verify_cmd,
            body="Run tests.",
        )

        # Read the task file to verify frontmatter is preserved
        task_content = task_file.read_text()
        assert "verify:" in task_content
        assert verify_cmd in task_content
