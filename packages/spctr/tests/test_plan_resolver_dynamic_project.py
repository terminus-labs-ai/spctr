"""Tests for dynamic project resolution in PlanResolver (obsidian-hg3).

Covers the ``project: __auto__`` sentinel that triggers runtime project
derivation from: (1) SR2_PROJECT env var, (2) cwd → .git → repo name.

Also validates that explicit project values still work unchanged.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sr2.config.models import ResolverConfig
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2_spectre.planning.models import TaskStatus
from sr2_spectre.planning.resolver import PlanResolver


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
    body: str = "Contract.",
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
# 1. Explicit project (unchanged behavior)
# ---------------------------------------------------------------------------


class TestExplicitProject:
    """Explicit project values work exactly as before (regression)."""

    def test_explicit_project_accepted(self, tmp_path):
        """A literal project name is accepted at init."""
        resolver = PlanResolver(make_config(project="myproject"))
        assert resolver._config.config["project"] == "myproject"

    @pytest.mark.asyncio
    async def test_explicit_project_filters_l1(self, tmp_path):
        """Explicit project filters L1 knowledge correctly."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        write_knowledge_file(
            knowledge_dir, "a.md", project="myproject", body="Mine."
        )
        write_knowledge_file(
            knowledge_dir, "b.md", project="otherproject", body="Not mine."
        )

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                project="myproject",
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Mine." in text
        assert "Not mine." not in text


# ---------------------------------------------------------------------------
# 2. project: __auto__ with SR2_PROJECT env var
# ---------------------------------------------------------------------------


class TestAutoProjectEnvVar:
    """project=__auto__ derives from SR2_PROJECT env var."""

    @pytest.mark.asyncio
    async def test_auto_resolves_from_env_var(self, tmp_path, monkeypatch):
        """When project=__auto__ and SR2_PROJECT is set, use the env var."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge" / "envproject"
        knowledge_dir.mkdir(parents=True)
        write_knowledge_file(
            knowledge_dir, "arch.md", project="envproject", body="Env knowledge."
        )

        monkeypatch.setenv("SR2_PROJECT", "envproject")

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                project="__auto__",
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Env knowledge." in text

    @pytest.mark.asyncio
    async def test_auto_filters_correct_project(self, tmp_path, monkeypatch):
        """Auto-derived project filters L1 correctly (excludes other projects)."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        # Knowledge for the auto-derived project
        write_knowledge_file(
            knowledge_dir, "right.md", project="auto-proj", body="Correct."
        )
        # Knowledge for a different project
        write_knowledge_file(
            knowledge_dir, "wrong.md", project="other-proj", body="Wrong."
        )

        monkeypatch.setenv("SR2_PROJECT", "auto-proj")

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                project="__auto__",
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Correct." in text
        assert "Wrong." not in text

    def test_auto_project_does_not_require_init_validation(self, tmp_path):
        """__auto__ passes init (no ValueError)."""
        resolver = PlanResolver(make_config(project="__auto__"))
        assert True  # Didn't raise


# ---------------------------------------------------------------------------
# 3. project: __auto__ with cwd derivation (fallback)
# ---------------------------------------------------------------------------


class TestAutoProjectCwdFallback:
    """When SR2_PROJECT is not set, derive from cwd → .git → repo name."""

    @pytest.mark.asyncio
    async def test_auto_derives_from_cwd_git(self, tmp_path, monkeypatch):
        """Auto derives project name from the directory containing .git."""
        # Create a fake git repo structure
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        plans_dir = repo_dir / "plans"
        plans_dir.mkdir()
        knowledge_dir = repo_dir / "knowledge"
        knowledge_dir.mkdir()
        write_knowledge_file(
            knowledge_dir, "arch.md", project="my-repo", body="Repo knowledge."
        )

        monkeypatch.delenv("SR2_PROJECT", raising=False)
        monkeypatch.chdir(repo_dir)

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                project="__auto__",
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Repo knowledge." in text

    @pytest.mark.asyncio
    async def test_auto_walks_up_for_git(self, tmp_path, monkeypatch):
        """Auto walks up from nested cwd to find .git."""
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        nested_dir = repo_dir / "src" / "deep" / "folder"
        nested_dir.mkdir(parents=True)

        plans_dir = repo_dir / "plans"
        plans_dir.mkdir()
        knowledge_dir = repo_dir / "knowledge"
        knowledge_dir.mkdir()
        write_knowledge_file(
            knowledge_dir, "arch.md", project="my-repo", body="Deep knowledge."
        )

        monkeypatch.delenv("SR2_PROJECT", raising=False)
        monkeypatch.chdir(nested_dir)

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                project="__auto__",
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Deep knowledge." in text


# ---------------------------------------------------------------------------
# 4. project: __auto__ fallback when no .git found
# ---------------------------------------------------------------------------


class TestAutoProjectNoGit:
    """When no .git is found, log a warning and use cwd name as fallback."""

    @pytest.mark.asyncio
    async def test_auto_uses_cwd_name_when_no_git(self, tmp_path, monkeypatch):
        """If no .git found walking up, use the cwd directory name."""
        work_dir = tmp_path / "some-work-dir"
        work_dir.mkdir()

        plans_dir = work_dir / "plans"
        plans_dir.mkdir()
        knowledge_dir = work_dir / "knowledge"
        knowledge_dir.mkdir()
        # Knowledge matching the cwd directory name
        write_knowledge_file(
            knowledge_dir, "arch.md", project="some-work-dir", body="Fallback knowledge."
        )

        monkeypatch.delenv("SR2_PROJECT", raising=False)
        monkeypatch.chdir(work_dir)

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                project="__auto__",
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Fallback knowledge." in text


# ---------------------------------------------------------------------------
# 5. SR2_PROJECT takes priority over cwd derivation
# ---------------------------------------------------------------------------


class TestAutoProjectPriority:
    """SR2_PROJECT env var takes priority over cwd-based derivation."""

    @pytest.mark.asyncio
    async def test_env_var_beats_cwd(self, tmp_path, monkeypatch):
        """SR2_PROJECT overrides cwd-derived project name."""
        # Create repo named "wrong-repo"
        repo_dir = tmp_path / "wrong-repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        plans_dir = repo_dir / "plans"
        plans_dir.mkdir()
        knowledge_dir = repo_dir / "knowledge"
        knowledge_dir.mkdir()

        # Knowledge for "correct-proj" (the env var value)
        write_knowledge_file(
            knowledge_dir, "right.md", project="correct-proj", body="Env wins."
        )
        # Knowledge for "wrong-repo" (the cwd-derived value)
        write_knowledge_file(
            knowledge_dir, "wrong.md", project="wrong-repo", body="Cwd loses."
        )

        monkeypatch.setenv("SR2_PROJECT", "correct-proj")
        monkeypatch.chdir(repo_dir)

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                project="__auto__",
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Env wins." in text
        assert "Cwd loses." not in text


# ---------------------------------------------------------------------------
# 6. Dynamic re-evaluation per turn
# ---------------------------------------------------------------------------


class TestAutoProjectPerTurn:
    """Auto project is re-evaluated each turn (not cached at init)."""

    @pytest.mark.asyncio
    async def test_project_can_change_between_turns(self, tmp_path, monkeypatch):
        """Changing SR2_PROJECT between turns changes which L1 is injected."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        write_knowledge_file(
            knowledge_dir, "proj_a.md", project="proj-a", body="Project A."
        )
        write_knowledge_file(
            knowledge_dir, "proj_b.md", project="proj-b", body="Project B."
        )

        monkeypatch.setenv("SR2_PROJECT", "proj-a")

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                project="__auto__",
            )
        )

        # Turn 1: proj-a
        result = await resolver.resolve([make_turn_start_event()])
        assert "Project A." in result.content[0].text
        assert "Project B." not in result.content[0].text

        # Change env var mid-run
        monkeypatch.setenv("SR2_PROJECT", "proj-b")

        # Turn 2: proj-b (dynamic re-read)
        result = await resolver.resolve([make_turn_start_event()])
        assert "Project B." in result.content[0].text
        assert "Project A." not in result.content[0].text


# ---------------------------------------------------------------------------
# 7. Default knowledge_root with __auto__
# ---------------------------------------------------------------------------


class TestAutoProjectDefaultKnowledgeRoot:
    """When knowledge_root is not explicitly set, __auto__ derives it."""

    @pytest.mark.asyncio
    async def test_default_knowledge_root_with_auto_project(
        self, tmp_path, monkeypatch
    ):
        """Default knowledge_root resolves to $SR2_HOME/knowledge/<derived-project>."""
        monkeypatch.setenv("SR2_HOME", str(tmp_path))
        monkeypatch.setenv("SR2_PROJECT", "derived-proj")

        # Create the default knowledge dir structure
        knowledge_dir = tmp_path / "knowledge" / "derived-proj"
        knowledge_dir.mkdir(parents=True)
        write_knowledge_file(
            knowledge_dir, "arch.md", project="derived-proj", body="Default root knowledge."
        )

        # Create plans dir at default location
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir(parents=True)

        resolver = PlanResolver(make_config(project="__auto__"))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Default root knowledge." in text
