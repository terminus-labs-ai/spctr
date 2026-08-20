"""PlanResolver reads the run context's area in ``__auto__`` mode.

Spec: ``specs/channel-area-injection.md`` (bead spc-48).

  AC 8  — ``__auto__`` uses the run context's area ahead of ``SR2_PROJECT``
          and ahead of the cwd ``.git`` walk.
  AC 9  — an explicitly empty area injects no knowledge and does not fall
          through to env/cwd.
  AC 10 — an absent ``area`` key behaves exactly as before (env -> .git ->
          basename), including when ``run_context_provider`` is ``None``.
  AC 11 — a non-``__auto__`` configured ``project:`` ignores the run context
          entirely.

``PlanResolver`` keeps its ``project:`` vocabulary; this is the one
area -> project translation point (FR 13).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sr2.config.models import ResolverConfig
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2_spectre.planning.resolver import PlanResolver

ALPHA = "ALPHA-KNOWLEDGE"
ENV = "ENV-KNOWLEDGE"
GITREPO = "GITREPO-KNOWLEDGE"
PINNED = "PINNED-KNOWLEDGE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(plans_root: Path, knowledge_root: Path, project: str) -> ResolverConfig:
    return ResolverConfig(
        type="plan",
        config={
            "project": project,
            "plans_root": str(plans_root),
            "knowledge_root": str(knowledge_root),
        },
    )


def turn_start() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


def write_knowledge(knowledge_dir: Path, filename: str, project: str, body: str) -> None:
    (knowledge_dir / filename).write_text(
        f"---\nkind: project-knowledge\nproject: {project}\n---\n\n{body}\n"
    )


def deps_with_area(area: str) -> Dependencies:
    return Dependencies(
        run_context_provider=lambda: {
            "mode": "interactive",
            "source": "",
            "area": area,
        }
    )


def deps_without_area_key() -> Dependencies:
    return Dependencies(
        run_context_provider=lambda: {"mode": "interactive", "source": ""}
    )


def three_candidates(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Stage three competing project names and return (plans_root, knowledge_root).

    ``alpha`` is only reachable through the run context, ``envproj`` only
    through ``SR2_PROJECT``, ``gitrepo`` only through the cwd ``.git`` walk.
    Whichever body shows up in the injection names the winner.
    """
    repo = tmp_path / "gitrepo"
    repo.mkdir()
    (repo / ".git").mkdir()
    plans = tmp_path / "plans"
    plans.mkdir()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()

    write_knowledge(knowledge, "alpha.md", "alpha", ALPHA)
    write_knowledge(knowledge, "env.md", "envproj", ENV)
    write_knowledge(knowledge, "git.md", "gitrepo", GITREPO)

    monkeypatch.setenv("SR2_PROJECT", "envproj")
    monkeypatch.chdir(repo)
    return plans, knowledge


async def injected(resolver: PlanResolver) -> str:
    return (await resolver.resolve([turn_start()])).content[0].text


# ---------------------------------------------------------------------------
# AC 8 — area outranks SR2_PROJECT and the .git walk
# ---------------------------------------------------------------------------

class TestAreaPriority:
    async def test_area_beats_env_var_and_git_walk(self, tmp_path, monkeypatch) -> None:
        plans, knowledge = three_candidates(tmp_path, monkeypatch)
        resolver = PlanResolver.build(
            make_config(plans, knowledge, "__auto__"), deps_with_area("alpha")
        )

        text = await injected(resolver)

        assert ALPHA in text
        assert ENV not in text
        assert GITREPO not in text

    async def test_area_is_re_read_every_turn(self, tmp_path, monkeypatch) -> None:
        """The provider is consulted at resolve time, not frozen at build."""
        plans, knowledge = three_candidates(tmp_path, monkeypatch)
        write_knowledge(knowledge, "beta.md", "beta", "BETA-KNOWLEDGE")
        current = {"area": "alpha"}
        deps = Dependencies(
            run_context_provider=lambda: {
                "mode": "interactive",
                "source": "",
                "area": current["area"],
            }
        )
        resolver = PlanResolver.build(make_config(plans, knowledge, "__auto__"), deps)

        assert ALPHA in await injected(resolver)

        current["area"] = "beta"
        second = await injected(resolver)
        assert "BETA-KNOWLEDGE" in second
        assert ALPHA not in second


# ---------------------------------------------------------------------------
# AC 9 — explicitly empty area stops resolution
# ---------------------------------------------------------------------------

class TestEmptyArea:
    async def test_empty_area_injects_no_knowledge(self, tmp_path, monkeypatch) -> None:
        plans, knowledge = three_candidates(tmp_path, monkeypatch)
        resolver = PlanResolver.build(
            make_config(plans, knowledge, "__auto__"), deps_with_area("")
        )

        result = await resolver.resolve([turn_start()])

        assert result.resolver_name == "plan"  # resolved, did not raise
        assert ALPHA not in result.content[0].text
        assert "## Project Knowledge" not in result.content[0].text

    async def test_empty_area_does_not_fall_through(self, tmp_path, monkeypatch) -> None:
        """An empty area is a decision, not an absence — env and cwd stay out."""
        plans, knowledge = three_candidates(tmp_path, monkeypatch)
        resolver = PlanResolver.build(
            make_config(plans, knowledge, "__auto__"), deps_with_area("")
        )

        text = await injected(resolver)

        assert ENV not in text
        assert GITREPO not in text


# ---------------------------------------------------------------------------
# AC 10 — absent area key is the pre-existing behaviour
# ---------------------------------------------------------------------------

NO_AREA_DEPS = {
    "key-absent": deps_without_area_key,
    "no-provider": Dependencies,
    "provider-returns-none": lambda: Dependencies(run_context_provider=lambda: None),
}


class TestAbsentAreaKey:
    @pytest.mark.parametrize("deps_factory", NO_AREA_DEPS.values(), ids=NO_AREA_DEPS)
    async def test_falls_through_to_env(
        self, deps_factory, tmp_path, monkeypatch
    ) -> None:
        plans, knowledge = three_candidates(tmp_path, monkeypatch)
        resolver = PlanResolver.build(
            make_config(plans, knowledge, "__auto__"), deps_factory()
        )

        text = await injected(resolver)

        assert ENV in text
        assert ALPHA not in text

    @pytest.mark.parametrize("deps_factory", NO_AREA_DEPS.values(), ids=NO_AREA_DEPS)
    async def test_falls_through_to_git_walk_without_env(
        self, deps_factory, tmp_path, monkeypatch
    ) -> None:
        plans, knowledge = three_candidates(tmp_path, monkeypatch)
        monkeypatch.delenv("SR2_PROJECT", raising=False)
        resolver = PlanResolver.build(
            make_config(plans, knowledge, "__auto__"), deps_factory()
        )

        text = await injected(resolver)

        assert GITREPO in text
        assert ALPHA not in text


# ---------------------------------------------------------------------------
# AC 11 — an explicit project: ignores the run context
# ---------------------------------------------------------------------------

class TestExplicitProjectIgnoresArea:
    async def test_named_area_does_not_override_explicit_project(
        self, tmp_path, monkeypatch
    ) -> None:
        plans, knowledge = three_candidates(tmp_path, monkeypatch)
        write_knowledge(knowledge, "pinned.md", "pinned", PINNED)
        resolver = PlanResolver.build(
            make_config(plans, knowledge, "pinned"), deps_with_area("alpha")
        )

        text = await injected(resolver)

        assert PINNED in text
        assert ALPHA not in text

    async def test_empty_area_does_not_suppress_explicit_project(
        self, tmp_path, monkeypatch
    ) -> None:
        plans, knowledge = three_candidates(tmp_path, monkeypatch)
        write_knowledge(knowledge, "pinned.md", "pinned", PINNED)
        resolver = PlanResolver.build(
            make_config(plans, knowledge, "pinned"), deps_with_area("")
        )

        assert PINNED in await injected(resolver)
