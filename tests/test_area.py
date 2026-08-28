"""Launch-directory area derivation (obsidian-7qdu).

sr2_spectre/area.py derives an area name from the launch directory so the
REPL (and PlanResolver's fallback chain) can resolve {area}-templated
resources. Order: SR2_AREA, nearest CLAUDE.md, nearest .git, cwd basename
(one WARNING). CLAUDE.md precedes .git because a vault area such as
/data/obsidian/projects/<area> is a folder inside the vault repo — the git
root would name the vault, not the area.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sr2_spectre.area import derive_area


# ---------------------------------------------------------------------------
# 1. SR2_AREA env var
# ---------------------------------------------------------------------------


class TestSr2AreaEnvVar:
    def test_env_var_wins(self, tmp_path, monkeypatch) -> None:
        area = tmp_path / "area"
        area.mkdir()
        (area / "CLAUDE.md").write_text("# area")
        (area / ".git").mkdir()
        monkeypatch.setenv("SR2_AREA", "override")
        assert derive_area(area) == "override"

    def test_empty_env_var_is_authoritative_no_area(
        self, tmp_path, monkeypatch
    ) -> None:
        """An empty SR2_AREA means 'no area' — no fallback to CLAUDE.md/.git."""
        area = tmp_path / "area"
        area.mkdir()
        (area / "CLAUDE.md").write_text("# area")
        monkeypatch.setenv("SR2_AREA", "")
        assert derive_area(area) == ""

    def test_env_var_beats_cwd_walk(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SR2_AREA", "explicit")
        assert derive_area(tmp_path) == "explicit"


# ---------------------------------------------------------------------------
# 2. Nearest CLAUDE.md
# ---------------------------------------------------------------------------


class TestClaudeMd:
    def test_cwd_with_claude_md(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("SR2_AREA", raising=False)
        area = tmp_path / "fractured-roots"
        area.mkdir()
        (area / "CLAUDE.md").write_text("# area")
        assert derive_area(area) == "fractured-roots"

    def test_walks_up_to_nearest_claude_md(self, tmp_path, monkeypatch) -> None:
        """The nearest ancestor with CLAUDE.md wins over outer markers."""
        monkeypatch.delenv("SR2_AREA", raising=False)
        outer = tmp_path / "outer"
        outer.mkdir()
        (outer / "CLAUDE.md").write_text("# outer")

        area = outer / "inner-area"
        area.mkdir()
        (area / "CLAUDE.md").write_text("# area")
        nested = area / "deep" / "work"
        nested.mkdir(parents=True)
        assert derive_area(nested) == "inner-area"

    def test_claude_md_precedes_git(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        """Vault case: an area folder inside the vault repo.

        /data/obsidian/.git would name the vault ('obsidian'); the area's
        own CLAUDE.md must win and name the area.
        """
        monkeypatch.delenv("SR2_AREA", raising=False)
        vault = tmp_path / "obsidian"
        vault.mkdir()
        (vault / ".git").mkdir()

        area = vault / "projects" / "fractured-roots"
        area.mkdir(parents=True)
        (area / "CLAUDE.md").write_text("# area")

        with caplog.at_level("WARNING"):
            assert derive_area(area) == "fractured-roots"
        assert not caplog.records


# ---------------------------------------------------------------------------
# 3. Nearest .git
# ---------------------------------------------------------------------------


class TestGit:
    def test_cwd_with_git(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("SR2_AREA", raising=False)
        repo = tmp_path / "sr2-spectre"
        repo.mkdir()
        (repo / ".git").mkdir()
        assert derive_area(repo) == "sr2-spectre"

    def test_walks_up_to_git_file(self, tmp_path, monkeypatch) -> None:
        """A .git *file* (worktree/submodule pointer) counts too."""
        monkeypatch.delenv("SR2_AREA", raising=False)
        repo = tmp_path / "worktree-repo"
        repo.mkdir()
        (repo / ".git").write_text("gitdir: /elsewhere/.git")
        nested = repo / "src" / "deep"
        nested.mkdir(parents=True)
        assert derive_area(nested) == "worktree-repo"

    def test_git_without_claude_md(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("SR2_AREA", raising=False)
        repo = tmp_path / "bare-repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        assert derive_area(repo) == "bare-repo"


# ---------------------------------------------------------------------------
# 4. Cwd basename fallback
# ---------------------------------------------------------------------------


class TestCwdBasenameFallback:
    def test_fallback_uses_start_basename(self, tmp_path, monkeypatch, caplog) -> None:
        monkeypatch.delenv("SR2_AREA", raising=False)
        work = tmp_path / "scratch-work"
        work.mkdir()
        with caplog.at_level("WARNING"):
            assert derive_area(work) == "scratch-work"
            assert "Area derivation" in caplog.text
            assert caplog.text.count("Area derivation") == 1

    def test_defaults_to_cwd(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("SR2_AREA", raising=False)
        work = tmp_path / "the-area"
        work.mkdir()
        (work / "CLAUDE.md").write_text("# area")
        monkeypatch.chdir(work)
        assert derive_area() == "the-area"


# ---------------------------------------------------------------------------
# 5. End-to-end: PlanResolver fallback now consults CLAUDE.md and SR2_AREA
# ---------------------------------------------------------------------------


class TestPlanResolverDelegation:
    @pytest.mark.asyncio
    async def test_claude_md_names_the_project(
        self, tmp_path, monkeypatch
    ) -> None:
        """PlanResolver's cwd fallback resolves through area.py: a CLAUDE.md
        inside a repo names the project, not the repo root."""
        from sr2.config.models import ResolverConfig
        from sr2.pipeline.events import Event, EventPhase
        from sr2_spectre.planning.resolver import PlanResolver

        home = tmp_path / "home"
        (home / ".sr2" / "plans").mkdir(parents=True)

        vault = tmp_path / "obsidian"
        vault.mkdir()
        (vault / ".git").mkdir()
        area = vault / "projects" / "my-area"
        area.mkdir(parents=True)
        (area / "CLAUDE.md").write_text("# area")

        knowledge = home / ".sr2" / "knowledge" / "my-area"
        knowledge.mkdir(parents=True)
        (knowledge / "k.md").write_text(
            "---\nkind: project-knowledge\nproject: my-area\n---\n\nArea body.\n"
        )

        monkeypatch.delenv("SR2_PROJECT", raising=False)
        monkeypatch.delenv("SR2_AREA", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: home)
        monkeypatch.chdir(area)

        resolver = PlanResolver(
            ResolverConfig(
                type="plan",
                config={
                    "plans_root": str(home / ".sr2" / "plans"),
                    "project": "__auto__",
                },
            )
        )
        result = await resolver.resolve(
            [Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")]
        )
        text = result.content[0].text
        assert "Area body." in text

    @pytest.mark.asyncio
    async def test_sr2_area_env_is_honored(self, tmp_path, monkeypatch) -> None:
        from sr2.config.models import ResolverConfig
        from sr2.pipeline.events import Event, EventPhase
        from sr2_spectre.planning.resolver import PlanResolver

        home = tmp_path / "home"
        (home / ".sr2" / "plans").mkdir(parents=True)
        knowledge = home / ".sr2" / "knowledge" / "env-area"
        knowledge.mkdir(parents=True)
        (knowledge / "k.md").write_text(
            "---\nkind: project-knowledge\nproject: env-area\n---\n\nEnv body.\n"
        )

        work = tmp_path / "unrelated"
        work.mkdir()
        monkeypatch.delenv("SR2_PROJECT", raising=False)
        monkeypatch.setenv("SR2_AREA", "env-area")
        monkeypatch.setattr("pathlib.Path.home", lambda: home)
        monkeypatch.chdir(work)

        resolver = PlanResolver(
            ResolverConfig(
                type="plan",
                config={
                    "plans_root": str(home / ".sr2" / "plans"),
                    "project": "__auto__",
                },
            )
        )
        result = await resolver.resolve(
            [Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")]
        )
        assert "Env body." in result.content[0].text
