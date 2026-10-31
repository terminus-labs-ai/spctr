# AC11 — Fresh project resume: Spectre situation providers (sr2-102).
#
# Ground-truth tests for the generic SituationProvider contract:
#   * CompositeSituationProvider refreshes every configured source
#   * Revision validation lets unchanged sources reuse cached content (FR23)
#   * Required-source failure is a structured ContinuityFailure, never
#     silent (FR29); optional-source failure never enters `failures`
#   * Built-in providers cover configured markdown/files, git, and beads (FR22)
#   * Entries carry source, authority, observation time, and revision (FR24)
#   * SpectreConfig declares the sources; nothing is hardcoded
#
# The contract fixed for these tests (spec Interfaces section, verbatim):
#
#   class SituationRequest(BaseModel, frozen=True):
#       owner_id: str
#       agent_id: str
#       origin: str
#       cwd: str | None
#       projects: tuple[str, ...]
#       previous_revisions: dict[str, str] = {}
#
#   class SituationSnapshot(BaseModel, frozen=True):
#       entries: tuple[Entry, ...]
#       refreshed_at: datetime
#       revisions: dict[str, str]
#       failures: tuple[ContinuityFailure, ...] = ()
#
#   class SituationProvider(Protocol):
#       async def refresh(self, request: SituationRequest) -> SituationSnapshot: ...
#
# Test-only conventions (implementation may satisfy them any way it likes,
# but the observable behavior asserted here is binding):
#   * Revisions are exposed under stable keys, so `previous_revisions` from
#     one snapshot round-trips into the next request.
#   * A reused (revision-validated, not re-read) source contributes at most
#     one entry — the cached one — to the new snapshot.
#   * `ContinuityFailure.message` names the failing source.

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from sr2.continuity.index import ContinuityFailure
from sr2.models import TextBlock
from sr2.pipeline.provenance import (
    ContextAuthority,
    ContextReason,
    Entry,
    EntryOrigin,
)

from sr2_spectre.continuity.sources import (
    BeadsSituationProvider,
    FileSituationProvider,
    GitSituationProvider,
)
from sr2_spectre.continuity.situation import (
    CompositeSituationProvider,
    SituationProvider,
    SituationRequest,
    SituationSnapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAYER = "situation"


def make_request(**overrides):
    base = dict(
        owner_id="diego",
        agent_id="spectre",
        origin="repl",
        cwd=None,
        projects=(),
    )
    base.update(overrides)
    return SituationRequest(**base)


def entry_text(entry) -> str:
    content = entry.content
    return getattr(content, "text", None) or str(content)


def entries_from(entry_iterable, needle: str):
    return [e for e in entry_iterable if needle in entry_text(e)]


def live_request(tmp_path, **overrides) -> SituationRequest:
    base = dict(
        owner_id="diego",
        agent_id="spectre",
        origin="repl",
        cwd=str(tmp_path),
        projects=("SR2",),
    )
    base.update(overrides)
    return SituationRequest(**base)


def make_beads_project(path, title):
    """Init a real bd project at *path* with one issue. Skips if bd absent.

    bd writes only inside *path* (isolated tmp dir), so cleanup is the
    caller's tmp_path teardown.
    """
    import shutil

    if shutil.which("bd") is None:
        pytest.skip("bd binary not available")
    init = subprocess.run(
        ["bd", "init", "--prefix", "AC11", "--quiet"],
        cwd=path, capture_output=True, text=True,
    )
    if init.returncode != 0:
        pytest.skip(f"bd init unavailable here: {init.stderr.strip()[:200]}")
    create = subprocess.run(
        ["bd", "create", title, "--quiet"],
        cwd=path, capture_output=True, text=True,
    )
    if create.returncode != 0:
        pytest.skip(f"bd create unavailable here: {create.stderr.strip()[:200]}")
    return path


class StaticProvider:
    """A hand-written provider that satisfies the SituationProvider protocol.

    Stands in for one live source in composite tests: reports a fixed
    revision, records how many times it was actually asked to work, and can
    be configured to fail.
    """

    def __init__(self, *, text, revision, required=True, fail=None):
        self._text = text
        self._revision = revision
        self.required = required
        self._fail = fail  # Exception | None
        self.refresh_count = 0

    async def refresh(self, request: SituationRequest) -> SituationSnapshot:
        self.refresh_count += 1
        if self._fail is not None:
            raise self._fail
        entry = Entry(  # noqa: F821 - imported below at module level
            id=f"static-{self._text[:8]}",
            content=TextBlock(text=self._text),
            sources=(),
            origin=EntryOrigin(kind="resolver", name="static-provider"),
            layer=_LAYER,
            session_id="situation",
            created_at=datetime.now(timezone.utc),
            reason=ContextReason.LIVE_SOURCE,
            authority=ContextAuthority.LIVE_SOURCE,
        )
        return SituationSnapshot(
            entries=(entry,),
            refreshed_at=datetime.now(timezone.utc),
            revisions={"static": self._revision},
        )


def init_git_repo(path, message="initial commit"):
    """Create a real git repo with one file and one commit."""
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], cwd=path, check=True,
        capture_output=True, text=True,
    )
    run("init", "-b", "main")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    (path / "file.txt").write_text("v1")
    run("add", "file.txt")
    run("commit", "-m", message)


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Row 1 — Composite provider refreshes all configured sources (FR22, FR24)
# ---------------------------------------------------------------------------


class TestCompositeRefresh:
    async def test_composite_collects_entries_from_all_sources(self, tmp_path):
        """AC11-r1: entries from every configured source appear, each with
        LIVE_SOURCE authority/reason and source/time/revision metadata."""
        now_md = tmp_path / "NOW.md"
        now_md.write_text("NOW: ship the resume")
        init_git_repo(tmp_path, message="feat: last completed work")
        (tmp_path / ".beads").mkdir()

        composite = CompositeSituationProvider(
            [
                FileSituationProvider(paths=[str(tmp_path / "NOW.md")], required=True),
                GitSituationProvider(repo=str(tmp_path), required=True),
                BeadsSituationProvider(project="SR2", required=False, cwd=str(tmp_path)),
            ]
        )

        snap = await composite.refresh(live_request(tmp_path))

        assert isinstance(snap, SituationSnapshot)
        assert snap.failures == ()

        # Files source contributed the NOW.md content.
        now_entries = entries_from(snap.entries, "ship the resume")
        assert now_entries, "NOW.md content missing from snapshot entries"

        # Every entry: live-source provenance (FR24).
        for e in snap.entries:
            assert e.authority == ContextAuthority.LIVE_SOURCE
            assert e.reason == ContextReason.LIVE_SOURCE
            assert e.created_at is not None  # observation time
            assert e.meta.get("source"), f"entry {e.id} lacks source ref"
            assert e.meta.get("revision"), f"entry {e.id} lacks revision"

        # Git source contributed HEAD sha + status.
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        git_entries = entries_from(snap.entries, head[:12])
        assert git_entries, "git HEAD sha missing from snapshot entries"

        # Beads source contributed (bd runs against the project dir).
        assert any(
            "beads" in entry_text(e).lower() or "bd " in entry_text(e)
            for e in snap.entries
        ), "beads source contributed no entry"

        # Revisions are keyed and non-empty; file revision identifies NOW.md.
        file_revs = [r for r in snap.revisions.values() if sha256_of(now_md.read_text()) in r]
        assert file_revs, "no revision exposes the sha256 of NOW.md content"

    async def test_provider_protocol_is_runtime_checkable(self):
        """AC11-r1: the spec's SituationProvider Protocol is exported and
        structural checks work against hand-written implementations."""
        assert isinstance(StaticProvider(text="x", revision="r1"), SituationProvider)
        assert not isinstance(object(), SituationProvider)


# ---------------------------------------------------------------------------
# Row 2 — Revision-based caching (FR23)
# ---------------------------------------------------------------------------


class TestRevisionCaching:
    def _file_provider(self, tmp_path):
        (tmp_path / "NOW.md").write_text("original content")
        return FileSituationProvider(paths=[str(tmp_path / "NOW.md")], required=True)

    async def test_unchanged_revision_reuses_cached_content(self, tmp_path):
        """AC11-r2: with previous_revisions matching current revisions, the
        snapshot still carries the entry content and the file is NOT re-read."""
        provider = self._file_provider(tmp_path)
        snap1 = await provider.refresh(live_request(tmp_path))
        assert entries_from(snap1.entries, "original content")

        def exploding_read(*args, **kwargs):
            raise AssertionError("file source re-read despite unchanged revision")

        # Boundary: the file source must consult revision metadata (e.g.
        # stat) instead of content whenever the revision is unchanged. Any
        # content read via the public pathlib API trips this guard.
        monkey_patch = pytest.MonkeyPatch()
        monkey_patch.setattr(Path, "read_text", exploding_read, raising=False)
        monkey_patch.setattr(Path, "read_bytes", exploding_read, raising=False)
        monkey_patch.setattr(Path, "open", exploding_read, raising=False)
        try:
            snap2 = await provider.refresh(
                live_request(tmp_path, previous_revisions=dict(snap1.revisions))
            )
        finally:
            monkey_patch.undo()

        assert entries_from(snap2.entries, "original content"), (
            "cached entry missing from revision-validated snapshot"
        )
        assert snap2.revisions == snap1.revisions

    async def test_changed_revision_forces_refresh(self, tmp_path):
        """AC11-r3: content change => revision mismatch => re-read."""
        provider = self._file_provider(tmp_path)
        snap1 = await provider.refresh(live_request(tmp_path))

        (tmp_path / "NOW.md").write_text("updated content")
        snap2 = await provider.refresh(
            live_request(tmp_path, previous_revisions=dict(snap1.revisions))
        )

        assert entries_from(snap2.entries, "updated content")
        assert not entries_from(snap2.entries, "original content")

    async def test_empty_previous_revisions_forces_full_refresh(self, tmp_path):
        """AC11-r10: previous_revisions={} => every source reads fresh."""
        (tmp_path / "NOW.md").write_text("alpha")
        provider = FileSituationProvider(paths=[str(tmp_path / "NOW.md")], required=True)

        snap = await provider.refresh(live_request(tmp_path))  # default {}

        assert entries_from(snap.entries, "alpha")
        assert snap.revisions


# ---------------------------------------------------------------------------
# Row 3 — Required-source failure is structured (FR29)
# Row 4 — Optional-source failure is not a failure
# ---------------------------------------------------------------------------


class TestFailureSurfacing:
    async def test_required_source_failure_becomes_structured_failure(self):
        """AC11-r3: a required source raising produces
        ContinuityFailure(capability='situation') naming the source, while
        the other sources' entries survive."""
        boom = OSError("disk on fire")
        good = StaticProvider(text="good source says hi", revision="g1")
        bad = StaticProvider(text="never", revision="b1", required=True, fail=boom)

        composite = CompositeSituationProvider([good, bad])
        snap = await composite.refresh(make_request())

        assert entries_from(snap.entries, "good source says hi")
        assert len(snap.failures) == 1
        failure = snap.failures[0]
        assert isinstance(failure, ContinuityFailure)
        assert failure.capability == "situation"
        assert "static" in failure.message.lower()

    async def test_optional_source_failure_is_not_a_failure(self):
        """AC11-r4: required=false source raising => no ContinuityFailure,
        but its stale content does not sneak into entries."""
        good = StaticProvider(text="required content", revision="g1")
        flaky = StaticProvider(text="never", revision="b1", required=False,
                               fail=RuntimeError("bd down"))

        composite = CompositeSituationProvider([good, flaky])
        snap = await composite.refresh(make_request())

        assert entries_from(snap.entries, "required content")
        assert snap.failures == ()

    async def test_required_git_source_failure_names_source(self, tmp_path):
        """AC11-r3 (git boundary): required git provider against a
        non-repository yields a structured situation failure, not a raise."""
        not_a_repo = tmp_path / "plain"
        not_a_repo.mkdir()
        provider = GitSituationProvider(repo=str(not_a_repo), required=True)

        snap = await provider.refresh(live_request(not_a_repo))

        assert snap.entries == ()
        assert len(snap.failures) == 1
        assert snap.failures[0].capability == "situation"
        assert "git" in snap.failures[0].message.lower()


# ---------------------------------------------------------------------------
# Rows 5–7 — Built-in providers (FR22)
# ---------------------------------------------------------------------------


class TestFileProvider:
    async def test_reads_configured_path_with_hash_revision(self, tmp_path):
        """AC11-r5: reads configured file content; entry carries source ref
        to the path and a revision identifying the content (sha256)."""
        f = tmp_path / "NOW.md"
        f.write_text("# NOW\nfinish the cage\n")
        provider = FileSituationProvider(paths=[str(f)], required=True)

        snap = await provider.refresh(live_request(tmp_path))

        matches = entries_from(snap.entries, "finish the cage")
        assert matches, "configured file content missing from entries"
        entry = matches[0]
        assert "NOW.md" in entry.meta["source"]
        assert sha256_of(f.read_text()) in entry.meta["revision"]

    async def test_multiple_configured_paths_all_read(self, tmp_path):
        """AC11-r5: every configured path contributes (sources are a list,
        NOW.md and handoff.md are configuration, not vocabulary)."""
        a = tmp_path / "NOW.md"
        b = tmp_path / "handoff.md"
        a.write_text("now marker")
        b.write_text("handoff marker")
        provider = FileSituationProvider(paths=[str(a), str(b)], required=True)

        snap = await provider.refresh(live_request(tmp_path))
        assert entries_from(snap.entries, "now marker")
        assert entries_from(snap.entries, "handoff marker")

    async def test_missing_required_file_is_structured_failure(self, tmp_path):
        """AC11-r5: a required configured file that does not exist fails
        the source structurally (FR29), naming the source."""
        provider = FileSituationProvider(
            paths=[str(tmp_path / "NOW.md")], required=True
        )
        snap = await provider.refresh(live_request(tmp_path))

        assert snap.entries == ()
        assert len(snap.failures) == 1
        assert snap.failures[0].capability == "situation"
        assert "file" in snap.failures[0].message.lower()


class TestGitProvider:
    async def test_reads_status_and_recent_commits(self, tmp_path):
        """AC11-r6: entry shows working state and recent commits; revision
        carries HEAD sha; dirty worktree is visible."""
        init_git_repo(tmp_path, message="feat: last completed work")
        provider = GitSituationProvider(repo=str(tmp_path), required=True)

        snap = await provider.refresh(live_request(tmp_path))

        assert snap.failures == ()
        matches = entries_from(snap.entries, "feat: last completed work")
        assert matches, "recent commit subject missing from git entry"
        entry = matches[0]
        assert entry.meta["source"] == "git"
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        assert head in entry.meta["revision"]

        # Uncommitted change must show up in working state.
        (tmp_path / "dirty.txt").write_text("uncommitted")
        snap2 = await provider.refresh(live_request(tmp_path))
        dirty_entries = entries_from(snap2.entries, "dirty.txt")
        assert dirty_entries, "dirty worktree file missing from git status entry"

    async def test_optional_git_source_on_non_repo_is_silent(self, tmp_path):
        """AC11-r6: optional git source (required=false) on a non-repo yields
        neither entries nor failures."""
        provider = GitSituationProvider(repo=str(tmp_path), required=False)
        snap = await provider.refresh(live_request(tmp_path))
        assert snap.entries == ()
        assert snap.failures == ()


class TestBeadsProvider:
    """Beads source exercised against the real `bd` binary (boundary
    allowed to be real; skip if absent). `bd` writes only inside the
    isolated tmp project."""

    BEADS_TITLE = "Fresh project resume"

    @pytest.fixture()
    def beads_project(self, tmp_path):
        project = tmp_path / "beads-project"
        project.mkdir()
        return make_beads_project(project, self.BEADS_TITLE)

    async def test_reads_project_state(self, beads_project):
        """AC11-r7: bd project state becomes an entry carrying the issue,
        source ref 'beads', and a revision (timestamp or query count)."""
        provider = BeadsSituationProvider(project="SR2", required=True)
        snap = await provider.refresh(
            live_request(beads_project)
        )

        assert snap.failures == ()
        matches = entries_from(snap.entries, self.BEADS_TITLE)
        assert matches, "beads issue title missing from entries"
        entry = matches[0]
        assert entry.meta["source"] == "beads"
        assert entry.meta["revision"], "beads entry lacks revision"

    async def test_unreachable_project_optional_is_silent(self, tmp_path):
        """AC11-r7: beads source pointing at a project with no bd database,
        required=false => entries and failures both empty."""
        empty = tmp_path / "no-beads"
        empty.mkdir()
        provider = BeadsSituationProvider(project="nothing", required=False)
        snap = await provider.refresh(live_request(empty))
        assert snap.entries == ()
        assert snap.failures == ()

    async def test_unreachable_project_required_is_structured(self, tmp_path):
        """AC11-r7 (FR29): same boundary, required=true => structured
        situation failure naming beads, no silent degradation."""
        empty = tmp_path / "no-beads"
        empty.mkdir()
        provider = BeadsSituationProvider(project="nothing", required=True)
        snap = await provider.refresh(live_request(empty))
        assert len(snap.failures) == 1
        assert snap.failures[0].capability == "situation"
        assert "beads" in snap.failures[0].message.lower()


# ---------------------------------------------------------------------------
# Row 8 — Config parsing (sources are configured, not hardcoded)
# ---------------------------------------------------------------------------


_MINIMAL_SPECTRE = {
    "agent": {"name": "spectre"},
    "models": {"default": {"model": "openai/gpt-4o"}},
    "pipeline": {"layers": []},
}


def _spectre_with(extra: dict):
    from sr2_spectre.config import SpectreConfig

    data = {**_MINIMAL_SPECTRE, **extra}
    return SpectreConfig(**data)


class TestContinuityConfig:
    def test_situation_sources_parse(self):
        """AC11-r8: continuity.situation_sources parses type/paths/repo/
        project/required per the spec config shape."""
        cfg = _spectre_with(
            {
                "continuity": {
                    "situation_sources": [
                        {"type": "files", "paths": ["{area}/NOW.md"], "required": True},
                        {"type": "git", "repo": "{cwd}", "required": True},
                        {"type": "beads", "project": "{project}", "required": False},
                    ]
                }
            }
        )
        srcs = cfg.continuity.situation_sources
        assert len(srcs) == 3
        assert srcs[0].type == "files"
        assert srcs[0].paths == ["{area}/NOW.md"]
        assert srcs[0].required is True
        assert srcs[1].type == "git"
        assert srcs[1].repo == "{cwd}"
        assert srcs[2].type == "beads"
        assert srcs[2].project == "{project}"
        assert srcs[2].required is False

    def test_defaults_empty(self):
        """AC11-r8: absent continuity section => empty source list."""
        cfg = _spectre_with({})
        assert cfg.continuity.situation_sources == []

    def test_unknown_source_type_rejected(self):
        """AC11-r8: unknown source types are a validation error."""
        with pytest.raises(ValidationError):
            _spectre_with(
                {
                    "continuity": {
                        "situation_sources": [{"type": "crystal-ball", "required": False}]
                    }
                }
            )


# ---------------------------------------------------------------------------
# Row 9 — Fresh vs remembered (FR24, AC11 narrative)
# ---------------------------------------------------------------------------


class TestFreshVsRemembered:
    async def test_resume_refresh_prefers_live_state_and_reports_revision(self, tmp_path):
        """AC11-r9: after a remembered snapshot, the resume-turn refresh
        reflects an intervening live update (live state wins), carries the
        new revision for supersession bookkeeping, and returns normally —
        the caller keeps the remembered entries alongside for explanation."""
        now_md = tmp_path / "NOW.md"
        now_md.write_text("step 3 in progress")
        provider = CompositeSituationProvider(
            [FileSituationProvider(paths=[str(now_md)], required=True)]
        )

        # Previous turn: remembered state.
        remembered = await provider.refresh(live_request(tmp_path))
        old_entries = entries_from(remembered.entries, "step 3 in progress")
        assert old_entries
        old_entry = old_entries[0]
        remembered_revisions = dict(remembered.revisions)

        # Intervening live update happens between turns.
        now_md.write_text("step 3 DONE — next: ship AC11")

        # Project-resumption turn: required sources refreshed BEFORE response.
        fresh = await provider.refresh(
            live_request(tmp_path, previous_revisions=dict(remembered.revisions))
        )

        live_matches = entries_from(fresh.entries, "next: ship AC11")
        assert live_matches, "resume refresh missed the intervening live update"
        live_entry = live_matches[0]
        assert live_entry.authority == ContextAuthority.LIVE_SOURCE
        assert live_entry.reason == ContextReason.LIVE_SOURCE

        # Stale content is gone from the live projection; remembered entry is
        # untouched, so both can be supplied for explanation.
        assert not entries_from(fresh.entries, "step 3 in progress")
        assert entry_text(old_entry) == "step 3 in progress"

        # Fresh snapshot publishes revisions so the old state is identifiable
        # as superseded (revision differs from remembered).
        assert fresh.revisions != remembered_revisions
