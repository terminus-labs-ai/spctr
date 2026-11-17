"""obsidian-vq2k (sr2-91 follow-up) — run_baseline retention safety.

`run_baseline()` cleans the pinned default results tree of superseded run
dirs; the committed (Git-tracked) published run dir must never be deleted
by a local re-run under a different run id, and nothing may be deleted
when the tracked-paths probe fails (Git absent / not a repository).

The tests monkeypatch ``_tracked_result_dirs`` — the Git probe — rather
than running git inside ``tmp_path``. A real ``run_baseline`` execution
drives the actual retention loop against the default root, so the
consumer is never tested through a fake-shaped seam. The current run's
id is computed before the new run dir exists (the id hashes the corpus
tree, so it is stable only while the tree is unchanged).

WARNING: these tests drive the real default results root
(``evals/continuity/results``) — they create and delete untracked run
dirs in their own worktree's checkout. They must NOT run concurrently
with anything else (another pytest, a ``run_baseline`` run, or a human)
inside the same worktree: a concurrent process could create or remove
run dirs mid-test, and the teardowns remove exactly the dirs they made
but the retention loop under test removes by mtime order, so a
concurrent untracked dir could be deleted out from under it.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evals.continuity import runner as crunner

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "evals" / "continuity" / "results"
TRACKED_PUBLISHED_DIR = RESULTS_ROOT / (
    "b75ddfa04ce6b4ea3383ab7194cb666aa711205241788d44ec310e054b14f964"
    "+3a9dbbe8aa119c05afc64314c16fff96dee484c2"
)

# Two untracked stale run dirs, ordered by mtime (A older than B).
_STALE_A = ("aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000"
            "+b1110000b1110000b1110000b1110000b1110000")
_STALE_B = ("cccc0000cccc0000cccc0000cccc0000cccc0000"
            "+d2220000d2220000d2220000d2220000d2220000")

_ONE_WEEK = timedelta(weeks=1)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True)


def _make_stale_dir(name: str, age: timedelta) -> Path:
    """Create an untracked stale run dir under the default results root."""
    d = RESULTS_ROOT / name
    d.mkdir(parents=True)
    shutil.copyfile(TRACKED_PUBLISHED_DIR / "metrics.json", d / "metrics.json")
    (d / "observations.jsonl").write_text("")
    ts = (datetime.now(tz=timezone.utc) - age).timestamp()
    os.utime(d, (ts, ts))
    return d


@pytest.fixture
def tracked_probe(monkeypatch):
    """Pin the tracked-paths probe to the committed published dir.

    Verifies the real probe agrees (so the fake is a true stand-in for
    this worktree), then replaces it via monkeypatch so the retention
    tests never depend on the repo's incidental tracked state.
    """
    assert TRACKED_PUBLISHED_DIR.is_dir(), (
        "the committed published run dir must exist in this worktree")
    real = crunner._tracked_result_dirs(RESULTS_ROOT)
    assert real is not None and TRACKED_PUBLISHED_DIR in real, (
        f"committed published dir {TRACKED_PUBLISHED_DIR.name!r} "
        "must appear in git ls-files of the results root")
    monkeypatch.setattr(
        crunner, "_tracked_result_dirs",
        lambda root: {TRACKED_PUBLISHED_DIR})
    yield TRACKED_PUBLISHED_DIR


@pytest.fixture
def stale_dirs():
    """Two untracked stale run dirs; removed again in teardown."""
    stale_a = _make_stale_dir(_STALE_A, _ONE_WEEK)
    stale_b = _make_stale_dir(_STALE_B, _ONE_WEEK / 2)
    try:
        yield stale_a, stale_b
    finally:
        for d in (stale_a, stale_b):
            if d.exists():
                shutil.rmtree(d)


def test_run_baseline_removes_only_untracked_stale_dirs(
        tracked_probe, stale_dirs):
    """AC1: one tracked dir + two untracked stale dirs — the retention
    loop removes only the untracked ones; the tracked dir and its
    artifacts survive."""
    stale_a, stale_b = stale_dirs
    current = RESULTS_ROOT / crunner.run_id()  # before the run dir exists

    run_dir = crunner.run_baseline()
    try:
        assert Path(run_dir) == current, (
            f"AC1: run_baseline must publish to the default root under "
            f"run_id(); got {Path(run_dir)!r}, expected {current!r}")
        assert not stale_a.exists(), (
            "AC1: untracked stale dir A must be removed")
        assert not stale_b.exists(), (
            "AC1: untracked stale dir B must be removed")
        assert TRACKED_PUBLISHED_DIR.is_dir(), (
            "AC1: the Git-tracked published run dir must survive a re-run")
        for fname in ("metrics.json", "observations.jsonl"):
            assert (TRACKED_PUBLISHED_DIR / fname).is_file(), (
                f"AC1: tracked artifact {fname!r} missing after a re-run")
    finally:
        shutil.rmtree(run_dir)


def test_run_baseline_probe_failure_removes_nothing_and_warns(
        tracked_probe, stale_dirs, monkeypatch, caplog):
    """AC1: when the tracked-paths probe fails (Git absent), nothing is
    removed and a warning is logged."""
    stale_a, stale_b = stale_dirs
    current = RESULTS_ROOT / crunner.run_id()  # before the run dir exists

    def _probe_absent(root: Path):
        raise OSError("git: command not found")

    monkeypatch.setattr(crunner, "_tracked_result_dirs", _probe_absent)
    try:
        with caplog.at_level(logging.WARNING,
                             logger="evals.continuity.runner"):
            crunner.run_baseline()
    finally:
        if current.exists():
            shutil.rmtree(current)

    assert stale_a.exists(), "AC1: probe failure must not remove stale dir A"
    assert stale_b.exists(), "AC1: probe failure must not remove stale dir B"
    assert TRACKED_PUBLISHED_DIR.is_dir(), (
        "AC1: probe failure must not remove the tracked dir")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "AC1: a warning must be logged when the probe fails"
    assert any("git" in r.getMessage().lower()
               or "tracked" in r.getMessage().lower() for r in warnings), (
        "AC1: the warning must name the tracked-paths probe that failed; "
        f"got {[r.getMessage() for r in warnings]!r}")


def test_tracked_result_dirs_helper_shape():
    """The helper reports the committed published run dir as tracked
    (first path component of each listed file, resolved under the root),
    excludes untracked run-dir names, and returns None outside a
    repository (delete-nothing contract)."""
    tracked = crunner._tracked_result_dirs(RESULTS_ROOT)
    assert tracked is not None, (
        "the results root must sit inside a git repository")
    assert TRACKED_PUBLISHED_DIR in tracked
    assert (RESULTS_ROOT / _STALE_A) not in tracked
    assert (RESULTS_ROOT / _STALE_B) not in tracked
    # Every reported dir sits directly under the results root.
    for d in tracked:
        assert d.parent == RESULTS_ROOT
    # Outside any repository: unknown, not "nothing tracked".
    outside = Path("/proc")
    assert subprocess.run(
        ["git", "-C", str(outside), "rev-parse", "--is-inside-work-tree"],
        capture_output=True).returncode != 0, (
        f"probe sandbox {outside} unexpectedly inside a repository")
    assert crunner._tracked_result_dirs(outside) is None


def test_committed_results_unchanged_by_branch():
    """AC2: the committed published run's bytes match HEAD exactly (the
    branch must not alter them). Paths are converted to repo-root
    relative before the HEAD lookup."""
    proc = _git("ls-files", "-z", "--", "evals/continuity/results")
    assert proc.returncode == 0
    committed = [n.decode() for n in proc.stdout.split(b"\0") if n]
    assert committed, "the published run dir must be Git-tracked"
    for rel in committed:
        # ls-files prints cwd-relative paths; HEAD: needs repo-root ones.
        at_root = (REPO_ROOT / rel).resolve().relative_to(
            _repo_toplevel()).as_posix()
        blob = _git("cat-file", "blob", f"HEAD:{at_root}")
        assert blob.returncode == 0, f"AC2: {at_root!r} missing from HEAD"
        assert (REPO_ROOT / rel).read_bytes() == blob.stdout, (
            f"AC2: worktree bytes of {rel!r} diverge from HEAD")


def _repo_toplevel() -> Path:
    proc = _git("rev-parse", "--show-toplevel")
    assert proc.returncode == 0
    return Path(proc.stdout.decode().strip())


def _results_prefix() -> str:
    """The results root's repo-root-relative prefix, ``""`` at the root.

    Mirrors the probe's own call: ``rev-parse --show-prefix`` from the
    results root; the output already ends in ``/`` (or is empty at the
    root itself)."""
    proc = _git("-C", str(RESULTS_ROOT), "rev-parse", "--show-prefix")
    assert proc.returncode == 0, (
        f"results root {RESULTS_ROOT} unexpectedly outside a repository")
    return proc.stdout.decode().strip()


def test_tracked_result_dirs_maps_repo_root_relative_listing(
        monkeypatch):
    """AC1: an exported absolute ``GIT_DIR`` makes the probe list
    repo-root-relative names for the whole repository; the helper must
    still report exactly the committed published dir under the results
    root (the base code mapped the repo's first components, e.g.
    ``<root>/.github``, so the published dir was missing and eligible
    for deletion). The listing is simulated as the probe sees it."""
    prefix = _results_prefix()
    # The listing is repo-root-relative (poisoned GIT_DIR view):
    # paths under *root* are ``<prefix><run-dir>/<file>``, other
    # repo paths are unrelated.
    published_rel = (
        f"{prefix}{TRACKED_PUBLISHED_DIR.name}/metrics.json\0"
        f"{prefix}{TRACKED_PUBLISHED_DIR.name}/observations.jsonl\0"
    )
    listing = (published_rel
               + ".github/workflows/ci.yml\0.gitignore\0")

    def _fake_run(cmd, *a, **kw):
        import types

        if "ls-files" in cmd:
            return types.SimpleNamespace(returncode=0, stdout=listing)
        if "--show-prefix" in cmd:
            # The poisoned view reports the repo root.
            return types.SimpleNamespace(returncode=0, stdout="")
        if "--is-inside-work-tree" in cmd:
            return types.SimpleNamespace(returncode=0, stdout="true")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(crunner.subprocess, "run", _fake_run)
    tracked = crunner._tracked_result_dirs(RESULTS_ROOT)
    assert tracked is not None, (
        "AC1: the probe must not bail out on a poisoned GIT_DIR view")
    assert TRACKED_PUBLISHED_DIR in tracked, (
        "AC1: the committed published dir must survive a "
        "repo-root-relative listing; the helper would hand the caller "
        f"a set lacking it and the retention loop would rmtree it; "
        f"got {tracked!r}")
    for d in tracked:
        assert d.parent == RESULTS_ROOT, (
            f"AC1: every reported dir must sit directly under the "
            f"results root; got {d!r}")
    assert (RESULTS_ROOT / "packages") not in tracked
    assert (RESULTS_ROOT / ".github") not in tracked


def test_tracked_result_dirs_hostile_git_dir_env(monkeypatch):
    """AC1 (real environment): with ``GIT_DIR`` exported to this
    worktree's absolute git dir, the real probe must either report the
    committed published dir with every dir under the results root, or
    refuse the view and return None (delete nothing). It must never
    return a set that lacks the published dir while still containing
    dirs the retention loop would treat as the tracked set."""
    git_dir = _repo_toplevel() / ".git"
    monkeypatch.setenv("GIT_DIR", str(git_dir))
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)
    tracked = crunner._tracked_result_dirs(RESULTS_ROOT)
    assert tracked is None or (
        TRACKED_PUBLISHED_DIR in tracked
        and all(d.parent == RESULTS_ROOT for d in tracked)), (
        "AC1: under an exported absolute GIT_DIR the helper must either "
        "report the published dir (every dir under the results root) or "
        f"return None; got {tracked!r}")
