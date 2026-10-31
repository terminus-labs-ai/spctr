"""Monorepo layout checks for spc-1038 (spec spctr-public-v0, AC1/AC2).

C-owned acceptance checks for the uv-workspace assembly. One test per
case-table row; run them from the workspace root with::

    uv run --frozen python -m pytest tests/test_monorepo_layout.py

AC1 (FR1): ``uv sync --frozen`` at the root installs both packages from
the workspace — no git-URL source for sr2 anywhere in the metadata, one
root lock. AC2 (FR2): file history under ``packages/sr2`` follows into
commits merged in from the old sr2 repository, and history under
``packages/spctr`` follows through the ``git mv`` into the pre-move
sr2-spectre history.

AC1-r1 clones the revision under test into a temp dir and syncs it with
a throwaway venv (``UV_PROJECT_ENVIRONMENT``); it needs network access,
like the initial root sync does, and is skipped when cloning fails.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ancestry anchors for AC2, committed by the assembly itself.
MERGE_COMMIT = "a4ecb78da6f7ec4c6eeea045e75b794cb61e8855"  # sr2 history -> packages/sr2
PRE_MERGE_SPCT_BASE = "e9403371e2a20d026bddc1c7e73577fff46fc281"  # last old-repo commit

SR2_EARLY_COMMITS = {
    "53d29381cc67acfdbe62abac66740b0a0bdc2dd5",  # "Initial commit: SR2 v1.0.0"
    "962ce486ef963496331ff928f8bd6fbd772087eb",  # "feat: plugin architecture refactoring"
}

SOURCE_KEYS = (
    "{ git =",
    "{ git = ",
    "git+https://",
    "git+ssh://",
    "git+file://",
)


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _commit_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
    )
    return result.returncode == 0


def _pyproject_sources(path: Path) -> dict[str, str]:
    """Return {tool.uv.sources} entries as name -> raw line."""
    sources: dict[str, str] = {}
    in_sources = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_sources = stripped == "[tool.uv.sources]"
        elif in_sources and "=" in stripped:
            name, _, value = stripped.partition("=")
            sources[name.strip()] = value.strip()
    return sources


def test_ac1_r1_root_uv_sync_frozen_installs_both_packages_from_workspace(
    tmp_path: Path,
) -> None:
    """AC1-r1: fresh clone of this revision; root uv sync --frozen exits 0
    and installs both packages editable from paths under packages/."""
    clone = tmp_path / "clone"
    head = _git("rev-parse", "HEAD")
    # Prefer a real clone of origin (the criterion says "fresh clone");
    # fall back to a local clone when the remote lacks this build ref or
    # the network is down. Either way uv runs on a pristine checkout.
    attempts = [
        ["git", "clone", "--quiet", "--branch", head, _git("remote", "get-url", "origin"), str(clone)],
        ["git", "clone", "--quiet", str(REPO_ROOT), str(clone)],
    ]
    for argv in attempts:
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode == 0:
            break
    else:
        pytest.skip(f"cannot clone for AC1-r1: {result.stderr.strip()[:200]}")
    if _git("rev-parse", "HEAD", cwd=clone) != head:
        _git("checkout", "--quiet", "--detach", head, cwd=clone)

    env = {
        **os.environ,
        # Keep the throwaway venv outside the clone so uv sees a pristine
        # fresh-checkout state (no .venv, no __pycache__).
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
    }
    sync = subprocess.run(
        ["uv", "sync", "--frozen"],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
    )
    assert sync.returncode == 0, f"uv sync --frozen failed:\n{sync.stderr[-1500:]}"

    venv_python = tmp_path / "venv" / "bin" / "python"
    probe = subprocess.run(
        [
            str(venv_python),
            "-c",
            "import json\n"
            "from importlib.metadata import distribution\n"
            "out = {}\n"
            "for name in ('sr2', 'sr2-spectre'):\n"
            "    raw = distribution(name).read_text('direct_url.json')\n"
            "    out[name] = json.loads(raw) if raw else None\n"
            "print(json.dumps(out))",
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, f"import probe failed:\n{probe.stderr[-800:]}"
    installed = json.loads(probe.stdout)

    expected = {
        "sr2": "{clone}/packages/sr2",
        "sr2-spectre": "{clone}/packages/spctr",
    }
    for name, template in expected.items():
        want = Path(template.format(clone=clone)).resolve()
        info = installed[name]
        assert info is not None, f"{name} has no direct_url.json (not a local install)"
        dir_info = info.get("dir_info") or {}
        assert dir_info.get("editable") is True, (
            f"{name} is not an editable install: {info}"
        )
        got = Path(info["url"].removeprefix("file://")).resolve()
        assert got == want, f"{name}: expected workspace path {want}, got {got}"


def test_ac1_r2_no_git_source_for_sr2_in_any_pyproject() -> None:
    """AC1-r2: no root or member pyproject declares sr2 (or anything)
    through a git source; both workspace members use { workspace = true }."""
    pyprojects = [REPO_ROOT / "pyproject.toml"] + sorted(
        (REPO_ROOT / "packages").glob("*/pyproject.toml")
    )
    assert len(pyprojects) == 3, f"expected 3 pyproject.toml files, got {pyprojects}"

    for path in pyprojects:
        text = path.read_text()
        assert not any(key in text for key in SOURCE_KEYS), (
            f"git source found in {path.relative_to(REPO_ROOT)}"
        )

    root_sources = _pyproject_sources(REPO_ROOT / "pyproject.toml")
    member = _pyproject_sources(REPO_ROOT / "packages" / "spctr" / "pyproject.toml")
    assert re.search(r"workspace\s*=\s*true", root_sources.get("sr2", "")), (
        "root pyproject: sr2 is not { workspace = true }"
    )
    assert re.search(r"workspace\s*=\s*true", member.get("sr2", "")), (
        "packages/spctr: sr2 is not { workspace = true }"
    )


def test_ac1_r3_single_root_uv_lock_declares_both_members() -> None:
    """AC1-r3: uv reads a member's own uv.lock only while that member is
    visible in the working tree (uv errors on a stale member lock), so
    "one root lock" means: no lock exists for any current member, and
    the root lock declares every member. The historical
    `packages/sr2/uv.lock` survives only as a deleted-in-HEAD history
    record for `git log --follow`; it must stay deleted on disk.
    The behavioural half of the row (both suites run from the root at
    the known baseline) is scripts/run_tests.sh."""
    for member in ("sr2", "spctr"):
        member_lock = REPO_ROOT / "packages" / member / "uv.lock"
        assert not member_lock.exists(), f"stale member lock on disk: {member_lock}"
        listed = _git("ls-files", "--", str(member_lock.relative_to(REPO_ROOT)))
        assert listed == "", f"member lock still tracked in the index: {member_lock}"

    text = (REPO_ROOT / "uv.lock").read_text()
    for member in ("sr2", "sr2-spectre"):
        assert f'name = "{member}"' in text, f"uv.lock does not pin {member}"
    workspace_section = re.search(
        r"\[manifest\]\nmembers = \[\n(.*?)\]", text, re.DOTALL
    )
    assert workspace_section is not None, "uv.lock has no [manifest] members list"
    members = workspace_section.group(1)
    assert '"sr2"' in members and '"sr2-spectre"' in members, (
        f"uv.lock manifest members incomplete:\n{members}"
    )


def test_ac2_r1_sr2_file_history_follows_into_premerge_commits() -> None:
    """AC2-r1: git log --follow on a file under packages/sr2 returns the
    commits sr2 carried before the merge, including its initial commit."""
    follow = set(
        _git("log", "--follow", "--format=%H", "--", "packages/sr2/src/sr2/__init__.py").split()
    )
    assert SR2_EARLY_COMMITS <= follow, (
        "packages/sr2 history does not follow into the merged sr2 history"
    )
    for commit in SR2_EARLY_COMMITS:
        assert _commit_is_ancestor(commit, MERGE_COMMIT), (
            f"{commit} is not older than the merge commit"
        )


def test_ac2_r2_spctr_file_history_follows_through_the_move() -> None:
    """AC2-r2: git log --follow on a file under packages/spctr returns
    the pre-move sr2-spectre commits, which predate the merge commit."""
    follow = set(
        _git("log", "--follow", "--format=%H", "--", "packages/spctr/pyproject.toml").split()
    )
    assert PRE_MERGE_SPCT_BASE in follow, (
        "packages/spctr history does not follow through the move into the old repo"
    )
    assert _commit_is_ancestor(PRE_MERGE_SPCT_BASE, MERGE_COMMIT), (
        "the pre-move base commit is not older than the merge commit"
    )
