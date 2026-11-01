"""Residue check for the sr2_spectre -> spctr rename (spc-1039, AC5).

FR4 (spec spctr-public-v0): ``sr2_spectre`` and ``sr2-spectre`` must not
appear in tracked files of the repository, except the allow-list below —
the rename script itself (it names the old tokens as its rewrite targets)
and this test (it must name them to forbid them). ``git grep`` is fixed
to the committed tree, so the check judges what would actually ship.

AC5-r2: planting the old name in a tracked file must make the check fail,
naming the file — exercised against a scratch git repo, never this one.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# REPO_ROOT: repo root seen from packages/spctr/tests/ (three levels up).
REPO_ROOT = Path(__file__).resolve().parents[3]

OLD_NAME = re.compile(r"sr2[-_]spectre")

# Allow-list, repo-relative: files that name the old tokens by design.
ALLOW_LIST = frozenset(
    {
        "scripts/rename_spctr.py",
        "packages/spctr/tests/test_entry_points.py",
        "packages/spctr/tests/test_rename_residue.py",
    }
)


def _grep_old_name(root: Path) -> list[str]:
    """Tracked file paths in ``root``'s HEAD containing an old name.

    ``-l -I -F``: file names only, binary files skipped, literal fixed
    strings for the two spellings (joined with the file list).
    """
    result = subprocess.run(
        ["git", "grep", "-l", "-I", "-F", "-e", "sr2_spectre", "-e", "sr2-spectre", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), (
        f"git grep failed ({result.returncode}): {result.stderr.strip()}"
    )
    paths = set()
    for line in result.stdout.splitlines():
        # HEAD:path form
        _, _, rel = line.partition(":")
        if rel:
            paths.add(rel)
    return sorted(paths)


def test_ac5_r1_no_old_name_outside_allow_list() -> None:
    """AC5-r1: no tracked file outside the allow-list carries the old name."""
    offenders = [
        p for p in _grep_old_name(REPO_ROOT) if p not in ALLOW_LIST
    ]
    assert offenders == [], (
        "sr2_spectre/sr2-spectre residue found outside the allow-list: "
        + ", ".join(offenders)
    )


def test_ac5_r2_planted_residue_fails_naming_the_file(tmp_path: Path) -> None:
    """AC5-r2: a planted old name in a tracked file fails the check, and
    the failure names that file."""
    subprocess.run(
        ["git", "init", "-q"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    planted = tmp_path / "planted_module.py"
    planted.write_text("from sr2_spectre.config import SpectreConfig\n", encoding="utf-8")
    clean = tmp_path / "clean_module.py"
    clean.write_text("from spctr.config import SpctrConfig\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-A"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=spc-1039 residue test",
            "commit",
            "-q",
            "-m",
            "plant residue",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    found = _grep_old_name(tmp_path)
    assert found == ["planted_module.py"], (
        "residue check must fail naming exactly the planted file"
    )
    assert "clean_module.py" not in found


def test_allow_list_entries_exist() -> None:
    """Guard the guard: an allow-list entry that no longer exists hides a
    revoked exemption instead of a real problem."""
    for rel in ALLOW_LIST:
        assert (REPO_ROOT / rel).is_file(), f"allow-list entry missing: {rel}"
