"""Guard test for the ``baseline_debt(bead_id)`` marker (FR5, spc-1040 AC6).

A ``baseline_debt`` marker cites the bead that tracks the known failure.
This guard fails when any collected ``baseline_debt`` marker carries no bead
id or a malformed one (not matching ``^(spc|sr2)-\\d+$``), so a marker can
never silently lose its ticket.  It also asserts that each of the five beads
FR5 names (spc-1017, spc-1029, spc-1030, spc-179, spc-965) has at least one
test carrying the marker, so a re-marking pass cannot quietly drop one.

The marker report is produced by a small pytest plugin
(``_synthetic_baseline_debt_plugin.py``) that a child ``pytest --collect-only
--report-baseline-debt`` run loads with ``-p``; the child collects a
workspace member's full suite (AC6-r1) or a throwaway synthetic module
(AC6-r2/r3).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# The beads FR5 names; each must have >= 1 test marked baseline_debt.
REQUIRED_BEADS = ("spc-1017", "spc-1029", "spc-1030", "spc-179", "spc-965")

BEAD_ID = r"(spc|sr2)-\d+"

ROOT = Path(__file__).resolve().parents[3]
PACKAGES = ROOT / "packages"
PLUGIN = Path(__file__).resolve().parent / "_synthetic_baseline_debt_plugin.py"


def _bead_id(mark: dict) -> str | None:
    """The bead id a marker carries, or None when it carries none."""
    if mark["kwargs"]:
        return mark["kwargs"].get("bead_id")
    if len(mark["args"]) == 1:
        return mark["args"][0]
    return None


def _validate_markers(
    markers: list[dict],
    require_beads: tuple[str, ...] = (),
) -> None:
    """Fail when any marker has no bead id or a malformed one, or when a
    required bead has no marked test."""
    bad = [
        f"{m['nodeid']} ({json.dumps(list(m['args']) + [v for v in m['kwargs'].values()])})"
        for m in markers
        if not isinstance(_bead_id(m), str) or not re.fullmatch(BEAD_ID, _bead_id(m))
    ]
    if bad:
        pytest.fail(
            "baseline_debt marker with no or malformed bead id "
            f"(must match ^{BEAD_ID}$):\n" + "\n".join(bad)
        )
    covered = {_bead_id(m) for m in markers}
    missing = [b for b in require_beads if b not in covered]
    if missing:
        pytest.fail(
            "FR5 beads with no test marked baseline_debt: " + ", ".join(missing)
        )


def _run_pytest_collect(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(PLUGIN.parent) + os.pathsep + env.get("PYTHONPATH", "")
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args,
         "-p", "no:cacheprovider",
         "-p", "_synthetic_baseline_debt_plugin",
         "--report-baseline-debt"],
        capture_output=True, text=True, cwd=cwd, env=env, timeout=300,
    )


def _markers_from_report(proc: subprocess.CompletedProcess[str]) -> list[dict]:
    for line in proc.stdout.splitlines():
        if line.startswith("BASELINE_DEBT_MARKERS "):
            return json.loads(line[len("BASELINE_DEBT_MARKERS "):])
    pytest.fail(
        f"no BASELINE_DEBT_MARKERS report from pytest run\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    )


@pytest.mark.baseline_debt("spc-1040")
def test_real_suite_markers_are_valid_and_cover_the_five_beads() -> None:
    """AC6-r1: the collected suite carries only valid bead ids and at least
    one marked test per bead of the five FR5 names."""
    markers: list[dict] = []
    for member in (PACKAGES / "sr2", PACKAGES / "spctr"):
        proc = _run_pytest_collect(
            [member.name, "--collect-only", "-q"], cwd=PACKAGES
        )
        markers.extend(_markers_from_report(proc))
    _validate_markers(markers, require_beads=REQUIRED_BEADS)


@pytest.mark.baseline_debt("spc-1040")
def test_guard_rejects_marker_without_bead_id(tmp_path: Path) -> None:
    """AC6-r2: a synthetic @pytest.mark.baseline_debt() makes the guard fail."""
    (tmp_path / "test_synthetic.py").write_text(
        "import pytest\n"
        "@pytest.mark.baseline_debt()\n"
        "def test_synthetic_no_bead_id(): pass\n",
        encoding="utf-8",
    )
    proc = _run_pytest_collect(["--collect-only", "-q"], cwd=tmp_path)
    markers = _markers_from_report(proc)
    with pytest.raises(pytest.fail.Exception) as exc:
        _validate_markers(markers)
    assert "test_synthetic_no_bead_id" in str(exc.value)


@pytest.mark.baseline_debt("spc-1040")
def test_guard_rejects_malformed_bead_id(tmp_path: Path) -> None:
    """AC6-r3: a synthetic baseline_debt("TODO") makes the guard fail."""
    (tmp_path / "test_synthetic.py").write_text(
        "import pytest\n"
        '@pytest.mark.baseline_debt("TODO")\n'
        "def test_synthetic_bad_bead_id(): pass\n",
        encoding="utf-8",
    )
    proc = _run_pytest_collect(["--collect-only", "-q"], cwd=tmp_path)
    markers = _markers_from_report(proc)
    with pytest.raises(pytest.fail.Exception) as exc:
        _validate_markers(markers)
    assert "test_synthetic_bad_bead_id" in str(exc.value)
