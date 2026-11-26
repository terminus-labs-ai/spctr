"""AC4 for scripts/collect_diff.py (spc-1066): silent-test-loss discrimination.

The gate must diff collected pytest node-ID sets between a base ref and the
current head, per workspace member, so an added test cancelling a deleted
one in pass/fail counts (the spc-1058 shape) is caught by name.

Case-table rows, end-to-end on a throwaway two-member git repo:

* r1  base == head             -> no additions, no removals; exit 0
* r2  head deletes one test    -> removed node ID reported exactly; nonzero exit
* r3  head adds one test       -> addition reported; exit 0
* r4  add + remove (spc-1058)  -> removal flagged despite cancelling counts
* r5  renamed test             -> one removal + one addition; nonzero exit
* r6  removal in second member -> detected; nonzero exit

Unit rows cover the pure logic: node-ID parsing of ``--collect-only -q``
output, set diffing, and the --allow sanction rules.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "scripts" / "collect_diff.py"

_GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
_ALPHA_BASE = (
    "def test_kept():\n"
    "    assert True\n"
    "\n"
    "\n"
    "def test_gone_at_head():\n"
    "    assert True\n"
)
_BETA_BASE = (
    "def test_beta_kept():\n"
    "    assert True\n"
    "\n"
    "\n"
    "def test_beta_gone():\n"
    "    assert True\n"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("collect_diff_1066", TOOL)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: GateVerdict is a @dataclass with lazy annotations,
    # which resolves field types through sys.modules by the module's name.
    sys.modules["collect_diff_1066"] = module
    spec.loader.exec_module(module)
    return module


def _make_repo(root: Path) -> None:
    """Throwaway two-member git repo: alpha/tests and beta/tests."""
    (root / "alpha" / "tests").mkdir(parents=True)
    (root / "beta" / "tests").mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "alpha" / "tests" / "test_a.py").write_text(_ALPHA_BASE)
    (root / "beta" / "tests" / "test_b.py").write_text(_BETA_BASE)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run([*_GIT, "add", "-A"], cwd=root, check=True)
    subprocess.run([*_GIT, "commit", "-q", "-m", "base"], cwd=root, check=True)


def _branch(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, capture_output=True,
        text=True, check=True,
    ).stdout.strip()


def _run_tool(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable, str(TOOL),
        "--repo-root", str(root),
        "--members", "alpha", "beta",
        "--pytest-cmd", f"{sys.executable} -m pytest",
        *extra,
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300, cwd=root)


def _added_removed(stdout: str) -> tuple[list[str], list[str]]:
    added = [l.split(None, 1)[1].strip() for l in stdout.splitlines() if l.startswith("  + ")]
    removed = [l.split(None, 1)[1].strip() for l in stdout.splitlines() if l.startswith("  - ")]
    return added, removed


# --- AC4, case-table rows (end-to-end) ---------------------------------------


def test_ac4_r1_base_equals_head_no_changes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _make_repo(root)
    res = _run_tool(root, _branch(root))
    assert res.returncode == 0, res.stdout + res.stderr
    added, removed = _added_removed(res.stdout)
    assert added == [] and removed == []
    assert "result: OK" in res.stdout


def test_ac4_r2_head_deletes_one_test_flags_exact_node_id(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _make_repo(root)
    (root / "alpha" / "tests" / "test_a.py").write_text("def test_kept():\n    assert True\n")
    res = _run_tool(root, _branch(root))
    assert res.returncode != 0, res.stdout + res.stderr
    added, removed = _added_removed(res.stdout)
    assert removed == ["alpha/tests/test_a.py::test_gone_at_head"]
    assert added == []


def test_ac4_r3_head_adds_one_test_is_tolerated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _make_repo(root)
    (root / "alpha" / "tests" / "test_new.py").write_text(
        "def test_fresh():\n    assert True\n"
    )
    res = _run_tool(root, _branch(root))
    assert res.returncode == 0, res.stdout + res.stderr
    added, removed = _added_removed(res.stdout)
    assert added == ["alpha/tests/test_new.py::test_fresh"]
    assert removed == []


def test_ac4_r4_spc1058_shape_add_and_remove_cancels_counts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _make_repo(root)
    # Replace the deleted test with a new one: member totals stay equal.
    (root / "alpha" / "tests" / "test_a.py").write_text(
        "def test_kept():\n    assert True\n\n\ndef test_replacement():\n    assert True\n"
    )
    res = _run_tool(root, _branch(root))
    assert res.returncode != 0, res.stdout + res.stderr
    added, removed = _added_removed(res.stdout)
    assert added == ["alpha/tests/test_a.py::test_replacement"]
    assert removed == ["alpha/tests/test_a.py::test_gone_at_head"]
    # The cancellation the finding warns about: per-member counts are equal.
    assert "alpha: base 2 head 2" in res.stdout


def test_ac4_r5_rename_shows_one_removal_plus_one_addition(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _make_repo(root)
    (root / "alpha" / "tests" / "test_a.py").write_text(
        _ALPHA_BASE.replace("def test_kept():", "def test_renamed():")
    )
    res = _run_tool(root, _branch(root))
    assert res.returncode != 0, res.stdout + res.stderr
    added, removed = _added_removed(res.stdout)
    assert added == ["alpha/tests/test_a.py::test_renamed"]
    assert removed == ["alpha/tests/test_a.py::test_kept"]


def test_ac4_r6_removal_in_second_member_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _make_repo(root)
    (root / "beta" / "tests" / "test_b.py").write_text("def test_beta_kept():\n    assert True\n")
    res = _run_tool(root, _branch(root))
    assert res.returncode != 0, res.stdout + res.stderr
    added, removed = _added_removed(res.stdout)
    assert removed == ["beta/tests/test_b.py::test_beta_gone"]
    assert added == []


def test_sanctioned_removal_via_allow_exits_zero(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _make_repo(root)
    (root / "alpha" / "tests" / "test_a.py").write_text("def test_kept():\n    assert True\n")
    res = _run_tool(
        root, _branch(root), "--allow", "alpha/tests/test_a.py::test_gone_at_head"
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "  ~ alpha/tests/test_a.py::test_gone_at_head" in res.stdout


# --- AC4, unit rows on the pure logic -----------------------------------------


def test_parse_collected_node_ids_drops_summary_and_noise() -> None:
    tool = _load_tool()
    output = (
        "tests/test_a.py::test_one\n"
        "tests/test_b.py::test_two[param with spaces]\n"
        "tests/test_b.py::test_three\n"
        "\n"
        "3 tests collected in 0.12s\n"
    )
    assert tool.parse_collected_node_ids(output) == {
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two[param with spaces]",
        "tests/test_b.py::test_three",
    }


def test_parse_collected_node_ids_ignores_error_sections() -> None:
    tool = _load_tool()
    output = (
        "tests/test_ok.py::test_a\n"
        "==================================== ERRORS ====================================\n"
        "______________________ ERROR collecting tests/test_bad.py ______________________\n"
        "E     SyntaxError: invalid syntax\n"
        "=========================== short test summary info ============================\n"
        "ERROR tests/test_bad.py\n"
        "1 test collected, 1 error in 0.10s\n"
    )
    assert tool.parse_collected_node_ids(output) == {"tests/test_ok.py::test_a"}


def test_diff_and_gate_exit_codes() -> None:
    tool = _load_tool()
    base = {"m/tests/t.py::a", "m/tests/t.py::b", "m/tests/t.py::c"}
    head = {"m/tests/t.py::a", "m/tests/t.py::c", "m/tests/t.py::new"}
    verdict = tool.GateVerdict.evaluate(base, head, [])
    assert verdict.added == ["m/tests/t.py::new"]
    assert verdict.removed == ["m/tests/t.py::b"]
    assert verdict.exit_code == tool.EXIT_REMOVED

    clean = tool.GateVerdict.evaluate(base, base, [])
    assert clean.added == [] and clean.removed == []
    assert clean.exit_code == tool.EXIT_OK

    sanctioned = tool.GateVerdict.evaluate(base, head, ["m/tests/t.py"])
    assert sanctioned.removed == [] and sanctioned.sanctioned == ["m/tests/t.py::b"]
    assert sanctioned.exit_code == tool.EXIT_OK


def test_is_sanctioned_match_rules() -> None:
    tool = _load_tool()
    node = "m/tests/test_x.py::TestClass::test_y"
    assert tool.is_sanctioned(node, [node])
    assert tool.is_sanctioned(node, ["m/tests/test_x.py"])
    assert tool.is_sanctioned(node, ["m/tests/test_x.py::TestClass::"])
    assert not tool.is_sanctioned(node, ["m/tests/test_x"])
    assert not tool.is_sanctioned(node, ["m/tests/test_x.py::TestClass::other"])
