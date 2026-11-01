"""AC3 for the scripted rename (spec spctr-public-v0 FR3, dec-277, dec-280, dec-292).

AC3-r1: a second run of scripts/rename_spctr.py changes nothing.
AC3-r2: ``import spctr`` works; ``sr2_spectre`` is not importable (no shim).
AC3-r3: the ``spctr`` console script runs ``--help``; no ``sr2-spectre`` script.
AC3-r4: spctr's on-disk names are the new ones, with no fallback to the old.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "rename_spctr.py"
SRC = REPO_ROOT / "packages" / "spctr" / "src" / "spctr"
BIN = Path(sys.executable).parent


def _load_script():
    spec = importlib.util.spec_from_file_location("rename_spctr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ac3_r1_second_run_changes_nothing(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    clone.mkdir()
    # Snapshot the working tree as it stands (tracked files, uncommitted edits
    # included) into a fresh repository, so the script's git calls see it.
    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    for rel in filter(None, files):
        src = REPO_ROOT / rel
        if src.is_file():
            dst = clone / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q"], cwd=clone, check=True)
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "snapshot"], cwd=clone, check=True)
    before = subprocess.run(["git", "status", "--porcelain"], cwd=clone, capture_output=True, text=True, check=True).stdout
    result = subprocess.run(
        [sys.executable, "scripts/rename_spctr.py"], cwd=clone, capture_output=True, text=True, check=True
    )
    assert "0 file(s) rewritten" in result.stdout
    assert "already renamed" in result.stdout
    after = subprocess.run(["git", "status", "--porcelain"], cwd=clone, capture_output=True, text=True, check=True).stdout
    assert after == before


def test_ac3_r1_rewrite_is_idempotent_on_its_own_output() -> None:
    rewrite = _load_script().rewrite_text
    sample = "import sr2_spectre.cli\nSpectreConfig ~/.sr2-spectre spectre.yaml x.spectre.zuliprc SPECTRE_X\n"
    once = rewrite(sample)
    assert once == "import spctr.cli\nSpctrConfig ~/.spctr spctr.yaml x.spctr.zuliprc SPCTR_X\n"
    assert rewrite(once) == once


def test_rewrite_preserves_external_environment_names() -> None:
    rewrite = _load_script().rewrite_text
    text = 'os.environ.get("SPECTRE_MEMORY_TEST_DSN", ".../spectre_memory_test") # Spectre'
    assert rewrite(text) == 'os.environ.get("SPECTRE_MEMORY_TEST_DSN", ".../spectre_memory_test") # spctr'


def test_ac3_r2_spctr_imports_and_old_name_does_not() -> None:
    import spctr  # noqa: F401

    assert importlib.util.find_spec("sr2_spectre") is None
    probe = subprocess.run([sys.executable, "-c", "import sr2_spectre"], capture_output=True, text=True, check=False)
    assert probe.returncode != 0
    assert "ModuleNotFoundError" in probe.stderr


def test_ac3_r3_console_script_runs_and_old_one_is_absent() -> None:
    result = subprocess.run([str(BIN / "spctr"), "--help"], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert not (BIN / "sr2-spectre").exists()


def test_ac3_r4_on_disk_names_are_new_with_no_fallback() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py"))
    for old in (r"\.sr2-spectre", r"spectre\.yaml", r"\.spectre\.zuliprc"):
        assert not re.search(old, text), f"old on-disk name {old!r} still in spctr source"
    for new in (".spctr", "spctr.yaml"):
        assert new in text, f"new on-disk name {new!r} missing from spctr source"
