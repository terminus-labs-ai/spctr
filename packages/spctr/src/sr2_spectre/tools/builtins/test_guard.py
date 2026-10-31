"""Test Guard tool — detect authored-but-uncollected tests (phantom coverage).

Scans test files for functions that look like tests (def test_*) and compares
against pytest's actual collection. Reports the delta so the agent knows when
a green suite hides untested cases due to typos, naming errors, or collection
filters.

Registered as a builtin tool; the agent calls it AFTER running pytest to
confirm the green result is genuine.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Matches `def test_<name>` or `async def test_<name>` at any indentation.
_TEST_FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+(test_\w+)\s*\(", re.MULTILINE)


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the subprocess's whole process group (spawns run in their own
    session, so its PGID is the child's PID). Killing only the direct child
    would orphan its children, which can outlive a timed-out or cancelled
    call (spc-947 defect class). Falls back to killing just the child if the
    group is already gone."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Best-effort reap so the transport is cleaned up. Bounded: a killed
    shell can leave an orphaned child holding the pipes open, which makes an
    unbounded wait() hang; the exit status is not needed here.
    CancelledError must never be swallowed: it re-raises so an outside
    cancellation keeps propagating."""
    try:
        await asyncio.wait_for(proc.wait(), timeout=0.5)
    except asyncio.CancelledError:
        raise
    except Exception:
        pass


@dataclass(frozen=True)
class GuardResult:
    """Result of a phantom-coverage check."""

    total_collected: int
    total_authored: int
    uncollected: list[str] = field(default_factory=list)
    clean: bool = True


class GuardTool:
    """Detect authored-but-uncollected test functions.

    Runs ``pytest --collect-only --quiet`` in *test_dir*, counts collected
    test ids, then scans all ``test_*.py`` files for ``def test_*``
    definitions. Any authored name not present in the collected set is
    reported as a phantom-coverage risk.
    """

    name = "test_guard"
    description = (
        "Check for authored-but-uncollected tests (phantom coverage). "
        "Scans test_*.py files for 'def test_*' patterns and compares "
        "against pytest's collection output. Run AFTER pytest to confirm "
        "a green suite is genuine."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "test_dir": {
                "type": "string",
                "description": (
                    "Path to the directory containing test_*.py files "
                    "(default: current working directory)."
                ),
            },
        },
        "required": [],
    }

    _COLLECT_TIMEOUT = 30

    def __init__(self, cwd: str | None = None) -> None:
        self._cwd = Path(cwd).resolve() if cwd else Path.cwd()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def __call__(self, test_dir: str | None = None) -> str:
        """Run the phantom-coverage check. Returns a formatted report."""
        target = Path(test_dir).resolve() if test_dir else self._cwd
        result = await self._check(target)
        return self._format(result)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _check(self, test_dir: Path) -> GuardResult:
        collected = await self._collect_tests(test_dir)
        authored = self._scan_authored(test_dir)

        uncollected = sorted(authored - collected)
        return GuardResult(
            total_collected=len(collected),
            total_authored=len(authored),
            uncollected=uncollected,
            clean=len(uncollected) == 0,
        )

    async def _collect_tests(self, test_dir: Path) -> set[str]:
        """Run pytest --collect-only and return the set of short function names.

        Extracts just the function name (last component after `::`) from each
        collected test id so we can compare against authored names.
        """
        try:
            proc = await asyncio.create_subprocess_shell(
                f"cd {test_dir} && python -m pytest --collect-only -q 2>&1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Own session/process group: on timeout or cancellation the
                # whole group is killed, so a collect subprocess cannot
                # outlive the tool call (same idiom as the terminal tool).
                start_new_session=True,
            )
        except OSError:
            return set()

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._COLLECT_TIMEOUT
            )
        except asyncio.CancelledError:
            # Outside cancel (dispatch turn timeout, REPL Ctrl-C): kill the
            # group — the pre-fix code killed nothing — then propagate.
            _kill_process_group(proc)
            await _reap(proc)
            raise
        except (asyncio.TimeoutError, TimeoutError):
            _kill_process_group(proc)
            await _reap(proc)
            return set()

        output = (stdout.decode(errors="replace") + stderr.decode(errors="replace"))

        collected: set[str] = set()
        for line in output.splitlines():
            # Lines look like: tests/test_foo.py::test_bar [module]
            # or: <Function test_baz>
            # We want the short name after the last `::` or inside `<Function ...>`
            stripped = line.strip()
            if "::" in stripped:
                # Take the last segment after :: (e.g. "test_bar")
                func_name = stripped.rsplit("::", 1)[-1].split()[0]
                if func_name.startswith("test_"):
                    collected.add(func_name)
            elif "<Function " in stripped:
                # Fallback: <Function test_name>
                match = re.search(r"<Function\s+(test_\w+)>", stripped)
                if match:
                    collected.add(match.group(1))
        return collected

    @staticmethod
    def _scan_authored(test_dir: Path) -> set[str]:
        """Scan test_*.py files for def test_* patterns. Returns set of names."""
        authored: set[str] = set()

        for filepath in test_dir.glob("test_*.py"):
            try:
                text = filepath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            for match in _TEST_FUNC_RE.finditer(text):
                authored.add(match.group(1))

        return authored

    @staticmethod
    def _format(result: GuardResult) -> str:
        if result.clean:
            return (
                f"Test guard: CLEAN — {result.total_collected} collected, "
                f"{result.total_authored} authored. No phantom coverage detected."
            )

        lines = [
            f"Test guard: {len(result.uncollected)} UNCOLLECTED test(s) detected!",
            f"  Collected: {result.total_collected} | Authored: {result.total_authored}",
            "",
            "  Uncollected (authored but pytest did not collect):",
        ]
        for name in result.uncollected:
            lines.append(f"    - {name}")
        lines.append(
            "\n  FIX: Check for typos in function names, missing 'test_' prefix, "
            "or collection filters in conftest.py."
        )
        return "\n".join(lines)
