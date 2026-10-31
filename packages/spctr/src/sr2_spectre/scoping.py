"""Per-session systemd scope for sr2-spectre processes.

When launched without an existing dedicated scope, re-exec the process under
``systemd-run --user --scope`` with a memory cap so a leak in any child is
confined to that scope and ``systemctl --user stop <unit>.scope`` cleans up
every orphan.

Skip conditions (the process continues unsoped):
- ``SR2_SCOPE_UNIT`` is set — gmap or workflow dispatch already scoped us.
- ``INVOCATION_ID`` is set — running as a systemd service (e.g. sr2-discord@*).
- ``SR2_NO_SCOPE=1`` — explicit opt-out for debugging or constrained environments.
- ``systemd-run`` is not on PATH — print a warning and continue.

Configuration via environment:
- ``SR2_SCOPE_MEMORY_MAX``: memory limit, default ``6G``.
- ``SR2_SCOPE_MEMORY_SWAP_MAX``: swap limit, default ``0``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

_DEFAULT_MEMORY_MAX = "6G"
_DEFAULT_MEMORY_SWAP_MAX = "0"

_UNIT_PREFIX = "spectre-"
_SAFE_UNIT_RE = re.compile(r"[^A-Za-z0-9:_.\-]")


def _sanitize_unit(name: str) -> str:
    return _SAFE_UNIT_RE.sub("_", name)


def _unit_name(argv: list[str]) -> str:
    """Derive a scope unit name from the command arguments or PID."""
    for i, arg in enumerate(argv):
        if arg == "--session-id" and i + 1 < len(argv):
            return _UNIT_PREFIX + _sanitize_unit(argv[i + 1])
        if arg.startswith("--session-id="):
            return _UNIT_PREFIX + _sanitize_unit(arg.split("=", 1)[1])
    return f"{_UNIT_PREFIX}{os.getpid()}"


def _already_scoped() -> bool:
    return bool(os.environ.get("SR2_SCOPE_UNIT"))


def _is_systemd_service() -> bool:
    return bool(os.environ.get("INVOCATION_ID"))


def _opt_out() -> bool:
    return os.environ.get("SR2_NO_SCOPE", "").strip() == "1"


def _systemd_run_available() -> bool:
    return shutil.which("systemd-run") is not None


def _memory_max() -> str:
    return os.environ.get("SR2_SCOPE_MEMORY_MAX", _DEFAULT_MEMORY_MAX)


def _memory_swap_max() -> str:
    return os.environ.get("SR2_SCOPE_MEMORY_SWAP_MAX", _DEFAULT_MEMORY_SWAP_MAX)


def _systemd_user_running() -> bool:
    """Check whether the user systemd instance is reachable."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0 or result.stdout.strip() in (
            b"running",
            b"degraded",
            b"starting",
            b"initializing",
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def maybe_enter_scope(argv: list[str]) -> None:
    """Re-exec the process inside a named systemd scope, or continue unsoped.

    On success this call never returns (execvp replaces the process).
    On any skip condition the function returns and execution continues
    normally.
    """
    if _already_scoped():
        return
    if _is_systemd_service():
        return
    if _opt_out():
        return
    if not _systemd_run_available():
        print(
            "sr2-spectre: systemd-run not found; running without memory scope.",
            file=sys.stderr,
        )
        return
    if not _systemd_user_running():
        print(
            "sr2-spectre: user systemd instance not reachable; running without memory scope.",
            file=sys.stderr,
        )
        return

    unit = _unit_name(argv)
    mem_max = _memory_max()
    swap_max = _memory_swap_max()

    os.environ["SR2_SCOPE_UNIT"] = unit

    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--unit={unit}",
        "-p", f"MemoryMax={mem_max}",
        "-p", f"MemorySwapMax={swap_max}",
        "--",
    ] + sys.argv

    os.execvp("systemd-run", cmd)
