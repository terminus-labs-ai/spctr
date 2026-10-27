"""Command payload executor (spc-185, FR26).

Runs a shell command with captured stdout/stderr to a log file under
``~/.sr2/logs/schedules/<agent>/<YYYY-MM-DD>/``, with a timeout kill.
Returns a result indicating success, failure, or timeout.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Cap for the error line posted to the channel (NFR Isolation).
ERROR_LINE_CAP = 200


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one command execution."""

    exit_code: int | None  # None on timeout
    timed_out: bool
    stdout: str
    stderr: str
    log_path: str

    @property
    def success(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    @property
    def error_line(self) -> str:
        """First stderr line or timeout note, capped at ERROR_LINE_CAP."""
        if self.timed_out:
            return "command timed out"
        first_line = self.stderr.strip().split("\n")[0] if self.stderr.strip() else ""
        if not first_line:
            first_line = f"exit code {self.exit_code}"
        if len(first_line) > ERROR_LINE_CAP:
            first_line = first_line[:ERROR_LINE_CAP - 1] + "…"
        return first_line


def log_dir_for(agent: str, now: datetime | None = None) -> Path:
    """Log directory: ``$SR2_HOME/logs/schedules/<agent>/<YYYY-MM-DD>/``

    SR2_HOME defaults to ``~/.sr2`` (spc-1004). Imported lazily:
    ``sr2_spectre.config`` reaches the discord interface, which imports
    ``sr2_spectre.schedules`` — a module-level import here would be circular.
    """
    from sr2_spectre.config import resolve_sr2_home

    if now is None:
        now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    return resolve_sr2_home() / "logs" / "schedules" / agent / date_str


async def run_command(
    command: str,
    *,
    agent: str,
    cwd: str | None = None,
    timeout: float = 600.0,
    clock: callable | None = None,
) -> CommandResult:
    """Execute a shell command with timeout and captured output.

    Args:
        command: Shell command string.
        agent: Agent name for the log path.
        cwd: Working directory (default: process cwd).
        timeout: Timeout in seconds.
        clock: Callable returning current datetime (for log filename).

    Returns:
        CommandResult with exit_code, timed_out, stdout, stderr, log_path.
    """
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    log_dir = log_dir_for(agent, now)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Unique log filename based on timestamp
    ts = now.strftime("%H%M%S")
    log_path = log_dir / f"{ts}.log"

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            exit_code = proc.returncode
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout_bytes = b""
            stderr_bytes = b""
            exit_code = None
            timed_out = True
    except Exception as exc:
        # Subprocess creation failed
        stdout_bytes = b""
        stderr_bytes = str(exc).encode()
        exit_code = -1
        timed_out = False

    stdout_text = stdout_bytes.decode(errors="replace")
    stderr_text = stderr_bytes.decode(errors="replace")

    # Write log file
    log_content = f"$ {command}\n"
    if timed_out:
        log_content += f"[TIMED OUT after {timeout}s]\n"
    else:
        log_content += f"[exit code: {exit_code}]\n"
    log_content += f"\n--- stdout ---\n{stdout_text}\n--- stderr ---\n{stderr_text}\n"
    log_path.write_text(log_content, encoding="utf-8")

    return CommandResult(
        exit_code=exit_code,
        timed_out=timed_out,
        stdout=stdout_text,
        stderr=stderr_text,
        log_path=str(log_path),
    )
