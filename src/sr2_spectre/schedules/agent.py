"""Agent payload executor (spc-187, FR27).

Runs a headless dispatch turn via the ``sr2-spectre dispatch`` CLI,
reads the JSON envelope, and returns a structured result.  The engine
uses this to execute ``agent``-type schedules: on ``done`` the final
message is posted verbatim to the destination (split at 2000 chars);
on any other status a one-line ⚠️ warning is posted instead.

Log placement mirrors the command executor: ``~/.sr2/logs/schedules/<agent>/<YYYY-MM-DD>/``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Discord message hard limit.
MESSAGE_SPLIT_LIMIT = 2000

# Cap for the error line posted to the channel.
ERROR_LINE_CAP = 200

# Valid envelope statuses.
VALID_STATUSES = frozenset({"done", "failed", "needs-input", "escalation", "timeout", "error"})


@dataclass(frozen=True)
class AgentResult:
    """Outcome of one agent dispatch execution."""

    status: str  # done | failed | needs-input | escalation | timeout | error
    message: str  # the envelope's final message (empty on non-done)
    exit_code: int | None
    duration_s: float
    log_path: str

    @property
    def success(self) -> bool:
        return self.status == "done"

    @property
    def error_line(self) -> str:
        """One-line summary for the ⚠️ warning."""
        if self.status == "timeout":
            return "agent dispatch status 'timeout'"
        line = f"agent dispatch returned status '{self.status}'"
        if len(line) > ERROR_LINE_CAP:
            line = line[: ERROR_LINE_CAP - 1] + "…"
        return line


def split_message(text: str, limit: int = MESSAGE_SPLIT_LIMIT) -> list[str]:
    """Split *text* into chunks of at most *limit* characters.

    Breaks at the last newline at or before the limit; never mid-line if
    avoidable.  A single line longer than *limit* is force-split.
    """
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        # Find the last newline at or before the limit
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            # No newline found — force-split at the limit
            cut = limit
        else:
            # Include the newline in the chunk (it's part of the line)
            cut += 1  # include the newline itself
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    return chunks


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


# Type for the injectable dispatch runner (used in tests to avoid real subprocesses).
DispatchRunner = Callable[..., Awaitable[AgentResult]]


async def run_agent_dispatch(
    agent_name: str,
    instruction: str,
    *,
    timeout: float = 600.0,
    agent: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> AgentResult:
    """Execute one agent dispatch turn via the sr2-spectre CLI.

    Args:
        agent_name: The target agent to dispatch to.
        instruction: The prompt/instruction text.
        timeout: Timeout in seconds for the dispatch.
        agent: Agent name for log paths (defaults to agent_name).
        clock: Callable returning current datetime (for log filename).

    Returns:
        AgentResult with status, message, exit_code, duration_s, log_path.
    """
    log_agent = agent or agent_name
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    log_dir = log_dir_for(log_agent, now)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = now.strftime("%H%M%S")
    log_path = log_dir / f"{ts}-agent.log"

    # Write instruction to a temp file
    prompt_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="schedule-prompt-"
    )
    try:
        prompt_file.write(instruction)
        prompt_file.close()

        # Write envelope to a temp file
        out_fd, out_path = tempfile.mkstemp(suffix=".json", prefix="schedule-envelope-")
        os.close(out_fd)  # we just need the path; dispatch writes to it

        try:
            cmd = [
                "sr2-spectre", "dispatch",
                "--agent", agent_name,
                "--prompt-file", prompt_file.name,
                "--timeout", str(timeout),
                "--out", out_path,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout + 30  # grace period beyond dispatch timeout
                )
                proc_exit_code = proc.returncode
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                proc_exit_code = None
                # Write timeout envelope manually
                result = AgentResult(
                    status="timeout",
                    message="",
                    exit_code=None,
                    duration_s=0.0,
                    log_path=str(log_path),
                )
                _write_log(log_path, agent_name, instruction, None, proc_exit_code)
                return result

            # Read the envelope
            envelope = _read_envelope(out_path)
            if envelope is None:
                # Could not parse envelope — treat as error
                stderr_text = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
                result = AgentResult(
                    status="error",
                    message=stderr_text[:ERROR_LINE_CAP] if stderr_text else "",
                    exit_code=proc_exit_code,
                    duration_s=0.0,
                    log_path=str(log_path),
                )
                _write_log(log_path, agent_name, instruction, None, proc_exit_code)
                return result

            status = envelope.get("status", "error")
            if status not in VALID_STATUSES:
                status = "error"
            message = envelope.get("message", "")
            duration_s = envelope.get("duration_s", 0.0)
            env_exit_code = envelope.get("exit_code", proc_exit_code)

            result = AgentResult(
                status=status,
                message=message,
                exit_code=env_exit_code,
                duration_s=duration_s,
                log_path=str(log_path),
            )
            _write_log(log_path, agent_name, instruction, envelope, proc_exit_code)
            return result

        finally:
            # Clean up temp files
            try:
                Path(out_path).unlink(missing_ok=True)
            except OSError:
                pass
    finally:
        try:
            Path(prompt_file.name).unlink(missing_ok=True)
        except OSError:
            pass


def _read_envelope(path: str) -> dict | None:
    """Read and parse the dispatch envelope JSON from *path*."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        if not text.strip():
            return None
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None


def _write_log(
    log_path: Path,
    agent_name: str,
    instruction: str,
    envelope: dict | None,
    proc_exit_code: int | None,
) -> None:
    """Write the agent run log (prompt + envelope)."""
    lines = [
        f"$ sr2-spectre dispatch --agent {agent_name}\n",
        f"\n--- instruction ---\n{instruction}\n",
    ]
    if envelope is not None:
        lines.append(f"\n--- envelope ---\n{json.dumps(envelope, indent=2)}\n")
    else:
        lines.append("\n--- envelope ---\n(unparseable or missing)\n")
    lines.append(f"\n[process exit code: {proc_exit_code}]\n")
    try:
        log_path.write_text("".join(lines), encoding="utf-8")
    except OSError:
        logger.warning("Failed to write agent schedule log: %s", log_path)
