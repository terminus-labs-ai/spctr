"""Terminal tool — run shell commands as a subprocess."""
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from sr2_spectre.session_identity import current_session_id


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the command's whole process group (spawns run in their own
    session, so its PGID is the shell's PID). Killing only the shell would
    orphan children (e.g. the ``sleep`` in ``sh -c 'sleep 30'``), which can
    outlive a timed-out or cancelled turn. Falls back to killing just the
    shell if the group is already gone."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


class TerminalTool:
    """Execute shell commands and return combined stdout+stderr.

    When *workspace_root* is set, the subprocess runs with ``cwd`` set to
    the workspace root (FR4 — workspace floor).
    """

    name = "terminal"
    description = "Run a shell command and return its output (stdout and stderr combined)."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
        },
        "required": ["command"],
    }

    def __init__(
        self, timeout: int = 30, workspace_root: str | None = None
    ) -> None:
        """Initialize the terminal tool.

        Args:
            timeout: Maximum execution time in seconds.
            workspace_root: When set, commands run with cwd set to this
                directory. When None, cwd is inherited (back-compat).
        """
        self.timeout = timeout
        if workspace_root is not None:
            self.workspace_root = str(Path(workspace_root).resolve())
        else:
            self.workspace_root = None

    async def __call__(self, command: str) -> str:
        kwargs: dict = {}
        if self.workspace_root is not None:
            kwargs["cwd"] = self.workspace_root
        # spc-137: during a session turn, name the session to the command
        # (interaction-signal capture reads $SPECTRE_SESSION_ID, mirroring
        # ${CLAUDE_CODE_SESSION_ID:--} / ${CODEX_THREAD_ID:--}). When unset
        # (outside a turn) pass no env at all: byte-identical inherited
        # behavior.
        session_id = current_session_id.get()
        if session_id is not None:
            kwargs["env"] = {**os.environ, "SPECTRE_SESSION_ID": session_id}

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Own session/process group: on timeout or cancellation we kill
            # the group so children cannot outlive the command.
            start_new_session=True,
            **kwargs,
        )

        # Pump stdout/stderr into buffers concurrently rather than using
        # communicate(). communicate() is all-or-nothing: on timeout its
        # buffered reads are discarded, so a killed command (e.g. a slow
        # pytest) returns nothing and a small model treats the empty result as
        # a dead end and silently ends its turn. By accumulating into buffers
        # as the process runs, output already produced survives the kill.
        stdout_buf = bytearray()
        stderr_buf = bytearray()

        async def _pump(reader: asyncio.StreamReader | None, buf: bytearray) -> None:
            if reader is None:
                return
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                buf.extend(chunk)

        pumps = asyncio.gather(
            _pump(proc.stdout, stdout_buf),
            _pump(proc.stderr, stderr_buf),
        )
        try:
            await asyncio.wait_for(pumps, timeout=self.timeout)
            await proc.wait()
        except asyncio.CancelledError:
            # A cancel from outside (dispatch turn timeout, REPL Ctrl-C)
            # arrives here, not via the timeout path above: the shell and
            # its children must not survive the turn. Kill the group, then
            # let the cancellation propagate unchanged.
            _kill_process_group(proc)
            pumps.cancel()
            raise
        except (asyncio.TimeoutError, TimeoutError):
            _kill_process_group(proc)
            # Cancel the pumps and reap them; whatever was read before the
            # timeout is already in the buffers.
            pumps.cancel()
            try:
                await pumps
            except (asyncio.CancelledError, Exception):
                pass
            # Best-effort reap so the transport is cleaned up. Bounded: a
            # killed shell can leave an orphaned child (e.g. `sleep`) holding
            # the pipes open, which makes an unbounded wait() hang. We don't
            # need the exit status, so cap the wait and move on.
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except Exception:
                pass
            partial = (
                stdout_buf.decode(errors="replace")
                + stderr_buf.decode(errors="replace")
            )
            msg = f"Command timed out after {self.timeout}s: {command}"
            if partial:
                msg += f"\n\nPartial output before timeout:\n{partial}"
            raise TimeoutError(msg)

        return stdout_buf.decode(errors="replace") + stderr_buf.decode(errors="replace")
