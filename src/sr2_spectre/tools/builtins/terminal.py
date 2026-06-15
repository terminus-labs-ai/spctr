"""Terminal tool — run shell commands as a subprocess."""
from __future__ import annotations

import asyncio
from pathlib import Path


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

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        try:
            communicate_coro = proc.communicate()
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                communicate_coro, timeout=self.timeout
            )
        except (asyncio.TimeoutError, TimeoutError):
            proc.kill()
            raise TimeoutError(f"Command timed out: {command}")

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
        return stdout + stderr
