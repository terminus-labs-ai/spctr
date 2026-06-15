"""File write tool — write content to a file on disk."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path


class FileWriteTool:
    """Write content to a file, creating parent directories as needed.

    When *workspace_root* is set, paths resolving outside the workspace
    are rejected with ValueError (FR3 — workspace floor).
    """

    name = "file_write"
    description = "Write a string to a file on disk. Creates missing parent directories."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace_root: str | None = None) -> None:
        """Initialize the file write tool.

        Args:
            workspace_root: When set, paths resolving outside this root
                are rejected with ValueError. When None, no enforcement
                (back-compat for standalone runs without SR2_WORKSPACE).
        """
        if workspace_root is not None:
            self.workspace_root: Path | None = Path(workspace_root).resolve()
        else:
            self.workspace_root = None

    async def __call__(self, path: str, content: str) -> str:
        self._check_path(path)

        # Resolve the effective path: relative paths resolve against
        # the workspace root, not raw process cwd.
        effective_path = self._resolve_path(path)

        loop = asyncio.get_event_loop()
        n_bytes = await loop.run_in_executor(None, _write_file, effective_path, content)
        return f"Written {n_bytes} bytes to {effective_path}"

    def _resolve_path(self, path: str) -> str:
        """Resolve a path for use within the workspace.

        Relative paths are resolved against the workspace root (not cwd).
        Absolute paths are used as-is.
        """
        p = Path(path)
        if self.workspace_root is not None and not p.is_absolute():
            return str(self.workspace_root / p)
        return path

    def _check_path(self, path: str) -> None:
        """Reject paths resolving outside the workspace root.

        Relative paths are resolved against the workspace root (not cwd).
        When workspace_root is None (not configured), skip enforcement.
        """
        if self.workspace_root is None:
            return

        resolved = Path(self._resolve_path(path)).resolve()

        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise ValueError(
                f"Path {resolved} is outside workspace {self.workspace_root}. "
                f"All file writes must be within the workspace root."
            )


def _write_file(path: str, content: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    encoded = content.encode("utf-8")
    with open(path, "wb") as f:
        f.write(encoded)
    return len(encoded)
