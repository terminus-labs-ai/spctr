"""File read tool — read a file from disk."""
from __future__ import annotations

import asyncio
import os

from sr2_spectre.tools.workspace_floor import WorkspaceFloor


class FileReadTool:
    """Read the contents of a file from the local filesystem.

    When *workspace_root* is set, paths resolving outside the workspace are
    rejected. Reads used to be unconfined even with SR2_WORKSPACE set, which
    made the workspace floor a write-only guarantee.
    """

    name = "file_read"
    description = "Read the contents of a file from disk and return them as a string."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self, max_bytes: int = 1_000_000, workspace_root: str | None = None
    ) -> None:
        self.max_bytes = max_bytes
        self.floor = WorkspaceFloor(workspace_root)

    async def __call__(self, path: str) -> str:
        effective = self.floor.checked(path)

        if not os.path.exists(effective):
            raise FileNotFoundError(f"No such file: {effective}")

        size = os.path.getsize(effective)
        if size > self.max_bytes:
            raise ValueError(
                f"File too large: {effective} is {size} bytes (limit: {self.max_bytes})"
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read_file, effective)


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()
