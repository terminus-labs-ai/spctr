"""Glob tool — find files matching a shell-style pattern."""
from __future__ import annotations

import asyncio
import glob as _glob
import os

from sr2_spectre.tools.workspace_floor import WorkspaceFloor


class GlobTool:
    """Find files matching a glob pattern, relative to a root directory.

    Returns matching paths relative to ``path``, sorted ascending and
    newline-joined. Supports recursive ``**`` matching.
    """

    name = "glob"
    description = (
        "Find files matching a shell-style glob pattern. Returns paths "
        "relative to the search directory, sorted ascending. Use '**' for "
        "recursive matching across nested directories."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g. '*.py' or '**/*.md').",
            },
            "path": {
                "type": "string",
                "description": "Root directory to search. Returned paths are relative to it.",
                "default": ".",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace_root: str | None = None) -> None:
        self.floor = WorkspaceFloor(workspace_root)

    async def __call__(self, pattern: str, path: str = ".") -> str:
        # "." means the workspace root under a floor, not the process cwd.
        search_root = self.floor.default_dir(path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _glob_search, pattern, search_root, self.floor
        )


def _glob_search(
    pattern: str, path: str, floor: WorkspaceFloor | None = None
) -> str:
    matches = _glob.glob(pattern, root_dir=path, recursive=True)
    if floor is not None and floor.enforced:
        # A pattern like '../../etc/*' escapes root_dir, so filter the results
        # rather than trusting the search root alone.
        matches = [
            m for m in matches if floor.contains(os.path.join(path, m))
        ]
    if not matches:
        return "No files found."
    return "\n".join(sorted(matches))
