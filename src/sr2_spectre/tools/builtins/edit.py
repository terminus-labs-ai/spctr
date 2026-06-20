"""File edit tool — exact substring replacement in a file on disk."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path


class EditTool:
    """Replace an exact substring in a file.

    Replaces a unique occurrence of ``old_string`` with ``new_string``. When
    ``replace_all`` is true, every occurrence is replaced. A zero-match or an
    ambiguous (multi-match without ``replace_all``) request raises ``ValueError``
    and leaves the file untouched.

    When *workspace_root* is set, paths resolving outside the workspace
    are rejected with ValueError (FR3 — workspace floor).
    """

    name = "edit"
    description = (
        "Replace an exact substring in a file. Requires a unique match unless "
        "replace_all is set, in which case all occurrences are replaced."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact substring to find and replace.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text for old_string.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match.",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, workspace_root: str | None = None) -> None:
        """Initialize the edit tool.

        Args:
            workspace_root: When set, paths resolving outside this root
                are rejected with ValueError. When None, no enforcement
                (back-compat for standalone runs without SR2_WORKSPACE).
        """
        if workspace_root is not None:
            self.workspace_root: Path | None = Path(workspace_root).resolve()
        else:
            self.workspace_root = None

    async def __call__(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        self._check_path(path)

        # Resolve the effective path: relative paths resolve against
        # the workspace root, not raw process cwd.
        effective_path = self._resolve_path(path)

        if not os.path.exists(effective_path):
            raise FileNotFoundError(f"No such file: {effective_path}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _edit_file, effective_path, old_string, new_string, replace_all
        )

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
                f"All file edits must be within the workspace root."
            )


def _edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> str:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    count = content.count(old_string)
    if count == 0:
        raise ValueError(f"old_string not found in {path}: {old_string!r}")
    if count > 1 and not replace_all:
        raise ValueError(
            f"Ambiguous edit: old_string matches {count} times in {path}. "
            "Set replace_all=True or provide a more specific old_string."
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
        n_replaced = count
    else:
        new_content = content.replace(old_string, new_string, 1)
        n_replaced = 1

    encoded = new_content.encode("utf-8")
    with open(path, "wb") as f:
        f.write(encoded)

    return f"Made {n_replaced} replacement(s) in {path}"
