"""Workspace floor — shared path confinement for tools that touch the disk.

``SR2_WORKSPACE`` is meant to be a floor under everything a tool can reach, but
until now only the writing tools enforced it (FR3/FR4). Reads were unconfined,
so an agent with ``file_read`` could reach anything the process could, workspace
or not — which matters the moment an agent runs somewhere its operator does not
fully trust it, as the Grindforge Discord bot does.

The contract, matching the one file_write.py established:

* relative paths resolve against the workspace root, not the process cwd;
* any path resolving outside the root is rejected with ValueError;
* ``workspace_root=None`` disables enforcement, preserving standalone
  behaviour for callers that never set SR2_WORKSPACE.
"""
from __future__ import annotations

from pathlib import Path


class WorkspaceFloor:
    """Resolves and validates paths against an optional workspace root."""

    def __init__(self, workspace_root: str | None = None) -> None:
        if workspace_root is not None:
            self.workspace_root: Path | None = Path(workspace_root).expanduser().resolve()
        else:
            self.workspace_root = None

    @property
    def enforced(self) -> bool:
        return self.workspace_root is not None

    def resolve(self, path: str) -> str:
        """Resolve *path* for use, anchoring relative paths to the root."""
        if self.workspace_root is None:
            return path
        p = Path(path).expanduser()
        if not p.is_absolute():
            return str(self.workspace_root / p)
        return str(p)

    def check(self, path: str) -> None:
        """Raise ValueError if *path* resolves outside the workspace root.

        Resolution follows symlinks, so a link planted inside the workspace
        cannot be used to step outside it.
        """
        if self.workspace_root is None:
            return
        resolved = Path(self.resolve(path)).resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise ValueError(
                f"Path {resolved} is outside workspace {self.workspace_root}. "
                f"This tool may only reach paths within the workspace root."
            ) from None

    def checked(self, path: str) -> str:
        """Resolve *path* and confirm it is inside the root. Returns the path."""
        self.check(path)
        return self.resolve(path)

    def contains(self, path: str) -> bool:
        """Return True if *path* is within the root (always True when off)."""
        if self.workspace_root is None:
            return True
        try:
            Path(path).resolve().relative_to(self.workspace_root)
            return True
        except ValueError:
            return False

    def default_dir(self, path: str) -> str:
        """Resolve a directory argument whose default is the cwd (``.``).

        Under a floor, "." means the workspace root rather than wherever the
        process happens to be.
        """
        if self.workspace_root is not None and path == ".":
            return str(self.workspace_root)
        return self.checked(path)
