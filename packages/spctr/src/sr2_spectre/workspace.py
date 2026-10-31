"""Workspace confinement — resolve workspace root and enforce boundaries.

FR1: The agent runtime takes a workspace root from SR2_WORKSPACE env var.
When unset, falls back to os.getcwd(). The resolved root is absolute and
canonicalized via os.path.realpath.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_workspace_root() -> Path:
    """Resolve the workspace root from environment.

    Priority:
    1. ``SR2_WORKSPACE`` env var — set by harbinger for worktree isolation.
    2. ``os.getcwd()`` — standalone behavior (preserves current cwd as root).

    Returns an absolute, canonicalized Path.
    """
    raw = os.environ.get("SR2_WORKSPACE")
    if raw:
        return Path(raw).resolve()
    return Path.cwd().resolve()


def is_within_workspace(path: str, workspace_root: Path) -> bool:
    """Check if a resolved path lies within the workspace root.

    Resolves the path (handling relative paths against the workspace root,
    and applying realpath to canonicalize symlinks and .. traversal).

    Returns True if the resolved path starts with the workspace root.
    """
    resolved = Path(path).resolve()
    root = workspace_root.resolve()

    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False
