"""Launch-directory area derivation for interfaces that know a directory.

Shared by the REPL interface (which stamps ``RunContext.area`` at startup)
and by ``PlanResolver`` (which delegates its cwd/.git fallback walk here
instead of duplicating it).

Derivation order, first hit wins:
1. ``SR2_AREA`` env var — explicit override; an empty value means "no area"
   and is authoritative (no fallback).
2. Nearest ancestor (cwd first) containing a ``CLAUDE.md`` — the area doc.
   Checked before ``.git`` because a vault area such as
   ``/data/obsidian/projects/<area>`` is a folder *inside* the vault git
   repo: the git root would name the vault, not the area.
3. Nearest ancestor containing ``.git`` — the repo name.
4. The cwd basename — logged with a single WARNING.

The result is a name (or `""`), never a path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_git_marker(path: Path) -> bool:
    marker = path / ".git"
    return marker.is_dir() or marker.is_file()


def _walk_up_from(start: Path) -> tuple[str, str]:
    """Walk from *start* (inclusive) toward the filesystem root.

    Returns ``(name, rule)`` where rule is ``"claude_md"``, ``"git"``, or
    ``"cwd_basename"`` for the fallback.
    """
    current = start
    while True:
        if (current / "CLAUDE.md").is_file():
            return current.name, "claude_md"
        if _has_git_marker(current):
            return current.name, "git"
        parent = current.parent
        if parent == current:
            break
        current = parent
    return start.name, "cwd_basename"


def derive_area(start: str | Path | None = None) -> str:
    """Derive the area name for this process from the launch directory.

    Args:
        start: Directory to derive from. Defaults to the current working
            directory (resolved).

    Returns:
        The area name, or ``""`` for an explicit no-area (empty ``SR2_AREA``).
    """
    env_area = os.environ.get("SR2_AREA")
    if env_area is not None:
        return env_area

    if start is None:
        try:
            start = Path.cwd().resolve()
        except OSError:
            start = Path("/tmp")
    else:
        start = Path(start)
        if not start.is_absolute():
            try:
                start = start.resolve()
            except OSError:
                start = Path("/tmp")

    name, rule = _walk_up_from(start)
    if rule == "cwd_basename":
        logger.warning(
            "Area derivation: no CLAUDE.md or .git found walking up from %s "
            "— using cwd name %r as area fallback.",
            start,
            name,
        )
    return name
