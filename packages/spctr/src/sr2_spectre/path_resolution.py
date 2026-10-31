"""Unified path resolution utility for Spectre config files (FR10).

Implements resolution rules in order:
1. ${VAR} interpolation — unset vars are a startup error.
2. Absolute paths (after interpolation) — used as-is.
3. Relative paths — resolved against the declaring file's own directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_VAR_RE = re.compile(r"\$\{([^}]+)\}")


class ConfigPathError(Exception):
    """Raised when a path value in a config file cannot be resolved."""


def resolve_path(
    raw: str,
    declaring_file: Path,
    env: dict[str, str] | None = None,
) -> Path:
    """Resolve a path value from a config file.

    Args:
        raw: The raw path string (may contain ${VAR} tokens).
        declaring_file: The config file in which this path appears.
                        Relative paths resolve against its parent directory.
        env: Environment variables for interpolation. Defaults to os.environ.
             Always includes SR2_HOME (caller must ensure this or pass it).

    Returns:
        Resolved absolute Path.

    Raises:
        ConfigPathError: If a ${VAR} token references an unset env var.
    """
    if env is None:
        env = dict(os.environ)

    # Step 1: interpolate all ${VAR} tokens.
    def _replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in env:
            raise ConfigPathError(
                "Unresolved environment variable '${" + name + "}' in path "
                + repr(raw) + ". "
                "Set the $" + name + " environment variable before starting Spectre."
            )
        return env[name]

    interpolated = _VAR_RE.sub(_replace, raw)

    # Step 2: absolute vs relative.
    p = Path(interpolated)
    if p.is_absolute():
        return p.resolve()

    # Step 3: relative — anchor to declaring file's parent directory.
    return (declaring_file.parent / p).resolve()
