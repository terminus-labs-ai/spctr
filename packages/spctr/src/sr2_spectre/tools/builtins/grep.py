"""Grep tool — search file contents for a pattern, line by line."""
from __future__ import annotations

import asyncio
import fnmatch
import os
import re

from sr2_spectre.tools.workspace_floor import WorkspaceFloor

# Directories that are pruned from recursive walks by default. These are
# never useful to grep and (e.g. a 221MB .venv) can blow up output.
_DEFAULT_IGNORE_DIRS = {
    ".venv",
    ".git",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}


class GrepTool:
    """Search files for lines matching a pattern.

    Searches a single file or recursively walks a directory, emitting each
    matching line as ``{filepath}:{lineno}:{linetext}`` (1-based line numbers).
    Supports regex or literal matching and an optional basename glob filter.
    Binary files (undecodable or containing a NUL byte) are skipped silently.

    Output is bounded to keep an agent from being flooded:

    * recursive walks prune ``ignore_dirs`` (merged with the built-in defaults);
    * any matching line longer than ``max_line_length`` is truncated;
    * at most ``max_matches`` matching lines are emitted, with a suppression
      notice appended when there are more.
    """

    name = "grep"
    description = (
        "Search file contents for a pattern. Accepts a single file or a "
        "directory (searched recursively). Supports regex or literal matching "
        "and an optional glob to scope which files are searched."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Pattern to search for in each line.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search. Directories are searched recursively.",
                "default": ".",
            },
            "glob": {
                "type": "string",
                "description": "Only search files whose basename matches this glob (e.g. '*.py').",
            },
            "regex": {
                "type": "boolean",
                "description": "Treat pattern as a regular expression (default true). If false, match literally.",
                "default": True,
            },
        },
        "required": ["pattern"],
    }

    def __init__(
        self,
        max_line_length: int = 500,
        max_matches: int = 100,
        ignore_dirs: set[str] | None = None,
        workspace_root: str | None = None,
    ) -> None:
        self.max_line_length = max_line_length
        self.max_matches = max_matches
        # Merge with defaults — never replace. Defaults always apply.
        self.ignore_dirs = _DEFAULT_IGNORE_DIRS | (ignore_dirs or set())
        self.floor = WorkspaceFloor(workspace_root)

    async def __call__(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        regex: bool = True,
        ignore_dirs: set[str] | None = None,
    ) -> str:
        # A per-call ignore set merges on top of the instance set (which itself
        # already includes the defaults). Merge, never replace.
        effective_ignore = self.ignore_dirs | (ignore_dirs or set())
        # "." means the workspace root under a floor, not the process cwd.
        search_root = self.floor.default_dir(path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _grep,
            pattern,
            search_root,
            glob,
            regex,
            self.max_line_length,
            self.max_matches,
            effective_ignore,
            self.floor,
        )


def _grep(
    pattern: str,
    path: str,
    glob: str | None,
    regex: bool,
    max_line_length: int,
    max_matches: int,
    ignore_dirs: set[str],
    floor: WorkspaceFloor | None = None,
) -> str:
    if regex:
        compiled = re.compile(pattern)
    else:
        compiled = re.compile(re.escape(pattern))

    if os.path.isdir(path):
        files = []
        for dirpath, dirnames, filenames in os.walk(path):
            # Prune ignored directories in place so os.walk won't descend.
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
            for filename in filenames:
                if glob is not None and not fnmatch.fnmatch(filename, glob):
                    continue
                candidate = os.path.join(dirpath, filename)
                # A symlinked file inside the tree can still point outside it.
                if floor is not None and not floor.contains(candidate):
                    continue
                files.append(candidate)
    else:
        if glob is not None and not fnmatch.fnmatch(os.path.basename(path), glob):
            files = []
        else:
            files = [path]

    matches: list[str] = []
    total_found = 0
    capped = False
    for filepath in files:
        try:
            with open(filepath, "rb") as fb:
                raw = fb.read()
        except OSError:
            continue

        # Robust binary skip: a NUL byte means binary even if UTF-8-decodable.
        if b"\x00" in raw:
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue

        for lineno, line in enumerate(content.splitlines(), start=1):
            if compiled.search(line):
                total_found += 1
                if len(matches) < max_matches:
                    text = _truncate_line(line, max_line_length)
                    matches.append(f"{filepath}:{lineno}:{text}")
                else:
                    capped = True

    if not matches:
        return "No matches found."

    output = "\n".join(matches)
    if capped:
        omitted = total_found - len(matches)
        output += (
            f"\n[output limited to {len(matches)} matches; "
            f"{omitted} more match(es) omitted]"
        )
    return output


def _truncate_line(line: str, max_line_length: int) -> str:
    if len(line) <= max_line_length:
        return line
    return line[:max_line_length] + "... [truncated]"
