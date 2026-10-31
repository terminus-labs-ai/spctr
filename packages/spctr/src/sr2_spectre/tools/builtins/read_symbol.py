"""Read-symbol tool — extract a full type/function definition from source.

Grep returns matching lines; understanding a class or function needs its full
signature. This tool reads the definition block (class body, function header +
docstring + first code section) so an agent can see required fields, method
signatures, and type annotations before constructing an instance.

Designed to solve the grounding gap observed in bead obsidian-ye0: when an
agent needs to instantiate a Pydantic model or call a constructor, grep of a
single field name only returns that one line, not the full field list.

Handles Python and GDScript. The declaration syntax differs — ``func name():``
and ``class_name Name`` rather than ``def name():`` and ``class Name`` — but
both languages scope blocks by indentation, so only the patterns change. The
language is chosen by file extension.
"""
from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from typing import Literal

from sr2_spectre.tools.workspace_floor import WorkspaceFloor

#: Declarations that occupy a single line, so there is no block to capture.
_SINGLE_LINE_KINDS = {"signal", "variable", "constant"}

GDSCRIPT_EXTENSIONS = {".gd"}


def _gdscript_patterns(symbol_name: str) -> list[tuple[re.Pattern, str]]:
    """Declaration patterns for GDScript, in precedence order."""
    name = re.escape(symbol_name)
    return [
        # `class_name Foo` names the whole file's type; its body is the file.
        (re.compile(r"^(\s*)class_name\s+(" + name + r")\b"), "class_name"),
        (re.compile(r"^(\s*)class\s+(" + name + r")\b"), "class"),
        (re.compile(r"^(\s*)(?:static\s+)?func\s+(" + name + r")\s*\("), "func"),
        (re.compile(r"^(\s*)signal\s+(" + name + r")\b"), "signal"),
        (re.compile(r"^(\s*)enum\s+(" + name + r")\b"), "enum"),
        (re.compile(r"^(\s*)const\s+(" + name + r")\b"), "constant"),
        (
            re.compile(
                r"^(\s*)(?:@export[^\s]*\s+)?(?:static\s+)?var\s+(" + name + r")\b"
            ),
            "variable",
        ),
    ]


def _find_gdscript_symbol(
    lines: list[str], symbol_name: str, file_path: str
) -> tuple[int, int, str]:
    """Locate a GDScript declaration. Returns (line index, indent, kind)."""
    for idx, line in enumerate(lines):
        for pattern, kind in _gdscript_patterns(symbol_name):
            m = pattern.match(line)
            if not m:
                continue
            indent = len(m.group(1))
            if kind == "func":
                return idx, indent, "method" if indent > 0 else "function"
            if kind == "class_name":
                return idx, indent, "class"
            return idx, indent, kind
    raise ValueError(
        f"Symbol '{symbol_name}' not found in {file_path} (searched for a "
        f"GDScript func, class, class_name, signal, enum, const or var)"
    )


@dataclass(frozen=True)
class SymbolInfo:
    """Structured result of a symbol lookup."""
    name: str
    kind: Literal[
        "class", "function", "method", "signal", "variable", "constant", "enum"
    ]
    start_line: int          # 1-based
    end_line: int            # 1-based (inclusive)
    body: str                # raw source of the definition
    file_path: str


def find_symbol(
    file_path: str,
    symbol_name: str,
    context_lines: int = 0,
) -> SymbolInfo:
    """Find a class or function definition in a Python source file.

    Scans for ``class SymbolName`` or ``def symbol_name(`` at the correct
    indentation level. For classes, captures the entire class body (all lines
    at the class indent or deeper). For functions, captures the signature +
    docstring + body.

    Args:
        file_path: Path to the Python source file.
        symbol_name: The class or function name to find (exact match).
        context_lines: Number of blank/comment lines of context to include
            after the definition ends (default 0).

    Returns:
        SymbolInfo with the raw source text of the definition.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If the symbol is not found in the file, or the file is
            not valid Python (empty, or the symbol matches nothing recognizable
            as a class/function definition).
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read file: {file_path}") from exc

    if os.path.splitext(file_path)[1].lower() in GDSCRIPT_EXTENSIONS:
        match_line, match_indent, gd_kind = _find_gdscript_symbol(
            lines, symbol_name, file_path
        )
        if gd_kind in _SINGLE_LINE_KINDS:
            end_line = match_line
        elif _is_file_scoped_class(lines, match_line, symbol_name):
            # `class_name Foo` types the whole file, so the file is the body.
            end_line = len(lines) - 1
        else:
            end_line = _find_definition_end(
                lines, match_line, match_indent,
                "class" if gd_kind in ("class", "enum") else gd_kind,
            )
        body_lines = lines[match_line : end_line + 1]
        if context_lines > 0:
            body_lines += lines[end_line + 1 : end_line + 1 + context_lines]
        return SymbolInfo(
            name=symbol_name,
            kind=gd_kind,
            start_line=match_line + 1,
            end_line=end_line + 1,
            body="\n".join(body_lines),
            file_path=file_path,
        )

    # Class definitions: 'class Name' or 'class Name(Base)'
    class_pattern = re.compile(
        r"^(\s*)class\s+(" + re.escape(symbol_name) + r")\b"
    )
    # Function/method definitions: 'def name('
    func_pattern = re.compile(
        r"^(\s*)def\s+(" + re.escape(symbol_name) + r")\s*\("
    )

    match_line: int | None = None
    match_indent: int = 0
    match_kind: Literal["class", "function", "method"] = "function"

    for idx, line in enumerate(lines):
        # Check class first (higher precedence for exact match)
        cm = class_pattern.match(line)
        if cm:
            match_line = idx
            match_indent = len(cm.group(1))
            match_kind = "class"
            break

        # Then function/method
        fm = func_pattern.match(line)
        if fm:
            match_line = idx
            indent = len(fm.group(1))
            match_indent = indent
            match_kind = "method" if indent > 0 else "function"
            break

    if match_line is None:
        raise ValueError(
            f"Symbol '{symbol_name}' not found in {file_path} "
            f"(searched for class/function definition)"
        )

    # Calculate the end of the definition block
    end_line = _find_definition_end(lines, match_line, match_indent, match_kind)

    body_lines = lines[match_line : end_line + 1]

    # Add context lines if requested
    if context_lines > 0:
        body_lines = body_lines + lines[end_line + 1 : end_line + 1 + context_lines]

    return SymbolInfo(
        name=symbol_name,
        kind=match_kind,
        start_line=match_line + 1,  # 1-based
        end_line=end_line + 1,      # 1-based
        body="\n".join(body_lines),
        file_path=file_path,
    )


def _is_file_scoped_class(lines: list[str], idx: int, symbol_name: str) -> bool:
    """True when the matched line is a `class_name` declaration."""
    return bool(
        re.match(r"^\s*class_name\s+" + re.escape(symbol_name) + r"\b", lines[idx])
    )


def _find_definition_end(
    lines: list[str],
    start: int,
    base_indent: int,
    kind: Literal[
        "class", "function", "method", "signal", "variable", "constant", "enum"
    ],
) -> int:
    """Find the last line of a class/function definition.

    A definition ends when we encounter a non-blank, non-comment line at an
    indent level <= the base indent (for classes/methods) or a line that's
    structurally at the same level (for top-level functions).

    Args:
        lines: All lines of the file.
        start: 0-based index of the definition start line.
        base_indent: Indentation of the def/class header.
        kind: Whether this is a class, function, or method.

    Returns:
        0-based index of the last line belonging to the definition.
    """
    expected_child_indent = base_indent + 4  # standard Python convention

    # For classes: body is everything indented deeper than the class header
    if kind == "class":
        last_content = start
        for i in range(start + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()

            # Blank lines and comments are part of the definition
            if not stripped or stripped.startswith("#"):
                continue

            # Calculate current indent
            current_indent = len(line) - len(line.lstrip())

            # If we hit something at the same or lower indent as the class
            # header, the class body has ended
            if current_indent <= base_indent:
                break

            last_content = i
        return last_content

    # For functions/methods: body is indented deeper than the def header
    # We include the signature, docstring, and body until we hit something
    # at the same or lower indent
    last_content = start
    in_docstring = False
    docstring_char = None

    for i in range(start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()

        # Handle docstrings — they're part of the function body
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                # Check if it's a single-line docstring
                if stripped.count(docstring_char) >= 2 and len(stripped) > 3:
                    # Single-line docstring like """text"""
                    last_content = i
                    continue
                else:
                    in_docstring = True
                    last_content = i
                    continue
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
            last_content = i
            continue

        # Blank lines are part of the body
        if not stripped:
            last_content = i
            continue

        # Comments are part of the body
        if stripped.startswith("#"):
            last_content = i
            continue

        # Calculate current indent
        current_indent = len(line) - len(line.lstrip())

        # If we hit something at the same or lower indent, the body has ended
        if current_indent <= base_indent:
            break

        last_content = i

    return last_content


# ---------------------------------------------------------------------------
# Tool interface (MCP-compatible)
# ---------------------------------------------------------------------------

class ReadSymbolTool:
    """Extract a full class or function definition from a source file.

    Unlike grep (which returns individual matching lines), this tool returns
    the complete definition block: class body with all fields, or function
    signature with docstring and body. Use this when you need to understand
    a type's required fields before constructing an instance, or a function's
    parameters before calling it.
    """

    name = "read_symbol"
    description = (
        "Read the full definition of a class or function from a Python or "
        "GDScript source file. Returns the complete definition block "
        "including all fields, "
        "type annotations, and docstrings. Use this before constructing "
        "instances or calling functions to see required parameters and field "
        "defaults. Unlike grep (which returns individual matching lines), this "
        "returns the entire definition so you can see the full structure."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "Absolute or relative path to the source file containing "
                    "the symbol (.py or .gd)."
                ),
            },
            "symbol_name": {
                "type": "string",
                "description": (
                    "Exact name of the symbol to look up (e.g. "
                    "'McpServerConfig', 'load_config', '_ready')."
                ),
            },
            "context_lines": {
                "type": "integer",
                "description": (
                    "Optional: number of context lines to include after the "
                    "definition ends. Useful to see what follows. Defaults to 0."
                ),
                "default": 0,
            },
        },
        "required": ["file_path", "symbol_name"],
    }

    def __init__(self, workspace_root: str | None = None) -> None:
        self.floor = WorkspaceFloor(workspace_root)

    async def __call__(
        self,
        file_path: str,
        symbol_name: str,
        context_lines: int = 0,
    ) -> str:
        effective = self.floor.checked(file_path)
        info = find_symbol(effective, symbol_name, context_lines=context_lines)

        result_lines = [
            f"Symbol: {info.name} ({info.kind})",
            f"File: {info.file_path}",
            f"Lines: {info.start_line}-{info.end_line}",
            "---",
            info.body,
            "---",
        ]
        return "\n".join(result_lines)
