"""Frontmatter parsing for plan, task, and project-knowledge markdown files.

Extracts the YAML block between the opening `---` and the next `---` from a
markdown file and parses it into the appropriate frozen dataclass per kind.

Tolerance guarantees (FR5):
  - Files lacking a recognized `kind` are skipped (logged).
  - Files that fail to parse (bad YAML, missing fields) are skipped (logged).
  - Never crashes resolve — returns None on any failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from sr2_spectre.planning.models import (
    KnowledgeFrontmatter,
    PlanFileFrontmatter,
    PlanFrontmatter,
    PlanStatus,
    RECOGNIZED_KINDS,
    TaskFrontmatter,
    TaskStatus,
    get_frontmatter_class,
)

logger = logging.getLogger(__name__)


def split_frontmatter(text: str) -> Optional[tuple[str, str]]:
    """Split text into (frontmatter_block, body) at the frontmatter boundary.

    Scans for a YAML frontmatter block delimited by opening ``---`` and
    closing ``---`` (on their own line).  Handles leading whitespace
    gracefully by stripping before detection.

    Returns:
        ``(frontmatter_block, body)`` where *frontmatter_block* includes the
        opening ``---``, YAML content, closing ``---``, and the trailing
        newline after the closing delimiter.  *body* is everything after
        that newline.  Returns ``None`` if no valid frontmatter block is
        found.

    Examples:
        >>> split_frontmatter("---\\nkind: task\\n---\\n# Body\\n")
        ('---\\nkind: task\\n---\\n', '# Body\\n')

        >>> split_frontmatter("# No frontmatter\\n") is None
        True
    """
    stripped = text.strip()
    if not stripped.startswith("---"):
        return None

    rest = stripped[3:]  # skip opening ---

    try:
        idx = rest.index("\n---")
    except ValueError:
        return None

    # frontmatter_block: opening --- through closing --- (inclusive) + trailing \n
    # idx is position of "\n---" in rest (rest = stripped[3:]).
    # fm_block in stripped: positions 0..3+idx+3 = idx+6 (covers "---" + YAML content + "\n---")
    fm_end = 3 + idx + 4  # 3 (opening ---) + idx (to \n of closing) + 4 ("\n---")
    fm_block = stripped[:fm_end]
    body = stripped[fm_end:]

    # If body starts with \n after the closing ---, that's the paragraph break.
    # We return the body as-is (with or without that leading newline).
    return fm_block, body


def extract_raw_frontmatter(text: str) -> Optional[str]:
    """Extract the raw YAML string between the first `---` and next `---`.

    Returns None if no valid frontmatter block is found.

    .. note::
       This is a thin wrapper around :func:`split_frontmatter` that
       extracts only the YAML content between delimiters.
    """
    result = split_frontmatter(text)
    if result is None:
        return None

    fm_block, _body = result
    # fm_block is "---\n<yaml>\n---\n" — strip delimiters to get raw YAML.
    inner = fm_block[3:]  # remove leading "---"
    inner = inner.rstrip("\n---")  # remove trailing "\n---\n"
    raw = inner.strip()
    return raw if raw else None


def parse_frontmatter(
    text: str,
    file_path: Optional[Path] = None,
) -> Optional[PlanFileFrontmatter]:
    """Parse frontmatter from a markdown string into a frozen dataclass.

    Args:
        text: Full markdown file content.
        file_path: Optional path for logging context.

    Returns:
        A frozen frontmatter dataclass for the recognized kind, or None if
        the file has no frontmatter, an unrecognized kind, or fails to parse.
    """
    label = f"{file_path}" if file_path else "inline text"

    raw = extract_raw_frontmatter(text)
    if raw is None:
        logger.debug("No frontmatter found in %s — skipping", label)
        return None

    # Parse YAML
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("YAML parse error in %s: %s — skipping", label, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Frontmatter in %s is not a mapping — skipping", label)
        return None

    # Check kind
    kind = data.get("kind")
    if kind is None:
        logger.warning("No 'kind' in frontmatter of %s — skipping", label)
        return None

    kind = str(kind).strip().lower()
    if kind not in RECOGNIZED_KINDS:
        logger.warning(
            "Unrecognized kind %r in frontmatter of %s — skipping", kind, label
        )
        return None

    # Build the appropriate dataclass
    cls = get_frontmatter_class(kind)
    if cls is None:
        # Shouldn't happen if kind is in RECOGNIZED_KINDS, but defensive.
        logger.warning("No dataclass for kind %r in %s — skipping", kind, label)
        return None

    try:
        return _build_frontmatter(cls, data, kind, label)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("Frontmatter field error in %s: %s — skipping", label, exc)
        return None


# --- Field coercion helpers ---

def _coerce_str(data: dict, key: str, default: str) -> str:
    """Get a value from *data*, coerce to string, strip whitespace.

    Handles the repeated pattern of ``data.get(key, default)`` with
    None-sentinel treatment (None → default) and ``str().strip()``.
    """
    value = data.get(key, default)
    if value is None:
        value = default
    return str(value).strip()


# --- Kind-specific parse functions ---

def _parse_task(data: dict, label: str) -> TaskFrontmatter:
    """Parse task frontmatter with validation."""
    plan = _coerce_str(data, "plan", "")

    order = data.get("order", 0)
    try:
        order = int(order)
    except (TypeError, ValueError):
        logger.warning("Invalid 'order' in %s — defaulting to 0", label)
        order = 0

    status_raw = _coerce_str(data, "status", "pending").lower()
    try:
        status = TaskStatus(status_raw)
    except ValueError:
        logger.warning(
            "Invalid 'status' %r in %s — defaulting to pending", status_raw, label
        )
        status = TaskStatus.PENDING

    verify = _coerce_str(data, "verify", "")

    title = _coerce_str(data, "title", "")

    return TaskFrontmatter(
        kind="task",
        plan=plan,
        order=order,
        status=status,
        verify=verify,
        title=title,
    )


def _parse_plan(data: dict, label: str) -> PlanFrontmatter:
    """Parse plan frontmatter with validation."""
    slug = _coerce_str(data, "slug", "")

    status_raw = _coerce_str(data, "status", "open").lower()
    try:
        status = PlanStatus(status_raw)
    except ValueError:
        logger.warning(
            "Invalid 'status' %r in %s — defaulting to open", status_raw, label
        )
        status = PlanStatus.OPEN

    goal = _coerce_str(data, "goal", "")

    return PlanFrontmatter(
        kind="plan",
        slug=slug,
        status=status,
        goal=goal,
    )


def _parse_knowledge(data: dict, label: str) -> KnowledgeFrontmatter:
    """Parse project-knowledge frontmatter."""
    project = _coerce_str(data, "project", "")

    return KnowledgeFrontmatter(
        kind="project-knowledge",
        project=project,
    )


# --- Dispatch table (OCP-compliant) ---

# Mapping from kind string to the corresponding parse function.
# Adding a new kind requires adding an entry here and a new _parse_X function —
# no if/elif chain to modify in _build_frontmatter.
_FRONTMATTER_PARSERS: dict[str, callable] = {
    "task": _parse_task,
    "plan": _parse_plan,
    "project-knowledge": _parse_knowledge,
}


def _build_frontmatter(
    cls: type[PlanFileFrontmatter],
    data: dict,
    kind: str,
    label: str,
) -> PlanFileFrontmatter:
    """Instantiate the correct frontmatter dataclass from parsed YAML data.

    Dispatches through ``_FRONTMATTER_PARSERS`` so adding a new kind
    requires only a dict entry — no if/elif chain (OCP).
    """
    parser = _FRONTMATTER_PARSERS.get(kind)
    if parser is None:
        raise ValueError(f"Unknown kind: {kind}")
    return parser(data, label)


def parse_file(
    path: Path,
) -> Optional[PlanFileFrontmatter]:
    """Read a file from disk and parse its frontmatter.

    Wraps parse_frontmatter with file I/O. Returns None on any error
    (file not found, permission denied, parse failure).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Cannot read %s: %s — skipping", path, exc)
        return None

    return parse_frontmatter(text, file_path=path)
