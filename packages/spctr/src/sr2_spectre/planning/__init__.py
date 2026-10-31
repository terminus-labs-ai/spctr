"""Planning subsystem for Spectre.

Durable, plan-aware context injection: a directory-per-plan layout on disk
plus a dynamic resolver that re-anchors the relevant plan/task/knowledge into
the LLM context each turn. Planning is a Spectre concern (not SR2 core); the
resolver plugs into SR2 via the ``sr2.resolvers`` entry point.

Public API:
  - ``LayerBudget`` — priority-aware token-budget allocator (pure logic).
  - ``PlanResolver`` / ``PlanResolverError`` — the dynamic resolver.
  - frontmatter parsing (``parse_file``, ``parse_frontmatter``,
    ``extract_raw_frontmatter``) and the frozen frontmatter models.
"""

from __future__ import annotations

from sr2_spectre.planning.budget import LayerBudget
from sr2_spectre.planning.frontmatter import (
    extract_raw_frontmatter,
    parse_file,
    parse_frontmatter,
    split_frontmatter,
)
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

__all__ = [
    "LayerBudget",
    "PlanResolver",
    "PlanResolverError",
    "extract_raw_frontmatter",
    "parse_file",
    "parse_frontmatter",
    "split_frontmatter",
    "KnowledgeFrontmatter",
    "PlanFileFrontmatter",
    "PlanFrontmatter",
    "PlanStatus",
    "RECOGNIZED_KINDS",
    "TaskFrontmatter",
    "TaskStatus",
    "get_frontmatter_class",
]


# ---------------------------------------------------------------------------
# Lazy imports for PlanResolver (avoids pulling in sr2.config.models at
# import time, which breaks in environments where sr2 is out of sync).
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    if name in ("PlanResolver", "PlanResolverError"):
        from sr2_spectre.planning.resolver import (
            PlanResolver,
            PlanResolverError,
        )
        # Cache in module globals so repeated access is fast.
        globals()["PlanResolver"] = PlanResolver
        globals()["PlanResolverError"] = PlanResolverError
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
