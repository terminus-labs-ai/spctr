"""PlanResolver: dynamic, per-turn plan + knowledge injection for SR2.

Implements FR1-8 from the auto-decomposition spec:
  - Directory-per-plan layout under configurable plans_root.
  - Configurable knowledge_root for project-knowledge files.
  - Layered L1/L2/L3 injection with clear delimiters.
  - Per-turn re-read (not frozen at init) so mid-run status changes are
    reflected on the next turn.

Planning is a Spectre concern; this resolver plugs into SR2's pipeline via the
``sr2.resolvers`` entry point (name ``plan``) — opt-in via pipeline config.
"""

from __future__ import annotations

import glob as _glob
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.token_counting import CHARS_PER_TOKEN
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

from sr2_spectre.planning.budget import LayerBudget
from sr2_spectre.planning.frontmatter import (
    parse_file,
    parse_frontmatter,
    split_frontmatter,
)
from sr2_spectre.planning.models import (
    KnowledgeFrontmatter,
    PlanFrontmatter,
    PlanStatus,
    TaskFrontmatter,
    TaskStatus,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_SUBSCRIPTION = EventSubscription(
    event_name="turn_start", phase=EventPhase.STARTING
)


# ---------------------------------------------------------------------------
# Layer delimiter constants
# ---------------------------------------------------------------------------

_PLANNING_HEADER = "## Planning"
_LAYER1_HEADER = "## Project Knowledge"
_LAYER2_HEADER = "## Active Plan"
_LAYER2_FINDINGS_HEADER = "## Active Findings"
_LAYER3_HEADER = "## Current Task"

_LAYER_SEPARATOR = "\n---\n"

# Layer priority for budget enforcement (lower number = higher priority = more protected).
# L3 (current task) is most important — the agent needs it to execute right now.
# L2 (plan overview) provides context for the current task.
# L1 (project knowledge) is background — useful but survivable without.
_LAYER_PRIORITY: dict[str, int] = {
    _LAYER1_HEADER: 3,  # dropped first
    _LAYER2_HEADER: 2,  # dropped second
    _LAYER3_HEADER: 1,  # dropped last (most protected)
    _PLANNING_HEADER: 3,  # same priority as L1 — trigger is disposable
}

# Sentinel value for dynamic project derivation at resolve time.
_AUTO_PROJECT = "__auto__"


# ---------------------------------------------------------------------------
# PlanResolver
# ---------------------------------------------------------------------------


class PlanResolver:
    """Dynamically resolves plan + knowledge content on every turn.

    Config fields (in ``ResolverConfig.config``)
    -------------------------------------------
    plans_root : str
        Root directory holding one sub-directory per plan.
        Default: ``~/.sr2/plans``
    knowledge_root : str
        Directory holding project-knowledge markdown files.
        Default: ``~/.sr2/knowledge/<project>`` (expanded at init, or
        dynamically when ``project`` is ``__auto__``).
    project : str
        Active project name; used to filter L1 knowledge files.
        **Required.** Set to ``__auto__`` to derive dynamically per-turn
        from ``SR2_PROJECT`` env var or cwd-based ``.git`` discovery.
    max_tokens : int | None
        Optional token budget for the combined injection.
    planning_guide_path : str | None
        Path to the planning-guide.md file. When set and no open plan
        exists, a short nudge is injected directing the agent to load
        this guide for multi-step work. Suppressed when a plan is open.
    """

    name: str = "plan"

    def __init__(
        self,
        config: ResolverConfig,
        run_context_provider: Callable[[], dict[str, str] | None] | None = None,
    ) -> None:
        self._config = config
        self._run_context_provider = run_context_provider
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, [_DEFAULT_SUBSCRIPTION]
        )

        # Required: project name (or __auto__ sentinel)
        raw_project: str | None = config.config.get("project")
        if not raw_project:
            raise ValueError(
                "PlanResolver requires config['project'] to be set. "
                "Use a literal project name (e.g. 'sr2-spectre') or "
                "'__auto__' for dynamic per-turn derivation."
            )

        # Track whether project is auto-derived
        self._is_auto: bool = raw_project == _AUTO_PROJECT

        # For explicit (non-auto) projects, store the value directly.
        # For __auto__, _resolve_project() is called at resolve time.
        if not self._is_auto:
            self._project: str = raw_project

        # Optional: roots with defaults
        plans_root_raw: str = config.config.get(
            "plans_root", str(Path.home() / ".sr2" / "plans")
        )
        self._plans_root = Path(plans_root_raw).expanduser().resolve()

        # knowledge_root: if explicitly provided, store resolved path.
        # If not provided (default), we need to re-resolve each turn when __auto__.
        self._knowledge_root_explicit: bool = "knowledge_root" in config.config
        if self._knowledge_root_explicit:
            knowledge_root_raw: str = config.config.get(
                "knowledge_root",
                str(Path.home() / ".sr2" / "knowledge" / raw_project),
            )
            self._knowledge_root = Path(knowledge_root_raw).expanduser().resolve()
        else:
            # Default knowledge_root — will be resolved dynamically when __auto__.
            # For explicit projects, resolve now.
            if not self._is_auto:
                self._knowledge_root = Path.home().expanduser() / ".sr2" / "knowledge" / raw_project
            else:
                # Placeholder — resolved in _resolve_knowledge_root().
                self._knowledge_root = Path.home() / ".sr2" / "knowledge" / raw_project

        self._max_tokens: int | None = config.config.get("max_tokens")

        # Optional: planning guide path for state-aware trigger injection.
        self._planning_guide_path: str | None = config.config.get(
            "planning_guide_path"
        )

    # ------------------------------------------------------------------
    # Dynamic project resolution
    # ------------------------------------------------------------------

    def _resolve_project(self) -> str | None:
        """Resolve the active project name — the one area -> project
        translation point (FR 13).

        If ``project`` was set to ``__auto__``, derive dynamically, highest
        priority first:
        1. The run context's ``area`` key, when present and non-empty — the
           interface's resolved area IS the project name.
        2. The run context's ``area`` key present and empty — explicitly no
           project; resolution stops here (returns None), no fallback.
        3. ``SR2_PROJECT`` env var.
        4. Walk up from ``os.getcwd()`` looking for ``.git`` — use the
           containing directory's name.
        5. Fallback: use ``os.getcwd()`` basename.

        Returns the project name string, or None when the run context
        supplies an explicitly empty area (case 2 above).
        """
        if not self._is_auto:
            return self._project

        # 1 & 2. Run context area, when the provider supplies the key at all.
        if self._run_context_provider is not None:
            ctx = self._run_context_provider()
            if ctx is not None and "area" in ctx:
                area = ctx["area"]
                return area if area else None

        # 3. Env var
        env_project = os.environ.get("SR2_PROJECT")
        if env_project:
            return env_project

        # 4. Walk up from cwd looking for .git
        try:
            cwd = Path.cwd().resolve()
        except OSError:
            cwd = Path("/tmp")

        current = cwd
        while True:
            git_dir = current / ".git"
            if git_dir.is_dir() or git_dir.is_file():
                return current.name
            parent = current.parent
            if parent == current:
                # Reached filesystem root without finding .git
                break
            current = parent

        # 5. Fallback: use cwd basename
        logger.warning(
            "PlanResolver: could not find .git walking up from %s — "
            "using cwd name '%s' as project fallback.",
            cwd,
            cwd.name,
        )
        return cwd.name

    def _resolve_knowledge_root(self) -> Path | None:
        """Resolve the knowledge_root path.

        If knowledge_root was explicitly configured in YAML, return the
        pre-resolved path (unchanged). If using the default and project is
        ``__auto__``, re-derive the path from the resolved project name.
        Returns None when the resolved project is None (explicitly no
        project/area) — there is no default directory to point at.
        """
        if not self._is_auto or self._knowledge_root_explicit:
            return self._knowledge_root

        project = self._resolve_project()
        if project is None:
            return None
        return Path.home().expanduser() / ".sr2" / "knowledge" / project

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, config: ResolverConfig, deps: Dependencies) -> "PlanResolver":
        return cls(config, run_context_provider=deps.run_context_provider)

    def current_frame_id(self) -> str | None:
        """Return the active frame id for the lowest-order pending task.

        Returns ``plan:<plan-slug>/<task-slug>`` when an open plan has pending
        tasks, or ``None`` when no open plan exists or all tasks are complete.

        Designed to be used as ``active_frame_provider`` on
        ``Dependencies`` — the orchestrator stamps this value into
        ``block.meta["frame"]`` on every emitted content block.
        """
        open_plan_dir, plan_fm = self._find_open_plan()
        if not open_plan_dir or not plan_fm:
            return None

        plan_slug = plan_fm.slug

        # Find the lowest-order pending task
        pending = self._find_pending_tasks(open_plan_dir)
        if not pending:
            return None

        # Already sorted by order; pick first
        _, task_slug = pending[0]
        return f"plan:{plan_slug}/{task_slug}"

    def _find_pending_tasks(
        self, plan_dir: Path
    ) -> list[tuple[int, str]]:
        """Return sorted list of (order, slug) for pending tasks.

        Slug is derived from the filename (NN-slug.md → slug).
        """
        pattern = str(plan_dir / "*.md")
        paths = sorted(_glob.glob(pattern, recursive=True))

        pending: list[tuple[int, str]] = []

        for path_str in paths:
            path = Path(path_str)
            if path.name == "_plan.md":
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Cannot read %s: %s — skipping", path, exc)
                continue

            fm = parse_file(path)
            if isinstance(fm, TaskFrontmatter) and fm.status == TaskStatus.PENDING:
                slug = self._extract_slug_from_filename(path.name)
                pending.append((fm.order, slug))

        pending.sort(key=lambda x: x[0])
        return pending

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        """Resolve L1 + L2 + L3 content dynamically from disk.

        Re-reads plan and knowledge directories on every call so that
        mid-run status changes are reflected immediately.
        """
        self.execution_count += 1

        # Resolve the active project (dynamic when __auto__)
        project = self._resolve_project()

        layers: list[tuple[str, str]] = []  # (layer_header, content)

        # Resolve the single open plan (or none) — called once, used for both
        # the planning trigger decision and L2/L3 injection.
        open_plan_dir, plan_frontmatter = self._find_open_plan()

        # Planning trigger: inject only when no open plan and guide path configured.
        # When a plan IS open, L2/L3 already carry the context, so the nudge is
        # redundant noise.
        if self._planning_guide_path and not open_plan_dir:
            trigger = (
                f"> For multi-step tasks, load the planning guide "
                f"(`file_read` `{self._planning_guide_path}`) and create a plan."
            )
            layers.append((_PLANNING_HEADER, trigger))

        # L1: project knowledge
        l1_content = self._resolve_layer1(project)
        if l1_content:
            layers.append((_LAYER1_HEADER, l1_content))

        if open_plan_dir and plan_frontmatter:
            # L2: plan-shared (_plan.md) — always injected when plan is open,
            # even if body is empty (signals to the model that a plan exists).
            l2_content = self._resolve_layer2(open_plan_dir)
            layers.append((_LAYER2_HEADER, l2_content))

            # L2-findings: _findings.md — injected alongside L2 when present
            # and non-empty; omitted when absent or empty.
            findings_content = self._resolve_findings(open_plan_dir)
            if findings_content.strip():
                layers.append((_LAYER2_FINDINGS_HEADER, findings_content))

            # L3: current task (lowest-order pending)
            l3_content = self._resolve_layer3(open_plan_dir)
            if l3_content.strip():
                layers.append((_LAYER3_HEADER, l3_content))

        # Token budget enforcement — layer-priority-aware (FR8, obsidian-2v2).
        # Drop layers from lowest to highest priority before joining.
        if self._max_tokens is not None:
            layers = self._enforce_budget(layers, self._max_tokens)

        # Build the combined text with delimiters
        combined_parts: list[str] = []
        for header, content in layers:
            combined_parts.append(header)
            combined_parts.append(content)

        combined = _LAYER_SEPARATOR.join(combined_parts)

        tokens = len(combined) // CHARS_PER_TOKEN

        return ResolvedContent(
            resolver_name=self.name,
            source_layer="plan",
            content=[TextBlock(text=combined)],
            token_count=tokens,
        )

    # ------------------------------------------------------------------
    # Layer resolution helpers
    # ------------------------------------------------------------------

    def _resolve_layer1(self, project: str | None) -> str:
        """L1: Load all project-knowledge files matching the active project.

        Globs ``*.md`` under ``knowledge_root``, parses frontmatter, filters
        by ``kind: project-knowledge`` with matching ``project`` field.
        Returns concatenated content of matching files (body only, frontmatter
        stripped).

        *project* is None when resolution explicitly yielded no project (an
        empty run-context area) — in that case no knowledge is injected and
        the knowledge root is never even computed.
        """
        if project is None:
            return ""

        knowledge_root = self._resolve_knowledge_root()

        if knowledge_root is None or not knowledge_root.is_dir():
            return ""

        pattern = str(knowledge_root / "*.md")
        paths = sorted(_glob.glob(pattern, recursive=True))

        parts: list[str] = []
        for path_str in paths:
            path = Path(path_str)
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Cannot read %s: %s — skipping", path, exc)
                continue

            fm = parse_frontmatter(text, file_path=path)
            if isinstance(fm, KnowledgeFrontmatter) and fm.project == project:
                body = self._strip_frontmatter(text)
                if body.strip():
                    parts.append(body)

        return "\n\n".join(parts)

    def _find_open_plan(
        self,
    ) -> tuple[Path | None, PlanFrontmatter | None]:
        """Find the single open plan directory.

        Globs immediate sub-directories under ``plans_root``. Parses each
        ``_plan.md`` looking for ``status: open``.

        Returns:
            (plan_dir, plan_frontmatter) or (None, None) if no open plans,
            or raises PlanResolverError if multiple open plans found.
        """
        if not self._plans_root.is_dir():
            return None, None

        plan_dirs: list[Path] = sorted(
            p for p in self._plans_root.iterdir() if p.is_dir()
        )

        open_plans: list[tuple[Path, PlanFrontmatter]] = []

        for plan_dir in plan_dirs:
            plan_file = plan_dir / "_plan.md"
            if not plan_file.is_file():
                continue

            fm = parse_file(plan_file)
            if isinstance(fm, PlanFrontmatter) and fm.status == PlanStatus.OPEN:
                open_plans.append((plan_dir, fm))

        if len(open_plans) > 1:
            slugs = [str(p[0].name) for p in open_plans]
            raise PlanResolverError(
                f"Multiple open plans detected: {', '.join(slugs)}. "
                "PlanResolver v1 supports at most one open plan. "
                "Close or complete the plans you don't need."
            )

        if not open_plans:
            return None, None

        return open_plans[0]

    def _resolve_layer2(self, plan_dir: Path) -> str:
        """L2: Load the _plan.md body (goal + constraints contract)."""
        plan_file = plan_dir / "_plan.md"
        if not plan_file.is_file():
            return ""

        text = plan_file.read_text(encoding="utf-8")
        return self._strip_frontmatter(text)

    def _resolve_findings(self, plan_dir: Path) -> str:
        """Resolve _findings.md: inject body when present and non-empty.

        Returns the raw file body (no frontmatter stripping — _findings.md
        has no required frontmatter). Returns empty string when absent or
        whitespace-only.
        """
        findings_file = plan_dir / "_findings.md"
        if not findings_file.is_file():
            return ""

        try:
            text = findings_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

        return text

    def _resolve_layer3(self, plan_dir: Path) -> str:
        """L3: Load the lowest-order pending task file body.

        Globs ``NN-*.md`` files in the plan directory, parses frontmatter,
        filters for ``kind: task`` with ``status: pending``, sorts by ``order``,
        returns the body of the first match.
        """
        pending_tasks: list[tuple[int, str]] = []  # (order, body_text)

        for order, slug in self._find_pending_tasks(plan_dir):
            task_file = plan_dir / f"{order:02d}-{slug}.md"
            try:
                text = task_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            body = self._strip_frontmatter(text)
            pending_tasks.append((order, body))

        if not pending_tasks:
            return ""

        return pending_tasks[0][1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_slug_from_filename(filename: str) -> str:
        """Extract the slug from a task filename like '02-cli-flag.md'.

        Strips the leading NN- prefix and the .md suffix.
        """
        # Remove .md suffix
        name = filename.removesuffix(".md")
        # Remove leading NN- prefix
        dash_idx = name.index("-")
        return name[dash_idx + 1 :]

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Strip the YAML frontmatter block (--- ... ---) from text.

        Returns the body after the closing ``---``. If no frontmatter is
        found, returns the original text.
        """
        result = split_frontmatter(text)
        if result is None:
            return text

        _fm_block, body = result
        return body

    def _enforce_budget(
        self,
        layers: list[tuple[str, str]],
        max_tokens: int,
    ) -> list[tuple[str, str]]:
        """Enforce token budget via LayerBudget allocator.

        Delegates to :class:`LayerBudget` which handles priority-aware
        eviction and tail truncation.
        """
        # Map (header, content) → (header, content, priority).
        prioritized: list[tuple[str, str, int]] = [
            (h, c, _LAYER_PRIORITY.get(h, 0)) for h, c in layers
        ]
        budget = LayerBudget(max_tokens=max_tokens)
        return budget.allocate(prioritized)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class PlanResolverError(Exception):
    """Raised when PlanResolver encounters a non-tolerable error.

    Used for: multiple open plans (v1 limitation).
    """
