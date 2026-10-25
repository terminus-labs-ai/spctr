"""Runtime — shared sub-runtime for all frames.

Holds config, LLM callable, MCP clients, tool registry, and shared stores.
One Runtime instance serves N per-frame Sessions, each with its own SR2.
"""

from __future__ import annotations

import asyncio
import logging
import os
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sr2.pipeline.provenance import ProvenanceStore
    from sr2.pipeline.tracing import Tracer

from sr2.memory import InMemoryMemoryStore, PostgresMemoryStore
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2_spectre.config import SpectreConfig
from sr2_spectre.live_llm import LiveLLM
from sr2_spectre.mcp.client import MCPClient, MCPConnectionError
from sr2_spectre.path_resolution import ConfigPathError
from sr2_spectre.run_log import SessionLogManager
from sr2_spectre.session import Session
from sr2_spectre.skills.builtin import DEFAULT_SKILLS
from sr2_spectre.skills.core import SkillRegistry, discover_skills, load_skill_from_path
from sr2_spectre.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _tool_accepts_workspace_root(class_path: str) -> bool:
    """Return True if the tool class's __init__ accepts a workspace_root kwarg.

    Used to inject the workspace floor only into tools that support it,
    avoiding a TypeError when a non-confining tool (e.g. FileReadTool) is
    handed an unexpected kwarg once SR2_WORKSPACE is set.
    """
    import importlib
    import inspect

    try:
        module_path, class_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        params = inspect.signature(cls.__init__).parameters
    except (ImportError, AttributeError, ValueError, TypeError):
        return False
    # Only an EXPLICIT named parameter counts. A bare object.__init__ reports
    # *args/**kwargs, which would falsely match every argument-less tool.
    return "workspace_root" in params


class Runtime:
    """Shared sub-runtime for all frames.

    Owns:
    - SpectreConfig (single source of truth)
    - LiveLLM (one LLM path, retargetable on config reload)
    - ToolRegistry (tool definitions; stateless executors)
    - MCPClient instances (connected once)
    - Shared in-memory MemoryStore (backs memory resolver/transformer; spc-49)
    - Shared persistent ProvenanceStore (spc-50)

    Creates per-frame Session instances via new_session().
    """

    def __init__(self, config: SpectreConfig) -> None:
        self.config = config
        self.registry = ToolRegistry()

        # Resolve workspace root for confinement (FR1)
        self.workspace_root: str | None = None
        workspace_raw = os.environ.get("SR2_WORKSPACE")
        if workspace_raw:
            from pathlib import Path
            self.workspace_root = str(Path(workspace_raw).resolve())

        # Names registered from agent.tools, tracked so a later config reload
        # can tell a tool this config dropped from one an MCP server supplied.
        self._config_tool_names: set[str] = set()
        self._register_config_tools(config, strict=True)

        # Auto-inject complete_step when a plan resolver is configured.
        # complete_step + plan resolver are one feature unit: if the pipeline
        # declares a plan resolver, the agent needs complete_step to mark
        # tasks done. Scanning pipeline.layers for type=='plan' gives us the
        # resolver's plans_root so there's a single source of truth.
        self._auto_inject_complete_step(config)

        # Bootstrap SkillRegistry: the full three-tier catalog (builtins,
        # discovered skills_dirs, per-file skills[]), registered in place by
        # the same rebuild path the per-message rescan uses (spc-110).
        self.skill_registry = SkillRegistry()
        self._rebuild_skills(config)

        # Filesystem signature of skills_dirs as of the bootstrap scan above.
        # Seeded here so the first per-message apply_config on an unchanged
        # filesystem does not rebuild (spc-110).
        self._skills_fs_signature: "tuple | None" = None
        # Last rescan failure message, for warn-once suppression (spc-110,
        # mirrors config_source.LiveConfigSource._load_error): repeats of the
        # same failure are quiet, a different failure warns again, and a
        # success clears it.
        self._skills_rescan_error: "str | None" = None
        self._seed_skills_fs_signature(config)

        # MCP clients — one per mcp_servers entry; connected lazily via initialize()
        self._mcp_clients: list[MCPClient] = []
        for mcp_cfg in config.agent.mcp_servers:
            if mcp_cfg.type == "stdio":
                client = MCPClient(server_type="stdio", command=mcp_cfg.command, args=mcp_cfg.args, env=mcp_cfg.env)
            elif mcp_cfg.type in ("streamable-http", "streamable_http"):
                client = MCPClient(server_type="streamable-http", url=mcp_cfg.url)
            else:
                client = MCPClient(server_type="http", url=mcp_cfg.url)
            self._mcp_clients.append(client)

        # Persistent provenance store — shared across all sessions.
        # Constructed here (path resolved), connected in initialize(), closed in aclose().
        self._provenance_store: "ProvenanceStore | None" = None
        self._provenance_store_path = self._resolve_provenance_path(config)

        # Shared memory store — one per Runtime, threaded into every Session so
        # memory accrued in one frame is visible to the resolver in later turns.
        # Backend is config/env-selectable (obsidian-cor): a DSN selects the
        # persistent Postgres store (survives restart, shared across processes);
        # absent/disabled falls back to the synchronous dict-backed in-memory
        # store (lost on exit). Both are synchronous and constructed eagerly.
        memory_dsn = self._resolve_memory_dsn(config)
        if memory_dsn is not None:
            self._memory_store = PostgresMemoryStore(memory_dsn)
            logger.info("Using persistent Postgres memory store")
        else:
            self._memory_store = InMemoryMemoryStore()

        # Build the LLM handle. LiveLLM, not a bare LiteLLMCallable: sessions
        # capture their LLM for life, so the indirection is what lets a
        # corrected model or base_url reach conversations that are already
        # open (see live_llm.LiveLLM).
        self.llm = LiveLLM(config.active_model_config)

        # Live sessions, weakly held so a closed frame is not kept alive just
        # to be told about config reloads.
        self._sessions: "weakref.WeakSet[Session]" = weakref.WeakSet()
        self._session_logs = SessionLogManager()

        self._active_frame_provider: Callable[[str], str | None] | None = None
        self._plan_resolver: Any = None
        self._plan_resolver_config = None
        self._wire_plan_resolver(config)

        # Reminders (spc-139, FR6): one switch registers the ``reminder``
        # tool and the delivery loop. Disabled (the default) registers
        # neither, so an unchanged config keeps behavior identical.
        # _reminders_seq counts enabled re-seats (startup included);
        # _reminders_invalidated counts the re-seats a live owning
        # interface must react to (spc-143, apply_config only).
        self.reminders_enabled: bool = False
        self.reminders: Any = None
        # spc-159: the owning interface's named-channel resolver (FR10),
        # supplied once through set_reminders_channel_resolver. Every
        # ReminderTool construction (startup and live-reload re-seat,
        # spc-143) passes it through the constructor; a tool already
        # seated when the resolver arrives is rebuilt through
        # _reminders_for so the new one takes it there too (spc-166).
        # None until an owning interface registers one.
        self._reminders_channel_resolver: Any = None
        self._reminders_origin: Any = None
        self._reminders_seq: int = 0
        self._reminders_invalidated: int = 0
        self._reminders_for(config)

        # Schedules (spc-185, FR23): one switch registers the ``schedule``
        # tool and the engine. Disabled (the default) registers neither.
        self.schedules_enabled: bool = False
        self.schedules: Any = None
        self._schedules_seq: int = 0
        self._schedules_invalidated: int = 0
        self._schedules_for(config)

    def _reminders_for(self, config: SpectreConfig) -> None:
        """Register (or skip) the reminder tool per ``agent.reminders``.

        The tool is registered in the Runtime's shared registry; the
        delivery loop is started/stopped by the owning interface (which
        supplies the send callable), through the ``reminders`` handle.
        Disabled is a no-op — nothing is registered and no loop exists.
        """
        from sr2_spectre.reminders import (
            DeliveryLoop,
            ReminderStore,
            ReminderTool,
        )
        from zoneinfo import ZoneInfo

        block = config.agent.reminders
        agent_name = config.agent.name
        self.reminders_enabled = bool(block.enabled)
        if not block.enabled:
            self.reminders = None
            self.registry.unregister(ReminderTool.name)
            return
        if not block.timezone:
            raise ValueError(
                "agent.reminders.enabled is true but agent.reminders.timezone "
                "is empty; an IANA timezone is required when reminders are "
                "enabled."
            )
        store = ReminderStore(block.resolved_store_path(agent_name))
        tz = ZoneInfo(block.timezone)
        tool = ReminderTool(
            store=store,
            timezone=block.timezone,
            agent_name=agent_name,
            origin_getter=self._reminders_origin_getter,
            topic=block.zulip_topic,
            # spc-159: the constructor is the resolver's only
            # construction path; the Runtime passes the seam's resolver
            # (or None, until an owning interface registers one) here —
            # every re-seat rebuild carries it too.
            channel_resolver=self._reminders_channel_resolver,
        )
        self.registry.register(
            name=ReminderTool.name,
            description=ReminderTool.description,
            input_schema=ReminderTool.input_schema,
            fn=tool.__call__,
        )
        # The delivery loop is owned by the interface (it owns the send
        # path); the Runtime only keeps the store and settings the
        # interface needs to build its loop.
        self.reminders = SimpleNamespace(
            store=store,
            tool=tool,
            agent_name=agent_name,
            poll_seconds=block.poll_seconds,
            tz=tz,
            topic=block.zulip_topic,
        )
        # spc-143: every re-seat (including the startup one) bumps the
        # sequence, so an interface holding a stale loop notices the handle
        # behind it was rebuilt even when enabled stays true.
        self._reminders_seq += 1

    def set_reminders_channel_resolver(self, resolver: Callable) -> None:
        """Register the owning interface's named-channel resolver (spc-159).

        The single path a resolver takes: the interface that owns the
        platform calls this once (at start), and every ReminderTool this
        Runtime builds — the startup one and each live-reload re-seat
        (spc-143) — receives it through its constructor. ``resolver`` is
        ``async (platform, name) -> list`` of matches, each carrying
        ``channel_id`` and ``channel_name`` (the FR10 match shape).

        Platform-gated per FR11: only a chat platform's resolver is
        kept; any other interface's is ignored, so a non-Discord/
        non-Zulip interface behaves exactly as with no resolver at all.

        spc-166: the Runtime seats its tool at startup, before any
        interface exists to register a resolver, so registering one
        rebuilds an already-seated tool through ``_reminders_for`` — the
        existing single construction site — and the registered tool
        carries the resolver from its constructor. A rebuild is NOT a
        reload: ``reminders_stamped`` does not move, so a running
        delivery loop keeps running (it shares only the store, whose
        path a same-config rebuild does not change). With reminders
        disabled there is no tool to rebuild and nothing is built.
        """
        platform = getattr(resolver, "spectre_platform", None)
        if platform not in ("discord", "zulip"):
            logger.warning(
                "Ignored a reminders channel resolver from unknown platform "
                "%r (expected 'discord' or 'zulip')",
                platform,
            )
            return
        self._reminders_channel_resolver = resolver
        # spc-166: the constructor is the resolver's only way into a tool,
        # so an already-seated tool is replaced by one built with it.
        if self.reminders is not None:
            self._reminders_for(self.config)

    def _reminders_origin_getter(self) -> Any:
        """The current turn's reminder origin, or None outside one.

        The owning interface records the origin on this Runtime right
        before its turn runs (and clears it after), so concurrent turns in
        different channels never record each other's origin — this reads
        per-turn state, not shared mutable config.
        """
        return getattr(self, "_reminders_origin", None)

    def set_reminders_origin(self, origin: Any) -> None:
        """Record the origin of the turn about to run (interface-owned)."""
        self._reminders_origin = origin

    def clear_reminders_origin(self) -> None:
        """Clear the turn origin after the turn ends (interface-owned)."""
        self._reminders_origin = None

    @property
    def reminders_seq(self) -> int:
        """Count of reminder re-seats — enabled constructions, startup
        included (spc-143).

        Changes whenever ``_reminders_for`` ran with an enabled block.
        Disable clears the handle without a seat, so this alone cannot
        tell an interface what to do — owning interfaces compare
        ``reminders_stamped``; this distinguishes a rebuild (R4) from a
        first start for consumers that want it.
        """
        return self._reminders_seq

    @property
    def reminders_stamped(self) -> int:
        """Count of reminder re-seats a live interface must react to.

        ``apply_config`` bumps it whenever ``agent.reminders`` changed
        (enable, disable, settings) — never for an unchanged block, so
        the no-op reload stays a single equality check end to end (R3).
        The owning interface compares it against the stamp taken when its
        delivery loop started; different: re-assert the loop from the
        current handle (start — was disabled, stop — now disabled with
        pending rows intact, restart — new settings took effect). Startup
        with an enabled block counts too, so ``start()`` and the sync
        share one mechanism.
        """
        return self._reminders_invalidated

    # ---- schedules (spc-185, FR23) ----------------------------------------

    def _schedules_for(self, config: SpectreConfig) -> None:
        """Register (or skip) the schedule tool per ``agent.schedules``."""
        from sr2_spectre.schedules import ScheduleStore
        from sr2_spectre.tools.schedule_tool import ScheduleTool
        from zoneinfo import ZoneInfo

        block = config.agent.schedules
        agent_name = config.agent.name
        self.schedules_enabled = bool(block.enabled)
        if not block.enabled:
            self.schedules = None
            self.registry.unregister(ScheduleTool.name)
            return
        if not block.timezone:
            raise ValueError(
                "agent.schedules.enabled is true but agent.schedules.timezone "
                "is empty; an IANA timezone is required when schedules are "
                "enabled."
            )
        store = ScheduleStore(block.resolved_store_path(agent_name))
        tz = ZoneInfo(block.timezone)
        tool = ScheduleTool(
            store=store,
            timezone=block.timezone,
            agent_name=agent_name,
            origin_getter=lambda: None,  # interface sets scope via ContextVar
        )
        self.registry.register(
            name=ScheduleTool.name,
            description=ScheduleTool.description,
            input_schema=ScheduleTool.input_schema,
            fn=tool.__call__,
        )
        self.schedules = SimpleNamespace(
            store=store,
            tool=tool,
            agent_name=agent_name,
            poll_seconds=block.poll_seconds,
            tz=tz,
            command_timeout=block.command_timeout_seconds,
            agent_timeout=block.agent_timeout_seconds,
        )
        self._schedules_seq += 1

    @property
    def schedules_seq(self) -> int:
        return self._schedules_seq

    @property
    def schedules_stamped(self) -> int:
        return self._schedules_invalidated

    def _wire_plan_resolver(self, config: SpectreConfig) -> None:
        """Build the PlanResolver and the active_frame_provider SR2 gets.

        ``PlanResolver.current_frame_id()`` returns the active frame for the
        lowest-order pending task. It is wrapped as a
        ``Callable[[str], str | None]`` to match the active_frame_provider
        signature (the origin parameter is ignored).

        Safe to re-run on a config reload. The closure reads
        ``self._plan_resolver`` at call time rather than capturing it, so a
        reload that only changes the resolver's settings is picked up by
        sessions that already hold the closure.
        """
        self._plan_resolver_config = self._find_plan_resolver_config(config)
        if not self._plan_resolver_config:
            self._plan_resolver = None
            self._active_frame_provider = None
            return

        from sr2_spectre.planning import PlanResolver

        self._plan_resolver = PlanResolver(self._plan_resolver_config)

        if self._active_frame_provider is None:
            def _frame_provider(origin: str) -> str | None:
                # origin parameter accepted for SR2 signature compatibility;
                # PlanResolver.current_frame_id() is origin-agnostic.
                _ = origin
                return self._plan_resolver.current_frame_id()

            self._active_frame_provider = _frame_provider

        logger.info("PlanResolver active — wiring active_frame_provider into sessions")

        # Warn if step_compaction transformer is not declared in any layer.
        # A PlanResolver stamps frame metadata on blocks, but without the
        # transformer those frames are never burned — context grows unbounded.
        self._check_step_compaction_config(config)

    def apply_config(self, config: SpectreConfig) -> list[str]:
        """Adopt a freshly reloaded config into this running process.

        Called once per inbound message, off the back of
        ``SpectreConfigSource.reload()``. Reloading answers "what does the file
        say now"; this decides what a live process can actually do about it.

        Every branch below is gated on an observed change, so the steady-state
        cost of a reload that found nothing new is one equality check — not a
        rebuild of the tool registry on every message. The one exception is
        the skills_dirs freshness check (spc-110): skill *files* change
        without the config changing, so it runs on EVERY call, before the
        early return, and its own steady-state cost is one stat per file.

        Fields that a running process cannot re-seat (MCP servers, the store
        connections, the agent identity) are pinned upstream by
        ``SpectreConfigSource`` and never reach this method; the operator is
        warned once that those need a restart.

        Returns:
            Names of the config areas that were actually applied, for logging
            and for tests to assert against.
        """
        # Skill-file freshness runs before the early return: this method is
        # called once per inbound message, most often with an unchanged
        # config object, and an edited SKILL.md must not wait for a config
        # edit (or a restart) to be noticed.
        self._rescan_skills_if_changed(config)

        previous = self.config

        # Reminders (spc-139, FR6, spc-143): the same switch feeds the tool
        # AND the delivery loop, so a changed block re-runs the exact
        # construction __init__ does — register plus rebuild the handle, or
        # unregister plus clear it — and re-stamps, so the owning interface
        # starts, stops or restarts its delivery loop on the next message
        # without restarting the interface. An unchanged block costs one
        # equality check — no re-registration, and the interface stops at
        # one int compare. Checked BEFORE the unchanged-config early return
        # below: a reload whose only edit is the reminders block must still
        # re-seat, and a broken enabled block must keep raising instead of
        # being swallowed as "no change" there.
        reminders_changed = config.agent.reminders != previous.agent.reminders
        if reminders_changed:
            # A raise here (enabled with an empty timezone — the startup
            # contract) aborts the reload; the config source keeps serving
            # last-good config, so a broken edit retries (and keeps
            # raising) until the file is fixed.
            self._reminders_for(config)
            # The seat moved: re-stamp so the owning interface re-checks
            # its delivery loop on the next message. NOT stamped when the
            # block is unchanged — a no-op reload must not restart a live
            # loop (R3).
            self._reminders_invalidated += 1

        # Schedules (spc-185, FR23): same pattern as reminders.
        schedules_changed = config.agent.schedules != previous.agent.schedules
        if schedules_changed:
            self._schedules_for(config)
            self._schedules_invalidated += 1

        if config == previous:
            # Nothing but the reload itself happened (the reminders branch
            # above already re-seated if the block moved).
            changed: list[str] = []
            if reminders_changed:
                changed.append("agent.reminders")
            if schedules_changed:
                changed.append("agent.schedules")
            if changed:
                logger.info("Applied config reload: %s", ", ".join(changed))
                return changed
            return []

        self.config = config
        applied: list[str] = []
        if reminders_changed:
            applied.append("agent.reminders")
        if schedules_changed:
            applied.append("agent.schedules")
        if reminders_changed or schedules_changed:
            # Align the comparison base for everything below: the reminders
            # branch already re-seated, and the remaining branches and the
            # late early-return judge the REST of the config.
            agent_updates: dict = {}
            if reminders_changed:
                agent_updates["reminders"] = config.agent.reminders
            if schedules_changed:
                agent_updates["schedules"] = config.agent.schedules
            previous = previous.model_copy(
                update={"agent": previous.agent.model_copy(update=agent_updates)}
            )

        # Models and endpoints. The LiveLLM swap reaches sessions that are
        # already open, which is the whole point of the indirection.
        if self.llm.retarget(config.active_model_config):
            applied.append("models")

        if config.agent.tools != previous.agent.tools:
            self._register_config_tools(config, strict=False)
            applied.append("agent.tools")

        if (
            config.agent.skills != previous.agent.skills
            or config.agent.skills_dirs != previous.agent.skills_dirs
            or config.agent.default_skills != previous.agent.default_skills
        ):
            self._rebuild_skills(config)
            applied.append("agent.skills")

        # The pipeline is baked into each frame's SR2, so a pipeline edit means
        # those have to be rebuilt. Sessions defer the rebuild to their next
        # turn, under their own lock — an in-flight reply must not have its SR2
        # swapped mid-stream.
        pipeline_changed = config.pipeline != previous.pipeline
        if pipeline_changed:
            self._wire_plan_resolver(config)
            self._auto_inject_complete_step(config)
            applied.append("pipeline")
        for session in list(self._sessions):
            session.apply_config(
                config,
                rebuild_sr2=pipeline_changed,
                active_frame_provider=self._active_frame_provider,
            )

        if applied:
            logger.info("Applied config reload: %s", ", ".join(applied))
        return applied

    def _register_config_tools(self, config: SpectreConfig, strict: bool) -> None:
        """(Re)register the tools declared in ``agent.tools``.

        workspace_root is injected ONLY into tools whose constructor accepts it
        (FileWriteTool, EditTool, TerminalTool). Blanket injection crashes tools
        that don't take the kwarg (e.g. FileReadTool) the moment SR2_WORKSPACE
        is set.

        Args:
            strict: True at startup, where a tool that will not construct is a
                broken config and should stop the process. False on reload,
                where the process is already serving traffic and a bad edit must
                cost the operator that one tool, not the bot.
        """
        registered: set[str] = set()
        for tool_cfg in config.agent.tools:
            tool_config = dict(tool_cfg.config)
            if (
                self.workspace_root is not None
                and "workspace_root" not in tool_config
                and _tool_accepts_workspace_root(tool_cfg.class_path)
            ):
                tool_config["workspace_root"] = self.workspace_root
            try:
                name = self.registry.register_from_class_path(
                    tool_cfg.class_path, tool_config
                )
            except Exception as exc:  # noqa: BLE001 — see `strict`
                if strict:
                    raise
                logger.warning(
                    "Tool '%s' failed to register on config reload (keeping the "
                    "previous registration if there was one): %s",
                    tool_cfg.class_path,
                    exc,
                )
                continue
            registered.add(name)

        # Retire tools an earlier config declared and this one does not. MCP
        # bridges and auto-injected tools are absent from _config_tool_names,
        # so they survive untouched.
        for stale in self._config_tool_names - registered:
            if self.registry.unregister(stale):
                logger.info("Tool '%s' dropped by config reload", stale)

        self._config_tool_names = registered

    def _rebuild_skills(self, config: SpectreConfig) -> None:
        """Repopulate the skill registry after a config reload or rescan.

        Both the config-value-change branch and the spc-110 per-message
        rescan rebuild through here, and both go through ``_skill_swap``:
        clear + refill in place rather than replacing the registry object
        (the auto-injected ``load_skill`` tool holds a reference to this
        exact object), and restore the exact previous contents if the
        refill raises.

        The registry is a pure function of config + filesystem: the full
        three-tier set (builtins, discovered dirs, per-file ``skills[]``)
        is built and registered inside the swap, so a rescan can never
        drop a tier the bootstrap had.

        ``_auto_inject_load_skill`` is re-run after every refill (not inside
        ``_bootstrap_skills``) so the load_skill tool's registration always
        matches the catalog in BOTH directions: empty→non-empty registers it;
        non-empty→empty withdraws it.
        """
        with self._skill_swap():
            for skill in self._bootstrap_skills(config):
                self.skill_registry.register(skill)
        self._auto_inject_load_skill()

    @contextmanager
    def _skill_swap(self) -> "Iterator[None]":
        """Clear the skill registry, refilling it from the ``with`` body.

        If the body raises, the registry's exact previous contents are
        restored instead — a clear-then-refill never leaves it partially
        filled. The registry OBJECT is never replaced either way, so the
        auto-injected ``load_skill`` tool keeps serving.
        """
        previous_skills = dict(self.skill_registry._skills)
        self.skill_registry.clear()
        try:
            yield
        except BaseException:
            self.skill_registry._skills.clear()
            self.skill_registry._skills.update(previous_skills)
            raise

    # ------------------------------------------------------------------
    # Skills_dirs freshness rescan (spc-110)
    # ------------------------------------------------------------------

    def _seed_skills_fs_signature(self, config: SpectreConfig) -> None:
        """Record the skills_dirs filesystem signature after a full scan.

        Called from ``__init__`` right after ``_bootstrap_skills``: bootstrap
        already scanned the dirs, so seeding here makes the first message on
        an unchanged filesystem cost zero rebuilds. A failed seed leaves the
        signature None, which makes the next rescan rebuild once and re-seed
        — safe, if one message late.
        """
        try:
            self._skills_fs_signature = self._skills_fs_signature_now(config)
        except Exception:  # noqa: BLE001 — a bad seed only costs one rebuild
            # Warn-once (mirrors _note_rescan_failure); no recovery line at
            # startup — there is no prior state to have lost.
            self._skills_rescan_error = "signature seeding failed at startup"
            self._skills_fs_signature = None

    @staticmethod
    def _resolve_skills_dir(dir_raw: str) -> Path | None:
        """Resolve one ``agent.skills_dirs`` entry to a concrete directory.

        Same resolution as core discovery (``resolve_path``: ``~`` expansion,
        ``${VAR}`` interpolation, relative anchoring against the process
        cwd) so the rescan signature and the rebuild agree on which files
        they mean.

        Returns:
            The resolved directory, or None when the path cannot be resolved
            (unset ``${VAR}``). Callers must treat None the way core's
            ``discover_skills_in_dir`` treats it — skip the entry (a stable
            signature component; no failure) — not raise: a raise here fires
            once per message, which is the log noise warn-once exists to
            prevent, and ``resolve_path`` bakes the raw string into the
            exception message so even that could not deduplicate across
            differing entries.
        """
        from sr2_spectre.path_resolution import resolve_path

        root = Path(dir_raw).expanduser()
        try:
            return resolve_path(str(root), Path.cwd(), dict(os.environ))
        except ConfigPathError:
            return None

    @classmethod
    def _skills_fs_signature_now(cls, config: SpectreConfig) -> tuple:
        """Change signature over the configured ``agent.skills_dirs``.

        Covers each configured directory (existence only) and, inside each
        one, the entries the two discovery layouts recognise — flat
        ``<dir>/<name>.md`` and bundled ``<dir>/<name>/SKILL.md`` — as
        (path, mtime_ns, size). Unrecognised entries are not skill-visible,
        so they stay out: steady state costs one stat per skill file.

        The directory itself is signed by existence only, never mtime: the
        files live inside it, so any add/remove already flips their entries,
        and signing the dir would double-count the mtime churn ``os.utime``
        causes on the parent.

        Raises:
            OSError: If a configured directory cannot be stat'd or listed.
                Unreadable and empty ARE distinguishable here (missing dir
                signs as ``exists=False``; unreadable dir raises), but both
                stay out of the recorded signature so the caller's last-good
                path owns the failure instead of silently freezing the
                catalog. An unresolvable path (unset ``${VAR}``) is not a
                failure: it signs as ``(raw, False, (), "unresolvable")``,
                matching the way core discovery skips it while staying
                distinct from the same path resolving to a missing dir.
        """
        signature: list[tuple] = []
        for dir_raw in config.agent.skills_dirs:
            root = cls._resolve_skills_dir(dir_raw)
            if root is None:
                # Unresolvable (${VAR} unset): a stable non-change while it
                # stays unset. Marked unresolvable=True so the message that
                # later sets the var flips the signature even when the dir
                # still does not exist — core discovery would pick it up
                # then, and so must we.
                signature.append((dir_raw, False, (), "unresolvable"))
                continue
            exists = root.is_dir()
            files: list[tuple] = []
            if exists:
                for name in sorted(os.listdir(root)):
                    child = root / name
                    skill_file: Path | None = None
                    if name.endswith(".md") and child.is_file():
                        skill_file = child
                    elif child.is_dir():
                        bundled = child / "SKILL.md"
                        if bundled.is_file():
                            skill_file = bundled
                    if skill_file is not None:
                        stat = skill_file.stat()
                        files.append(
                            (str(skill_file), stat.st_mtime_ns, stat.st_size)
                        )
            signature.append((str(root), exists, tuple(files)))
        return tuple(signature)

    def _note_rescan_failure(self, exc: BaseException) -> None:
        """Warn about a rescan failure, once per distinct message.

        Mirrors ``config_source.LiveConfigSource._note_failure``: repeats of
        the same failure are quiet (a persistently broken tree must not log
        one WARNING per inbound message), a DIFFERENT failure warns again,
        and the next successful scan clears the suppressed state.
        """
        message = f"{type(exc).__name__}: {exc}"
        if message != self._skills_rescan_error:
            logger.warning(
                "Skill directories rescan failed — keeping the previous "
                "skill registry until the next reload succeeds: %s",
                exc,
            )
            self._skills_rescan_error = message

    def _rescan_skills_if_changed(self, config: SpectreConfig) -> None:
        """Rebuild the skill catalog iff the skills_dirs filesystem changed.

        Runs on every ``apply_config`` call, including the unchanged-config
        ones the early return swallows. Steady state: signature match, zero
        log output, no rebuild.

        Last-good contract: on any rescan failure (unreadable configured
        dir, unreadable skill file) the previous registry stays exactly in
        force, this returns normally — the message is processed against the
        last known-good catalog — and at most one warning is logged per
        distinct failure (repeats suppressed; see ``_note_rescan_failure``).
        The rebuild goes through ``_rebuild_skills`` on this same config, so
        the registry is a pure function of config + filesystem: builtins,
        disk discovery and the per-file ``skills[]`` tier are all re-asserted
        in the established order (builtins weakest, per-file strongest).
        An unreadable skill FILE counts as a failed rescan here rather than
        as core's per-file skip: core silently dropping it would change the
        catalog on a transient mid-edit error. The signature is left
        untouched on failure so the next message retries.
        """
        if not config.agent.skills_dirs:
            # Nothing to watch: keep the signature aligned with this config
            # (one bool check) so switching skills_dirs off via a config
            # change cannot leave a stale signature triggering rebuilds.
            self._skills_fs_signature = ()
            if self._skills_rescan_error is not None:
                logger.info("skills_dirs rescans recovered")
                self._skills_rescan_error = None
            return

        try:
            signature = self._skills_fs_signature_now(config)
        except Exception as exc:  # noqa: BLE001 — last-good: any scan failure keeps old registry
            self._note_rescan_failure(exc)
            return

        if signature == self._skills_fs_signature:
            return

        # Something changed. The strict discovery below raises on any
        # per-file read failure (``discover_skills`` would skip and warn,
        # silently shrinking the catalog on a transient mid-edit error), so
        # a failed rebuild attempt never reaches the registry.
        try:
            self._discover_skills_strict(config.agent.skills_dirs)
        except Exception as exc:  # noqa: BLE001 — last-good: any failure keeps old registry
            self._note_rescan_failure(exc)
            return

        try:
            self._rebuild_skills(config)
        except Exception as exc:  # noqa: BLE001 — _skill_swap restores the previous registry
            self._note_rescan_failure(exc)
            return

        self._skills_fs_signature = signature
        if self._skills_rescan_error is not None:
            logger.info("skills_dirs rescans recovered")
            self._skills_rescan_error = None

    def _discover_skills_strict(self, skills_dirs: list[str]) -> list:
        """Discover every skill in *skills_dirs*, raising on any file failure.

        Mirrors ``skills.core.discover_skills_in_dir`` — same two layouts,
        same ordering, same parser — but a file that cannot be read raises
        instead of being skipped and warned about, so the caller's last-good
        path owns the failure. This is the "build the replacement contents
        before clearing" half of that contract, kept entirely inside the
        Runtime.
        """
        replacement: list = []
        for dir_raw in skills_dirs:
            root = self._resolve_skills_dir(dir_raw)
            if root is None:
                # Unresolvable (${VAR} unset): core skips it with a warning,
                # so the rebuilt catalog matches without it. Not a failure.
                continue
            names = sorted(os.listdir(root))
            # Flat form first, then bundled — the order discover_skills uses
            # so last-wins registration stays deterministic.
            for name in names:
                if name.endswith(".md") and (root / name).is_file():
                    skill = self._load_skill_file_strict(root / name, None)
                    if skill is not None:
                        replacement.append(skill)
            for name in names:
                child = root / name
                if child.is_dir():
                    bundled = child / "SKILL.md"
                    if bundled.is_file():
                        skill = self._load_skill_file_strict(bundled, name)
                        if skill is not None:
                            replacement.append(skill)
        return replacement

    @staticmethod
    def _load_skill_file_strict(md_file: Path, fallback_name: str | None):
        """Read and parse one skill file, raising on failure.

        Same read and parser as ``discover_skills_in_dir`` — except a file
        that cannot be read propagates instead of being skipped, which is
        the whole difference the last-good contract needs. A file the parser
        legitimately skips (no usable frontmatter in the flat form) yields
        None and is dropped by the caller.
        """
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        skill = _parse_skill_frontmatter(
            md_file.read_text(encoding="utf-8"),
            md_file,
            fallback_name=fallback_name,
        )
        return skill

    async def initialize(self) -> None:
        """Connect all MCP clients and register their tool bridges into the registry.

        Also connects the persistent provenance store if configured.
        """
        await self._session_logs.start()

        # Connect persistent provenance store (before MCP so tool bridges
        # can reference it if needed).
        if self._provenance_store_path is not None:
            from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

            self._provenance_store = SQLiteProvenanceStore(
                db_path=self._provenance_store_path
            )
            await self._provenance_store.connect()
            logger.info(
                "Connected persistent provenance store: %s",
                self._provenance_store_path,
            )

        for client in self._mcp_clients:
            try:
                bridges = await client.connect()
                for bridge in bridges:
                    self.registry.register(
                        name=bridge.name,
                        description=bridge.description,
                        input_schema=bridge.input_schema,
                        fn=bridge,
                    )
            except MCPConnectionError as exc:
                logger.warning("MCP server failed to connect: %s", exc)

    def new_session(
        self,
        frame_id: str,
        tracer: "Tracer | None" = None,
    ) -> Session:
        """Create a new per-frame Session with its own SR2 instance.

        The Session shares the Runtime's tool registry, LLM callable, and
        pipeline config, but has independent history and serialization.

        When a PlanResolver is configured, the active_frame_provider is wired
        through so SR2 stamps block.meta["frame"] with the current task's frame
        id, enabling step-compaction to burn completed step context.

        The shared provenance store (if initialized) is threaded through so
        all sessions write pipeline provenance to the same persistent store.
        The shared in-memory memory store is threaded through so the memory
        resolver/transformer read and write the same store across frames.
        """
        session = Session(
            frame_id=frame_id,
            config=self.config,
            llm=self.llm,
            registry=self.registry,
            tracer=tracer,
            active_frame_provider=self._active_frame_provider,
            provenance_store=self._provenance_store,
            memory_store=self._memory_store,
            log_manager=self._session_logs if self._session_logs.enabled else None,
        )
        self._sessions.add(session)
        return session

    async def aclose(self) -> None:
        """Close all MCP client transports and the provenance store.

        Safe to call even if initialize() was never called.
        """
        await self._session_logs.aclose()
        if self._provenance_store is not None:
            await self._provenance_store.close()
            self._provenance_store = None
        # Close the memory store if the selected backend is closeable
        # (PostgresMemoryStore.close() is synchronous; InMemoryMemoryStore has
        # no close() and is skipped).
        close = getattr(self._memory_store, "close", None)
        if callable(close):
            close()
        # Best-effort teardown: a single MCP client failing to close must not
        # strand the others or escape this coroutine. CancelledError (a
        # BaseException, NOT caught by `except Exception`) is caught explicitly
        # because httpcore/anyio can raise it while tearing down an HTTP
        # transport during an otherwise-clean shutdown — that race gave the
        # process exit code 1 even after the task succeeded (spc-63).
        for client in self._mcp_clients:
            try:
                await client.close()
            except (Exception, asyncio.CancelledError) as exc:
                logger.warning(
                    "MCP client close failed during shutdown (ignored): %s", exc
                )

    # ------------------------------------------------------------------
    # Auto-injection helpers
    # ------------------------------------------------------------------

    def _auto_inject_complete_step(self, config: SpectreConfig) -> None:
        """Auto-register complete_step if a plan resolver is in the pipeline.

        Scans all pipeline layers for a resolver with type=='plan'. If found,
        extracts the plans_root from the resolver's config and registers
        CompleteStepTool with that plans_root.

        Skips registration if complete_step is already in the registry
        (explicitly declared in agent.tools).
        """
        if "complete_step" in self.registry:
            # Already registered explicitly — don't duplicate.
            return

        plans_root = self._find_plans_root(config)
        # _find_plans_root returns None when no plan resolver exists.
        # When a plan resolver exists but plans_root isn't set explicitly,
        # we pass None to CompleteStepTool so it uses its own default (~/.sr2/plans).

        has_plan_resolver = False
        for layer in config.pipeline.layers:
            for resolver in layer.resolvers:
                if resolver.type == "plan":
                    has_plan_resolver = True
                    break
            if has_plan_resolver:
                break

        if not has_plan_resolver:
            return

        # Import locally to avoid circular dependency at module level.
        from sr2_spectre.tools.builtins.complete_step import CompleteStepTool

        self.registry.register(
            name=CompleteStepTool.name,
            description=CompleteStepTool.description,
            input_schema=CompleteStepTool.input_schema,
            fn=CompleteStepTool(plans_root=plans_root).__call__,
        )
        logger.info(
            "Auto-injected complete_step tool (plans_root=%s)",
            plans_root or "~/.sr2/plans (default)",
        )

    @staticmethod
    def _find_plan_resolver_config(
        config: SpectreConfig,
    ) -> "ResolverConfig | None":
        """Scan pipeline layers for a plan resolver and return its ResolverConfig.

        Returns the first ResolverConfig with type=='plan', or None.
        Used to build the PlanResolver instance for active_frame_provider wiring.
        """
        from sr2.config.models import ResolverConfig

        for layer in config.pipeline.layers:
            for resolver in layer.resolvers:
                if resolver.type == "plan":
                    return resolver
        return None

    @staticmethod
    def _find_plans_root(config: SpectreConfig) -> str | None:
        """Scan pipeline layers for a plan resolver and return its plans_root.

        Walks pipeline.layers[*].resolvers[*] looking for type=='plan'.
        Returns the plans_root from the first match's config dict, or None
        if no plan resolver is found or plans_root isn't explicitly set.
        """
        layers = config.pipeline.layers
        for layer in layers:
            for resolver in layer.resolvers:
                if resolver.type == "plan":
                    return resolver.config.get("plans_root")
        return None

    @staticmethod
    def _find_step_compaction_transformer(config: SpectreConfig) -> bool:
        """Return True if any pipeline layer declares a step_compaction transformer.

        Walks pipeline.layers[*].transformers[*] looking for type=='step_compaction'.
        """
        for layer in config.pipeline.layers:
            if layer.transformers:
                for transformer in layer.transformers:
                    if transformer.type == "step_compaction":
                        return True
        return False

    def _check_step_compaction_config(self, config: SpectreConfig) -> None:
        """Warn at startup if a plan resolver exists but no step_compaction transformer is declared.

        A PlanResolver stamps block.meta['frame'] on every block, enabling
        step-compaction to burn completed frame context. Without the
        transformer, stamped frames accumulate unbounded — the spc-3 failure
        mode. This warning catches the half-configured state early.
        """
        if self._find_step_compaction_transformer(config):
            return
        logger.warning(
            "PlanResolver is configured but no step_compaction transformer was found "
            "in any pipeline layer. Block frames will accumulate unbounded. "
            "Add a step_compaction transformer to a pipeline layer (see "
            "sr2_spectre.planning.transformer:StepCompactionTransformer)."
        )

    @staticmethod
    def _resolve_provenance_path(config: SpectreConfig) -> str | None:
        """Resolve the provenance store path from config.

        Returns:
            Absolute path string if persistent provenance is enabled,
            None if disabled (empty string in config) or not set.
        """
        raw = config.provenance_store_path
        if raw is None:
            # Default: ~/.sr2-spectre/provenance.db
            return str(Path.home() / ".sr2-spectre" / "provenance.db")
        if raw == "":
            # Explicitly disabled
            return None
        # User-provided path — resolve ~ and variables
        return str(Path(raw).expanduser().resolve())

    @staticmethod
    def _resolve_memory_dsn(config: SpectreConfig) -> str | None:
        """Resolve the memory store DSN from config, falling back to env.

        Precedence:
          * config.memory_store_dsn is authoritative when not None. An empty
            string means explicitly disabled -> None (in-memory), even if the
            env var is set.
          * When config.memory_store_dsn is None, fall back to the
            SPECTRE_MEMORY_DSN environment variable (non-empty).
          * Otherwise None -> in-memory store.

        Returns:
            A DSN string if the persistent Postgres store is selected, else None.
        """
        raw = config.memory_store_dsn
        if raw is not None:
            # Explicit config value wins. "" == disabled.
            return raw or None
        env = os.environ.get("SPECTRE_MEMORY_DSN")
        return env or None

    def _bootstrap_skills(self, config: SpectreConfig) -> list:
        """Build the full three-tier skill catalog for *config* WITHOUT registering.

        Returns the skill list in registration order — the caller
        (``_rebuild_skills``) registers it, so the registry is a pure
        function of config + filesystem and both rebuild paths (startup and
        per-message rescan) produce identical catalogs.

        The tiers and their order: builtin DEFAULT_SKILLS (unless
        ``agent.default_skills`` is false), then ``agent.skills_dirs[]``
        discovery, then ``agent.skills[]`` per-file loads. Later
        registrations win, so the order is deliberate: builtins are the
        weakest claim on a name and an explicit per-file skill is the
        strongest.
        """
        skills: list = []

        # 1. Builtin defaults, unless the agent opted out.
        #
        # Opt-out exists because these describe SR2's own architecture and
        # file layout. An agent facing people who are not the operator has no
        # use for them and should not be able to read them out (obsidian-4z94).
        if config.agent.default_skills:
            skills.extend(DEFAULT_SKILLS)
            logger.info("Registered %d default skill(s)", len(DEFAULT_SKILLS))
        else:
            logger.info(
                "Builtin default skills withheld (agent.default_skills=false)"
            )

        # 2. Discover skills from skills_dirs (bulk loading)
        if config.agent.skills_dirs:
            discovered = discover_skills(config.agent.skills_dirs)
            skills.extend(discovered)
            for skill in discovered:
                logger.info(
                    "Discovered skill '%s' (v%s) from skills_dirs",
                    skill.name,
                    skill.version,
                )
            logger.info(
                "Discovered %d skill(s) from skills_dirs (%s)",
                len(discovered),
                ", ".join(config.agent.skills_dirs),
            )

        # 3. Load config-declared skill files (per-file override path)
        for skill_cfg in config.agent.skills:
            try:
                skill = load_skill_from_path(
                    name=skill_cfg.name,
                    path=skill_cfg.path,
                    version=skill_cfg.version,
                    description=skill_cfg.description,
                    tags=skill_cfg.tags,
                )
            except FileNotFoundError:
                logger.warning(
                    "Skill file not found: '%s' at %s — skipping",
                    skill_cfg.name,
                    skill_cfg.path,
                )
                continue
            except ConfigPathError as exc:
                logger.warning(
                    "Skill path unresolvable: '%s' at %s — skipping (%s)",
                    skill_cfg.name,
                    skill_cfg.path,
                    exc,
                )
                continue
            skills.append(skill)
            logger.info("Loaded skill '%s' from %s", skill_cfg.name, skill_cfg.path)

        return skills

    def _auto_inject_load_skill(self) -> None:
        """Register or withdraw the load_skill tool to match the registry.

        The tool is the runtime entry point for the skills subsystem, so it
        follows the skills: an agent with none gets a tool whose only possible
        answer is "No skills registered.", which is schema tokens spent on
        nothing and a menu entry inviting a pointless call.

        Withdrawal matters as much as injection. A config reload can empty the
        registry — ``agent.default_skills`` flipped to false, a skills_dirs
        entry removed — and the tool has to go with it, or a guest-facing
        agent keeps a live handle to skills it is no longer meant to have.
        """
        if not len(self.skill_registry):
            if self.registry.unregister("load_skill"):
                logger.info("Withdrew load_skill tool (no skills registered)")
            return

        if "load_skill" in self.registry:
            return

        # Import locally to avoid circular dependency at module level.
        from sr2_spectre.tools.builtins.load_skill import LoadSkillTool

        self.registry.register(
            name=LoadSkillTool.name,
            description=LoadSkillTool.description,
            input_schema=LoadSkillTool.input_schema,
            fn=LoadSkillTool(registry=self.skill_registry).__call__,
        )
        logger.info(
            "Auto-injected load_skill tool (%d skill(s) available)",
            len(self.skill_registry),
        )
