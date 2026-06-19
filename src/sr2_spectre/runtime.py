"""Runtime — shared sub-runtime for all frames.

Holds config, LLM callable, MCP clients, tool registry, and shared stores.
One Runtime instance serves N per-frame Sessions, each with its own SR2.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sr2.pipeline.provenance import ProvenanceStore
    from sr2.pipeline.tracing import Tracer

from sr2.integrations.litellm import LiteLLMCallable
from sr2.memory import InMemoryMemoryStore, PostgresMemoryStore
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2_spectre.config import SpectreConfig
from sr2_spectre.mcp.client import MCPClient, MCPConnectionError
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
    - LiteLLMCallable (one LLM path)
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

        # Register tools from config, injecting workspace_root ONLY into tools
        # whose constructor accepts it (FileWriteTool, EditTool, TerminalTool).
        # Blanket injection crashes tools that don't take the kwarg (e.g.
        # FileReadTool) the moment SR2_WORKSPACE is set.
        for tool_cfg in config.agent.tools:
            tool_config = dict(tool_cfg.config)
            if (
                self.workspace_root is not None
                and "workspace_root" not in tool_config
                and _tool_accepts_workspace_root(tool_cfg.class_path)
            ):
                tool_config["workspace_root"] = self.workspace_root
            self.registry.register_from_class_path(tool_cfg.class_path, tool_config)

        # Auto-inject complete_step when a plan resolver is configured.
        # complete_step + plan resolver are one feature unit: if the pipeline
        # declares a plan resolver, the agent needs complete_step to mark
        # tasks done. Scanning pipeline.layers for type=='plan' gives us the
        # resolver's plans_root so there's a single source of truth.
        self._auto_inject_complete_step(config)

        # Bootstrap SkillRegistry: builtin DEFAULT_SKILLS + config-declared files.
        self.skill_registry = SkillRegistry()
        self._bootstrap_skills(config)

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

        # Build LLM callable — forward per-agent sampling params
        model_cfg = config.models["default"]
        llm_kwargs: dict = {
            "model": model_cfg.model,
            "base_url": model_cfg.base_url,
        }
        if model_cfg.api_key:
            llm_kwargs["api_key"] = model_cfg.api_key
        if model_cfg.params:
            llm_kwargs.update(model_cfg.params)
        self.llm = LiteLLMCallable(**llm_kwargs)

        # Build PlanResolver instance to wire active_frame_provider into SR2.
        # The PlanResolver.current_frame_id() returns the active frame for the
        # lowest-order pending task. We wrap it as a Callable[[str], str | None]
        # to match the active_frame_provider signature (origin parameter ignored).
        self._active_frame_provider: Callable[[str], str | None] | None = None
        self._plan_resolver_config = self._find_plan_resolver_config(config)
        if self._plan_resolver_config:
            from sr2_spectre.planning import PlanResolver

            self._plan_resolver = PlanResolver(self._plan_resolver_config)

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

    async def initialize(self) -> None:
        """Connect all MCP clients and register their tool bridges into the registry.

        Also connects the persistent provenance store if configured.
        """
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
        return Session(
            frame_id=frame_id,
            config=self.config,
            llm=self.llm,
            registry=self.registry,
            tracer=tracer,
            active_frame_provider=self._active_frame_provider,
            provenance_store=self._provenance_store,
            memory_store=self._memory_store,
        )

    async def aclose(self) -> None:
        """Close all MCP client transports and the provenance store.

        Safe to call even if initialize() was never called.
        """
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

    def _bootstrap_skills(self, config: SpectreConfig) -> None:
        """Bootstrap the SkillRegistry with DEFAULT_SKILLS + config-declared skills.

        Always registers the builtin DEFAULT_SKILLS (sr2-conventions). Then
        loads any additional skills declared in agent.skills[] from disk
        using load_skill_from_path.

        Also discovers skills from agent.skills_dirs[] via directory scanning
        with frontmatter parsing.

        Auto-injects the load_skill tool so the agent can discover and load
        skills at runtime.
        """
        # 1. Register builtin defaults
        for skill in DEFAULT_SKILLS:
            self.skill_registry.register(skill)
        logger.info("Registered %d default skill(s)", len(DEFAULT_SKILLS))

        # 2. Discover skills from skills_dirs (bulk loading)
        if config.agent.skills_dirs:
            discovered = discover_skills(config.agent.skills_dirs)
            for skill in discovered:
                self.skill_registry.register(skill)
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
                self.skill_registry.register(skill)
                logger.info("Loaded skill '%s' from %s", skill_cfg.name, skill_cfg.path)
            except FileNotFoundError:
                logger.warning(
                    "Skill file not found: '%s' at %s — skipping",
                    skill_cfg.name,
                    skill_cfg.path,
                )

        # 4. Auto-inject load_skill tool
        self._auto_inject_load_skill()

    def _auto_inject_load_skill(self) -> None:
        """Auto-register the load_skill tool if not already present.

        The load_skill tool is always available — it's the runtime entry
        point for the skills subsystem. Unlike complete_step (which is
        conditional on a plan resolver), skills are always useful.
        """
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
