"""Agent — SR2-powered spectre agent (backward-compat facade).

The Agent is now a thin facade over Runtime + Session. It maintains full
backward compatibility: all attributes (session_id, history, registry, sr2)
and methods (stream_message, handle_user_message, new_session, register_tool,
initialize, aclose) work identically.

Internally the Agent:
- Constructs a Runtime (shared config, LLM, MCP clients, tool registry)
- Creates a single Session (frame_id = session_id)
- Delegates all conversation operations to the Session

This design enables future multi-frame operation (spc-13) where N Sessions
share one Runtime. The single-frame path remains behaviorally identical.

Owns (via delegation):
- Session identity and authoritative conversation history (list[Message])
- Tool registry (execution side — provides tool_executor callback to SR2)
- SR2 instance (context compilation, LLM call, token budgets)

SR2 is invoked as a stateless context compiler + streamer.
SR2 drives the tool execution loop internally via the tool_executor callback;
spectre owns all conversation state and history.

Design: spectre owns history; SR2 owns the tool loop.
Each round: seed prior history → turn(increment) → SR2 loops internally
→ SR2 yields stream events (text, tool_use_emitted, tool_result_received)
→ spectre translates to AgentEvents → Interface renders.
The tool-loop limit is owned and enforced by SR2 (pipeline.max_tool_iterations);
spectre catches SR2's ToolLoopLimitError and stops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from sr2.pipeline.tracing import Tracer

    from sr2_spectre.session import Session

from sr2_spectre.config import SpectreConfig
from sr2_spectre.core import RunContext, TurnResult
from sr2_spectre.events import AgentEvent
from sr2_spectre.runtime import Runtime

# ---------------------------------------------------------------------------
# Backward-compat aliases — existing code imports these from agent.py
# ---------------------------------------------------------------------------

# session_id → frame_id mapping: Agent's session_id *is* the frame_id
# for the single-frame path.


class Agent:
    """SR2 Spectre agent — backward-compatible facade over Runtime + Session.

    Maintains identical external API to the pre-split Agent. All attributes
    and methods delegate to the internal Runtime and Session instances.

    For backward compatibility the following attributes remain accessible:
    - session_id: str (the frame_id of the internal Session)
    - history: list[Message] (delegated to Session.history)
    - registry: ToolRegistry (delegated to Runtime.registry)
    - sr2: SR2 (delegated to Session.sr2)
    """

    def __init__(
        self,
        config: SpectreConfig,
        session_id: str | None = None,
        tracer: "Tracer | None" = None,
    ) -> None:
        self._config = config
        self._runtime = Runtime(config)
        self._tracer = tracer

        # session_id is the frame_id for single-frame operation
        frame_id = session_id or f"{config.agent.name}-default"
        self._session = self._runtime.new_session(
            frame_id=frame_id, tracer=self._tracer
        )

    # ---- Backward-compat property accessors ----

    @property
    def config(self) -> SpectreConfig:
        """DEPRECATED — use _config. Kept for backward compatibility."""
        return self._config

    @property
    def session_id(self) -> str:
        """Session identity (= frame_id for single-frame path)."""
        return self._session.frame_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Allow setting session_id (used by new_session())."""
        # For backward compat, we recreate the session under the Runtime
        self._replace_session(value)

    @property
    def history(self) -> list:
        """Conversation history (delegated to Session)."""
        return self._session.history

    @property
    def registry(self) -> Any:
        """Tool registry (delegated to Runtime)."""
        return self._runtime.registry

    @property
    def sr2(self) -> Any:
        """SR2 instance (delegated to Session)."""
        return self._session.sr2

    @property
    def _execute_tool(self) -> Any:
        """Tool executor callback (delegated to Session). Kept for test backward compat."""
        return self._session._execute_tool

    @property
    def provenance_store(self) -> Any:
        """The Runtime's shared provenance store, or None when unconfigured (spc-132).

        Read-only passthrough — the store belongs to the Runtime. Interfaces
        read it to list archived sessions (``list_sessions``) for a resume
        picker; ``None`` before ``initialize()`` or with provenance disabled.
        """
        return self._runtime.provenance_store

    async def restore_history(self) -> Any:
        """Restore the facade's Session from its archive (delegated to Session).

        Passthrough to ``Session.restore_history``: rebuilds this frame's
        resumable history from the provenance archive before the first turn,
        returning the ``ResumeResult`` (history, interrupted, restore_failed).
        Used by an interface adopting a key mid-process — the identity switch
        comes first, so this restores the *adopted* key's archive (spc-132).
        """
        return await self._session.restore_history()

    @property
    def reminders(self) -> Any:
        """The Runtime's reminders handle (None when disabled, FR6).

        The owning interface reads this in ``start()`` to launch the
        delivery loop with its own send callable, and in ``stop()`` to end
        it. None while ``agent.reminders.enabled`` is false — nothing is
        registered and no loop exists.
        """
        return self._runtime.reminders

    def set_reminders_channel_resolver(self, resolver: Any) -> None:
        """Register the owning interface's named-channel resolver (spc-159).

        Delegates to ``Runtime.set_reminders_channel_resolver``: the
        interface supplies its resolver once, and the Runtime passes it
        into every ReminderTool it builds (construction and re-seat),
        rebuilding a tool that is already seated so it carries the
        resolver too (spc-166).
        """
        self._runtime.set_reminders_channel_resolver(resolver)

    @property
    def reminders_seq(self) -> int:
        """The Runtime's reseat counter (spc-143). See Runtime.reminders_seq."""
        return self._runtime.reminders_seq

    @property
    def reminders_stamped(self) -> int:
        """The Runtime's reload stamp (spc-143).

        The owning interface compares it against the stamp recorded when
        its delivery loop started; when they differ, a reload happened and
        the loop re-asserts from the current handle (start, stop, or
        restart on fresh settings).
        """
        return self._runtime.reminders_stamped

    @property
    def schedules(self) -> Any:
        """The Runtime's schedules handle (None when disabled, FR23).

        The owning interface reads this in ``start()`` to launch the
        schedule engine with its own send callable.
        """
        return self._runtime.schedules

    @property
    def schedules_seq(self) -> int:
        """The Runtime's schedules reseat counter. See Runtime.schedules_seq."""
        return self._runtime.schedules_seq

    @property
    def schedules_stamped(self) -> int:
        """The Runtime's schedules reload stamp; see ``reminders_stamped``."""
        return self._runtime.schedules_stamped

    # ---- Run context delegation ----

    @property
    def run_context(self) -> RunContext | None:
        """Return the run context set by the Interface, or None."""
        return self._session.run_context

    def set_run_context(self, ctx: RunContext) -> None:
        """Set the run context. Called by the Interface during start()."""
        self._session.set_run_context(ctx)

    # ---- Delegated methods ----

    def apply_config(self, config: SpectreConfig) -> list[str]:
        """Adopt a reloaded config, returning the config areas actually applied.

        Interfaces that reload config while running (Discord) call this once
        per inbound message, so a corrected model, endpoint, pipeline or tool
        list takes effect on the next reply instead of the next restart.

        Returns:
            Names of the applied config areas — empty when nothing changed.
        """
        self._config = config
        return self._runtime.apply_config(config)

    async def initialize(self) -> None:
        """Connect all MCP clients and register their tool bridges."""
        await self._runtime.initialize()

    async def stream_message(self, text: str) -> AsyncIterator[AgentEvent]:
        """Stream agent events for a user message."""
        async for ev in self._session.stream_message(text):
            yield ev

    async def handle_user_message(self, text: str) -> TurnResult:
        """Process a user message through the SR2-powered tool loop."""
        return await self._session.handle_user_message(text)

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        fn: Any,
    ) -> None:
        """Register a tool at runtime (not from config)."""
        self._runtime.registry.register(name, description, input_schema, fn)

    async def aclose(self) -> None:
        """Close all MCP client transports."""
        await self._runtime.aclose()

    def create_session(self, frame_id: str) -> Session:
        """Create an independent Session under this Agent's Runtime.

        The returned Session shares the Runtime (config, LLM, MCP, tool
        registry, memory) and carries the Agent's tracer, but has its own
        history and SR2. The facade's current conversation is untouched:
        nothing here resets, replaces, or appends to it. The caller owns
        the returned Session and closes it when its conversation ends.

        Unlike new_session(), this never touches the facade's Session.
        """
        return self._runtime.new_session(frame_id=frame_id, tracer=self._tracer)

    def new_session(self, session_id: str | None = None) -> None:
        """Reset conversation history and start a new session.

        Creates a new Session under the same Runtime (shared config,
        LLM, MCP, tool registry) with fresh history.
        """
        frame_id = session_id or f"{self._config.agent.name}-default"
        self._replace_session(frame_id)

    def _replace_session(self, frame_id: str) -> None:
        """Swap in a fresh Session, carrying over the tracer and any run context.

        Re-applying the context makes the replacement open its own session log.
        """
        old = self._session
        ctx = old.run_context
        old.close()
        self._session = self._runtime.new_session(
            frame_id=frame_id, tracer=self._tracer
        )
        if ctx is not None:
            self._session.set_run_context(ctx)
