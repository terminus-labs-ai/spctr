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

        # session_id is the frame_id for single-frame operation
        frame_id = session_id or f"{config.agent.name}-default"
        self._session = self._runtime.new_session(frame_id=frame_id, tracer=tracer)

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
        self._session = self._runtime.new_session(frame_id=value)

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

    def new_session(self, session_id: str | None = None) -> None:
        """Reset conversation history and start a new session.

        Creates a new Session under the same Runtime (shared config,
        LLM, MCP, tool registry) with fresh history.
        """
        frame_id = session_id or f"{self._config.agent.name}-default"
        self._session = self._runtime.new_session(frame_id=frame_id)
