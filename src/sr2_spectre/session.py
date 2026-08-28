"""Session — per-frame conversation state.

Each Session owns an SR2 instance (session_id = frame_id), its own history,
and a per-frame asyncio.Lock serializing turns. The SR2 shares the Runtime's
tool registry and LLM callable but maintains independent conversation state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

if TYPE_CHECKING:
    from sr2.memory import MemoryStore
    from sr2.pipeline.provenance import ProvenanceStore
    from sr2.pipeline.tracing import Tracer
    from sr2.protocols.llm import LLMCallable

from sr2.config.models import ToolLoopLimitError
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.token_counting import CharacterTokenCounter

from sr2_spectre.config import SpectreConfig
from sr2_spectre.core import RunContext, TurnResult
from sr2_spectre.events import (
    AgentDone,
    AgentEvent,
    AgentThinkingDelta,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
    AgentUsage,
)
from sr2_spectre.tools.output import ToolOutput
from sr2_spectre.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Session:
    """Per-frame conversation state.

    Owns:
    - frame_id: stable identity (= SR2 session_id)
    - SR2 instance: constructed per-frame with shared provenance injected
    - history: list[Message] — only this frame's transcript
    - _lock: asyncio.Lock — serializes turns within this frame
    """

    def __init__(
        self,
        frame_id: str,
        config: SpectreConfig,
        llm: "LLMCallable",
        registry: ToolRegistry,
        tracer: "Tracer | None" = None,
        active_frame_provider: Callable[[str], str | None] | None = None,
        provenance_store: "ProvenanceStore | None" = None,
        memory_store: "MemoryStore | None" = None,
    ) -> None:
        self.frame_id = frame_id
        self.config = config
        self._registry = registry
        self._token_counter = CharacterTokenCounter()
        self.history: list[Message] = []
        self._lock = asyncio.Lock()

        # Kept so the SR2 can be rebuilt when a config reload changes the
        # pipeline. Everything here is either shared process-wide (stores, the
        # LLM handle) or fixed for this frame (tracer, providers), so a rebuild
        # only re-reads the pipeline.
        self._llm = llm
        self._tracer = tracer
        self._active_frame_provider = active_frame_provider
        self._provenance_store = provenance_store
        self._memory_store = memory_store
        # Set by apply_config(); consumed at the top of the next turn, under
        # the lock, so an in-flight reply never has the SR2 swapped underneath
        # it (its tool executor publishes onto sr2.bus while it runs).
        self._sr2_stale = False

        # Run context — set by the Interface at start(); None until then.
        self._run_context: RunContext | None = None

        # Build the run_context_provider callback that reads self._run_context
        # at resolve time (not at construction time).  SR2 stores the callable
        # and passes it to resolvers via Dependencies.run_context_provider.
        def _run_context_provider() -> dict[str, str] | None:
            ctx = self._run_context
            if ctx is None:
                return None
            out = {
                "mode": ctx.mode,
                "source": ctx.source or "",
            }
            if ctx.area is not None:
                out["area"] = ctx.area
            return out

        self._run_context_provider = _run_context_provider
        self.sr2 = self._build_sr2()

    def _build_sr2(self) -> SR2:
        """Construct an SR2 for this frame from the config in force.

        SR2 owns context compilation, tool definition injection, and LLM calls.
        When a shared provenance_store is provided (from Runtime), all sessions
        write pipeline provenance to the same persistent store. The shared
        memory_store (when provided) backs the memory resolver/transformer so
        agents accrue cross-session memory within the process.

        Safe to call more than once: spectre owns the authoritative history in
        ``self.history`` and re-seeds SR2 from it at the top of every turn, so
        a fresh SR2 loses no conversation state.
        """
        return SR2(
            pipeline_config=self.config.pipeline,
            llm={"default": self._llm},
            token_counter=self._token_counter,
            session_id=self.frame_id,
            tool_source=self._registry,
            tracer=self._tracer,
            tool_executor=self._execute_tool,
            active_frame_provider=self._active_frame_provider,
            run_context_provider=self._run_context_provider,
            provenance_store=self._provenance_store,
            memory_store=self._memory_store,
        )

    def apply_config(
        self,
        config: SpectreConfig,
        rebuild_sr2: bool = False,
        active_frame_provider: Callable[[str], str | None] | None = None,
    ) -> None:
        """Adopt a reloaded config for the next turn.

        Args:
            config: The config now in force process-wide.
            rebuild_sr2: True when the pipeline changed, so this frame's SR2
                has to be rebuilt. The rebuild is deferred to the next turn
                rather than done here — see ``_sr2_stale``.
            active_frame_provider: The provider the Runtime holds now. A
                pipeline edit can add or remove the plan resolver this comes
                from, so it is re-supplied alongside the rebuild instead of
                being frozen at construction.
        """
        self.config = config
        if rebuild_sr2:
            self._active_frame_provider = active_frame_provider
            self._sr2_stale = True

    def _refresh_sr2_if_stale(self) -> None:
        """Rebuild the SR2 if a config reload invalidated it. Call under lock."""
        if not self._sr2_stale:
            return
        self.sr2 = self._build_sr2()
        self._sr2_stale = False
        logger.info("Rebuilt SR2 for frame '%s' — pipeline changed", self.frame_id)

    @property
    def run_context(self) -> RunContext | None:
        """Return the run context set by the Interface, or None."""
        return self._run_context

    def set_run_context(self, ctx: RunContext) -> None:
        """Set the run context. Called by the Interface during start()."""
        self._run_context = ctx

    async def _execute_tool(self, block: ToolUseBlock) -> ToolResultBlock:
        """SR2 tool_executor callback. Executes a tool via the shared registry.

        Truncates oversized results before they enter context.
        Dispatches post-execute bus events declared in ``ToolOutput`` wrappers.
        """
        max_bytes = self.config.agent.tool_result_max_bytes

        def _truncate(content: str, name: str) -> str:
            if len(content) <= max_bytes:
                return content
            truncated = content[:max_bytes]
            return (
                f"{truncated}\n\n"
                f"[TRUNCATED: output exceeded {max_bytes} bytes "
                f"(original size: {len(content)} bytes, tool: {name})]"
            )

        try:
            out = await self._registry.execute(block.name, block.input)

            # Check for post-execute events (generic dispatch — no name-magic)
            events_to_dispatch: list[Any] = []
            if isinstance(out, ToolOutput):
                events_to_dispatch = out.events
                out = out.result

            content = _truncate(str(out), block.name)
            result = ToolResultBlock(tool_use_id=block.id, content=content)

            # Dispatch any post-execute events declared by the tool
            for pe_event in events_to_dispatch:
                self.sr2.bus.queue(
                    Event(
                        name=pe_event.event_name,
                        phase=getattr(
                            EventPhase,
                            pe_event.phase.upper(),
                            EventPhase.COMPLETED,
                        ),
                        source_layer=pe_event.source_layer,
                        data=pe_event.data,
                    )
                )

            return result
        except Exception as exc:
            logger.warning("Tool %r failed: %s", block.name, exc)
            content = _truncate(f"ERROR: {exc}", block.name)
            return ToolResultBlock(tool_use_id=block.id, content=content, is_error=True)

    async def stream_message(self, text: str) -> AsyncIterator[AgentEvent]:
        """Stream agent events for a user message, serialized by _lock."""
        async with self._lock:
            self._refresh_sr2_if_stale()
            self.history.append(Message(role="user", content=[TextBlock(text=text)]))

            prior = self.history[:-1]
            increment = self.history[-1].content
            self.sr2.seed_session(prior)

            text_acc: list[str] = []
            thinking_acc: list[str] = []
            total_tool_calls = 0
            tool_id_to_name: dict[str, str] = {}
            turn_input_tokens = 0
            turn_output_tokens = 0
            # Local estimate of the context size at the start of this turn —
            # available regardless of whether the backend reports usage.
            context_tokens = sum(
                self._token_counter.count(msg.content) for msg in prior
            )

            try:
                async for ev in self.sr2.turn(user_input=increment):
                    if ev.type == "text" and ev.text:
                        text_acc.append(ev.text)
                        yield AgentTextDelta(text=ev.text)
                    elif ev.type == "thinking" and ev.text:
                        thinking_acc.append(ev.text)
                        yield AgentThinkingDelta(text=ev.text)
                    elif ev.type == "usage" and ev.usage is not None:
                        turn_input_tokens += ev.usage.input_tokens
                        turn_output_tokens += ev.usage.output_tokens
                        yield AgentUsage(
                            input_tokens=ev.usage.input_tokens,
                            output_tokens=ev.usage.output_tokens,
                        )
                    elif ev.type == "tool_use_emitted" and ev.tool_uses:
                        for tu in ev.tool_uses:
                            total_tool_calls += 1
                            tool_id_to_name[tu.id] = tu.name
                            yield AgentToolStart(
                                tool_id=tu.id, name=tu.name, input=tu.input
                            )
                    elif ev.type == "tool_result_received" and ev.tool_results:
                        for tr in ev.tool_results:
                            yield AgentToolResult(
                                tool_id=tr.tool_use_id,
                                name=tool_id_to_name.get(tr.tool_use_id, ""),
                                content=tr.content,
                                is_error=getattr(tr, "is_error", False),
                            )
            except ToolLoopLimitError:
                notice = "Tool iteration limit reached; stopping."
                text_acc.append(notice)
                yield AgentTextDelta(text=notice)

            last_text = "".join(text_acc)
            assistant_content = [TextBlock(text=last_text)] if last_text else []
            self.history.append(Message(role="assistant", content=assistant_content))

            logger.debug(
                "Turn complete, %d tool calls",
                total_tool_calls,
            )
            yield AgentDone(
                tool_calls_executed=total_tool_calls,
                input_tokens=turn_input_tokens,
                output_tokens=turn_output_tokens,
                context_tokens=context_tokens,
            )

    async def handle_user_message(self, text: str) -> TurnResult:
        """Process a user message and return a TurnResult."""
        text_parts: list[str] = []
        total = 0
        tokens = 0
        async for ev in self.stream_message(text):
            if isinstance(ev, AgentTextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, AgentDone):
                total = ev.tool_calls_executed
                tokens = ev.input_tokens + ev.output_tokens
        return TurnResult(
            text="".join(text_parts),
            tool_calls_executed=total,
            total_tokens=tokens,
        )
