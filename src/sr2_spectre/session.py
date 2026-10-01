"""Session — per-frame conversation state.

Each Session owns an SR2 instance (session_id = frame_id), its own history,
and a per-frame asyncio.Lock serializing turns. The SR2 shares the Runtime's
tool registry and LLM callable but maintains independent conversation state.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import weakref
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
)
from sr2_spectre.live_llm import LoggedLLM
from sr2_spectre.run_log import SessionLog, SessionLogManager, SessionTracer
from sr2_spectre.tools.output import ToolOutput
from sr2_spectre.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _ms_since(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


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
        log_manager: SessionLogManager | None = None,
    ) -> None:
        self.frame_id = frame_id
        self.config = config
        self._registry = registry
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

        # Live session log: opened lazily on the first run context (which is
        # when the interface is known) and released when this Session is
        # garbage collected. None when no manager was supplied or opening failed.
        self._log_manager = log_manager
        self._log: SessionLog | None = None
        self._log_attempted = False
        # Turns between turn.start and stream_message exit (queued ones count);
        # a close() requested meanwhile is deferred until the last one ends.
        self._turns_in_flight = 0
        self._close_requested = False

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
        llm = self._llm
        tracer = self._tracer
        if self._log_manager is not None:
            # Both read self._log at call time: the log opens lazily, after
            # this SR2 is built, and a reload-driven rebuild must keep logging.
            tracer = SessionTracer(lambda: self._log, self._tracer)
            if hasattr(llm, "retarget"):
                llm = LoggedLLM(llm, lambda: self._log, lambda: self.config.active_model)
        return SR2(
            pipeline_config=self.config.pipeline,
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            session_id=self.frame_id,
            tool_source=self._registry,
            tracer=tracer,
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
        if self._log is not None:
            self._log.set_interface(ctx.interface)
        elif not self._log_attempted:
            self._open_log(ctx)

    def _open_log(self, ctx: RunContext) -> None:
        """Create this Session's log, announce its path, record session.start."""
        self._log_attempted = True
        if self._log_manager is None:
            return
        try:
            log = self._log_manager.open_session(self.frame_id, self.config.agent.name)
        except Exception as exc:
            logger.warning(
                "Session log unavailable in %s: %s", self._log_manager.directory, exc
            )
            return
        weakref.finalize(self, log.close)
        self._log = log
        log.set_interface(ctx.interface)
        log.append("session.start", {"agent": self.config.agent.name})
        print(f"Session log: {log.path}", file=sys.stderr)

    def close(self) -> None:
        """Close this Session's log; idempotent. Other Sessions are untouched.

        Closes immediately when no turn is in flight; otherwise defers until
        the last in-flight turn ends so its remaining events are not dropped.
        """
        self._close_requested = True
        if self._turns_in_flight == 0:
            self._close_log_now()

    def _close_log_now(self) -> None:
        if self._log is not None:
            self._log.close()

    def _log_event(self, event: str, data: dict[str, Any]) -> None:
        if self._log is not None:
            self._log.append(event, data)

    async def _execute_tool(self, block: ToolUseBlock) -> ToolResultBlock:
        """SR2 tool_executor callback: run the tool, logging actual start/end."""
        self._log_event(
            "tool.start",
            {"tool_use_id": block.id, "name": block.name, "arguments": block.input},
        )
        started = time.monotonic()
        result, original_bytes = await self._run_tool(block)
        self._log_event(
            "tool.complete",
            {
                "tool_use_id": block.id,
                "name": block.name,
                "is_error": bool(result.is_error),
                "result_bytes": original_bytes,
                "result": str(result.content),
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
            },
        )
        return result

    async def _run_tool(self, block: ToolUseBlock) -> tuple[ToolResultBlock, int]:
        """Execute a tool via the shared registry.

        Truncates oversized results before they enter context.
        Dispatches post-execute bus events declared in ``ToolOutput`` wrappers.
        Returns the result and the original (pre-truncation) UTF-8 size in bytes.
        """
        max_bytes = self.config.agent.tool_result_max_bytes
        original_bytes = 0

        def _truncate(content: str, name: str) -> str:
            nonlocal original_bytes
            encoded = content.encode("utf-8")
            original_bytes = len(encoded)
            if original_bytes <= max_bytes:
                return content
            truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
            return (
                f"{truncated}\n\n"
                f"[TRUNCATED: output exceeded {max_bytes} bytes "
                f"(original size: {original_bytes} bytes, tool: {name})]"
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

            return result, original_bytes
        except Exception as exc:
            logger.warning("Tool %r failed: %s", block.name, exc)
            content = _truncate(f"ERROR: {exc}", block.name)
            return (
                ToolResultBlock(tool_use_id=block.id, content=content, is_error=True),
                original_bytes,
            )

    async def stream_message(self, text: str) -> AsyncIterator[AgentEvent]:
        """Stream agent events for a user message, logging turn boundaries."""
        self._log_event("turn.start", {"input": text})
        self._turns_in_flight += 1
        try:
            started = time.monotonic()
            tool_calls = 0
            try:
                async for ev in self._stream_turn(text):
                    if isinstance(ev, AgentDone):
                        tool_calls = ev.tool_calls_executed
                    yield ev
            except asyncio.CancelledError:
                self._log_event("turn.cancel", {"duration_ms": _ms_since(started)})
                raise
            except Exception as exc:
                self._log_event(
                    "turn.error",
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "duration_ms": _ms_since(started),
                    },
                )
                raise
            self._log_event(
                "turn.complete",
                {"duration_ms": _ms_since(started), "tool_calls": tool_calls},
            )
        finally:
            self._turns_in_flight -= 1
            if self._close_requested and self._turns_in_flight == 0:
                self._close_log_now()

    async def _stream_turn(self, text: str) -> AsyncIterator[AgentEvent]:
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

            try:
                async for ev in self.sr2.turn(user_input=increment):
                    if ev.type == "text" and ev.text:
                        text_acc.append(ev.text)
                        yield AgentTextDelta(text=ev.text)
                    elif ev.type == "thinking" and ev.text:
                        thinking_acc.append(ev.text)
                        yield AgentThinkingDelta(text=ev.text)
                    elif ev.type == "retry":
                        self._log_event(
                            "sr2.retry",
                            {**(ev.meta or {}), "iteration": ev.iteration},
                        )
                    elif ev.type == "error" and ev.errors:
                        self._log_event(
                            "pipeline.error",
                            {"errors": list(ev.errors), "iteration": ev.iteration},
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
            yield AgentDone(tool_calls_executed=total_tool_calls)

    async def handle_user_message(self, text: str) -> TurnResult:
        """Process a user message and return a TurnResult."""
        text_parts: list[str] = []
        total = 0
        async for ev in self.stream_message(text):
            if isinstance(ev, AgentTextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, AgentDone):
                total = ev.tool_calls_executed
        return TurnResult(
            text="".join(text_parts), tool_calls_executed=total
        )
