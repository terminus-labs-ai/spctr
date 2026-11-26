"""SessionResolver: accumulates conversation history across turns.

The turn-start fire (the batch carrying ``user_input``) returns the PRIOR
history; every later fire in the turn returns only the messages it adds. Each
fire also captures its events (user_input, assistant_response,
assistant_iteration_response, tool_result_received) into the history for the
next turn's snapshot.
"""

from __future__ import annotations

import sys

from sr2.config.models import ResolverConfig
from sr2.models import ContentBlock, Message, ToolResultBlock, ToolUseBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions
from sr2.protocols.llm import CompletionResponse

_DEFAULT_SUBSCRIPTIONS = [
    EventSubscription(event_name="user_input", phase=EventPhase.STARTING),
    EventSubscription(event_name="assistant_response", phase=EventPhase.COMPLETED),
    EventSubscription(event_name="assistant_iteration_response", phase=EventPhase.COMPLETED),
    EventSubscription(event_name="tool_result_received", phase=EventPhase.COMPLETED),
]

# Tool-loop event names, ranked in the order their messages are appended.
_TOOL_LOOP_ORDER = {
    "assistant_iteration_response": 0,
    "tool_use_emitted": 0,
    "tool_result_received": 1,
}

# Event names this resolver consumes. Anything else can ride in the same
# pending batch (``Layer.check_budget`` queues ``token_threshold`` on every
# pass over the threshold) and must not change how the batch is read.
_HANDLED_EVENTS = frozenset({"user_input", "assistant_response", *_TOOL_LOOP_ORDER})


class SessionResolver:
    """Accumulates user/assistant messages across turns.

    Acts as an accumulator — max_executions is set to sys.maxsize so it fires
    on every matching event throughout the turn (including mid-turn tool loop
    iterations).

    Each resolve() call:
      1. Emits the prior history (copy) when the batch starts a turn, else the
         delta of the messages captured by this call. Emitting the snapshot on
         any other fire appends a second copy of the conversation to a layer
         that already holds it (spc-1125).
      2. Captures matching events into internal history for next turn.
      3. Increments execution_count.

    Supported events:
      - user_input: appends a user Message from event.data (list[ContentBlock])
      - assistant_response: appends an assistant Message from CompletionResponse.content
      - assistant_iteration_response: appends the tool iteration's full assistant
        Message (thinking, text and tool_use blocks) from CompletionResponse.content
      - tool_use_emitted: appends an assistant Message containing tool_use blocks
        (only when a config subscribes to it explicitly)
      - tool_result_received: appends a user Message containing tool_result blocks
    """

    name: str = "session"

    def __init__(self, config: ResolverConfig) -> None:
        self._config = config
        self.max_executions: int = sys.maxsize
        self.execution_count: int = 0
        self._history: list[Message] = []
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, _DEFAULT_SUBSCRIPTIONS
        )

    @classmethod
    def build(cls, config: ResolverConfig, deps: "Dependencies") -> "SessionResolver":
        return cls(config)

    def seed(self, messages: list[Message]) -> None:
        """Pre-populate conversation history.

        Replaces any existing history with independent copies of *messages*.
        Call before the first turn to inject prior context (e.g. restored sessions).
        """
        self._history = [m.model_copy() for m in messages]

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1

        # Only the turn-start fire (the batch carrying ``user_input``) emits the
        # history snapshot. Every later fire in the turn (tool loop, end of
        # turn) returns only the messages it adds: the layer already holds the
        # snapshot, and re-emitting it appends a second copy of the whole
        # conversation. Foreign events in the batch are ignored here — spc-1125
        # was ``token_threshold`` riding in with the tool events and flipping
        # this fire back to a full replay on every iteration past the threshold.
        handled = [e for e in events if e.name in _HANDLED_EVENTS]
        emits_delta_only = not any(e.name == "user_input" for e in handled)

        # Snapshot prior history for output (copy so mutations don't corrupt state).
        output: list[Message] = [] if emits_delta_only else list(self._history)
        history_len_before = len(self._history)
        if all(e.name in _TOOL_LOOP_ORDER for e in handled):
            # The assistant message precedes its tool results on the wire,
            # whatever order the orchestrator queued the events in.
            handled = sorted(handled, key=lambda e: _TOOL_LOOP_ORDER[e.name])

        # Capture new events into history for subsequent calls
        for event in handled:
            if event.name == "user_input":
                data: list[ContentBlock] | None = event.data
                if data:
                    self._history.append(Message(role="user", content=list(data)))

            elif event.name == "assistant_response":
                response: CompletionResponse | None = event.data
                if response is not None:
                    self._history.append(
                        Message(role="assistant", content=list(response.content))
                    )

            elif event.name == "assistant_iteration_response":
                iteration: CompletionResponse | None = event.data
                if iteration is not None:
                    self._history.append(
                        Message(role="assistant", content=list(iteration.content))
                    )

            elif event.name == "tool_use_emitted":
                tool_blocks: list[ContentBlock] | None = event.data
                if tool_blocks:
                    self._history.append(
                        Message(role="assistant", content=list(tool_blocks))
                    )

            elif event.name == "tool_result_received":
                result_blocks: list[ToolResultBlock] | None = event.data
                if result_blocks:
                    self._history.append(
                        Message(role="user", content=list(result_blocks))
                    )

        if emits_delta_only:
            output = list(self._history[history_len_before:])

        return ResolvedContent(
            resolver_name=self.name,
            source_layer="session",
            content=output,
        )
