"""Tests for SessionResolver.

Covers:
  - Construction: default subscriptions (user_input + assistant_response), name, max_executions
  - Custom subscriptions override defaults
  - Protocol conformance with Resolver protocol
  - History capture: user_input events captured into internal history
  - History capture: assistant_response events captured into internal history
  - Delta fire (spc-1120): an assistant_response fire after the turn start
    returns only the message it adds, not a history snapshot
  - Resolve output: first call with no prior history returns empty content
  - Resolve output: current turn's events are captured but not included in output
  - Replay scope (spc-1125): only a turn-start fire (a batch carrying
    ``user_input``) replays history; every other fire emits its own delta, so
    a foreign event riding in the batch (``token_threshold``) cannot append a
    second copy of the conversation to the layer
  - Foreign-event hardening (spc-1129): a batch mixing a handled event with
    ``token_threshold`` still emits the delta and keeps the handled events in
    their own order (spec clause 11, third part)
  - Multi-turn accumulation: history grows across resolve() calls
  - Chronological ordering preserved
  - execution_count increments per call
  - Edge cases: no matching events, empty events list
"""

import pytest
from sr2.config.models import EventSubscriptionConfig, ResolverConfig
from sr2.models import Message, TextBlock, TokenUsage
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.protocols import Resolver
from sr2.pipeline.resolvers.session import SessionResolver
from sr2.protocols.llm import CompletionResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(**kwargs) -> ResolverConfig:
    """Build a minimal ResolverConfig for SessionResolver."""
    return ResolverConfig(type="session", **kwargs)


def make_user_input_event(text: str = "hello") -> Event:
    """Create a user_input event with a single TextBlock."""
    return Event(
        name="user_input",
        phase=EventPhase.STARTING,
        source_layer="core",
        data=[TextBlock(text=text)],
    )


def make_assistant_response_event(text: str = "hi there") -> Event:
    """Create an assistant_response event with a CompletionResponse."""
    return Event(
        name="assistant_response",
        phase=EventPhase.COMPLETED,
        source_layer="core",
        data=CompletionResponse(
            id="resp-001",
            content=[TextBlock(text=text)],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        ),
    )


def make_other_event(name: str = "turn_start") -> Event:
    """Create a non-session event."""
    return Event(
        name=name,
        phase=EventPhase.STARTING,
        source_layer="core",
    )


# ---------------------------------------------------------------------------
# 1. Construction — defaults
# ---------------------------------------------------------------------------


class TestSessionResolverConstruction:
    def test_constructs_with_valid_config(self):
        """SessionResolver builds without error from a basic config."""
        resolver = SessionResolver(make_config())
        assert resolver is not None

    def test_name_is_session(self):
        """Resolver name attribute must be 'session'."""
        resolver = SessionResolver(make_config())
        assert resolver.name == "session"

    def test_default_subscriptions_include_user_input(self):
        """Default subscriptions must include 'user_input'."""
        resolver = SessionResolver(make_config())
        names = [s.event_name for s in resolver.subscriptions]
        assert "user_input" in names

    def test_default_subscriptions_include_assistant_response(self):
        """Default subscriptions must include 'assistant_response'."""
        resolver = SessionResolver(make_config())
        names = [s.event_name for s in resolver.subscriptions]
        assert "assistant_response" in names

    def test_default_max_executions_is_unbounded(self):
        """SessionResolver is an accumulator — max_executions defaults to sys.maxsize
        so it fires on every matching event throughout a turn (including mid-turn
        tool loop iterations), regardless of the config value."""
        import sys
        resolver = SessionResolver(make_config())
        assert resolver.max_executions == sys.maxsize

    def test_max_executions_is_always_unbounded(self):
        """max_executions is always sys.maxsize regardless of config.max_executions.
        SessionResolver overrides it in __init__ to enable accumulator behaviour."""
        import sys
        resolver = SessionResolver(make_config(max_executions=10))
        assert resolver.max_executions == sys.maxsize


# ---------------------------------------------------------------------------
# 2. Custom subscriptions override
# ---------------------------------------------------------------------------


class TestSessionResolverSubscriptionOverride:
    def test_custom_subscriptions_replace_defaults(self):
        """Non-empty config.subscriptions override the defaults."""
        cfg = make_config(
            subscriptions=[
                EventSubscriptionConfig(event="custom_event", phase="completed"),
            ],
        )
        resolver = SessionResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "custom_event" in names
        assert "user_input" not in names
        assert "assistant_response" not in names

    def test_empty_config_subscriptions_falls_back_to_defaults(self):
        """Empty config.subscriptions => use default subscriptions."""
        cfg = make_config(subscriptions=[])
        resolver = SessionResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "user_input" in names
        assert "assistant_response" in names


# ---------------------------------------------------------------------------
# 3. Protocol conformance
# ---------------------------------------------------------------------------


class TestSessionResolverProtocolConformance:
    def test_isinstance_resolver(self):
        """SessionResolver must satisfy the Resolver protocol."""
        resolver = SessionResolver(make_config())
        assert isinstance(resolver, Resolver)

    def test_has_subscriptions_attribute(self):
        resolver = SessionResolver(make_config())
        assert hasattr(resolver, "subscriptions")
        assert isinstance(resolver.subscriptions, list)

    def test_has_max_executions_attribute(self):
        resolver = SessionResolver(make_config())
        assert hasattr(resolver, "max_executions")
        assert isinstance(resolver.max_executions, int)

    def test_has_execution_count_attribute(self):
        resolver = SessionResolver(make_config())
        assert hasattr(resolver, "execution_count")
        assert isinstance(resolver.execution_count, int)


# ---------------------------------------------------------------------------
# 4. First turn — no prior history
# ---------------------------------------------------------------------------


class TestSessionResolverFirstTurn:
    @pytest.mark.asyncio
    async def test_first_resolve_with_user_input_returns_empty(self):
        """First call returns empty content — there's no prior history yet.
        The current user_input is captured for next turn, not returned."""
        resolver = SessionResolver(make_config())
        event = make_user_input_event("first message")
        result = await resolver.resolve([event])

        assert isinstance(result, ResolvedContent)
        assert result.content == []

    @pytest.mark.asyncio
    async def test_first_resolve_with_no_events_returns_empty(self):
        """First call with no events returns empty content."""
        resolver = SessionResolver(make_config())
        result = await resolver.resolve([])

        assert isinstance(result, ResolvedContent)
        assert result.content == []

    @pytest.mark.asyncio
    async def test_first_resolve_sets_resolver_name(self):
        """ResolvedContent.resolver_name must be 'session'."""
        resolver = SessionResolver(make_config())
        result = await resolver.resolve([])
        assert result.resolver_name == "session"

    @pytest.mark.asyncio
    async def test_first_resolve_sets_source_layer(self):
        """ResolvedContent.source_layer must be set to a non-empty string."""
        resolver = SessionResolver(make_config())
        result = await resolver.resolve([])
        assert isinstance(result.source_layer, str)
        assert result.source_layer


# ---------------------------------------------------------------------------
# 5. History capture — user_input events
# ---------------------------------------------------------------------------


class TestSessionResolverUserInputCapture:
    @pytest.mark.asyncio
    async def test_user_input_captured_into_history(self):
        """A captured user_input message appears in the NEXT TURN's snapshot.

        spc-1125 retarget: only a turn-start fire (a batch carrying
        ``user_input``) replays history, because the layer appends whatever a
        fire emits and a replay outside the turn start duplicates the whole
        conversation. A later fire therefore returns just its own messages.
        """
        resolver = SessionResolver(make_config(max_executions=100))
        user_event = make_user_input_event("hello")

        # Turn 1: captures "hello", returns empty (no prior history)
        await resolver.resolve([user_event])

        # Turn 2's snapshot carries the captured user message.
        result = await resolver.resolve([make_user_input_event("turn 2")])
        assert len(result.content) == 1
        msg = result.content[0]
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert msg.content[0].text == "hello"

    @pytest.mark.asyncio
    async def test_user_input_not_in_same_call_output(self):
        """The current turn's user_input is NOT in the output of the
        same resolve() call — it's captured for next turn."""
        resolver = SessionResolver(make_config())
        user_event = make_user_input_event("current turn input")
        result = await resolver.resolve([user_event])

        # Output should not contain the current turn's message
        for item in result.content:
            if isinstance(item, Message) and item.role == "user":
                assert item.content[0].text != "current turn input"


# ---------------------------------------------------------------------------
# 6. History capture — assistant_response events
# ---------------------------------------------------------------------------


class TestSessionResolverAssistantResponseCapture:
    @pytest.mark.asyncio
    async def test_assistant_response_captured_into_history(self):
        """An assistant_response fire emits the message it captures (spc-1125).

        The fire is a delta fire: what it captures is what it returns, and the
        turn's next snapshot holds it as part of the history.
        """
        resolver = SessionResolver(make_config(max_executions=100))
        assistant_event = make_assistant_response_event("I can help")

        result = await resolver.resolve([assistant_event])

        assert len(result.content) == 1
        msg = result.content[0]
        assert isinstance(msg, Message)
        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert msg.content[0].text == "I can help"

    @pytest.mark.asyncio
    async def test_assistant_response_after_turn_start_emits_only_new_message(self):
        """End-of-turn fire returns the delta, never the history snapshot (spc-1120).

        The turn-start fire already handed the layer "hi", so replaying the
        snapshot here would append the whole conversation a second time. Fails
        against the pre-spc-1120 resolver, which emitted the prior history.
        """
        resolver = SessionResolver(make_config(max_executions=100))

        # Turn start captures "hi" — the layer now holds it.
        await resolver.resolve([make_user_input_event("hi")])

        result = await resolver.resolve([make_assistant_response_event("Done")])

        assert result.content == [
            Message(role="assistant", content=[TextBlock(text="Done")])
        ]

    @pytest.mark.asyncio
    async def test_assistant_response_extracts_content_from_completion_response(self):
        """CompletionResponse.content is extracted into Message(role='assistant').
        The CompletionResponse wrapper itself is not stored."""
        resolver = SessionResolver(make_config(max_executions=100))
        assistant_event = make_assistant_response_event("response text")

        result = await resolver.resolve([assistant_event])

        msg = result.content[0]
        assert isinstance(msg, Message)
        assert msg.role == "assistant"
        # Content should be the list of ContentBlocks, not the CompletionResponse
        assert msg.content[0].text == "response text"


# ---------------------------------------------------------------------------
# 7. Multi-turn accumulation
# ---------------------------------------------------------------------------


class TestSessionResolverMultiTurn:
    @pytest.mark.asyncio
    async def test_second_turn_returns_first_turn_history(self):
        """After one complete turn (user + assistant), the next resolve
        returns both messages from that prior turn."""
        resolver = SessionResolver(make_config(max_executions=100))

        # Turn 1: user input
        await resolver.resolve([make_user_input_event("hi")])
        # Turn 1: assistant response
        await resolver.resolve([make_assistant_response_event("hello!")])

        # Turn 2: should return prior turn's history
        result = await resolver.resolve([make_user_input_event("what's up?")])

        assert len(result.content) == 2
        assert result.content[0].role == "user"
        assert result.content[0].content[0].text == "hi"
        assert result.content[1].role == "assistant"
        assert result.content[1].content[0].text == "hello!"

    @pytest.mark.asyncio
    async def test_three_turns_returns_accumulated_history(self):
        """After two complete turns, the third resolve returns all four
        prior messages in chronological order."""
        resolver = SessionResolver(make_config(max_executions=100))

        # Turn 1
        await resolver.resolve([make_user_input_event("turn 1 user")])
        await resolver.resolve([make_assistant_response_event("turn 1 assistant")])

        # Turn 2
        await resolver.resolve([make_user_input_event("turn 2 user")])
        await resolver.resolve([make_assistant_response_event("turn 2 assistant")])

        # Turn 3: should return all 4 prior messages
        result = await resolver.resolve([make_user_input_event("turn 3 user")])

        assert len(result.content) == 4
        assert result.content[0].role == "user"
        assert result.content[0].content[0].text == "turn 1 user"
        assert result.content[1].role == "assistant"
        assert result.content[1].content[0].text == "turn 1 assistant"
        assert result.content[2].role == "user"
        assert result.content[2].content[0].text == "turn 2 user"
        assert result.content[3].role == "assistant"
        assert result.content[3].content[0].text == "turn 2 assistant"

    @pytest.mark.asyncio
    async def test_history_preserves_chronological_order(self):
        """Messages appear in the order they were captured:
        user, assistant, user, assistant, ..."""
        resolver = SessionResolver(make_config(max_executions=100))

        await resolver.resolve([make_user_input_event("u1")])
        await resolver.resolve([make_assistant_response_event("a1")])
        await resolver.resolve([make_user_input_event("u2")])
        await resolver.resolve([make_assistant_response_event("a2")])

        result = await resolver.resolve([make_user_input_event("u3")])
        roles = [msg.role for msg in result.content]
        assert roles == ["user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------


class TestSessionResolverEdgeCases:
    @pytest.mark.asyncio
    async def test_resolve_with_no_matching_events_replays_nothing(self):
        """A foreign-event fire emits nothing (spc-1125).

        ``Layer.check_budget`` queues ``token_threshold`` on every pass over
        the threshold, and that event rides into the tool-loop batches. The old
        contract — hand back the whole history — appended a second copy of the
        conversation to a layer that already held it (obsidian-4pkd: 88 wire
        messages became 152). History replays only at the turn start.
        """
        resolver = SessionResolver(make_config(max_executions=100))

        # Seed one turn of history
        await resolver.resolve([make_user_input_event("seeded")])
        await resolver.resolve([make_assistant_response_event("reply")])

        # A fire woken by an unrelated event adds nothing and emits nothing.
        result = await resolver.resolve([make_other_event("layer_ready")])

        assert result.content == []

    @pytest.mark.asyncio
    async def test_resolve_with_empty_events_replays_nothing(self):
        """Empty events list captures nothing and emits nothing (spc-1125)."""
        resolver = SessionResolver(make_config(max_executions=100))

        await resolver.resolve([make_user_input_event("msg")])
        result = await resolver.resolve([])

        assert result.content == []

    @pytest.mark.asyncio
    async def test_user_input_event_with_no_data_is_ignored(self):
        """A user_input event with data=None should not add to history."""
        resolver = SessionResolver(make_config(max_executions=100))

        empty_event = Event(
            name="user_input",
            phase=EventPhase.STARTING,
            source_layer="core",
            data=None,
        )
        await resolver.resolve([empty_event])
        result = await resolver.resolve([])

        assert result.content == []

    @pytest.mark.asyncio
    async def test_user_input_event_with_empty_data_is_ignored(self):
        """A user_input event with data=[] should not add to history."""
        resolver = SessionResolver(make_config(max_executions=100))

        empty_event = Event(
            name="user_input",
            phase=EventPhase.STARTING,
            source_layer="core",
            data=[],
        )
        await resolver.resolve([empty_event])
        result = await resolver.resolve([])

        assert result.content == []

    @pytest.mark.asyncio
    async def test_assistant_response_event_with_no_data_is_ignored(self):
        """An assistant_response event with data=None should not add to history."""
        resolver = SessionResolver(make_config(max_executions=100))

        empty_event = Event(
            name="assistant_response",
            phase=EventPhase.COMPLETED,
            source_layer="core",
            data=None,
        )
        await resolver.resolve([empty_event])
        result = await resolver.resolve([])

        assert result.content == []

    @pytest.mark.asyncio
    async def test_output_is_copy_not_reference(self):
        """Returned content should not be a mutable reference to internal state.
        Mutating the output must not corrupt future resolve() calls."""
        resolver = SessionResolver(make_config(max_executions=100))

        await resolver.resolve([make_user_input_event("persistent")])
        result1 = await resolver.resolve([make_user_input_event("turn 2")])

        # Mutate the returned list
        result1.content.clear()

        # Internal history should be unaffected
        result2 = await resolver.resolve([make_user_input_event("turn 3")])
        assert len(result2.content) == 2
        assert result2.content[0].content[0].text == "persistent"


# ---------------------------------------------------------------------------
# 9. execution_count
# ---------------------------------------------------------------------------


class TestSessionResolverExecutionCount:
    def test_execution_count_starts_at_zero(self):
        """Fresh resolver should have execution_count == 0."""
        resolver = SessionResolver(make_config())
        assert resolver.execution_count == 0

    @pytest.mark.asyncio
    async def test_execution_count_increments_after_resolve(self):
        """execution_count increments once per resolve() call."""
        resolver = SessionResolver(make_config())
        await resolver.resolve([make_user_input_event()])
        assert resolver.execution_count == 1

    @pytest.mark.asyncio
    async def test_execution_count_increments_on_each_call(self):
        """Multiple calls accumulate in execution_count."""
        resolver = SessionResolver(make_config(max_executions=100))
        await resolver.resolve([make_user_input_event()])
        await resolver.resolve([make_assistant_response_event()])
        await resolver.resolve([make_user_input_event()])
        assert resolver.execution_count == 3

    @pytest.mark.asyncio
    async def test_execution_count_increments_even_with_no_matching_events(self):
        """execution_count increments even when no matching events found."""
        resolver = SessionResolver(make_config())
        await resolver.resolve([make_other_event()])
        assert resolver.execution_count == 1


# ---------------------------------------------------------------------------
# 10. build() classmethod
# ---------------------------------------------------------------------------


class TestSessionResolverBuild:
    def test_build_returns_session_resolver_instance(self):
        """build() must return a SessionResolver instance."""
        config = make_config()
        result = SessionResolver.build(config, Dependencies())
        assert isinstance(result, SessionResolver)

    def test_build_with_populated_deps_also_works(self):
        """build() must accept and ignore a non-empty Dependencies container."""
        config = make_config()
        deps = Dependencies(llm={"default": lambda *a, **kw: None})
        result = SessionResolver.build(config, deps)
        assert isinstance(result, SessionResolver)

    def test_build_result_satisfies_resolver_protocol(self):
        """Instance returned by build() must satisfy isinstance(x, Resolver)."""
        config = make_config()
        result = SessionResolver.build(config, Dependencies())
        assert isinstance(result, Resolver)

    def test_build_state_matches_direct_construction(self):
        """build() must produce an instance with the same observable state
        as one constructed via SessionResolver(config) directly."""
        config = make_config(max_executions=3)
        via_build = SessionResolver.build(config, Dependencies())
        via_init = SessionResolver(config)
        assert via_build.max_executions == via_init.max_executions
        assert via_build.name == via_init.name
        assert via_build.execution_count == via_init.execution_count


# ---------------------------------------------------------------------------
# spc-1118 (FR 5, AC4): tool iterations reach the session layer as they happen
# ---------------------------------------------------------------------------
#
# Events are the ones the orchestrator queues for one tool iteration, in its
# order: tool_use_emitted, tool_result_received, then
# assistant_iteration_response (the iteration's full assistant content).
# Like the layer, _deliver passes the resolver only the events it subscribes to.

from sr2.models import ThinkingBlock, ToolResultBlock, ToolUseBlock


def _tool_iteration_events(tc_id: str, text: str | None = None, thinking: str | None = None) -> list[Event]:
    content = []
    if thinking:
        content.append(ThinkingBlock(text=thinking))
    if text:
        content.append(TextBlock(text=text))
    tool_use = ToolUseBlock(id=tc_id, name="get_weather", input={"location": "Oslo"})
    content.append(tool_use)
    result = ToolResultBlock(tool_use_id=tc_id, content=f"result {tc_id}")
    return [
        Event(name="tool_use_emitted", phase=EventPhase.COMPLETED, source_layer="orchestrator", data=[tool_use]),
        Event(name="tool_result_received", phase=EventPhase.COMPLETED, source_layer="orchestrator", data=[result]),
        Event(
            name="assistant_iteration_response",
            phase=EventPhase.COMPLETED,
            source_layer="orchestrator",
            data=CompletionResponse(
                id="iter-response",
                content=content,
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            ),
        ),
    ]


async def _deliver(resolver: SessionResolver, events: list[Event]) -> ResolvedContent:
    return await resolver.resolve(
        [e for e in events if any(s.matches(e) for s in resolver.subscriptions)]
    )


def _expected_assistant(tc_id: str, text: str | None = None, thinking: str | None = None) -> Message:
    content = []
    if thinking:
        content.append(ThinkingBlock(text=thinking))
    if text:
        content.append(TextBlock(text=text))
    content.append(ToolUseBlock(id=tc_id, name="get_weather", input={"location": "Oslo"}))
    return Message(role="assistant", content=content)


def _expected_results(tc_id: str) -> Message:
    return Message(role="user", content=[ToolResultBlock(tool_use_id=tc_id, content=f"result {tc_id}")])


class TestSessionResolverToolIterations:
    @pytest.mark.asyncio
    async def test_ac4_r1_tool_iteration_adds_full_assistant_then_results(self):
        resolver = SessionResolver(make_config())
        await resolver.resolve([make_user_input_event("run a tool")])

        result = await _deliver(resolver, _tool_iteration_events("tc1", text="Let me look."))

        assert result.content == [_expected_assistant("tc1", text="Let me look."), _expected_results("tc1")]

    @pytest.mark.asyncio
    async def test_ac4_r2_assistant_message_carries_thinking_block(self):
        resolver = SessionResolver(make_config())
        await resolver.resolve([make_user_input_event("run a tool")])

        result = await _deliver(
            resolver,
            _tool_iteration_events("tc1", text="Let me look.", thinking="I should check.")
        )

        assert result.content[0] == _expected_assistant("tc1", text="Let me look.", thinking="I should check.")
        assert isinstance(result.content[0].content[0], ThinkingBlock)

    @pytest.mark.asyncio
    async def test_ac4_r3_next_turn_history_holds_full_assistant_then_results(self):
        resolver = SessionResolver(make_config())
        await resolver.resolve([make_user_input_event("run a tool")])
        await _deliver(resolver, _tool_iteration_events("tc1", text="Let me look."))
        await resolver.resolve([make_assistant_response_event("Done.")])

        next_turn = await resolver.resolve([make_user_input_event("next")])

        assert next_turn.content == [
            Message(role="user", content=[TextBlock(text="run a tool")]),
            _expected_assistant("tc1", text="Let me look."),
            _expected_results("tc1"),
            Message(role="assistant", content=[TextBlock(text="Done.")]),
        ]

    @pytest.mark.asyncio
    async def test_ac4_r4_two_iterations_add_a1_r1_a2_r2_without_duplicates(self):
        resolver = SessionResolver(make_config())
        await resolver.resolve([make_user_input_event("run two tools")])

        first = await _deliver(resolver, _tool_iteration_events("tc1", text="First."))
        second = await _deliver(resolver, _tool_iteration_events("tc2", text="Second."))
        await resolver.resolve([make_assistant_response_event("Done.")])
        next_turn = await resolver.resolve([make_user_input_event("next")])

        a1, r1 = _expected_assistant("tc1", text="First."), _expected_results("tc1")
        a2, r2 = _expected_assistant("tc2", text="Second."), _expected_results("tc2")
        assert first.content + second.content == [a1, r1, a2, r2]
        assert next_turn.content == [
            Message(role="user", content=[TextBlock(text="run two tools")]),
            a1,
            r1,
            a2,
            r2,
            Message(role="assistant", content=[TextBlock(text="Done.")]),
        ]


# ---------------------------------------------------------------------------
# 10. Foreign event in the batch (spc-1129) — spec clause 11, third part
#
# ``Layer.check_budget`` queues ``token_threshold`` on every pass over the
# threshold, and that event rides into the pending batch beside the resolver's
# own events. The resolver reads it as foreign: it must not flip an end-of-turn
# fire back to a history replay, and it must not become part of the batch's own
# ordering. ``token_threshold`` is not in the resolver's subscriptions, so
# unlike ``_deliver`` these calls hand ``resolve()`` the raw mixed batch — that
# is what no earlier test in this file does.
# ---------------------------------------------------------------------------


def token_threshold_event() -> Event:
    """The event Layer.check_budget queues while a layer stays over threshold."""
    return Event(
        name="token_threshold",
        phase=EventPhase.COMPLETED,
        source_layer="session",
        data={"used": 99999, "budget": 4000, "pct": 0.6},
    )


class TestSessionResolverForeignEventInBatch:
    @pytest.mark.asyncio
    async def test_mixed_foreign_batch_still_emits_the_delta(self):
        """A foreign ``token_threshold`` must not flip the fire to a replay (spc-1129).

        Turn 1 is seeded at the turn start, so a snapshot on this fire would
        hand the layer a second copy of the conversation. The fire returns
        exactly the assistant message it captures.

        Fails against the pre-spc-1125 form that reads the unfiltered batch
        (``any(e.name == "user_input" for e in events)``): there the foreign
        event makes the batch look like a turn start and the content comes
        back as turn 1's history instead of the delta.
        """
        resolver = SessionResolver(make_config(max_executions=100))

        # Turn 1: the turn-start fire, which is the only snapshot fire.
        await resolver.resolve([make_user_input_event("hi")])

        # End of turn: the delta fire, with a foreign event riding in its batch.
        result = await resolver.resolve(
            [make_assistant_response_event("Done"), token_threshold_event()]
        )

        assert result.content == [
            Message(role="assistant", content=[TextBlock(text="Done")])
        ]

    @pytest.mark.asyncio
    async def test_foreign_event_still_does_not_replay_on_a_later_fire(self):
        """The next end-of-turn fire also emits its delta, not a snapshot (spc-1129).

        Guards the delta decision on a fire whose batch holds no event with a
        rank in ``_TOOL_LOOP_ORDER`` either: an unfiltered read that keys off
        tool-loop events would replay the whole history here, so the content is
        pinned to the single new message.
        """
        resolver = SessionResolver(make_config(max_executions=100))

        await resolver.resolve([make_user_input_event("hi")])
        await resolver.resolve([make_assistant_response_event("Done")])

        result = await resolver.resolve(
            [make_assistant_response_event("Later"), token_threshold_event()]
        )

        assert result.content == [
            Message(role="assistant", content=[TextBlock(text="Later")])
        ]

    @pytest.mark.asyncio
    async def test_mixed_foreign_batch_does_not_reorder_the_handled_events(self):
        """A foreign event between two handled events must not defeat their ranking (spc-1129).

        The batch's handled events are queued OUT of their ``_TOOL_LOOP_ORDER``
        rank order — ``tool_result_received`` (rank 1) before
        ``assistant_iteration_response`` (rank 0) — with the foreign
        ``token_threshold`` riding between them. The resolver must still emit
        the ranked content (assistant message, then tool results), so the
        only way this passes is the sort over the batch's own handled events:
        without the ``_HANDLED_EVENTS`` filter (the unfiltered batch then
        contains a non-ranked event, so the guard skips the sort), or with the
        guard reading the unfiltered batch, the result comes back unsorted
        (tool results first) and this assertion fails.
        """
        resolver = SessionResolver(make_config(max_executions=100))
        await resolver.resolve([make_user_input_event("run a tool")])

        iteration = _tool_iteration_events("tc1", text="Let me look.")
        # iteration == [tool_use_emitted, tool_result_received,
        # assistant_iteration_response]; queue the two handled events out of
        # rank order (results rank 1, then iteration response rank 0) with the
        # foreign event between them.
        batch = [iteration[1], token_threshold_event(), iteration[2]]

        result = await resolver.resolve(batch)

        assert [m.role for m in result.content] == ["assistant", "user"]
        assert result.content == [
            _expected_assistant("tc1", text="Let me look."),
            _expected_results("tc1"),
        ]
