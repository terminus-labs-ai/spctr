"""spc-1125 — a long run must never send a request without the user query.

Live failure (obsidian-4pkd / obsidian-8qpc.15, release ``41b6b21``): after the
session layer's ``compact`` transformer summarised once, the conversation kept
growing, the litellm guard logged ``dropped 19/18/12 oldest message(s) to fit
context_limit=194650`` and the next request died with provider
``400 invalid_prompt 'No user query found in messages.'``.

Three defects combine:

1. ``Layer.check_budget`` queues ``token_threshold`` on every pass while the
   layer stays over its threshold. That event rides into the
   ``SessionResolver``'s next fire, and the resolver decided "snapshot vs.
   delta" by testing whether the batch was a *subset* of its own event names —
   so one foreign trigger event flipped it back to replaying the whole history,
   which the layer appended to the copy it already held (88 → 152 wire
   messages, 114k → 296k layer tokens, on every later iteration).
2. ``Layer.force_truncate`` then popped content off the front until the layer
   fitted ``token_budget``, eating the run's prompt and the summary message.
   Tool results compile to ``role="tool"`` wire messages, so what left the
   pipeline had no user-role message at all. The guard's shed rule protects the
   prompt; it cannot restore one.
3. ``TransformerConfig.max_executions`` defaults to 1 and the live FR 12 config
   does not set it, so the summariser ran exactly once per run and never
   protected the run again.

Spec: topics/sr2/spctr/specs/context-window-guard.md (FR 9-12, dec-854).
Expected values are written out here, never read back from the code.
"""

from __future__ import annotations

import yaml
from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig, TransformerConfig
from sr2.models import Message, TextBlock, TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget, ResolvedContent
from sr2.pipeline.resolvers.session import SessionResolver
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent

from conftest import SequentialMockLLM, make_user_input, tool_use_event

SYSTEM_TEXT = "You are a context-pressure test assistant."
PROMPT = "Build the thing."
TOOL_NAME = "read_file"
TOOL_INPUT = {"path": "a"}
RESULT_CHARS = 3200  # 800 tokens at CharacterTokenCounter's chars // 4
HEADER = "[Summary of {count} earlier messages]"
SUMMARY_TEXT = "S"
SUMMARY_PROMPT = "Summarise the earlier tool work."

# Budget 4000 / threshold 0.6 → the threshold is 2400 tokens; each tool result
# is 800, so the layer crosses it, gets compacted, and crosses it again.
FR12_YAML = f"""
layers:
  - name: system
    target: system
    resolvers:
      - type: static
        config:
          text: "{SYSTEM_TEXT}"
  - name: session
    target: messages
    token_budget: 4000
    token_threshold_pct: 0.6
    resolvers:
      - {{name: session, type: session}}
      - type: input
        subscriptions: [{{event: user_input, phase: completed}}]
    transformers:
      - name: compact
        type: summarize
        subscriptions: [{{event: token_threshold}}]
        config:
          keep_strategy: keep_within_tokens
          keep_tokens: 1000
          retry_growth_tokens: 10000
          summarization_prompt: "{SUMMARY_PROMPT}"
          summary_header: "{HEADER}"
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def token_threshold_event() -> Event:
    """The event Layer.check_budget queues while a layer stays over threshold."""
    return Event(
        name="token_threshold",
        phase=EventPhase.COMPLETED,
        source_layer="session",
        data={"used": 99999, "budget": 4000, "pct": 0.6},
    )


def iteration_events(tool_call_id: str) -> list[Event]:
    """The two events one tool iteration leaves behind, orchestrator order."""
    return [
        Event(
            name="assistant_iteration_response",
            phase=EventPhase.COMPLETED,
            source_layer="core",
            data=CompletionResponse(
                id=f"resp-{tool_call_id}",
                content=[ToolUseBlock(id=tool_call_id, name=TOOL_NAME, input=dict(TOOL_INPUT))],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            ),
        ),
        Event(
            name="tool_result_received",
            phase=EventPhase.COMPLETED,
            source_layer="core",
            data=[
                ToolResultBlock(
                    tool_use_id=tool_call_id, content=result_text(tool_call_id)
                )
            ],
        ),
    ]


def result_text(tool_call_id: str) -> str:
    return (tool_call_id + "-") * (RESULT_CHARS // (len(tool_call_id) + 1))


def make_user_input_event(text: str = PROMPT) -> Event:
    return Event(
        name="user_input",
        phase=EventPhase.STARTING,
        source_layer="core",
        data=[TextBlock(text=text)],
    )


class _LoopLLM(SequentialMockLLM):
    """Scripted stream replies for the loop; ``complete`` is the summary call."""

    def __init__(self, call_sequences) -> None:
        super().__init__(call_sequences)
        self.complete_calls: list[CompletionRequest] = []
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.complete_calls.append(request)
        return CompletionResponse(
            id="summary",
            content=[TextBlock(text=SUMMARY_TEXT)],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


def _noop_llm() -> "_LoopLLM":
    """A scripted LLM that is never asked to stream; used for build-time tests."""
    return _LoopLLM([[StreamEvent(type="end")]])


async def _executor(block: ToolUseBlock) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=block.id, content=result_text(block.id))


def _reply_tool(tc_id: str):
    return [
        tool_use_event(tool_use_id=tc_id, tool_name=TOOL_NAME, tool_input=dict(TOOL_INPUT)),
        StreamEvent(type="end"),
    ]


def _script(n_tools: int):
    return [_reply_tool(f"tc{i}") for i in range(1, n_tools + 1)] + [
        [StreamEvent(type="text", text="Done."), StreamEvent(type="end")]
    ]


async def _run(llm: _LoopLLM, pipeline: PipelineConfig) -> None:
    sr2 = SR2(
        pipeline_config=pipeline,
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=_executor,
    )
    async for _ in sr2.turn(make_user_input(PROMPT)):
        pass


def _pipeline() -> PipelineConfig:
    return PipelineConfig.model_validate(yaml.safe_load(FR12_YAML))


def messages_layer(token_budget: int | None) -> Layer:
    return Layer(
        name="conversation",
        target=CompilationTarget.MESSAGES,
        position=AppendStrategy(),
        token_budget=token_budget,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
    )


def put(layer: Layer, content: list) -> None:
    layer.add_content(
        ResolvedContent(resolver_name="stub", source_layer=layer.name, content=content)
    )


def U(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


def A(tc_id: str) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseBlock(id=tc_id, name=TOOL_NAME, input=dict(TOOL_INPUT))],
    )


def R(tc_id: str, chars: int = RESULT_CHARS) -> Message:
    return Message(
        role="user",
        content=[ToolResultBlock(tool_use_id=tc_id, content=result_text(tc_id)[:chars])],
    )


def has_user_query(content: list) -> bool:
    """True when some item is a user message carrying real text (not a tool result)."""
    return any(
        isinstance(item, Message)
        and item.role == "user"
        and any(isinstance(b, TextBlock) and b.text.strip() for b in item.content)
        for item in content
    )


# ===========================================================================
# 1 — SessionResolver: a foreign trigger event must not replay the history
# ===========================================================================


class TestForeignEventDoesNotReplayHistory:
    async def test_r1_delta_contains_only_the_new_iteration_messages(self):
        resolver = SessionResolver(ResolverConfig(type="session"))
        await resolver.resolve([make_user_input_event()])
        await resolver.resolve(iteration_events("tc1"))

        result = await resolver.resolve([*iteration_events("tc2"), token_threshold_event()])

        assert len(result.content) == 2
        assistant, tool_result = result.content
        assert assistant.role == "assistant"
        assert isinstance(assistant.content[0], ToolUseBlock)
        assert assistant.content[0].id == "tc2"
        assert tool_result.role == "user"
        assert isinstance(tool_result.content[0], ToolResultBlock)

    async def test_r2_foreign_event_never_reorders_an_assistant_past_its_results(self):
        resolver = SessionResolver(ResolverConfig(type="session"))
        await resolver.resolve([make_user_input_event()])

        # The orchestrator may queue the tool result before the iteration
        # response; the assistant message still precedes its results.
        batch = [
            iteration_events("tc1")[1],
            token_threshold_event(),
            iteration_events("tc1")[0],
        ]
        result = await resolver.resolve(batch)

        assert [type(b).__name__ for b in result.content[0].content] == ["ToolUseBlock"]
        assert [type(b).__name__ for b in result.content[1].content] == ["ToolResultBlock"]


# ===========================================================================
# 2 — Layer.force_truncate keeps the run's prompt and its tool pairs
# ===========================================================================


class TestForceTruncateKeepsPrompt:
    def test_r3_prompt_survives_a_messages_layer_truncation(self):
        layer = messages_layer(3000)
        put(
            layer,
            [U(PROMPT)]
            + [item for i in range(1, 5) for item in (A(f"tc{i}"), R(f"tc{i}"))],
        )

        warning = layer.force_truncate()

        assert warning is not None
        content = layer.get_content()
        assert has_user_query(content), [type(i).__name__ for i in content]
        assert content[0] == U(PROMPT)
        # Never leaves a tool result whose assistant call was truncated away.
        assert not starts_with_tool_result(content)

    def test_r4_pairs_are_dropped_whole(self):
        layer = messages_layer(1700)
        content = [U(PROMPT), A("tc1"), R("tc1"), A("tc2"), R("tc2"), A("tc3"), R("tc3")]
        put(layer, content)

        layer.force_truncate()

        kept = layer.get_content()
        ids_kept = [
            b.tool_use_id
            for m in kept
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
        calls_kept = [
            b.id
            for m in kept
            for b in m.content
            if isinstance(b, ToolUseBlock)
        ]
        assert set(ids_kept) <= set(calls_kept)

    def test_r5_cannot_fit_without_the_prompt_leaves_it_in_place(self):
        big_prompt = U("read the whole vault " * 100)  # 2300 tokens
        layer = messages_layer(10)
        put(layer, [big_prompt])

        warning = layer.force_truncate()

        assert layer.get_content() == [big_prompt]
        assert warning is not None and "prompt" in warning

    def test_r6_system_target_truncation_is_unchanged(self):
        layer = Layer(
            name="system",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=100,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )
        put(layer, [TextBlock(text="a" * 800), TextBlock(text="b" * 800)])

        warning = layer.force_truncate()

        assert warning is not None
        assert CharacterTokenCounter().count(layer.get_content()) <= 100


def starts_with_tool_result(content: list) -> bool:
    if not content or not isinstance(content[0], Message):
        return False
    return any(isinstance(b, ToolResultBlock) for b in content[0].content)


# ===========================================================================
# 3 — Compaction repeats: max_executions must not cap a pressure-triggered
#     summariser at one execution
# ===========================================================================


class TestCompactionRepeats:
    def test_r7_subscribed_summariser_is_not_capped_at_one_execution(self):
        from sr2.pipeline.dependencies import Dependencies
        from sr2.pipeline.transformers.summarization import SummarizationTransformer

        config = TransformerConfig(
            type="summarize",
            name="compact",
            subscriptions=[{"event": "token_threshold"}],
            config={"keep_strategy": "keep_within_tokens", "keep_tokens": 100},
        )
        transformer = SummarizationTransformer.build(
            config, Dependencies(llm={"default": _noop_llm()}, session_id="s")
        )

        assert transformer.max_executions > 1

    def test_r8_explicit_max_executions_is_respected(self):
        from sr2.pipeline.dependencies import Dependencies
        from sr2.pipeline.transformers.summarization import SummarizationTransformer

        config = TransformerConfig(
            type="summarize",
            name="compact",
            max_executions=3,
            subscriptions=[{"event": "token_threshold"}],
            config={"keep_strategy": "keep_within_tokens", "keep_tokens": 100},
        )
        transformer = SummarizationTransformer.build(
            config, Dependencies(llm={"default": _noop_llm()}, session_id="s")
        )

        assert transformer.max_executions == 3

    def test_r9_unsubscribed_summariser_keeps_the_default_cap(self):
        from sr2.pipeline.dependencies import Dependencies
        from sr2.pipeline.transformers.summarization import SummarizationTransformer

        config = TransformerConfig(
            type="summarize",
            name="compact",
            config={"keep_strategy": "keep_last_n", "keep_last_n": 2},
        )
        transformer = SummarizationTransformer.build(
            config, Dependencies(llm={"default": _noop_llm()}, session_id="s")
        )

        assert transformer.max_executions == 1


class TestLongRunKeepsAUserQuery:
    async def test_r10_long_run_summarises_more_than_once(self):
        llm = _LoopLLM(_script(8))

        await _run(llm, _pipeline())

        assert len(llm.complete_calls) >= 2, (
            f"expected the summariser to fire again after the second threshold "
            f"crossing; got {len(llm.complete_calls)} summary call(s)"
        )

    async def test_r11_no_request_loses_the_run_prompt(self):
        llm = _LoopLLM(_script(8))

        await _run(llm, _pipeline())

        for i, request in enumerate(llm.stream_calls):
            assert has_user_query(request.messages), (
                f"request {i} has no user message: "
                f"{[(m.role, type(m.content[0]).__name__) for m in request.messages]}"
            )
