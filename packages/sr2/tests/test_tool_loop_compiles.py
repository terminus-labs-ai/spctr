"""spc-1118 — the tool loop sends what the pipeline compiles.

Spec: topics/sr2/spctr/specs/context-window-guard.md, FR 5-7, AC4-AC5.

AC4 (FR 5): the session layer holds each tool iteration's full assistant
message and its tool results as they happen, and the next turn's history
matches.

AC5 (FR 6-7): every request the tool loop sends equals the request the
pre-change hand-built loop sends (system, messages, tools) when no
transformer changes content; a transformer that changes the session layer
on ``tool_result_received`` changes the next request.

Expected requests are written out from each case's construction (the
user input, the scripted model replies and the stub executor's results),
never read back from the code under test. The real orchestrator, engine,
bus and resolvers run; only the LLM, the tool executor and one test
transformer are stand-ins.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import pytest
import sr2.orchestrator as orch
from sr2.config.models import (
    EventSubscriptionConfig,
    FinalAnswerConfig,
    LayerConfig,
    PipelineConfig,
    RepeatGuardConfig,
    ResolverConfig,
    TransformerConfig,
)
from sr2.models import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
from sr2.pipeline.models import TransformationResult
from sr2.pipeline.resolvers import static_text as static_text_module
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions
from sr2.protocols.llm import CompletionRequest, StreamEvent

from conftest import SequentialMockLLM, make_user_input, stub_executor, tool_use_event

SYSTEM_TEXT = "You are a tool loop test assistant."
TOOL_NAME = "get_weather"
TOOL_INPUT = {"location": "Oslo"}
TOOL_RESULT = f"result_for_{TOOL_NAME}"  # what conftest.stub_executor returns


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _conversation_layer(*, with_clock: bool, transformers=None) -> LayerConfig:
    resolvers = [ResolverConfig(type="session")]
    if with_clock:
        resolvers.append(
            ResolverConfig(
                type="static_text",
                name="clock",
                config={"timezone": "America/Phoenix", "text": "CLOCK {time}"},
                subscriptions=[EventSubscriptionConfig(event="user_input", phase="completed")],
            )
        )
    resolvers.append(
        ResolverConfig(
            type="input",
            subscriptions=[EventSubscriptionConfig(event="user_input", phase="completed")],
        )
    )
    return LayerConfig(
        name="conversation",
        target="messages",
        resolvers=resolvers,
        transformers=transformers,
    )


def _pipeline(*, with_clock: bool = False, transformers=None, **kwargs) -> PipelineConfig:
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[ResolverConfig(type="static", config={"text": SYSTEM_TEXT})],
            ),
            _conversation_layer(with_clock=with_clock, transformers=transformers),
        ],
        **kwargs,
    )


class _LayerSnapshotLLM(SequentialMockLLM):
    """SequentialMockLLM that also snapshots the conversation layer's content
    at the moment of each call (the state the request was built from)."""

    def __init__(self, call_sequences):
        super().__init__(call_sequences)
        self.sr2: SR2 | None = None
        self.layer_snapshots: list[list] = []

    async def stream(self, request):
        assert self.sr2 is not None
        layer = next(l for l in self.sr2._engine.layers if l.name == "conversation")
        self.layer_snapshots.append([m.model_copy(deep=True) for m in layer.get_content()])
        async for event in super().stream(request):
            yield event


def _make_sr2(llm, pipeline_config: PipelineConfig) -> SR2:
    sr2 = SR2(
        pipeline_config=pipeline_config,
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=stub_executor,
    )
    if isinstance(llm, _LayerSnapshotLLM):
        llm.sr2 = sr2
    return sr2


async def _drive(sr2: SR2, text: str) -> None:
    async for _ in sr2.turn(make_user_input(text)):
        pass


# Scripted replies -----------------------------------------------------------


def _reply_tool(tc_id: str, text: str | None = None, thinking: str | None = None):
    events = []
    if thinking:
        events.append(StreamEvent(type="thinking", text=thinking))
    if text:
        events.append(StreamEvent(type="text", text=text))
    events.append(tool_use_event(tool_use_id=tc_id, tool_name=TOOL_NAME, tool_input=dict(TOOL_INPUT)))
    events.append(StreamEvent(type="end"))
    return events


def _reply_text(text: str):
    return [StreamEvent(type="text", text=text), StreamEvent(type="end")]


def _reply_empty():
    return [StreamEvent(type="end")]


# Expected messages ----------------------------------------------------------


def U(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


def A_text(text: str) -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)])


def A(tc_id: str, text: str | None = None, thinking: str | None = None) -> Message:
    content = []
    if thinking:
        content.append(ThinkingBlock(text=thinking))
    if text:
        content.append(TextBlock(text=text))
    content.append(ToolUseBlock(id=tc_id, name=TOOL_NAME, input=dict(TOOL_INPUT)))
    return Message(role="assistant", content=content)


def R(tc_id: str, content: str = TOOL_RESULT) -> Message:
    return Message(role="user", content=[ToolResultBlock(tool_use_id=tc_id, content=content)])


SYSTEM = [TextBlock(text=SYSTEM_TEXT)]


def _assert_request(actual: CompletionRequest, messages: list[Message], tools=None) -> None:
    assert actual.system == SYSTEM
    assert actual.messages == messages
    assert actual.tools == tools


# ===========================================================================
# AC4 — the session layer holds each tool iteration as it happens (FR 5)
# ===========================================================================


class TestAC4SessionLayerHoldsToolIterations:
    @pytest.mark.asyncio
    async def test_ac4_r1_layer_ends_with_assistant_then_results_after_iteration_one(self):
        llm = _LayerSnapshotLLM([_reply_tool("tc1", text="Let me look."), _reply_text("Done.")])
        sr2 = _make_sr2(llm, _pipeline())

        await _drive(sr2, "run a tool")

        assert len(llm.layer_snapshots) == 2
        after_iteration_one = llm.layer_snapshots[1]
        assert after_iteration_one[-2:] == [A("tc1", text="Let me look."), R("tc1")]

    @pytest.mark.asyncio
    async def test_ac4_r2_layer_assistant_message_carries_thinking_block(self):
        llm = _LayerSnapshotLLM(
            [_reply_tool("tc1", text="Let me look.", thinking="I should check."), _reply_text("Done.")]
        )
        sr2 = _make_sr2(llm, _pipeline())

        await _drive(sr2, "run a tool")

        after_iteration_one = llm.layer_snapshots[1]
        assert after_iteration_one[-2:] == [
            A("tc1", text="Let me look.", thinking="I should check."),
            R("tc1"),
        ]
        assert any(isinstance(b, ThinkingBlock) for b in after_iteration_one[-2].content)

    @pytest.mark.asyncio
    async def test_ac4_r3_next_turn_history_holds_full_assistant_message_then_results(self):
        llm = SequentialMockLLM(
            [_reply_tool("tc1", text="Let me look."), _reply_text("Done."), _reply_text("Ack.")]
        )
        sr2 = _make_sr2(llm, _pipeline())

        await _drive(sr2, "run a tool")
        await _drive(sr2, "next")

        _assert_request(
            llm.stream_calls[2],
            [
                U("run a tool"),
                A("tc1", text="Let me look."),
                R("tc1"),
                A_text("Done."),
                U("next"),
            ],
        )

    @pytest.mark.asyncio
    async def test_ac4_r4_two_iterations_layer_holds_a1_r1_a2_r2_once_each(self):
        llm = _LayerSnapshotLLM(
            [
                _reply_tool("tc1", text="First."),
                _reply_tool("tc2", text="Second."),
                _reply_text("Done."),
            ]
        )
        sr2 = _make_sr2(llm, _pipeline())

        await _drive(sr2, "run two tools")

        assert len(llm.layer_snapshots) == 3
        layer = llm.layer_snapshots[2]
        expected_tail = [A("tc1", text="First."), R("tc1"), A("tc2", text="Second."), R("tc2")]
        assert layer[-4:] == expected_tail
        for message in expected_tail:
            assert layer.count(message) == 1, f"duplicated in session layer: {message}"


# ===========================================================================
# AC5 — requests equal the pre-change hand-built requests (FR 6-7)
# ===========================================================================


@pytest.fixture
def stepping_clock(monkeypatch):
    """Each clock render advances one minute, so a re-rendered clock differs."""
    state = {"n": 0}

    def _now(tz=None):
        state["n"] += 1
        minutes = 9 * 60 + state["n"] - 1
        return datetime.datetime(
            2026, 7, 22, minutes // 60, minutes % 60, tzinfo=ZoneInfo("America/Phoenix")
        )

    monkeypatch.setattr(static_text_module.date, "now", _now)


class TestAC5RequestsEqualHandBuilt:
    @pytest.mark.asyncio
    async def test_ac5_r1_three_iteration_turn_with_live_shaped_layers(self, stepping_clock):
        llm = SequentialMockLLM(
            [
                _reply_tool("tc1", text="One."),
                _reply_tool("tc2", text="Two."),
                _reply_tool("tc3", text="Three."),
                _reply_text("Done."),
            ]
        )
        sr2 = _make_sr2(llm, _pipeline(with_clock=True))

        await _drive(sr2, "go")

        assert len(llm.stream_calls) == 4
        # The turn's compiled prefix: [clock, input]. The clock renders once,
        # at the turn's compile; its value is whatever that render produced.
        first = llm.stream_calls[0]
        assert len(first.messages) == 2
        clock_text = first.messages[0].content[0].text
        assert clock_text.startswith("CLOCK ")
        prefix = [U(clock_text), U("go")]

        _assert_request(llm.stream_calls[0], prefix)
        _assert_request(llm.stream_calls[1], prefix + [A("tc1", text="One."), R("tc1")])
        _assert_request(
            llm.stream_calls[2],
            prefix + [A("tc1", text="One."), R("tc1"), A("tc2", text="Two."), R("tc2")],
        )
        _assert_request(
            llm.stream_calls[3],
            prefix
            + [
                A("tc1", text="One."),
                R("tc1"),
                A("tc2", text="Two."),
                R("tc2"),
                A("tc3", text="Three."),
                R("tc3"),
            ],
        )

    @pytest.mark.asyncio
    async def test_ac5_r2_empty_first_iteration_retry_sends_the_original_request(self):
        llm = SequentialMockLLM([_reply_empty(), _reply_text("Recovered.")])
        sr2 = _make_sr2(llm, _pipeline())

        await _drive(sr2, "hello")

        assert len(llm.stream_calls) == 2
        _assert_request(llm.stream_calls[0], [U("hello")])
        _assert_request(llm.stream_calls[1], [U("hello")])
        assert llm.stream_calls[1] == llm.stream_calls[0]

    @pytest.mark.asyncio
    async def test_ac5_r2_empty_retry_after_tool_iteration_sends_the_same_request(self):
        """Same row on a request the tool loop built (not the turn's first)."""
        llm = SequentialMockLLM(
            [_reply_tool("tc1", text="Let me look."), _reply_empty(), _reply_text("Done.")]
        )
        sr2 = _make_sr2(llm, _pipeline())

        await _drive(sr2, "run a tool")

        assert len(llm.stream_calls) == 3
        expected = [U("run a tool"), A("tc1", text="Let me look."), R("tc1")]
        _assert_request(llm.stream_calls[1], expected)
        _assert_request(llm.stream_calls[2], expected)
        assert llm.stream_calls[2] == llm.stream_calls[1]

    @pytest.mark.asyncio
    async def test_ac5_r3_repeat_guard_note_replaces_result_in_next_request(self):
        note = "REPEATED: do not call this again."
        llm = SequentialMockLLM(
            [_reply_tool("tc1"), _reply_tool("tc2"), _reply_text("Done.")]
        )
        sr2 = _make_sr2(
            llm,
            _pipeline(
                repeat_guard=RepeatGuardConfig(
                    enabled=True, notes_before_stop=5, note=note, warning=""
                )
            ),
        )

        await _drive(sr2, "check twice")

        assert len(llm.stream_calls) == 3
        _assert_request(
            llm.stream_calls[2],
            [U("check twice"), A("tc1"), R("tc1"), A("tc2"), R("tc2", content=note)],
        )

    @pytest.mark.asyncio
    async def test_ac5_r4_forced_final_answer_request_appends_instruction_without_tools(self):
        instruction = "FINISH NOW."
        not_run = "NOT RUN."
        llm = SequentialMockLLM(
            [_reply_tool("tc1"), _reply_tool("tc2"), _reply_text("Final.")]
        )
        sr2 = _make_sr2(
            llm,
            _pipeline(
                max_tool_iterations=1,
                final_answer=FinalAnswerConfig(
                    enabled=True, instruction=instruction, not_run=not_run
                ),
            ),
        )

        await _drive(sr2, "loop")

        assert len(llm.stream_calls) == 3
        _assert_request(llm.stream_calls[1], [U("loop"), A("tc1"), R("tc1")])
        _assert_request(
            llm.stream_calls[2],
            [
                U("loop"),
                A("tc1"),
                R("tc1"),
                A("tc2"),
                R("tc2", content=not_run),
                U(instruction),
            ],
            tools=None,
        )


# ===========================================================================
# AC5-r5 — a transformer on the session layer changes the running turn (FR 6)
# ===========================================================================


class _DropOldestTransformer:
    """Test transformer: drops the oldest item of the layer's content."""

    def __init__(self, config: TransformerConfig) -> None:
        self.name = config.name or "drop_oldest"
        self.max_executions = config.max_executions
        self.execution_count = 0
        self.subscriptions = build_subscriptions(config.subscriptions, PHASE_MAP, [])

    @classmethod
    def build(cls, config, deps, *args):
        return cls(config)

    async def transform(self, content, events):
        self.execution_count += 1
        return TransformationResult(
            transformer_name="drop_oldest",
            source_layer="conversation",
            content=list(content[1:]),
        )


@pytest.fixture
def drop_oldest_registered(monkeypatch):
    original_get = orch._TRANSFORMERS.get

    def _get(name):
        if name == "test_drop_oldest":
            return _DropOldestTransformer
        return original_get(name)

    monkeypatch.setattr(orch._TRANSFORMERS, "get", _get)


class TestAC5TransformerChangesRunningTurn:
    @pytest.mark.asyncio
    async def test_ac5_r5_drop_oldest_on_tool_result_received_reflected_in_next_request(
        self, drop_oldest_registered
    ):
        llm = SequentialMockLLM(
            [_reply_text("Ack one."), _reply_tool("tc1", text="Let me look."), _reply_text("Done.")]
        )
        transformers = [
            TransformerConfig(
                type="test_drop_oldest",
                name="drop_oldest",
                max_executions=1,
                subscriptions=[
                    EventSubscriptionConfig(event="tool_result_received", phase="completed")
                ],
            )
        ]
        sr2 = _make_sr2(llm, _pipeline(transformers=transformers))

        await _drive(sr2, "turn one")
        await _drive(sr2, "run a tool")

        assert len(llm.stream_calls) == 3
        # Turn two's first request is the full compiled history.
        _assert_request(
            llm.stream_calls[1], [U("turn one"), A_text("Ack one."), U("run a tool")]
        )
        # The transformer dropped the oldest message ("turn one") when the
        # tool results arrived; the next request reflects it.
        _assert_request(
            llm.stream_calls[2],
            [A_text("Ack one."), U("run a tool"), A("tc1", text="Let me look."), R("tc1")],
        )
