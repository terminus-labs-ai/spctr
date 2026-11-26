"""spc-1120 — bus-triggered summarisation in a live tool loop.

Spec: topics/sr2/spctr/specs/context-window-guard.md, FR 11-12, AC9 and the
NFR on documentation.

The pipeline is built from YAML in the FR 12 shape, scaled down
(``token_budget: 4000``, ``token_threshold_pct: 0.6``, ``keep_tokens: 1000``).
The real orchestrator, engine, bus, session resolver and
``SummarizationTransformer`` run; only the LLM and the tool executor are
fakes. Each tool result is 3200 characters, which ``CharacterTokenCounter``
counts as 800 tokens, so the session content passes the 2400-token
threshold after the third tool result.

Expected requests are written out from each case's construction, never
read back from the code under test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from sr2.config.models import PipelineConfig
from sr2.models import Message, TextBlock, TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent

from conftest import SequentialMockLLM, make_user_input, tool_use_event

SYSTEM_TEXT = "You are a summarisation loop test assistant."
PROMPT = "Build the thing."
TOOL_NAME = "read_file"
TOOL_INPUT = {"path": "a"}
RESULT_CHARS = 3200  # 800 tokens at CharacterTokenCounter's chars // 4
HEADER = "[Summary of {count} earlier messages]"
SUMMARY_TEXT = "S"
SUMMARY_PROMPT = "Summarise the earlier tool work."

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


def _pipeline(*, with_transformer: bool = True) -> PipelineConfig:
    raw = yaml.safe_load(FR12_YAML)
    if not with_transformer:
        del raw["layers"][1]["transformers"]
    return PipelineConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _LoopLLM(SequentialMockLLM):
    """Scripted stream replies; ``complete`` is the summary call."""

    def __init__(self, call_sequences, *, summary_raises: bool = False) -> None:
        super().__init__(call_sequences)
        self.summary_raises = summary_raises
        self.complete_calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.complete_calls.append(request)
        if self.summary_raises:
            raise RuntimeError("summary backend down")
        return CompletionResponse(
            id="summary",
            content=[TextBlock(text=SUMMARY_TEXT)],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


def _result_text(tc_id: str) -> str:
    return (tc_id + "-") * (RESULT_CHARS // (len(tc_id) + 1))


async def _executor(block: ToolUseBlock) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=block.id, content=_result_text(block.id))


def _reply_tool(tc_id: str):
    return [
        tool_use_event(tool_use_id=tc_id, tool_name=TOOL_NAME, tool_input=dict(TOOL_INPUT)),
        StreamEvent(type="end"),
    ]


def _reply_text(text: str):
    return [StreamEvent(type="text", text=text), StreamEvent(type="end")]


def _script(n_tools: int):
    return [_reply_tool(f"tc{i}") for i in range(1, n_tools + 1)] + [_reply_text("Done.")]


async def _run(llm: _LoopLLM, pipeline: PipelineConfig) -> None:
    sr2 = SR2(
        pipeline_config=pipeline,
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=_executor,
    )
    async for _ in sr2.turn(make_user_input(PROMPT)):
        pass


# Expected messages ------------------------------------------------------------


def U(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


def A(tc_id: str) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseBlock(id=tc_id, name=TOOL_NAME, input=dict(TOOL_INPUT))],
    )


def R(tc_id: str) -> Message:
    return Message(
        role="user",
        content=[ToolResultBlock(tool_use_id=tc_id, content=_result_text(tc_id))],
    )


SYSTEM = [TextBlock(text=SYSTEM_TEXT)]


def test_fixture_sizes_match_the_case_table():
    counter = CharacterTokenCounter()
    assert counter.count([R("tc1")]) == 800
    # Two iterations stay under the 2400-token threshold, three pass it.
    two = [U(PROMPT), A("tc1"), R("tc1"), A("tc2"), R("tc2")]
    assert counter.count(two) < 2400
    assert counter.count(two + [A("tc3"), R("tc3")]) >= 2400


# ===========================================================================
# AC9 — FR 12 config: the next request carries the summary in place of the span
# ===========================================================================


class TestAC9ToolLoopSummarisation:
    @pytest.mark.asyncio
    async def test_ac9_r1_request_after_threshold_has_summary_in_place_of_span(self):
        llm = _LoopLLM(_script(3))

        await _run(llm, _pipeline())

        assert len(llm.stream_calls) == 4
        # Requests before the threshold are unsummarised.
        assert llm.stream_calls[2].messages == [U(PROMPT), A("tc1"), R("tc1"), A("tc2"), R("tc2")]
        # One summary call, carrying the span (iterations 1-2) and not the kept window.
        assert len(llm.complete_calls) == 1
        summary_req = llm.complete_calls[0]
        assert summary_req.system == [TextBlock(text=SUMMARY_PROMPT)]
        span_text = summary_req.messages[0].content[0].text
        assert _result_text("tc1") in span_text and _result_text("tc2") in span_text
        assert _result_text("tc3") not in span_text
        assert PROMPT not in span_text
        # The first request after content passes 2400 tokens: prompt, one
        # summary message (header + "S"), kept window; earlier tool messages gone.
        after = llm.stream_calls[3]
        assert after.system == SYSTEM
        assert after.messages == [
            U(PROMPT),
            U("[Summary of 4 earlier messages]\n\nS"),
            A("tc3"),
            R("tc3"),
        ]

    @pytest.mark.asyncio
    async def test_ac9_r2_threshold_never_reached_makes_no_summary_call(self):
        llm = _LoopLLM(_script(2))
        baseline = _LoopLLM(_script(2))

        await _run(llm, _pipeline())
        await _run(baseline, _pipeline(with_transformer=False))

        assert llm.complete_calls == []
        assert len(llm.stream_calls) == 3
        assert llm.stream_calls[2].messages == [U(PROMPT), A("tc1"), R("tc1"), A("tc2"), R("tc2")]
        assert [(c.system, c.messages, c.tools) for c in llm.stream_calls] == [
            (c.system, c.messages, c.tools) for c in baseline.stream_calls
        ]

    @pytest.mark.asyncio
    async def test_ac9_r3_summary_call_raises_loop_continues_unsummarised(self, caplog):
        llm = _LoopLLM(_script(3), summary_raises=True)

        with caplog.at_level(logging.WARNING, logger="sr2.pipeline.transformers.summarization"):
            await _run(llm, _pipeline())

        assert len(llm.complete_calls) == 1
        assert len(llm.stream_calls) == 4
        assert llm.stream_calls[3].messages == [
            U(PROMPT),
            A("tc1"),
            R("tc1"),
            A("tc2"),
            R("tc2"),
            A("tc3"),
            R("tc3"),
        ]
        # spc-1125 adds a wiring warning from the same logger at build time, so
        # count the warnings this row is about: the failed summary call.
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == "sr2.pipeline.transformers.summarization"
            and "content left unchanged" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert "summary backend down" in warnings[0].getMessage()


# ===========================================================================
# AC9-r4 — config.example.yaml documents the FR 12 shape and validates
# ===========================================================================


EXAMPLE = Path(__file__).resolve().parents[2] / "spctr" / "config.example.yaml"


def test_ac9_r4_config_example_loads_and_documents_fr12_block():
    from spctr.config import load_config

    cfg = load_config(EXAMPLE)

    layers = {layer.name: layer for layer in cfg.pipeline.layers}
    session = next(
        layer for layer in layers.values() if any(r.type == "session" for r in layer.resolvers)
    )
    assert session.target == "messages"
    assert session.token_budget == 190000
    assert session.token_threshold_pct == 0.6
    assert len(session.transformers) == 1
    compact = session.transformers[0]
    assert compact.type == "summarize"  # registered key (pyproject entry point)
    assert [(s.event, s.phase) for s in compact.subscriptions] == [("token_threshold", None)]
    assert compact.config["keep_strategy"] == "keep_within_tokens"
    assert compact.config["keep_tokens"] == 50000
    assert compact.config["retry_growth_tokens"] == 10000
    assert isinstance(compact.config["summarization_prompt"], str)
    assert compact.config["summarization_prompt"].strip()
    assert compact.config["summary_header"] == "[Summary of {count} earlier messages]"
    # The guard field is documented next to the model it applies to.
    assert "context_safety_ratio" in EXAMPLE.read_text()
