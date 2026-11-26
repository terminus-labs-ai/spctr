"""spc-1062 AC5-r7..r9: the tripwire inside a real SR2 turn.

Spec agent-workflow-v3 D38: when the test runs keep showing the same failure
without progress, the harness writes a note to C; if it recurs, it ends the
turn with a harness-written escalation. Off by default.

Each test drives ``SR2.turn()`` over a scripted LLM whose every iteration
runs ``pytest tests/test_x.py -q`` and an executor that always returns the
same failing output, and asserts on the requests the model actually received
and the caller-visible events.
"""

from __future__ import annotations

from typing import Any

from sr2.config.models import PipelineConfig
from sr2.models import ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import StreamEvent
from conftest import SequentialMockLLM, make_minimal_config, make_user_input, tool_use_event

FAILING_ID = "tests/test_x.py::test_a"
FAILING_OUTPUT = (
    "F                                                                        [100%]\n"
    "E       assert 1 == 2\n"
    f"FAILED {FAILING_ID} - assert 1 == 2\n"
    "1 failed in 0.01s"
)
# More calls than the default 30-call span needs for two detections, so a
# tripwire that were on by default would stop the r9 turn.
ITERATIONS = 40

NOTE_AT_3 = (
    "Harness tripwire: your test runs have shown the same failure since call "
    f"1 without progress:\n- {FAILING_ID}\nDo not retry the same approach. If "
    "the cause is outside your code (tests, fixtures, contract or "
    "environment), stop now and escalate in your contract's format. If this "
    "failure recurs, the harness ends the run with an escalation."
)


class FailingExecutor:
    """Every call returns the same failing pytest output."""

    def __init__(self) -> None:
        self.order: list[str] = []

    async def __call__(self, block: ToolUseBlock) -> ToolResultBlock:
        self.order.append(block.id)
        return ToolResultBlock(tool_use_id=block.id, content=FAILING_OUTPUT)


def _config(tripwire: dict[str, Any] | None) -> PipelineConfig:
    raw: dict[str, Any] = make_minimal_config().model_dump()
    raw["max_tool_iterations"] = 50
    if tripwire is not None:
        raw["tripwire"] = tripwire
    return PipelineConfig(**raw)


async def _run(config: PipelineConfig):
    from sr2.orchestrator import SR2

    sequences = [
        [
            tool_use_event(
                tool_use_id=f"c{i}",
                tool_name="terminal",
                tool_input={"command": "cd /w && pytest tests/test_x.py -q 2>&1 | tail -5"},
            ),
            StreamEvent(type="end"),
        ]
        for i in range(1, ITERATIONS + 1)
    ]
    sequences.append([StreamEvent(type="text", text="FINAL"), StreamEvent(type="end")])
    llm = SequentialMockLLM(sequences)
    executor = FailingExecutor()
    sr2 = SR2(
        pipeline_config=config,
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=executor,
    )
    events = [e async for e in sr2.turn(make_user_input())]
    return events, llm, executor


def _results_in(request) -> dict[str, Any]:
    """tool_use_id → result content, for every tool result in one request."""
    return {
        block.tool_use_id: block.content
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


# Detection 1 at call 3 (same failure as call 1, two calls apart), detection 2
# at call 4: the note goes to call 3's result, the stop follows at call 4.
ENABLED = {"enabled": True, "min_span_calls": 2, "notes_before_stop": 1}


async def test_first_detection_note_reaches_model_and_turn_continues():
    """AC5-r7: the note is in the next request as text; the turn goes on."""
    events, llm, executor = await _run(_config(ENABLED))

    after_call_2 = _results_in(llm.stream_calls[2])
    assert after_call_2 == {"c1": FAILING_OUTPUT, "c2": FAILING_OUTPUT}
    after_call_3 = _results_in(llm.stream_calls[3])
    assert after_call_3["c3"] == f"{FAILING_OUTPUT}\n\n{NOTE_AT_3}"
    assert after_call_3["c1"] == FAILING_OUTPUT and after_call_3["c2"] == FAILING_OUTPUT
    assert "c4" in executor.order, "the turn continued after the note"
    tool_results = [r for e in events if e.type == "tool_result_received" for r in e.tool_results]
    assert [r.content for r in tool_results if r.tool_use_id == "c3"] == [
        f"{FAILING_OUTPUT}\n\n{NOTE_AT_3}"
    ]


async def test_second_detection_ends_turn_with_harness_escalation():
    """AC5-r8: the turn ends with a harness-written escalation; no more calls."""
    events, llm, executor = await _run(_config(ENABLED))

    assert executor.order == ["c1", "c2", "c3", "c4"], "no tool call runs after the stop"
    assert len(llm.stream_calls) == 4, "no LLM call is made after the stop"
    texts = [e.text for e in events if e.type == "text"]
    assert texts == [
        f"ESCALATION: tripwire\n- {FAILING_ID}\nThe harness ended the run: the same "
        "test failure repeated without progress from call 1 to call 4."
    ]
    assert [e.meta for e in events if e.type == "final_answer"] == [
        {"reason": "tripwire", "fallback_used": False}
    ]
    assert events[-1].type == "end"
    assert [e.type for e in events].count("end") == 1


async def test_no_tripwire_configuration_leaves_turn_untouched():
    """AC5-r9: off by default — every call runs, every result is real."""
    events, llm, executor = await _run(_config(None))

    assert executor.order == [f"c{i}" for i in range(1, ITERATIONS + 1)]
    assert len(llm.stream_calls) == ITERATIONS + 1
    assert _results_in(llm.stream_calls[-1]) == {
        f"c{i}": FAILING_OUTPUT for i in range(1, ITERATIONS + 1)
    }
    assert [e.text for e in events if e.type == "text"] == ["FINAL"]
    assert not [e for e in events if e.type == "final_answer"]


def test_tripwire_defaults():
    """The YAML block ``pipeline.tripwire``: off, 30-call span, one note."""
    tripwire = _config(None).tripwire
    assert tripwire.enabled is False
    assert tripwire.min_span_calls == 30
    assert tripwire.notes_before_stop == 1
    assert tripwire.tools == ["terminal"]
    assert tripwire.command_argument == "command"
    assert tripwire.command_pattern == r"pytest|run_tests"
