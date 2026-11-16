"""Tests for sr2-122 — consecutive repeat guard in the tool loop.

Spec: role-run-stop-rules (spc-959), FR 7-18 and 22, AC 4-6.

With ``pipeline.repeat_guard.enabled``, every executed tool call still runs;
an identical call (same tool, same sorted-key JSON arguments as the previous
executed call) with the same result gets ``note`` instead of its real
result, ``warning`` is appended on repeat ``notes_before_stop``, and the
next same-result repeat ends the turn through the sr2-121 forced final
answer with reason ``repeat`` — regardless of ``final_answer.enabled``.
``max_identical_calls`` consecutive identical calls end the turn whatever
the results. With the block absent or disabled, results are never altered.

Case table rows: AC4-r1..r6, AC5-r1..r4, AC6-r1..r7, plus config defaults.
"""

from __future__ import annotations

from typing import Any

import pytest

from sr2.config.models import (
    DEFAULT_FINAL_ANSWER_FALLBACK,
    DEFAULT_FINAL_ANSWER_INSTRUCTION,
    DEFAULT_FINAL_ANSWER_NOT_RUN,
    DEFAULT_REPEAT_NOTE,
    DEFAULT_REPEAT_WARNING,
    PipelineConfig,
    RepeatGuardConfig,
)
from sr2.models import TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    tool_use_event,
)

REASON_1_4 = "you repeated call 1 4 times"
REASON_1_10 = "you repeated call 1 10 times"
REASON_1_5 = "you repeated call 1 5 times"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**pipeline_kwargs: Any) -> PipelineConfig:
    """Minimal pipeline config. The case-table default supplies no
    ``final_answer`` block — the repeat stop must force the final answer
    anyway (FR 11)."""
    raw: dict[str, Any] = make_minimal_config().model_dump()
    raw.update(pipeline_kwargs)
    return PipelineConfig(**raw)


def _guard(**overrides: Any) -> dict[str, Any]:
    """repeat_guard block for the pipeline config dict (case-table default)."""
    return {"enabled": True, "notes_before_stop": 2, "max_identical_calls": 10, **overrides}


SHELL_INPUT = {"command": "pytest"}


def _x(call_id: str) -> StreamEvent:
    """Case-table X: shell {"command": "pytest"}."""
    return tool_use_event(tool_use_id=call_id, tool_name="shell", tool_input=dict(SHELL_INPUT))


def _iter_events(*calls: StreamEvent) -> list[StreamEvent]:
    return list(calls) + [StreamEvent(type="end")]


def _iter_text_events(text: str, *calls: StreamEvent) -> list[StreamEvent]:
    """A tool iteration whose reply carries text before the calls (AC9-r3)."""
    return [StreamEvent(type="text", text=text)] + list(calls) + [StreamEvent(type="end")]


def _text_reply(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text", text=text), StreamEvent(type="end")]


class ScriptedExecutor:
    """Fake executor running one scripted result per call id.

    Each entry maps a tool_use id to either a result content string or a
    ``(content, is_error)`` tuple. Every call for an id re-serves that id's
    scripted entry: the lookup is by id, so there is no positional advance
    through the script. A call for an id the script does not list raises
    ``KeyError``.
    """

    def __init__(self, script: dict[str, Any]) -> None:
        self.script = script
        self.order: list[str] = []

    async def __call__(self, block: ToolUseBlock) -> ToolResultBlock:
        self.order.append(block.id)
        spec = self.script[block.id]
        content, is_error = spec if isinstance(spec, tuple) else (spec, False)
        return ToolResultBlock(tool_use_id=block.id, content=content, is_error=is_error)


def _build_sr2(llm, executor, config: PipelineConfig):
    from sr2.orchestrator import SR2

    return SR2(
        pipeline_config=config,
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=executor,
    )


async def _run_turn(
    config: PipelineConfig,
    sequences: list[list[StreamEvent]],
    script: dict[str, Any],
    *,
    final_text: str = "FINAL",
    repeat_last: bool = False,
):
    """Drive a turn: scripted LLM sequences, then a text reply ending the
    turn (appended unless the caller's sequences already end in text).

    Returns (events, llm, executor).
    """
    seqs = list(sequences)
    if seqs[-1][0].type == "tool_use":
        # Two identical text replies: the guard-stop path consumes the reply
        # after its stop iteration as the forced final answer's reply; turns
        # that never stop simply never reach the second one.
        seqs = seqs + [_text_reply(final_text)] * 2
    executor = ScriptedExecutor(script)
    llm = SequentialMockLLM(seqs, repeat_last=repeat_last)
    sr2 = _build_sr2(llm, executor, config)
    events = [e async for e in sr2.turn(make_user_input())]
    return events, llm, executor


def _note(first_call: int, repeat: int, note: str = DEFAULT_REPEAT_NOTE) -> str:
    return note.format(first_call=first_call, repeat=repeat)


def _expected_result(call_id: str, script: dict[str, Any], notes: dict[str, str]) -> str:
    """What the model must see as call ``call_id``'s result: its note when
    the guard replaced it, else its scripted real content."""
    if call_id in notes:
        return notes[call_id]
    spec = script[call_id]
    return spec[0] if isinstance(spec, tuple) else spec


def _results_seen(llm: SequentialMockLLM) -> list[str]:
    """Every tool_result content the model was ever shown, in order.

    Requests are cumulative, so each request after the first adds exactly
    its own new results — except the guard-stop path's forced call, which
    re-sends the pre-forced request: skip it by message identity.
    """
    seen: list[str] = []
    prev_messages: list | None = None
    for request in llm.stream_calls[1:]:
        if request.messages is prev_messages or (
            prev_messages is not None and len(request.messages) <= len(prev_messages)
        ):
            # Re-sent (or shorter) history: the forced call. Contributes nothing.
            prev_messages = request.messages
            continue
        for message in request.messages[len(prev_messages) :] if prev_messages else request.messages:
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    seen.append(block.content)
        prev_messages = request.messages
    return seen


def _assert_results(
    llm: SequentialMockLLM, script: dict[str, Any], notes: dict[str, str]
) -> None:
    expected = [_expected_result(cid, script, notes) for cid in script]
    assert _results_seen(llm) == expected


def _rg_events(events: list[StreamEvent]) -> list[StreamEvent]:
    return [e for e in events if e.type == "repeat_guard"]


def _fa_events(events: list[StreamEvent]) -> list[StreamEvent]:
    return [e for e in events if e.type == "final_answer"]


def _collect_bus(sr2_instance: Any, event_name: str) -> list[Any]:
    """Subscribe to the engine bus for event_name before the turn (test_bus_events idiom)."""
    collected: list[Any] = []
    sr2_instance._engine.bus.subscribe(event_name, lambda e: collected.append(e))
    return collected


def _final_text(events: list[StreamEvent]) -> str:
    """The turn's final text: the text events after the last end event."""
    last_end = max(i for i, e in enumerate(events) if e.type == "end")
    return "".join(e.text for e in events[last_end - 1 : -1] if e.type == "text")


def _malformed_event(call_id: str) -> StreamEvent:
    """A tool_use event flagged invalid by the LLM integration (obsidian-1fmv)."""
    return StreamEvent(
        type="tool_use",
        tool_use_id=call_id,
        tool_name="shell",
        tool_input={},
        meta={"invalid_arguments": True, "error": "Expecting value: line 1 column 1"},
    )


# ---------------------------------------------------------------------------
# AC4 — note, warning, then stop through the forced final answer
# ---------------------------------------------------------------------------


class TestAC4NoteWarningStop:
    @pytest.mark.asyncio
    async def test_ac4_r1_second_call_gets_the_note_no_warning(self):
        """AC4-r1: X, X (same result). Executor runs twice; the model sees
        call 2 as the note naming call 1, with no warning appended."""
        script = {"c1": "1 failed", "c2": "1 failed"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()), [_iter_events(_x("c1")), _iter_events(_x("c2"))], script
        )

        assert executor.order == ["c1", "c2"], "every call still executes"
        _assert_results(llm, script, {"c2": _note(1, 1)})
        assert "Do not repeat it." in _results_seen(llm)[-1]
        assert "ends the run" not in _results_seen(llm)[-1], "no warning on repeat 1"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac4_r2_repeat_two_note_plus_warning(self):
        """AC4-r2: X ×3 (same). Call 3's note names call 1 and ends with the
        warning for repeat 2."""
        script = {"c1": "1 failed", "c2": "1 failed", "c3": "1 failed"}
        expected = _note(1, 2) + DEFAULT_REPEAT_WARNING.format(repeat=2)
        assert expected.endswith(" This is repeat 2; the next identical call ends the run with an escalation.")
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x("c1")), _iter_events(_x("c2")), _iter_events(_x("c3"))],
            script,
        )

        assert executor.order == ["c1", "c2", "c3"]
        _assert_results(llm, script, {"c2": _note(1, 1), "c3": expected})

    @pytest.mark.asyncio
    async def test_ac4_r3_fourth_call_stops_via_forced_final_answer(self):
        """AC4-r3: X ×4 (same). Turn ends through the forced final answer:
        next request tools=None, reason filled, one final_answer event with
        reason "repeat" — with no final_answer config block at all.

        The stop note is the repeat-3 form of a custom note that includes
        ``{repeat}``, and is asserted to differ from its repeat-2 form
        (the default note has no ``{repeat}``, so the two would be
        indistinguishable)."""
        custom_note = "DUP {first_call} r{repeat} END"
        script = {"c1": "1 failed", "c2": "1 failed", "c3": "1 failed", "c4": "1 failed"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard(note=custom_note)),
            [_iter_events(_x(f"c{i}")) for i in range(1, 5)],
            script,
        )

        assert executor.order == ["c1", "c2", "c3", "c4"]
        assert len(llm.stream_calls) == 5, "4 iterations plus exactly one forced call"
        forced_request = llm.stream_calls[4]
        # The forced request pairs c4 (the stop call, repeat 3) with its
        # stop note. Repeat 3 is not notes_before_stop (2), so no warning
        # is appended: the note is exactly the repeat-3 form, and it is
        # distinguishable from the repeat-2 form because the custom note
        # carries {repeat}.
        c4_result = next(
            block.content
            for message in forced_request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.tool_use_id == "c4"
        )
        assert c4_result == custom_note.format(first_call=1, repeat=3)
        assert c4_result != custom_note.format(first_call=1, repeat=2)
        assert forced_request.tools is None
        instruction = "".join(
            b.text
            for b in forced_request.messages[-1].content
            if isinstance(b, TextBlock)
        )
        assert instruction == DEFAULT_FINAL_ANSWER_INSTRUCTION.format(reason=REASON_1_4)
        fa = _fa_events(events)
        assert len(fa) == 1 and fa[0].meta == {"reason": "repeat", "fallback_used": False}
        assert _final_text(events) == "FINAL"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac4_r4_note_names_the_previous_run(self):
        """AC4-r4: A, X, X. The note on call 3 says 'identical to call 2'."""
        a = tool_use_event(tool_use_id="c1", tool_name="read", tool_input={"path": "a"})
        script = {"c1": "src", "c2": "1 failed", "c3": "1 failed"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(a), _iter_events(_x("c2")), _iter_events(_x("c3"))],
            script,
        )

        _assert_results(llm, script, {"c3": _note(2, 1)})

    @pytest.mark.asyncio
    async def test_ac4_r5_argument_key_order_is_irrelevant(self):
        """AC4-r5: {"a":1,"b":2} then {"b":2,"a":1}, same tool and result:
        identical — the second call gets the note."""
        c1 = tool_use_event(tool_use_id="c1", tool_name="shell", tool_input={"a": 1, "b": 2})
        c2 = tool_use_event(tool_use_id="c2", tool_name="shell", tool_input={"b": 2, "a": 1})
        script = {"c1": "same", "c2": "same"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()), [_iter_events(c1), _iter_events(c2)], script
        )

        _assert_results(llm, script, {"c2": _note(1, 1)})

    @pytest.mark.asyncio
    async def test_ac4_r6_custom_note_placeholders(self):
        """AC4-r6: custom note 'DUP {first_call}/{repeat}' — call 2's result
        is exactly 'DUP 1/1'."""
        script = {"c1": "1 failed", "c2": "1 failed"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard(note="DUP {first_call}/{repeat}")),
            [_iter_events(_x("c1")), _iter_events(_x("c2"))],
            script,
        )

        _assert_results(llm, script, {"c2": "DUP 1/1"})


# ---------------------------------------------------------------------------
# AC5 — different results, and the identical-call cap
# ---------------------------------------------------------------------------


class TestAC5DifferentResultsAndCap:
    @pytest.mark.asyncio
    async def test_ac5_r1_different_result_passes_and_resets(self):
        """AC5-r1: X→waiting, X→done, X→done. Call 2 gets its real 'done';
        call 3 gets the note with repeat 1."""
        script = {"c1": "waiting", "c2": "done", "c3": "done"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 4)],
            script,
        )

        assert executor.order == ["c1", "c2", "c3"]
        _assert_results(llm, script, {"c3": _note(2, 1)})

    @pytest.mark.asyncio
    async def test_ac5_r2_max_identical_calls_stops_whatever_the_results(self):
        """AC5-r2: X ×10 with distinct results. All 10 execute with their
        real results; after call 10 the turn ends with reason
        'you repeated call 1 10 times'."""
        script = {f"c{i}": f"r{i}" for i in range(1, 11)}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 11)],
            script,
        )

        assert executor.order == [f"c{i}" for i in range(1, 11)]
        assert len(llm.stream_calls) == 11, "10 iterations plus exactly one forced call"
        forced_request = llm.stream_calls[10]
        assert forced_request.tools is None
        instruction = "".join(
            b.text
            for b in forced_request.messages[-1].content
            if isinstance(b, TextBlock)
        )
        assert instruction == DEFAULT_FINAL_ANSWER_INSTRUCTION.format(reason=REASON_1_10)
        fa = _fa_events(events)
        assert len(fa) == 1 and fa[0].meta == {"reason": "repeat", "fallback_used": False}
        rg = _rg_events(events)
        assert len(rg) == 1 and rg[0].meta["action"] == "stop"
        assert _final_text(events) == "FINAL"
        # The stop fired on call 10: its real result r10 is what the model is
        # shown (FR 13 as amended 2026-10-20) — the forced request's history
        # ends with the executed iteration's results, and no call of it is
        # final_answer.not_run (AC9).
        history_results = [
            block
            for message in forced_request.messages[:-1]
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert [r.tool_use_id for r in history_results] == [f"c{i}" for i in range(1, 11)]
        assert history_results[-1].content == "r10"
        not_run = [
            block
            for message in forced_request.messages[-2:]
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert not_run[0].tool_use_id == "c10"
        assert not_run[0].content == "r10"
        assert DEFAULT_FINAL_ANSWER_NOT_RUN not in [r.content for r in history_results]

    @pytest.mark.asyncio
    async def test_ac5_r3_comparison_uses_the_real_previous_result(self):
        """AC5-r3: X→x, X→x, X→x. Call 3 is compared against the real 'x',
        never the note: it gets note plus warning (repeat 2)."""
        script = {"c1": "x", "c2": "x", "c3": "x"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 4)],
            script,
        )

        _assert_results(
            llm,
            script,
            {"c2": _note(1, 1), "c3": _note(1, 2) + DEFAULT_REPEAT_WARNING.format(repeat=2)},
        )

    @pytest.mark.asyncio
    async def test_ac5_r5_different_result_restarts_the_same_result_chain(self):
        """AC1(a) / FR 12: X→x, X→x, X→y, X→x, X→x (5 calls). The different
        result on call 3 (y) resets the same-result chain at call 3. Call 4
        (x) is identical to call 3 (y)? No — different result again, so the
        chain resets at call 4. Call 5 (x) is same as call 4 (x): repeat 1,
        first_call 4. A ``no_same_result_reset`` mutant would keep first_call
        1 throughout, giving repeat 3 on call 5 and stopping at call 5
        (repeat 3 > notes_before_stop 2). The absent stop event and
        first_call 4 in the call-5 meta are the kill signals."""
        script = {"c1": "x", "c2": "x", "c3": "y", "c4": "x", "c5": "x"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 6)],
            script,
        )

        rg = _rg_events(events)
        # Call 2: same as call 1 → repeat 1, first_call 1.
        assert len(rg) >= 1
        assert rg[0].meta == {
            "call": 2,
            "first_call": 1,
            "identical_calls": 2,
            "same_result_repeats": 1,
            "action": "note",
        }
        # Call 3: different result (y) — real result, no event. Chain resets
        # at call 3.
        # Call 4: identical to call 3, but different result (x vs y) — real
        # result, no event. Chain resets at call 4.
        # Call 5: identical to call 4, same result (x vs x) — repeat 1,
        # first_call 4. No stop (repeat 1 is not > 2).
        assert len(rg) == 2, f"expected exactly 2 repeat_guard events (calls 2 and 5); got {[e.meta for e in rg]}"
        assert rg[1].meta == {
            "call": 5,
            "first_call": 4,
            "identical_calls": 5,
            "same_result_repeats": 1,
            "action": "note",
        }
        _assert_results(
            llm,
            script,
            {
                "c2": _note(1, 1),
                "c3": "y",
                "c5": _note(4, 1),
            },
        )

    @pytest.mark.asyncio
    async def test_ac5_r6_same_arguments_different_tool_name_is_not_identical(self):
        """AC1(b) / FR 8: identical arguments on two different tools are not
        an identical call — both get their real results, no note, no
        repeat_guard event. A guard keyed on arguments alone
        (``args_only_key``) would note call 2."""
        c1 = tool_use_event(tool_use_id="c1", tool_name="shell", tool_input=dict(SHELL_INPUT))
        c2 = tool_use_event(tool_use_id="c2", tool_name="bash", tool_input=dict(SHELL_INPUT))
        script = {"c1": "1 failed", "c2": "1 failed"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()), [_iter_events(c1), _iter_events(c2)], script
        )

        _assert_results(llm, script, {})
        assert _rg_events(events) == []

    @pytest.mark.asyncio
    async def test_ac5_r4_is_error_difference_is_a_different_result(self):
        """AC5-r4: same content, is_error false then true: not the same
        result — call 2 gets its real result."""
        script = {"c1": "boom", "c2": ("boom", True)}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x("c1")), _iter_events(_x("c2"))],
            script,
        )

        _assert_results(llm, script, {})
        assert _rg_events(events) == []


# ---------------------------------------------------------------------------
# AC6 — resets, disabled behaviour, stream events
# ---------------------------------------------------------------------------


class TestAC6ResetsDisabledAndEvents:
    @pytest.mark.asyncio
    async def test_ac6_r1_different_tool_resets_both_counts(self):
        """AC6-r1: X, X, Y, Y. Call 3 (Y) is a real result; call 4's note
        says 'identical to call 3', repeat 1."""
        y1 = tool_use_event(tool_use_id="c3", tool_name="grep", tool_input={"pattern": "x"})
        y2 = tool_use_event(tool_use_id="c4", tool_name="grep", tool_input={"pattern": "x"})
        script = {"c1": "1 failed", "c2": "1 failed", "c3": "hit", "c4": "hit"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [
                _iter_events(_x("c1")),
                _iter_events(_x("c2")),
                _iter_events(y1),
                _iter_events(y2),
            ],
            script,
        )

        _assert_results(llm, script, {"c2": _note(1, 1), "c4": _note(3, 1)})
        assert [e.meta for e in _rg_events(events)] == [
            {"call": 2, "first_call": 1, "identical_calls": 2, "same_result_repeats": 1, "action": "note"},
            {"call": 4, "first_call": 3, "identical_calls": 2, "same_result_repeats": 1, "action": "note"},
        ]

    @pytest.mark.asyncio
    async def test_ac6_r2_different_arguments_no_note(self):
        """AC6-r2: 'pytest' then 'pytest -x': both real results, no note."""
        c2 = tool_use_event(
            tool_use_id="c2", tool_name="shell", tool_input={"command": "pytest -x"}
        )
        script = {"c1": "1 failed", "c2": "2 failed"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()), [_iter_events(_x("c1")), _iter_events(c2)], script
        )

        _assert_results(llm, script, {})
        assert _rg_events(events) == []

    @pytest.mark.asyncio
    async def test_ac6_r3_guard_absent_results_never_altered(self):
        """AC6-r3: no repeat_guard block; X ×5 same result. All 5 real, no
        repeat_guard events, the turn continues normally."""
        script = {f"c{i}": "1 failed" for i in range(1, 6)}
        events, llm, executor = await _run_turn(
            _config(), [_iter_events(_x(f"c{i}")) for i in range(1, 6)], script
        )

        assert executor.order == [f"c{i}" for i in range(1, 6)]
        _assert_results(llm, script, {})
        assert _rg_events(events) == []
        assert _fa_events(events) == []
        assert _final_text(events) == "FINAL"

    @pytest.mark.asyncio
    async def test_ac6_r4_repeat_guard_event_meta_sequence(self):
        """AC6-r4: the AC4-r3 sequence emits three repeat_guard events with
        the specified meta, the last one action 'stop'."""
        script = {f"c{i}": "1 failed" for i in range(1, 5)}
        events, _, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 5)],
            script,
        )

        rg = _rg_events(events)
        assert [e.meta for e in rg] == [
            {"call": 2, "first_call": 1, "identical_calls": 2, "same_result_repeats": 1, "action": "note"},
            {"call": 3, "first_call": 1, "identical_calls": 3, "same_result_repeats": 2, "action": "note"},
            {"call": 4, "first_call": 1, "identical_calls": 4, "same_result_repeats": 3, "action": "stop"},
        ]

    @pytest.mark.asyncio
    async def test_ac6_r5_two_identical_calls_in_one_message(self):
        """AC6-r5: one assistant message with two X calls, same result: the
        second gets the note 'identical to call 1'."""
        script = {"c1": "1 failed", "c2": "1 failed"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()), [_iter_events(_x("c1"), _x("c2"))], script
        )

        assert executor.order == ["c1", "c2"]
        _assert_results(llm, script, {"c2": _note(1, 1)})
        assert [e.meta for e in _rg_events(events)] == [
            {"call": 2, "first_call": 1, "identical_calls": 2, "same_result_repeats": 1, "action": "note"}
        ]

    @pytest.mark.asyncio
    async def test_ac6_r6_malformed_call_not_numbered_no_reset(self):
        """AC6-r6: X, malformed call, X. The malformed call is not numbered
        and does not reset: the second X is call 2 and gets the note
        'identical to call 1'."""
        script = {"c1": "1 failed", "c3": "1 failed"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x("c1")), _iter_events(_malformed_event("c2")), _iter_events(_x("c3"))],
            script,
        )

        assert executor.order == ["c1", "c3"], "the malformed call never executes"
        # The malformed call's explanatory result is real and passes through;
        # the guard never numbers it (FR 7).
        malformed_note = (
            "Tool call 'shell' was not executed: its arguments are invalid JSON "
            "(parse error: Expecting value: line 1 column 1). "
            "Retry with valid JSON arguments."
        )
        notes = {"c2": malformed_note, "c3": _note(1, 1)}
        full_script = dict(script)
        full_script["c2"] = malformed_note
        # Results reach the model in message order: c1's real result, then
        # the malformed call's explanation, then c3's note.
        ordered_ids = ["c1", "c2", "c3"]
        expected = [_expected_result(cid, full_script, notes) for cid in ordered_ids]
        assert _results_seen(llm) == expected
        assert [e.meta["call"] for e in _rg_events(events)] == [2]

    @pytest.mark.asyncio
    async def test_ac6_r7_guard_state_is_per_turn(self):
        """AC6-r7: turn 1 ends after X; turn 2 starts with X, same result:
        turn 2's X gets its real result."""
        script = {"c1": "1 failed", "c2": "1 failed"}
        executor = ScriptedExecutor(script)
        llm = SequentialMockLLM(
            [_iter_events(_x("c1")), _text_reply("T1"), _iter_events(_x("c2")), _text_reply("T2")]
        )
        sr2 = _build_sr2(llm, executor, _config(repeat_guard=_guard()))

        events1 = [e async for e in sr2.turn(make_user_input())]
        events2 = [e async for e in sr2.turn(make_user_input())]

        assert _final_text(events1) == "T1" and _final_text(events2) == "T2"
        assert executor.order == ["c1", "c2"]
        _assert_results(llm, script, {})
        assert _rg_events(events1 + events2) == []
        await sr2.aclose()


# ---------------------------------------------------------------------------
# AC1 — non-default thresholds, full stop meta, fallback, explicit disable
# ---------------------------------------------------------------------------


class TestAC1ThresholdsMetaFallbackDisabled:
    @pytest.mark.asyncio
    async def test_ac1_c_non_default_thresholds_note_at_4_stop_at_5(self):
        """AC1(c): ``notes_before_stop=3, max_identical_calls=5`` — X ×5
        (same). A guard hardcoded to the defaults (2 / 10) would stop at call
        4 and emit a warning on repeat 2; here the warning lands at call 4
        (repeat 3) and the stop at call 5 (repeat 4). Pinned by the full
        per-event meta sequence, the note texts, and the forced-call count."""
        script = {f"c{i}": "1 failed" for i in range(1, 6)}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard(notes_before_stop=3, max_identical_calls=5)),
            [_iter_events(_x(f"c{i}")) for i in range(1, 6)],
            script,
        )

        assert executor.order == ["c1", "c2", "c3", "c4", "c5"], "call 6 never executes"
        assert len(llm.stream_calls) == 6, "5 iterations plus exactly one forced call"
        assert [e.meta for e in _rg_events(events)] == [
            {"call": 2, "first_call": 1, "identical_calls": 2, "same_result_repeats": 1, "action": "note"},
            {"call": 3, "first_call": 1, "identical_calls": 3, "same_result_repeats": 2, "action": "note"},
            # Warning on the repeat whose count equals notes_before_stop (3).
            {"call": 4, "first_call": 1, "identical_calls": 4, "same_result_repeats": 3, "action": "note"},
            {"call": 5, "first_call": 1, "identical_calls": 5, "same_result_repeats": 4, "action": "stop"},
        ]
        _assert_results(
            llm,
            script,
            {
                "c2": _note(1, 1),
                "c3": _note(1, 2),
                "c4": _note(1, 3) + DEFAULT_REPEAT_WARNING.format(repeat=3),
                # The stop note carries the warning's repeat (3), per the
                # note-plus-warning shape pinned by test_ac4_r3.
                "c5": _note(1, 4),
            },
        )
        forced_request = llm.stream_calls[5]
        assert forced_request.tools is None
        instruction = "".join(
            b.text
            for b in forced_request.messages[-1].content
            if isinstance(b, TextBlock)
        )
        assert instruction == DEFAULT_FINAL_ANSWER_INSTRUCTION.format(reason=REASON_1_5)
        fa = _fa_events(events)
        assert len(fa) == 1 and fa[0].meta == {"reason": "repeat", "fallback_used": False}
        assert _final_text(events) == "FINAL"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac1_d_cap_stop_full_meta(self):
        """AC1(d): a ``max_identical_calls=2`` stop (distinct results)
        asserts the stop event's meta field by field — not just ``action`` —
        plus the forced request and the ``final_answer`` event's
        reason/fallback_used."""
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard(max_identical_calls=2)),
            [_iter_events(_x("c1")), _iter_events(_x("c2"))],
            {"c1": "r1", "c2": "r2"},
        )

        assert executor.order == ["c1", "c2"]
        rg = _rg_events(events)
        assert len(rg) == 1
        # first_call is the restart point of the same-result chain: the
        # different result on call 2 restarts it at call 2, and
        # same_result_repeats is 0 (the cap stop is result-independent).
        assert rg[0].meta == {
            "call": 2,
            "first_call": 2,
            "identical_calls": 2,
            "same_result_repeats": 0,
            "action": "stop",
        }
        assert len(llm.stream_calls) == 3, "2 iterations plus exactly one forced call"
        forced_request = llm.stream_calls[2]
        assert forced_request.tools is None
        instruction = "".join(
            b.text
            for b in forced_request.messages[-1].content
            if isinstance(b, TextBlock)
        )
        assert instruction == DEFAULT_FINAL_ANSWER_INSTRUCTION.format(
            reason="you repeated call 1 2 times"
        )
        fa = _fa_events(events)
        assert len(fa) == 1 and fa[0].meta == {"reason": "repeat", "fallback_used": False}
        assert _final_text(events) == "FINAL"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac1_e_fallback_when_forced_reply_is_a_tool_call(self):
        """AC1(e): the forced final answer's reply is a tool call — the turn's
        final text is ``final_answer.fallback`` with ``{reason}`` filled and
        the ``final_answer`` event reports ``fallback_used=True``, reason
        ``repeat``; the tool call never executes."""
        script = {"c1": "1 failed", "c2": "1 failed", "c3": "1 failed", "c4": "1 failed"}
        sequences = (
            [_iter_events(_x(f"c{i}")) for i in range(1, 5)]
            + [_iter_events(tool_use_event(tool_use_id="c_forced", tool_name="shell", tool_input=dict(SHELL_INPUT)))]
        )
        executor = ScriptedExecutor(script)
        llm = SequentialMockLLM(sequences)
        sr2 = _build_sr2(llm, executor, _config(repeat_guard=_guard()))
        events = [e async for e in sr2.turn(make_user_input())]

        assert executor.order == ["c1", "c2", "c3", "c4"], "the forced reply's tool call never executes"
        assert len(llm.stream_calls) == 5, "4 iterations plus exactly one forced call"
        assert llm.stream_calls[4].tools is None
        assert _final_text(events) == DEFAULT_FINAL_ANSWER_FALLBACK.format(reason=REASON_1_4)
        fa = _fa_events(events)
        assert len(fa) == 1 and fa[0].meta == {"reason": "repeat", "fallback_used": True}
        assert events[-1].type == "end"
        await sr2.aclose()

    @pytest.mark.asyncio
    async def test_ac1_f_enabled_false_never_triggers(self):
        """AC1(f): an explicit ``enabled: false`` block — X ×5 (same result):
        every result passes through untouched, no ``repeat_guard`` or
        ``final_answer`` events, the turn ends normally."""
        script = {f"c{i}": "1 failed" for i in range(1, 6)}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard(enabled=False)),
            [_iter_events(_x(f"c{i}")) for i in range(1, 6)],
            script,
        )

        assert executor.order == [f"c{i}" for i in range(1, 6)]
        _assert_results(llm, script, {})
        assert _rg_events(events) == []
        assert _fa_events(events) == []
        assert _final_text(events) == "FINAL"
        assert events[-1].type == "end"
        # No forced call: exactly one LLM call per iteration plus the
        # terminal text reply.
        assert len(llm.stream_calls) == 6


# ---------------------------------------------------------------------------
# AC9 — the stop keeps the results the guard gave the executed calls
# ---------------------------------------------------------------------------


class TestAC9StopKeepsExecutedResults:
    """The forced final-answer request of a guard stop pairs every executed
    call of the stop iteration with the result the guard gave it (spec
    spc-959 FR 11, 13 amended 2026-10-20, 14), never ``not_run``, and the
    iteration's tool pair reaches the engine bus exactly once.
    """

    @staticmethod
    def _forced_blocks(forced_request):
        """The forced request's ending-iteration pair: the ToolUseBlocks of
        its assistant message (third from the end) and the tool results of
        the user message before the instruction.

        Returns (tool_use_ids, results_by_id)."""
        tool_use_ids = [
            block.id
            for block in forced_request.messages[-3].content
            if isinstance(block, ToolUseBlock)
        ]
        results = [
            block
            for message in forced_request.messages[-2:-1]
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        return tool_use_ids, {r.tool_use_id: r.content for r in results}

    @pytest.mark.asyncio
    async def test_ac9_r1_stop_iteration_pairs_c4_with_the_note(self):
        """AC9-r1: X ×4 (same result). The forced request pairs c4 with the
        note (first_call 1, repeat 3, no warning); no not_run anywhere."""
        script = {f"c{i}": "1 failed" for i in range(1, 5)}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 5)],
            script,
        )

        forced_request = llm.stream_calls[4]
        tool_use_ids, results = self._forced_blocks(forced_request)
        assert tool_use_ids == ["c4"], "the stop iteration's call keeps its pairing"
        assert results["c4"].startswith(_note(1, 3)), "the note, not not_run"
        assert not results["c4"].endswith(DEFAULT_REPEAT_WARNING.format(repeat=2))
        all_results = [
            block
            for message in forced_request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert DEFAULT_FINAL_ANSWER_NOT_RUN not in [r.content for r in all_results]

    @pytest.mark.asyncio
    async def test_ac9_r2_different_result_stop_pairs_the_real_result(self):
        """AC9-r2: X ×10 with distinct results r1..r10. The forced request
        pairs c10 with its real result r10 (FR 13 as amended); no not_run."""
        script = {f"c{i}": f"r{i}" for i in range(1, 11)}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 11)],
            script,
        )

        forced_request = llm.stream_calls[10]
        tool_use_ids, results = self._forced_blocks(forced_request)
        assert tool_use_ids == ["c10"]
        assert results["c10"] == "r10", "the real result, not the note or not_run"
        all_results = [
            block
            for message in forced_request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert DEFAULT_FINAL_ANSWER_NOT_RUN not in [r.content for r in all_results]

    @pytest.mark.asyncio
    async def test_ac9_r3_stop_in_a_multi_call_message_keeps_every_result(self):
        """AC9-r3: X, X, X (same), then one message [X, Y]. The X stops; the
        forced request pairs it with the note and Y with its real result, in
        assistant order; no not_run."""
        y = tool_use_event(tool_use_id="c5", tool_name="grep", tool_input={"pattern": "x"})
        script = {"c1": "1 failed", "c2": "1 failed", "c3": "1 failed", "c4": "1 failed", "c5": "hit"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [
                _iter_events(_x("c1")),
                _iter_events(_x("c2")),
                _iter_events(_x("c3")),
                _iter_text_events("wrapping up", _x("c4"), y),
            ],
            script,
            repeat_last=True,
        )

        assert executor.order == ["c1", "c2", "c3", "c4", "c5"], "both calls ran"
        forced_request = llm.stream_calls[4]
        tool_use_ids, results = self._forced_blocks(forced_request)
        assert tool_use_ids == ["c4", "c5"], "assistant order preserved"
        assert results["c4"].startswith(_note(1, 3)), "the stop note"
        assert results["c5"] == "hit", "the other executed call keeps its real result"
        assert events[-1].type == "end"
        all_results = [
            block
            for message in forced_request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert DEFAULT_FINAL_ANSWER_NOT_RUN not in [r.content for r in all_results]

    @pytest.mark.asyncio
    async def test_ac9_error_result_is_preserved_in_mixed_stop_request(self):
        """AC9-error1: the cap stops on c2 and preserves both full results."""
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard(max_identical_calls=2)),
            [_iter_events(_x("c1"), _x("c2"))],
            {"c1": ("success", False), "c2": ("tool failed", True)},
        )

        assert executor.order == ["c1", "c2"]
        assert len(llm.stream_calls) == 2, "one batch plus exactly one forced call"
        forced_request = llm.stream_calls[1]
        assert forced_request.tools is None
        tool_use_ids = [
            block.id
            for block in forced_request.messages[-3].content
            if isinstance(block, ToolUseBlock)
        ]
        assert tool_use_ids == ["c1", "c2"]
        results = [
            (block.tool_use_id, block.content, block.is_error)
            for block in forced_request.messages[-2].content
            if isinstance(block, ToolResultBlock)
        ]
        assert results == [("c1", "success", False), ("c2", "tool failed", True)]
        fa = _fa_events(events)
        assert len(fa) == 1 and fa[0].meta == {"reason": "repeat", "fallback_used": False}
        assert _final_text(events) == "FINAL"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac9_r4_stop_iteration_tool_pair_is_queued_once(self):
        """AC9-r4: X ×4 (same result). The stop iteration's tool_use_emitted
        and tool_result_received bus events are each queued exactly once, with
        the note as c4's result."""
        script = {f"c{i}": "1 failed" for i in range(1, 5)}
        executor = ScriptedExecutor(script)
        sequences = [_iter_events(_x(f"c{i}")) for i in range(1, 5)] + [_text_reply("FINAL")] * 2
        llm = SequentialMockLLM(sequences)
        sr2 = _build_sr2(llm, executor, _config(repeat_guard=_guard()))
        use_events = _collect_bus(sr2, "tool_use_emitted")
        result_events = _collect_bus(sr2, "tool_result_received")
        [e async for e in sr2.turn(make_user_input())]

        assert len(use_events) == 4, (
            f"one tool_use_emitted per iteration, the stop included once; got {len(use_events)}"
        )
        assert len(result_events) == 4, (
            f"one tool_result_received per iteration, the stop included once; got {len(result_events)}"
        )
        stop_results = [
            block for event in result_events for block in event.data if block.tool_use_id == "c4"
        ]
        assert len(stop_results) == 1, "c4's result is queued exactly once"
        assert stop_results[0].content.startswith(_note(1, 3)), "the note, not not_run"
        assert [block.id for event in use_events for block in event.data] == [
            "c1",
            "c2",
            "c3",
            "c4",
        ]
        await sr2.aclose()


# ---------------------------------------------------------------------------
# Config defaults (FR 22: every text is a YAML field with a default)
# ---------------------------------------------------------------------------


class TestRepeatGuardConfigDefaults:
    def test_defaults_off_with_spec_texts(self):
        g = RepeatGuardConfig()
        assert g.enabled is False
        assert g.notes_before_stop == 2
        assert g.max_identical_calls == 10
        assert g.note == (
            "Result not shown: this call is identical to call {first_call} and "
            "returned the same result. Do not repeat it. Change your approach, or if "
            "you cannot make progress, stop and escalate in your contract's format."
        )
        assert g.warning == (
            " This is repeat {repeat}; the next identical call ends the run with an escalation."
        )

    def test_pipeline_default_is_off(self):
        assert make_minimal_config().repeat_guard.enabled is False

    def test_reason_text_shape(self):
        from sr2.orchestrator import _repeat_reason_text

        assert _repeat_reason_text(1, 4) == REASON_1_4
        assert _repeat_reason_text(2, 10) == "you repeated call 2 10 times"
