"""Tests for sr2-170 — SequentialMockLLM fails loudly when its script runs out.

AC6: a call past the last scripted sequence raises AssertionError unless the
test opted in with ``repeat_last=True``, which repeats the last sequence.

Case table rows: AC6-r1, AC6-r2.
"""

from __future__ import annotations

import pytest

from sr2.protocols.llm import CompletionRequest, StreamEvent
from conftest import SequentialMockLLM


def _seq(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text", text=text), StreamEvent(type="end")]


async def _consume(llm: SequentialMockLLM) -> list[StreamEvent]:
    return [event async for event in llm.stream(CompletionRequest(messages=[]))]


class TestSequentialMockLLMExhaustion:
    @pytest.mark.asyncio
    async def test_ac6_r1_third_call_past_two_sequences_raises(self):
        """AC6-r1: calls 1 and 2 yield seq_a and seq_b; call 3 raises with the
        exact exhaustion message; stream_calls still records all 3 calls."""
        seq_a, seq_b = _seq("a"), _seq("b")
        llm = SequentialMockLLM([seq_a, seq_b])

        assert await _consume(llm) == seq_a
        assert await _consume(llm) == seq_b
        with pytest.raises(AssertionError) as exc_info:
            await _consume(llm)

        assert str(exc_info.value) == (
            "SequentialMockLLM script exhausted: call 3 with 2 scripted sequences"
        )
        assert len(llm.stream_calls) == 3

    @pytest.mark.asyncio
    async def test_ac6_r2_repeat_last_yields_last_sequence_again(self):
        """AC6-r2: with repeat_last=True the third call yields seq_b, no error."""
        seq_a, seq_b = _seq("a"), _seq("b")
        llm = SequentialMockLLM([seq_a, seq_b], repeat_last=True)

        assert await _consume(llm) == seq_a
        assert await _consume(llm) == seq_b
        assert await _consume(llm) == seq_b
        assert len(llm.stream_calls) == 3
