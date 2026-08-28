"""Tests for Agent.stream_message() — streaming event API (obsidian-556).

Covers:
  A. Text-only response: yields AgentTextDelta(s) then AgentDone(tool_calls_executed=0)
  B. AgentDone is always last event, even on empty text
  C. Tool call sequence: AgentToolStart → AgentToolResult → ... → AgentDone
  D. tool_calls_executed counter in AgentDone is correct
  E. Tool errors yield AgentToolResult(is_error=True) — not raised
  F. tool-loop limit exceeded: AgentDone still emitted
  G. History is updated the same as handle_user_message()
  H. handle_user_message() still works (re-implemented on top of stream_message())
  I. Multi-round: text + tool + text emitted in correct order

NOTE: Imports from sr2_spectre.events will fail until that module is implemented.
This is intentional — these are TDD tests written before the implementation exists.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from sr2.config.models import ToolLoopLimitError
from sr2.models import Message, TextBlock, TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.protocols.llm import StreamEvent
from sr2_spectre.agent import Agent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import TurnResult
from sr2_spectre.events import (
    AgentDone,
    AgentEvent,
    AgentThinkingDelta,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
    AgentUsage,
)


# ---------------------------------------------------------------------------
# Shared helpers — same patterns as test_agent.py
# ---------------------------------------------------------------------------

def _minimal_pipeline_dict() -> dict:
    return {
        "layers": [
            {
                "name": "system",
                "target": "system",
                "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
            },
            {
                "name": "tools",
                "target": "tools",
                "resolvers": [],
                "tool_providers": [{"type": "spectre_tools"}],
            },
            {
                "name": "conversation",
                "target": "messages",
                "resolvers": [{"type": "session"}, {"type": "input"}],
            },
        ]
    }


def _make_config(**agent_kwargs) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test", **agent_kwargs),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=_minimal_pipeline_dict(),
    )


def _mock_sr2_with_rounds(*round_event_lists: list[StreamEvent]) -> MagicMock:
    """Return a mock SR2 whose turn() yields successive rounds of events."""
    call_count = 0
    rounds = list(round_event_lists)

    mock_sr2 = MagicMock()
    mock_sr2.seed_session = MagicMock()

    async def _turn(user_input):
        nonlocal call_count
        events = rounds[call_count]
        call_count += 1
        for ev in events:
            yield ev

    mock_sr2.turn = _turn
    return mock_sr2


def _make_agent(mock_sr2: MagicMock, **config_kwargs) -> Agent:
    with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
        return Agent(config=_make_config(**config_kwargs), session_id="test-session")


async def _collect(gen: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [ev async for ev in gen]


# ---------------------------------------------------------------------------
# A. Text-only response
# ---------------------------------------------------------------------------

class TestStreamMessageTextOnly:
    @pytest.mark.asyncio
    async def test_text_only_yields_text_delta_then_done(self):
        """Text-only response: AgentTextDelta(s) followed by AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Hello "),
            StreamEvent(type="text", text="world!"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Hi"))

        text_deltas = [e for e in events if isinstance(e, AgentTextDelta)]
        assert len(text_deltas) >= 1
        assert "".join(e.text for e in text_deltas) == "Hello world!"

        done_events = [e for e in events if isinstance(e, AgentDone)]
        assert len(done_events) == 1

        # AgentDone is last
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_text_only_done_has_zero_tool_calls(self):
        """AgentDone.tool_calls_executed is 0 for text-only response."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Answer"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Question"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 0

    @pytest.mark.asyncio
    async def test_text_delta_type_field_is_text_delta(self):
        """AgentTextDelta.type == 'text_delta'."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Hi"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Hello"))

        deltas = [e for e in events if isinstance(e, AgentTextDelta)]
        assert all(e.type == "text_delta" for e in deltas)


# ---------------------------------------------------------------------------
# B. AgentDone is always last — including empty text
# ---------------------------------------------------------------------------

class TestStreamMessageAgentDoneAlwaysLast:
    @pytest.mark.asyncio
    async def test_done_emitted_when_text_is_empty(self):
        """AgentDone is emitted even when the LLM yields no text."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Hi"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 0

    @pytest.mark.asyncio
    async def test_done_is_exactly_last_event(self):
        """No events come after AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Stuff"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Prompt"))

        done_indices = [i for i, e in enumerate(events) if isinstance(e, AgentDone)]
        assert len(done_indices) == 1
        assert done_indices[0] == len(events) - 1

    @pytest.mark.asyncio
    async def test_done_type_field_is_done(self):
        """AgentDone.type == 'done'."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="ok"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("test"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.type == "done"


# ---------------------------------------------------------------------------
# C. Tool call sequence
# ---------------------------------------------------------------------------

class TestStreamMessageToolSequence:
    @pytest.mark.asyncio
    async def test_tool_start_emitted_before_execution(self):
        """AgentToolStart is yielded when SR2 emits tool_use_emitted."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="add", input={"a": 1, "b": 2})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="3")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="Result is 3"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("1+2?"))

        starts = [e for e in events if isinstance(e, AgentToolStart)]
        assert len(starts) == 1
        assert starts[0].type == "tool_start"
        assert starts[0].tool_id == "tu1"
        assert starts[0].name == "add"
        assert starts[0].input == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_tool_result_emitted_after_execution(self):
        """AgentToolResult is yielded when SR2 emits tool_result_received."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="greet", input={"name": "world"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="Hello world")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="Done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Greet"))

        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(results) == 1
        assert results[0].type == "tool_result"
        assert results[0].tool_id == "tu1"
        assert results[0].is_error is False
        assert "Hello world" in results[0].content

    @pytest.mark.asyncio
    async def test_tool_start_before_tool_result_in_sequence(self):
        """AgentToolStart always precedes AgentToolResult for the same tool_id."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="ping", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="pong")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="Pong"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Ping"))

        start_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentToolStart))
        result_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentToolResult))
        assert start_idx < result_idx

    @pytest.mark.asyncio
    async def test_tool_result_before_done(self):
        """AgentToolResult precedes AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="calc", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="42")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="ok"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        result_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentToolResult))
        done_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentDone))
        assert result_idx < done_idx


# ---------------------------------------------------------------------------
# D. tool_calls_executed counter
# ---------------------------------------------------------------------------

class TestStreamMessageToolCallsCounter:
    @pytest.mark.asyncio
    async def test_single_tool_call_counted(self):
        """AgentDone.tool_calls_executed == 1 after one tool call."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="t", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="result")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 1

    @pytest.mark.asyncio
    async def test_two_tool_calls_in_one_round_counted(self):
        """Two tool_use blocks in one iteration -> tool_calls_executed == 2."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[
                    ToolUseBlock(id="tu1", name="t1", input={}),
                    ToolUseBlock(id="tu2", name="t2", input={}),
                ],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[
                    ToolResultBlock(tool_use_id="tu1", content="r1"),
                    ToolResultBlock(tool_use_id="tu2", content="r2"),
                ],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="both done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 2

        starts = [e for e in events if isinstance(e, AgentToolStart)]
        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(starts) == 2
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_tool_calls_across_multiple_iterations_counted(self):
        """Tool calls across two iterations sum correctly in AgentDone."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="t1", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="result")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu2", name="t1", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu2", content="result")],
            ),
            StreamEvent(type="iteration_complete", iteration=1),
            StreamEvent(type="text", text="final"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 2


# ---------------------------------------------------------------------------
# E. Tool errors
# ---------------------------------------------------------------------------

class TestStreamMessageToolErrors:
    @pytest.mark.asyncio
    async def test_tool_error_yields_tool_result_with_is_error_true(self):
        """Failing tool yields AgentToolResult(is_error=True), does not raise."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="boom", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="kaboom", is_error=True)],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="recovered"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("trigger"))

        error_results = [
            e for e in events
            if isinstance(e, AgentToolResult) and e.is_error
        ]
        assert len(error_results) == 1
        assert "kaboom" in error_results[0].content

    @pytest.mark.asyncio
    async def test_unregistered_tool_name_yields_is_error_true(self):
        """Tool result with is_error=True yields AgentToolResult(is_error=True)."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="no_such_tool", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="tool not found", is_error=True)],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="recovered"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("trigger"))

        error_results = [e for e in events if isinstance(e, AgentToolResult) and e.is_error]
        assert len(error_results) == 1
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_tool_error_does_not_raise_from_stream_message(self):
        """stream_message() must not raise even when tool result has error."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="crash", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="fatal", is_error=True)],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="fine"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("trigger"))
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_successful_tool_yields_is_error_false(self):
        """Successful tool yields AgentToolResult(is_error=False)."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="ok_tool", input={"x": 1})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="got 1")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("run"))

        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(results) == 1
        assert results[0].is_error is False


# ---------------------------------------------------------------------------
# F. tool-loop limit exceeded
# ---------------------------------------------------------------------------

class TestStreamMessageMaxToolRounds:
    @pytest.mark.asyncio
    async def test_done_emitted_when_max_tool_rounds_exceeded(self):
        """AgentDone is still the last event when SR2 stops after max iterations."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="still going")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu2", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu2", content="still going")],
            ),
            StreamEvent(type="iteration_complete", iteration=1),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu3", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu3", content="still going")],
            ),
            StreamEvent(type="iteration_complete", iteration=2),
            StreamEvent(type="text", text="stopped"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("start"))
        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_tool_calls_executed_reflects_max_rounds(self):
        """tool_calls_executed reflects the number of tool calls before SR2 stopped."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="ok")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu2", name="loop", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu2", content="ok")],
            ),
            StreamEvent(type="iteration_complete", iteration=1),
            StreamEvent(type="text", text="done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("start"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed == 2


# ---------------------------------------------------------------------------
# F2. ToolLoopLimitError raised mid-stream -> graceful stop (obsidian-ydt, Behavior 2)
#
# When self.sr2.turn(...) raises ToolLoopLimitError partway through streaming,
# stream_message must catch it and end gracefully: emit the text/tool events
# seen before the raise, emit a final notice (an AgentTextDelta whose text
# indicates the tool/iteration limit was reached), and ALWAYS still emit
# AgentDone as the final event. No exception may escape stream_message.
# ---------------------------------------------------------------------------

def _mock_sr2_raising_loop_limit() -> MagicMock:
    """Mock SR2 whose turn() yields a tool round, then raises ToolLoopLimitError."""
    mock_sr2 = MagicMock()
    mock_sr2.seed_session = MagicMock()

    async def _turn(user_input):
        yield StreamEvent(
            type="tool_use_emitted",
            tool_uses=[ToolUseBlock(id="tu1", name="loop", input={})],
        )
        yield StreamEvent(
            type="tool_result_received",
            tool_results=[ToolResultBlock(tool_use_id="tu1", content="still going")],
        )
        raise ToolLoopLimitError("tool loop iteration limit reached")

    mock_sr2.turn = _turn
    return mock_sr2


class TestStreamMessageToolLoopLimitError:
    @pytest.mark.asyncio
    async def test_no_exception_escapes_and_done_is_last(self):
        """ToolLoopLimitError must be caught; AgentDone is still the final event."""
        agent = _make_agent(_mock_sr2_raising_loop_limit())

        events = await _collect(agent.stream_message("go"))

        assert isinstance(events[-1], AgentDone)

    @pytest.mark.asyncio
    async def test_tool_calls_before_raise_are_reflected(self):
        """tool_calls_executed reflects the tool calls emitted before the raise (>=1)."""
        agent = _make_agent(_mock_sr2_raising_loop_limit())

        events = await _collect(agent.stream_message("go"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.tool_calls_executed >= 1

        # The tool events seen before the raise are still surfaced.
        starts = [e for e in events if isinstance(e, AgentToolStart)]
        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(starts) >= 1
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_final_notice_text_delta_emitted(self):
        """At least one AgentTextDelta carries a limit/iteration notice (not exact wording)."""
        agent = _make_agent(_mock_sr2_raising_loop_limit())

        events = await _collect(agent.stream_message("go"))

        delta_texts = [e.text.lower() for e in events if isinstance(e, AgentTextDelta)]
        # NOTE: wording is intentionally not pinned; accept any of these tokens.
        assert any(
            ("limit" in t) or ("iteration" in t) or ("stopped" in t)
            for t in delta_texts
        ), f"No limit/iteration notice found in text deltas: {delta_texts}"


# ---------------------------------------------------------------------------
# E2. Tool result name correlation (obsidian-syp)
# ---------------------------------------------------------------------------

class TestStreamMessageToolResultName:
    @pytest.mark.asyncio
    async def test_tool_result_name_matches_tool_use_name(self):
        """AgentToolResult.name must be populated from the corresponding tool_use_emitted."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="calculator", input={"a": 1, "b": 2})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="3")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="The result is 3"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("1+2?"))

        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(results) == 1
        assert results[0].name == "calculator"

    @pytest.mark.asyncio
    async def test_tool_result_name_multiple_tools(self):
        """Each AgentToolResult.name matches its corresponding tool_use name by tool_id."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[
                    ToolUseBlock(id="tu1", name="search", input={"q": "foo"}),
                    ToolUseBlock(id="tu2", name="fetch", input={"url": "http://example.com"}),
                ],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[
                    ToolResultBlock(tool_use_id="tu1", content="search results"),
                    ToolResultBlock(tool_use_id="tu2", content="<html>"),
                ],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="Done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("search and fetch"))

        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(results) == 2
        name_by_id = {r.tool_id: r.name for r in results}
        assert name_by_id["tu1"] == "search"
        assert name_by_id["tu2"] == "fetch"

    @pytest.mark.asyncio
    async def test_tool_result_name_across_iterations(self):
        """Tool result name is correctly tracked across multiple iterations."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="lookup", input={"key": "a"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="found a")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu2", name="transform", input={"val": "a"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu2", content="transformed a")],
            ),
            StreamEvent(type="iteration_complete", iteration=1),
            StreamEvent(type="text", text="Final answer"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        results = [e for e in events if isinstance(e, AgentToolResult)]
        assert len(results) == 2
        assert results[0].name == "lookup"
        assert results[1].name == "transform"


# ---------------------------------------------------------------------------
# G. History
# ---------------------------------------------------------------------------

class TestStreamMessageHistory:
    @pytest.mark.asyncio
    async def test_history_updated_after_stream(self):
        """History has user + assistant messages after stream_message completes."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Response"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        await _collect(agent.stream_message("Question"))

        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_tool_result_appended_to_history(self):
        """When tools are used, history still gets user + assistant (SR2 handles tool loop)."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="lookup", input={"q": "x"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="42")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="The answer is 42"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        await _collect(agent.stream_message("What is x?"))

        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_history_grows_after_each_turn(self):
        """Each stream_message call appends user + assistant to history (observable proxy for seed_session being called)."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Hi"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        await _collect(agent.stream_message("Hello"))
        assert len(agent.history) == 2

        # Second turn grows history further — proves seed_session was called again
        async def _second_turn(user_input):
            yield StreamEvent(type="text", text="Again")
            yield StreamEvent(type="end")

        mock_sr2.turn = _second_turn
        await _collect(agent.stream_message("Second"))
        assert len(agent.history) == 4


# ---------------------------------------------------------------------------
# H. handle_user_message on top of stream_message
# ---------------------------------------------------------------------------

class TestHandleUserMessageOnStreamMessage:
    @pytest.mark.asyncio
    async def test_handle_user_message_returns_turn_result(self):
        """handle_user_message still returns TurnResult."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Answer"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        result = await agent.handle_user_message("Question")
        assert isinstance(result, TurnResult)
        assert result.text == "Answer"

    @pytest.mark.asyncio
    async def test_handle_user_message_with_tool_returns_correct_count(self):
        """handle_user_message counts tool calls from SR2 internal tool loop."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="calc", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="42")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="42"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        result = await agent.handle_user_message("calc")
        assert isinstance(result, TurnResult)
        assert result.tool_calls_executed == 1


# ---------------------------------------------------------------------------
# I. Multi-round ordering
# ---------------------------------------------------------------------------

class TestStreamMessageMultiRoundOrdering:
    @pytest.mark.asyncio
    async def test_text_then_tool_then_text_event_order(self):
        """Events appear in correct order: text, tool_start, tool_result, text, done."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Thinking..."),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="lookup", input={"q": "x"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="42")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="The answer is 42"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)
        agent.register_tool("lookup", "Lookup", {}, lambda q: "42")

        events = await _collect(agent.stream_message("What is x?"))

        types = [type(e).__name__ for e in events]

        assert "AgentTextDelta" in types
        assert "AgentToolStart" in types
        assert "AgentToolResult" in types
        assert "AgentDone" in types

        text_delta_indices = [i for i, t in enumerate(types) if t == "AgentTextDelta"]
        tool_start_idx = types.index("AgentToolStart")
        tool_result_idx = types.index("AgentToolResult")
        done_idx = types.index("AgentDone")

        assert min(text_delta_indices) < tool_start_idx
        assert tool_start_idx < tool_result_idx
        assert tool_result_idx < done_idx


# ---------------------------------------------------------------------------
# J. Thinking events pass-through (spc-26)
# ---------------------------------------------------------------------------

class TestStreamMessageThinking:
    @pytest.mark.asyncio
    async def test_thinking_events_passed_through(self):
        """SR2 thinking events are translated to AgentThinkingDelta."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="thinking", text="Let me think"),
            StreamEvent(type="text", text="The answer is 42"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("What is 6*7?"))

        thinking_events = [e for e in events if isinstance(e, AgentThinkingDelta)]
        assert len(thinking_events) == 1
        assert thinking_events[0].text == "Let me think"
        assert thinking_events[0].type == "thinking_delta"

    @pytest.mark.asyncio
    async def test_thinking_not_accumulated_in_history(self):
        """Thinking text is NOT appended to assistant history — only regular text is."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="thinking", text="reasoning here"),
            StreamEvent(type="text", text="final answer"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        await _collect(agent.stream_message("question"))

        # History should have user + assistant, assistant should NOT contain thinking
        assert len(agent.history) == 2
        assistant = agent.history[1]
        assert assistant.role == "assistant"
        # The text content should be the regular response, not the thinking
        text_content = " ".join(
            block.text for block in assistant.content if hasattr(block, "text")
        )
        assert "final answer" in text_content
        assert "reasoning here" not in text_content

    @pytest.mark.asyncio
    async def test_multiple_thinking_chunks_passed_through(self):
        """Multiple thinking events are all yielded as separate AgentThinkingDelta."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="thinking", text="Part 1 "),
            StreamEvent(type="thinking", text="Part 2"),
            StreamEvent(type="text", text="Answer"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("go"))

        thinking_events = [e for e in events if isinstance(e, AgentThinkingDelta)]
        assert len(thinking_events) == 2
        assert thinking_events[0].text == "Part 1 "
        assert thinking_events[1].text == "Part 2"

    @pytest.mark.asyncio
    async def test_thinking_then_tool_then_text_order(self):
        """Thinking events appear before tool calls when reasoning precedes tools."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="thinking", text="Planning approach"),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="search", input={"q": "x"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="found")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="Found it!"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("search"))

        thinking_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentThinkingDelta))
        tool_start_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentToolStart))
        text_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentTextDelta))

        assert thinking_idx < tool_start_idx
        assert tool_start_idx < text_idx


# ---------------------------------------------------------------------------
# J. Token usage capture (spc-82)
# ---------------------------------------------------------------------------

class TestStreamMessageUsageCapture:
    @pytest.mark.asyncio
    async def test_usage_events_yield_agent_usage(self):
        """StreamEvent(type='usage') events are surfaced as AgentUsage."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Answer"),
            StreamEvent(
                type="usage",
                usage=TokenUsage(input_tokens=100, output_tokens=25),
            ),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Question"))

        usages = [e for e in events if isinstance(e, AgentUsage)]
        assert len(usages) == 1
        assert usages[0].input_tokens == 100
        assert usages[0].output_tokens == 25

    @pytest.mark.asyncio
    async def test_done_carries_usage_totals(self):
        """AgentDone sums real usage across the turn."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(
                type="usage",
                usage=TokenUsage(input_tokens=100, output_tokens=25),
            ),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="search", input={"q": "x"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="found")],
            ),
            StreamEvent(
                type="usage",
                usage=TokenUsage(input_tokens=40, output_tokens=10),
            ),
            StreamEvent(type="text", text="Done"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Question"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.input_tokens == 140
        assert done.output_tokens == 35

    @pytest.mark.asyncio
    async def test_done_zero_usage_when_endpoint_silent(self):
        """No usage events -> AgentDone token fields are 0 (unknown, not fake)."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Answer"),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        events = await _collect(agent.stream_message("Question"))

        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.input_tokens == 0
        assert done.output_tokens == 0

    @pytest.mark.asyncio
    async def test_done_context_tokens_is_local_estimate(self):
        """context_tokens is the local estimate of prior history, available
        regardless of whether the endpoint reports usage."""
        mock_sr2 = _mock_sr2_with_rounds(
            [
                StreamEvent(type="text", text="Answer"),
                StreamEvent(type="end"),
            ],
            [
                StreamEvent(type="text", text="Follow-up answer"),
                StreamEvent(type="end"),
            ],
        )
        agent = _make_agent(mock_sr2)

        # First turn: no prior history -> estimate is 0.
        events = await _collect(agent.stream_message("Question"))
        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.context_tokens == 0

        # Second turn: prior history includes the user + assistant messages
        # from the first turn -> estimate is positive.
        events = await _collect(agent.stream_message("Follow-up"))
        done = events[-1]
        assert isinstance(done, AgentDone)
        assert done.context_tokens > 0

    @pytest.mark.asyncio
    async def test_handle_user_message_populates_total_tokens(self):
        """TurnResult.total_tokens reflects real usage reported by the LLM."""
        mock_sr2 = _mock_sr2_with_rounds([
            StreamEvent(type="text", text="Answer"),
            StreamEvent(
                type="usage",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
            StreamEvent(type="end"),
        ])
        agent = _make_agent(mock_sr2)

        result = await agent.handle_user_message("Question")

        assert isinstance(result, TurnResult)
        assert result.total_tokens == 15
