"""Tests for thinking/reasoning visibility in the TUI (spc-26).

Covers:
  1. AgentThinkingDelta event type exists with correct type field
  2. Session.stream_message passes through thinking events from SR2
  3. Thinking events are distinct from text events

Note: TUI rendering of thinking events (FR5/6) is deferred to spc-55.
These tests verify the event types and pass-through behavior.
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest

from sr2.protocols.llm import StreamEvent
from sr2_spectre.events import (
    AgentDone,
    AgentEvent,
    AgentThinkingDelta,
    AgentTextDelta,
)


# ---------------------------------------------------------------------------
# 1. Event type
# ---------------------------------------------------------------------------

def test_thinking_delta_event_type():
    """AgentThinkingDelta has type='thinking_delta'."""
    ev = AgentThinkingDelta(text="reasoning")
    assert ev.type == "thinking_delta"
    assert ev.text == "reasoning"
    assert isinstance(ev, AgentEvent)


def test_thinking_delta_is_distinct_from_text_delta():
    """AgentThinkingDelta is not an AgentTextDelta."""
    ev = AgentThinkingDelta(text="reasoning")
    assert not isinstance(ev, AgentTextDelta)


# ---------------------------------------------------------------------------
# 2. Session passes through thinking events
# ---------------------------------------------------------------------------

def _make_agent_for_thinking(round_events: list[StreamEvent]) -> MagicMock:
    """Create a mock agent whose stream_message yields the given events."""
    agent = MagicMock()
    agent.session_id = "test-session"
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=[])
    agent.history = []

    async def _stream(text: str) -> AsyncIterator:
        for ev in round_events:
            if ev.type == "text" and ev.text:
                yield AgentTextDelta(text=ev.text)
            elif ev.type == "thinking" and ev.text:
                yield AgentThinkingDelta(text=ev.text)
            elif ev.type == "end":
                yield AgentDone()

    agent.stream_message = _stream
    return agent


@pytest.mark.asyncio
async def test_thinking_events_stream_correctly():
    """AgentThinkingDelta events are yielded correctly from stream_message."""
    events = [
        AgentThinkingDelta(text="Let me think about this"),
        AgentTextDelta(text="The answer is 42"),
        AgentDone(),
    ]
    agent = _make_agent_for_thinking([
        StreamEvent(type="thinking", text="Let me think about this"),
        StreamEvent(type="text", text="The answer is 42"),
        StreamEvent(type="end"),
    ])

    received = []
    async for ev in agent.stream_message("question"):
        received.append(ev)

    assert len(received) == 3
    assert isinstance(received[0], AgentThinkingDelta)
    assert received[0].text == "Let me think about this"
    assert isinstance(received[1], AgentTextDelta)
    assert received[1].text == "The answer is 42"
    assert isinstance(received[2], AgentDone)


@pytest.mark.asyncio
async def test_thinking_then_tool_then_text():
    """Thinking before tools: thinking block closes, tool runs, text resumes."""
    from sr2_spectre.events import AgentToolStart, AgentToolResult

    agent = MagicMock()
    agent.session_id = "test"
    agent.new_session = MagicMock()
    agent.registry = MagicMock()
    agent.registry.list_names = MagicMock(return_value=[])
    agent.history = []

    async def _stream(text: str):
        yield AgentThinkingDelta(text="Planning approach")
        yield AgentToolStart(tool_id="t1", name="search", input={"q": "x"})
        yield AgentToolResult(tool_id="t1", name="search", content="found", is_error=False)
        yield AgentTextDelta(text="Found it!")
        yield AgentDone()

    agent.stream_message = _stream

    received = []
    async for ev in agent.stream_message("q"):
        received.append(ev)

    assert len(received) == 5
    assert isinstance(received[0], AgentThinkingDelta)
    assert received[0].text == "Planning approach"
    assert isinstance(received[1], AgentToolStart)
    assert received[1].name == "search"
    assert isinstance(received[2], AgentToolResult)
    assert not received[2].is_error
    assert isinstance(received[3], AgentTextDelta)
    assert received[3].text == "Found it!"
    assert isinstance(received[4], AgentDone)
