"""spc-1119 AC6: a messages layer over token_threshold_pct queues token_threshold
when its content is Message items (FR 8)."""

from __future__ import annotations

import pytest

from sr2.models import Message, TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.token_counting import CharacterTokenCounter


def _layer(bus: EventBus) -> Layer:
    return Layer(
        name="session",
        target=CompilationTarget.MESSAGES,
        position=AppendStrategy(),
        token_budget=1000,
        token_threshold_pct=0.5,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=bus,
    )


def _messages(tokens: int) -> list[Message]:
    # Two messages of tokens/2 each; 4 chars = 1 token.
    half = tokens // 2
    return [
        Message(role="user", content=[TextBlock(text="u" * (half * 4))]),
        Message(role="assistant", content=[TextBlock(text="a" * (half * 4))]),
    ]


async def _run(tokens: int) -> list[Event]:
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(EventSubscription(event_name="token_threshold"), seen.append)
    layer = _layer(bus)
    layer.set_content(_messages(tokens))
    layer.handle_event(Event(name="turn_start", phase=EventPhase.COMPLETED, source_layer="engine"))
    await layer.process_pending()
    await bus.drain()
    return seen


@pytest.mark.asyncio
async def test_ac6_r3_message_content_over_threshold_queues_event():
    """AC6-r3: 600 counted tokens of Messages -> one token_threshold, used == 600."""
    seen = await _run(600)
    assert len(seen) == 1
    assert seen[0].data["used"] == 600


@pytest.mark.asyncio
async def test_ac6_r4_message_content_under_threshold_no_event():
    """AC6-r4: 400 counted tokens of Messages -> no token_threshold event."""
    assert await _run(400) == []
