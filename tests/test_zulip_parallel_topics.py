"""Zulip topics run concurrently (zulip-migration FR3, FR4, AC19).

The poller never waits on a turn; a turn in one topic never waits on a turn
in another; messages in one topic keep their order on that topic's Session;
each turn's reminder origin is its own even while turns overlap.

The real ``ZulipInterface`` poll loop runs over a fake transport and a fake
agent whose Session streams block on per-message gates the test opens. Every
test is bounded by a timeout, so a serialized loop fails instead of hanging.
No model, no network.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from sr2_spectre.events import AgentDone, AgentTextDelta
from sr2_spectre.interfaces.zulip import ZulipConfig, ZulipInterface
from sr2_spectre.interfaces.zulip.client import ZulipCredentials
from sr2_spectre.reminders import tool as reminders_tool

DIEGO_ID = 8
BOT_ID = 9
CHANNEL = 2
TIMEOUT = 5.0


class _Harness:
    """Shared state: the ordered log, one gate per message text, and the
    reminder origin topic each turn saw while it ran."""

    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.started: dict[str, asyncio.Event] = {}
        self.origin_topics: dict[str, str] = {}

    def gate(self, text: str) -> asyncio.Event:
        return self.gates.setdefault(text, asyncio.Event())

    def started_event(self, text: str) -> asyncio.Event:
        return self.started.setdefault(text, asyncio.Event())


class _FakeSession:
    def __init__(self, harness: _Harness) -> None:
        self._h = harness

    def set_run_context(self, context: Any) -> None:
        pass

    async def stream_message(self, text: str):
        self._h.log.append(("START", text))
        self._h.started_event(text).set()
        await self._h.gate(text).wait()
        scope = reminders_tool.turn_scope.get()
        self._h.origin_topics[text] = scope.origin.topic if scope else ""
        yield AgentTextDelta(text=f"answer {text}")
        yield AgentDone(tool_calls_executed=0)
        self._h.log.append(("END", text))

    def close(self) -> None:
        pass


class _FakeAgent:
    def __init__(self, harness: _Harness) -> None:
        self.config = SimpleNamespace(agent=SimpleNamespace(name="miranda"))
        self._h = harness

    def create_session(self, frame_id: str) -> _FakeSession:
        return _FakeSession(self._h)

    def apply_config(self, config: Any) -> None:
        pass


class _FakeClient:
    """Hands out each batch once its release event is set; polls are
    recorded so a test can see the loop kept polling."""

    def __init__(self, batches: list[tuple[asyncio.Event | None, list[dict]]]):
        self._batches = list(batches)
        self.polls = 0
        self.idle = asyncio.Event()
        self._next_id = 500

    async def register(self) -> dict:
        return {"queue_id": "q", "last_event_id": 0, "user_id": BOT_ID,
                "full_name": "Miranda",
                "realm_users": [{"user_id": DIEGO_ID, "is_bot": False},
                                {"user_id": BOT_ID, "is_bot": True}]}

    async def get_events(self, queue_id: str, last_event_id: int) -> dict:
        self.polls += 1
        if not self._batches:
            self.idle.set()
            await asyncio.sleep(0.01)
            return {"events": []}
        release, messages = self._batches[0]
        if release is not None and not release.is_set():
            await asyncio.sleep(0.01)
            return {"events": []}
        self._batches.pop(0)
        return {"events": [{"id": m["id"], "type": "message", "message": m}
                           for m in messages]}

    async def get_topic_messages(self, channel_id: int, topic: str) -> dict:
        return {"messages": []}

    async def send_message(self, **kwargs: Any) -> int:
        self._next_id += 1
        return self._next_id

    async def update_message(self, message_id: int, content: str) -> None:
        pass

    async def aclose(self) -> None:
        pass


_ids = iter(range(100, 10_000))


def _msg(text: str, topic: str) -> dict:
    return {"id": next(_ids), "type": "stream", "stream_id": CHANNEL,
            "subject": topic, "sender_id": DIEGO_ID, "content": text,
            "display_recipient": "sandbox"}


@pytest.fixture
def start(monkeypatch):
    """Start the real interface over a fake client; yields a runner."""
    import sr2_spectre.interfaces.zulip.interface as iface_mod

    monkeypatch.setattr(iface_mod, "load_zulip_credentials", lambda path: (
        ZulipCredentials(email="miranda-bot@zulip.example", api_key="k",
                         site="https://zulip.example")))
    started: list[tuple[ZulipInterface, asyncio.Task]] = []

    async def _start(batches, harness: _Harness):
        client = _FakeClient(batches)
        monkeypatch.setattr(iface_mod, "ZulipClient", lambda **kw: client)
        interface = ZulipInterface(ZulipConfig(
            zuliprc="unused", users=[DIEGO_ID], edit_stream_interval=0,
        ))
        await interface.start(_FakeAgent(harness))
        task = asyncio.create_task(interface.run(None))
        started.append((interface, task))
        return interface, client, task

    yield _start
    for interface, task in started:
        interface._running = False
        task.cancel()


async def _wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), TIMEOUT)


async def test_a_later_topic_answers_while_an_earlier_one_is_still_running(start):
    """FR3/FR4: topic B, arriving in a later batch, finishes while topic A's
    turn is still blocked, so the poller kept polling and B never waited."""
    h = _Harness()
    later = asyncio.Event()
    interface, client, task = await start(
        [(None, [_msg("a", "topic-a")]), (later, [_msg("b", "topic-b")])], h,
    )
    await _wait(h.started_event("a"))
    later.set()
    await _wait(h.started_event("b"))
    h.gate("b").set()
    await asyncio.wait_for(_until(lambda: ("END", "b") in h.log), TIMEOUT)
    assert ("END", "a") not in h.log
    h.gate("a").set()
    await _wait(client.idle)
    interface._running = False
    await asyncio.wait_for(task, TIMEOUT)
    assert h.log.index(("END", "b")) < h.log.index(("END", "a"))


async def test_messages_in_one_topic_run_in_order(start):
    """FR4: a second message in the same topic starts only after the first
    finished, on the same topic Session."""
    h = _Harness()
    interface, client, task = await start(
        [(None, [_msg("one", "same"), _msg("two", "same")])], h,
    )
    await _wait(h.started_event("one"))
    await _wait(client.idle)
    assert ("START", "two") not in h.log
    h.gate("two").set()
    h.gate("one").set()
    interface._running = False
    await asyncio.wait_for(task, TIMEOUT)
    assert h.log == [("START", "one"), ("END", "one"),
                     ("START", "two"), ("END", "two")]


async def test_overlapping_turns_each_keep_their_own_reminder_origin(start):
    """Delivery seam: each turn's reminder origin is its own topic even while
    another topic's turn runs at the same time."""
    h = _Harness()
    interface, client, task = await start(
        [(None, [_msg("a", "topic-a"), _msg("b", "topic-b")])], h,
    )
    await _wait(h.started_event("a"))
    await _wait(h.started_event("b"))
    h.gate("b").set()
    h.gate("a").set()
    await _wait(client.idle)
    interface._running = False
    await asyncio.wait_for(task, TIMEOUT)
    assert h.origin_topics == {"a": "topic-a", "b": "topic-b"}
    assert reminders_tool.turn_scope.get() is None


async def test_stop_cancels_a_running_turn(start):
    h = _Harness()
    interface, client, task = await start([(None, [_msg("a", "topic-a")])], h)
    await _wait(h.started_event("a"))
    await asyncio.wait_for(interface.stop(), TIMEOUT)
    await asyncio.wait_for(task, TIMEOUT)
    assert ("END", "a") not in h.log


async def _until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0.005)
