"""Concurrent Discord turns stay on their own channel's Session (obsidian-hw0f).

One shared Agent serves every Discord channel, and each inbound message runs
as its own concurrent task. A turn restores its channel's Session and then
awaits the typing indicator before it streams. If another channel's message
arrives during that await, the first turn must still run on its own channel's
Session: its own history reaches the model, its reply reaches its own channel,
and its turn is recorded in its own channel's session log (spec
live-session-log: one log per Session, joined per channel on the session ID
``discord-<channel_id>``).

The tests drive the real ``DiscordInterface`` -> ``Agent`` -> ``Session`` path
with a fake Discord adapter and a scripted model. The fake typing indicator
holds the turn inside ``channel_typing`` until the other channel's turn has
also reached its typing indicator (bounded, so an implementation that runs
the two turns one after the other is not deadlocked).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sr2.protocols.llm import StreamEvent
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.interface import DiscordInterface
from sr2_spectre.run_log import SessionLogManager
from tests.test_session_run_log import _agent, _events

CHANNEL_A = 1001
CHANNEL_B = 2002
MARKER = re.compile(r"(?:alpha|bravo)-\d+")
TERMINAL = ("turn.complete", "turn.error", "turn.cancel")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _texts(request) -> list[str]:
    """Every text block in the request's conversation messages."""
    out: list[str] = []
    for msg in request.messages:
        content = msg.content
        if isinstance(content, str):
            out.append(content)
            continue
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                out.append(text)
    return out


def _user_texts(request) -> list[str]:
    out: list[str] = []
    for msg in request.messages:
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, str):
            out.append(content)
            continue
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                out.append(text)
    return out


class ContextEchoLLM:
    """Model that replies with the input it answers and the context it saw.

    The reply to input ``X`` is ``re:X ctx:<every marker in the request>``,
    so both the reply text and the recorded requests show which conversation
    history reached the model. ``fail_on`` makes the call for that input
    raise instead. Once ``adapter`` is set, ``typing_at_call`` records which
    channels were inside their typing indicator when each call began.
    """

    model = "test-model"

    def __init__(self, fail_on: str | None = None) -> None:
        self.requests: dict[str, Any] = {}
        self.typing_at_call: dict[str, set[int]] = {}
        self.adapter: FakeAdapter | None = None
        self._fail_on = fail_on

    async def complete(self, request):
        raise NotImplementedError

    async def stream(self, request):
        current = _user_texts(request)[-1]
        self.requests[current] = request
        if self.adapter is not None:
            self.typing_at_call[current] = set(self.adapter.typing_now)
        if current == self._fail_on:
            raise RuntimeError(f"model failed on {current}")
        seen = sorted({m for t in _texts(request) for m in MARKER.findall(t)})
        yield StreamEvent(type="text", text=f"re:{current} ctx:{','.join(seen)}")
        yield StreamEvent(type="end")


class _Typing:
    """Typing indicator that holds until the other channel is typing too.

    Entering yields to the event loop until some other channel is inside
    its own typing indicator, or until a bounded number of yields has
    passed. That reproduces a second channel's message being handled while
    this turn awaits ``channel_typing``.
    """

    MAX_YIELDS = 200

    def __init__(self, adapter: "FakeAdapter", channel_id: int) -> None:
        self._adapter = adapter
        self._channel_id = channel_id

    async def __aenter__(self):
        self._adapter.typing_now.add(self._channel_id)
        for _ in range(self.MAX_YIELDS):
            if self._adapter.typing_now - {self._channel_id}:
                break
            await asyncio.sleep(0)
        return None

    async def __aexit__(self, *exc) -> bool:
        self._adapter.typing_now.discard(self._channel_id)
        return False


class FakeAdapter:
    """Discord adapter stand-in recording what each channel was sent."""

    bot_id = 11111
    bot_mentions = ["<@11111>"]

    def __init__(self) -> None:
        self.typing_now: set[int] = set()
        self.posted: dict[int, dict[int, str]] = {}
        self._next_id = 500

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def set_message_handler(self, handler) -> None:
        self.handler = handler

    def set_slash_handler(self, handler) -> None:
        self.slash_handler = handler

    def is_thread_channel(self, channel_obj) -> bool:
        return False

    def area_channel(self, channel_obj):
        return (None, None)

    def channel_typing(self, channel_id: int) -> _Typing:
        return _Typing(self, channel_id)

    async def send_message(self, channel_id: int, text: str):
        self._next_id += 1
        self.posted.setdefault(channel_id, {})[self._next_id] = text
        return MagicMock(id=self._next_id)

    async def edit_message(self, channel_id: int, message_id: int, text: str) -> None:
        self.posted.setdefault(channel_id, {})[message_id] = text

    async def send_embed(self, *args, **kwargs) -> None:
        return None

    def shown(self, channel_id: int) -> list[str]:
        """What is currently displayed in a channel, oldest message first."""
        return [self.posted[channel_id][k] for k in sorted(self.posted.get(channel_id, {}))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def log_dir() -> Path:
    """The session-log directory under the conftest-isolated SR2_HOME."""
    return SessionLogManager().directory


async def _started(llm: ContextEchoLLM) -> tuple[DiscordInterface, FakeAdapter]:
    adapter = FakeAdapter()
    llm.adapter = adapter
    iface = DiscordInterface(config=DiscordConfig(edit_stream_interval=0))
    agent = _agent(llm)
    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter",
        return_value=adapter,
    ):
        await iface.start(agent)
    return iface, adapter


def _message(content: str, channel_id: int) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.id = hash(content) & 0xFFFF
    message.channel = MagicMock(id=channel_id)
    message.author = MagicMock(id=99999)
    return message


async def _send(iface: DiscordInterface, content: str, channel_id: int) -> None:
    await asyncio.wait_for(iface._process_message(_message(content, channel_id)), timeout=10)


async def _send_together(iface: DiscordInterface, *msgs: tuple[str, int]) -> None:
    """Deliver messages as concurrent tasks, as discord.py dispatches them."""
    await asyncio.wait_for(
        asyncio.gather(*(iface._process_message(_message(c, ch)) for c, ch in msgs)),
        timeout=10,
    )


def _logs(log_dir: Path) -> list[list[dict]]:
    files = sorted(log_dir.glob("*.jsonl")) if log_dir.exists() else []
    return [_events(f) for f in files]


def _sid(channel_id: int) -> str:
    return f"discord-{channel_id}"


def _turn_inputs(log_dir: Path, channel_id: int) -> list[str]:
    """Inputs of every turn.start in this channel's session logs, sorted.

    Sorted because log file names share a UTC-second prefix and end in a
    random suffix, so file order is not message order.
    """
    out: list[str] = []
    for events in _logs(log_dir):
        for e in events:
            if e["event"] == "turn.start" and e["session_id"] == _sid(channel_id):
                out.append(e["data"]["input"])
    return sorted(out)


def _log_holding(log_dir: Path, text: str) -> list[dict]:
    """The single log file whose turn.start carries ``text``."""
    matches = [
        events for events in _logs(log_dir)
        if any(e["event"] == "turn.start" and e["data"]["input"] == text for e in events)
    ]
    assert len(matches) == 1, f"expected exactly one log with turn {text!r}, got {len(matches)}"
    return matches[0]


def _turn_events(events: list[dict], text: str) -> list[str]:
    """Event names from ``text``'s turn.start through its terminal event."""
    names = [e["event"] for e in events]
    start = next(
        i for i, e in enumerate(events)
        if e["event"] == "turn.start" and e["data"]["input"] == text
    )
    tail = names[start:]
    end = next(i for i, n in enumerate(tail) if n in TERMINAL)
    return tail[: end + 1]


def _markers(request) -> set[str]:
    return {m for t in _texts(request) for m in MARKER.findall(t)}


async def _seed_two_channels(iface: DiscordInterface) -> None:
    """Give each channel one completed, non-overlapping turn of history."""
    await _send(iface, "alpha-1", CHANNEL_A)
    await _send(iface, "bravo-1", CHANNEL_B)


# ---------------------------------------------------------------------------
# AC1 — each of two interleaved channel turns is logged in its own channel's
# session log
# ---------------------------------------------------------------------------

class TestInterleavedTurnsLogToOwnChannel:
    async def test_each_turn_start_is_in_its_own_channels_log(self, log_dir):
        iface, _ = await _started(ContextEchoLLM())

        await _send_together(iface, ("alpha-1", CHANNEL_A), ("bravo-1", CHANNEL_B))

        assert _turn_inputs(log_dir, CHANNEL_A) == ["alpha-1"]
        assert _turn_inputs(log_dir, CHANNEL_B) == ["bravo-1"]

    async def test_each_turn_runs_to_completion_in_its_own_log(self, log_dir):
        iface, _ = await _started(ContextEchoLLM())
        await _seed_two_channels(iface)

        await _send_together(iface, ("alpha-2", CHANNEL_A), ("bravo-2", CHANNEL_B))

        for text, channel in (("alpha-2", CHANNEL_A), ("bravo-2", CHANNEL_B)):
            events = _log_holding(log_dir, text)
            assert {e["session_id"] for e in events} == {_sid(channel)}, (
                f"{text}'s log mixes sessions"
            )
            turn = _turn_events(events, text)
            assert turn[-1] == "turn.complete", f"{text} did not complete in its log: {turn}"
            assert "model.start" in turn and "model.end" in turn, turn

    async def test_other_channels_turn_never_appears_in_a_log(self, log_dir):
        iface, _ = await _started(ContextEchoLLM())
        await _seed_two_channels(iface)

        await _send_together(iface, ("alpha-2", CHANNEL_A), ("bravo-2", CHANNEL_B))

        assert _turn_inputs(log_dir, CHANNEL_A) == ["alpha-1", "alpha-2"]
        assert _turn_inputs(log_dir, CHANNEL_B) == ["bravo-1", "bravo-2"]

    async def test_arrival_order_does_not_matter(self, log_dir):
        iface, _ = await _started(ContextEchoLLM())
        await _seed_two_channels(iface)

        await _send_together(iface, ("bravo-2", CHANNEL_B), ("alpha-2", CHANNEL_A))

        assert _turn_inputs(log_dir, CHANNEL_A) == ["alpha-1", "alpha-2"]
        assert _turn_inputs(log_dir, CHANNEL_B) == ["bravo-1", "bravo-2"]

    async def test_failed_turn_is_logged_in_its_own_channel_only(self, log_dir):
        iface, adapter = await _started(ContextEchoLLM(fail_on="alpha-2"))
        await _seed_two_channels(iface)

        await _send_together(iface, ("alpha-2", CHANNEL_A), ("bravo-2", CHANNEL_B))

        a_turn = _turn_events(_log_holding(log_dir, "alpha-2"), "alpha-2")
        b_turn = _turn_events(_log_holding(log_dir, "bravo-2"), "bravo-2")
        assert a_turn[-1] == "turn.error", a_turn
        assert b_turn[-1] == "turn.complete", b_turn
        assert _turn_inputs(log_dir, CHANNEL_A) == ["alpha-1", "alpha-2"]
        assert _turn_inputs(log_dir, CHANNEL_B) == ["bravo-1", "bravo-2"]
        assert "model failed on alpha-2" in adapter.shown(CHANNEL_A)[-1]
        assert adapter.shown(CHANNEL_B)[-1].startswith("re:bravo-2")


# ---------------------------------------------------------------------------
# AC2 — each interleaved turn sees its own channel's history, not the other's,
# and its reply goes to its own channel
# ---------------------------------------------------------------------------

class TestInterleavedTurnsSeeOwnHistory:
    async def test_model_sees_only_own_channels_history(self):
        llm = ContextEchoLLM()
        iface, _ = await _started(llm)
        await _seed_two_channels(iface)

        await _send_together(iface, ("alpha-2", CHANNEL_A), ("bravo-2", CHANNEL_B))

        assert _markers(llm.requests["alpha-2"]) == {"alpha-1", "alpha-2"}
        assert _markers(llm.requests["bravo-2"]) == {"bravo-1", "bravo-2"}

    async def test_model_sees_only_own_channels_history_either_order(self):
        llm = ContextEchoLLM()
        iface, _ = await _started(llm)
        await _seed_two_channels(iface)

        await _send_together(iface, ("bravo-2", CHANNEL_B), ("alpha-2", CHANNEL_A))

        assert _markers(llm.requests["alpha-2"]) == {"alpha-1", "alpha-2"}
        assert _markers(llm.requests["bravo-2"]) == {"bravo-1", "bravo-2"}

    async def test_first_message_in_each_channel_sees_no_other_channel(self):
        llm = ContextEchoLLM()
        iface, _ = await _started(llm)

        await _send_together(iface, ("alpha-1", CHANNEL_A), ("bravo-1", CHANNEL_B))

        assert _markers(llm.requests["alpha-1"]) == {"alpha-1"}
        assert _markers(llm.requests["bravo-1"]) == {"bravo-1"}

    async def test_each_channel_shows_its_own_reply(self):
        llm = ContextEchoLLM()
        iface, adapter = await _started(llm)
        await _seed_two_channels(iface)

        await _send_together(iface, ("alpha-2", CHANNEL_A), ("bravo-2", CHANNEL_B))

        assert adapter.shown(CHANNEL_A)[-1] == "re:alpha-2 ctx:alpha-1,alpha-2"
        assert adapter.shown(CHANNEL_B)[-1] == "re:bravo-2 ctx:bravo-1,bravo-2"
        # The typing indicator still covers each turn's model call.
        assert CHANNEL_A in llm.typing_at_call["alpha-2"]
        assert CHANNEL_B in llm.typing_at_call["bravo-2"]

    async def test_history_stays_separate_on_the_turn_after_the_race(self):
        llm = ContextEchoLLM()
        iface, adapter = await _started(llm)
        await _send_together(iface, ("alpha-1", CHANNEL_A), ("bravo-1", CHANNEL_B))

        await _send(iface, "alpha-2", CHANNEL_A)
        await _send(iface, "bravo-2", CHANNEL_B)

        assert _markers(llm.requests["alpha-2"]) == {"alpha-1", "alpha-2"}
        assert _markers(llm.requests["bravo-2"]) == {"bravo-1", "bravo-2"}
        assert adapter.shown(CHANNEL_A)[-1] == "re:alpha-2 ctx:alpha-1,alpha-2"


# ---------------------------------------------------------------------------
# AC3 — single-channel and non-overlapping behaviour is unchanged
# ---------------------------------------------------------------------------

class TestNonOverlappingBehaviourUnchanged:
    async def test_single_channel_turns_log_under_that_channel(self, log_dir):
        iface, _ = await _started(ContextEchoLLM())

        await _send(iface, "alpha-1", CHANNEL_A)
        await _send(iface, "alpha-2", CHANNEL_A)

        assert _turn_inputs(log_dir, CHANNEL_A) == ["alpha-1", "alpha-2"]
        for text in ("alpha-1", "alpha-2"):
            assert _turn_events(_log_holding(log_dir, text), text)[-1] == "turn.complete"

    async def test_single_channel_history_accumulates(self):
        llm = ContextEchoLLM()
        iface, adapter = await _started(llm)

        await _send(iface, "alpha-1", CHANNEL_A)
        await _send(iface, "alpha-2", CHANNEL_A)
        await _send(iface, "alpha-3", CHANNEL_A)

        assert _markers(llm.requests["alpha-1"]) == {"alpha-1"}
        assert _markers(llm.requests["alpha-3"]) == {"alpha-1", "alpha-2", "alpha-3"}
        assert adapter.shown(CHANNEL_A)[-1] == "re:alpha-3 ctx:alpha-1,alpha-2,alpha-3"

    async def test_sequential_channels_keep_separate_logs_and_history(self, log_dir):
        llm = ContextEchoLLM()
        iface, adapter = await _started(llm)

        await _send(iface, "alpha-1", CHANNEL_A)
        await _send(iface, "bravo-1", CHANNEL_B)
        await _send(iface, "alpha-2", CHANNEL_A)
        await _send(iface, "bravo-2", CHANNEL_B)

        assert _turn_inputs(log_dir, CHANNEL_A) == ["alpha-1", "alpha-2"]
        assert _turn_inputs(log_dir, CHANNEL_B) == ["bravo-1", "bravo-2"]
        assert _markers(llm.requests["alpha-2"]) == {"alpha-1", "alpha-2"}
        assert _markers(llm.requests["bravo-2"]) == {"bravo-1", "bravo-2"}
        assert adapter.shown(CHANNEL_A)[-1] == "re:alpha-2 ctx:alpha-1,alpha-2"
        assert adapter.shown(CHANNEL_B)[-1] == "re:bravo-2 ctx:bravo-1,bravo-2"
        for text, channel in (
            ("alpha-1", CHANNEL_A), ("bravo-1", CHANNEL_B),
            ("alpha-2", CHANNEL_A), ("bravo-2", CHANNEL_B),
        ):
            assert channel in llm.typing_at_call[text], (
                f"{text}'s model call ran outside its channel's typing indicator"
            )

    async def test_typing_indicator_covers_interleaved_model_calls(self):
        llm = ContextEchoLLM()
        iface, _ = await _started(llm)
        await _seed_two_channels(iface)

        await _send_together(iface, ("bravo-2", CHANNEL_B), ("alpha-2", CHANNEL_A))

        assert CHANNEL_A in llm.typing_at_call["alpha-2"]
        assert CHANNEL_B in llm.typing_at_call["bravo-2"]

    async def test_each_message_still_gets_its_own_log_file(self, log_dir):
        iface, _ = await _started(ContextEchoLLM())

        await _send(iface, "alpha-1", CHANNEL_A)
        await _send(iface, "alpha-2", CHANNEL_A)

        a = _log_holding(log_dir, "alpha-1")
        b = _log_holding(log_dir, "alpha-2")
        assert a != b, "both turns landed in one log file"
        assert [e["event"] for e in a].count("turn.start") == 1
        assert [e["event"] for e in b].count("turn.start") == 1
