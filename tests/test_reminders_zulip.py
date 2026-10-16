"""Reminders — Zulip creation and delivery (spc-139).

Direct-profile unit tests for the spc-139 AC subset (AC3, AC4, AC5, AC7,
AC8, AC12). Everything is faked at the declared seams:

- the clock is an injected callable returning a controlled UTC datetime
  (no real waiting anywhere);
- the Zulip send boundary is a recording fake (no server, no credentials);
- the store path is a per-test temp file — no network, no ~/.sr2 live
  state.

The tool, store and delivery loop run on the real asyncio event loop so
the claim/mark transaction ordering (FR17) is actually exercised.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from sr2_spectre.reminders import (
    DeliveryLoop,
    ReminderOrigin,
    ReminderStore,
    ReminderTool,
)

TZ = ZoneInfo("America/Phoenix")
UTC = ZoneInfo("UTC")

# Fixed "now" so no test ever waits on the wall clock.
NOW = datetime(2026, 10, 15, 22, 0, 0, tzinfo=UTC)  # 15:00 MST


def _clock_at(moment: datetime):
    """A clock frozen at *moment*."""
    return lambda: moment


class FakeSender:
    """Records every (channel, topic, content) the loop asked to send."""

    def __init__(self) -> None:
        self.sends: list[tuple[int, str, str]] = []

    async def send(self, channel_id: int, topic: str, content: str) -> None:
        self.sends.append((int(channel_id), topic, content))


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "reminders.sqlite"


@pytest.fixture
def store(store_path: Path) -> ReminderStore:
    return ReminderStore(store_path)


def _origin(**overrides) -> ReminderOrigin:
    """A Zulip turn origin: Diego in channel 42, topic 'plan'."""
    base = dict(
        agent="miranda",
        platform="zulip",
        requester_id="8",
        requester_mention="@8",
        channel_id="42",
        channel_name="sr2-spectre",
        topic="plan",
    )
    base.update(overrides)
    return ReminderOrigin(**base)


def _tool(
    store: ReminderStore,
    now: datetime = NOW,
    origin: ReminderOrigin | None = None,
) -> ReminderTool:
    return ReminderTool(
        store=store,
        timezone="America/Phoenix",
        agent_name="miranda",
        origin_getter=lambda: origin,
        clock=_clock_at(now),
    )


def _loop(
    store: ReminderStore,
    sender: FakeSender,
    now: datetime,
    poll_seconds: float = 0.01,
) -> DeliveryLoop:
    return DeliveryLoop(
        store,
        send=sender.send,
        poll_seconds=poll_seconds,
        tz=TZ,
        agent_name="miranda",
        clock=_clock_at(now),
    )


async def _settle(loop: DeliveryLoop, polls: int = 3) -> None:
    """Let the loop run a few polls (injected clock: no real waiting)."""
    for _ in range(polls):
        await asyncio.sleep(0.02)


def _agent_config(reminders: dict | None) -> Any:
    """A minimal real SpectreConfig for Runtime/Agent construction."""
    from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig

    agent: dict[str, Any] = {"name": "miranda", "default_skills": False}
    if reminders is not None:
        agent["reminders"] = reminders
    return SpectreConfig(
        agent=AgentConfig(**agent),
        models={"default": ModelConfig(model="test-model")},
        pipeline={"layers": []},
        provenance_store_path="",
        memory_store_dsn="",
    )


def _make_agent(config: Any) -> Any:
    """A real Agent with the documented LLM seam patched (no model call)."""
    from sr2_spectre.agent import Agent

    llm = object.__new__(type("FakeLLM", (), {}))
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=llm):
        return Agent(config=config)


def _make_interface(tmp_path: Path) -> Any:
    """A ZulipInterface with a fake client: no server, no credentials,
    no network. Only the reminders wiring is exercised via start/stop.

    The fake client stays patched for the interface's life; the test's
    event loop is the single loop, so the patch's exit is a no-op re-entry
    guard (same object swapped back in).
    """
    from sr2_spectre.interfaces.zulip.client import ZulipCredentials
    import sr2_spectre.interfaces.zulip.interface as iface_mod
    from sr2_spectre.interfaces.zulip.config import ZulipConfig

    config = ZulipConfig(
        zuliprc=str(tmp_path / "zuliprc"),
        allowed_user_ids=[8],
        channels={42: ""},
    )

    class _FakeClient:
        def __init__(self) -> None:
            self.sends: list[dict] = []

        async def register(self):
            return {
                "queue_id": "q-test",
                "last_event_id": 0,
                "realm_users": [{"user_id": 9, "is_bot": True}],
                "realm_bots": [],
            }

        async def aclose(self):
            pass

        async def send_message(self, **kwargs):
            self.sends.append(kwargs)
            return 1

    # Patch for the interface's whole life (construction + start + stop all
    # happen inside one test, one event loop).
    iface_mod.load_zulip_credentials = lambda path: ZulipCredentials(
        email="bot@test.local", api_key="k", site="http://127.0.0.1:1"
    )
    iface_mod.ZulipClient = lambda **kw: _FakeClient()
    return iface_mod.ZulipInterface(config)


# ---------------------------------------------------------------------------
# AC3 — the one switch: disabled registers nothing, enabled registers both
# ---------------------------------------------------------------------------


class TestAC3Switch:
    async def test_disabled_registers_nothing(self, tmp_path):
        """AC3: reminders disabled (the default) registers no tool and
        starts no loop."""
        agent = _make_agent(_agent_config(reminders=None))
        try:
            assert "reminder" not in agent.registry
            assert agent.reminders is None
            assert agent._runtime.reminders_enabled is False
            interface = _make_interface(tmp_path)
            await interface.start(agent)
            try:
                assert interface._reminders_loop is None
            finally:
                await interface.stop()
        finally:
            await agent.aclose()

    async def test_enabled_registers_tool_and_starts_loop(self, tmp_path):
        """AC3: reminders enabled registers the tool and the interface
        start() begins the delivery loop on the shared store."""
        config = _agent_config(
            reminders={
                "enabled": True,
                "timezone": "America/Phoenix",
                "store_path": str(tmp_path / "r.sqlite"),
                "poll_seconds": 30,
            }
        )
        agent = _make_agent(config)
        try:
            assert "reminder" in agent.registry
            assert agent.reminders is not None
            assert agent._runtime.reminders_enabled is True

            interface = _make_interface(tmp_path)
            await interface.start(agent)
            try:
                assert interface._reminders_loop is not None
                loop = interface._reminders_loop
                # The loop shares the tool's store (one source of truth).
                assert loop._store is agent.reminders.store
                # The delivery loop was started: a start task was created.
                assert interface._reminders_start_task is not None
            finally:
                await interface.stop()
            assert interface._reminders_loop is None
        finally:
            await agent.aclose()

    async def test_enabled_requires_timezone(self, tmp_path):
        """AC3 (FR6): enabled without an IANA timezone fails startup."""
        config = _agent_config(reminders={"enabled": True, "timezone": ""})
        with pytest.raises(ValueError, match="timezone"):
            _make_agent(config)


# ---------------------------------------------------------------------------
# AC4 — create validation per FR8: each invalid case errors, store unchanged
# ---------------------------------------------------------------------------


class TestAC4CreateValidation:
    async def test_empty_text(self, store):
        """AC4: empty/whitespace text → tool error, store unchanged."""
        tool = _tool(store, origin=_origin())
        result = await tool(action="create", text="   ", in_minutes=60)
        assert result.startswith("Error:")
        assert store.count_by_status() == {}

    async def test_missing_text(self, store):
        """AC4: text omitted → tool error, store unchanged."""
        tool = _tool(store, origin=_origin())
        result = await tool(action="create", in_minutes=60)
        assert result.startswith("Error:")
        assert store.count_by_status() == {}

    async def test_text_over_1000_chars(self, store):
        """AC4: text > 1000 chars → tool error, store unchanged."""
        tool = _tool(store, origin=_origin())
        result = await tool(action="create", text="x" * 1001, in_minutes=60)
        assert result.startswith("Error:")
        assert store.count_by_status() == {}

    async def test_both_fields(self, store):
        """AC4: both in_minutes and at → tool error, store unchanged."""
        tool = _tool(store, origin=_origin())
        result = await tool(
            action="create", text="hi", in_minutes=60, at="2026-10-15 16:00"
        )
        assert result.startswith("Error:")
        assert store.count_by_status() == {}

    async def test_neither_field(self, store):
        """AC4: neither in_minutes nor at → tool error, store unchanged."""
        tool = _tool(store, origin=_origin())
        result = await tool(action="create", text="hi")
        assert result.startswith("Error:")
        assert store.count_by_status() == {}

    async def test_past_time(self, store):
        """AC4: at in the past → tool error, store unchanged."""
        tool = _tool(store, origin=_origin())
        result = await tool(action="create", text="hi", at="2026-10-15 14:00")
        assert result.startswith("Error:")
        assert store.count_by_status() == {}

    async def test_due_less_than_one_minute_ahead(self, store):
        """AC4: due < 1 minute ahead (in_minutes and at) → tool error."""
        tool = _tool(store, origin=_origin())
        # in_minutes=1: due exactly 1 minute ahead, not ahead of it.
        r1 = await tool(action="create", text="hi", in_minutes=1)
        assert r1.startswith("Error:")
        # at: 30s ahead of now (15:00:30 MST) — less than one minute.
        r2 = await tool(action="create", text="hi", at="2026-10-15 15:00:30")
        assert r2.startswith("Error:")
        assert store.count_by_status() == {}

    async def test_in_minutes_out_of_range(self, store):
        """AC4: in_minutes 0, negative, or > 525600 → tool error."""
        tool = _tool(store, origin=_origin())
        for bad in (0, -5, 525601):
            result = await tool(action="create", text="hi", in_minutes=bad)
            assert result.startswith("Error:"), f"in_minutes={bad}: {result}"
        assert store.count_by_status() == {}

    async def test_malformed_at(self, store):
        """AC4: at not in YYYY-MM-DD HH:MM → tool error, store unchanged."""
        tool = _tool(store, origin=_origin())
        for bad in ("2026-10-15T16:00", "15:00", "2026-13-40 25:99", "soon"):
            result = await tool(action="create", text="hi", at=bad)
            assert result.startswith("Error:"), f"at={bad!r}: {result}"
        assert store.count_by_status() == {}

    async def test_no_origin_errors(self, store):
        """AC4 (FR21): no turn origin (non-chat interface) → tool error."""
        tool = _tool(store, origin=None)
        result = await tool(action="create", text="hi", in_minutes=60)
        assert result.startswith("Error:")
        assert store.count_by_status() == {}


# ---------------------------------------------------------------------------
# AC5 — create without channel: origin channel, reply with ID/time/channel
# ---------------------------------------------------------------------------


class TestAC5CreateDefaultDestination:
    async def test_stores_origin_channel_and_topic(self, store):
        """AC5 (FR9): destination = origin channel, topic = the configured
        Zulip topic (default 'reminders'); text and requester stored."""
        tool = _tool(store, origin=_origin())
        result = await tool(action="create", text="look into x", in_minutes=180)
        assert result.startswith("Reminder r")

        assert store.count_by_status() == {"pending": 1}
        row = store.due(now=NOW + timedelta(hours=4))[0]
        assert row.dest_channel_id == "42"
        assert row.dest_channel_name == "sr2-spectre"
        # FR9: the Zulip destination topic is reminders.zulip_topic
        # (default 'reminders'), NOT the origin topic.
        assert row.dest_topic == "reminders"
        assert row.requester_id == "8"
        assert row.requester_mention == "@8"
        assert row.text == "look into x"
        assert row.due_at == NOW + timedelta(minutes=180)
        assert row.platform == "zulip"

    async def test_at_interpreted_in_timezone(self, store):
        """AC5: 'at' is local time in reminders.timezone (stored UTC)."""
        tool = _tool(store, origin=_origin())
        result = await tool(action="create", text="t", at="2026-10-15 17:00")
        assert result.startswith("Reminder r")
        row = store.due(now=NOW + timedelta(hours=3))[0]
        # 17:00 MST == 00:00 UTC the next day.
        assert row.due_at == datetime(2026, 10, 16, 0, 0, tzinfo=UTC)

    async def test_reply_has_id_local_time_and_channel(self, store):
        """AC5 (FR12): reply returns ID, local due time (weekday, date,
        time, zone) and the destination channel name."""
        tool = _tool(store, origin=_origin())
        result = await tool(action="create", text="t", in_minutes=180)
        # 15:00 MST now + 3h = Thu 18:00 MST.
        assert result == "Reminder r1 set for Thu 2026-10-15 18:00 MST in #sr2-spectre"

    async def test_id_format_r_n_increasing(self, store):
        """AC5: IDs are r<N>, unique per store and increasing."""
        tool = _tool(store, origin=_origin())
        r1 = await tool(action="create", text="a", in_minutes=60)
        r2 = await tool(action="create", text="b", in_minutes=120)
        assert r1.startswith("Reminder r1 set for")
        assert r2.startswith("Reminder r2 set for")

    async def test_channel_argument_not_supported_yet(self, store):
        """spc-139 contract: a channel argument returns the explicit tool
        error and stores nothing (spc-140 replaces this)."""
        tool = _tool(store, origin=_origin())
        result = await tool(
            action="create", text="t", in_minutes=60, channel="general"
        )
        assert result == "Error: Named channels are not supported yet"
        assert store.count_by_status() == {}


# ---------------------------------------------------------------------------
# AC7 — due reminder sent once, mention + verbatim text, marked delivered;
#         not-yet-due never sent; late suffix
# ---------------------------------------------------------------------------


class TestAC7Delivery:
    async def test_due_sent_once_and_delivered(self, store):
        """AC7: a due reminder is sent exactly once to its destination with
        the requester mention and verbatim text, then marked delivered."""
        sender = FakeSender()
        # Due at NOW+100min; the loop's clock is past that, frozen.
        loop = _loop(store, sender, now=NOW + timedelta(minutes=101))
        await loop.start()

        tool = _tool(store, now=NOW, origin=_origin())
        await tool(action="create", text="look into https://example.com/x", in_minutes=100)

        await _settle(loop)
        await loop.stop()

        assert len(sender.sends) == 1
        channel, topic, content = sender.sends[0]
        assert channel == 42
        assert topic == "reminders"  # FR9 destination topic
        assert content == "@8 Reminder: look into https://example.com/x"

        row = store.get("r1")
        assert row.status == "delivered"
        assert row.delivered_at == NOW + timedelta(minutes=101)

    async def test_not_yet_due_not_sent(self, store):
        """AC7 (FR14): a not-yet-due reminder is never sent."""
        sender = FakeSender()
        loop = _loop(store, sender, now=NOW + timedelta(minutes=10))
        await loop.start()

        tool = _tool(store, now=NOW, origin=_origin())
        await tool(action="create", text="later", in_minutes=120)

        await _settle(loop)
        await loop.stop()
        assert sender.sends == []
        assert store.get("r1").status == "pending"

    async def test_late_suffix_after_five_minutes(self, store):
        """AC7 (FR15): sent >5 min after due carries '(due <local>,
        sent late)'; on time it does not."""
        sender = FakeSender()
        # r1: due NOW+10min, poll at NOW+20min → 10 min late.
        # r2: due NOW+19min, poll at NOW+20min → 1 min late → no suffix.
        loop = _loop(store, sender, now=NOW + timedelta(minutes=20))
        await loop.start()
        tool = _tool(store, now=NOW, origin=_origin())
        await tool(action="create", text="late one", in_minutes=10)
        await tool(action="create", text="on time", at="2026-10-15 15:19")

        await _settle(loop)
        await loop.stop()
        assert len(sender.sends) == 2
        late = next(s[2] for s in sender.sends if "late one" in s[2])
        on_time = next(s[2] for s in sender.sends if "on time" in s[2])
        assert late.startswith("@8 Reminder: late one")
        assert "sent late" in late
        assert "due Thu 2026-10-15 15:10 MST" in late
        assert "sent late" not in on_time


# ---------------------------------------------------------------------------
# AC8 — overdue-at-start sent in due order with late suffix; sending rows
#         are resent
# ---------------------------------------------------------------------------


class TestAC8StartupCatchup:
    async def test_overdue_sent_in_due_order_with_late_suffix(self, store):
        """AC8 (FR16): reminders overdue at start are sent in due-time
        order, each marked late per FR15."""
        # Seed two overdue rows (created before this process started).
        tool_old = _tool(store, now=NOW - timedelta(hours=3), origin=_origin())
        await tool_old(action="create", text="oldest", at="2026-10-15 13:00")
        tool_newer = _tool(store, now=NOW - timedelta(hours=2), origin=_origin())
        await tool_newer(action="create", text="newer", at="2026-10-15 14:00")
        assert [r.text for r in store.due(now=NOW)] == ["oldest", "newer"]

        sender = FakeSender()
        loop = _loop(store, sender, now=NOW)
        await loop.start()  # the first poll is the startup catch-up
        await _settle(loop)
        await loop.stop()

        assert len(sender.sends) == 2
        first, second = sender.sends
        # Due order: r1 (13:00) before r2 (14:00).
        assert "oldest" in first[2]
        assert "newer" in second[2]
        # Both were sent >5 min after their due times → late suffix on both.
        assert "due Thu 2026-10-15 13:00 MST" in first[2] and "sent late" in first[2]
        assert "due Thu 2026-10-15 14:00 MST" in second[2] and "sent late" in second[2]
        assert store.get("r1").status == "delivered"
        assert store.get("r2").status == "delivered"

    async def test_sending_rows_are_resent_on_start(self, store):
        """AC8 (FR17): a row left 'sending' by a crashed process is returned
        to pending at start and resent."""
        tool = _tool(store, now=NOW - timedelta(hours=2), origin=_origin())
        await tool(action="create", text="crash mid-send", at="2026-10-15 14:00")
        assert store.claim("r1") is True  # pending → sending, as a crash left it
        assert store.get("r1").status == "sending"

        sender = FakeSender()
        loop = _loop(store, sender, now=NOW)
        await loop.start()  # start() resets sending → pending

        await _settle(loop)
        await loop.stop()

        assert len(sender.sends) == 1
        assert "crash mid-send" in sender.sends[0][2]
        assert store.get("r1").status == "delivered"


# ---------------------------------------------------------------------------
# AC12 — stop() ends the loop, pending rows intact
# ---------------------------------------------------------------------------


class TestAC12Stop:
    async def test_stop_ends_loop_and_leaves_pending_rows(self, store):
        """AC12 (FR22): stop() ends the delivery loop; pending rows are left
        intact, and a claim interrupted mid-flight is retried at the next
        start (FR17)."""
        sender = FakeSender()
        loop = _loop(store, sender, now=NOW)
        await loop.start()

        # One reminder due before the loop polls; one far in the future.
        tool = _tool(store, now=NOW - timedelta(minutes=10), origin=_origin())
        await tool(action="create", text="due", at="2026-10-15 15:00")
        tool2 = _tool(store, now=NOW - timedelta(minutes=10), origin=_origin())
        await tool2(action="create", text="future", at="2027-01-01 00:00")

        await _settle(loop)
        assert [s[2] for s in sender.sends] == ["@8 Reminder: due"]

        # A third row is claimed (mid-flight) when the loop stops before
        # its mark: the next start must resend it.
        tool3 = _tool(store, now=NOW - timedelta(minutes=10), origin=_origin())
        await tool3(action="create", text="interrupted", at="2026-10-15 15:30")
        assert store.claim("r3") is True  # simulate an in-flight claim

        await loop.stop()
        assert loop._task is None

        statuses = store.count_by_status()
        assert statuses.get("delivered") == 1
        assert statuses.get("pending") == 1  # 'future' untouched
        assert statuses.get("sending") == 1  # the interrupted claim survives stop()

        # Next process start: sending → pending; the due row is resent,
        # the not-due one is not.
        sender2 = FakeSender()
        loop2 = _loop(store, sender2, now=NOW + timedelta(minutes=31))
        await loop2.start()
        await _settle(loop2)
        await loop2.stop()
        assert any("interrupted" in s[2] for s in sender2.sends)
        assert not any("future" in s[2] for s in sender2.sends)
        assert store.get("r3").status == "delivered"
        assert store.get("r2").status == "pending"
