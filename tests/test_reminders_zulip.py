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
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from sr2_spectre.reminders import (
    DeliveryLoop,
    Reminder,
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
    """Records every (channel, topic, content) the loop asked to send.

    ``fail``: "always" raises on every send; "twice" raises on the first
    two and succeeds after (the retry-then-deliver path).
    """

    def __init__(self, fail: str | None = None) -> None:
        self.sends: list[tuple[int, str, str]] = []  # successful sends only
        self.attempts = 0  # every send call, successful or not
        self._fail = fail
        self._failures = 0

    async def send(self, channel_id: int, topic: str, content: str) -> None:
        self.attempts += 1
        if self._fail == "always" or (self._fail == "twice" and self._failures < 2):
            self._failures += 1
            raise RuntimeError("boom")
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
    channel_resolver=None,
) -> ReminderTool:
    return ReminderTool(
        store=store,
        timezone="America/Phoenix",
        agent_name="miranda",
        origin_getter=lambda: origin,
        clock=_clock_at(now),
        channel_resolver=channel_resolver,
    )


class FakeChannel:
    """One channel match as the interface's resolver reports it (FR10:
    an object with channel_id and channel_name)."""

    def __init__(self, channel_id, channel_name) -> None:
        self.channel_id = channel_id
        self.channel_name = channel_name


def _resolver(match_lists: dict):
    """A fake resolver: name (lowercased) → list of FakeChannel."""

    async def resolve(platform: str, name: str):
        return match_lists.get(name.lower(), [])

    return resolve


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
                "streams": [
                    {"stream_id": 42, "name": "sr2-spectre"},
                    {"stream_id": 7, "name": "general"},
                ],
            }

        async def aclose(self):
            pass

        async def get_streams(self):
            return {
                "streams": [
                    {"stream_id": 42, "name": "sr2-spectre", "can_send": True},
                    {"stream_id": 7, "name": "general", "can_send": True},
                    {"stream_id": 8, "name": "read-only", "can_send": False},
                ]
            }

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

    async def test_channel_without_resolver_errors(self, store):
        """spc-140 replaces the spc-139 placeholder: without a resolver
        (the Runtime builds the tool before any interface exists) a channel
        argument errors and stores nothing."""
        tool = _tool(store, origin=_origin())
        result = await tool(
            action="create", text="t", in_minutes=60, channel="general"
        )
        assert result.startswith("Error:")
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


# ---------------------------------------------------------------------------
# AC6 — create with channel: named resolution per FR10/FR11 (spc-140)
# ---------------------------------------------------------------------------


class TestAC6NamedChannel:
    async def test_named_channel_resolves_by_name(self, store):
        """AC6 (FR10): a leading # is stripped and the name resolves
        case-insensitively; the resolved ID and name are stored and the
        reply names the resolved destination."""
        resolver = _resolver({"general": [FakeChannel(7, "general")]})
        tool = _tool(store, origin=_origin(), channel_resolver=resolver)
        result = await tool(
            action="create", text="t", in_minutes=60, channel="#General"
        )
        assert result == "Reminder r1 set for Thu 2026-10-15 16:00 MST in #general"
        row = store.get("r1")
        assert row.dest_channel_id == "7"
        assert row.dest_channel_name == "general"
        # FR10: Zulip keeps the reminders topic even for a named channel.
        assert row.dest_topic == "reminders"

    async def test_unknown_channel_errors_stores_nothing(self, store):
        """AC6 (FR10): a name with no match is a tool error, stored nothing."""
        tool = _tool(store, origin=_origin(), channel_resolver=_resolver({}))
        result = await tool(
            action="create", text="t", in_minutes=60, channel="nope"
        )
        assert result.startswith("Error:")
        assert store.count_by_status() == {}

    async def test_ambiguous_channel_errors_stores_nothing(self, store):
        """AC6 (FR10): more than one match is a tool error, stored nothing."""
        resolver = _resolver(
            {"dup": [FakeChannel(1, "Dup"), FakeChannel(2, "dup")]}
        )
        tool = _tool(store, origin=_origin(), channel_resolver=resolver)
        result = await tool(
            action="create", text="t", in_minutes=60, channel="dup"
        )
        assert result.startswith("Error:")
        assert store.count_by_status() == {}

    async def test_cross_platform_names_refused(self, store):
        """AC6 (FR11): 'discord#x' from a Zulip turn is refused with the
        exact error, nothing stored and no platform lookup attempted."""
        calls: list = []

        async def spy(platform, name):
            calls.append((platform, name))
            return []

        tool = _tool(store, origin=_origin(), channel_resolver=spy)
        for bad in ("discord#general", "discord#general"):
            result = await tool(
                action="create", text="t", in_minutes=60, channel=bad
            )
            assert result == "Error: Cross-platform reminders are not supported"
        assert store.count_by_status() == {}
        assert calls == []

    async def test_interface_resolver_from_registration_snapshot(self, tmp_path):
        """AC6 through the real ZulipInterface: registration streams give
        the origin its channel NAME (FR12), and the interface resolver
        matches by name case-insensitively, dropping can_send=False."""
        interface = _make_interface(tmp_path)
        await interface.start(object())  # start() only reads .reminders
        try:
            assert interface._channel_display_name(42) == "sr2-spectre"
            assert interface._channel_display_name(99) == "99"
            assert interface._channel_display_name(7) == "general"

            matches = await interface._resolve_channel("zulip", "GeneRal")
            assert [(m.channel_id, m.channel_name) for m in matches] == [(7, "general")]
            assert await interface._resolve_channel("zulip", "read-only") == []
            assert await interface._resolve_channel("discord", "general") == []
        finally:
            await interface.stop()

    async def test_seam_named_channel_through_the_real_resolver(self, tmp_path):
        """AC6 seam (spc-166): start() registers the real interface
        resolver on the agent's Runtime seam, which seats a tool built
        with it in the startup tool's place — so a named-channel create
        through the CURRENT handle stores the resolved destination
        instead of crashing (the acceptor-repro regression).

        The agent is a real Agent over a real Runtime (as ``FakeAgent``
        in tests/test_reminders_resolver_seam.py mirrors), so the tool
        under test is the Runtime's own seat and the resolver can only
        have arrived through the seam and its constructor. The store file
        is the assertions' store; the clock is pinned by READBACK so the
        reply text is exact (the Runtime seat runs the real clock).
        """
        store_file = tmp_path / "r-seam.sqlite"
        agent = _make_agent(
            _agent_config(
                reminders={
                    "enabled": True,
                    "timezone": "America/Phoenix",
                    "store_path": str(store_file),
                    "poll_seconds": 30,
                }
            )
        )
        try:
            interface = _make_interface(tmp_path)
            tool_before = agent.reminders.tool
            await interface.start(agent)
            try:
                # spc-166: registration rebuilt the seat through the
                # Runtime's single construction site.
                tool = agent.reminders.tool
                assert tool is not tool_before
                tool._origin_getter = lambda: _origin()  # READBACK
                tool._clock = _clock_at(NOW)  # READBACK
                result = await tool(
                    action="create", text="t", in_minutes=60, channel="general"
                )
                assert result == "Reminder r1 set for Thu 2026-10-15 16:00 MST in #general"
                row = ReminderStore(store_file).get("r1")
                assert row.dest_channel_id == "7"
                assert row.dest_channel_name == "general"
            finally:
                await interface.stop()
        finally:
            await agent.aclose()

    async def test_malformed_resolver_result_is_a_tool_error(self, store):
        """A resolver yielding raw dicts returns the guard error, stores
        nothing, raises nothing."""
        from sr2_spectre.reminders.tool import MALFORMED_RESOLVER_ERROR

        async def bad_resolver(platform, name):
            return [{"stream_id": 7, "name": "general"}]

        tool = _tool(store, origin=_origin(), channel_resolver=bad_resolver)
        result = await tool(
            action="create", text="t", in_minutes=60, channel="general"
        )
        assert result == MALFORMED_RESOLVER_ERROR
        assert store.count_by_status() == {}


# ---------------------------------------------------------------------------
# AC9 — bounded retry per FR18: +1 min, +5 min, failed with ERROR log
# ---------------------------------------------------------------------------


class TestAC9RetryPolicy:
    async def test_retries_at_plus_one_and_plus_five_then_fails(
        self, store, caplog
    ):
        """AC9 (FR18): a failing send re-seats due_at at due+1min and
        due+5min (attempts 1 and 2); the third failure marks the row
        failed with an ERROR log naming its ID and the error. A 30s
        polling cadence (two seconds after each schedule point, always
        within the FR15 late threshold) proves each attempt lands at its
        own time."""
        sender = FakeSender(fail="always")
        _seed(store, "flaky", NOW)

        def stored_due():
            with store._connect() as conn:
                return tuple(conn.execute(
                    "SELECT due_at, next_attempt_at FROM reminders WHERE id = 'r1'"
                ).fetchone())

        original_due, original_next = stored_due()
        dues = []
        with caplog.at_level(logging.INFO, logger="sr2_spectre.reminders.loop"):
            for offset in range(14):  # NOW .. NOW + 6.5 min
                loop = _loop(store, sender, now=NOW + timedelta(seconds=30 * offset + 2))
                await loop.start()
                await _settle(loop)
                await loop.stop()
                dues.append(stored_due())

        assert sender.attempts == 3
        assert len(sender.sends) == 0
        row = store.get("r1")
        assert row.status == "failed"
        assert row.attempts == 3
        # The retry schedule: due → due+1min → due+5min → terminal. dues[k]
        # is read after the poll at NOW+30k+2s: the first poll (attempt at
        # due) already re-seated to +1min; the change to +5min waits for
        # the attempt at due+1min, i.e. the poll at NOW+62s; that attempt
        # re-seats to (due+1min)+5min = due+6min — the retry schedule is
        # cumulative, matching "retried 1 minute and then 5 minutes later".
        assert all(due == original_due for due, _ in dues)  # due_at never moves
        nexts = [nxt for _, nxt in dues]
        assert nexts[0] != original_next and nexts[1] == nexts[0]
        assert _dt_from(nexts[0]) == NOW + timedelta(minutes=1)
        changed = [n for n in nexts if n != nexts[0]]
        assert changed and nexts.index(changed[0]) == 2  # poll at NOW+62s
        assert _dt_from(changed[0]) == NOW + timedelta(minutes=6)
        assert all(n == changed[0] for n in changed)  # terminal: no re-seat
        errors = [r for r in caplog.records
                  if r.levelno == logging.ERROR and "r1" in r.getMessage()]
        assert errors and "boom" in errors[-1].getMessage()

    async def test_recovers_after_two_failures(self, store):
        """AC9: a send that fails twice still delivers on the +5 retry."""
        sender = FakeSender(fail="twice")
        _seed(store, "stubborn", NOW)
        for offset in range(14):  # 30s cadence, two seconds past each point
            loop = _loop(store, sender,
                         now=NOW + timedelta(seconds=30 * offset + 2))
            await loop.start()
            await _settle(loop)
            await loop.stop()
        assert sender.attempts == 3  # due, +1 min, +5 min
        assert len(sender.sends) == 1
        assert "stubborn" in sender.sends[-1][2]
        assert store.get("r1").status == "delivered"

    async def test_other_due_reminder_still_sent(self, store):
        """AC9 (FR18): one row failing every poll never blocks another
        due reminder, which is delivered normally."""
        sender = FakeSender()

        async def send(channel_id, topic, content):
            if "bad" in content:
                raise RuntimeError("boom")
            sender.sends.append((int(channel_id), topic, content))
            sender.attempts += 1

        loop = DeliveryLoop(
            store, send=send, poll_seconds=0.01, tz=TZ, agent_name="miranda",
            clock=_clock_at(NOW),
        )
        # Both rows seeded overdue; NOW is frozen, so the bad row fails on
        # every poll (attempts 1, 2, 3 → failed) while the good row is
        # delivered on the first.
        _seed(store, "bad one", NOW - timedelta(minutes=10))
        _seed(store, "good one", NOW - timedelta(minutes=9))

        await loop.start()
        await _settle(loop, polls=6)
        await loop.stop()

        assert [s[2].split("(due")[0] for s in sender.sends] == [
            "@8 Reminder: good one"
        ]
        assert store.get("r1").status == "failed"
        assert store.get("r1").attempts == 3
        assert store.get("r2").status == "delivered"


# ---------------------------------------------------------------------------
# AC10 — list and cancel per FR19/FR20; cancelled is never sent
# ---------------------------------------------------------------------------


def _seed(store: ReminderStore, text: str, due_at: datetime,
          created_at: datetime | None = None, agent: str = "miranda") -> str:
    """Insert one row directly (bypasses tool validation)."""
    return store.create(
        Reminder(
            id="r0", agent=agent, platform="zulip",
            requester_id="8", requester_mention="@8",
            origin_channel_id="42", dest_channel_id="42",
            dest_channel_name="sr2-spectre", dest_topic="reminders",
            text=text, due_at=due_at,
            created_at=created_at if created_at is not None else NOW,
            delivered_at=None, status="pending", attempts=0, last_error=None,
        )
    )


def _mark(store: ReminderStore, reminder_id: str, status: str) -> None:
    conn = store._connect()
    conn.execute(
        "UPDATE reminders SET status = ? WHERE id = ?", (status, reminder_id)
    )
    conn.commit()
    conn.close()


def _ts(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _dt_from(value: str) -> datetime:
    return datetime.fromisoformat(value)


class TestAC10ListCancel:
    async def test_list_shows_pending_sorted_and_recent_failed(self, store):
        """AC10 (FR19): pending plus failed-within-7-days, sorted by due
        time; each line carries ID, local due time, destination, status
        and text truncated to 80 characters. A retry-failed row's due_at
        already holds its final attempt time (the FR18 re-seat), so the
        7-day window filters on due_at directly. Cancelled, delivered and
        older failed rows stay out."""
        r_second = _seed(store, "second", NOW + timedelta(minutes=30))
        r_first = _seed(store, "first", NOW + timedelta(minutes=10))
        r_long = _seed(store, "x" * 100, NOW + timedelta(minutes=20))
        r_recent = _seed(store, "recent failed", NOW - timedelta(days=1))
        _mark(store, r_recent, "failed")
        # Failed 6 days ago (its due_at is its last attempt): inside the
        # window, and first in due order.
        r_old = _seed(store, "old failed", NOW - timedelta(days=6))
        _mark(store, r_old, "failed")
        # Failed 8 days ago: outside the 7-day window, stays out.
        r_stale = _seed(store, "stale failed", NOW - timedelta(days=8))
        _mark(store, r_stale, "failed")
        r_gone = _seed(store, "gone", NOW + timedelta(minutes=40))
        _mark(store, r_gone, "cancelled")
        r_done = _seed(store, "delivered one", NOW + timedelta(minutes=15))
        _mark(store, r_done, "delivered")
        _seed(store, "other agent", NOW + timedelta(minutes=5), agent="anna")

        tool = _tool(store, origin=_origin())
        result = await tool(action="list")
        lines = result.splitlines()
        assert [line.split()[0] for line in lines] == [
            r_old, r_recent, r_first, r_long, r_second,
        ]
        # ID, local due, destination, status, text — the first line in full.
        assert lines[0] == (
            f"{r_old} Fri 2026-10-09 15:00 MST #sr2-spectre failed old failed"
        )
        # 100 chars of text render as 79 + the ellipsis marker.
        long_line = next(line for line in lines if line.startswith(r_long))
        assert ("x" * 79 + "…") in long_line
        assert "x" * 80 not in result
        assert r_gone not in result and r_done not in result
        assert "stale failed" not in result
        assert "other agent" not in result

    async def test_list_empty(self, store):
        """AC10 (FR19): nothing pending → an explicit empty list."""
        tool = _tool(store, origin=_origin())
        assert await tool(action="list") == "No reminders."

    async def test_cancel_pending_never_sent(self, store):
        """AC10 (FR20): a pending reminder becomes cancelled; a loop
        running past its due time never sends it."""
        _seed(store, "cancelled one", NOW + timedelta(minutes=10))
        tool = _tool(store, origin=_origin())
        assert await tool(action="cancel", id="r1") == "Cancelled reminder r1"
        assert store.get("r1").status == "cancelled"

        sender = FakeSender()
        loop = _loop(store, sender, now=NOW + timedelta(minutes=30))
        await loop.start()
        await _settle(loop)
        await loop.stop()
        assert sender.sends == []

    async def test_cancel_unknown_and_terminal_states(self, store):
        """AC10 (FR20): unknown IDs and delivered/failed/cancelled rows
        each return a tool error and change nothing."""
        _seed(store, "a", NOW + timedelta(minutes=10))
        tool = _tool(store, origin=_origin())

        err = await tool(action="cancel", id="r99")
        assert err.startswith("Error:")
        for status in ("delivered", "failed", "cancelled"):
            _mark(store, "r1", status)
            err = await tool(action="cancel", id="r1")
            assert err.startswith("Error:"), status
            assert store.get("r1").status == status

    async def test_cancel_sending_reports_being_delivered(self, store):
        """AC10 (FR20): a reminder already claimed for sending completes
        its send; cancel only reports it as already being delivered."""
        _seed(store, "in flight", NOW - timedelta(minutes=1))
        assert store.claim("r1") is True
        tool = _tool(store, origin=_origin())
        result = await tool(action="cancel", id="r1")
        assert result == "Reminder r1 is already being delivered"
        assert store.get("r1").status == "sending"

        sender = FakeSender()
        loop = _loop(store, sender, now=NOW)
        await loop.start()
        await _settle(loop)
        await loop.stop()
        assert len(sender.sends) == 1  # the in-flight send completed
        assert store.get("r1").status == "delivered"


# ---------------------------------------------------------------------------
# AC11 — FR21: no chat origin → every action gets the same error
# ---------------------------------------------------------------------------


class TestAC11NoChatInterface:
    async def test_every_action_gets_the_fr21_error(self, store):
        """AC11 (FR21): an interface that is neither Discord nor Zulip
        records no origin; create, list and cancel each return the exact
        FR21 tool error and the store stays untouched."""
        tool = _tool(store, origin=None)
        expected = "Error: Reminders need the Discord or Zulip interface"
        results = [
            await tool(action="create", text="t", in_minutes=60),
            await tool(action="list"),
            await tool(action="cancel", id="r1"),
        ]
        assert results == [expected, expected, expected]
        assert store.count_by_status() == {}
