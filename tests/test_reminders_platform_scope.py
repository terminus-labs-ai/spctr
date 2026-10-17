"""Reminders — per-platform delivery scoping (spc-176).

Regression tests for the cross-claim bug: one agent (e.g. Miranda) runs
the Discord and Zulip interfaces at once over the agent's ONE reminder
store. Each ``DeliveryLoop`` serves exactly one platform, so due
selection, claim and startup recovery only ever touch rows whose
``platform`` matches the loop's platform.

One shared temporary store holds rows of BOTH platforms; every loop uses
a frozen clock and a recording fake sender (no network, no real waiting,
no ~/.sr2 live state). The existing single-platform tests
(test_reminders_zulip.py, test_reminders_discord.py) cover the
platform-agnostic default; these cover the isolation.
"""
from __future__ import annotations

import asyncio
import inspect
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from sr2_spectre.reminders import DeliveryLoop, Reminder, ReminderStore

UTC = ZoneInfo("UTC")

# Fixed "now" so no test ever waits on the wall clock.
NOW = datetime(2026, 10, 15, 22, 0, 0, tzinfo=UTC)
TZ = ZoneInfo("America/Phoenix")


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
def store(tmp_path: Path) -> ReminderStore:
    """One store shared by both platforms' loops (the bug's precondition)."""
    return ReminderStore(tmp_path / "reminders.sqlite")


def _seed(
    store: ReminderStore,
    platform: str,
    text: str,
    due_at: datetime,
    status: str = "pending",
    attempts: int = 0,
) -> str:
    """Insert one row directly and return its ID.

    ``create`` always inserts ``pending`` (spc-144); ``status``/``attempts``
    are applied by a follow-up UPDATE so a test can seed a row mid-send
    the way a crash would leave it.
    """
    reminder_id = store.create(
        Reminder(
            id="r0",
            agent="miranda",
            platform=platform,
            requester_id="8",
            requester_mention="@8",
            origin_channel_id="42",
            dest_channel_id="42",
            dest_channel_name="sr2-spectre",
            dest_topic="reminders",
            text=text,
            due_at=due_at,
            created_at=NOW,
            delivered_at=None,
            status="pending",
            attempts=0,
            last_error=None,
        )
    )
    if status != "pending":
        with store._connect() as conn:
            conn.execute(
                "UPDATE reminders SET status = ?, attempts = ? WHERE id = ?",
                (status, attempts, reminder_id),
            )
    return reminder_id


def _loop(
    store: ReminderStore,
    sender: FakeSender,
    platform: str,
    now: datetime = NOW,
    poll_seconds: float = 0.01,
) -> DeliveryLoop:
    """A loop serving exactly one platform."""
    return DeliveryLoop(
        store,
        send=sender.send,
        platform=platform,
        poll_seconds=poll_seconds,
        tz=TZ,
        agent_name="miranda",
        clock=_clock_at(now),
    )


async def _settle(loop: DeliveryLoop, polls: int = 3) -> None:
    """Let the loop run a few polls (injected clock: no real waiting)."""
    for _ in range(polls):
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# AC1 — DeliveryLoop takes the platform; the production interfaces pass it
# ---------------------------------------------------------------------------


def test_delivery_loop_requires_platform():
    """DeliveryLoop's signature takes the platform it serves."""
    params = inspect.signature(DeliveryLoop.__init__).parameters
    assert "platform" in params
    # Keyword-only, like every non-store setting on the constructor.
    assert params["platform"].kind is inspect.Parameter.KEYWORD_ONLY


def test_interfaces_build_their_loop_with_their_platform():
    """Each interface's DeliveryLoop construction names its own platform.

    Reading the call site (the interfaces need a live bot to run) is the
    cheapest honest check that the platform shipped is the one written on
    reminder creation ("zulip" / "discord").
    """
    import sr2_spectre.interfaces.discord.interface as discord_mod
    import sr2_spectre.interfaces.zulip.interface as zulip_mod

    def source(mod) -> str:
        return Path(inspect.getsourcefile(mod)).read_text(encoding="utf-8")

    zulip_src = source(zulip_mod)
    discord_src = source(discord_mod)

    def loop_call(src: str) -> str:
        # The DeliveryLoop(...) construction spanning the kwargs.
        match = re.search(r"DeliveryLoop\((?:.|\n)*?\)\n", src)
        assert match is not None, "DeliveryLoop construction vanished"
        return match.group(0)

    assert 'platform="zulip"' in loop_call(zulip_src)
    assert 'platform="discord"' in loop_call(discord_src)


def test_platform_defaults_keep_existing_call_shapes_working():
    """AC5: an unscoped store call and an unplatformed loop still work —
    the existing single-platform tests must pass unmodified. A fresh
    temp-file store proves it without the shared fixture."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ReminderStore(Path(tmp) / "reminders.sqlite")
        _seed(store, "zulip", "legacy due", NOW - timedelta(minutes=1))

        # Store API: no platform → every platform, as before.
        assert [r.text for r in store.due(now=NOW)] == ["legacy due"]
        assert store.claim("r1") is True
        assert store.reset_sending_to_pending() == 1

    async def send(channel_id, topic, content):
        pass

    loop = DeliveryLoop(
        store,
        send=send,
        poll_seconds=0.01,
        tz=TZ,
        agent_name="miranda",
        clock=_clock_at(NOW),
    )
    assert loop._store is store  # constructed; defaults accepted


# ---------------------------------------------------------------------------
# AC2 — a loop delivers only its platform's rows
# ---------------------------------------------------------------------------


async def test_loop_delivers_only_its_platform(store):
    zulip_id = _seed(store, "zulip", "zulip due", NOW - timedelta(minutes=2))
    discord_id = _seed(
        store, "discord", "discord due", NOW - timedelta(minutes=1)
    )
    sender = FakeSender()

    loop = _loop(store, sender, "zulip")
    await loop.start()
    await _settle(loop)
    await loop.stop()

    # Only the zulip row was sent.
    assert [s[2] for s in sender.sends] == ["@8 Reminder: zulip due"]
    assert store.get(zulip_id).status == "delivered"
    # The discord row stays pending with attempts untouched.
    row = store.get(discord_id)
    assert row.status == "pending"
    assert row.attempts == 0


async def test_other_platform_loop_gets_the_other_rows(store):
    """The mirror image: a discord loop over the same shared store."""
    zulip_id = _seed(store, "zulip", "zulip due", NOW - timedelta(minutes=2))
    discord_id = _seed(
        store, "discord", "discord due", NOW - timedelta(minutes=1)
    )
    sender = FakeSender()

    loop = _loop(store, sender, "discord")
    await loop.start()
    await _settle(loop)
    await loop.stop()

    assert [s[2] for s in sender.sends] == ["@8 Reminder: discord due"]
    assert store.get(discord_id).status == "delivered"
    row = store.get(zulip_id)
    assert row.status == "pending"
    assert row.attempts == 0


async def test_two_loops_one_store_deliver_their_own_rows_each(store):
    """Both loops running at once over one store: each delivers exactly
    its platform's rows and the two never cross."""
    zulip_id = _seed(store, "zulip", "zulip due", NOW - timedelta(minutes=2))
    discord_id = _seed(
        store, "discord", "discord due", NOW - timedelta(minutes=1)
    )
    zulip_sender = FakeSender()
    discord_sender = FakeSender()

    zulip_loop = _loop(store, zulip_sender, "zulip")
    discord_loop = _loop(store, discord_sender, "discord")
    await zulip_loop.start()
    await discord_loop.start()
    await _settle(zulip_loop, polls=5)
    await zulip_loop.stop()
    await discord_loop.stop()

    assert [s[2] for s in zulip_sender.sends] == ["@8 Reminder: zulip due"]
    assert [s[2] for s in discord_sender.sends] == ["@8 Reminder: discord due"]
    assert store.get(zulip_id).status == "delivered"
    assert store.get(discord_id).status == "delivered"


# ---------------------------------------------------------------------------
# AC3 — startup recovery only resets its own platform's sending rows
# ---------------------------------------------------------------------------


async def test_start_recovery_only_touches_own_platform(store):
    zulip_sending = _seed(
        store, "zulip", "zulip mid-send", NOW - timedelta(minutes=5),
        status="sending", attempts=1,
    )
    discord_sending = _seed(
        store, "discord", "discord mid-send", NOW - timedelta(minutes=4),
        status="sending", attempts=1,
    )
    sender = FakeSender()

    loop = _loop(store, sender, "zulip")
    await loop.start()
    await _settle(loop)
    await loop.stop()

    # Own platform: recovered, then delivered by this loop.
    assert store.get(zulip_sending).status == "delivered"
    assert [s[2] for s in sender.sends] == ["@8 Reminder: zulip mid-send"]
    # Other platform: untouched — still sending, nothing sent.
    row = store.get(discord_sending)
    assert row.status == "sending"
    assert row.attempts == 1


def test_reset_sending_scoped_by_platform(store):
    zulip_id = _seed(store, "zulip", "z", NOW, status="sending")
    discord_id = _seed(store, "discord", "d", NOW, status="sending")

    assert store.reset_sending_to_pending(platform="zulip") == 1
    assert store.get(zulip_id).status == "pending"
    assert store.get(discord_id).status == "sending"

    assert store.reset_sending_to_pending(platform="discord") == 1
    assert store.get(discord_id).status == "pending"

    # Unscoped still resets everything (AC5 default).
    store.claim(zulip_id)
    store.claim(discord_id)
    assert store.reset_sending_to_pending() == 2


# ---------------------------------------------------------------------------
# AC4 — claim cannot move another platform's row to sending
# ---------------------------------------------------------------------------


def test_claim_scoped_by_platform(store):
    zulip_id = _seed(store, "zulip", "z", NOW)
    discord_id = _seed(store, "discord", "d", NOW)

    # Claiming the discord row as zulip loses; the row stays pending.
    assert store.claim(discord_id, platform="zulip") is False
    assert store.get(discord_id).status == "pending"
    # Claiming it as its own platform wins.
    assert store.claim(discord_id, platform="discord") is True
    assert store.get(discord_id).status == "sending"
    # The zulip row was never touched.
    assert store.get(zulip_id).status == "pending"


def test_due_scoped_by_platform(store):
    _seed(store, "zulip", "z due", NOW - timedelta(minutes=2))
    _seed(store, "discord", "d due", NOW - timedelta(minutes=1))

    zulip_due = store.due(now=NOW, platform="zulip")
    discord_due = store.due(now=NOW, platform="discord")
    assert [r.text for r in zulip_due] == ["z due"]
    assert [r.text for r in discord_due] == ["d due"]


async def test_claim_async_scoped_by_platform(store):
    discord_id = _seed(store, "discord", "d", NOW)
    assert await store.claim_async(discord_id, platform="zulip") is False
    assert store.get(discord_id).status == "pending"


async def test_zulip_loop_cannot_even_see_the_discord_row(store):
    """End to end: a zulip loop's due selection skips the discord row, so
    no claim is attempted and the row stays untouched pending."""
    discord_id = _seed(
        store, "discord", "discord due", NOW - timedelta(minutes=1)
    )
    sender = FakeSender()

    loop = _loop(store, sender, "zulip")
    await loop.start()
    await _settle(loop)
    await loop.stop()

    assert sender.sends == []
    row = store.get(discord_id)
    assert row.status == "pending"
    assert row.attempts == 0
