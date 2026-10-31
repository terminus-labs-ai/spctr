"""spc-157: a retry keeps the original due time (AC R1–R4).

The FR18 retry path used to move ``due_at`` to each next-attempt time,
so a reminder delivered on a late retry lost its FR15 late suffix and
``list`` showed the retry time. Retries are now scheduled from
``next_attempt_at`` and ``due_at`` never moves.

- R1: after one or two failed sends the stored ``due_at`` is still the
  original due time, and the next attempt is still made at +1 then +5
  minutes on that.
- R2: a send that succeeds on the +5 min retry posts the FR15 suffix
  with the original due local time.
- R3: ``list`` shows, and sorts by, the original due time — for retried
  pending rows and, for the 7-day window, failed rows.
- R4: a store file written with the previous schema (no next-attempt
  column) opens without error, gains the column additively, and its rows
  stay deliverable.

Injected clock and recording fake sender as in the existing reminder
tests: no network, no real waiting. Every store is an explicit path
under tmp_path — never ``~/.sr2`` live state.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
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

# The schema as of base commit 2c741b4 (spc-139/spc-140, before spc-157):
# sixteen columns, no next-attempt column. Verbatim, so R4 exercises the
# real previous schema rather than a summary of it.
OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
  id TEXT PRIMARY KEY,
  agent TEXT NOT NULL,
  platform TEXT NOT NULL,
  requester_id TEXT NOT NULL,
  requester_mention TEXT NOT NULL,
  origin_channel_id TEXT NOT NULL,
  dest_channel_id TEXT NOT NULL,
  dest_channel_name TEXT NOT NULL,
  dest_topic TEXT NOT NULL,
  text TEXT NOT NULL,
  due_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  delivered_at TEXT NULL,
  status TEXT NOT NULL CHECK (status IN
    ('pending', 'sending', 'delivered', 'failed', 'cancelled')),
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT NULL
);
"""


def _origin(**overrides) -> ReminderOrigin:
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


def _tool(store: ReminderStore, now: datetime = NOW) -> ReminderTool:
    return ReminderTool(
        store=store,
        timezone="America/Phoenix",
        agent_name="miranda",
        origin_getter=lambda: _origin(),
        clock=lambda: now,
    )


def _loop(store: ReminderStore, send, now: datetime) -> DeliveryLoop:
    return DeliveryLoop(
        store,
        send=send,
        poll_seconds=0.01,
        tz=TZ,
        agent_name="miranda",
        clock=lambda: now,
    )


def _seed(store: ReminderStore, text: str, due_at: datetime) -> str:
    return store.create(
        Reminder(
            id="r0",
            agent="miranda",
            platform="zulip",
            requester_id="8",
            requester_mention="@8",
            origin_channel_id="42",
            dest_channel_id="42",
            dest_channel_name="sr2-spectre",
            dest_topic="reminders",
            text=text,
            due_at=due_at,
            created_at=due_at - timedelta(minutes=5),
            delivered_at=None,
            status="pending",
            attempts=0,
            last_error=None,
        )
    )


async def _poll(store: ReminderStore, send, moments) -> None:
    """Run one short-lived loop per scheduled point, as the AC9 tests do
    (the clock is injected — none of this is real waiting)."""
    for moment in moments:
        loop = _loop(store, send, now=moment)
        await loop.start()
        for _ in range(3):
            await asyncio.sleep(0.02)
        await loop.stop()


def _columns(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [r[1] for r in conn.execute("PRAGMA table_info(reminders)")]
    finally:
        conn.close()


async def _always_fails_async(channel_id: int, topic: str, content: str) -> None:
    raise RuntimeError("boom")


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "reminders.sqlite"


@pytest.fixture
def store(store_path: Path) -> ReminderStore:
    return ReminderStore(store_path)


# ---------------------------------------------------------------------------
# R1 — due_at keeps the original due time; FR18 timing unchanged
# ---------------------------------------------------------------------------


async def test_r1_due_at_unchanged_after_two_failed_sends(store):
    """R1: after one and after two failed sends the stored due_at still
    equals the original due time, and the next attempt is scheduled at
    the FR18 times (+1 min, then +5 min on that = due+6 min)."""
    _seed(store, "flaky", NOW)
    send = _always_fails_async

    await _poll(store, send, [NOW + timedelta(seconds=2)])
    row = store.get("r1")
    assert row.attempts == 1
    assert row.status == "pending"
    assert row.due_at == NOW
    assert row.next_attempt_at == NOW + timedelta(minutes=1)

    await _poll(store, send, [NOW + timedelta(seconds=62)])
    row = store.get("r1")
    assert row.attempts == 2
    assert row.due_at == NOW
    assert row.next_attempt_at == NOW + timedelta(minutes=6)


async def test_r1_third_failure_fails_without_touching_due_at(store):
    """R1: the third failure marks the row failed and still leaves
    due_at at the original due time (FR19 dates the failed row by it)."""
    _seed(store, "flaky", NOW)
    await _poll(
        store,
        _always_fails_async,
        [NOW, NOW + timedelta(seconds=62), NOW + timedelta(minutes=6, seconds=2)],
    )

    row = store.get("r1")
    assert row.status == "failed"
    assert row.attempts == 3
    assert row.due_at == NOW


async def test_r1_no_attempt_before_the_scheduled_retry(store):
    """R1: with due_at left at the original time, a poll between due and
    +1 min must not attempt again — the retry gate is next_attempt_at."""
    _seed(store, "flaky", NOW)
    calls: list[str] = []

    async def counting_fail(channel_id, topic, content):
        calls.append(content)
        raise RuntimeError("boom")

    await _poll(store, counting_fail, [NOW + timedelta(seconds=2)])
    assert len(calls) == 1
    await _poll(store, counting_fail, [NOW + timedelta(seconds=30)])
    assert len(calls) == 1  # before +1 min: the row is not due again
    await _poll(store, counting_fail, [NOW + timedelta(seconds=62)])
    assert len(calls) == 2
    await _poll(store, counting_fail, [NOW + timedelta(minutes=5)])
    assert len(calls) == 2  # before +6 min: still not due
    await _poll(store, counting_fail, [NOW + timedelta(minutes=6, seconds=2)])
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# R2 — the FR15 late suffix names the original due local time
# ---------------------------------------------------------------------------


async def test_r2_late_retry_posts_original_due_time_with_late_suffix(store):
    """R2: a reminder whose send succeeds on the +5 min retry (more than
    5 min after its original due time) is posted with the FR15 suffix
    showing the original due local time."""
    _seed(store, "pay the toll", NOW)
    attempts = {"n": 0}

    async def fails_twice(channel_id, topic, content):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise RuntimeError("boom")
        sends.append((int(channel_id), topic, content))

    sends: list[tuple[int, str, str]] = []
    await _poll(
        store,
        fails_twice,
        [NOW, NOW + timedelta(seconds=62), NOW + timedelta(minutes=6, seconds=2)],
    )

    assert len(sends) == 1
    content = sends[0][2]
    assert content.startswith("@8 Reminder: pay the toll")
    assert "sent late" in content
    assert "due Thu 2026-10-15 15:00 MST" in content
    assert store.get("r1").status == "delivered"


async def test_r2_early_retry_carries_no_late_suffix(store):
    """R2 (negative half): a retry that lands within 5 min of the
    original due time carries no late suffix."""
    _seed(store, "pay the toll", NOW)
    attempts = {"n": 0}
    sends: list[tuple[int, str, str]] = []

    async def fails_once(channel_id, topic, content):
        attempts["n"] += 1
        if attempts["n"] <= 1:
            raise RuntimeError("boom")
        sends.append((int(channel_id), topic, content))

    await _poll(store, fails_once, [NOW, NOW + timedelta(seconds=62)])

    assert len(sends) == 1
    assert sends[0][2] == "@8 Reminder: pay the toll"
    assert store.get("r1").status == "delivered"


# ---------------------------------------------------------------------------
# R3 — list shows and sorts by the original due time
# ---------------------------------------------------------------------------


async def test_r3_list_shows_original_due_time_of_a_retried_row(store):
    """R3 (FR19): a pending reminder that has been retried lists the
    original due local time, not the retry time, and rows sort by that
    original due time."""
    _seed(store, "later one", NOW + timedelta(minutes=40))  # r1
    _seed(store, "retried one", NOW + timedelta(minutes=10))  # r2
    await _poll(store, _always_fails_async, [NOW + timedelta(minutes=10, seconds=2)])

    retried = store.get("r2")
    assert retried.attempts == 1
    assert retried.due_at == NOW + timedelta(minutes=10)
    assert retried.next_attempt_at == NOW + timedelta(minutes=11)

    result = await _tool(store, now=NOW + timedelta(minutes=10, seconds=5))(
        action="list"
    )
    lines = result.splitlines()
    assert [line.split()[0] for line in lines] == ["r2", "r1"]
    assert lines[0] == (
        "r2 Thu 2026-10-15 15:10 MST #sr2-spectre pending retried one"
    )


async def test_r3_failed_row_window_uses_original_due_time(store):
    """R3 (FR19): the 7-day failed window is dated by the original due
    time — a row that failed today lists, and a row whose original due
    time is 8 days ago does not."""
    _seed(store, "old but failing", NOW - timedelta(days=8))  # r1
    _seed(store, "failed today", NOW - timedelta(minutes=20))  # r2
    # One poll now hits both rows; r2 fails its third attempt this poll,
    # r1 too (both already carry 0 attempts; run three polls spaced to
    # the retry schedule so both end terminal 'failed').
    await _poll(
        store,
        _always_fails_async,
        [
            NOW - timedelta(minutes=20),
            NOW - timedelta(minutes=19),
            NOW - timedelta(minutes=14),
        ],
    )

    assert store.get("r1").status == "failed"
    assert store.get("r2").status == "failed"
    assert store.get("r2").due_at == NOW - timedelta(minutes=20)

    result = await _tool(store, now=NOW)(action="list")
    assert "failed today" in result
    assert "old but failing" not in result


# ---------------------------------------------------------------------------
# R4 — an old-schema store file opens, gains the column, stays deliverable
# ---------------------------------------------------------------------------


def _insert_legacy_row(path: Path, reminder_id: str, text: str, due_at: datetime):
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        "INSERT INTO reminders (id, agent, platform, requester_id, "
        "requester_mention, origin_channel_id, dest_channel_id, "
        "dest_channel_name, dest_topic, text, due_at, created_at, status, "
        "attempts) VALUES (?, 'miranda', 'zulip', '8', '@8', '42', '42', "
        "'sr2-spectre', 'reminders', ?, ?, ?, 'pending', 0)",
        (
            reminder_id,
            text,
            due_at.astimezone(UTC).isoformat(),
            (due_at - timedelta(minutes=10)).astimezone(UTC).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def test_r4_old_schema_file_opens_and_gains_the_column(store_path: Path):
    """R4: a file created with the base commit's CREATE TABLE (no
    next-attempt column) opens with the new store without error, gains
    the column additively, and keeps every earlier column and row."""
    _insert_legacy_row(store_path, "r1", "legacy row", NOW + timedelta(minutes=30))
    before = _columns(store_path)
    assert "next_attempt_at" not in before

    store = ReminderStore(store_path)
    after = _columns(store_path)
    assert after[: len(before)] == before  # additive: old columns keep order
    assert after[-1] == "next_attempt_at"

    row = store.get("r1")
    assert row.text == "legacy row"
    assert row.due_at == NOW + timedelta(minutes=30)
    assert row.next_attempt_at is None
    # A legacy row is due on its own due time.
    assert [r.id for r in store.due(now=NOW + timedelta(minutes=31))] == ["r1"]
    assert store.due(now=NOW + timedelta(minutes=29)) == []


async def test_r4_legacy_row_delivers_through_the_loop(store_path: Path):
    """R4: a row written under the old schema is delivered by the new
    loop with its original due time intact."""
    _insert_legacy_row(store_path, "r1", "legacy deliver", NOW - timedelta(minutes=2))
    store = ReminderStore(store_path)
    sends: list[tuple[int, str, str]] = []

    async def record(channel_id, topic, content):
        sends.append((int(channel_id), topic, content))

    await _poll(store, record, [NOW])

    assert len(sends) == 1
    assert "legacy deliver" in sends[0][2]
    row = store.get("r1")
    assert row.status == "delivered"
    assert row.due_at == NOW - timedelta(minutes=2)


async def test_r4_legacy_row_retries_without_losing_its_due_time(store_path: Path):
    """R4: a legacy row that fails follows the new retry schedule while
    due_at keeps the value the old schema stored."""
    _insert_legacy_row(store_path, "r1", "legacy flaky", NOW)
    store = ReminderStore(store_path)
    await _poll(store, _always_fails_async, [NOW + timedelta(seconds=2)])

    row = store.get("r1")
    assert row.attempts == 1
    assert row.due_at == NOW
    assert row.next_attempt_at == NOW + timedelta(minutes=1)
