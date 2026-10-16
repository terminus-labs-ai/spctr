"""Reminders store column dedup (spc-144).

Direct-profile unit tests for the spc-144 AC subset (R1, R2). The store
refactor names the reminder column set once and maps rows to ``Reminder``
by column name; behavior must not change.

- R1: one module-level column definition; ``due()``, ``get()`` and
  ``create()`` derive from it; row mapping is by name, not index.
- R2: a full round trip (create → get → due) preserves every ``Reminder``
  field, including create()'s fixed 'pending'/0 status and the timestamp
  parsing (due_at, created_at, delivered_at).

Everything is a temp-file SQLite database under tmp_path — no network,
no real waiting, no ~/.sr2 live state.
"""
from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sr2_spectre import reminders
from sr2_spectre.reminders.store import _COLUMNS, Reminder, ReminderStore

UTC = timezone.utc

NOW = datetime(2026, 10, 15, 22, 0, 0, tzinfo=UTC)


def _row(**overrides) -> Reminder:
    """A fully populated pending reminder row (all fifteen fields)."""
    base = dict(
        id="r1",
        agent="miranda",
        platform="zulip",
        requester_id="8",
        requester_mention="@8",
        origin_channel_id="42",
        dest_channel_id="42",
        dest_channel_name="sr2-spectre",
        dest_topic="reminders",
        text="look into https://example.com/x",
        due_at=NOW + timedelta(minutes=60),
        created_at=NOW,
        delivered_at=None,
        status="pending",
        attempts=0,
        last_error=None,
    )
    base.update(overrides)
    return Reminder(**base)


@pytest.fixture
def store(tmp_path: Path) -> ReminderStore:
    return ReminderStore(tmp_path / "reminders.sqlite")


# ---------------------------------------------------------------------------
# R1 — the column names appear once; due()/get()/create() derive from it
# ---------------------------------------------------------------------------


def test_r1_single_column_definition():
    """R1: store.py names the column set once as a module-level tuple of
    the same names as the Reminder fields."""
    store_mod = reminders.store
    assert isinstance(store_mod._COLUMNS, tuple)
    field_names = tuple(f.name for f in dataclasses.fields(Reminder))
    assert store_mod._COLUMNS == field_names


def test_r1_queries_derive_from_columns(store):
    """R1: the SQL due(), get() and create() actually run is built from
    ``_COLUMNS``. The store hands its connections to ``with ...``;
    ``Connection.__exit__`` runs COMMIT/ROLLBACK on the context object
    only (no statement), so a bare subclass that records every execute
    observes the real statements without breaking transactions."""
    record_sql: list[str] = []

    class SpyConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            record_sql.append(sql)
            return super().execute(sql, *args, **kwargs)

    real_connect = sqlite3.connect

    def spy_connect(path, *args, **kwargs):
        kwargs["factory"] = SpyConnection
        return real_connect(path, *args, **kwargs)

    sqlite3.connect = spy_connect
    try:
        store.create(_row())
        store.get("r1")
        store.due(now=NOW + timedelta(hours=1))
    finally:
        sqlite3.connect = real_connect

    select_sqls = [s for s in record_sql if s.lstrip().startswith("SELECT id,")]
    assert len(select_sqls) == 2  # get() and due()
    joined = ", ".join(_COLUMNS)
    for sql in select_sqls:
        assert joined in sql

    insert_sqls = [s for s in record_sql if s.lstrip().startswith("INSERT INTO reminders")]
    assert len(insert_sqls) == 1
    # create() lists every column except delivered_at (which stays NULL).
    inside = insert_sqls[0].split("INSERT INTO reminders", 1)[1]
    inside = inside.split("VALUES")[0]
    listed = [c.strip() for c in inside.strip().strip("()").split(",")]
    assert listed == [c for c in _COLUMNS if c not in ("delivered_at", "next_attempt_at")]
    # One bound placeholder per listed column.
    values_part = insert_sqls[0].split("VALUES", 1)[1]
    assert values_part.count("?") == len(_COLUMNS) - 2


# ---------------------------------------------------------------------------
# R2 — round trip create → get → due preserves every field
# ---------------------------------------------------------------------------


def test_r2_round_trip_preserves_every_field(store):
    """R2: a reminder written by create() reads back through get() and
    due() with every Reminder field preserved: timestamps parsed
    (due_at, created_at, delivered_at), status/attempts the fixed
    'pending'/0 of create()."""
    created_id = store.create(_row())
    assert created_id == "r1"

    future = NOW + timedelta(hours=1)
    by_id = store.get(created_id)
    in_due = store.due(now=future)

    assert [r.id for r in in_due] == [created_id]
    for got in (by_id, in_due[0]):
        # Field-by-field equality over all sixteen dataclass fields.
        assert dataclasses.asdict(got) == dataclasses.asdict(_row())
        assert isinstance(got.due_at, datetime)
        assert isinstance(got.created_at, datetime)
        assert got.due_at == NOW + timedelta(minutes=60)
        assert got.created_at == NOW
        assert got.delivered_at is None
        assert got.status == "pending"
        assert got.attempts == 0
        assert got.last_error is None


def test_r2_pending_statuses_default(store):
    """R2: due() with its default statuses returns pending rows; a row
    claimed to 'sending' is not due until 'sending' is asked for."""
    store.create(_row())
    assert store.claim("r1") is True
    future = NOW + timedelta(hours=1)
    assert store.due(now=future) == []
    assert [r.id for r in store.due(now=future, statuses=("sending",))] == ["r1"]


def test_r2_round_trip_after_delivered(store):
    """R2: delivered_at parses back as a datetime and last_error survives
    a failed attempt, mapped by column name."""
    due_at = NOW + timedelta(minutes=30)
    store.create(_row(due_at=due_at))
    store.mark_failed_attempt("r1", "boom")
    assert store.get("r1").attempts == 1
    assert store.get("r1").last_error == "boom"

    delivered_at = NOW + timedelta(minutes=31)
    store.claim("r1")
    store.mark_delivered("r1", delivered_at)

    got = store.get("r1")
    assert got.status == "delivered"
    assert got.delivered_at == delivered_at
    # Past-due delivered rows only surface when 'delivered' is requested.
    in_due = store.due(now=delivered_at, statuses=("delivered",))[0]
    assert dataclasses.asdict(in_due) == dataclasses.asdict(got)
