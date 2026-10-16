"""SQLite persistence for reminders (spc-139, FR17/NFR Persistence).

The store is the shared state between the reminder tool (inside a model
turn) and the delivery loop: both access it concurrently from the same
event loop. Every mutating operation runs as one ``sqlite3`` transaction,
so a reminder is claimed by exactly one sender and every status change is
atomic.

``sqlite3`` via ``asyncio.to_thread`` rather than ``aiosqlite``: the
known aiosqlite shutdown hang (spc-5, boot race) must not be repeated.
Each method opens a fresh short-lived connection, so there is no long-lived
connection to hang on shutdown; the loop must not delay process exit
beyond one in-flight send (NFR Shutdown).
"""
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

# The reminder column set, named once (spc-144). ``_SCHEMA``, the SELECTs
# (``due``, ``get``) and the INSERT (``create``) all derive from this
# single definition, and rows map back to ``Reminder`` by column name via
# ``sqlite3.Row`` — adding a column means editing one tuple (plus its
# CREATE TABLE clause).
_COLUMNS = (
    "id",
    "agent",
    "platform",
    "requester_id",
    "requester_mention",
    "origin_channel_id",
    "dest_channel_id",
    "dest_channel_name",
    "dest_topic",
    "text",
    "due_at",
    "created_at",
    "delivered_at",
    "status",
    "attempts",
    "last_error",
)

# Columns ``create`` inserts: the whole set except ``delivered_at``.
# ``status`` and ``attempts`` carry the fixed 'pending'/0 of the
# pre-spc-144 INSERT and ``last_error`` starts NULL.
_INSERT_COLUMNS = tuple(c for c in _COLUMNS if c != "delivered_at")

_SCHEMA = """
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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Reminder:
    """One stored reminder row."""

    id: str
    agent: str
    platform: str
    requester_id: str
    requester_mention: str
    origin_channel_id: str
    dest_channel_id: str
    dest_channel_name: str
    dest_topic: str
    text: str
    due_at: datetime
    created_at: datetime
    delivered_at: datetime | None
    status: str
    attempts: int
    last_error: str | None


class ReminderStore:
    """Synchronous SQLite store with async facades.

    Args:
        path: File path of the SQLite database; parent directories are
            created on demand. ``:memory:`` is supported for tests.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._connect()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=10.0)
        # Map rows to Reminder by column name, never by position (spc-144).
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute(_SCHEMA)
        conn.commit()
        return conn

    # ---- sync core (each call = one short-lived connection/transaction) ----

    def create(self, row: Reminder) -> str:
        """Insert a pending reminder. Returns the assigned ``r<N>`` ID.

        Raises ``ValueError`` if an ID collision would occur (should never
        happen: IDs are allocated from the max existing N).
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT id FROM reminders ORDER BY id"
            )
            next_n = 1  # first ID is r1
            for (existing,) in cursor:
                if existing.startswith("r"):
                    try:
                        next_n = max(next_n, int(existing[1:]) + 1)
                    except ValueError:
                        continue
            new_id = f"r{next_n}"
            conn.execute(
                f"""INSERT INTO reminders ({", ".join(_INSERT_COLUMNS)})
                    VALUES ({", ".join(['?'] * len(_INSERT_COLUMNS))})""",
                (
                    new_id,
                    row.agent,
                    row.platform,
                    row.requester_id,
                    row.requester_mention,
                    row.origin_channel_id,
                    row.dest_channel_id,
                    row.dest_channel_name,
                    row.dest_topic,
                    row.text,
                    _ts(row.due_at),
                    _ts(row.created_at),
                    "pending",
                    0,
                    None,
                ),
            )
        return new_id

    def due(self, now: datetime, statuses: Iterable[str] = ("pending",)) -> list[Reminder]:
        """Rows whose due time is at or before ``now``, in due order."""
        statuses = tuple(statuses)
        placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT {", ".join(_COLUMNS)}
                    FROM reminders
                    WHERE status IN ({placeholders}) AND due_at <= ?
                    ORDER BY due_at, id""",
                (*statuses, _ts(now)),
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def claim(self, reminder_id: str) -> bool:
        """Atomically move one row ``pending`` → ``sending`` (FR17 claim).

        Returns True only if this call won the claim; a row that is not
        ``pending`` (already claimed/delivered/...) is left untouched.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET status = 'sending' "
                "WHERE id = ? AND status = 'pending'",
                (reminder_id,),
            )
            return cursor.rowcount == 1

    def mark_delivered(self, reminder_id: str, at: datetime) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE reminders SET status = 'delivered', delivered_at = ? "
                "WHERE id = ?",
                (_ts(at), reminder_id),
            )

    def mark_failed_attempt(self, reminder_id: str, error: str) -> None:
        """A send failed: leave the row for the next poll (spc-139 policy).

        The row returns to ``pending`` (retried on the next poll),
        ``attempts`` increments and ``last_error`` carries the reason.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE reminders SET status = 'pending', attempts = attempts + 1, "
                "last_error = ? WHERE id = ?",
                (error, reminder_id),
            )

    def reset_sending_to_pending(self) -> int:
        """On start, return every ``sending`` row to ``pending`` (FR17/FR22)."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET status = 'pending' WHERE status = 'sending'"
            )
            return cursor.rowcount

    def get(self, reminder_id: str) -> Reminder | None:
        with self._connect() as conn:
            row = conn.execute(
                f"""SELECT {", ".join(_COLUMNS)}
                    FROM reminders WHERE id = ?""",
                (reminder_id,),
            ).fetchone()
        return None if row is None else self._row_to_reminder(row)

    def count_by_status(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM reminders GROUP BY status"
            ).fetchall()
        return {status: n for status, n in rows}

    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        """Build a ``Reminder`` from a row, keyed by column name (spc-144).

        Timestamp columns parse from their stored ISO text; every other
        column maps straight onto the field of the same name.
        """
        values = dict(row)
        for ts_field in ("due_at", "created_at", "delivered_at"):
            raw = values[ts_field]
            values[ts_field] = None if raw is None else datetime.fromisoformat(raw)
        return Reminder(**values)

    # ---- async facades (run the sync core off the event loop) --------------

    async def create_async(self, row: Reminder) -> str:
        return await asyncio.to_thread(self.create, row)

    async def due_async(self, now: datetime,
                        statuses: Iterable[str] = ("pending",)) -> list[Reminder]:
        return await asyncio.to_thread(self.due, now, statuses)

    async def claim_async(self, reminder_id: str) -> bool:
        return await asyncio.to_thread(self.claim, reminder_id)

    async def mark_delivered_async(self, reminder_id: str, at: datetime) -> None:
        await asyncio.to_thread(self.mark_delivered, reminder_id, at)

    async def mark_failed_attempt_async(self, reminder_id: str, error: str) -> None:
        await asyncio.to_thread(self.mark_failed_attempt, reminder_id, error)

    async def reset_sending_to_pending_async(self) -> int:
        return await asyncio.to_thread(self.reset_sending_to_pending)
