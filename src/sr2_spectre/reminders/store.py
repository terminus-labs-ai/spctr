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
    # spc-157: FR18 retries schedule from this column, never from due_at.
    # It must stay out of ``_INSERT_COLUMNS`` (fresh rows start NULL) and
    # out of the Reminder dataclass (rows map back by name; the field
    # would otherwise break every frozen positional/keyword construction).
    "next_attempt_at",
)

# Columns ``create`` inserts: every column above except the two that
# start NULL. ``status`` and ``attempts`` carry the fixed 'pending'/0 of
# the pre-spc-144 INSERT; ``last_error`` starts NULL and
# ``next_attempt_at`` starts NULL (a first attempt is due at ``due_at``).
_INSERT_COLUMNS = tuple(
    c for c in _COLUMNS if c not in ("delivered_at", "next_attempt_at")
)

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
  last_error TEXT NULL,
  next_attempt_at TEXT NULL
);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# spc-157: retries are scheduled from this column instead of moving
# ``due_at``. A store file created before it existed (the spc-139 schema)
# gains the column additively on connect; existing rows keep NULL and
# stay deliverable (their next attempt is due at ``due_at``).
_RETRY_COLUMN = "next_attempt_at"


def _ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Reminder:
    """One stored reminder row.

    ``next_attempt_at`` is the FR18 retry schedule and stays None until a
    send fails; ``due_at`` is the time the user asked for and never moves
    (spc-157). Declared positionally last with a default so every other
    field keeps its old position for positional construction.
    """

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
    # FR18 retry schedule (spc-157). Declared last, with a default, so a
    # caller building a fresh row never names it and old call sites keep
    # working.
    next_attempt_at: datetime | None = None


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
        self._migrate(conn)
        conn.commit()
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add missing columns to a store file from an earlier schema.

        ``_SCHEMA`` is a no-op on an existing table, so a file written
        before spc-157 needs the retry column added in place; the change
        is additive (NULL for every existing row) and the rows stay
        deliverable.
        """
        existing = {
            row["name"] for row in conn.execute("PRAGMA table_info(reminders)")
        }
        if _RETRY_COLUMN not in existing:
            conn.execute(
                "ALTER TABLE reminders ADD COLUMN next_attempt_at TEXT NULL"
            )

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

    def due(
        self,
        now: datetime,
        statuses: Iterable[str] = ("pending",),
        platform: str | None = None,
    ) -> list[Reminder]:
        """Rows whose next attempt is at or before ``now``, in due order.

        A row that has never been retried attempts at its ``due_at``; a
        retried row attempts at ``next_attempt_at`` while ``due_at``
        stays the time the user asked for (spc-157). Ordering and the
        ``ORDER BY due_at, id`` the FR19 list relies on are unchanged.

        ``platform`` (spc-176): when set, only rows of that platform are
        due — one agent's Discord loop must never see its Zulip rows.
        ``None`` (the default) keeps the old all-platforms behavior.
        """
        statuses = tuple(statuses)
        placeholders = ", ".join("?" for _ in statuses)
        # spc-176: an empty clause keeps the old all-platforms SQL exactly.
        platform_clause = "AND platform = ?" if platform is not None else ""
        params = [*(statuses), _ts(now)]
        if platform is not None:
            params.append(platform)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT {", ".join(_COLUMNS)}
                    FROM reminders
                    WHERE status IN ({placeholders})
                      AND COALESCE(next_attempt_at, due_at) <= ?
                      {platform_clause}
                    ORDER BY due_at, id""",
                params,
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def claim(self, reminder_id: str, platform: str | None = None) -> bool:
        """Atomically move one row ``pending`` → ``sending`` (FR17 claim).

        Returns True only if this call won the claim; a row that is not
        ``pending`` (already claimed/delivered/...) is left untouched.
        With ``platform`` set (spc-176) a row of another platform is not
        claimable at all — the claim doubles as the cross-loop guard.
        """
        platform_clause = " AND platform = ?" if platform is not None else ""
        params = (reminder_id, platform) if platform is not None else (reminder_id,)
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET status = 'sending' "
                "WHERE id = ? AND status = 'pending'" + platform_clause,
                params,
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

    def due_with_retry(
        self,
        reminder_id: str,
        error: str,
        next_attempt: datetime,
        max_attempts: int = 3,
    ) -> str:
        """One failed send with the bounded retry policy (FR18, spc-140;
        spc-157 keeps ``due_at``).

        Until the attempt budget runs out the row returns to ``pending``
        with its next attempt scheduled at ``next_attempt`` — ``due_at``
        is left as the time the user asked for, so the FR15 late suffix
        and the FR19 list keep the original due time. Once this failure
        reaches ``max_attempts`` it is terminal ``failed``. Returns the
        new status.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE reminders SET attempts = attempts + 1, last_error = ? "
                "WHERE id = ?",
                (error, reminder_id),
            )
            row = conn.execute(
                "SELECT attempts FROM reminders WHERE id = ?", (reminder_id,)
            ).fetchone()
            attempts = row["attempts"] if row else max_attempts
            if attempts >= max_attempts:
                conn.execute(
                    "UPDATE reminders SET status = 'failed' WHERE id = ?",
                    (reminder_id,),
                )
                return "failed"
            conn.execute(
                "UPDATE reminders SET status = 'pending', next_attempt_at = ? "
                "WHERE id = ?",
                (_ts(next_attempt), reminder_id),
            )
            return "pending"

    def cancel(self, reminder_id: str) -> bool:
        """Atomically move one row ``pending`` → ``cancelled`` (FR20).

        Returns True only if this call won before a sender claimed it;
        any other status (sending, delivered, failed, cancelled) is left
        untouched.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET status = 'cancelled' "
                "WHERE id = ? AND status = 'pending'",
                (reminder_id,),
            )
            return cursor.rowcount == 1

    def reset_sending_to_pending(self, platform: str | None = None) -> int:
        """On start, return every ``sending`` row to ``pending`` (FR17/FR22).

        ``platform`` (spc-176): when set, only that platform's rows are
        reset — one interface's startup must not reset the other's
        mid-send row and cause a double delivery. ``None`` (the default)
        keeps the old all-platforms behavior.
        """
        platform_clause = " AND platform = ?" if platform is not None else ""
        params = (platform,) if platform is not None else ()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET status = 'pending' WHERE status = 'sending'"
                + platform_clause,
                params,
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
        for ts_field in ("due_at", "next_attempt_at", "created_at", "delivered_at"):
            raw = values[ts_field]
            values[ts_field] = None if raw is None else datetime.fromisoformat(raw)
        return Reminder(**values)

    # ---- async facades (run the sync core off the event loop) --------------

    async def create_async(self, row: Reminder) -> str:
        return await asyncio.to_thread(self.create, row)

    async def due_async(
        self,
        now: datetime,
        statuses: Iterable[str] = ("pending",),
        platform: str | None = None,
    ) -> list[Reminder]:
        return await asyncio.to_thread(self.due, now, statuses, platform)

    async def claim_async(self, reminder_id: str, platform: str | None = None) -> bool:
        return await asyncio.to_thread(self.claim, reminder_id, platform)

    async def get_async(self, reminder_id: str) -> Reminder | None:
        return await asyncio.to_thread(self.get, reminder_id)

    async def mark_delivered_async(self, reminder_id: str, at: datetime) -> None:
        await asyncio.to_thread(self.mark_delivered, reminder_id, at)

    async def mark_failed_attempt_async(self, reminder_id: str, error: str) -> None:
        await asyncio.to_thread(self.mark_failed_attempt, reminder_id, error)

    async def due_with_retry_async(
        self, reminder_id: str, error: str, next_attempt: datetime,
        max_attempts: int = 3,
    ) -> str:
        return await asyncio.to_thread(
            self.due_with_retry, reminder_id, error, next_attempt, max_attempts
        )

    async def cancel_async(self, reminder_id: str) -> bool:
        return await asyncio.to_thread(self.cancel, reminder_id)

    async def reset_sending_to_pending_async(
        self, platform: str | None = None
    ) -> int:
        return await asyncio.to_thread(self.reset_sending_to_pending, platform)
