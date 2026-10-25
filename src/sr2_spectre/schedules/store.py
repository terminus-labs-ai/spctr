"""SQLite persistence for schedules (spc-185, FR23–FR35).

Mirrors the reminder store pattern: synchronous SQLite with async facades,
one short-lived connection per operation, platform-scoped claim.
"""
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_COLUMNS = (
    "id",
    "agent",
    "platform",
    "requester_id",
    "requester_mention",
    "dest_channel_id",
    "dest_channel_name",
    "label",
    "cron",
    "payload_type",
    "payload",
    "on_miss",
    "next_run_at",
    "last_run_at",
    "created_at",
    "status",
    "last_status",
    "last_error",
)

_TS_BINDINGS = frozenset({"next_run_at", "last_run_at", "created_at"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
  id TEXT PRIMARY KEY,
  agent TEXT NOT NULL,
  platform TEXT NOT NULL,
  requester_id TEXT NOT NULL,
  requester_mention TEXT NOT NULL,
  dest_channel_id TEXT NOT NULL,
  dest_channel_name TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  cron TEXT NOT NULL,
  payload_type TEXT NOT NULL CHECK (payload_type IN ('command', 'agent')),
  payload TEXT NOT NULL,
  on_miss TEXT NOT NULL DEFAULT 'run_late' CHECK (on_miss IN ('run_late', 'skip')),
  next_run_at TEXT NOT NULL,
  last_run_at TEXT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'cancelled')),
  last_status TEXT NULL,
  last_error TEXT NULL
);
"""


def _ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Schedule:
    """One stored schedule row."""

    id: str
    agent: str
    platform: str
    requester_id: str
    requester_mention: str
    dest_channel_id: str
    dest_channel_name: str
    label: str
    cron: str
    payload_type: str
    payload: str
    on_miss: str
    next_run_at: datetime
    last_run_at: datetime | None
    created_at: datetime
    status: str
    last_status: str | None
    last_error: str | None


class ScheduleStore:
    """Synchronous SQLite store with async facades.

    Args:
        path: File path of the SQLite database; parent directories are
            created on demand. ``:memory:`` is supported for tests.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute(_SCHEMA)
        return conn

    # ---- sync core ----

    def create(self, row: Schedule) -> str:
        """Insert an active schedule. Returns the assigned ``s<N>`` ID."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT id FROM schedules ORDER BY id")
            next_n = 1
            for (existing,) in cursor:
                if existing.startswith("s"):
                    try:
                        next_n = max(next_n, int(existing[1:]) + 1)
                    except ValueError:
                        continue
            new_id = f"s{next_n}"
            values = []
            for col in _COLUMNS:
                if col == "id":
                    values.append(new_id)
                elif col in _TS_BINDINGS:
                    raw = getattr(row, col)
                    values.append(None if raw is None else _ts(raw))
                else:
                    values.append(getattr(row, col))
            conn.execute(
                f"""INSERT INTO schedules ({", ".join(_COLUMNS)})
                    VALUES ({", ".join(['?'] * len(_COLUMNS))})""",
                values,
            )
        return new_id

    def due(
        self,
        now: datetime,
        platform: str | None = None,
    ) -> list[Schedule]:
        """Active schedules whose next_run_at is at or before *now*."""
        platform_clause = "AND platform = ?" if platform is not None else ""
        params: list = [_ts(now)]
        if platform is not None:
            params.append(platform)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT {", ".join(_COLUMNS)}
                    FROM schedules
                    WHERE status = 'active'
                      AND next_run_at <= ?
                      {platform_clause}
                    ORDER BY next_run_at, id""",
                params,
            ).fetchall()
        return [self._row_to_schedule(r) for r in rows]

    def claim(self, schedule_id: str, platform: str | None = None) -> bool:
        """Atomically mark one row as running (platform-scoped).

        Returns True only if this call won the claim. The claim is
        implemented by advancing next_run_at to a sentinel far future
        value atomically; if the row was already claimed (next_run_at
        changed), rowcount is 0.
        """
        SENTINEL = "9999-12-31T23:59:59+00:00"
        platform_clause = " AND platform = ?" if platform is not None else ""
        params: list = [schedule_id]
        if platform is not None:
            params.append(platform)
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT next_run_at FROM schedules WHERE id = ? AND status = 'active'{platform_clause}",
                tuple(params),
            ).fetchone()
            if row is None:
                return False
            current_next = row["next_run_at"]
            if current_next == SENTINEL:
                # Already claimed
                return False
            cursor = conn.execute(
                "UPDATE schedules SET next_run_at = '9999-12-31T23:59:59+00:00' "
                "WHERE id = ? AND status = 'active' AND next_run_at = ?" + platform_clause,
                (schedule_id, current_next) + (tuple(params[1:]) if platform is not None else ()),
            )
            return cursor.rowcount == 1

    def advance_after_run(
        self,
        schedule_id: str,
        next_run_at: datetime,
        last_run_at: datetime,
        last_status: str,
        last_error: str | None = None,
    ) -> None:
        """After a run completes, set next_run_at and record the outcome."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE schedules SET next_run_at = ?, last_run_at = ?, "
                "last_status = ?, last_error = ? WHERE id = ?",
                (_ts(next_run_at), _ts(last_run_at), last_status, last_error, schedule_id),
            )

    def cancel(self, schedule_id: str) -> bool:
        """Remove a schedule (FR32). Returns True if it was active."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE schedules SET status = 'cancelled' "
                "WHERE id = ? AND status = 'active'",
                (schedule_id,),
            )
            return cursor.rowcount == 1

    def get(self, schedule_id: str) -> Schedule | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM schedules WHERE id = ?",
                (schedule_id,),
            ).fetchone()
        return None if row is None else self._row_to_schedule(row)

    def recover_claimed(self, now: datetime, platform: str | None = None) -> list[Schedule]:
        """On start, find rows stuck at the sentinel (crashed mid-run).

        These are schedules whose next_run_at was advanced to the sentinel
        but never restored — the process died mid-execution. They are
        returned so the engine can treat them as missed occurrences.
        """
        platform_clause = "AND platform = ?" if platform is not None else ""
        params: list = ["9999-12-31T23:59:59+00:00"]
        if platform is not None:
            params.append(platform)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT {", ".join(_COLUMNS)}
                    FROM schedules
                    WHERE status = 'active'
                      AND next_run_at = ?
                      {platform_clause}""",
                params,
            ).fetchall()
        return [self._row_to_schedule(r) for r in rows]

    def list_for_agent(
        self, agent: str, statuses: Iterable[str] = ("active",)
    ) -> list[Schedule]:
        statuses = tuple(statuses)
        placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT {", ".join(_COLUMNS)}
                    FROM schedules
                    WHERE agent = ? AND status IN ({placeholders})
                    ORDER BY next_run_at, id""",
                (agent, *statuses),
            ).fetchall()
        return [self._row_to_schedule(r) for r in rows]

    @staticmethod
    def _row_to_schedule(row: sqlite3.Row) -> Schedule:
        values = dict(row)
        for ts_field in ("next_run_at", "last_run_at", "created_at"):
            raw = values[ts_field]
            values[ts_field] = None if raw is None else datetime.fromisoformat(raw)
        return Schedule(**values)

    # ---- async facades ----

    async def create_async(self, row: Schedule) -> str:
        return await asyncio.to_thread(self.create, row)

    async def due_async(self, now: datetime, platform: str | None = None) -> list[Schedule]:
        return await asyncio.to_thread(self.due, now, platform)

    async def claim_async(self, schedule_id: str, platform: str | None = None) -> bool:
        return await asyncio.to_thread(self.claim, schedule_id, platform)

    async def advance_after_run_async(
        self,
        schedule_id: str,
        next_run_at: datetime,
        last_run_at: datetime,
        last_status: str,
        last_error: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.advance_after_run, schedule_id, next_run_at, last_run_at, last_status, last_error
        )

    async def cancel_async(self, schedule_id: str) -> bool:
        return await asyncio.to_thread(self.cancel, schedule_id)

    async def get_async(self, schedule_id: str) -> Schedule | None:
        return await asyncio.to_thread(self.get, schedule_id)

    async def recover_claimed_async(
        self, now: datetime, platform: str | None = None
    ) -> list[Schedule]:
        return await asyncio.to_thread(self.recover_claimed, now, platform)

    async def list_for_agent_async(
        self, agent: str, statuses: Iterable[str] = ("active",)
    ) -> list[Schedule]:
        return await asyncio.to_thread(self.list_for_agent, agent, statuses)
