"""Schedule engine (spc-185, FR28–FR30, FR35).

Polls the store every ``poll_seconds``, claims each due schedule
(platform-scoped per spc-176), runs its payload serially, advances
``next_run_at``, and posts failures to the destination channel.

On start, schedules whose ``next_run_at`` is in the past are handled
per FR30: ``run_late`` fires once then advances past now; ``skip``
advances without firing. A schedule stuck at the claim sentinel (crashed
mid-run) is treated as a missed occurrence.

Payloads run serially (one at a time per process). A payload failure
never stops the engine (FR29).

Each poll scans the store directory (spc-995, FR39–FR44): leftover temp
files are swept; invalid files are skipped and logged at ERROR once per
content hash; a file first seen without ``next_run_at`` is scheduled for
its next occurrence (never retro-fires); a file first seen with a past
``next_run_at`` follows the FR30 ``on_miss`` policy; a cron edit
reschedules; ``enabled: false`` files are never due or advanced, and a
flip back to enabled advances silently to the next occurrence.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from spctr.schedules.agent import AgentResult, run_agent_dispatch, split_message
from spctr.schedules.command import CommandResult, run_command
from spctr.schedules.cron import CronSpec, parse_cron
from spctr.schedules.store import CLAIM_SENTINEL, InvalidEntry, Schedule, ScheduleStore

logger = logging.getLogger(__name__)

LATE_THRESHOLD = timedelta(minutes=5)
_SENTINEL = datetime.fromisoformat(CLAIM_SENTINEL)


class ScheduleEngine:
    """Polls the schedule store and executes due payloads.

    Args:
        store: The agent's persistent schedule store.
        send: ``async (channel_id, content) -> None``. Posts a message
            to the destination channel. Must raise on failure.
        poll_seconds: Seconds between polls.
        tz: IANA zone for cron interpretation.
        agent_name: Agent name for log lines.
        command_timeout: Default timeout for command payloads.
        platform: The platform this engine serves (spc-176 scoping).
        clock: Callable returning the current UTC datetime.
    """

    def __init__(
        self,
        store: ScheduleStore,
        send: Callable[[str, str], Awaitable[None]],
        *,
        poll_seconds: float,
        tz: ZoneInfo,
        agent_name: str,
        command_timeout: float = 600.0,
        agent_timeout: float = 600.0,
        platform: str | None = None,
        clock: Callable[[], datetime] | None = None,
        agent_runner: Callable | None = None,
    ) -> None:
        self._schedule_store = store
        self._send = send
        self._poll_seconds = poll_seconds
        self._timezone = tz
        self._name = agent_name
        self._command_timeout = command_timeout
        self._agent_timeout = agent_timeout
        self._platform = platform
        self._now_fn = clock or (lambda: datetime.now(ZoneInfo("UTC")))
        self._agent_runner = agent_runner  # injectable for tests
        self._task: asyncio.Task | None = None
        # FR39/FR44: last seen (status, cron) per schedule id, this process.
        self._seen: dict[str, tuple[str, str]] = {}
        # FR40: (filename, content hash) pairs already logged at ERROR.
        self._invalid_logged: set[tuple[str, str]] = set()

    def _now(self) -> datetime:
        return self._now_fn()

    def _next_after(self, cron: str, now: datetime) -> datetime:
        try:
            return parse_cron(cron).next_after(now, self._timezone)
        except Exception:
            return now + timedelta(hours=1)

    async def start(self) -> None:
        """Catch up missed runs, recover crashed runs, start the polling task."""
        await self._recover()
        await self._catch_up()
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        """End the engine; unrun rows are left intact (FR35)."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _recover(self) -> None:
        """Handle schedules stuck at the claim sentinel (FR30)."""
        now = self._now()
        stuck = await self._schedule_store.recover_claimed_async(now, self._platform)
        for sched in stuck:
            await self._schedule_store.advance_after_run_async(
                sched.id, self._next_after(sched.cron, now), now,
                "error", "recovered from crash"
            )
            logger.info(
                "schedule recovered: agent=%s id=%s", self._name, sched.id
            )

    async def _catch_up(self) -> None:
        """FR30: missed occurrences run once at start, then advance past now.

        An ordinary overdue active row (``next_run_at`` in the past, not
        the claim sentinel — those are the crash rows ``_recover`` handles)
        is caught up per its ``on_miss``:

        - ``skip``: advance past now without firing. ``last_status`` and
          ``last_error`` record the skip so the row is inspectable.
        - otherwise (``run_late``): claim the row (the platform-scoped
          claim, so a concurrent second process stays out and one start
          catches up a row at most once), then execute through the same
          ``_execute`` path the poll uses — the fired log line is marked
          late because the fire time is more than five minutes past the
          missed occurrence. ``_execute`` advances to the next cron
          occurrence strictly after now, so nothing replays or refires.
        """
        now = self._now()
        overdue = await self._schedule_store.due_async(now, self._platform)
        for sched in overdue:
            if sched.on_miss == "skip":
                await self._skip_missed(sched, now)
                continue
            if not await self._schedule_store.claim_async(sched.id, self._platform):
                continue
            try:
                await self._execute(sched, now)
            except Exception:
                # FR29: a payload failure never stops the engine.
                logger.exception(
                    "schedule catch-up error: agent=%s id=%s",
                    self._name, sched.id,
                )
                await self._schedule_store.advance_after_run_async(
                    sched.id, self._next_after(sched.cron, now), now,
                    "failed", "catch-up error"
                )

    async def _skip_missed(self, sched: Schedule, now: datetime) -> None:
        """FR30 ``on_miss: skip``: advance past now without firing."""
        await self._schedule_store.advance_after_run_async(
            sched.id, self._next_after(sched.cron, now), now, "skipped", None
        )
        logger.info(
            "schedule missed run skipped: agent=%s id=%s",
            self._name, sched.id,
        )

    async def _run(self) -> None:
        while True:
            await self._poll_once()
            await asyncio.sleep(self._poll_seconds)

    def _report_invalid(self, invalid: list[InvalidEntry]) -> None:
        """FR40: one ERROR line per invalid file per content hash."""
        for entry in invalid:
            key = (entry.filename, entry.content_hash)
            if key in self._invalid_logged:
                continue
            self._invalid_logged.add(key)
            logger.error(
                "schedule file invalid: agent=%s file=%s reason=%s",
                self._name, entry.filename, entry.reason,
            )

    async def _reschedule(self, sched: Schedule, now: datetime, what: str) -> bool:
        """Silently set ``next_run_at`` to the next occurrence after *now*."""
        next_run = self._next_after(sched.cron, now)
        done = await self._schedule_store.reschedule_async(
            sched.id, sched.next_run_at, next_run
        )
        if done:
            logger.info(
                "schedule %s: agent=%s id=%s next_run_at=%s",
                what, self._name, sched.id, next_run.isoformat(),
            )
        return done

    async def _observe(self, sched: Schedule, now: datetime) -> bool:
        """Per-file bookkeeping for one scanned row; True when it is due.

        FR44: a paused file is never due, caught up or advanced; seeing it
        flip back to enabled advances it silently.  FR39: a file with no
        ``next_run_at`` is scheduled, never retro-fired; a cron edit
        reschedules; a file first seen with a past ``next_run_at`` follows
        its FR30 ``on_miss``.
        """
        prev = self._seen.get(sched.id)
        self._seen[sched.id] = (sched.status, sched.cron)
        if sched.status != "active" or sched.next_run_at == _SENTINEL:
            return False
        if sched.next_run_at is None:
            what = "scheduled"
        elif prev is not None and prev[0] != "active":
            what = "resumed"
        elif prev is not None and prev[1] != sched.cron:
            what = "rescheduled"
        else:
            if sched.next_run_at > now:
                return False
            if prev is None and sched.on_miss == "skip":
                await self._skip_missed(sched, now)
                return False
            return True
        if not await self._reschedule(sched, now, what):
            # The file changed under us: retry from the old view next poll.
            if prev is None:
                self._seen.pop(sched.id, None)
            else:
                self._seen[sched.id] = prev
        return False

    async def _scan(self, now: datetime) -> list[Schedule]:
        """One poll's directory scan (FR39–FR41); returns the due rows."""
        try:
            await asyncio.to_thread(
                self._schedule_store.sweep_temp_files, now, self._poll_seconds
            )
        except OSError:
            logger.exception("schedule temp sweep error: agent=%s", self._name)
        rows, invalid = await asyncio.to_thread(self._schedule_store.scan)
        self._report_invalid(invalid)
        present = {row.id for row in rows}
        for gone in set(self._seen) - present:
            del self._seen[gone]
        due: list[Schedule] = []
        for sched in rows:
            if self._platform is not None and sched.platform != self._platform:
                continue
            if await self._observe(sched, now):
                due.append(sched)
        return sorted(due, key=lambda r: (r.next_run_at, r.id))

    async def _poll_once(self) -> None:
        now = self._now()
        due = await self._scan(now)
        for sched in due:
            if not await self._schedule_store.claim_async(sched.id, self._platform):
                continue
            try:
                await self._execute(sched, now)
            except Exception as exc:
                # FR29: a payload failure never stops the engine.
                logger.exception(
                    "schedule execution error: agent=%s id=%s",
                    self._name, sched.id,
                )
                # Still advance so it doesn't fire in a loop
                await self._schedule_store.advance_after_run_async(
                    sched.id, self._next_after(sched.cron, now), now,
                    "failed", str(exc)[:200]
                )

    async def _execute(self, sched: Schedule, fire_time: datetime) -> None:
        """Run one schedule's payload and advance."""
        next_run = self._next_after(sched.cron, fire_time)

        late = fire_time - sched.next_run_at > LATE_THRESHOLD
        payload = json.loads(sched.payload)

        result = None
        if sched.payload_type == "command":
            result = await self._run_command_payload(payload, sched)
            await self._handle_command_result(sched, result)
        elif sched.payload_type == "agent":
            result = await self._run_agent_payload(payload, sched)
            await self._handle_agent_result(sched, result)

        success = result is None or result.success
        await self._schedule_store.advance_after_run_async(
            sched.id,
            next_run,
            fire_time,
            "ok" if success else "failed",
            None if success else (result.error_line if result else None),
        )

        if late:
            logger.info(
                "schedule fired late: agent=%s id=%s", self._name, sched.id
            )
        else:
            logger.info(
                "schedule fired: agent=%s id=%s", self._name, sched.id
            )

    async def _run_command_payload(self, payload: dict, sched: Schedule) -> CommandResult:
        """Execute a command payload."""
        command = payload.get("command", "")
        cwd = payload.get("cwd")
        return await run_command(
            command,
            agent=self._name,
            cwd=cwd,
            timeout=self._command_timeout,
            clock=self._now_fn,
        )

    async def _handle_command_result(self, sched: Schedule, result: CommandResult) -> None:
        """Post failure to the destination channel (FR26)."""
        if result.success:
            return
        label_part = f" ({sched.label})" if sched.label else ""
        message = f"⚠️ Schedule {sched.id}{label_part} failed: {result.error_line}"
        try:
            await self._send(sched.dest_channel_id, message)
        except Exception:
            logger.exception(
                "schedule failure post error: agent=%s id=%s",
                self._name, sched.id,
            )

    async def _run_agent_payload(self, payload: dict, sched: Schedule) -> AgentResult:
        """Execute an agent payload via the dispatch CLI (spc-187, FR27)."""
        target_agent = payload.get("agent", self._name)
        instruction = payload.get("instruction", "")
        runner = self._agent_runner or run_agent_dispatch
        return await runner(
            target_agent,
            instruction,
            timeout=self._agent_timeout,
            agent=self._name,
            clock=self._now_fn,
        )

    async def _handle_agent_result(self, sched: Schedule, result: AgentResult) -> None:
        """Post agent result to the destination channel (spc-187, FR27).

        On ``done``: post the final message verbatim, split at 2000 chars.
        On any other status: post the one-line ⚠️ form with the status.
        The schedule stays active regardless.
        """
        label_part = f" ({sched.label})" if sched.label else ""
        if result.success:
            if not result.message:
                return
            chunks = split_message(result.message)
            for chunk in chunks:
                try:
                    await self._send(sched.dest_channel_id, chunk)
                except Exception:
                    logger.exception(
                        "schedule agent post error: agent=%s id=%s",
                        self._name, sched.id,
                    )
                    break
        else:
            message = f"⚠️ Schedule {sched.id}{label_part} failed: {result.error_line}"
            try:
                await self._send(sched.dest_channel_id, message)
            except Exception:
                logger.exception(
                    "schedule failure post error: agent=%s id=%s",
                    self._name, sched.id,
                )

    async def execute_now(self, sched: Schedule) -> None:
        """Run a schedule immediately (FR33: run_now).

        Executes through the normal path (posts and logs) but leaves
        ``next_run_at`` set to the next cron occurrence after now.
        """
        now = self._now()
        next_run = self._next_after(sched.cron, now)

        payload = json.loads(sched.payload)

        result = None
        if sched.payload_type == "command":
            result = await self._run_command_payload(payload, sched)
            await self._handle_command_result(sched, result)
            status = "ok" if result.success else "failed"
            error = None if result.success else result.error_line
        elif sched.payload_type == "agent":
            result = await self._run_agent_payload(payload, sched)
            await self._handle_agent_result(sched, result)
            status = "ok" if result.success else "failed"
            error = None if result.success else result.error_line
        else:
            status = "ok"
            error = None

        await self._schedule_store.advance_after_run_async(
            sched.id, next_run, now, status, error
        )
        logger.info(
            "schedule manual run: agent=%s id=%s", self._name, sched.id
        )
