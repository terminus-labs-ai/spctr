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
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from sr2_spectre.schedules.command import CommandResult, run_command
from sr2_spectre.schedules.cron import CronSpec, parse_cron
from sr2_spectre.schedules.store import Schedule, ScheduleStore

logger = logging.getLogger(__name__)

LATE_THRESHOLD = timedelta(minutes=5)


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
        platform: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._schedule_store = store
        self._send = send
        self._poll_seconds = poll_seconds
        self._timezone = tz
        self._name = agent_name
        self._command_timeout = command_timeout
        self._platform = platform
        self._now_fn = clock or (lambda: datetime.now(ZoneInfo("UTC")))
        self._task: asyncio.Task | None = None

    def _now(self) -> datetime:
        return self._now_fn()

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
            try:
                spec = parse_cron(sched.cron)
                next_run = spec.next_after(now, self._timezone)
            except Exception:
                next_run = now + timedelta(hours=1)
            await self._schedule_store.advance_after_run_async(
                sched.id, next_run, now, "error", "recovered from crash"
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
                try:
                    spec = parse_cron(sched.cron)
                    next_run = spec.next_after(now, self._timezone)
                except Exception:
                    next_run = now + timedelta(hours=1)
                await self._schedule_store.advance_after_run_async(
                    sched.id, next_run, now, "skipped", None
                )
                logger.info(
                    "schedule missed run skipped: agent=%s id=%s",
                    self._name, sched.id,
                )
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
                try:
                    spec = parse_cron(sched.cron)
                    next_run = spec.next_after(now, self._timezone)
                except Exception:
                    next_run = now + timedelta(hours=1)
                await self._schedule_store.advance_after_run_async(
                    sched.id, next_run, now, "failed", "catch-up error"
                )

    async def _run(self) -> None:
        while True:
            await self._poll_once()
            await asyncio.sleep(self._poll_seconds)

    async def _poll_once(self) -> None:
        now = self._now()
        due = await self._schedule_store.due_async(now, self._platform)
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
                try:
                    spec = parse_cron(sched.cron)
                    next_run = spec.next_after(now, self._timezone)
                except Exception:
                    next_run = now + timedelta(hours=1)
                await self._schedule_store.advance_after_run_async(
                    sched.id, next_run, now, "failed", str(exc)[:200]
                )

    async def _execute(self, sched: Schedule, fire_time: datetime) -> None:
        """Run one schedule's payload and advance."""
        try:
            spec = parse_cron(sched.cron)
            next_run = spec.next_after(fire_time, self._timezone)
        except Exception:
            next_run = fire_time + timedelta(hours=1)

        late = fire_time - sched.next_run_at > LATE_THRESHOLD
        payload = json.loads(sched.payload)

        result = None
        if sched.payload_type == "command":
            result = await self._run_command_payload(payload, sched)
            await self._handle_command_result(sched, result)
        elif sched.payload_type == "agent":
            # spc-187 scope — inert for now
            pass

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

    async def execute_now(self, sched: Schedule) -> None:
        """Run a schedule immediately (FR33: run_now).

        Executes through the normal path (posts and logs) but leaves
        ``next_run_at`` set to the next cron occurrence after now.
        """
        now = self._now()
        try:
            spec = parse_cron(sched.cron)
            next_run = spec.next_after(now, self._timezone)
        except Exception:
            next_run = now + timedelta(hours=1)

        payload = json.loads(sched.payload)

        result = None
        if sched.payload_type == "command":
            result = await self._run_command_payload(payload, sched)
            await self._handle_command_result(sched, result)
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
