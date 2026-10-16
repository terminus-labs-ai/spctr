"""Delivery loop for reminders (spc-139; FR13-FR17, FR22).

Polls the store every ``poll_seconds``, claims each due row
(``pending`` → ``sending`` in one transaction, so one reminder is sent by
one sender only), sends it through the owning interface's send callable,
and marks it delivered only after the platform accepts the send. A send
that raises follows the FR18 bounded retry policy: the due time moves to
+1 minute, then +5 minutes, and the third failure marks the row
``failed`` with an ERROR log (spc-140); other due rows in the same poll
are still processed.

``start()`` first returns every ``sending`` row to ``pending`` (a crash
between send and mark must not lose the reminder — FR17) and the first
poll sends everything already due in due-time order, with the late suffix
per FR15. ``stop()`` ends the loop and leaves pending rows intact (FR22).
The loop is one ``asyncio.Task``; stopping cancels it, so process exit
waits for at most one in-flight send (NFR Shutdown).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from sr2_spectre.reminders.tool import LATE_SUFFIX_THRESHOLD
from sr2_spectre.reminders.store import Reminder, ReminderStore

logger = logging.getLogger(__name__)

LATE_SUFFIX = "(due {due}, sent late)"

# FR18: a failed send is retried at +1 and +5 minutes; the third failure
# marks the reminder failed.
RETRY_DELAYS = (timedelta(minutes=1), timedelta(minutes=5))
MAX_SEND_ATTEMPTS = len(RETRY_DELAYS) + 1


def _next_attempt(reminder: Reminder) -> datetime:
    """When this row's failed attempt should be re-run (FR18, spc-157).

    The delay counts from the row's current attempt time — the original
    due time on the first failure, the stored ``next_attempt_at`` after
    that — so the schedule is due+1 then due+6 (cumulative), the timing
    the AC9 tests pin.
    """
    current = reminder.next_attempt_at or reminder.due_at
    return current + RETRY_DELAYS[
        min(reminder.attempts, len(RETRY_DELAYS) - 1)
    ]


def format_reminder_message(reminder: Reminder, tz: ZoneInfo, now: datetime) -> str:
    """``<mention> Reminder: <text>`` plus the late suffix when sent >5 min
    after due (FR13, FR15)."""
    message = f"{reminder.requester_mention} Reminder: {reminder.text}"
    if now - reminder.due_at > LATE_SUFFIX_THRESHOLD:
        due_local = reminder.due_at.astimezone(tz).strftime("%a %Y-%m-%d %H:%M %Z")
        message += LATE_SUFFIX.format(due=due_local)
    return message


class DeliveryLoop:
    """Polls the store and delivers due reminders through a send callable.

    Args:
        store: The agent's persistent reminder store.
        send: ``async (channel_id, topic, content) -> None``. Must raise
            on failure; the loop does not inspect return values.
        poll_seconds: Seconds between polls (``reminders.poll_seconds``).
        tz: IANA zone for rendering local times in messages and logs.
        agent_name: Agent name for log lines (never the text).
        clock: Callable returning the current UTC datetime. Injected for
            tests; defaults to the real clock.
    """

    def __init__(
        self,
        store: ReminderStore,
        send: Callable[[str, str, str], Awaitable[None]],
        *,
        poll_seconds: float,
        tz: ZoneInfo,
        agent_name: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._send = send
        self._poll_seconds = poll_seconds
        self._tz = tz
        self._agent_name = agent_name
        self._clock = clock or (lambda: datetime.now(ZoneInfo("UTC")))
        self._task: asyncio.Task | None = None

    def _now(self) -> datetime:
        return self._clock()

    async def start(self) -> None:
        """Reset interrupted sends and start the polling task."""
        await self._store.reset_sending_to_pending_async()
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        """End the loop; pending rows are left intact for the next start."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while True:
            await self._poll_once()
            await asyncio.sleep(self._poll_seconds)

    async def _poll_once(self) -> None:
        now = self._now()
        due = await self._store.due_async(now)
        for reminder in due:
            if not await self._store.claim_async(reminder.id):
                # Someone else already claimed it (single-loop in practice,
                # but the claim is the only thing that guarantees it).
                continue
            content = format_reminder_message(reminder, self._tz, now)
            try:
                await self._send(
                    reminder.dest_channel_id, reminder.dest_topic, content
                )
            except Exception as exc:  # noqa: BLE001 — any send failure is retried
                # FR18 policy (spc-140): due +1 min, then +5 min, scheduled
                # on next_attempt_at so due_at keeps the original due time
                # (spc-157); the third failure marks the row failed with an
                # ERROR log naming the ID and the error.
                status = await self._store.due_with_retry_async(
                    reminder.id,
                    str(exc),
                    _next_attempt(reminder),
                    max_attempts=MAX_SEND_ATTEMPTS,
                )
                if status == "failed":
                    logger.error(
                        "reminder failed after %d attempts: "
                        "agent=%s id=%s error=%s",
                        MAX_SEND_ATTEMPTS, self._agent_name, reminder.id, exc,
                    )
                else:
                    logger.warning(
                        "reminder send failed (retrying later): "
                        "agent=%s id=%s error=%s",
                        self._agent_name, reminder.id, exc,
                    )
                continue
            late = now - reminder.due_at > LATE_SUFFIX_THRESHOLD
            await self._store.mark_delivered_async(reminder.id, self._now())
            if late:
                logger.info(
                    "reminder delivered late: agent=%s id=%s",
                    self._agent_name, reminder.id,
                )
            else:
                logger.info(
                    "reminder delivered: agent=%s id=%s",
                    self._agent_name, reminder.id,
                )
