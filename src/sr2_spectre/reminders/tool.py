"""The ``reminder`` tool (spc-139; FR7 in this slice: ``create`` only).

One tool with an ``action`` field rather than three tools (NFR Tool schema
economy). ``list`` and ``cancel`` arrive in spc-140 alongside named-channel
resolution (FR10/FR11); until then they return explicit tool errors.

Origin plumbing: the owning interface records each turn's origin —
platform, channel, requester and mention — on its ``Agent`` under
``reminders_origin`` BEFORE the turn runs (and clears it afterwards), so
concurrent turns in different channels never record each other's origin.
The tool reads it at call time, inside the turn that owns it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sr2_spectre.reminders.store import Reminder, ReminderStore
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

LATE_SUFFIX_THRESHOLD = timedelta(minutes=5)  # FR15


@dataclass(frozen=True)
class ReminderOrigin:
    """Where a reminder request came from (one turn's worth)."""

    agent: str
    platform: str
    requester_id: str
    requester_mention: str
    channel_id: str
    channel_name: str
    topic: str  # the Zulip topic the request came from (Discord: "")


def format_local_due(due_utc: datetime, tz: ZoneInfo) -> str:
    """Local due time: weekday, date, time and zone abbreviation (FR12)."""
    local = due_utc.astimezone(tz)
    return local.strftime("%a %Y-%m-%d %H:%M %Z")


class ReminderTool:
    """The ``reminder`` tool callable, registered on the agent's registry."""

    name = "reminder"
    description = (
        "Create a one-shot reminder that is posted to a chat channel at a "
        "later time with the stored text verbatim."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"enum": ["create", "list", "cancel"]},
            "text": {"type": "string"},
            "in_minutes": {"type": "integer"},
            "at": {
                "type": "string",
                "description": "YYYY-MM-DD HH:MM local time",
            },
            "channel": {
                "type": "string",
                "description": "Same-platform channel name; omit for this channel",
            },
            "id": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(
        self,
        store: ReminderStore,
        timezone: str,
        agent_name: str,
        origin_getter: Any,
        clock: Any = None,
        topic: str = "reminders",
    ) -> None:
        """
        Args:
            store: The agent's persistent reminder store.
            timezone: IANA zone from ``reminders.timezone``; ``at`` is
                interpreted in this zone and all replies render in it.
            agent_name: Agent name for log lines (never the text).
            origin_getter: Callable returning the current turn's
                ``ReminderOrigin`` (set by the interface) or None.
            clock: Callable returning the current UTC datetime. Injected
                for tests; defaults to the real clock.
            topic: The Zulip topic reminders are posted to
                (``reminders.zulip_topic``, default ``reminders`` — FR9).
        """
        self._store = store
        self._tz = ZoneInfo(timezone)
        self._agent_name = agent_name
        self._origin_getter = origin_getter
        self._clock = clock or (lambda: datetime.now(tz=ZoneInfo("UTC")))
        self._topic = topic

    def _now(self) -> datetime:
        return self._clock()

    async def __call__(
        self,
        action: str,
        text: str | None = None,
        in_minutes: int | None = None,
        at: str | None = None,
        channel: str | None = None,
        id: str | None = None,
    ) -> str:
        if action != "create":
            # spc-140 ships list/cancel; spc-140 also ships named channels.
            return (
                "Error: only the 'create' action is available in this build. "
                "Named channels are not supported yet."
            )
        return await self._create(text, in_minutes, at, channel)

    async def _create(
        self,
        text: str | None,
        in_minutes: int | None,
        at: str | None,
        channel: str | None,
    ) -> str:
        if channel is not None:
            # spc-140: named channels. Error text per the spc-139 contract.
            return "Error: Named channels are not supported yet"

        # FR8 validation — each failure returns a tool error, stores nothing.
        if text is None or not text.strip():
            return "Error: 'text' is required and must not be empty"
        if len(text) > 1000:
            return "Error: 'text' is over 1000 characters"
        if (in_minutes is None) == (at is None):
            return (
                "Error: provide exactly one of 'in_minutes' or 'at'"
            )

        now = self._now()
        if in_minutes is not None:
            if not isinstance(in_minutes, int) or isinstance(in_minutes, bool):
                return "Error: 'in_minutes' must be an integer"
            if not 1 <= in_minutes <= 525600:
                return "Error: 'in_minutes' must be between 1 and 525600"
            due = now + timedelta(minutes=in_minutes)
        else:
            try:
                local = datetime.strptime(at, "%Y-%m-%d %H:%M").replace(
                    tzinfo=self._tz
                )
            except (TypeError, ValueError):
                return "Error: 'at' must be formatted YYYY-MM-DD HH:MM"
            due = local.astimezone(ZoneInfo("UTC"))
        if due <= now + timedelta(minutes=1):
            return "Error: the due time must be at least one minute ahead"

        origin = self._origin_getter()
        if origin is None:
            return "Error: Reminders need the Discord or Zulip interface"

        row = Reminder(
            id="r0",  # placeholder; the store assigns the real r<N>
            agent=origin.agent,
            platform=origin.platform,
            requester_id=origin.requester_id,
            requester_mention=origin.requester_mention,
            origin_channel_id=origin.channel_id,
            dest_channel_id=origin.channel_id,
            dest_channel_name=origin.channel_name,
            dest_topic=self._topic,
            text=text,
            due_at=due,
            created_at=now,
            delivered_at=None,
            status="pending",
            attempts=0,
            last_error=None,
        )
        new_id = await self._store.create_async(row)
        logger.info(
            "reminder create: agent=%s id=%s", self._agent_name, new_id
        )
        # FR12: ID, local due time (weekday, date, time, zone), destination.
        return (
            f"Reminder {new_id} set for "
            f"{format_local_due(due, self._tz)} in #{origin.channel_name}"
        )
