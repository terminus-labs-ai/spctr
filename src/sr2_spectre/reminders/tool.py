"""The ``reminder`` tool (spc-139 create/deliver; spc-140 completes FR7).

One tool with an ``action`` field rather than three tools (NFR Tool schema
economy). ``create`` optionally resolves a same-platform named channel
(FR10/FR11) through an injected resolver; ``list`` and ``cancel`` follow
FR19/FR20.

Origin plumbing: the owning interface records each turn's origin —
platform, channel, requester and mention — on its ``Agent`` under
``reminders_origin`` BEFORE the turn runs (and clears it afterwards), so
concurrent turns in different channels never record each other's origin.
The tool reads it at call time, inside the turn that owns it.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sr2_spectre.reminders.store import Reminder, ReminderStore
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

LATE_SUFFIX_THRESHOLD = timedelta(minutes=5)  # FR15

# FR11: a channel naming the other platform is refused outright.
CROSS_PLATFORM_ERROR = "Error: Cross-platform reminders are not supported"

# FR21: no chat origin (an interface that is neither Discord nor Zulip).
NO_INTERFACE_ERROR = "Error: Reminders need the Discord or Zulip interface"

# FR11: the platform a channel name may prefix ("discord#x", "zulip#x").
_PLATFORM_PREFIXES = ("discord", "zulip")

# FR10: the guard message when a resolver yields an unusable match.
MALFORMED_RESOLVER_ERROR = "Error: channel lookup returned an unusable result"

# FR10/FR21: the tool's answer to a named-channel create when this build
# has no resolver (no owning interface registered one). Extracted for the
# tests to pin the exact text; behavior unchanged.
NO_RESOLVER_SENTINEL = "Error: named channels are not available in this build"

# FR19: list shows pending rows plus failed ones within this window.
FAILED_LIST_WINDOW = timedelta(days=7)
LIST_TEXT_LIMIT = 80


def _ts(value: datetime) -> str:
    """Timestamp in the store's UTC ISO text form (window compares only)."""
    return value.astimezone(timezone.utc).isoformat()


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


@dataclass(frozen=True)
class ChannelMatch:
    """FR10: one named-channel resolver match (what the tool consumes)."""

    channel_id: int
    channel_name: str


@dataclass(frozen=True)
class TurnScope:
    """What the owning interface carries for one turn (spc-141)."""

    origin: ReminderOrigin
    guild_id: int | None = None  # Discord only: the guild to resolve inside


# This turn's reminder scope (spc-141, spec Delivery seam): set by the
# owning interface around its turn, read by the tool at call time. A task
# is the unit of isolation — concurrent tasks never share a context — so
# two turns in different channels cannot record each other's origin, with
# or without a lock. Outside a turn the default None leaves the tool on its
# origin_getter, so every other interface is unchanged.
turn_scope: ContextVar[TurnScope | None] = ContextVar(
    "spectre_reminders_turn_scope", default=None
)


def _match_has_channels(matches: Any) -> bool:
    """True when every match carries channel_id and channel_name."""
    return all(
        hasattr(match, "channel_id") and hasattr(match, "channel_name")
        for match in matches
    )


def format_local_due(due_utc: datetime, tz: ZoneInfo) -> str:
    """Local due time: weekday, date, time and zone abbreviation (FR12)."""
    local = due_utc.astimezone(tz)
    return local.strftime("%a %Y-%m-%d %H:%M %Z")


class ReminderTool:
    """The ``reminder`` tool callable, registered on the agent's registry.

    spc-159 resolver ownership: the only ways a resolver reaches the
    tool are the constructor's ``channel_resolver`` argument and the
    public :meth:`attach_channel_resolver` landing point. Both live in
    this class; no code outside ``reminders/`` reads or writes the
    resolver state.
    """

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
        channel_resolver: Any = None,
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
            channel_resolver: ``async (platform, name) -> list`` of matches,
                each with ``channel_id`` and ``channel_name`` attributes
                (FR10). The constructor is the only construction path for
                the resolver (spc-159): the owning interface registers it
                on the Runtime, which passes it in here. None rejects
                every named-channel create.
        """
        self._store = store
        self._tz = ZoneInfo(timezone)
        self._agent_name = agent_name
        self._origin_getter = origin_getter
        self._clock = clock or (lambda: datetime.now(tz=ZoneInfo("UTC")))
        self._topic = topic
        self._channel_resolver = channel_resolver

    def attach_channel_resolver(self, resolver: Any) -> None:
        """Give THIS built tool the owning interface's resolver.

        The constructor is the single way a resolver is *built into* a
        ReminderTool (spc-159); this public landing point exists because
        the Runtime registers its tool before any interface exists to
        register a resolver (FR6 ordering), and the registration must
        still reach an already-built tool. Both paths write the same
        field through ReminderTool's own code — no outside reader or
        writer of the resolver exists, and every built tool ends with
        the same hand-off the constructor takes.
        """
        self._channel_resolver = resolver

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
        # FR21: without a chat turn there is no origin, and every action
        # refuses (checked before anything else). This turn's scope wins
        # over the shared getter, so an overlapping turn can never supply
        # another channel's origin.
        scoped = turn_scope.get()
        origin = scoped.origin if scoped is not None else self._origin_getter()
        if origin is None:
            return NO_INTERFACE_ERROR
        if action == "create":
            return await self._create(text, in_minutes, at, channel, origin)
        if action == "list":
            return await self._list(origin)
        if action == "cancel":
            return await self._cancel(id, origin)
        return f"Error: unknown action '{action}'"

    async def _resolve_channel(
        self, channel: str, origin: ReminderOrigin
    ) -> Any:
        """FR10/FR11: resolve a named destination. Returns an error string
        or the single match."""
        name = channel.strip()
        if name.startswith("#"):
            name = name[1:].strip()
        prefix, sep, rest = name.partition("#")
        if sep and prefix.lower() in _PLATFORM_PREFIXES:
            # FR11: naming the other platform is refused before lookup.
            if prefix.lower() != origin.platform:
                return CROSS_PLATFORM_ERROR
            name = rest.strip()
        if self._channel_resolver is None:
            return NO_RESOLVER_SENTINEL
        # FR10: a Discord resolver scopes its match to the origin guild;
        # it is the only resolver that takes a scope, and the owning
        # interface's seam names it. Others are called exactly as before.
        if getattr(self._channel_resolver, "spectre_platform", None) == "discord":
            scope = turn_scope.get()
            matches = await self._channel_resolver(
                origin.platform, name, scope
            )
        else:
            matches = await self._channel_resolver(origin.platform, name)
        # Guard: a resolver returning anything but the documented match
        # shape (e.g. raw stream dicts) is a tool error, not an exception.
        if matches and not _match_has_channels(matches):
            logger.error(
                "%s: resolver returned %r", MALFORMED_RESOLVER_ERROR, matches[0]
            )
            return MALFORMED_RESOLVER_ERROR
        if not matches:
            return f"Error: channel '{channel}' not found on this platform"
        if len(matches) > 1:
            return f"Error: channel '{channel}' is ambiguous on this platform"
        return matches[0]

    async def _create(
        self,
        text: str | None,
        in_minutes: int | None,
        at: str | None,
        channel: str | None,
        origin: ReminderOrigin,
    ) -> str:
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

        # FR10: resolve a named destination after validation; nothing is
        # stored when the name does not resolve to exactly one channel.
        dest_channel_id, dest_channel_name = origin.channel_id, origin.channel_name
        if channel is not None:
            resolved = await self._resolve_channel(channel, origin)
            if isinstance(resolved, str):
                return resolved
            dest_channel_id = str(resolved.channel_id)
            dest_channel_name = str(resolved.channel_name)

        row = Reminder(
            id="r0",  # placeholder; the store assigns the real r<N>
            agent=origin.agent,
            platform=origin.platform,
            requester_id=origin.requester_id,
            requester_mention=origin.requester_mention,
            origin_channel_id=origin.channel_id,
            dest_channel_id=dest_channel_id,
            dest_channel_name=dest_channel_name,
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
            f"{format_local_due(due, self._tz)} in #{dest_channel_name}"
        )

    async def _list(self, origin: ReminderOrigin) -> str:
        """FR19: this agent's pending reminders plus failed ones within
        FAILED_LIST_WINDOW, sorted by due time.

        ``due_at`` is the original due time (spc-157): FR18 retries are
        scheduled from ``next_attempt_at``, so each line shows and the
        failed-row window is dated by the time the user asked for.
        """
        now = self._now()
        cutoff = _ts(now - FAILED_LIST_WINDOW)
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders "
                "WHERE agent = ? AND (status = 'pending' "
                "       OR (status = 'failed' AND due_at >= ?)) "
                "ORDER BY due_at, id",
                (origin.agent, cutoff),
            ).fetchall()
        if not rows:
            return "No reminders."
        lines = []
        for r in rows:
            reminder = self._store._row_to_reminder(r)
            text = reminder.text
            if len(text) > LIST_TEXT_LIMIT:
                text = text[: LIST_TEXT_LIMIT - 1] + "…"
            lines.append(
                f"{reminder.id} {format_local_due(reminder.due_at, self._tz)} "
                f"#{reminder.dest_channel_name} {reminder.status} {text}"
            )
        return "\n".join(lines)

    async def _cancel(self, id: str | None, origin: ReminderOrigin) -> str:
        """FR20: a pending reminder becomes cancelled and is never sent."""
        if not id:
            return "Error: 'id' is required for cancel"
        row = await self._store.get_async(id)
        if row is None or row.agent != origin.agent:
            return f"Error: reminder '{id}' not found"
        if row.status == "sending":
            # The claim is in flight; the send completes (FR20).
            return f"Reminder {id} is already being delivered"
        if not await self._store.cancel_async(id):
            return (
                f"Error: reminder {id} is {row.status}, only pending "
                "reminders can be cancelled"
            )
        logger.info("reminder cancel: agent=%s id=%s", self._agent_name, id)
        return f"Cancelled reminder {id}"
