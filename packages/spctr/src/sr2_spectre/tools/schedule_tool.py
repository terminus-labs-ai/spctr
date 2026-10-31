"""The ``schedule`` tool (spc-185, FR24–FR26, FR32–FR33; spc-186, FR31, FR9–FR11).

One tool with an ``action`` field (NFR Tool schema economy). spc-185
implemented ``create``, ``cancel`` and ``run_now``; spc-186 completes
``list`` (FR31) and the ``channel`` destination (FR9–FR11); the agent
payload lands in spc-187.

Origin plumbing follows the reminder tool's pattern: the owning interface
records the turn's origin on a ContextVar before the turn runs, and the
tool reads it at call time.
"""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sr2_spectre.schedules.cron import CronParseError, parse_cron
from sr2_spectre.schedules.store import Schedule, ScheduleStore

logger = logging.getLogger(__name__)

# FR34: no schedule delivery (non-Discord interfaces).
NO_INTERFACE_ERROR = "Error: Schedules need the Discord interface"

# FR25: max payload sizes.
COMMAND_MAX = 4000
AGENT_INSTRUCTION_MAX = 4000

# FR11: a channel naming the other platform is refused outright. Mirrors
# the reminder tool's constant with the tool's own noun.
CROSS_PLATFORM_ERROR = "Error: Cross-platform schedules are not supported"

# FR10/FR11: the platform a channel name may prefix ("discord#x", "zulip#x").
_PLATFORM_PREFIXES = ("discord", "zulip")

# FR10: the guard message when a resolver yields an unusable match.
MALFORMED_RESOLVER_ERROR = "Error: channel lookup returned an unusable result"

# FR10/FR11: the tool's answer to a named-channel create when no owning
# interface registered a resolver.
NO_RESOLVER_ERROR = "Error: named channels are not available in this build"


@dataclass(frozen=True)
class ScheduleOrigin:
    """Where a schedule request came from (one turn's worth)."""

    agent: str
    platform: str
    requester_id: str
    requester_mention: str
    channel_id: str
    channel_name: str


@dataclass(frozen=True)
class ScheduleTurnScope:
    """What the owning interface carries for one turn."""

    origin: ScheduleOrigin


# This turn's schedule scope: set by the owning interface around its turn,
# read by the tool at call time.
schedule_turn_scope: ContextVar[ScheduleTurnScope | None] = ContextVar(
    "spectre_schedules_turn_scope", default=None
)


class ScheduleTool:
    """The ``schedule`` tool callable, registered on the agent's registry."""

    name = "schedule"
    description = (
        "Create, cancel, or manually run a recurring scheduled job "
        "(command or agent payload) that fires on a cron schedule."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"enum": ["create", "list", "cancel", "run_now"]},
            "cron": {
                "type": "string",
                "description": "5-field cron, e.g. '0 7 * * *' or '0 9 * * 1'",
            },
            "type": {"enum": ["command", "agent"]},
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "instruction": {"type": "string"},
            "agent": {
                "type": "string",
                "description": "Target agent for an agent payload; omit for the creating agent",
            },
            "channel": {
                "type": "string",
                "description": "Same-platform channel name; omit for this channel",
            },
            "label": {
                "type": "string",
                "description": "Short name shown in posts and list",
            },
            "id": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(
        self,
        store: ScheduleStore,
        timezone: str,
        agent_name: str,
        origin_getter: Any,
        clock: Any = None,
        engine: Any = None,
    ) -> None:
        self._schedule_store = store
        self._timezone = ZoneInfo(timezone)
        self._name = agent_name
        self._get_origin = origin_getter
        self._now_fn = clock or (lambda: datetime.now(tz=ZoneInfo("UTC")))
        self._engine = engine
        # FR10: the owning interface registers its channel resolver here
        # (spc-186). The Runtime constructs this tool before any interface
        # exists, so the seam is post-construction, mirroring how the
        # engine is wired.
        self._schedule_channel_resolver: Any = None

    def set_channel_resolver(self, resolver: Any) -> None:
        """Register the owning interface's named-channel resolver (FR10/FR11).

        ``resolver`` is ``async (platform, name[, scope]) -> list`` of
        matches, each carrying ``channel_id`` and ``channel_name`` (the
        FR10 match shape). A resolver marked ``spectre_platform ==
        "discord"`` is called with the reminder turn scope, exactly as
        the reminder tool calls it: that scope carries the guild.
        """
        self._schedule_channel_resolver = resolver

    def _now(self) -> datetime:
        return self._now_fn()

    async def __call__(
        self,
        action: str,
        cron: str | None = None,
        type: str | None = None,
        command: str | None = None,
        cwd: str | None = None,
        instruction: str | None = None,
        agent: str | None = None,
        channel: str | None = None,
        label: str | None = None,
        id: str | None = None,
    ) -> str:
        scoped = schedule_turn_scope.get()
        origin = scoped.origin if scoped is not None else self._get_origin()
        if origin is None:
            return NO_INTERFACE_ERROR

        if action == "create":
            return await self._create(
                cron, type, command, cwd, instruction, agent, channel, label, origin
            )
        if action == "cancel":
            return await self._cancel(id, origin)
        if action == "run_now":
            return await self._run_now(id, origin)
        if action == "list":
            return await self._list(origin)
        return f"Error: unknown action '{action}'"

    async def _create(
        self,
        cron: str | None,
        payload_type: str | None,
        command: str | None,
        cwd: str | None,
        instruction: str | None,
        agent: str | None,
        channel: str | None,
        label: str | None,
        origin: ScheduleOrigin,
    ) -> str:
        # FR25 validation
        if not cron or not cron.strip():
            return "Error: 'cron' is required and must not be empty"
        try:
            spec = parse_cron(cron)
        except CronParseError as exc:
            return f"Error: invalid cron expression: {exc}"

        if not payload_type:
            return "Error: 'type' is required (command or agent)"
        if payload_type not in ("command", "agent"):
            return f"Error: unknown payload type '{payload_type}'"

        # Validate payload content
        if payload_type == "command":
            if not command or not command.strip():
                return "Error: 'command' is required for a command payload"
            if len(command) > COMMAND_MAX:
                return f"Error: 'command' exceeds {COMMAND_MAX} characters"
            payload_json = json.dumps({"command": command, "cwd": cwd})
        else:
            if not instruction or not instruction.strip():
                return "Error: 'instruction' is required for an agent payload"
            if len(instruction) > AGENT_INSTRUCTION_MAX:
                return f"Error: 'instruction' exceeds {AGENT_INSTRUCTION_MAX} characters"
            target_agent = agent or origin.agent
            payload_json = json.dumps({"agent": target_agent, "instruction": instruction})

        # FR24/FR9: destination defaults to origin channel root
        dest_channel_id = origin.channel_id
        dest_channel_name = origin.channel_name
        # FR10/FR11: resolve a named destination after validation; nothing
        # is stored when the name does not resolve to exactly one channel.
        if channel is not None:
            resolved = await self._resolve_channel(channel, origin)
            if isinstance(resolved, str):
                return resolved
            dest_channel_id = str(resolved.channel_id)
            dest_channel_name = str(resolved.channel_name)

        now = self._now()
        next_run = spec.next_after(now, self._timezone)

        row = Schedule(
            id="s0",  # placeholder; store assigns real s<N>
            agent=origin.agent,
            platform=origin.platform,
            requester_id=origin.requester_id,
            requester_mention=origin.requester_mention,
            dest_channel_id=dest_channel_id,
            dest_channel_name=dest_channel_name,
            label=label or "",
            cron=cron,
            payload_type=payload_type,
            payload=payload_json,
            on_miss="run_late",
            next_run_at=next_run,
            last_run_at=None,
            created_at=now,
            status="active",
            last_status=None,
            last_error=None,
        )
        new_id = await self._schedule_store.create_async(row)
        logger.info("schedule create: agent=%s id=%s", self._name, new_id)

        # FR12-style confirmation
        next_local = next_run.astimezone(self._timezone)
        next_str = next_local.strftime("%a %H:%M %Z")
        payload_summary = (command or instruction or "")[:40]
        return (
            f"Schedule {new_id} set: {next_str} in #{dest_channel_name} "
            f"({payload_type}: {payload_summary})"
        )

    async def _resolve_channel(
        self, channel: str, origin: ScheduleOrigin
    ) -> Any:
        """FR10/FR11: resolve a named destination. Returns an error string
        or the single match. Mirrors the reminder tool's resolver."""
        name = channel.strip()
        if name.startswith("#"):
            name = name[1:].strip()
        prefix, sep, rest = name.partition("#")
        if sep and prefix.lower() in _PLATFORM_PREFIXES:
            # FR11: naming the other platform is refused before lookup.
            if prefix.lower() != origin.platform:
                return CROSS_PLATFORM_ERROR
            name = rest.strip()
        if self._schedule_channel_resolver is None:
            return NO_RESOLVER_ERROR
        # FR10: a Discord resolver scopes its match to the origin guild;
        # it is the only resolver that takes a scope, and the reminder
        # turn scope is what carries the guild (spc-141).
        if getattr(self._schedule_channel_resolver, "spectre_platform", None) == "discord":
            from sr2_spectre.reminders import tool as reminders_tool

            scope = reminders_tool.turn_scope.get()
            matches = await self._schedule_channel_resolver(origin.platform, name, scope)
        else:
            matches = await self._schedule_channel_resolver(origin.platform, name)
        # Guard: a resolver returning anything but the documented match
        # shape (e.g. raw channel dicts) is a tool error, not an exception.
        if matches and not all(
            hasattr(m, "channel_id") and hasattr(m, "channel_name")
            for m in matches
        ):
            logger.error(
                "%s: resolver returned %r", MALFORMED_RESOLVER_ERROR, matches[0]
            )
            return MALFORMED_RESOLVER_ERROR
        if not matches:
            return f"Error: channel '{channel}' not found on this platform"
        if len(matches) > 1:
            return f"Error: channel '{channel}' is ambiguous on this platform"
        return matches[0]

    async def _cancel(self, id: str | None, origin: ScheduleOrigin) -> str:
        """FR32: cancel removes the schedule."""
        if not id:
            return "Error: 'id' is required for cancel"
        row = await self._schedule_store.get_async(id)
        if row is None or row.agent != origin.agent:
            return f"Error: schedule '{id}' not found"
        if row.status == "cancelled":
            return f"Error: schedule {id} is already cancelled"
        if not await self._schedule_store.cancel_async(id):
            return f"Error: schedule {id} could not be cancelled"
        logger.info("schedule cancel: agent=%s id=%s", self._name, id)
        return f"Cancelled schedule {id}"

    async def _run_now(self, id: str | None, origin: ScheduleOrigin) -> str:
        """FR33: run_now executes immediately through the normal path."""
        if not id:
            return "Error: 'id' is required for run_now"
        row = await self._schedule_store.get_async(id)
        if row is None or row.agent != origin.agent:
            return f"Error: schedule '{id}' not found"
        if row.status == "cancelled":
            return f"Error: schedule {id} is cancelled"
        if self._engine is None:
            return "Error: schedule engine not available"
        await self._engine.execute_now(row)
        logger.info("schedule run_now: agent=%s id=%s", self._name, id)
        return f"Schedule {id} executed. Next run remains as scheduled."

    async def _list(self, origin: ScheduleOrigin) -> str:
        """FR31: this agent's active schedules sorted by next run."""
        rows = await self._schedule_store.list_for_agent_async(origin.agent)
        if not rows:
            return "No schedules."
        lines = []
        for s in rows:
            next_local = s.next_run_at.astimezone(self._timezone)
            next_str = next_local.strftime("%a %Y-%m-%d %H:%M %Z")
            payload = json.loads(s.payload)
            summary = (payload.get("command") or payload.get("instruction") or "")[:40]
            lines.append(
                f"{s.id} {next_str} cron={s.cron} #{s.dest_channel_name} "
                f"{s.payload_type}: {summary} [{s.last_status or 'never run'}]"
            )
        return "\n".join(lines)
