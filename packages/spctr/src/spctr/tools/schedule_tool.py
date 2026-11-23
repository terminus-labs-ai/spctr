"""The ``schedule`` tool (spc-185, FR24–FR26, FR32–FR33; spc-186, FR31, FR9–FR11).

One tool with an ``action`` field (NFR Tool schema economy). spc-185
implemented ``create``, ``cancel`` and ``run_now``; spc-186 completes
``list`` (FR31) and the ``channel`` destination (FR9–FR11); the agent
payload lands in spc-187.

Every string this tool returns or advertises is a YAML parameter
(spc-1097, YAML-parameter mandate): the constructor takes a
``ScheduleTextsConfig`` (``agent.schedules.texts``) carrying the tool
description, the input-schema descriptions, the result formats and the
error texts. Its defaults are the spec-pinned strings verbatim, so an
untouched config reproduces the increment-2/3 forms byte-for-byte.

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
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from spctr.schedules.cron import CronParseError, parse_cron
from spctr.schedules.store import AGENT_INSTRUCTION_MAX, Schedule, ScheduleStore

if TYPE_CHECKING:
    from spctr.config import ScheduleTextsConfig

logger = logging.getLogger(__name__)


def _default_texts() -> ScheduleTextsConfig:
    """The spec-pinned text defaults, built on call.

    ``spctr.config`` imports the Discord interface package, which imports
    this module, so the config-model import stays inside a function: at
    construction time whichever import order started the process, both
    modules are loadable. (spc-1097.)
    """
    from spctr.config import ScheduleTextsConfig

    return ScheduleTextsConfig()

# FR25: max command-payload size.  The agent-instruction cap (FR27) lives in
# the store as ``AGENT_INSTRUCTION_MAX`` (imported above), so this create-time
# check and the store's load-time validity rule cannot drift apart.
COMMAND_MAX = 4000

# FR10/FR11: the platform a channel name may prefix ("discord#x", "zulip#x").
_PLATFORM_PREFIXES = ("discord", "zulip")


@dataclass(frozen=True)
class ScheduleOrigin:
    """Where a schedule request came from (one turn's worth)."""

    agent: str
    platform: str
    requester_id: str
    requester_mention: str
    channel_id: str
    channel_name: str
    # FR47: the post topic a create from this origin stores when the call
    # names none (Zulip: ``schedules.zulip_topic``; Discord: unused).
    default_topic: str = ""


@dataclass(frozen=True)
class ScheduleTurnScope:
    """What the owning interface carries for one turn."""

    origin: ScheduleOrigin


# This turn's schedule scope: set by the owning interface around its turn,
# read by the tool at call time.
schedule_turn_scope: ContextVar[ScheduleTurnScope | None] = ContextVar(
    "spctr_schedules_turn_scope", default=None
)


class ScheduleTool:
    """The ``schedule`` tool callable, registered on the agent's registry.

    ``description`` and ``input_schema`` are per-instance, built from the
    ``texts`` model (spc-1097); register the instance attributes, not
    class attributes.
    """

    name = "schedule"

    def __init__(
        self,
        store: ScheduleStore,
        timezone: str,
        agent_name: str,
        origin_getter: Any,
        clock: Any = None,
        engine: Any = None,
        texts: ScheduleTextsConfig | None = None,
    ) -> None:
        self._schedule_store = store
        self._timezone = ZoneInfo(timezone)
        self._name = agent_name
        self._get_origin = origin_getter
        self._now_fn = clock or (lambda: datetime.now(tz=ZoneInfo("UTC")))
        self._engine = engine
        self._texts = texts if texts is not None else _default_texts()
        # FR31 list rows fill ``last_status`` from the row, falling back
        # to the configurable never-run label.
        self.description = self._texts.tool_description
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {"enum": ["create", "list", "cancel", "run_now"]},
                "cron": {
                    "type": "string",
                    "description": self._texts.cron_description,
                },
                "type": {"enum": ["command", "agent"]},
                "command": {"type": "string"},
                "cwd": {"type": "string"},
                "instruction": {"type": "string"},
                "agent": {
                    "type": "string",
                    "description": self._texts.agent_description,
                },
                "channel": {
                    "type": "string",
                    "description": self._texts.channel_description,
                },
                "topic": {
                    "type": "string",
                    "description": self._texts.topic_description,
                },
                "label": {
                    "type": "string",
                    "description": self._texts.label_description,
                },
                "id": {"type": "string"},
            },
            "required": ["action"],
        }
        # FR10: the owning interface registers its channel resolver here
        # (spc-186). The Runtime constructs this tool before any interface
        # exists, so the seam is post-construction, mirroring how the
        # engine is wired.
        self._schedule_channel_resolver: Any = None
        # FR47: the owning interface's ``(platform, channel_id) -> name``
        # lookup, so ``list`` can name a stored destination ID ("" when
        # it cannot). Same post-construction seam as the resolver.
        self._schedule_channel_namer: Any = None

    def set_channel_namer(self, namer: Any) -> None:
        """Register the owning interface's channel-ID -> name lookup (FR47)."""
        self._schedule_channel_namer = namer

    def set_channel_resolver(self, resolver: Any) -> None:
        """Register the owning interface's named-channel resolver (FR10/FR11).

        ``resolver`` is ``async (platform, name[, scope]) -> list`` of
        matches, each carrying ``channel_id`` and ``channel_name`` (the
        FR10 match shape). A resolver marked ``spctr_platform ==
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
        topic: str | None = None,
    ) -> str:
        scoped = schedule_turn_scope.get()
        origin = scoped.origin if scoped is not None else self._get_origin()
        if origin is None:
            return self._texts.need_interface_error

        if action == "create":
            return await self._create(
                cron, type, command, cwd, instruction, agent, channel, label, origin,
                topic,
            )
        if action == "cancel":
            return await self._cancel(id, origin)
        if action == "run_now":
            return await self._run_now(id, origin)
        if action == "list":
            return await self._list(origin)
        return self._texts.unknown_action_error.format(action=action)

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
        topic: str | None = None,
    ) -> str:
        # FR25 validation
        if not cron or not cron.strip():
            return self._texts.cron_required_error
        try:
            spec = parse_cron(cron)
        except CronParseError as exc:
            return self._texts.invalid_cron_error.format(reason=exc)

        if not payload_type:
            return self._texts.type_required_error
        if payload_type not in ("command", "agent"):
            return self._texts.unknown_payload_type_error.format(type=payload_type)

        # Validate payload content
        if payload_type == "command":
            if not command or not command.strip():
                return self._texts.command_required_error
            if len(command) > COMMAND_MAX:
                return self._texts.command_too_long_error.format(max=COMMAND_MAX)
            payload_json = json.dumps({"command": command, "cwd": cwd})
        else:
            if not instruction or not instruction.strip():
                return self._texts.instruction_required_error
            if len(instruction) > AGENT_INSTRUCTION_MAX:
                return self._texts.instruction_too_long_error.format(
                    max=AGENT_INSTRUCTION_MAX
                )
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
        # FR47: a Zulip schedule records its post topic, the given one or
        # the origin's default; other platforms have no topic.
        dest_topic = ""
        if origin.platform == "zulip":
            dest_topic = (topic or "").strip() or origin.default_topic

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
            topic=dest_topic,
        )
        new_id = await self._schedule_store.create_async(row)
        logger.info("schedule create: agent=%s id=%s", self._name, new_id)

        # FR12-style confirmation
        next_local = next_run.astimezone(self._timezone)
        next_str = next_local.strftime("%a %H:%M %Z")
        payload_summary = (command or instruction or "")[:40]
        return self._texts.created_format.format(
            id=new_id,
            next=next_str,
            destination=self._destination(dest_channel_name, dest_topic),
            payload_type=payload_type,
            summary=payload_summary,
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
                return self._texts.cross_platform_error
            name = rest.strip()
        if self._schedule_channel_resolver is None:
            return self._texts.no_resolver_error
        # FR10: a Discord resolver scopes its match to the origin guild;
        # it is the only resolver that takes a scope, and the reminder
        # turn scope is what carries the guild (spc-141).
        if getattr(self._schedule_channel_resolver, "spctr_platform", None) == "discord":
            from spctr.reminders import tool as reminders_tool

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
                "%s: resolver returned %r", self._texts.resolver_unusable_error, matches[0]
            )
            return self._texts.resolver_unusable_error
        if not matches:
            return self._texts.channel_not_found_error.format(channel=channel)
        if len(matches) > 1:
            return self._texts.channel_ambiguous_error.format(channel=channel)
        return matches[0]

    async def _cancel(self, id: str | None, origin: ScheduleOrigin) -> str:
        """FR32: cancel removes the schedule."""
        if not id:
            return self._texts.id_required_cancel_error
        row = await self._schedule_store.get_async(id)
        if row is None or row.agent != origin.agent:
            return self._texts.schedule_not_found_error.format(id=id)
        if row.status == "cancelled":
            return self._texts.already_cancelled_error.format(id=id)
        if not await self._schedule_store.cancel_async(id):
            return self._texts.cancel_failed_error.format(id=id)
        logger.info("schedule cancel: agent=%s id=%s", self._name, id)
        return self._texts.cancelled_message.format(id=id)

    async def _run_now(self, id: str | None, origin: ScheduleOrigin) -> str:
        """FR33: run_now executes immediately through the normal path."""
        if not id:
            return self._texts.id_required_run_now_error
        row = await self._schedule_store.get_async(id)
        if row is None or row.agent != origin.agent:
            return self._texts.schedule_not_found_error.format(id=id)
        if row.status == "cancelled":
            return self._texts.cancelled_schedule_error.format(id=id)
        if self._engine is None:
            return self._texts.engine_unavailable_error
        await self._engine.execute_now(row)
        logger.info("schedule run_now: agent=%s id=%s", self._name, id)
        return self._texts.executed_message.format(id=id)

    async def _list(self, origin: ScheduleOrigin) -> str:
        """FR31: this agent's active schedules sorted by next run."""
        rows = await self._schedule_store.list_for_agent_async(origin.agent)
        if not rows:
            return self._texts.empty_list_message
        lines = []
        for s in rows:
            next_local = s.next_run_at.astimezone(self._timezone)
            next_str = next_local.strftime("%a %Y-%m-%d %H:%M %Z")
            payload = json.loads(s.payload)
            summary = (payload.get("command") or payload.get("instruction") or "")[:40]
            lines.append(
                self._texts.list_row_format.format(
                    id=s.id,
                    next=next_str,
                    cron=s.cron,
                    # The channel *name* resolves through the interface
                    # seam (spc-996); until a file has been through it,
                    # show the stored destination string (the resolved ID).
                    destination=self._destination(self._channel_name(s), s.topic),
                    payload_type=s.payload_type,
                    summary=summary,
                    last_status=s.last_status or self._texts.never_run_label,
                )
            )
        return "\n".join(lines)

    def _channel_name(self, row: Schedule) -> str:
        """The destination's display name: the row's, the owning
        interface's lookup (FR47), else the stored destination string."""
        if row.dest_channel_name:
            return row.dest_channel_name
        if self._schedule_channel_namer is not None:
            name = self._schedule_channel_namer(row.platform, row.dest_channel_id)
            if name:
                return str(name)
        return row.dest_channel_id

    def _destination(self, channel_name: str, topic: str) -> str:
        """``destination_format``, or ``destination_topic_format`` when a
        topic is set (Zulip's ``#stream > topic``)."""
        if topic:
            return self._texts.destination_topic_format.format(
                channel=channel_name, topic=topic
            )
        return self._texts.destination_format.format(channel=channel_name)
