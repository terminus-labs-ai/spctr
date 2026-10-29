"""ZulipInterface — the Interface protocol implementation for Zulip chat.

Built on the accepted spc-123/124 seams; spc-120 added area routing and
whole-config reload:

- One independent `Session` per (channel, exact topic), created through the
  public `Agent.create_session()` API and owned by this interface. The frame
  identity carries the agent name, the resolved server origin, the channel ID
  and the exact topic — a topic name alone is never an identity.
- Startup loads credentials, validates the routing mapping's shape and
  registers the event queue. It performs no history fetch and no model work:
  a topic that already holds messages when the process starts is only
  discovered lazily, on the first user message that triggers in it.
- Every inbound message re-reads the whole resolved config through a
  ``ZulipConfigView`` (one reload at the single entry point, last-good
  retained on a malformed edit), hands the same single read to
  ``Agent.apply_config``, and routes against the slice it returned. The
  ``zuliprc`` path and ``api_url`` override are pinned for the process:
  the transport, the registered queue and the Session identities are built
  from the startup values.
- Admission and areas follow Discord (spc-1012): ``channels`` lists the
  admitted channel IDs (empty = every channel the queue delivers; the
  queue registers with ``all_public_streams``), and the area is the
  ``channel_areas`` override or else the channel name derived by the
  shared Discord helper. No area is ``RunContext(area="")``; routing never
  falls back to the working directory.
- Bot identity comes from real server metadata: the registration response's
  ``realm_users`` (and ``realm_bots``) arrays carry ``user_id``/``is_bot``
  per https://zulip.com/api/register-queue; a message event never carries an
  invented bot flag.
- A triggered cold topic is probed through the topic history API
  (https://zulip.com/api/get-messages, narrow JSON-encoded). The current
  trigger and eligible siblings from the same event batch are already in
  that fetch, so their message IDs are excluded; ANY other message — including bot-only alerts — refuses the
  topic with an explicit unsupported-history reply and NO model/tool work
  until complete history restoration ships. Fresh messages this process
  already admitted in-process never make a new topic look populated. A probe
  failure or malformed answer blocks the model turn with an explicit error.
- Humans the ``users`` list admits (empty = any human) in admitted
  channels trigger turns; bots and the bot's own account (registration
  ``realm_users``/``user_id``) never do, and unadmitted channels are
  ignored without model work. With ``mention_only`` a topic with no active
  Session starts only on a mention of the bot (spc-1013).
- An admitted turn posts ``thinking_message`` in its topic before the
  model call, streams through ``Session.stream_message``, and the answer
  (or ``error_message``) replaces the placeholder by edit, chunked past
  ``max_message_length``; failed posts or edits fall back to new messages
  (spc-1014).
- While the turn runs the placeholder shows live progress (spc-1015): a
  templated line per tool start and result, then the streamed text so far
  plus ``...``, throttled by ``edit_stream_interval`` and tailed to
  ``max_message_length``. A pending progress edit is cancelled before the
  final edit, so progress never overwrites the answer.
- All model turns are serialized globally (one Miranda turn at a time);
  concurrent messages in the same topic share one Session and keep ordered
  history; distinct topics are isolated.
- `stop()` closes every interface-owned Session and the transport.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import urllib.parse
from typing import TYPE_CHECKING, Any, Callable

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import AgentTextDelta, AgentToolResult, AgentToolStart
from sr2_spectre.interfaces.common import chunk_message, tail_for_stream
from sr2_spectre.interfaces.zulip.client import (
    ZulipClient,
    ZulipError,
    load_zulip_credentials,
)
from sr2_spectre.interfaces.zulip.config import (
    ZulipConfig,
    bot_ids_from_registration,
    is_configured_no_area_channel,
    mentions_bot,
    resolve_channel_area,
    validate_no_area_only_routing,
)
from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView
from sr2_spectre.reminders import (
    ChannelMatch,
    DeliveryLoop,
    ReminderOrigin,
)

if TYPE_CHECKING:
    from sr2_spectre.config_source import SpectreConfigSource

logger = logging.getLogger(__name__)

# Sent as the explicit reply for a topic this cut cannot restore yet.
_REFUSAL_TEXT = (
    "This topic already contains messages I have no saved conversation for, "
    "and topic history restoration is not available in this build yet. "
    "I did not process the earlier content. Start a new topic for a fresh "
    "conversation."
)

_PROBE_FAILURE_TEXT = (
    "I could not check the earlier messages in this topic, so I will not "
    "answer without its history. Please try again later."
)


_TOOL_PLACEHOLDER = re.compile(r"\{(name|input|content)\}")


def render_tool_line(
    template: str, name: str, value: str, max_chars: int
) -> str:
    """Fill one tool-line template (spc-1015).

    *value* (a tool's ``str(input)`` or its result content) is cut to
    *max_chars* plus ``...`` and fills both ``{input}`` and ``{content}``;
    ``{name}`` is the tool name. One substitution pass, so a value that
    itself contains ``{name}`` is never re-filled.
    """
    if len(value) > max_chars:
        value = value[:max_chars] + "..."
    values = {"name": name, "input": value, "content": value}
    return _TOOL_PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


class _ProgressView:
    """Live progress of one turn's placeholder (spc-1015, Discord parity).

    Tool lines accumulate and the streamed text is remembered; every render
    shows the tool lines, a blank line, then the text so far plus ``...``
    (``thinking_message`` before any text), tailed to
    ``max_message_length``. Edits run as one tracked background task so the
    stream never waits on the transport; a newer render supersedes a
    pending one, and :meth:`cancel` drops it before the final edit.
    """

    def __init__(
        self,
        client: Any,
        placeholder_id: Any,
        config: ZulipConfig,
        clock: Callable[[], float],
    ) -> None:
        self._client = client
        self._placeholder_id = placeholder_id
        self._config = config
        self._now = clock
        self._lines: list[str] = []
        self._text = ""
        self._last_text_edit: float | None = None
        self._pending: asyncio.Task | None = None

    def tool_start(self, event: AgentToolStart) -> None:
        if not self._config.tool_embed_enabled:
            return
        self._lines.append(render_tool_line(
            self._config.tool_start_line, event.name, str(event.input),
            self._config.tool_line_max_chars,
        ))
        self._render()

    def tool_result(self, event: AgentToolResult) -> None:
        if not self._config.tool_embed_enabled:
            return
        template = (
            self._config.tool_result_error_line if event.is_error
            else self._config.tool_result_ok_line
        )
        self._lines.append(render_tool_line(
            template, event.name or "tool",
            str(event.content).replace("\n", "|"),
            self._config.tool_line_max_chars,
        ))
        self._render()

    def text(self, text_so_far: str) -> None:
        # Always remember the latest text so a tool render between
        # throttled edits shows it too.
        self._text = text_so_far
        interval = self._config.edit_stream_interval
        if interval <= 0:
            return
        now = self._now()
        if self._last_text_edit is None or now - self._last_text_edit >= interval:
            self._last_text_edit = now
            self._render()

    async def cancel(self) -> None:
        """Drop a pending progress edit and wait until it has stopped."""
        pending, self._pending = self._pending, None
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.wait([pending])

    def _render(self) -> None:
        if self._placeholder_id is None:
            return
        parts: list[str] = []
        if self._lines:
            parts.append("\n".join(self._lines))
        parts.append(
            self._text + "..." if self._text else self._config.thinking_message
        )
        content = tail_for_stream(
            "\n\n".join(parts), self._config.max_message_length
        )
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
        self._pending = asyncio.ensure_future(self._edit(content))

    async def _edit(self, content: str) -> None:
        try:
            await self._client.update_message(self._placeholder_id, content)
        except ZulipError as exc:
            logger.error("Zulip progress edit failed: %s", exc)


def topic_frame_id(agent_name: str, origin: str, channel_id: int, topic: str) -> str:
    """Build the Session frame identity for one Zulip topic.

    The identity includes the agent, the resolved canonical server origin,
    the integer channel ID and the URL-escaped exact topic, so two topics can
    never collide on a bare name and the server is part of the identity.
    """
    escaped = urllib.parse.quote(topic, safe="")
    return f"zulip:{agent_name}:{origin}#{channel_id}#{escaped}"


class ZulipInterface:
    """Zulip chat interface for SR2 Spectre (area routing + live reload)."""

    name = "zulip"

    def __init__(
        self,
        config: ZulipConfig | None = None,
        config_source: "SpectreConfigSource | None" = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Args:
            config: A fixed zulip block, frozen for the life of the process.
                Ignored when config_source is given.
            config_source: The live whole-config source (a
                ``SpectreConfigSource`` from the CLI). The adapter re-reads
                it once per inbound message through a ``ZulipConfigView``
                and hands the same single read to ``Agent.apply_config``,
                so routing and allowlist edits — and the agent's own
                models/endpoints — apply on the next message with last-good
                fallback. When omitted, *config* is frozen for the life of
                the process.
            clock: Monotonic seconds for the progress-edit throttle
                (default ``time.monotonic``); a test seam.
        """
        self._progress_clock = clock or time.monotonic
        self._view = (
            ZulipConfigView(config_source)
            if config_source is not None
            else None
        )
        self._config: ZulipConfig | None = config
        if self._view is not None:
            self._config = self._view.current
        if config is not None:
            # The startup block pins the transport, the registered queue and
            # the Session identities for the life of the process, even when
            # the rest of the slice reloads.
            self._startup_config = config
        elif self._config is not None:
            self._startup_config = self._config
        else:
            raise ValueError(
                "ZulipInterface needs a zulip config or a config source "
                "whose current config carries a zulip block."
            )
        self._agent: Any = None
        self._credentials: Any = None
        self._client: ZulipClient | None = None
        self._queue_id: str = ""
        self._last_event_id: int = -1
        # Frame ID -> (live Session, per-topic history reserved for the
        # history-restoration slice) for the topics this process owns.
        self._sessions: dict[str, tuple[Any, list[Any]]] = {}
        # Message IDs of the triggers this process ADMITTED to a Session;
        # they appear in a later topic fetch (a trigger "already appears in
        # fetched topic history") without making the new topic look
        # populated. Refused triggers never join: they stay prior content.
        self._admitted_ids: set[int] = set()
        # Bot user IDs from registration realm metadata (see _read_bot_ids).
        self._bot_ids: set[int] = set()
        # The bot's own account (registration ``user_id``/``full_name``):
        # never a trigger, and the name a mention_only message must mention.
        self._bot_user_id: int | None = None
        self._bot_full_name: str | None = None
        # Global model-turn serialization: one Miranda turn at a time.
        self._turn_lock = asyncio.Lock()
        # Reminders (spc-139, FR6): delivery loop + the per-turn origin.
        # The origin is recorded per turn (below), never shared globally.
        self._reminders_loop: Any = None
        # spc-143: the Runtime reload stamp the live loop was started
        # under, or None while no loop runs. The per-message sync starts,
        # stops or restarts the loop when the Runtime's stamp differs.
        self._reminders_stamp: int | None = None
        self._reminders_start_task: Any = None
        # Channel name → ID from the latest registration snapshot
        # (spc-140, FR10); empty when the snapshot is unavailable or
        # malformed, and a name then lives only in a later refresh.
        self._channel_names: dict[str, int] = {}
        self._running = False

    # ---- Interface protocol --------------------------------------------------

    async def start(self, agent: Any) -> None:
        """Validate config, load credentials, register the event queue.

        No history is fetched and no model request is made at startup:
        populated topics are only discovered when a user message triggers in
        them, and are refused until history restoration ships.
        """
        validate_no_area_only_routing(self._config)
        self._agent = agent
        self._credentials = load_zulip_credentials(self._config.zuliprc)
        self._client = ZulipClient(
            credentials=self._credentials, config=self._config
        )
        registration = await self._client.register()
        self._queue_id: str = str(registration["queue_id"])
        self._last_event_id: int = int(registration["last_event_id"])
        self._bot_ids = self._read_bot_ids(registration)
        own_id = registration.get("user_id")
        self._bot_user_id = own_id if isinstance(own_id, int) else None
        full_name = registration.get("full_name")
        self._bot_full_name = full_name if isinstance(full_name, str) else None
        self._channel_names = self._read_channel_names(registration)
        self._wire_reminder_channel_lookup(agent)
        self._start_reminders(agent)
        self._running = True
        logger.info(
            "Zulip interface started: origin=%s queue registered bot=%s",
            self._origin,
            self._credentials.email,
        )

    def _read_bot_ids(self, registration: dict[str, Any]) -> set[int]:
        """Bot user IDs from registration realm metadata.

        The shared reader (spc-992): both Zulip interface modes derive bot
        identity from the same register-queue metadata, never from a flag a
        message event carries.
        """
        return bot_ids_from_registration(registration)

    async def stop(self) -> None:
        """Close every interface-owned Session, then the transport."""
        self._running = False
        await self._stop_reminders()
        for session, _seen in self._sessions.values():
            session.close()
        self._sessions.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---- reminders (spc-139, FR6/FR22) --------------------------------------

    def _read_channel_names(self, registration: dict[str, Any]) -> dict[str, int]:
        """Name → channel ID from the registration snapshot (spc-140, FR10).

        Per the register-queue contract the snapshot is complete and
        authoritative at startup: ``streams`` carries every public channel.
        A missing or malformed array is tolerated — named-channel
        resolution then finds no match and the tool errors, which stores
        nothing.
        """
        names: dict[str, int] = {}
        for entry in registration.get("streams") or []:
            if isinstance(entry, dict):
                name = entry.get("name")
                channel_id = entry.get("stream_id")
                if isinstance(name, str) and isinstance(channel_id, int):
                    names[name.lower()] = channel_id
        return names

    def _wire_reminder_channel_lookup(self, agent: Any) -> None:
        """Hand this interface's named-channel resolver to the Runtime.

        The Runtime builds the tool before any interface exists (FR6); the
        platform lookup is interface knowledge, so the owning Zulip
        interface registers its resolver here once at start (spc-140,
        FR10/FR11) through the Runtime's public seam
        (``set_reminders_channel_resolver``). The seam is the only
        hand-off: the Runtime passes the resolver into every ReminderTool
        it constructs — the startup seat and each live-reload re-seat
        (spc-143) alike — and rebuilds the tool already seated through
        that same construction site (spc-166).

        The interface never reads or writes ReminderTool state (spc-159):
        the constructor is a resolver's only way into a tool.
        """
        registrar = getattr(agent, "set_reminders_channel_resolver", None)
        if registrar is not None:
            registrar(self._resolve_channel)

    async def _resolve_channel(self, platform: str, name: str) -> list[Any]:
        """FR10: exact case-insensitive Zulip channel match by name.

        ``can_send`` (the server's per-channel post permission) filters
        channels the bot cannot post to. A failed lookup answers "no
        match": the tool reports not found and stores nothing.

        ``spectre_platform`` marks this callable as the Zulip platform's
        resolver for the Runtime seam (spc-159): the seam only accepts a
        resolver that names the platform it resolves on (FR11).
        """
        if platform != "zulip" or self._client is None:
            return []
        try:
            response = await self._client.get_streams()
        except ZulipError as exc:
            logger.error("Zulip channel lookup failed: %s", exc)
            return []
        matches = []
        for entry in response.get("streams") or []:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("name"), str)
                and isinstance(entry.get("stream_id"), int)
                and entry["name"].lower() == name.lower()
                and entry.get("can_send", True)
            ):
                matches.append(
                    ChannelMatch(
                        channel_id=entry["stream_id"],
                        channel_name=entry["name"],
                    )
                )
        return matches

    # The seam's platform marker (spc-159): set on the underlying
    # function, so every bound ``interface._resolve_channel`` carries it.
    _resolve_channel.spectre_platform = "zulip"

    def _channel_display_name(self, channel_id: int) -> str:
        """Human name for an ID (origin plumbing, FR12); ID if unknown."""
        for name, cid in self._channel_names.items():
            if cid == channel_id:
                return name
        return str(channel_id)

    def _start_reminders(self, agent: Any) -> None:
        """Start the delivery loop when ``agent.reminders.enabled`` is on.

        The loop claims due rows from the Runtime's store and sends them
        through the interface's own transport (the Zulip client). Disabled
        config registers no tool (Runtime) and starts no loop (here) —
        AC3. The reload stamp recorded here is what ``_sync_reminders``
        compares against on later messages (spc-143).
        """
        reminders = getattr(agent, "reminders", None)
        if reminders is None:
            return

        async def send(channel_id: str, topic: str, content: str) -> None:
            await self._send_reminder_reply(int(channel_id), topic, content)

        loop = DeliveryLoop(
            reminders.store,
            send=send,
            poll_seconds=reminders.poll_seconds,
            tz=reminders.tz,
            agent_name=reminders.agent_name,
            # spc-176: this loop serves only Zulip — one agent's Discord
            # loop shares the store and must not touch these rows.
            platform="zulip",
        )
        self._reminders_loop = loop
        self._reminders_stamp = getattr(agent, "reminders_stamped", 0)
        self._reminders_start_task = asyncio.ensure_future(loop.start())

    def _runtime_of(self, agent: Any) -> Any:
        return getattr(agent, "_runtime", None)

    async def _stop_reminders(self) -> None:
        """End the delivery loop; pending rows are left intact (FR22)."""
        loop = self._reminders_loop
        self._reminders_loop = None
        self._reminders_stamp = None
        if loop is None:
            return
        try:
            await loop.stop()
        except asyncio.CancelledError:
            pass

    async def _sync_reminders(self) -> None:
        """Re-seat the delivery loop after a live config reload (spc-143).

        Runs once per message, after the per-message ``apply_config``
        (inside the turn lock, so restarts serialize with turns). Steady
        state costs one int compare: no reload since the loop started (or
        no loop and none since) leaves everything untouched (R3). Once the
        Runtime's stamp moves, re-assert from the current handle — stop
        any running loop and start again only if the handle exists, which
        covers all three edges with one path: enable starts (R1), disable
        stops (pending rows intact, R2), settings change restarts on the
        fresh settings (R4).
        """
        runtime = self._runtime_of(self._agent)
        stamp = getattr(runtime, "reminders_stamped", None) if runtime else None
        if self._reminders_stamp is not None:
            if stamp == self._reminders_stamp:
                return
            await self._stop_reminders()
        handle = getattr(self._agent, "reminders", None)
        if self._reminders_loop is None and handle is not None:
            self._start_reminders(self._agent)

    async def run(self, agent: Any) -> None:
        """Poll the Zulip event queue until stopped."""
        while self._running:
            try:
                response = await self._client.get_events(
                    self._queue_id, self._last_event_id
                )
            except ZulipError as exc:
                # Transport failure: log and back off; never model work.
                logger.error("Zulip event poll failed: %s", exc)
                await asyncio.sleep(2.0)
                continue
            events = response.get("events")
            if not isinstance(events, list):
                logger.error("Zulip event poll returned no events array")
                await asyncio.sleep(2.0)
                continue
            # One whole-config reload per inbound batch — the single entry
            # point for inbound traffic (docs/INTERFACE-DEV-GUIDE.md § Live
            # config reload): every message in this batch is admitted and
            # routed against this one read, and each topic's Session runs
            # its turn under this read's mapped area — existing Sessions
            # included (spec ruling 2026-10-16).
            config = self._reload_config()
            # Snapshot eligible triggers before processing the batch: the history
            # probe can already see later siblings from this same response.
            batch_ids = {
                message["id"]
                for event in events
                if isinstance(event, dict) and event.get("type") == "message"
                and isinstance(message := event.get("message"), dict)
                and self._is_allowed_message(message, config)
            }
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_id = event.get("id")
                if isinstance(event_id, int):
                    self._last_event_id = max(self._last_event_id, event_id)
                if event.get("type") != "message":
                    # Topic rename/move/resolve etc. never start work.
                    continue
                message = event.get("message")
                if isinstance(message, dict):
                    await self._handle_message(message, batch_ids, config)

    # ---- message handling ----------------------------------------------------

    def _reload_config(self) -> ZulipConfig:
        """One whole-config reload at the single entry point for inbound traffic.

        The view reload is the process's only re-read: it refreshes the
        Zulip routing slice AND the agent's half of the same load (via
        ``Agent.apply_config``), and a malformed edit keeps the last
        known-good config in force (``LiveConfigSource.reload`` absorbs the
        failure — never wrap it in a crash). The ``zuliprc`` path and
        ``api_url`` override are the adapter's own pinned fields, rebound
        to the startup block on every read: the transport, the registered
        queue and the Session identities were built from those values.
        Without a live source (a fixed config) this is a no-op.
        """
        if self._view is None:
            return self._config
        loaded = self._view.reload()
        # The transport, the registered queue and the Session identities
        # were built from the startup block, so the slice routed against
        # rebinds those two fields to it on every read — the adapter's own
        # pinning, kept out of the view so each reload stays the newly
        # loaded slice (or, on a malformed edit, the source's last-good).
        pinned = loaded.model_copy(
            update={
                "zuliprc": self._startup_config.zuliprc,
                "api_url": self._startup_config.api_url,
            }
        )
        self._config = pinned
        if self._agent is not None:
            self._agent.apply_config(self._view.source.current)
        return pinned

    def _is_allowed_message(
        self, message: dict[str, Any], config: ZulipConfig | None = None
    ) -> bool:
        """Recognize a human trigger in a configured channel on this read.

        ``config`` is the resolved config the batch arrived on; a caller
        that passes none (an existing single-caller convention) reads the
        config in force.
        """
        if config is None:
            config = self._config
        if message.get("type") != "stream":
            return False
        channel_id = message.get("stream_id")
        sender_id = message.get("sender_id")
        if not all(
            isinstance(value, int)
            for value in (channel_id, sender_id, message.get("id"))
        ):
            return False
        return is_configured_no_area_channel(
            channel_id=channel_id,
            sender_id=sender_id,
            sender_is_bot=sender_id in self._bot_ids,
            config=config,
            bot_user_id=self._bot_user_id,
        )

    async def _handle_message(
        self,
        message: dict[str, Any],
        batch_ids: set[int] | None = None,
        config: ZulipConfig | None = None,
        reload: bool = True,
    ) -> None:
        """Route one eligible event without changing its batch order.

        ``config`` is the resolved config this message arrived on (the poll
        loop re-reads once per batch). A direct caller that passes neither
        *config* nor ``reload=False`` reads one here, so every routed
        message reads exactly one config; ``reload=False`` routes against
        the config in force (the accepted pre-routing call convention).
        """
        if config is None and reload:
            config = self._reload_config()
        elif config is None:
            config = self._config
        if not self._is_allowed_message(message, config):
            return
        channel_id = message["stream_id"]
        message_id = message["id"]
        topic = str(message.get("subject", ""))
        # The area comes from this message's own read: the channel_areas
        # override, else the channel name (display_recipient) derived as
        # Discord derives it, else "" — never the working-directory area.
        channel_name = message.get("display_recipient")
        area = resolve_channel_area(
            channel_id=channel_id,
            config=config,
            channel_name=channel_name if isinstance(channel_name, str) else None,
        )
        frame_id = topic_frame_id(
            self._agent.config.agent.name, self._origin, channel_id, topic
        )
        async with self._turn_lock:
            # Same-topic concurrency: one Session per topic; the lock above
            # serializes model turns globally, so queued follow-ups in one
            # topic keep their order and distinct topics stay isolated.
            entry = self._sessions.get(frame_id)
            if entry is None and config.mention_only and not mentions_bot(
                str(message.get("content", "")),
                full_name=self._bot_full_name,
                user_id=self._bot_user_id,
            ):
                # mention_only (Discord parity): a topic with no active
                # Session starts only on a mention of the bot; a topic
                # that already has one needs none. No probe, no reply.
                return
            if entry is None:
                # Admit-and-remember only on success. The poller holds the
                # turn lock across probe and turn. Eligible siblings from this
                # batch are fresh too, but refused IDs are never remembered
                # after the batch ends.
                entry = await self._ensure_session(
                    frame_id, channel_id, topic, message_id,
                    self._admitted_ids | {message_id} | (batch_ids or set()),
                    area=area,
                )
                if entry is not None:
                    self._admitted_ids.add(message_id)
                    self._sessions[frame_id] = entry
            if entry is None:
                return  # refused or probe-failed; reply already sent
            session = entry[0]
            # A routing edit applies to EVERY topic in the channel on its
            # next message, including a topic whose Session already exists
            # (spec ruling 2026-10-16): re-stamp the run context from this
            # batch's own read through the public Session seam — the mapped
            # area, or the explicit no-area "" — never the area the topic
            # was created with, and never a working-directory fallback.
            session.set_run_context(RunContext(
                interface="zulip",
                mode=RunMode.INTERACTIVE,
                source=f"zulip:{channel_id}:{topic}",
                area=area if area is not None else "",
            ))
            await self._run_turn(session, channel_id, topic, message, config)

    async def _ensure_session(
        self,
        frame_id: str,
        channel_id: int,
        topic: str,
        trigger_id: int,
        candidate_ids: set[int],
        area: str | None = None,
    ) -> tuple[Any, list[Any]] | None:
        """Probe cold-topic history and create the Session for a new topic.

        Called only for topics this process does not yet own. The trigger
        message and eligible siblings in the current batch are already inside
        fetched history, so their IDs are excluded. Any other message,
        including bot-only alerts — means the topic is populated and is refused with an
        explicit reply and no model work until history restoration ships.
        Messages this process admitted to a Session earlier
        (``candidate_ids``) are likewise fresh in-process messages, not
        prior content; a refused trigger never joins them and stays prior
        content. A probe failure or malformed answer blocks the turn the
        same way. Returns the (Session, per-topic history) entry on
        success, or None when the turn was blocked (the blocking reply is
        sent here, while still holding the turn lock).

        ``area`` is the channel's mapped area on this message's config read:
        a name is carried into the Session's RunContext for the existing
        resolvers, and ``""`` is the explicit no-area — never collapsed,
        never inferred from the working directory. Omitted (``None``) by a
        caller with no routing read, which behaves as the explicit no-area.
        """
        try:
            response = await self._client.get_topic_messages(channel_id, topic)
            messages = response.get("messages")
            if not isinstance(messages, list):
                raise TypeError("history response has no messages array")
        except (ZulipError, KeyError, TypeError) as exc:
            # Probe failure blocks model work: explicit reply, no turn.
            logger.error(
                "Topic history probe failed for channel %s: %s", channel_id, exc
            )
            await self._send_reply(channel_id, topic, _PROBE_FAILURE_TEXT)
            return None
        # The trigger (and any trigger this process already admitted to a
        # Session) appears in the fetch without being prior content; each is
        # excluded by its unique id. Anything else — a bot alert included —
        # is prior content.
        def _is_prior(entry: Any) -> bool:
            if not isinstance(entry, dict):
                return True
            entry_id = entry.get("id")
            return entry_id != trigger_id and entry_id not in candidate_ids

        if any(_is_prior(entry) for entry in messages):
            logger.info(
                "Populated cold topic (channel %s, topic %r): refusing "
                "without history restoration",
                channel_id, topic,
            )
            await self._send_reply(channel_id, topic, _REFUSAL_TEXT)
            return None
        session = self._agent.create_session(frame_id)
        session.set_run_context(
            RunContext(
                interface="zulip",
                mode=RunMode.INTERACTIVE,
                source=f"zulip:{channel_id}:{topic}",
                # Mapped area from this message's config read; "" stays the
                # explicit no-area so no consumer falls back to the cwd.
                area=area if area is not None else "",
            )
        )
        return session, set()

    async def _run_turn(
        self,
        session: Any,
        channel_id: int,
        topic: str,
        message: dict[str, Any],
        config: ZulipConfig | None = None,
    ) -> None:
        """Run one admitted message through its topic's Session.

        Working indicator (spc-1014): ``thinking_message`` is posted in the
        topic before the model call, the turn runs through
        ``Session.stream_message``, and the answer (or ``error_message`` on
        a model failure) replaces the placeholder by edit. A failed post or
        edit falls back to new messages; no send or edit failure escapes.
        Meanwhile tool events and text deltas update the placeholder
        through a :class:`_ProgressView` (spc-1015), whose pending edit is
        cancelled before the answer lands. ``config`` is the read this message arrived on (default: the
        config in force).
        """
        if config is None:
            config = self._config
        origin = self._reminders_origin_for(channel_id, message)
        runtime = self._runtime_of(self._agent)
        if runtime is not None:
            runtime.set_reminders_origin(origin)
        try:
            # Live reload happens upstream in the per-message
            # handle_user_message path; re-seat the delivery loop here so
            # an enable/settings change lands before this turn's tool
            # could create a row (spc-143). The turn lock keeps restarts
            # serialized with turns.
            await self._sync_reminders()
            placeholder_id = await self._send_reply(
                channel_id, topic, config.thinking_message
            )
            progress = _ProgressView(
                self._client, placeholder_id, config, self._progress_clock
            )
            parts: list[str] = []
            try:
                async for event in session.stream_message(
                    str(message.get("content", ""))
                ):
                    if isinstance(event, AgentTextDelta):
                        parts.append(event.text)
                        progress.text("".join(parts))
                    elif isinstance(event, AgentToolStart):
                        progress.tool_start(event)
                    elif isinstance(event, AgentToolResult):
                        progress.tool_result(event)
            except Exception as exc:
                logger.error("Zulip model turn failed: %s", exc)
                text = config.error_message.replace("{error}", str(exc))
            else:
                text = "".join(parts) or config.empty_response_message
            finally:
                # A stale progress edit must never land on the answer.
                await progress.cancel()
            await self._deliver_answer(
                channel_id, topic, placeholder_id, text, config
            )
        finally:
            if runtime is not None:
                runtime.clear_reminders_origin()

    async def _deliver_answer(
        self,
        channel_id: int,
        topic: str,
        placeholder_id: Any,
        text: str,
        config: ZulipConfig,
    ) -> None:
        """Replace the placeholder with *text*, chunked past the limit.

        The first chunk edits the placeholder and the rest follow as new
        messages in order. With no placeholder, or when the edit fails (for
        example the realm's edit time limit expired), every chunk is sent
        as a new message instead.
        """
        chunks = chunk_message(text, config.max_message_length)
        if placeholder_id is not None:
            try:
                await self._client.update_message(placeholder_id, chunks[0])
            except ZulipError as exc:
                logger.error("Zulip reply edit failed: %s", exc)
            else:
                chunks = chunks[1:]
        for chunk in chunks:
            await self._send_reply(channel_id, topic, chunk)

    def _reminders_origin_for(self, channel_id: int, message: dict[str, Any]) -> ReminderOrigin:
        """The origin of THIS turn, recorded per message.

        Two concurrent turns in different channels must never record each
        other's origin (spec, Delivery seam): the origin is captured from
        this message's own event (channel, topic, sender, mention) and held
        on the Runtime for the duration of exactly this turn — never a
        global, and the turn lock is deliberately not relied on for it.
        """
        sender_id = message.get("sender_id")
        topic = str(message.get("subject", ""))
        return ReminderOrigin(
            agent=self._agent.config.agent.name,
            platform="zulip",
            requester_id=str(sender_id) if sender_id is not None else "",
            requester_mention=f"@{sender_id}" if sender_id is not None else "@user",
            channel_id=str(channel_id),
            channel_name=self._channel_display_name(channel_id),
            topic=topic,
        )

    async def _send_reply(self, channel_id: int, topic: str, content: str) -> Any:
        """Deliver one bot-authored message back to the original topic.

        Returns the new message ID, or None when the send failed (logged).
        """
        try:
            return await self._client.send_message(
                type="stream", to=channel_id, topic=topic, content=content
            )
        except ZulipError as exc:
            logger.error("Zulip reply send failed: %s", exc)
            return None

    async def _send_reminder_reply(
        self, channel_id: int, topic: str, content: str
    ) -> None:
        """Deliver one reminder to its destination channel/topic.

        Raises on transport failure: the delivery loop counts the attempt
        and leaves the row pending for the next poll (spc-139 policy).
        """
        await self._client.send_message(
            type="stream", to=channel_id, topic=topic, content=content
        )

    @property
    def _origin(self) -> str:
        """The resolved server origin requests go to (for identities)."""
        return (self._config.api_url or self._credentials.site).rstrip("/")
