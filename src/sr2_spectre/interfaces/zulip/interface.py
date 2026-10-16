"""ZulipInterface — the Interface protocol implementation for Zulip chat.

Splice 1 (spc-125) is a safe no-area chat pilot built on the accepted
spc-123/124 seams:

- One independent `Session` per (channel, exact topic), created through the
  public `Agent.create_session()` API and owned by this interface. The frame
  identity carries the agent name, the resolved server origin, the channel ID
  and the exact topic — a topic name alone is never an identity.
- Startup loads credentials, validates no-area-only routing and registers the
  event queue. It performs no history fetch and no model work: a topic that
  already holds messages when the process starts is only discovered lazily,
  on the first user message that triggers in it.
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
- Only allowed human users in configured no-area channels trigger turns; bot
  messages and unconfigured channels are ignored without model work.
- All model turns are serialized globally (one Miranda turn at a time);
  concurrent messages in the same topic share one Session and keep ordered
  history; distinct topics are isolated.
- `stop()` closes every interface-owned Session and the transport.
"""
from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import Any

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.interfaces.zulip.client import (
    ZulipClient,
    ZulipError,
    load_zulip_credentials,
)
from sr2_spectre.interfaces.zulip.config import (
    ZulipConfig,
    is_configured_no_area_channel,
    validate_no_area_only_routing,
)
from sr2_spectre.reminders import DeliveryLoop, ReminderOrigin

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


def topic_frame_id(agent_name: str, origin: str, channel_id: int, topic: str) -> str:
    """Build the Session frame identity for one Zulip topic.

    The identity includes the agent, the resolved canonical server origin,
    the integer channel ID and the URL-escaped exact topic, so two topics can
    never collide on a bare name and the server is part of the identity.
    """
    escaped = urllib.parse.quote(topic, safe="")
    return f"zulip:{agent_name}:{origin}#{channel_id}#{escaped}"


class ZulipInterface:
    """Zulip chat interface for SR2 Spectre (safe no-area pilot)."""

    name = "zulip"

    def __init__(self, config: ZulipConfig) -> None:
        self._config = config
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
        self._start_reminders(agent)
        self._running = True
        logger.info(
            "Zulip interface started: origin=%s queue registered bot=%s",
            self._origin,
            self._credentials.email,
        )

    def _read_bot_ids(self, registration: dict[str, Any]) -> set[int]:
        """Collect bot user IDs from registration realm metadata.

        Message events carry no bot flag; per the register-queue contract the
        ``realm_users`` array (every account, ``is_bot`` marks bots) plus the
        ``realm_bots`` array are the real server metadata that identifies
        bots by ``user_id``.
        """
        bot_ids: set[int] = set()
        for entry in registration.get("realm_users") or []:
            if isinstance(entry, dict) and entry.get("is_bot") is True:
                user_id = entry.get("user_id")
                if isinstance(user_id, int):
                    bot_ids.add(user_id)
        for entry in registration.get("realm_bots") or []:
            if isinstance(entry, dict):
                bot_id = entry.get("bot_id")
                if isinstance(bot_id, int):
                    bot_ids.add(bot_id)
        return bot_ids

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
            # Snapshot eligible triggers before processing the batch: the history
            # probe can already see later siblings from this same response.
            batch_ids = {
                message["id"]
                for event in events
                if isinstance(event, dict) and event.get("type") == "message"
                and isinstance(message := event.get("message"), dict)
                and self._is_allowed_message(message)
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
                    await self._handle_message(message, batch_ids)

    # ---- message handling ----------------------------------------------------

    def _is_allowed_message(self, message: dict[str, Any]) -> bool:
        """Recognize human triggers eligible for this no-area pilot."""
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
            config=self._config,
        )

    async def _handle_message(
        self, message: dict[str, Any], batch_ids: set[int] | None = None,
    ) -> None:
        """Route one eligible event without changing its batch order."""
        if not self._is_allowed_message(message):
            return
        channel_id = message["stream_id"]
        message_id = message["id"]
        topic = str(message.get("subject", ""))
        frame_id = topic_frame_id(
            self._agent.config.agent.name, self._origin, channel_id, topic
        )
        async with self._turn_lock:
            # Same-topic concurrency: one Session per topic; the lock above
            # serializes model turns globally, so queued follow-ups in one
            # topic keep their order and distinct topics stay isolated.
            entry = self._sessions.get(frame_id)
            if entry is None:
                # Admit-and-remember only on success. The poller holds the
                # turn lock across probe and turn. Eligible siblings from this
                # batch are fresh too, but refused IDs are never remembered
                # after the batch ends.
                entry = await self._ensure_session(
                    frame_id, channel_id, topic, message_id,
                    self._admitted_ids | {message_id} | (batch_ids or set()),
                )
                if entry is not None:
                    self._admitted_ids.add(message_id)
                    self._sessions[frame_id] = entry
            if entry is None:
                return  # refused or probe-failed; reply already sent
            session = entry[0]
            await self._run_turn(session, channel_id, topic, message)

    async def _ensure_session(
        self,
        frame_id: str,
        channel_id: int,
        topic: str,
        trigger_id: int,
        candidate_ids: set[int],
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
                area="",  # explicit no-area: never fall back to cwd
            )
        )
        return session, set()

    async def _run_turn(
        self, session: Any, channel_id: int, topic: str, message: dict[str, Any]
    ) -> None:
        """Run one admitted message through its topic's Session."""
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
            try:
                result = await session.handle_user_message(
                    str(message.get("content", ""))
                )
            except Exception as exc:
                logger.error("Zulip model turn failed: %s", exc)
                return
            if not result.text:
                return
            await self._send_reply(channel_id, topic, result.text)
        finally:
            if runtime is not None:
                runtime.clear_reminders_origin()

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
            channel_name=str(channel_id),
            topic=topic,
        )

    async def _send_reply(self, channel_id: int, topic: str, content: str) -> None:
        """Deliver one bot-authored message back to the original topic."""
        try:
            await self._client.send_message(
                type="stream", to=channel_id, topic=topic, content=content
            )
        except ZulipError as exc:
            logger.error("Zulip reply send failed: %s", exc)

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
