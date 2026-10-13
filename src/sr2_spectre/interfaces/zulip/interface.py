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
  (https://zulip.com/api/get-messages, narrow JSON-encoded). Every trigger
  this process has already delivered counts as an in-flight fresh message and
  is excluded by message id — the current trigger, and its siblings when the
  server returned several same-topic messages inside ONE events batch, are
  concurrent triggers of that topic's single Session and never prior content
  for each other. ANY other message — including bot-only alerts — refuses the
  topic with an explicit unsupported-history reply and NO model/tool work
  until complete history restoration ships; a trigger refused here stays
  prior content for every later process, which never saw it. A probe failure
  or malformed answer blocks the model turn with an explicit error.
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
        # Frame ID -> (live Session, per-topic history reserved for the
        # history-restoration slice) for the topics this process owns.
        self._sessions: dict[str, tuple[Any, list[Any]]] = {}
        # Message IDs this process DELIVERED as an event, admitted or refused.
        # They appear in a later topic fetch (an in-flight trigger "already
        # appears in fetched topic history") without making a new topic look
        # populated: several fresh same-topic triggers returned inside ONE
        # events batch are concurrent triggers of that topic's single
        # Session, never prior content for each other. A refused trigger
        # stays prior content for every later process, which never saw it.
        self._delivered_ids: set[int] = set()
        # Bot user IDs from registration realm metadata (see _read_bot_ids).
        self._bot_ids: set[int] = set()
        # Global model-turn serialization: one Miranda turn at a time.
        self._turn_lock = asyncio.Lock()
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
        for session, _seen in self._sessions.values():
            session.close()
        self._sessions.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

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
                    await self._handle_message(message)

    # ---- message handling ----------------------------------------------------

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Route one Zulip message event; ignored events do no model work."""
        # Official stream-message shape (get-events contract): ``type``
        # "stream", the channel under ``stream_id``, the exact topic under
        # ``subject`` (legacy name), the sender under ``sender_id``.
        if message.get("type") != "stream":
            return
        channel_id = message.get("stream_id")
        sender_id = message.get("sender_id")
        message_id = message.get("id")
        if (
            not isinstance(channel_id, int)
            or not isinstance(sender_id, int)
            or not isinstance(message_id, int)
        ):
            return
        if not is_configured_no_area_channel(
            channel_id=channel_id,
            sender_id=sender_id,
            sender_is_bot=sender_id in self._bot_ids,
            config=self._config,
        ):
            # Bot, disallowed user, or unconfigured channel: no model work.
            return

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
                # The poller holds the turn lock across probe and turn, so
                # every trigger this process already delivered (this one
                # included) is in the delivered set and none of them is prior
                # content for a cold topic.
                self._delivered_ids.add(message_id)
                entry = await self._ensure_session(
                    frame_id, channel_id, topic, message_id,
                    self._delivered_ids,
                )
                if entry is not None:
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

        Called only for topics this process does not yet own. The fetched
        history already contains this trigger and any sibling trigger the
        server returned in the same batch, so every id in ``candidate_ids``
        (this trigger included) is an in-flight fresh message, not prior
        content; ANY other message — including bot-only alerts — means the
        topic is populated and is refused with an explicit reply and no model
        work until history restoration ships. A refused trigger stays in
        ``candidate_ids`` for this process only, so it remains prior content
        for every later process. A probe failure or malformed answer blocks
        the turn the same way. Returns the (Session, per-topic history) entry
        on success, or None when the turn was blocked (the blocking reply is
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
        # The triggers this process delivered (this one and any batch
        # sibling) appear in the fetch without being prior content; each is
        # excluded by its unique id. Anything else — a bot alert included —
        # is prior content.
        def _is_prior(entry: Any) -> bool:
            if not isinstance(entry, dict):
                return True
            entry_id = entry.get("id")
            r = entry_id != trigger_id and entry_id not in candidate_ids
            import os as _os
            with open("/tmp/spc125/diag2.log", "a") as _fh:
                _fh.write(f"{_os.getpid()} trig={trigger_id} id={entry_id!r} "
                          f"type={type(entry_id).__name__} cand={sorted(candidate_ids)} -> {r}\n")
            return r

        import os as _os, json as _json, time as _t
        with open("/tmp/spc125/diag.log", "a") as _fh:
            _fh.write(f"{_t.monotonic()} pid={_os.getpid()} trigger={trigger_id} "
                      f"candidate={sorted(candidate_ids)} "
                      f"fetched={[(m.get('id') if isinstance(m, dict) else m) for m in messages]} "
                      f"prior={[(m.get('id') if isinstance(m, dict) else m) for m in messages if _is_prior(m)]}\n")
        _prior_flags = [(m.get("id") if isinstance(m, dict) else m, _is_prior(m))
                        for m in messages]
        _any_prior = any(_is_prior(entry) for entry in messages)
        import os as _os2
        with open("/tmp/spc125/diag3.log", "a") as _fh:
            _fh.write(f"CALL trig={trigger_id} flags={_prior_flags} "
                      f"any={_any_prior} any2={any(f for _, f in _prior_flags)}\n")
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
        import os as _o, time as _tm
        def _d(msg):
            with open("/tmp/spc125/diag4.log", "a") as _f2:
                _f2.write(f"{_tm.monotonic()} pid={_o.getpid()} msg={message.get('id')} {msg}\n")
        _d(f"ENTER session={type(session).__name__}")
        try:
            result = await session.handle_user_message(
                str(message.get("content", ""))
            )
        except Exception as exc:
            _d(f"EXC {type(exc).__name__}: {exc}")
            logger.error("Zulip model turn failed: %s", exc)
            return
        _d(f"result text_len={len(result.text or '')}")
        if not result.text:
            return
        await self._send_reply(channel_id, topic, result.text)
        _d("sent")

    async def _send_reply(self, channel_id: int, topic: str, content: str) -> None:
        """Deliver one bot-authored message back to the original topic."""
        try:
            await self._client.send_message(
                type="stream", to=channel_id, topic=topic, content=content
            )
        except ZulipError as exc:
            logger.error("Zulip reply send failed: %s", exc)

    @property
    def _origin(self) -> str:
        """The resolved server origin requests go to (for identities)."""
        return (self._config.api_url or self._credentials.site).rstrip("/")
