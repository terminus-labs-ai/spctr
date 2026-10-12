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
- A triggered topic that this process does not yet hold in a live Session is
  probed for prior content. Any populated topic — including bot-only alerts —
  is refused with an explicit unsupported-history reply and NO model/tool work
  until complete history restoration ships (spc-120 slice). A probe failure
  likewise blocks the model turn with an explicit error.
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
        # Frame ID -> the live Session this process owns for that topic.
        self._sessions: dict[str, Any] = {}
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
        self._running = True
        logger.info(
            "Zulip interface started: origin=%s queue registered",
            self._origin,
        )

    async def stop(self) -> None:
        """Close every interface-owned Session, then the transport."""
        self._running = False
        for session in self._sessions.values():
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
                self._last_event_id = max(
                    self._last_event_id, int(event.get("id", self._last_event_id))
                )
                if event.get("type") != "message":
                    # Topic rename/move/resolve etc. never start work.
                    continue
                message = event.get("message")
                if isinstance(message, dict):
                    await self._handle_message(message)

    # ---- message handling ----------------------------------------------------

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Route one Zulip message event; ignored events do no model work."""
        channel_id = message.get("channel_id")
        sender_id = message.get("sender_id")
        if not isinstance(channel_id, int) or not isinstance(sender_id, int):
            return
        sender_is_bot = bool(
            (message.get("sender_type") or "").lower() == "bot"
        )
        if not is_configured_no_area_channel(
            channel_id=channel_id,
            sender_id=sender_id,
            sender_is_bot=sender_is_bot,
            config=self._config,
        ):
            # Bot, disallowed user, or unconfigured channel: no model work.
            return

        topic = str(message.get("topic", ""))
        frame_id = topic_frame_id(
            self._agent.config.agent.name, self._origin, channel_id, topic
        )
        async with self._turn_lock:
            # Same-topic concurrency: one Session per topic; the lock above
            # serializes model turns globally, so queued follow-ups in one
            # topic keep their order and distinct topics stay isolated.
            session = self._sessions.get(frame_id)
            if session is None:
                session = await self._ensure_session(frame_id, channel_id, topic)
                if session is not None:
                    self._sessions[frame_id] = session
                else:
                    return  # refused or probe-failed; reply already sent
            await self._run_turn(session, message)

    async def _ensure_session(
        self, frame_id: str, channel_id: int, topic: str
    ) -> Any | None:
        """Probe cold-topic history and create the Session for a new topic.

        Called only for topics this process does not yet own. A topic that
        already holds ANY messages — including bot-only alerts — is refused
        with an explicit reply and no model work until history restoration
        ships; a probe failure blocks the turn the same way. Returns None
        when the turn was blocked (the reply is sent synchronously here,
        while still holding the turn lock).
        """
        try:
            probe = self._client.get_topic_messages(channel_id, topic)
            messages = (await probe).get("messages")
            populated = isinstance(messages, list) and len(messages) > 0
        except (ZulipError, KeyError, TypeError) as exc:
            # Probe failure blocks model work: explicit reply, no turn.
            logger.error(
                "Topic history probe failed for channel %s: %s", channel_id, exc
            )
            asyncio.get_running_loop().create_task(
                self._send_reply(channel_id, topic, _PROBE_FAILURE_TEXT)
            )
            return None
        if populated:
            logger.info(
                "Populated cold topic (channel %s, topic %r): refusing "
                "without history restoration",
                channel_id, topic,
            )
            asyncio.get_running_loop().create_task(
                self._send_reply(channel_id, topic, _REFUSAL_TEXT)
            )
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
        return session

    async def _run_turn(self, session: Any, message: dict[str, Any]) -> None:
        """Run one admitted message through its topic's Session."""
        try:
            result = await session.handle_user_message(
                str(message.get("content", ""))
            )
        except Exception as exc:
            logger.error("Zulip model turn failed: %s", exc)
            return
        if not result.text:
            return
        await self._send_reply(
            int(message["channel_id"]), str(message.get("topic", "")), result.text
        )

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
