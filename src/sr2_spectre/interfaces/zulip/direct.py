"""ZulipDMInterface — the DM-only Zulip variant (spc-992).

Promoted from the Afterlife character runtime's private channel so a DM-only
bot uses framework code instead of a copy of the admission rules. Shape
differences from the stream/topic ``ZulipInterface`` — and why they are not
omissions:

- No topic machinery at all: DMs have no topics, so there are no cold-topic
  history probes, no topic Sessions and no ``channels`` area routing. One
  Session serves the DM, created through the public ``Agent.create_session``
  API and owned by this interface; its ``RunContext`` carries the explicit
  no-area ``""`` (never ``None``, so no consumer falls back to the working
  directory).
- Admission is the SAME shared core the stream mode calls:
  ``config.admits_direct_message`` wraps ``admits_human_trigger`` (a bot or
  the bot's own account never triggers, ``users`` empty = any human, strict
  integer IDs) and adds only DM-shape checks (private type, integer fields,
  optional peer filter). ``mention_only`` is a stream-mode field and is
  ignored here. Bot identity comes from registration realm metadata, and
  the registration response's own ``user_id`` identifies this bot — no
  extra HTTP round trip.
- Outbound sends go through ``ZulipClient.send_direct`` — the JSON-list
  private destination is encoded there, never here.

Like the stream interface, inbound traffic has one entry point (``run()``)
that re-reads the whole resolved config once per batch through
``ZulipConfigView``, hands the same read to ``Agent.apply_config``, and
routes against the slice it returned; the ``zuliprc`` path and ``api_url``
override stay pinned to the startup block. Model turns are serialized
globally.

The poll/admit/send core (``start``, ``poll``, ``send``,
``should_consider_beat``, ``stop``) is also the public seam an external
runtime drives directly from config — that is how the Afterlife runtime
consumes it — while ``run(agent)`` is the framework-own loop that answers
admitted DMs with real agent turns, so ``--interface zulip`` plus
``mode: direct`` selects a working bot with no external driver.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.interfaces.zulip.client import (
    ZulipClient,
    ZulipError,
    ZulipQueueExpiredError,
    load_zulip_credentials,
)
from sr2_spectre.interfaces.zulip.config import (
    ZulipConfig,
    admits_direct_message,
    bot_ids_from_registration,
    validate_direct_mode,
)
from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView

if TYPE_CHECKING:
    from sr2_spectre.config_source import SpectreConfigSource

logger = logging.getLogger(__name__)


def dm_frame_id(agent_name: str, origin: str, peer_id: int) -> str:
    """The Session frame identity for one DM.

    The identity includes the agent, the resolved canonical server origin
    and the peer's integer user ID — a bare name is never an identity, and
    two realms never collide.
    """
    return f"zulip-dm:{agent_name}:{origin}#{peer_id}"


@dataclass(frozen=True)
class DMessage:
    """One admitted inbound DM."""

    message_id: int
    sender_id: int
    text: str


class ZulipDMInterface:
    """DM-only Zulip interface: private messages with one peer, no areas."""

    name = "zulip"

    def __init__(
        self,
        config: ZulipConfig | None = None,
        config_source: "SpectreConfigSource | None" = None,
        *,
        quiet_seconds: float = 15.0,
    ) -> None:
        """
        Args:
            config: A fixed zulip block (``mode: direct``), frozen for the
                life of the process. Ignored when config_source is given.
            config_source: The live whole-config source (a
                ``SpectreConfigSource`` from the CLI). The adapter re-reads
                it once per inbound batch through a ``ZulipConfigView`` and
                hands the same single read to ``Agent.apply_config``, so
                allowlist edits apply on the next message with last-good
                fallback. When omitted, *config* is frozen.
            quiet_seconds: Quiet period for ``should_consider_beat`` — the
                promoted channel policy that keeps a heartbeat beat from
                firing mid-exchange.

        Raises:
            ValueError: No config source and no config block, or a config
                without an integer ``peer_id`` (direct mode's one
                requirement).
        """
        self._view = (
            ZulipConfigView(config_source)
            if config_source is not None
            else None
        )
        self._config: ZulipConfig | None = config
        if self._view is not None:
            self._config = self._view.current
        if config is not None:
            self._startup_config = config
        elif self._config is not None:
            self._startup_config = self._config
        else:
            raise ValueError(
                "ZulipDMInterface needs a zulip config or a config source "
                "whose current config carries a zulip block."
            )
        validate_direct_mode(self._config)
        self._agent: Any = None
        self._credentials: Any = None
        self._client: ZulipClient | None = None
        self._queue_id: str = ""
        self._last_event_id: int = -1
        # Bot user IDs from registration realm metadata; her own account ID
        # comes from the same response's top-level user_id.
        self._bot_ids: set[int] = set()
        self._bot_user_id: int | None = None
        self._started = False
        # The one Session serving the DM, created on the first admitted
        # message and owned (closed) by this interface.
        self._session: Any = None
        # Global model-turn serialization: one turn at a time.
        self._turn_lock = asyncio.Lock()
        self._quiet_seconds = quiet_seconds
        self._running = False

    # -- lifecycle -------------------------------------------------------------

    async def start(self, agent: Any = None) -> None:
        """Validate the DM block, load credentials, register the event queue.

        No history is fetched and no model request is made at startup. The
        registration response carries this bot's own ``user_id`` (the
        register-queue contract) — the identity a caller used to need an
        extra HTTP call for — alongside the realm bot metadata.
        """
        validate_direct_mode(self._config)
        self._agent = agent
        self._credentials = load_zulip_credentials(self._config.zuliprc)
        self._client = ZulipClient(
            credentials=self._credentials, config=self._config
        )
        await self._register()
        self._started = True
        self._running = True
        logger.info(
            "Zulip DM interface started: origin=%s queue registered bot=%s",
            self._origin,
            self._credentials.email,
        )

    async def _register(self) -> None:
        """Register an event queue and take its snapshot (start and expiry)."""
        registration = await self._client.register()
        self._queue_id = str(registration["queue_id"])
        self._last_event_id = int(registration["last_event_id"])
        self._bot_ids = bot_ids_from_registration(registration)
        own = registration.get("user_id")
        self._bot_user_id = own if isinstance(own, int) else None

    async def stop(self) -> None:
        """Close the DM Session, then the transport."""
        self._running = False
        self._started = False
        if self._session is not None:
            try:
                self._session.close()
            except Exception as exc:  # noqa: BLE001 — teardown never blocks the close
                logger.error("Zulip DM session close failed: %s", exc)
            self._session = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- admission (shared framework core) ---------------------------------------

    def admits(self, message: dict[str, Any]) -> bool:
        """True when this message may trigger a turn (DM, human, allowlist).

        The decision is the shared ``admits_direct_message`` core — the
        same ``admits_human_trigger`` rule the stream interface admits
        through, plus DM-shape checks. The config in force is used.
        """
        return self.admits_with(message, self._config)

    def admits_with(
        self, message: dict[str, Any], config: ZulipConfig
    ) -> bool:
        """Admission against an explicit config read (the per-batch slice)."""
        return admits_direct_message(
            message=message,
            users=config.users,
            bot_ids=self._bot_ids,
            bot_user_id=self._bot_user_id,
            peer_id=config.peer_id,
        )

    # -- inbound -------------------------------------------------------------------

    async def poll(self) -> list[DMessage]:
        """Admitted DMs since the last poll.

        Transport errors propagate (``ZulipError``) — callers own the
        fail-soft policy; ``run`` below logs and backs off like the stream
        interface does. The event cursor advances over every event,
        admitted or not, so ignored events never re-arrive.
        """
        response = await self._client.get_events(
            self._queue_id, self._last_event_id
        )
        events = response.get("events")
        if not isinstance(events, list):
            return []
        return self._admit_batch(events)

    def _admit_batch(self, events: list[Any]) -> list[DMessage]:
        """Advance the cursor over one batch and return its admitted DMs."""
        config = self._config
        admitted: list[DMessage] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = event.get("id")
            if isinstance(event_id, int):
                self._last_event_id = max(self._last_event_id, event_id)
            if event.get("type") != "message":
                # Presence/rename/etc. events never start work.
                continue
            message = event.get("message")
            if not isinstance(message, dict) or not self.admits_with(
                message, config
            ):
                continue
            admitted.append(
                DMessage(
                    message_id=int(message["id"]),
                    sender_id=int(message["sender_id"]),
                    text=str(message.get("content", "")),
                )
            )
        return admitted

    # -- outbound --------------------------------------------------------------------

    @property
    def peer_id(self) -> int | None:
        return self._config.peer_id

    @property
    def bot_user_id(self) -> int | None:
        return self._bot_user_id

    async def send(self, text: str) -> int:
        """Send one DM to the peer; returns the realm message id."""
        peer_id = self._config.peer_id
        if peer_id is None:
            raise RuntimeError("DM channel has no peer to message")
        return await self._client.send_direct(peer_id, text)

    # -- heartbeat policy (promoted from the Afterlife channel, FR-7) ------------------

    def should_consider_beat(self, last_outbound_ts: float | None) -> bool:
        """Quiet-period gate: only beat when the conversation has settled.

        An external timer supplies cadence; this keeps a beat from firing
        mid-exchange (the promoted channel's contract, carried over as-is).
        """
        if last_outbound_ts is None:
            return True
        return (time.monotonic() - last_outbound_ts) >= self._quiet_seconds

    # -- Interface protocol: the framework-own DM loop ---------------------------------

    async def run(self, agent: Any) -> None:
        """Poll the DM queue until stopped, answering admitted messages.

        One whole-config reload per inbound batch (the stream interface's
        convention): every message in the batch is admitted against this
        one read, and the same read feeds ``Agent.apply_config``.
        """
        while self._running:
            try:
                response = await self._client.get_events(
                    self._queue_id, self._last_event_id
                )
            except ZulipQueueExpiredError:
                # The server collected the queue; the old ID never recovers.
                logger.warning("Zulip DM event queue expired; registering a new one")
                try:
                    await self._register()
                except ZulipError as exc:
                    logger.error("Zulip DM queue re-registration failed: %s", exc)
                    await asyncio.sleep(2.0)
                continue
            except ZulipError as exc:
                # Transport failure: log and back off; never model work.
                logger.error("Zulip DM event poll failed: %s", exc)
                await asyncio.sleep(2.0)
                continue
            events = response.get("events")
            if not isinstance(events, list):
                logger.error("Zulip DM event poll returned no events array")
                await asyncio.sleep(2.0)
                continue
            self._reload_config()
            admitted = self._admit_batch(events)
            for message in admitted:
                await self._handle_message(message)

    def _reload_config(self) -> ZulipConfig:
        """One whole-config reload per batch; pinned fields stay startup's.

        The view reload is the process's only re-read: it refreshes the
        Zulip slice AND the agent's half of the same load (via
        ``Agent.apply_config``), and a malformed edit keeps the last
        known-good config in force. The transport, the registered queue and
        the Session identity were built from the startup block, so the
        slice rebinds ``zuliprc``/``api_url`` to it on every read. Without
        a live source this is a no-op.
        """
        if self._view is None:
            return self._config
        loaded = self._view.reload()
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

    async def _handle_message(self, message: DMessage) -> None:
        """Run one admitted DM through the peer's Session and reply to it."""
        async with self._turn_lock:
            session = self._session
            if session is None:
                session = self._ensure_session(message.sender_id)
                self._session = session
                session.set_run_context(
                    RunContext(
                        interface="zulip",
                        mode=RunMode.INTERACTIVE,
                        source=f"zulip-dm:{message.sender_id}",
                        # DMs carry no area: "" is the explicit no-area,
                        # never the None a consumer might fall back on.
                        area="",
                    )
                )
            try:
                result = await session.handle_user_message(message.text)
            except Exception as exc:  # noqa: BLE001 — a failed turn never kills the loop
                logger.error("Zulip DM model turn failed: %s", exc)
                return
            if not result.text:
                return
            await self._send_reply(result.text)

    def _ensure_session(self, peer_id: int) -> Any:
        """Create the interface-owned Session for this DM."""
        frame_id = dm_frame_id(
            self._agent.config.agent.name, self._origin, peer_id
        )
        return self._agent.create_session(frame_id)

    async def _send_reply(self, content: str) -> None:
        """Deliver one bot-authored DM. Send failures log, never raise here."""
        try:
            await self.send(content)
        except ZulipError as exc:
            logger.error("Zulip DM reply send failed: %s", exc)

    @property
    def _origin(self) -> str:
        """The resolved server origin requests go to (for identities)."""
        return (self._config.api_url or self._credentials.site).rstrip("/")
