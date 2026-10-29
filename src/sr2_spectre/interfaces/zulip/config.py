"""Zulip interface configuration and the pure admission/routing decisions.

The schema follows the spec's Interfaces section: ``zuliprc`` (credential
path, no inline key), an optional ``api_url`` override (loopback test
endpoints; the zuliprc ``site`` stays the canonical origin), explicit
integer-ID allowlists, and Discord's channel options (spc-1012):
``channels`` is a list of integer channel IDs (empty = every channel the
event queue delivers) and ``channel_areas`` maps a channel ID written as a
string to an area identity (``""`` = explicitly no area). Without an
override the area is the channel name, derived by the one
:func:`sr2_spectre.interfaces.common.derive_area_name` Discord uses; it
never falls back to the working directory.

ROUTING CONTRACT: a nonempty area is valid configuration and routes
through the existing resolvers; no area is always ``""``. The startup
validator (:func:`validate_no_area_only_routing`) only checks shape — live
vault existence is never probed at config time.

INTERFACE MODES (spc-992): ``mode`` selects which Zulip interface the
``--interface zulip`` front door builds. ``"stream"`` (the default, so every
config written before spc-992 parses unchanged) is the stream/topic-shaped
``ZulipInterface``. ``"direct"`` is the DM-only variant
(``ZulipDMInterface``): private messages with ``peer_id`` only, no
topics, no channel admission or area routing. Both modes decide admission through the
ONE :func:`admits_human_trigger` core below, so the security-relevant rule
never exists in two copies.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sr2_spectre.interfaces.common import resolve_area

#: The accepted values of ``ZulipConfig.mode``.
ZULIP_MODES = ("stream", "direct")


class ZulipConfig(BaseModel):
    """Configuration for the Zulip interface.

    Attributes:
        zuliprc: Path to the bot's zuliprc credential file (email, key, site).
        api_url: Optional API origin override. When set (e.g. a loopback test
            endpoint), requests go there instead of the zuliprc site.
        users: Human user IDs whose messages may trigger a turn, in both
            modes (Discord's ``users``, spc-1013). Empty admits any human;
            nonempty admits only the listed IDs. A bot, or the bot's own
            account, never triggers even if its ID appears here (reply-loop
            prevention). Replaces the removed ``allowed_user_ids``, which
            now fails validation.
        mention_only: Stream mode: when true, a message in a topic with no
            active Session triggers only when it mentions the bot
            (:func:`mentions_bot`); a topic with an active Session needs no
            mention. Direct mode ignores it.
        channels: Integer channel IDs admitted in stream mode. Empty admits
            every channel the event queue delivers (every public channel,
            plus private channels the bot is subscribed to); nonempty admits
            only the listed IDs. Direct mode ignores it.
        channel_areas: Channel ID written as a string → area identity.
            ``""`` means explicitly no area. A channel with no key here
            takes its area from its name (:func:`resolve_channel_area`).
            Stream mode only.
        mode: Which Zulip interface variant the config drives — ``"stream"``
            (default; stream/topic chats, area routing) or ``"direct"`` (DMs
            with ``peer_id`` only, no topics and no area routing).
        peer_id: Direct mode: the human user ID this bot exchanges DMs with,
            in both directions. Strict integer, like every other ID here.
            Stream mode ignores it.
    """
    model_config = ConfigDict(strict=True)

    zuliprc: str
    api_url: str | None = None
    users: list[int] = Field(default_factory=list)
    mention_only: bool = False
    channels: list[int] = Field(default_factory=list)
    channel_areas: dict[str, str] = Field(default_factory=dict)
    mode: str = "stream"
    peer_id: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_removed_allowlist(cls, data: Any) -> Any:
        if isinstance(data, dict) and "allowed_user_ids" in data:
            raise ValueError(
                "Zulip `allowed_user_ids` was removed; use `users` instead "
                "(a list of integer user IDs, empty = any human)."
            )
        return data

    @field_validator("mode")
    @classmethod
    def _mode_is_known(cls, value: str) -> str:
        if value not in ZULIP_MODES:
            raise ValueError(
                "Zulip `mode` must be one of "
                f"{', '.join(repr(name) for name in ZULIP_MODES)}; "
                f"got {value!r}."
            )
        return value


def admits_human_trigger(
    *,
    sender_id: int,
    sender_is_bot: bool,
    users,
    bot_user_id: int | None = None,
) -> bool:
    """THE admission rule: a human the ``users`` list admits (spc-992, spc-1013).

    One implementation serves every Zulip interface mode — the stream/topic
    :func:`is_configured_no_area_channel` front door and the DM variant
    :func:`admits_direct_message` both delegate their human/bot decision
    here, so the security-relevant rule cannot drift between them.

    Clauses, in force order:

    1. A bot-authored message (``is_bot`` in the registration's
       ``realm_users``) or a message from the bot's own registration
       ``user_id`` never triggers, whatever ``users`` holds — reply-loop
       prevention never depends on the user filter.
    2. The sender ID must be a real integer: a stringified ID (or no ID) is
       never a match.
    3. An empty ``users`` admits any human; a nonempty one admits only the
       listed integer IDs.
    """
    if sender_is_bot:
        return False
    if not isinstance(sender_id, int):
        return False
    if bot_user_id is not None and sender_id == bot_user_id:
        return False
    if not users:
        return True
    return sender_id in users


def mentions_bot(
    content: str, *, full_name: str | None, user_id: int | None
) -> bool:
    """True when raw Markdown *content* mentions the bot by name (spc-1013).

    Event content is raw Markdown, so a mention is literal text. The bot is
    mentioned by ``@**<full_name>**``, ``@**<full_name>|<user_id>**`` or the
    silent ``@_**…**`` form of either, matched exactly: a wildcard mention
    (``@**all**``, ``@**everyone**``, ...) names no user and never matches,
    and a name that only starts with the bot's name (``@**Mirandas**``) is
    not a mention.
    """
    if not full_name:
        return False
    names = [full_name]
    if user_id is not None:
        names.append(f"{full_name}|{user_id}")
    return any(
        f"{prefix}**{name}**" in content
        for prefix in ("@", "@_")
        for name in names
    )


def is_configured_no_area_channel(
    *,
    channel_id: int,
    sender_id: int,
    sender_is_bot: bool,
    config: ZulipConfig,
    bot_user_id: int | None = None,
) -> bool:
    """Stream mode: decide whether one message event may trigger a model turn.

    A turn is admitted when the channel passes the ``channels`` list (empty
    admits every delivered channel; nonempty admits only the listed integer
    IDs — a stringified ID is not a match) and the human/bot rule below
    admits the sender (``bot_user_id`` is the bot's own registration
    ``user_id``). The human/bot decision is the single shared
    :func:`admits_human_trigger` core, never a second copy of it. The name
    is historical (spc-119/123 admitted no-area channels only); the routed
    area itself comes from :func:`resolve_channel_area`.
    """
    if config.channels and channel_id not in config.channels:
        return False
    return admits_human_trigger(
        sender_id=sender_id,
        sender_is_bot=sender_is_bot,
        users=config.users,
        bot_user_id=bot_user_id,
    )


def admits_direct_message(
    *,
    message: dict,
    users,
    bot_ids,
    bot_user_id: int | None = None,
    peer_id: int | None = None,
    message_id_field: str = "id",
) -> bool:
    """Direct mode: decide whether one event message may trigger a turn (spc-992).

    DM-shape checks on top of the same shared :func:`admits_human_trigger`
    core the stream mode calls — this function adds no human/bot policy of
    its own, only what a DM has that a stream event check does not:

    * the message is a private message (a stream post is never admitted,
      ``users`` or not);
    * the sender ID and the message ID are real integers — a stringified ID
      is not a match;
    * when a *peer_id* is configured, only that peer's DMs are admitted.

    The bot rule (realm-metadata ``bot_ids`` and the bot's own
    ``bot_user_id``) is the shared core's.
    """
    if message.get("type") != "private":
        return False
    sender_id = message.get("sender_id")
    if not isinstance(sender_id, int):
        return False
    if not isinstance(message.get(message_id_field), int):
        return False
    return admits_human_trigger(
        sender_id=sender_id,
        sender_is_bot=sender_id in bot_ids,
        users=users,
        bot_user_id=bot_user_id,
    ) and (peer_id is None or sender_id == peer_id)


def bot_ids_from_registration(registration: dict) -> set[int]:
    """Collect bot user IDs from registration realm metadata (spc-992: shared).

    Message events carry no bot flag; per the register-queue contract the
    ``realm_users`` array (every account, ``is_bot`` marks bots) plus the
    ``realm_bots`` array are the real server metadata that identifies bots
    by ``user_id``. Both Zulip interface modes read bot identity through
    this one function — an invented per-event bot flag never decides
    admission.
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


def validate_direct_mode(config: ZulipConfig) -> None:
    """Validate the DM block's shape at direct-mode startup (spc-992).

    Direct mode needs exactly one thing stream mode does not: a peer to DM.
    Without it the bot could receive but never reply, so startup fails
    explicitly (the strict schema already rejects a stringified
    ``peer_id``). Called by ``ZulipDMInterface.start()``; stream mode never
    calls it and never reads ``peer_id``.
    """
    if not isinstance(config.peer_id, int):
        raise ValueError(
            "Zulip direct mode needs a `peer_id`: the integer user ID of "
            "the human this bot exchanges DMs with. Add "
            "`zulip.peer_id: <user-id>` to the config."
        )


def resolve_channel_area(
    *,
    channel_id: int,
    config: ZulipConfig,
    channel_name: str | None = None,
) -> str:
    """Decide the area identity for one channel: a name, or ``""``.

    ``channel_areas[str(channel_id)]`` wins when the key exists (``""``
    stays the explicit no-area). Otherwise the area is *channel_name* (a
    stream message's ``display_recipient``) derived by the one
    :func:`sr2_spectre.interfaces.common.resolve_area` Discord uses. No area
    is always ``""``, never ``None``: this decision never consults the
    working directory or ``SR2_AREA``, so a channel can never accidentally
    inherit the service's area.
    """
    area, _provenance = resolve_area(
        channel_id, channel_name, config.channel_areas
    )
    return area or ""


def validate_no_area_only_routing(config: ZulipConfig) -> None:
    """Validate the channel options' shape at startup.

    A shape guard only — ``channels`` integer IDs, ``channel_areas`` string
    keys to string areas (the strict ``ZulipConfig`` schema enforces both);
    a named area that does not exist in the vault yet is valid, so this
    validation opens no vault. The name is historical (spc-119/123).
    """
    for channel_id in config.channels:
        if not isinstance(channel_id, int):
            raise ValueError(
                "Zulip `channels` is malformed: each entry must be an "
                f"integer channel ID, got {type(channel_id).__name__}."
            )
    for key, area in config.channel_areas.items():
        if not isinstance(key, str) or not isinstance(area, str):
            raise ValueError(
                "Zulip `channel_areas` is malformed: each entry must map a "
                "channel ID written as a string to a string area identity, "
                f"got {type(key).__name__}: {type(area).__name__}."
            )
