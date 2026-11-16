"""Zulip interface configuration and the pure admission/routing decisions.

The schema follows the spec's Interfaces section: ``zuliprc`` (credential
path, no inline key), an optional ``api_url`` override (loopback test
endpoints; the zuliprc ``site`` stays the canonical origin), explicit
integer-ID allowlists, and Discord's channel options (spc-1012):
``channels`` is a list of integer channel IDs (empty = every channel the
event queue delivers) and ``channel_areas`` maps a channel ID written as a
string to an area identity (``""`` = explicitly no area). Without an
override the area is the channel name, derived by the one
:func:`spctr.interfaces.common.derive_area_name` Discord uses; it
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

import re
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spctr.interfaces.common import resolve_area

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
        topic_owner_routing: Stream mode (spc-1071): when true, one owner bot
            per topic. The owner is recorded per topic in-process: from the
            topic's first message (one bounded fetch, cached whatever its
            value — any non-wildcard realm mention decides it), then from
            later BOT mentions only — a mention resolving to this bot
            records ``self`` and claims the topic, a mention of another
            registered bot records ``other``, and a mention of a human
            changes nothing. A message in a topic recorded ``other`` is
            dropped silently before any trigger, Session or indicator work.
            A topic this bot owns plays the mention exemption
            ``mention_only`` would otherwise require. ``False`` — the
            default — is zero behavior change anywhere. Direct mode
            ignores it.
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
        max_message_length: Stream mode: the longest single message the
            bot sends (Zulip's limit). A longer answer is split with the
            shared :func:`spctr.interfaces.common.chunk_message`.
        thinking_message: Stream mode: the placeholder posted in the topic
            when a turn is admitted, before the model call; the answer
            replaces it by edit.
        empty_response_message: Stream mode: what the placeholder shows
            when the model answers with no text.
        error_message: Stream mode: what the placeholder shows when the
            model turn fails; ``{error}`` is replaced by the error text.
        edit_stream_interval: Stream mode: the minimum seconds between
            text-progress edits of the placeholder; ``0`` disables them
            (tool-line edits still happen).
        tool_embed_enabled: Stream mode: when true, each tool start and
            result appends its templated line to the placeholder.
        tool_line_max_chars: Stream mode: ``{input}`` and ``{content}`` in a
            tool line are cut to this many characters plus ``...``.
        tool_start_line: Stream mode: the line a tool start appends;
            ``{name}`` and ``{input}`` are filled in.
        tool_result_ok_line: Stream mode: the line a successful tool result
            appends; ``{name}`` and ``{content}`` (newlines as ``|``).
        tool_result_error_line: Stream mode: the line a failed tool result
            appends; same placeholders as ``tool_result_ok_line``.
        handoff_start_message: Stream mode: posted when a handoff takes over.
        handoff_end_message: Stream mode: posted when a handoff returns.
        stop_handoff_message: Stream mode: reply after stopping a handoff.
        stop_turn_message: Stream mode: reply after stopping a topic turn.
        stop_idle_message: Stream mode: reply when ``!stop`` finds no turn.
    """
    model_config = ConfigDict(strict=True)

    zuliprc: str
    api_url: str | None = None
    users: list[int] = Field(default_factory=list)
    mention_only: bool = False
    topic_owner_routing: bool = False
    channels: list[int] = Field(default_factory=list)
    channel_areas: dict[str, str] = Field(default_factory=dict)
    mode: str = "stream"
    peer_id: int | None = None
    # Working indicator (stream mode; direct mode ignores these).
    max_message_length: int = 10000
    thinking_message: str = "⏳ Thinking..."
    empty_response_message: str = "(No response generated)"
    error_message: str = "Error: {error}"
    # Live progress (stream mode, spc-1015).
    edit_stream_interval: float = 1.0
    tool_embed_enabled: bool = True
    tool_line_max_chars: int = 80
    tool_start_line: str = "▶ `{name}` - `{input}`"
    tool_result_ok_line: str = "✓ `{name}` - `{content}`"
    tool_result_error_line: str = "✖ `{name}` - `{content}`"
    handoff_start_message: str = "→ {agent} took over ({frame_id})"
    handoff_end_message: str = "← {agent}: {status} in {duration}"
    stop_handoff_message: str = "⏹ stopped {agent}"
    stop_turn_message: str = "⏹ stopped"
    stop_idle_message: str = "nothing running"

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


#: One realm-mention operand inside ``@**X**`` / ``@_**X**`` (spc-1071).
_REALM_MENTION_RE = re.compile(r"@_?\*\*([^*]+?)\*\*")

#: Wildcard mention targets: they name no user and are never ownership
#: material (same set spc-1013 documents for ``mention_only``).
_WILDCARD_MENTION_NAMES = frozenset(
    {"all", "everyone", "channel", "stream", "topic"}
)


def _realm_mention_targets(
    content: str, *, full_name: str | None, user_id: int | None
) -> Iterator[tuple[str, int | None]]:
    """Yield ``(name, id-or-None)`` per realm-mention operand in *content*
    that is neither a wildcard nor this bot (spc-1071).

    Classification is by mention SYNTAX alone: every ``@**X**`` /
    ``@_**X**`` operand ``X`` (``<name>`` or ``<name>|<id>``) yields its
    display name and parsed ``|id`` unless it is a wildcard name
    (``all``/``everyone``/``channel``/``stream``/``topic`` — names no
    user) or resolves to this bot exactly as :func:`mentions_bot`
    resolves it — ``X == full_name`` or ``X == full_name|user_id`` — so
    the two functions can never disagree. User-group syntax
    (``@*group*``, single asterisks) and plain typed ``@name`` are not
    realm mentions and never match.
    """
    for match in _REALM_MENTION_RE.finditer(content):
        target = match.group(1)
        name, sep, suffix = target.partition("|")
        if name in _WILDCARD_MENTION_NAMES:
            continue
        if full_name and (
            target == full_name
            or (user_id is not None and target == f"{full_name}|{user_id}")
        ):
            continue
        yield name, int(suffix) if sep and suffix.isdigit() else None


def mentions_other(
    content: str, *, full_name: str | None, user_id: int | None
) -> bool:
    """True when *content* mentions a realm identity that is not this bot.

    Any non-wildcard realm-mention target that does not resolve to this
    bot counts (:func:`_realm_mention_targets`): a human's mention as
    well as another bot's — the OPENER rule (spc-1071).
    """
    return any(
        _realm_mention_targets(
            content, full_name=full_name, user_id=user_id
        )
    )


def mentions_other_bot(
    content: str,
    *,
    full_name: str | None,
    user_id: int | None,
    bot_ids: set[int],
    bot_names: set[str],
) -> bool:
    """True when *content* mentions a realm identity that is ANOTHER BOT.

    The later-message hand-over rule (spc-1071, Diego ruling 2026-11-16):
    a mention target counts when its ``|id`` is a registered bot other
    than this one (``bot_ids`` minus ``user_id`` itself) or its name is
    a registered bot name (``bot_names``; the own name is already out by
    the shared target rule). A human mention is never a hand-over.
    """
    for name, target_id in _realm_mention_targets(
        content, full_name=full_name, user_id=user_id
    ):
        if name in bot_names:
            return True
        if (
            target_id is not None
            and target_id != user_id
            and target_id in bot_ids
        ):
            return True
    return False


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


def bot_names_from_registration(registration: dict) -> set[str]:
    """Collect bot full names from registration realm metadata (spc-1071).

    The name twin of :func:`bot_ids_from_registration`, over the same
    data source: ``realm_users`` entries with ``is_bot`` carry every
    bot's ``full_name`` (the mention syntax names users by full name),
    and ``realm_bots`` entries may carry one too.
    """
    names: set[str] = set()
    for entry in registration.get("realm_users") or []:
        if (
            isinstance(entry, dict)
            and entry.get("is_bot") is True
            and isinstance(entry.get("full_name"), str)
        ):
            names.add(entry["full_name"])
    for entry in registration.get("realm_bots") or []:
        if isinstance(entry, dict) and isinstance(entry.get("full_name"), str):
            names.add(entry["full_name"])
    return names


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
    :func:`spctr.interfaces.common.resolve_area` Discord uses. No area
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
