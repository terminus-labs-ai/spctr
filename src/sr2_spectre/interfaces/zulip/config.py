"""Zulip interface configuration and the pure admission/routing decisions.

The schema follows the spec's Interfaces section: ``zuliprc`` (credential
path, no inline key), an optional ``api_url`` override (loopback test
endpoints; the zuliprc ``site`` stays the canonical origin), explicit
integer-ID allowlists, and ``channels`` mapping channel ID → area identity
(``""`` = explicitly no area). There is no channel-name or working-directory
fallback, and string forms of an ID (e.g. a YAML-quoted ``"42"``) are not
admitted in place of the integer.

ROUTING CONTRACT (spc-120): a mapped nonempty area is valid configuration
and routes through the existing resolvers; ``""`` stays the explicit
no-area and an unconfigured channel is never routed. The former temporary
gate (:func:`validate_no_area_only_routing`, spc-119/123 intermediate
contract) only validates mapping shape now — live vault existence is never
probed at config time.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ZulipConfig(BaseModel):
    """Configuration for the Zulip interface.

    Attributes:
        zuliprc: Path to the bot's zuliprc credential file (email, key, site).
        api_url: Optional API origin override. When set (e.g. a loopback test
            endpoint), requests go there instead of the zuliprc site.
        allowed_user_ids: Human user IDs whose messages may trigger a turn.
            Empty admits no one; bot-authored messages never trigger even if
            an ID appears here (reply-loop prevention).
        channels: Channel ID → area identity. ``""`` means explicitly no
            area; a nonempty value is that channel's area identity, mapped
            through :func:`resolve_channel_area` on every inbound message.
            There is no implicit channel-name or working-directory fallback.
    """
    model_config = ConfigDict(strict=True)

    zuliprc: str
    api_url: str | None = None
    allowed_user_ids: list[int] = Field(default_factory=list)
    channels: dict[int, str] = Field(default_factory=dict)


def is_configured_no_area_channel(
    *,
    channel_id: int,
    sender_id: int,
    sender_is_bot: bool,
    config: ZulipConfig,
) -> bool:
    """Decide whether one message event may trigger a model turn.

    A turn is admitted only when the channel is explicitly configured (any
    area, including the no-area ``""``), the sender is a human on the
    allowlist, and the message is not bot-authored (the bot rule outranks
    the allowlist so replies can never loop). Membership is decided on the
    configured integer IDs — an event carrying a stringified ID is not a
    match, and there is no channel-name fallback. The name is historical
    (spc-119/123 admitted no-area channels only); since spc-120 it is the
    trigger admission for every configured channel — the routed area itself
    comes from :func:`resolve_channel_area`.
    """
    if sender_is_bot:
        return False
    if channel_id not in config.channels:
        return False
    return sender_id in config.allowed_user_ids


def resolve_channel_area(
    *,
    channel_id: int,
    config: ZulipConfig,
) -> str | None:
    """Decide the area identity for one channel: ``""``, a name, or ``None``.

    The answer comes only from the configured mapping. A configured channel
    gets exactly its mapped value — ``""`` is the explicit no-area and must
    be returned as such, never collapsed into ``None`` (consumers treat
    ``None`` as "may fall back"). An unconfigured channel gets a positive
    non-answer of ``None``: this decision never consults the working
    directory, ``SR2_AREA``, or a channel name, so an unmapped channel can
    never accidentally inherit the service's area.
    """
    return config.channels.get(channel_id)


def validate_no_area_only_routing(config: ZulipConfig) -> None:
    """Validate the routing mapping's shape at startup.

    The temporary spc-119/123 gate that rejected any nonempty area is lifted
    (spc-120): mapped areas are supported routing now. What stays is the
    shape guard — every key an integer, every value a string (the strict
    ``ZulipConfig`` schema enforces both) — and nothing else: a named area
    that does not exist in the vault yet is a valid mapping, so this
    validation opens no vault.
    """
    for channel_id, area in config.channels.items():
        if not isinstance(channel_id, int) or not isinstance(area, str):
            raise ValueError(
                "Zulip channel mapping is malformed: each entry must map an "
                f"integer channel ID to a string area identity, got "
                f"{channel_id!r}: {type(area).__name__}."
            )
