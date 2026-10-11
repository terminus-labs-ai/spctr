"""Zulip interface configuration and the pure admission/routing decisions.

The schema follows the spec's Interfaces section: ``zuliprc`` (credential
path, no inline key), an optional ``api_url`` override (loopback test
endpoints; the zuliprc ``site`` stays the canonical origin), explicit
integer-ID allowlists, and ``channels`` mapping channel ID → area identity
(``""`` = explicitly no area). There is no channel-name or working-directory
fallback, and string forms of an ID (e.g. a YAML-quoted ``"42"``) are not
admitted in place of the integer.

INTERMEDIATE CONTRACT: nonempty area mappings are rejected until area
routing ships in spc-120
(``Add Zulip area routing and live config reload``), which owns lifting
both the :func:`validate_no_area_only_routing` gate and the nonempty-area
arm of :func:`is_configured_no_area_channel`. Until then a nonempty area
fails startup explicitly rather than silently routing without its area.
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
            area. In this cut only ``""`` values are admitted; nonempty areas
            fail startup (validate_no_area_only_routing) until spc-120.
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

    A turn is admitted only when the channel is explicitly configured as a
    no-area channel, the sender is a human on the allowlist, and the message
    is not bot-authored (the bot rule outranks the allowlist so replies can
    never loop). Membership is decided on the configured integer IDs — an
    event carrying a stringified ID is not a match, and there is no
    channel-name fallback.
    """
    if sender_is_bot:
        return False
    if config.channels.get(channel_id) != "":
        return False
    return sender_id in config.allowed_user_ids


def validate_no_area_only_routing(config: ZulipConfig) -> None:
    """Fail startup when any configured channel carries a nonempty area.

    INTERMEDIATE CONTRACT: area routing ships in spc-120. Until then a
    nonempty area mapping is rejected explicitly instead of silently routing
    without its area.
    """
    offending = sorted(cid for cid, area in config.channels.items() if area)
    if offending:
        raise ValueError(
            "Zulip area routing is not supported in this build: channel(s) "
            f"{offending} map to a nonempty area. Configure area-less "
            'channels (area "") or wait for the routing release (spc-120).'
        )
