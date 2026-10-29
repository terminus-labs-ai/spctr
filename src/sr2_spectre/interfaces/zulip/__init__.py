"""Zulip interface package — configuration, credentials, transport, chat.

Public seams: the ``zulip`` config block (modes ``stream`` and ``direct``,
spc-992), synthetic zuliprc credential loading, the ONE admission core
(``admits_human_trigger``, wrapped per mode by
``is_configured_no_area_channel`` and ``admits_direct_message``), the
``ZulipClient`` HTTP transport (spc-124, DM send included), the
``ZulipInterface`` stream/topic chat interface (spc-125) and the
``ZulipDMInterface`` DM-only variant (spc-992, promoted from the Afterlife
character runtime — its poll/admit/send core is the seam an external
runtime drives directly from config).
"""
from __future__ import annotations

from sr2_spectre.interfaces.zulip.client import (
    ZulipAuthError,
    ZulipClient,
    ZulipCredentialError,
    ZulipCredentials,
    ZulipError,
    ZulipTransportError,
    load_zulip_credentials,
)
from sr2_spectre.interfaces.zulip.config import (
    ZULIP_MODES,
    ZulipConfig,
    admits_direct_message,
    admits_human_trigger,
    bot_ids_from_registration,
    is_configured_no_area_channel,
    mentions_bot,
    validate_direct_mode,
    validate_no_area_only_routing,
)
from sr2_spectre.interfaces.zulip.direct import (
    DMessage,
    ZulipDMInterface,
    dm_frame_id,
)
from sr2_spectre.interfaces.zulip.interface import (
    ZulipInterface,
    topic_frame_id,
)

__all__ = [
    "DMessage",
    "ZULIP_MODES",
    "ZulipAuthError",
    "ZulipClient",
    "ZulipConfig",
    "ZulipCredentialError",
    "ZulipCredentials",
    "ZulipDMInterface",
    "ZulipError",
    "ZulipInterface",
    "ZulipTransportError",
    "admits_direct_message",
    "admits_human_trigger",
    "bot_ids_from_registration",
    "dm_frame_id",
    "is_configured_no_area_channel",
    "load_zulip_credentials",
    "mentions_bot",
    "topic_frame_id",
    "validate_direct_mode",
    "validate_no_area_only_routing",
]
