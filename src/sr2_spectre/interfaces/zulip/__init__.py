"""Zulip interface package — configuration, credentials, transport, chat.

Public seams shipped so far: the ``zulip`` config block, synthetic zuliprc
credential loading, trigger admission, the temporary nonempty-area routing
gate, the ``ZulipClient`` HTTP transport (spc-124), and the ``ZulipInterface``
safe no-area chat interface (spc-125).
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
    ZulipConfig,
    is_configured_no_area_channel,
    validate_no_area_only_routing,
)
from sr2_spectre.interfaces.zulip.interface import (
    ZulipInterface,
    topic_frame_id,
)

__all__ = [
    "ZulipAuthError",
    "ZulipClient",
    "ZulipConfig",
    "ZulipCredentialError",
    "ZulipCredentials",
    "ZulipError",
    "ZulipInterface",
    "ZulipTransportError",
    "is_configured_no_area_channel",
    "load_zulip_credentials",
    "topic_frame_id",
    "validate_no_area_only_routing",
]
