"""Zulip interface package — configuration, credentials, admission exports.

This cut (spc-123) ships the pure public seams: the ``zulip`` config block,
synthetic zuliprc credential loading, trigger admission and the temporary
nonempty-area routing gate. ``ZulipCredentialError`` / ``ZulipAuthError`` /
``ZulipTransportError`` are the declared failure boundaries, re-exported at
package level. Transport (spc-124), the interface and CLI wiring (spc-125)
build on these exports; nothing here performs HTTP or model work.
"""
from __future__ import annotations

from sr2_spectre.interfaces.zulip.client import (
    ZulipAuthError,
    ZulipCredentialError,
    ZulipCredentials,
    ZulipTransportError,
    load_zulip_credentials,
)
from sr2_spectre.interfaces.zulip.config import (
    ZulipConfig,
    is_configured_no_area_channel,
    validate_no_area_only_routing,
)

__all__ = [
    "ZulipAuthError",
    "ZulipConfig",
    "ZulipCredentialError",
    "ZulipCredentials",
    "ZulipTransportError",
    "is_configured_no_area_channel",
    "load_zulip_credentials",
    "validate_no_area_only_routing",
]
