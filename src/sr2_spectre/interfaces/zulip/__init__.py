"""Zulip interface package — configuration, credentials, transport exports.

Public seams shipped so far: the ``zulip`` config block, synthetic zuliprc
credential loading, trigger admission, the temporary nonempty-area routing
gate, and the ``ZulipClient`` HTTP transport (spc-124: register, get-events,
exact-topic history, send — configured server only). The interface and CLI
wiring (spc-125) build on these exports; nothing here performs model work.
"""
from __future__ import annotations

from sr2_spectre.interfaces.zulip.client import (
    ZulipAuthError,
    ZulipClient,
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
    "ZulipClient",
    "ZulipConfig",
    "ZulipCredentialError",
    "ZulipCredentials",
    "ZulipTransportError",
    "is_configured_no_area_channel",
    "load_zulip_credentials",
    "validate_no_area_only_routing",
]
