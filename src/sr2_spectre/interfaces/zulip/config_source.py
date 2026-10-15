"""Live Zulip configuration view.

The Zulip interface is a long-lived process, so the routing map and the
user allowlist it started with are not the ones it should be held to
forever. Since spc-120 the adapter re-reads the whole resolved config on
every inbound message instead.

The reload machinery itself — last-good fallback, atomic swap, pinned
fields — is generic and lives in :mod:`sr2_spectre.config_source`. This
module supplies the Zulip-shaped way of reaching it, mirroring
``DiscordConfigView``:

- ``ZulipConfigView`` is the Zulip slice of a live ``SpectreConfigSource``.
  One reload per inbound message refreshes the whole agent config — models,
  endpoints, pipeline, tools — and the adapter reads its own routing slice
  out of that same single load rather than parsing the file a second time;
  ``Agent.apply_config`` consumes the rest of that same read.

The ``zuliprc`` credential path and any ``api_url`` override are pinned by
the interface itself: the transport, the registered event queue and the
Session identities were built from them at startup, so they carry the
startup values for the life of the process regardless of what a reload
returns. Everything else in the slice — ``channels`` and
``allowed_user_ids`` above all — is hot and applies to the next inbound
message. A malformed edit keeps the last known-good slice in force
(``LiveConfigSource.reload`` absorbs the failure); it never takes the
process down.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sr2_spectre.interfaces.zulip.config import ZulipConfig

if TYPE_CHECKING:
    from sr2_spectre.config_source import SpectreConfigSource


class ZulipConfigView:
    """The Zulip slice of a live ``SpectreConfigSource``.

    Reloading the view reloads the whole Spectre config — that is the
    point: the interface's message dispatch is the process's single entry
    point, so it is the one place that can refresh the agent's models and
    endpoints too. The adapter sees only its own settings;
    ``Agent.apply_config`` consumes the rest of the same load.
    """

    def __init__(self, source: "SpectreConfigSource") -> None:
        self._source = source
        # Optional adapter-set slice in force (see ``pin_effective``).
        self._pinned: ZulipConfig | None = None

    @property
    def source(self) -> "SpectreConfigSource":
        """The underlying whole-config source."""
        return self._source

    @property
    def current(self) -> ZulipConfig:
        """The Zulip config in force, as of the last reload.

        A whole-config load without a ``zulip`` block reads as an
        explicitly empty slice (nobody admitted, nothing routed) rather
        than an error — startup already refused that configuration, and a
        mid-run reload must not crash on it.
        """
        if self._pinned is not None:
            return self._pinned
        return self._source.current.zulip or ZulipConfig(zuliprc="")

    def reload(self) -> ZulipConfig:
        """Reload the whole Spectre config; return the Zulip slice of it."""
        self._source.reload()
        return self.current

    def pin_effective(self, config: ZulipConfig | None) -> None:
        """Set the slice this view reports, or None to report the raw slice.

        The Zulip adapter uses this to keep its pinned fields (``zuliprc``,
        ``api_url`` — the transport and Session identities are built from
        them at startup) visible through the view, so the adapter reads
        exactly one effective config per message.
        """
        self._pinned = config
