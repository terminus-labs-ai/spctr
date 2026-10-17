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

Everything in the slice — ``channels``, ``allowed_user_ids``, the
``zuliprc`` path and the ``api_url`` override above all — is hot: each
reload reports the newly loaded slice and applies to the next inbound
message. The interface separately keeps its *own* startup block for the
transport, the registered event queue and the Session identities, which
were built from it and are rebound to it on every reload. A malformed
edit keeps the last known-good slice in force (``LiveConfigSource.reload``
absorbs the failure); it never takes the process down.
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

    @property
    def source(self) -> "SpectreConfigSource":
        """The underlying whole-config source."""
        return self._source

    @property
    def current(self) -> ZulipConfig:
        """The Zulip config in force — the slice of the last successful load.

        A whole-config load without a ``zulip`` block reads as an
        explicitly empty slice (nobody admitted, nothing routed) rather
        than an error — startup already refused that configuration, and a
        mid-run reload must not crash on it.
        """
        return self._source.current.zulip or ZulipConfig(zuliprc="")

    def reload(self) -> ZulipConfig:
        """Reload the whole Spectre config; return the Zulip slice of it.

        Every reload reports the newly loaded slice: on loader failure
        ``LiveConfigSource.reload`` keeps the last known-good whole config
        in force, so the slice returned is that config's slice, never a
        stale per-view copy.
        """
        self._source.reload()
        return self.current
