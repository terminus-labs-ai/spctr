"""Live Discord configuration source.

The Discord bot is a long-lived process, often running on a machine the
operator cannot reach, so holding the config in memory meant every edit needed
a restart. The interface re-reads its config from disk on every inbound
message instead.

The reload machinery itself — last-good fallback, atomic swap, pinned fields —
is generic and lives in :mod:`sr2_spectre.config_source`. This module supplies
the two Discord-shaped ways of reaching it:

- ``DiscordConfigSource`` reloads a ``DiscordConfig`` on its own. Used when the
  Discord settings are all a caller cares about.
- ``DiscordConfigView`` is the Discord slice of a live ``SpectreConfigSource``.
  This is the path the bot runs on: one reload per message refreshes the whole
  agent config — models, endpoints, pipeline, tools — and the adapter reads its
  own settings out of that same load rather than parsing the file a second
  time.

Readers are not pinned to a snapshot: a message that arrives while an earlier
reply is still streaming triggers its own reload, and the in-flight reply picks
up the new values from that point on.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sr2_spectre.config_source import LiveConfigSource
from sr2_spectre.interfaces.discord.config import DiscordConfig

if TYPE_CHECKING:
    from sr2_spectre.config_source import SpectreConfigSource


@runtime_checkable
class DiscordConfigProvider(Protocol):
    """What the adapter needs from whatever supplies its config."""

    @property
    def current(self) -> DiscordConfig: ...

    def reload(self) -> DiscordConfig: ...


class DiscordConfigSource(LiveConfigSource[DiscordConfig]):
    """Supplies the DiscordConfig in force right now, re-reading on demand.

    A different token needs a fresh gateway login, which a reload cannot
    perform, so the token loaded at startup is carried forward and the file
    value is ignored.
    """

    PINNED_FIELDS = ("token",)
    LABEL = "Discord config"

    def __init__(
        self,
        loader: Callable[[], DiscordConfig],
        initial: DiscordConfig | None = None,
    ) -> None:
        """
        Args:
            loader: Callable that re-reads and validates the config from
                disk. May raise — failures are absorbed by ``reload()``.
            initial: Config to start from, normally the one resolved at
                process start. Defaults to an empty DiscordConfig.
        """
        super().__init__(
            loader=loader,
            initial=initial if initial is not None else DiscordConfig(),
        )

    @classmethod
    def static(cls, config: DiscordConfig | None = None) -> "DiscordConfigSource":
        """Build a source that never changes.

        Used by callers that hand the interface a config object directly
        (tests, embedders) instead of a path to reload from.
        """
        cfg = config if config is not None else DiscordConfig()
        return cls(loader=lambda: cfg, initial=cfg)


class DiscordConfigView:
    """The Discord slice of a live ``SpectreConfigSource``.

    Reloading the view reloads the whole Spectre config — that is the point:
    the bot's message dispatch is the process's single entry point, so it is
    the one place that can refresh the agent's models and endpoints too. The
    adapter sees only its own settings; ``Runtime.apply_config`` consumes the
    rest of the same load.

    ``discord.token`` is pinned by ``SpectreConfigSource``, so a view carries
    the same restart-safety the standalone source has.
    """

    def __init__(self, source: "SpectreConfigSource") -> None:
        self._source = source

    @property
    def source(self) -> "SpectreConfigSource":
        """The underlying whole-config source."""
        return self._source

    @property
    def current(self) -> DiscordConfig:
        """The Discord config in force, as of the last reload."""
        return self._source.current.discord or DiscordConfig()

    def reload(self) -> DiscordConfig:
        """Reload the whole Spectre config; return the Discord slice of it."""
        self._source.reload()
        return self.current
