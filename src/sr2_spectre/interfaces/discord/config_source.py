"""Live Discord configuration source.

The Discord bot is a long-lived process, often running on a machine the
operator cannot reach. Holding the config in memory meant every edit needed
a restart, so the interface re-reads its config from disk on every inbound
message instead. ``DiscordConfigSource`` owns that read.

Two properties matter more than freshness:

- **The bot must survive a bad config.** A file that is missing, malformed
  or caught mid-save must not take the process down; the last known-good
  config stays in force until the file parses again.
- **A reload is atomic.** ``reload()`` swaps in a whole new config object
  rather than mutating fields, so no reader ever sees a half-applied edit.
  Readers are not pinned to a snapshot, though: a message that arrives while
  an earlier reply is still streaming triggers its own reload, and the
  in-flight reply picks up the new values from that point on.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from sr2_spectre.interfaces.discord.config import DiscordConfig

logger = logging.getLogger(__name__)

# Fields that cannot change while the bot is connected. A different token
# needs a fresh gateway login, which a reload cannot perform, so the value
# loaded at startup is carried forward and the file value is ignored.
_PINNED_FIELDS: tuple[str, ...] = ("token",)


class DiscordConfigSource:
    """Supplies the DiscordConfig in force right now, re-reading on demand.

    The interface and the adapter share one source. The adapter calls
    ``reload()`` once, at the top of the message dispatch path; everything
    downstream reads ``current``.
    """

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
        self._loader = loader
        self._current = initial if initial is not None else DiscordConfig()
        self._load_error: str | None = None
        self._pinned_warned: set[str] = set()

    @classmethod
    def static(cls, config: DiscordConfig | None = None) -> DiscordConfigSource:
        """Build a source that never changes.

        Used by callers that hand the interface a config object directly
        (tests, embedders) instead of a path to reload from.
        """
        cfg = config if config is not None else DiscordConfig()
        return cls(loader=lambda: cfg, initial=cfg)

    @property
    def current(self) -> DiscordConfig:
        """The config in force — the most recent successful load."""
        return self._current

    def reload(self) -> DiscordConfig:
        """Re-read the config, returning the config now in force.

        On loader failure the previous config is returned unchanged and the
        error is logged once (repeats of the same error are suppressed so a
        broken file does not flood the log with one line per message).
        """
        try:
            loaded = self._loader()
        except Exception as exc:  # noqa: BLE001 — any loader failure must be survivable
            self._note_failure(f"{type(exc).__name__}: {exc}")
            return self._current

        self._note_success()

        candidate = self._apply_pinned(loaded)
        changed = _changed_fields(self._current, candidate)
        if changed:
            # Field names only: values may include secrets.
            logger.info("Discord config reloaded — changed: %s", ", ".join(changed))
            self._current = candidate
        return self._current

    def _apply_pinned(self, loaded: DiscordConfig) -> DiscordConfig:
        """Carry pinned fields over from the config in force.

        An unset pinned value means the bot never started with one (the
        adapter refuses to connect without a token), so there is nothing to
        protect and the loaded value is adopted. Once a value is in force it
        is frozen: that is the token the gateway session was opened with.
        """
        pinned: dict[str, object] = {}
        for name in _PINNED_FIELDS:
            in_force = getattr(self._current, name)
            if not in_force:
                continue
            pinned[name] = in_force
            if getattr(loaded, name) != in_force and name not in self._pinned_warned:
                logger.warning(
                    "Discord config: '%s' changed on disk but cannot be applied "
                    "without a restart — keeping the value the bot started with.",
                    name,
                )
                self._pinned_warned.add(name)
        return loaded.model_copy(update=pinned)

    def _note_failure(self, error: str) -> None:
        if error != self._load_error:
            logger.warning(
                "Discord config reload failed (keeping the config in force): %s",
                error,
            )
            self._load_error = error

    def _note_success(self) -> None:
        if self._load_error is not None:
            logger.info("Discord config reload recovered")
            self._load_error = None


def _changed_fields(before: DiscordConfig, after: DiscordConfig) -> list[str]:
    """Names of the fields that differ between two configs."""
    return [
        name
        for name in type(after).model_fields
        if getattr(before, name) != getattr(after, name)
    ]
