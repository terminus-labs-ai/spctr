"""Live configuration sources — re-read config from disk while running.

A long-lived agent process (the Discord bot above all) usually runs somewhere
the operator cannot reach, so a config field that needs a restart is a field
that does not get changed. These sources re-read the fully resolved config on
every inbound message instead.

Three properties matter more than freshness:

- **The process must survive a bad config.** A file that is missing, malformed
  or caught mid-save must not take the process down; the last known-good
  config stays in force until the file parses again.
- **A swap is atomic.** ``reload()`` swaps in a whole new config object rather
  than mutating fields, so no reader ever sees a half-applied edit.
- **Some values cannot change under a running process.** A Discord token needs
  a fresh gateway login; an MCP server needs its subprocess respawned; a store
  DSN needs its connection torn down. Those fields are *pinned*: the value the
  process started with is carried forward, the file value is ignored, and the
  operator is warned once so the divergence is never silent.

Reloading is deliberately separated from *applying*. This module only answers
"what does the file say now"; ``Runtime.apply_config`` decides what can be
swapped into a live process and what needs a restart.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    # Deferred: sr2_spectre.config reaches into the interface packages, which
    # import this module. Only annotations need the name, so importing it
    # lazily keeps that cycle from closing at import time.
    from sr2_spectre.config import SpectreConfig

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def _get_path(model: BaseModel, path: str) -> object:
    """Read a dotted field path off a model, returning None if any hop is None."""
    current: object = model
    for part in path.split("."):
        if current is None:
            return None
        current = getattr(current, part, None)
    return current


def _pin_path(loaded: ConfigT, value: object, path: str) -> ConfigT:
    """Return a copy of *loaded* with the dotted *path* forced to *value*.

    Rebuilds only the models along the path, so unrelated branches of the
    config keep their freshly loaded values.
    """
    head, _, rest = path.partition(".")
    if not rest:
        return loaded.model_copy(update={head: value})

    child = getattr(loaded, head, None)
    if child is None:
        # Nothing to pin into — the loaded config dropped this whole branch.
        return loaded
    return loaded.model_copy(update={head: _pin_path(child, value, rest)})


class LiveConfigSource(Generic[ConfigT]):
    """Supplies the config in force right now, re-reading on demand.

    Subclasses declare which fields are pinned. Callers call ``reload()`` once,
    at the top of the message dispatch path; everything downstream reads
    ``current``.
    """

    #: Dotted field paths that cannot be changed without a process restart.
    PINNED_FIELDS: tuple[str, ...] = ()

    #: Used in log lines to say which config reloaded.
    LABEL: str = "Config"

    def __init__(
        self,
        loader: Callable[[], ConfigT],
        initial: ConfigT,
    ) -> None:
        """
        Args:
            loader: Callable that re-reads and validates the config from
                disk. May raise — failures are absorbed by ``reload()``.
            initial: Config to start from, normally the one resolved at
                process start.
        """
        self._loader = loader
        self._current = initial
        self._load_error: str | None = None
        self._pinned_warned: set[str] = set()
        # Logged under the concrete subclass's module, not this one, so an
        # operator filtering on `sr2_spectre.interfaces.discord` still sees the
        # bot's own reload lines.
        self._log = logging.getLogger(type(self).__module__)

    @property
    def current(self) -> ConfigT:
        """The config in force — the most recent successful load."""
        return self._current

    def reload(self) -> ConfigT:
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
            self._log.info("%s reloaded — changed: %s", self.LABEL, ", ".join(changed))
            self._current = candidate
        return self._current

    def _apply_pinned(self, loaded: ConfigT) -> ConfigT:
        """Carry pinned fields over from the config in force.

        An unset pinned value means the process never started with one, so
        there is nothing to protect and the loaded value is adopted. Once a
        value is in force it is frozen: that is the value the live connection,
        subprocess or session was opened with.
        """
        candidate = loaded
        for path in self.PINNED_FIELDS:
            in_force = _get_path(self._current, path)
            if not in_force:
                continue
            if _get_path(loaded, path) != in_force and path not in self._pinned_warned:
                self._log.warning(
                    "%s: '%s' changed on disk but cannot be applied "
                    "without a restart — keeping the value the process started with.",
                    self.LABEL,
                    path,
                )
                self._pinned_warned.add(path)
            candidate = _pin_path(candidate, in_force, path)
        return candidate

    def _note_failure(self, error: str) -> None:
        if error != self._load_error:
            self._log.warning(
                "%s reload failed (keeping the config in force): %s",
                self.LABEL,
                error,
            )
            self._load_error = error

    def _note_success(self) -> None:
        if self._load_error is not None:
            self._log.info("%s reload recovered", self.LABEL)
            self._load_error = None


class SpectreConfigSource(LiveConfigSource["SpectreConfig"]):
    """The whole resolved SpectreConfig, re-read on every inbound message.

    Pinned fields are the ones wired into a live connection or subprocess at
    startup, which a reload has no way to rebuild:

    - ``agent.name`` — frame ids are derived from it, so changing it mid-process
      would orphan every open conversation.
    - ``agent.mcp_servers`` — each server owns a connected transport, and stdio
      servers own a subprocess.
    - ``memory_store_dsn`` / ``provenance_store_path`` — both back a connection
      opened once at startup.
    - ``discord.token`` — the gateway session was opened with it.

    Everything else (models and endpoints above all, the pipeline, tools,
    skills) is hot-swappable and applied by ``Runtime.apply_config``.
    """

    PINNED_FIELDS = (
        "agent.name",
        "agent.mcp_servers",
        "memory_store_dsn",
        "provenance_store_path",
        "discord.token",
    )
    LABEL = "Spectre config"

    @classmethod
    def static(cls, config: "SpectreConfig") -> "SpectreConfigSource":
        """Build a source that never changes.

        Used by callers that hand the runtime a config object directly (tests,
        embedders, non-Discord interfaces) instead of a path to reload from.
        """
        return cls(loader=lambda: config, initial=config)


def _changed_fields(before: BaseModel, after: BaseModel) -> list[str]:
    """Names of the fields that differ between two configs."""
    return [
        name
        for name in type(after).model_fields
        if getattr(before, name) != getattr(after, name)
    ]
