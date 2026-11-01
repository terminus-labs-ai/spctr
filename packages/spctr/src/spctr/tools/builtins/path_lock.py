"""Per-path locks serializing file mutations across worker threads."""
from __future__ import annotations

import os
import threading

_registry: dict[str, threading.Lock] = {}
_registry_guard = threading.Lock()


def path_lock(path: str) -> threading.Lock:
    """Return the process-wide lock for the file at *path*, keyed by os.path.realpath(path)."""
    key = os.path.realpath(path)
    with _registry_guard:
        lock = _registry.get(key)
        if lock is None:
            lock = _registry[key] = threading.Lock()
        return lock
