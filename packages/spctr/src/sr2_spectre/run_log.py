"""Live session log — one flushed JSONL file per Session.

``SessionLogManager`` (owned by Runtime) creates session logs below
``$SR2_HOME/logs/sessions/`` and sweeps stale ones. ``$SR2_SESSION_LOG_DIR``
moves them elsewhere, or turns them off when set to ``off``. ``SessionLog`` appends
one JSON envelope per event and flushes each line so ``tail -f`` sees it
immediately. A writer holds an exclusive ``flock`` for its lifetime, which is
how a sweep in any process knows the file is still live.

Every failure here warns and is swallowed: logging must never change the
agent result or process exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import re
import secrets
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sr2_spectre.config import resolve_sr2_home

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_TEXT_BYTES = 16 * 1024
RETENTION = timedelta(hours=24)
_SECRET_KEYS = frozenset(
    {"api_key", "authorization", "token", "password", "secret", "cookie"}
)
_REDACTED = "[REDACTED]"
_PROCESS_START = time.monotonic()
_DIR_ENV = "SR2_SESSION_LOG_DIR"


def _bound_text(text: str) -> Any:
    """Return *text*, or a preview wrapper when it exceeds 16 KiB."""
    raw = text.encode("utf-8")
    if len(raw) <= MAX_TEXT_BYTES:
        return text
    return {
        "preview": raw[:MAX_TEXT_BYTES].decode("utf-8", errors="ignore"),
        "bytes": len(raw),
        "truncated": True,
    }


def _sanitize(value: Any) -> Any:
    """Recursively redact secret keys, cap text, and coerce to JSON types."""
    if isinstance(value, str):
        return _bound_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(k): _REDACTED if str(k).lower() in _SECRET_KEYS else _sanitize(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, bytes):
        return _bound_text(value.decode("utf-8", errors="replace"))
    return _bound_text(str(value))


class SessionLog:
    """Append-only JSONL writer for one Session."""

    def __init__(self, path: Path, fd: int, session_id: str) -> None:
        self._path = path
        self._fh = os.fdopen(fd, "a", encoding="utf-8")
        self._session_id = session_id
        self._interface = ""
        self._sequence = 0
        self._failed = False
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def set_interface(self, interface: str) -> None:
        self._interface = interface

    def append(self, event: str, data: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            if self._fh is None or self._failed:
                return
            self._sequence += 1
            try:
                line = json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sequence": self._sequence,
                        "timestamp": datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%S.%fZ"
                        ),
                        "elapsed_ms": round(
                            (time.monotonic() - _PROCESS_START) * 1000, 3
                        ),
                        "session_id": self._session_id,
                        "interface": self._interface,
                        "event": event,
                        "data": _sanitize(data or {}),
                    },
                    default=str,
                )
                self._fh.write(line + "\n")
                self._fh.flush()
            except Exception as exc:
                self._failed = True
                logger.warning("Session log write failed for %s: %s", self._path, exc)

    def close(self) -> None:
        with self._lock:
            fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            fh.close()  # closing the descriptor also releases the flock
        except Exception as exc:
            logger.warning("Session log close failed for %s: %s", self._path, exc)


class SessionLogManager:
    """Creates session logs and sweeps stale ones on a fixed cadence."""

    def __init__(self, cleanup_interval: float = 24 * 60 * 60) -> None:
        self._cleanup_interval = cleanup_interval
        self._task: asyncio.Task | None = None
        self._logs: "weakref.WeakSet[SessionLog]" = weakref.WeakSet()

    @property
    def directory(self) -> Path:
        raw = os.environ.get(_DIR_ENV, "").strip()
        if raw and raw.lower() != "off":
            return Path(raw).expanduser()
        return resolve_sr2_home() / "logs" / "sessions"

    @property
    def enabled(self) -> bool:
        """False when ``$SR2_SESSION_LOG_DIR`` is ``off``."""
        return os.environ.get(_DIR_ENV, "").strip().lower() != "off"

    def _ensure_directory(self) -> Path:
        if not self.enabled:
            raise OSError(f"session logs disabled by {_DIR_ENV}=off")
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
        return directory

    def open_session(self, session_id: str, agent_name: str) -> SessionLog:
        """Create and lock a new log file. Raises OSError if that is impossible."""
        directory = self._ensure_directory()
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)[:64]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        while True:
            path = directory / f"{stamp}-{safe_id}-{secrets.token_hex(4)}.jsonl"
            try:
                fd = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600
                )
                break
            except FileExistsError:
                continue
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            log = SessionLog(path, fd, session_id)
        except BaseException:
            os.close(fd)
            raise
        self._logs.add(log)
        return log

    def cleanup_once(self, now: datetime | None = None) -> list[Path]:
        """Delete unlocked logs whose last write is over 24 hours old."""
        now = now or datetime.now(timezone.utc)
        cutoff = (now - RETENTION).timestamp()
        deleted: list[Path] = []
        try:
            candidates = sorted(self.directory.glob("*.jsonl"))
        except OSError as exc:
            logger.warning("Session log cleanup failed for %s: %s", self.directory, exc)
            return deleted
        for path in candidates:
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                with open(path, "rb") as fh:
                    try:
                        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue  # held by a live writer
                    path.unlink()
                deleted.append(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("Session log cleanup failed for %s: %s", path, exc)
        return deleted

    def _sweep(self) -> None:
        try:
            self.cleanup_once()
        except Exception as exc:
            logger.warning("Session log cleanup failed: %s", exc)

    async def start(self) -> None:
        """Sweep once now, then every ``cleanup_interval`` seconds."""
        if self._task is not None or not self.enabled:
            return
        try:
            self._ensure_directory()
        except OSError as exc:
            logger.warning("Session log directory unusable (%s): %s", self.directory, exc)
        self._sweep()
        self._task = asyncio.create_task(self._periodic())

    async def _periodic(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval)
            self._sweep()

    async def aclose(self) -> None:
        """Cancel the periodic sweep and close every open log."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for log in list(self._logs):
            log.close()


def summarize_request(request: Any) -> dict[str, Any]:
    """Structural counts and a token estimate for a request; never its content."""
    try:
        system = request.system or []
        tools = request.tools or []
        chars = sum(len(b.text) for b in system)
        for msg in request.messages:
            for block in msg.content:
                if hasattr(block, "text"):
                    chars += len(block.text)
                else:
                    chars += len(json.dumps(block.model_dump(), default=str))
        chars += sum(len(t.model_dump_json()) for t in tools)
        return {
            "system_blocks": len(system),
            "messages": len(request.messages),
            "tools": len(tools),
            "request_tokens_estimate": chars // 4,
        }
    except Exception as exc:
        logger.warning("Session log request summary failed: %s", exc)
        return {}


class SessionTracer:
    """SR2 tracer that logs firings to a session log, then forwards to *inner*.

    Content before/after is deliberately omitted so static resolver text and
    compiled prompts never reach the log. The caller's own tracer (``--trace``)
    still receives every hook unchanged.
    """

    def __init__(
        self, log_provider: Callable[[], SessionLog | None], inner: Any = None
    ) -> None:
        self._log_provider = log_provider
        self._inner = inner

    def _append(self, event: str, data: dict[str, Any]) -> None:
        log = self._log_provider()
        if log is not None:
            log.append(event, data)

    def on_firing(self, record: Any) -> None:
        failed = record.status == "failed"
        self._append(
            "pipeline.error" if failed else "pipeline.firing",
            {
                "turn_seq": record.turn_seq,
                "iteration_seq": record.iteration_seq,
                "firing_seq": record.firing_seq,
                "kind": record.kind,
                "component": record.component_name,
                "layer": record.layer,
                "trigger_events": list(record.trigger_events),
                "tokens_before": record.tokens_before,
                "tokens_after": record.tokens_after,
                "tokens_delta": record.tokens_delta,
                "duration_ms": record.duration_ms,
                "status": record.status,
                "error": record.error,
            },
        )
        if self._inner is not None:
            self._inner.on_firing(record)

    def on_compile(self, request: Any) -> None:
        self._append("pipeline.compile", summarize_request(request))
        if self._inner is not None:
            self._inner.on_compile(request)
