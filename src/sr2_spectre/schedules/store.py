"""Markdown-file persistence for schedules (spc-994, FR36–FR41).

One ``.md`` file per schedule in one directory: YAML frontmatter between
``---`` fence lines carries the config, the markdown body after the
second fence carries the prompt (agent payload) or free-form notes
(command payload).  The SQLite store is replaced outright; the public
contract (``Schedule`` dataclass, one-path constructor, method names and
signatures) is unchanged so the engine, executors, tool and interface
wiring compile untouched.

Concurrent writers are serialized by an advisory ``flock`` on a lockfile
in the directory (FR41); every mutation writes a temp file in the same
directory and ``os.replace``s it over the target so readers never see a
partial file.  Leftover temp files older than one poll interval are
deleted at scan time (``sweep_temp_files``); the poll interval is
carried by the engine.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

logger = logging.getLogger(__name__)

# Claim sentinel (FR41): the exact string a claimed row carries in
# ``next_run_at``; startup recovery treats a sentinel-stuck file as a
# payload killed mid-run.
CLAIM_SENTINEL = "9999-12-31T23:59:59+00:00"

# FR43: a stem is valid when it matches this (stems are lowercase;
# uppercase stems are invalid files, never silently folded).
_ID_STEM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# FR40: keys the store owns.  User keys are config; runtime keys are
# engine-owned and rewritten after each run.  Any other key is
# "user-space": accepted and preserved, never clobbered.
USER_KEYS = (
    "agent",
    "platform",
    "requester_id",
    "requester_mention",
    "dest_channel_id",
    "dest_channel_name",
    "label",
    "cron",
    "type",
    "command",
    "cwd",
    "on_miss",
)
RUNTIME_KEYS = ("next_run_at", "last_run_at", "last_status", "last_error")

# Files matching this are never schedules: store bookkeeping, temp files
# (possibly orphaned) and non-markdown files (FR36).
_NON_SCHEDULE_RE = re.compile(r"^\.(lock|tmp-[^.]*\.\.lock|tmp-)(.+)?$")


@dataclass(frozen=True)
class Schedule:
    """One stored schedule row."""

    id: str
    agent: str
    platform: str
    requester_id: str
    requester_mention: str
    dest_channel_id: str
    dest_channel_name: str
    label: str
    cron: str
    payload_type: str
    payload: str
    on_miss: str
    next_run_at: datetime
    last_run_at: datetime | None
    created_at: datetime
    status: str
    last_status: str | None
    last_error: str | None


@dataclass(frozen=True)
class InvalidEntry:
    """A file the scan could not parse as a Schedule (FR40)."""

    filename: str
    reason: str


def _ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_ts(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _split_document(text: str) -> dict[str, str] | None:
    """Split one file into frontmatter head and body.

    Returns ``None`` when the file does not start with a ``---`` fence or
    has no closing fence.  The body is everything after the second fence
    line, byte-for-byte; a missing body is empty.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            head = "\n".join(lines[1:idx])
            # Body: everything after the closing fence line, byte-for-byte.
            body = text.split("\n", idx + 1)[1][idx + 1 :].lstrip("\n")
            return {"head": head, "body": body}
    return None


def _render_scalar(key: str, value: object) -> str:
    """Render one frontmatter key as a single line, round-trippable.

    ``safe_dump`` emits a YAML ``...`` document-end marker on plain
    scalar values that is invalid mid-document; a ``---``/``...`` dump
    minus the fences is the faithful one-line form, keeping whatever
    quoting YAML chose (quoted strings, block scalars, ...) exactly.
    """
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True)
    lines = text.split("\n")
    if lines and lines[0] == "---":
        lines = lines[1:]
    if lines and lines[-1] in ("", "..."):
        while lines and lines[-1] in ("", "..."):
            lines.pop()
    return key + ": " + "\n".join(lines)


def _render_document(user: list[tuple[str, object]], runtime: list[tuple[str, str]], body: str) -> str:
    """Render ``---`` fences, user keys, runtime keys, blank line, body.

    User keys render in their recorded order, then runtime keys in the
    canonical order; the body is emitted byte-for-byte, never re-rendered
    through a markdown library.
    """
    parts = ["---"]
    for key, value in user:
        parts.append(_render_scalar(key, value))
    for key, value in runtime:
        parts.append(_render_scalar(key, value))
    parts.append("---")
    if body:
        parts.append("")
    parts.append(body)
    return "\n".join(parts)


def _payload_json(payload_type: str, user: list[tuple[str, object]], body: str) -> str:
    """Reconstruct the increment-2 JSON ``payload`` string from the file."""
    values = dict(user)
    if payload_type == "agent":
        return json.dumps(
            {"agent": values.get("agent_target", "") or "", "instruction": body},
            ensure_ascii=False,
        )
    return json.dumps({"command": values.get("command", "") or "", "cwd": values.get("cwd")})


def _user_keys(user: list[tuple[str, object]], row: Schedule) -> list[tuple[str, object]]:
    """Order the file's user keys as: known user keys, then extras."""
    values = dict(user)
    ordered: list[tuple[str, object]] = []
    for key in USER_KEYS:
        if key in values:
            ordered.append((key, values[key]))
    seen = {key for key, _ in ordered}
    for key, value in user:
        if key not in seen:
            ordered.append((key, value))
    return ordered


def _runtime_list(row: Schedule) -> list[tuple[str, str]]:
    return [
        ("next_run_at", _ts(row.next_run_at)),
        ("last_run_at", "" if row.last_run_at is None else _ts(row.last_run_at)),
        ("last_status", row.last_status or ""),
        ("last_error", row.last_error or ""),
    ]


class ScheduleStore:
    """Directory-backed markdown store with async facades.

    Args:
        path: Directory holding one ``<id>.md`` file per schedule;
            created on demand along with its parents.
    """

    def __init__(self, path: str | Path) -> None:
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._dir / ".lock"

    # ---- locking and atomic file IO ----

    @contextmanager
    def _lock(self, exclusive: bool = True):
        """Advisory ``flock`` on the directory lockfile (FR41).

        The lockfile is created lazily and never removed: deleting it
        would let two writers hold different inodes and both believe they
        hold the lock.
        """
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _replace(self, filename: str, content: str) -> None:
        """Write *content* to *filename* atomically (temp + ``os.replace``)."""
        tmp = self._dir / f".tmp-{os.getpid()}-{secrets.token_hex(4)}-{filename}"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, self._dir / filename)

    def _read_file(self, filename: str) -> str | None:
        try:
            return (self._dir / filename).read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            return None

    def sweep_temp_files(self, now: datetime, poll_seconds: float) -> None:
        """Delete leftover temp files older than one poll interval (FR41).

        A process killed between temp write and ``os.replace`` leaves the
        original readable and its temp file here.  The engine calls this
        on every poll with the store's poll interval.
        """
        cutoff = now.timestamp() - poll_seconds
        try:
            entries = list(self._dir.iterdir())
        except FileNotFoundError:
            return
        for entry in entries:
            if not _NON_SCHEDULE_RE.match(entry.name):
                continue
            try:
                mtime = entry.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime <= cutoff:
                try:
                    entry.unlink()
                except FileNotFoundError:
                    pass

    # ---- parse / render one file ----

    def _parse_file(self, filename: str, text: str) -> tuple[Schedule | None, str | None, dict | None]:
        """Parse one schedule file.

        Returns ``(schedule, None, extra)`` when the file is valid, or
        ``(None, reason, None)`` when it is not (FR40).  ``extra`` is the
        ordered key/value list of user-space keys as written, for use by
        the writer; it carries the file's user keys (known plus unknown
        extras) with their on-disk ordering preserved.
        """
        stem = filename[:-3]
        if not _ID_STEM_RE.match(stem):
            return None, "invalid id stem", None
        doc = _split_document(text)
        if doc is None:
            return None, "no frontmatter fences", None
        try:
            data = yaml.safe_load(doc["head"]) or {}
        except yaml.YAMLError as exc:
            return None, f"frontmatter does not parse: {exc}", None
        if not isinstance(data, dict):
            return None, "frontmatter is not a mapping", None
        # User-space keys (user + unknown extras, minus runtime keys) in
        # file order, so a rewrite preserves their order byte-for-byte.
        seen = set()
        ordered: list[tuple[str, object]] = []
        for k, v in data.items():
            if k in RUNTIME_KEYS or k in seen:
                continue
            ordered.append((k, v))
            seen.add(k)
        if "cron" not in data:
            return None, "missing required key 'cron'", None
        payload_type = data.get("type", "")
        if payload_type not in ("command", "agent"):
            return None, f"unknown type {payload_type!r}", None
        if payload_type == "command" and not data.get("command"):
            return None, "command schedule missing 'command'", None
        if payload_type == "agent" and not doc["body"].strip():
            return None, "agent schedule has empty body", None
        on_miss = data.get("on_miss", "run_late")
        if on_miss not in ("run_late", "skip"):
            return None, f"unknown on_miss {on_miss!r}", None
        platform = data.get("platform", "")
        if not platform:
            return None, "missing required key 'platform'", None
        raw_next = data.get("next_run_at")
        if raw_next is None or not isinstance(raw_next, str) or raw_next in ("", "null"):
            return None, "missing required key 'next_run_at'", None
        try:
            next_run_at = _parse_ts(raw_next)
        except ValueError:
            return None, f"invalid next_run_at {raw_next!r}", None
        last_run_at = None
        if data.get("last_run_at"):
            try:
                last_run_at = _parse_ts(str(data["last_run_at"]))
            except ValueError:
                return None, f"invalid last_run_at {data['last_run_at']!r}", None
        created_at = None
        if data.get("created_at"):
            try:
                created_at = _parse_ts(str(data["created_at"]))
            except ValueError:
                return None, f"invalid created_at {data['created_at']!r}", None
        row = Schedule(
            id=stem,
            agent=str(data.get("agent", "")),
            platform=str(platform),
            requester_id=str(data.get("requester_id", "")),
            requester_mention=str(data.get("requester_mention", "")),
            dest_channel_id=str(data.get("dest_channel_id", "")),
            dest_channel_name=str(data.get("dest_channel_name", "")),
            label=str(data.get("label", "")),
            cron=str(data["cron"]),
            payload_type=str(payload_type),
            payload=_payload_json(payload_type, ordered, doc["body"]),
            on_miss=str(on_miss),
            next_run_at=next_run_at,
            last_run_at=last_run_at,
            created_at=created_at or next_run_at,
            status="active",
            last_status=(str(data["last_status"]) if data.get("last_status") else None),
            last_error=(str(data["last_error"]) if data.get("last_error") else None),
        )
        return row, None, {"user": ordered, "body": doc["body"]}

    def _render_row(self, row: Schedule) -> str:
        """Render a fresh file from a Schedule (create path)."""
        values = dict(
            agent=row.agent,
            platform=row.platform,
            requester_id=row.requester_id,
            requester_mention=row.requester_mention,
            dest_channel_id=row.dest_channel_id,
            dest_channel_name=row.dest_channel_name,
            label=row.label,
            cron=row.cron,
            type=row.payload_type,
            on_miss=row.on_miss,
        )
        if row.payload_type == "command":
            payload = json.loads(row.payload)
            if payload.get("command"):
                values["command"] = payload["command"]
            if payload.get("cwd"):
                values["cwd"] = payload["cwd"]
        else:
            values["agent_target"] = json.loads(row.payload).get("agent", "")
        ordered = [(k, values[k]) for k in USER_KEYS if k in values]
        body = ""
        if row.payload_type == "agent":
            body = json.loads(row.payload).get("instruction", "")
            if body:
                body += "\n"
        return _render_document(ordered, _runtime_list(row), body)

    def _load_one(self, filename: str) -> Schedule | None:
        text = self._read_file(filename)
        if text is None:
            return None
        row, _reason, _extra = self._parse_file(filename, text)
        return row

    def _load_mutating(self, filename: str):
        """Load one file for a mutation: returns (row, extra) or (None, reason)."""
        text = self._read_file(filename)
        if text is None:
            return None, "missing"
        row, reason, extra = self._parse_file(filename, text)
        if row is None:
            return None, reason
        return (row, extra), None

    def _rewrite(self, filename: str, row: Schedule, extra: dict) -> None:
        user = _user_keys(extra["user"], row)
        self._replace(filename, _render_document(user, _runtime_list(row), extra["body"]))

    # ---- sync core ----

    def create(self, row: Schedule) -> str:
        """Write ``<row.id>.md`` (FR43 stem generation is a later slice).

        Returns the ID that was written.
        """
        if not _ID_STEM_RE.match(row.id):
            raise ValueError(f"invalid id {row.id!r}")
        text = self._render_row(row)
        with self._lock():
            self._replace(f"{row.id}.md", text)
        return row.id

    def due(
        self,
        now: datetime,
        platform: str | None = None,
    ) -> list[Schedule]:
        """Active schedules whose next_run_at is at or before *now*."""
        with self._lock(exclusive=False):
            rows = [
                row
                for row in self._scan()
                if row is not None
                and row.status == "active"
                and row.next_run_at <= now
                and (platform is None or row.platform == platform)
            ]
        return sorted(rows, key=lambda r: (r.next_run_at, r.id))

    def claim(self, schedule_id: str, platform: str | None = None) -> bool:
        """Atomically mark one file as running (platform-scoped).

        Returns True only if this call won the claim.  The claim flips
        ``next_run_at`` to the sentinel atomically under the lock; a
        second call sees the sentinel and returns False.
        """
        filename = f"{schedule_id}.md"
        with self._lock():
            loaded, reason = self._load_mutating(filename)
            if loaded is None:
                return False
            row, extra = loaded
            if row.status != "active" or _ts(row.next_run_at) == CLAIM_SENTINEL:
                return False
            if platform is not None and row.platform != platform:
                return False
            new_row = replace(row, next_run_at=datetime.fromisoformat(CLAIM_SENTINEL))
            self._rewrite(filename, new_row, extra)
            return True

    def advance_after_run(
        self,
        schedule_id: str,
        next_run_at: datetime,
        last_run_at: datetime,
        last_status: str,
        last_error: str | None = None,
    ) -> None:
        """After a run completes, set next_run_at and record the outcome."""
        filename = f"{schedule_id}.md"
        with self._lock():
            loaded, reason = self._load_mutating(filename)
            if loaded is None:
                return  # file deleted while its payload ran (FR42): skip the write
            row, extra = loaded
            new_row = replace(
                row,
                next_run_at=next_run_at,
                last_run_at=last_run_at,
                last_status=last_status,
                last_error=last_error,
            )
            self._rewrite(filename, new_row, extra)

    def cancel(self, schedule_id: str) -> bool:
        """Delete the file (FR32).  Returns True if it was active."""
        filename = f"{schedule_id}.md"
        with self._lock():
            loaded, reason = self._load_mutating(filename)
            if loaded is None:
                return False
            row, _extra = loaded
            if row.status != "active":
                return False
            try:
                (self._dir / filename).unlink()
            except FileNotFoundError:
                return False
            return True

    def get(self, schedule_id: str) -> Schedule | None:
        return self._load_one(f"{schedule_id}.md")

    def recover_claimed(self, now: datetime, platform: str | None = None) -> list[Schedule]:
        """On start, find rows stuck at the sentinel (crashed mid-run).

        These are schedules whose ``next_run_at`` was advanced to the
        sentinel but never restored — the process died mid-execution.
        They are returned so the engine can treat them as missed
        occurrences.
        """
        with self._lock(exclusive=False):
            rows = [
                row
                for row in self._scan()
                if row is not None
                and row.status == "active"
                and _ts(row.next_run_at) == CLAIM_SENTINEL
                and (platform is None or row.platform == platform)
            ]
        return rows

    def list_for_agent(
        self, agent: str, statuses: Iterable[str] = ("active",)
    ) -> list[Schedule]:
        statuses = tuple(statuses)
        with self._lock(exclusive=False):
            rows = [
                row
                for row in self._scan()
                if row is not None and row.agent == agent and row.status in statuses
            ]
        return sorted(rows, key=lambda r: (r.next_run_at, r.id))

    def _scan(self) -> list[Schedule | None]:
        """One directory scan: one entry per ``*.md`` file, None when invalid."""
        result: list[Schedule | None] = []
        for path in sorted(self._dir.glob("*.md"), key=lambda p: p.name):
            row, reason, _extra = self._parse_file(path.name, self._read_file(path.name) or "")
            if row is None:
                logger.error("schedule file invalid: file=%s reason=%s", path.name, reason)
            result.append(row)
        return result

    def scan(self) -> tuple[list[Schedule], list[InvalidEntry]]:
        """One directory scan: valid Schedules plus invalid (filename, reason) pairs.

        The engine calls this on every poll (FR39/FR40).  Invalid files
        never run, never claim, and do not affect their siblings.
        """
        valid: list[Schedule] = []
        invalid: list[InvalidEntry] = []
        for path in sorted(self._dir.glob("*.md"), key=lambda p: p.name):
            row, reason, _extra = self._parse_file(path.name, self._read_file(path.name) or "")
            if row is None:
                invalid.append(InvalidEntry(filename=path.name, reason=reason))
            else:
                valid.append(row)
        return valid, invalid

    # ---- async facades ----

    async def create_async(self, row: Schedule) -> str:
        return await asyncio.to_thread(self.create, row)

    async def due_async(self, now: datetime, platform: str | None = None) -> list[Schedule]:
        return await asyncio.to_thread(self.due, now, platform)

    async def claim_async(self, schedule_id: str, platform: str | None = None) -> bool:
        return await asyncio.to_thread(self.claim, schedule_id, platform)

    async def advance_after_run_async(
        self,
        schedule_id: str,
        next_run_at: datetime,
        last_run_at: datetime,
        last_status: str,
        last_error: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.advance_after_run, schedule_id, next_run_at, last_run_at, last_status, last_error
        )

    async def cancel_async(self, schedule_id: str) -> bool:
        return await asyncio.to_thread(self.cancel, schedule_id)

    async def get_async(self, schedule_id: str) -> Schedule | None:
        return await asyncio.to_thread(self.get, schedule_id)

    async def recover_claimed_async(
        self, now: datetime, platform: str | None = None
    ) -> list[Schedule]:
        return await asyncio.to_thread(self.recover_claimed, now, platform)

    async def list_for_agent_async(
        self, agent: str, statuses: Iterable[str] = ("active",)
    ) -> list[Schedule]:
        return await asyncio.to_thread(self.list_for_agent, agent, statuses)
