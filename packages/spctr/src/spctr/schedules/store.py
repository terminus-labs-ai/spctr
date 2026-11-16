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

from spctr.schedules.cron import CronParseError, parse_cron

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
    "payload_agent",
    "created_at",
)
RUNTIME_KEYS = ("next_run_at", "last_run_at", "last_status", "last_error")

# Leftover mutation temps (possibly orphaned by a kill between temp write
# and os.replace); swept after one poll interval (FR41).  The lockfile is
# NOT matched: deleting it would let two writers hold different inodes.
_TEMP_RE = re.compile(r"^\.tmp-")


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


def _parse_ts(raw) -> datetime:
    # yaml.safe_load turns unquoted ISO timestamps into datetime objects;
    # quoted scalars come back as strings.
    if isinstance(raw, datetime):
        value = raw
    else:
        value = datetime.fromisoformat(str(raw))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _split_document(text: str) -> dict[str, str] | None:
    """Split one file into frontmatter head and body.

    Returns ``None`` when the file does not start with a ``---`` fence or
    has no closing fence.  The body is everything after the second fence
    line, byte-for-byte; a missing body is empty.
    """
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        # A closing fence at EOF (no trailing newline) is valid; the
        # body is empty.
        if text.endswith("\n---"):
            return {"head": text[4 : len(text) - 4], "body": ""}
        return None
    head = text[4:end]
    # Body: everything after the closing fence line, byte-for-byte, minus
    # exactly one separator newline (the blank line the renderer puts
    # between the fence and the body).
    rest = text[end + len("\n---\n") :]
    body = rest[1:] if rest.startswith("\n") else rest
    return {"head": head, "body": body}


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
        # The payload target is ``payload_agent`` when the file carries an
        # explicit one; otherwise the file's ``agent`` (owning agent)
        # default, exactly as increment 2 stored it.
        target = values.get("payload_agent")
        if not target:
            target = values.get("agent", "") or ""
        return json.dumps(
            {"agent": target, "instruction": body}, ensure_ascii=False
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
        except (OSError, UnicodeDecodeError):
            return None

    def sweep_temp_files(self, now: datetime, poll_seconds: float) -> None:
        """Delete leftover temp files older than one poll interval (FR41).

        A process killed between temp write and ``os.replace`` leaves the
        original readable and its temp file here.  The engine calls this
        on every poll with the store's poll interval.
        """
        cutoff = now.timestamp() - poll_seconds
        with self._lock():
            try:
                entries = list(self._dir.iterdir())
            except FileNotFoundError:
                return
            for entry in entries:
                if not _TEMP_RE.match(entry.name):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except FileNotFoundError:
                    continue
                if mtime <= cutoff:
                    try:
                        entry.unlink()
                    except OSError:
                        pass

    # ---- parse / render one file ----

    def _parse_file(self, filename: str, text: str | None) -> tuple[Schedule | None, str | None, dict | None]:
        """Parse one schedule file.

        Returns ``(schedule, None, extra)`` when the file is valid, or
        ``(None, reason, None)`` when it is not (FR40).  ``extra`` is the
        ordered key/value list of user-space keys as written, for use by
        the writer; it carries the file's user keys (known plus unknown
        extras) with their on-disk ordering preserved.

        A file with no ``next_run_at`` runtime key is VALID with
        ``next_run_at=None`` — a first-seen hand-authored file the engine
        schedules on its next poll (FR39).  ``enabled: false`` parses to
        ``status="paused"`` — valid, but never due, never claimed and
        never advanced (FR44).
        """
        stem = filename[:-3]
        if not _ID_STEM_RE.match(stem):
            return None, "invalid id stem", None
        if text is None:
            return None, "unreadable or not UTF-8", None
        if len(text.encode("utf-8")) > 64 * 1024:
            return None, "file exceeds 64 KiB", None
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
        if not data.get("cron"):
            return None, "missing required key 'cron'", None
        try:
            parse_cron(str(data["cron"]))
        except CronParseError:
            return None, f"invalid cron expression {data['cron']!r}", None
        payload_type = data.get("type", "")
        if payload_type not in ("command", "agent"):
            return None, f"unknown type {payload_type!r}", None
        if payload_type == "command" and not data.get("command"):
            return None, "command schedule missing 'command'", None
        if payload_type == "agent":
            if not doc["body"].strip():
                return None, "agent schedule has empty body", None
            if len(doc["body"]) > 4000:
                return None, "agent instruction exceeds 4000 characters", None
        # An explicit target agent renders as ``payload_agent``; the plain
        # ``agent`` key is the owning agent, never the payload target.
        if "payload_agent" in data and not isinstance(data["payload_agent"], str):
            return None, "invalid payload_agent value", None
        on_miss = data.get("on_miss", "run_late")
        if on_miss not in ("run_late", "skip"):
            return None, f"unknown on_miss {on_miss!r}", None
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            return None, f"unknown enabled {enabled!r}", None
        platform = data.get("platform", "")
        if not platform:
            return None, "platform required", None
        if platform not in ("discord", "zulip"):
            return None, f"invalid platform {platform!r}", None
        raw_next = data.get("next_run_at")
        if raw_next is None or raw_next in ("", "null"):
            next_run_at = None
        else:
            try:
                next_run_at = _parse_ts(raw_next)
            except (ValueError, TypeError):
                return None, f"invalid next_run_at {raw_next!r}", None
        last_run_at = None
        if data.get("last_run_at"):
            try:
                last_run_at = _parse_ts(data["last_run_at"])
            except (ValueError, TypeError):
                return None, f"invalid last_run_at {data['last_run_at']!r}", None
        created_at = None
        if data.get("created_at"):
            try:
                created_at = _parse_ts(data["created_at"])
            except (ValueError, TypeError):
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
            status="paused" if not enabled else "active",
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
        body = ""
        if row.payload_type == "command":
            payload = json.loads(row.payload)
            if payload.get("command"):
                values["command"] = payload["command"]
            if payload.get("cwd"):
                values["cwd"] = payload["cwd"]
        if row.payload_type == "agent":
            payload = json.loads(row.payload)
            # An explicit target agent renders as payload_agent; the
            # default (owning agent) round-trips through the agent key.
            if payload.get("agent"):
                values["payload_agent"] = payload["agent"]
            body = payload.get("instruction", "")
        if row.created_at is not None:
            values["created_at"] = _ts(row.created_at)
        ordered = [(k, values[k]) for k in USER_KEYS if k in values]
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
        """Insert an active schedule. Returns the assigned ``s<N>`` ID.

        The incoming ``row.id`` is a placeholder and ignored: the ID is
        the next free ``s<N>`` stem under the exclusive lock (highest
        numeric suffix among existing ``s<N>.md`` files + 1, starting at
        ``s1``), so a create never overwrites an existing file.  The
        FR43 label→stem generation for hand-meaningful names is a later
        slice.
        """
        text = self._render_row(row)
        with self._lock():
            next_n = 1
            for path in self._dir.glob("*.md"):
                stem = path.name[:-3]
                if not stem.startswith("s"):
                    continue
                try:
                    next_n = max(next_n, int(stem[1:]) + 1)
                except ValueError:
                    continue
            new_id = f"s{next_n}"
            self._replace(f"{new_id}.md", text)
        return new_id

    def due(
        self,
        now: datetime,
        platform: str | None = None,
    ) -> list[Schedule]:
        """Active schedules whose next_run_at is at or before *now*."""
        rows = [
            row
            for row in self._scan()
            if row.status == "active"
            and row.next_run_at is not None
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
            if (
                row.status != "active"
                or row.next_run_at is None
                or _ts(row.next_run_at) == CLAIM_SENTINEL
            ):
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
                # File deleted while its payload ran (FR42): skip the
                # result write; the run itself already completed.
                if reason == "missing":
                    logger.info(
                        "schedule result write skipped, file gone: id=%s",
                        schedule_id,
                    )
                return
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
        """Delete the file (FR32); deleting the file cancels the schedule.

        Returns False only when there is no file to remove — a paused
        (``enabled: false``) file is still a schedule and is removed too
        (FR44).
        """
        filename = f"{schedule_id}.md"
        with self._lock():
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
        rows = [
            row
            for row in self._scan()
            if row.status == "active"
            and row.next_run_at is not None
            and _ts(row.next_run_at) == CLAIM_SENTINEL
            and (platform is None or row.platform == platform)
        ]
        return rows

    def list_for_agent(
        self, agent: str, statuses: Iterable[str] = ("active",)
    ) -> list[Schedule]:
        statuses = tuple(statuses)
        rows = [
            row
            for row in self._scan()
            if row.agent == agent and row.status in statuses
        ]
        return sorted(
            rows,
            key=lambda r: (
                r.next_run_at is not None,
                r.next_run_at or datetime.max.replace(tzinfo=timezone.utc),
                r.id,
            ),
        )

    def _scan(self) -> list[Schedule]:
        """The valid schedules from one directory scan."""
        return self.scan()[0]

    def scan(self) -> tuple[list[Schedule], list[InvalidEntry]]:
        """One directory scan: valid Schedules plus invalid (filename, reason) pairs.

        The engine calls this on every poll (FR39/FR40); reporting an
        ERROR line per invalid file at most once per content hash is the
        caller's job.  Invalid files never run, never claim, and do not
        affect their siblings.
        """
        valid: list[Schedule] = []
        invalid: list[InvalidEntry] = []
        with self._lock(exclusive=False):
            for path in sorted(self._dir.glob("*.md"), key=lambda p: p.name):
                if not path.is_file() or _TEMP_RE.match(path.name):
                    continue
                row, reason, _extra = self._parse_file(path.name, self._read_file(path.name))
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
