"""One-shot SQLite-to-markdown schedule migration (spc-998, FR45).

``python -m spctr.schedules.migrate --store <sqlite-path>`` reads the
increment-2 legacy SQLite store once: every ``active`` row becomes an
``<id>.md`` file next to the database (the agent's schedule directory,
``schedules.store_path``) preserving id, label, cron, platform, channel
ID string, payload (command keys into frontmatter, the agent
instruction into the body) and the runtime keys; ``cancelled`` rows are
dropped.  The rows carry no timezone or topic column, so those FR37 keys
are omitted and fall back to the config defaults (``schedules.timezone``
/ ``schedules.zulip_topic``, FR37).

All-or-nothing (FR45: "any error leaves the SQLite file untouched"):
every row is read, parsed and rendered before a single file is written,
each file goes down with one atomic ``os.replace``, and the database is
renamed to ``<name>.sqlite.migrated`` only after the emitted files are
re-read through the store's own parser and the count matches.  A failed
migration leaves the database in place, its bytes unchanged.

Run this with the agent process stopped (FR45): nothing may be claiming
or advancing rows while they move.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from spctr.schedules.store import (
    Schedule,
    ScheduleStore,
    _parse_ts,
    _render_document,
    _runtime_list,
    _ts,
    _user_keys,
)

# FR45: the legacy table is read only; id, label, cron, platform,
# dest_channel_id, payload and on_miss are carried to the file, the
# runtime keys keep their exact UTC ISO strings.
_SELECT_ACTIVE = (
    "SELECT id, agent, platform, requester_id, requester_mention, "
    "dest_channel_id, dest_channel_name, label, cron, payload_type, "
    "payload, on_miss, next_run_at, last_run_at, created_at, status, "
    "last_status, last_error FROM schedules "
    "WHERE status = 'active' ORDER BY id"
)


class MigrationError(Exception):
    """Migration refused; the SQLite file is left untouched."""


def _row_to_schedule(raw: sqlite3.Row) -> Schedule:
    """One legacy row as a ``Schedule`` (column names match the fields)."""
    return Schedule(
        id=raw["id"],
        agent=raw["agent"],
        platform=raw["platform"],
        requester_id=raw["requester_id"],
        requester_mention=raw["requester_mention"],
        dest_channel_id=raw["dest_channel_id"],
        dest_channel_name=raw["dest_channel_name"],
        label=raw["label"],
        cron=raw["cron"],
        payload_type=raw["payload_type"],
        payload=raw["payload"],
        on_miss=raw["on_miss"],
        next_run_at=_parse_ts(raw["next_run_at"]),
        last_run_at=None if not raw["last_run_at"] else _parse_ts(raw["last_run_at"]),
        created_at=_parse_ts(raw["created_at"]),
        status=raw["status"],
        last_status=raw["last_status"],
        last_error=raw["last_error"],
    )


def _ts_text(raw: str) -> str:
    """The exact UTC ISO string for a legacy timestamp column value."""
    return _ts(_parse_ts(raw))


def _parse_row(raw: sqlite3.Row, store: ScheduleStore) -> tuple[list, str]:
    """Turn one legacy row into ``(frontmatter-user-keys, body)``.

    Rendering follows the FR37 key sets the store uses: ``_user_keys``
    orders known user keys before extras, ``_render_document`` emits the
    fences and the byte-for-byte body, ``_runtime_list`` renders the
    runtime keys in canonical order.  Raises ``MigrationError`` on a row
    whose payload cannot produce a valid file.
    """
    try:
        data = json.loads(raw["payload"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise MigrationError(
            f"row {raw['id']!r}: payload JSON does not parse: {exc}"
        ) from exc
    payload_type = raw["payload_type"]
    user: list[tuple[str, object]] = [
        ("platform", raw["platform"]),
        ("label", raw["label"]),
        ("cron", raw["cron"]),
        ("type", payload_type),
    ]
    # FR45: the channel ID string is preserved verbatim; ``_user_keys``
    # places it at its FR37 position when the file renders.
    if raw["dest_channel_id"]:
        user.append(("channel", str(raw["dest_channel_id"])))
    body = ""
    if payload_type == "command":
        if not isinstance(data, dict) or not data.get("command"):
            raise MigrationError(
                f"row {raw['id']!r}: command payload has no 'command'"
            )
        user.append(("command", str(data["command"])))
        if data.get("cwd"):
            user.append(("cwd", str(data["cwd"])))
    elif payload_type == "agent":
        if not isinstance(data, dict) or not str(data.get("instruction", "")).strip():
            raise MigrationError(
                f"row {raw['id']!r}: agent payload has no instruction"
            )
        target = str(data.get("agent") or "")
        # The owning agent is the store's (FR36), never frontmatter; the
        # key means "target" and is written only when it differs.
        if target and target != store.agent:
            user.append(("agent", target))
        body = str(data["instruction"])
    else:
        raise MigrationError(f"row {raw['id']!r}: unknown payload_type {payload_type!r}")
    user.append(("on_miss", raw["on_miss"]))
    for key in ("requester_id", "requester_mention", "created_at"):
        if raw[key]:
            value: object = raw[key]
            if key == "created_at":
                value = _ts_text(raw[key])
            user.append((key, value))
    return _user_keys(user, _row_to_schedule(raw)), body


def migrate(store_path: str | Path) -> tuple[int, str]:
    """Migrate one legacy SQLite store to markdown files (FR45).

    Returns ``(migrated_count, message)``.  Raises ``MigrationError``
    without touching the database on any error.
    """
    db = Path(store_path)
    migrated = db.with_name(db.name + ".migrated")
    if not db.exists():
        if migrated.exists():
            return 0, "nothing to migrate"
        raise MigrationError(f"no SQLite store at {db}")

    # Read-only open: migration never writes to the database, whatever
    # else goes wrong, and a missing file is an error rather than an
    # empty database.
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro&immutable=0", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise MigrationError(f"cannot open {db}: {exc}") from exc
    try:
        try:
            raw_rows = conn.execute(_SELECT_ACTIVE).fetchall()
        except sqlite3.Error as exc:
            raise MigrationError(f"cannot read {db}: {exc}") from exc
    finally:
        conn.close()

    if not raw_rows and not migrated.exists():
        # Nothing to migrate, and the legacy database is left exactly as
        # found: an empty store is not renamed away, since no file was
        # emitted to justify the handover.
        return 0, "nothing to migrate"

    # The store directory is the database's parent (FR36:
    # schedules.store_path, one directory per owning agent).
    target_dir = db.parent
    agent = raw_rows[0]["agent"] if raw_rows else ""
    store = ScheduleStore(target_dir, agent=agent)

    # Validate and render every row first (all-or-nothing): nothing is
    # written until the whole batch renders, so a bad row late in the
    # table cannot leave a half-written migration behind.
    rendered: list[tuple[str, str]] = []
    for raw in raw_rows:
        filename = f"{raw['id']}.md"
        if (target_dir / filename).exists():
            raise MigrationError(f"refusing to overwrite existing {filename}")
        user, body = _parse_row(raw, store)
        text = _render_document(user, _runtime_list(_row_to_schedule(raw)), body)
        rendered.append((filename, text))

    for filename, text in rendered:
        tmp = target_dir / f".tmp-migrate-{filename}"
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target_dir / filename)
        except OSError:
            # Mid-write failure: remove this file's temp so no partial
            # content is left; the database has not been touched and the
            # error propagates.
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    # FR45: re-read the emitted files through the store's parser and
    # match the count before the database may be renamed away.
    rows, invalid = store.scan()
    if len(rows) != len(rendered) or invalid:
        bad = ", ".join(f"{e.filename}: {e.reason}" for e in invalid)
        raise MigrationError(
            f"verification failed: expected {len(rendered)} readable "
            f"schedule files, got {len(rows)}"
            + (f" (invalid: {bad})" if bad else "")
        )
    if migrated.exists():
        raise MigrationError(f"refusing to overwrite existing {migrated}")
    os.rename(db, migrated)
    return len(rendered), f"migrated {len(rendered)} schedule(s) to {db.parent}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spctr.schedules.migrate",
        description=(
            "One-shot migration of a legacy SQLite schedule store to "
            "markdown schedule files (FR45).  Run with the agent process "
            "stopped."
        ),
    )
    parser.add_argument(
        "--store",
        required=True,
        help=(
            "Path to the legacy SQLite store (schedules.store_path).  "
            "Must run with the agent process stopped.  On success it is "
            "renamed to <name>.sqlite.migrated; on any error it is left "
            "untouched."
        ),
    )
    args = parser.parse_args(argv)
    try:
        _count, message = migrate(args.store)
    except (MigrationError, OSError, sqlite3.Error) as exc:
        # OSError too: a write or rename can still fail mid-migration
        # (read-only target directory, disk full); the database is left
        # untouched and no partial file survives our temp cleanup.
        print(f"migration failed: {exc}", file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
