"""One-shot SQLite-to-markdown migration (spc-998, FR45, AC40).

Fixture SQLite stores are built from the legacy increment-2 schema
(embedded verbatim below).  The CLI is exercised both in-process
(``main``) and through ``python -m spctr.schedules.migrate`` so the
FR45 command line itself is covered.  Migrated files are verified
through the real ``ScheduleStore`` parser (real-producer: the migration
wrote them, the shipped store reads them) plus byte-for-byte body and
key-text assertions on the raw file.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from spctr.schedules import ScheduleStore
from spctr.schedules.migrate import main

# The legacy increment-2 schedule table, embedded verbatim per contract.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
  id TEXT PRIMARY KEY,
  agent TEXT NOT NULL,
  platform TEXT NOT NULL,
  requester_id TEXT NOT NULL,
  requester_mention TEXT NOT NULL,
  dest_channel_id TEXT NOT NULL,
  dest_channel_name TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  cron TEXT NOT NULL,
  payload_type TEXT NOT NULL CHECK (payload_type IN ('command', 'agent')),
  payload TEXT NOT NULL,
  on_miss TEXT NOT NULL DEFAULT 'run_late' CHECK (on_miss IN ('run_late', 'skip')),
  next_run_at TEXT NOT NULL,
  last_run_at TEXT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'cancelled')),
  last_status TEXT NULL,
  last_error TEXT NULL
);
"""

_COLUMNS = (
    "id, agent, platform, requester_id, requester_mention, dest_channel_id, "
    "dest_channel_name, label, cron, payload_type, payload, on_miss, "
    "next_run_at, last_run_at, created_at, status, last_status, last_error"
)

_INSTRUCTION = "Review the open beads.\nThen post the summary.\n\nKeep it short.\n"


def _make_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def _insert(conn: sqlite3.Connection, **overrides) -> None:
    row = {
        "id": "s1",
        "agent": "miranda",
        "platform": "discord",
        "requester_id": "42",
        "requester_mention": "<@42>",
        "dest_channel_id": "100",
        "dest_channel_name": "general",
        "label": "morning",
        "cron": "0 7 * * *",
        "payload_type": "command",
        "payload": json.dumps({"command": "df -h", "cwd": "/tmp"}),
        "on_miss": "run_late",
        "next_run_at": "2026-10-21T07:00:00+00:00",
        "last_run_at": "2026-10-20T07:00:03+00:00",
        "created_at": "2026-10-01T00:00:00+00:00",
        "status": "active",
        "last_status": "ok",
        "last_error": None,
    }
    row.update(overrides)
    conn.execute(
        f"INSERT INTO schedules ({_COLUMNS}) VALUES "
        f"({', '.join('?' * 18)})",
        [row[name.strip()] for name in _COLUMNS.split(",")],
    )


def _fixture_store(tmp_path: Path) -> Path:
    """AC40 fixture: command row, agent row ``s1``... renamed below.

    Row ids: ``c1`` command, ``s1`` agent with the multiline
    instruction, ``x9`` cancelled.
    """
    db = tmp_path / "schedules.sqlite"
    conn = _make_db(db)
    _insert(
        conn,
        id="c1",
        payload_type="command",
        payload=json.dumps({"command": "df -h", "cwd": "/tmp"}),
    )
    _insert(
        conn,
        id="s1",
        label="brief",
        on_miss="skip",
        payload_type="agent",
        payload=json.dumps({"agent": "mirror", "instruction": _INSTRUCTION}),
        last_error=None,
    )
    _insert(conn, id="x9", status="cancelled")
    conn.commit()
    conn.close()
    return db


def _run_cli(store: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "spctr.schedules.migrate", "--store", str(store)],
        check=False,
        capture_output=True,
        text=True,
    )


class TestMigratesActiveRows:
    """AC40-r1: active rows become files; cancelled rows are dropped."""

    def test_active_rows_become_files_db_renamed(self, tmp_path):
        db = _fixture_store(tmp_path)
        rc = main(["--store", str(db)])
        assert rc == 0
        texts = {p.name: p.read_text(encoding="utf-8") for p in tmp_path.glob("*.md")}
        assert set(texts) == {"c1.md", "s1.md"}
        # DB renamed; the original name is gone.
        assert not db.exists()
        moved = tmp_path / "schedules.sqlite.migrated"
        assert moved.is_file()

    def test_migrated_files_load_through_the_real_store(self, tmp_path):
        db = _fixture_store(tmp_path)
        assert main(["--store", str(db)]) == 0
        store = ScheduleStore(tmp_path, agent="miranda")
        rows, invalid = store.scan()
        assert invalid == []
        assert sorted(r.id for r in rows) == ["c1", "s1"]
        by_id = {r.id: r for r in rows}

        c1 = by_id["c1"]
        assert (c1.label, c1.cron, c1.platform) == ("morning", "0 7 * * *", "discord")
        assert c1.on_miss == "run_late"
        # FR38: a hand-authored or migrated `channel:` that is not a
        # snowflake ID resolves through the platform seam at load (the
        # store without an interface seam hands back empty), so the
        # store-loaded value cannot pin the ID string; the raw file test
        # below pins it instead.
        assert json.loads(c1.payload) == {"command": "df -h", "cwd": "/tmp"}
        # Runtime state carried exactly.
        assert c1.next_run_at.isoformat() == "2026-10-21T07:00:00+00:00"
        assert c1.last_run_at.isoformat() == "2026-10-20T07:00:03+00:00"
        assert c1.last_status == "ok"
        assert c1.last_error is None
        # Agent row: target agent preserved, instruction is the body.
        s1 = by_id["s1"]
        assert json.loads(s1.payload) == {"agent": "mirror", "instruction": _INSTRUCTION}
        assert s1.on_miss == "skip"
        assert s1.next_run_at.isoformat() == "2026-10-21T07:00:00+00:00"
        # Cancelled row dropped entirely.
        assert "x9" not in by_id

    def test_s1_body_is_the_instruction_byte_for_byte(self, tmp_path):
        db = _fixture_store(tmp_path)
        assert main(["--store", str(db)]) == 0
        text = (tmp_path / "s1.md").read_text(encoding="utf-8")
        # Body after the closing fence: exactly one separator newline,
        # then the instruction byte-for-byte (contract: store semantics).
        assert text.endswith("\n---\n\n" + _INSTRUCTION)
        # The target agent is in frontmatter.
        assert "\nagent: mirror\n" in text

    def test_runtime_keys_are_the_exact_utc_iso_strings(self, tmp_path):
        db = _fixture_store(tmp_path)
        assert main(["--store", str(db)]) == 0
        text = (tmp_path / "c1.md").read_text(encoding="utf-8")
        # Channel ID string preserved verbatim in frontmatter.
        assert "\nchannel: '100'\n" in text
        assert "next_run_at: '2026-10-21T07:00:00+00:00'" in text
        assert "last_run_at: '2026-10-20T07:00:03+00:00'" in text
        assert "last_status: ok" in text
        # last_error None -> the store's empty rendering.
        assert "last_error: ''" in text

    def test_no_timezone_or_topic_keys_are_invented(self, tmp_path):
        db = _fixture_store(tmp_path)
        assert main(["--store", str(db)]) == 0
        for name in ("c1.md", "s1.md"):
            text = (tmp_path / name).read_text(encoding="utf-8")
            head = text.split("\n---\n")[0]
            assert "\ntimezone:" not in head + "\n"
            assert "\ntopic:" not in head + "\n"

    def test_cli_module_invocation_matches_contract(self, tmp_path):
        db = _fixture_store(tmp_path)
        proc = _run_cli(db)
        assert proc.returncode == 0, proc.stderr
        assert (tmp_path / "schedules.sqlite.migrated").is_file()
        assert {p.name for p in tmp_path.glob("*.md")} == {"c1.md", "s1.md"}


class TestRerunIsANoOp:
    """AC40-r2: re-running exits 0 printing 'nothing to migrate'."""

    def test_rerun_prints_nothing_to_migrate(self, tmp_path):
        db = _fixture_store(tmp_path)
        assert main(["--store", str(db)]) == 0
        proc = _run_cli(db)
        assert proc.returncode == 0, proc.stderr
        assert "nothing to migrate" in proc.stdout
        # The migrated database stays put; nothing is renamed again.
        assert not db.exists()
        assert (tmp_path / "schedules.sqlite.migrated").is_file()

    def test_rerun_writes_nothing(self, tmp_path):
        db = _fixture_store(tmp_path)
        assert main(["--store", str(db)]) == 0
        before = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(tmp_path.iterdir())
            if p.name.endswith((".md", ".migrated"))
        }
        assert main(["--store", str(db)]) == 0
        after = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(tmp_path.iterdir())
            if p.name.endswith((".md", ".migrated"))
        }
        assert after == before


class TestFailureLeavesDatabaseUntouched:
    """AC40-r3/r4: any error leaves the SQLite file untouched."""

    def test_read_only_target_dir_leaves_db_and_no_files(self, tmp_path, request):
        db = _fixture_store(tmp_path)
        db_bytes = db.read_bytes()
        if os.geteuid() == 0:
            # Root ignores directory write permissions; skip rather than
            # assert a denial that cannot happen in this container.
            pytest.skip("chmod 555 does not deny writes to root")
        os.chmod(tmp_path, 0o555)
        try:
            rc = main(["--store", str(db)])
        finally:
            os.chmod(tmp_path, 0o755)
        assert rc == 1
        assert db.is_file()
        assert db.read_bytes() == db_bytes
        assert not (tmp_path / "schedules.sqlite.migrated").exists()
        # No .md written; at most a sweepable .tmp-migrate-* leftover.
        for p in tmp_path.iterdir():
            if p.name.endswith(".md"):
                pytest.fail(f"migration wrote {p.name} despite failing")

    def test_forced_replace_failure_mid_migration(self, tmp_path, monkeypatch):
        # First file lands, the second replace fails: the DB must stay
        # untouched and un-renamed, and the error is reported.
        db = _fixture_store(tmp_path)
        db_bytes = db.read_bytes()
        real_replace = os.replace
        calls = {"n": 0}

        def flaky_replace(src, dst, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError(28, "No space left on device")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr("os.replace", flaky_replace)
        rc = main(["--store", str(db)])
        assert rc == 1
        assert db.is_file() and db.read_bytes() == db_bytes
        assert not (tmp_path / "schedules.sqlite.migrated").exists()
        # No .tmp-migrate-* survives the failed run.
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-migrate-")]
        assert leftovers == []

    def test_rename_waits_for_a_matching_re_read(self, tmp_path, monkeypatch):
        # FR45: the database is renamed "only after re-reading the
        # emitted files and matching the count".  With that re-read
        # rigged to see a mismatch, a compliant migration must NOT
        # rename; the rename target must also still exist at rename
        # time (i.e. the re-read happened before it, not after).
        db = _fixture_store(tmp_path)
        db_bytes = db.read_bytes()
        seen = {"existed_at_scan": None}

        def rigged_scan(self):
            # Observe ordering: the original DB must still exist here.
            seen["existed_at_scan"] = db.exists()
            return [], []  # pretend nothing landed

        monkeypatch.setattr(ScheduleStore, "scan", rigged_scan)
        rc = main(["--store", str(db)])
        assert rc == 1
        assert seen["existed_at_scan"] is True
        assert db.is_file() and db.read_bytes() == db_bytes
        assert not (tmp_path / "schedules.sqlite.migrated").exists()

    def test_malformed_payload_json_errors_db_untouched(self, tmp_path):
        db = tmp_path / "schedules.sqlite"
        conn = _make_db(db)
        _insert(conn, id="c1")
        _insert(conn, id="s2", payload_type="agent", payload="{not json")
        conn.commit()
        conn.close()
        db_bytes = db.read_bytes()
        rc = main(["--store", str(db)])
        assert rc == 1
        assert db.is_file() and db.read_bytes() == db_bytes
        assert not (tmp_path / "schedules.sqlite.migrated").exists()
        # All-or-nothing: even the valid row emitted no file.
        assert list(tmp_path.glob("*.md")) == []

    def test_missing_store_exits_nonzero(self, tmp_path):
        rc = main(["--store", str(tmp_path / "nope.sqlite")])
        assert rc == 1

    def test_empty_store_reports_nothing_and_keeps_db(self, tmp_path):
        db = tmp_path / "schedules.sqlite"
        conn = _make_db(db)
        conn.commit()
        conn.close()
        assert main(["--store", str(db)]) == 0
        assert db.is_file()  # nothing was emitted; the db is left alone
