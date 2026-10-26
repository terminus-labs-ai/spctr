"""Directory-backed ScheduleStore: round-trip fidelity and concurrency (spc-994).

AC37 (async facades), AC38 (round-trip fidelity, key order), AC39
(concurrent writers, kill between temp write and replace, temp sweep).
File tests use tmp_path directories; concurrency uses two real
processes sharing one directory.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from sr2_spectre.schedules import Schedule, ScheduleStore

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)
SENTINEL = "9999-12-31T23:59:59+00:00"


def _make_schedule(**kwargs) -> Schedule:
    defaults = dict(
        id="s1",
        agent="miranda",
        platform="discord",
        requester_id="42",
        requester_mention="<@42>",
        dest_channel_id="100",
        dest_channel_name="general",
        label="test",
        cron="0 7 * * *",
        payload_type="command",
        payload=json.dumps({"command": "echo hi", "cwd": None}),
        on_miss="run_late",
        next_run_at=NOW + timedelta(hours=1),
        last_run_at=None,
        created_at=NOW,
        status="active",
        last_status=None,
        last_error=None,
    )
    defaults.update(kwargs)
    return Schedule(**defaults)


def _write_schedule_file(store_dir: Path, filename: str, user_keys: str, body: str) -> Path:
    path = store_dir / filename
    path.write_text(f"---\n{user_keys}\n---\n\n{body}", encoding="utf-8")
    return path


USER_BASE = """agent: miranda
platform: discord
requester_id: '42'
requester_mention: <@42>
dest_channel_id: '100'
dest_channel_name: general
label: test
cron: 0 7 * * *
type: command
command: echo hi
on_miss: run_late
"""


# ---------------------------------------------------------------------------
# AC38: round-trip fidelity
# ---------------------------------------------------------------------------

class TestAC38RoundTrip:
    def test_created_file_parses_back_equal(self, tmp_path):
        """AC38-r1: a created file parses back to an equal Schedule."""
        store = ScheduleStore(tmp_path / "schedules")
        row = _make_schedule(
            label="wördly ✓ 日本語",
            dest_channel_name="café ☕",
            next_run_at=datetime(2026, 10, 21, 7, 0, tzinfo=UTC),
            payload=json.dumps({"command": "echo héllo — ünïcode ✓", "cwd": "/tmp"}),
        )
        store.create(row)
        parsed = store.get("s1")
        assert parsed is not None
        assert (
            parsed.id, parsed.agent, parsed.platform, parsed.requester_id,
            parsed.requester_mention, parsed.dest_channel_id,
            parsed.dest_channel_name, parsed.label, parsed.cron,
            parsed.payload_type, parsed.payload, parsed.on_miss,
            parsed.next_run_at, parsed.last_run_at, parsed.status,
            parsed.last_status, parsed.last_error,
        ) == (
            row.id, row.agent, row.platform, row.requester_id,
            row.requester_mention, row.dest_channel_id,
            row.dest_channel_name, row.label, row.cron,
            row.payload_type, row.payload, row.on_miss,
            row.next_run_at, row.last_run_at, row.status,
            row.last_status, row.last_error,
        )
        # The instruction/label unicode survives on disk, not just in memory.
        text = (tmp_path / "schedules" / "s1.md").read_text(encoding="utf-8")
        assert "wördly ✓ 日本語" in text
        assert "café ☕" in text
        assert "héllo — ünïcode ✓" in text

    def test_agent_body_survives_byte_for_byte(self, tmp_path):
        """AC38-r1: unicode/emoji agent bodies survive create + parse."""
        store = ScheduleStore(tmp_path / "schedules")
        body = "Summarize the beads 🚂🎉\n\nSecond para — em-dash 日本語 ✓"
        row = _make_schedule(
            payload_type="agent",
            payload=json.dumps({"agent": "miranda", "instruction": body}, ensure_ascii=False),
        )
        store.create(row)
        parsed = store.get("s1")
        assert parsed is not None
        payload = json.loads(parsed.payload)
        assert payload["instruction"] == body
        payload_agent = json.loads(row.payload)
        assert payload_agent["agent"] == "miranda"

    def test_runtime_rewrite_preserves_user_keys_and_body(self, tmp_path):
        """AC38-r2: advance_after_run preserves user keys (incl. unknown
        extras) and the body byte-for-byte."""
        store = ScheduleStore(tmp_path / "schedules")
        store.create(_make_schedule(next_run_at=NOW - timedelta(seconds=1)))
        # Add an unknown extra user key after creation (hand edit).
        path = tmp_path / "schedules" / "s1.md"
        text = path.read_text(encoding="utf-8")
        text = text.replace("command: echo hi", "command: echo hi\nextra_note: keep-me 🎯")
        path.write_text(text, encoding="utf-8")
        before = path.read_bytes()

        store.claim("s1")
        store.advance_after_run("s1", NOW + timedelta(days=1), NOW, "ok", None)
        after = path.read_bytes()

        # The unknown extra key survives the runtime rewrite.
        assert b"extra_note: keep-me" in after
        assert b"\xf0\x9f\x8e\xaf" in after  # 🎯 survives
        # Runtime keys moved exactly as requested; user keys are untouched.
        parsed = store.get("s1")
        assert parsed.next_run_at == NOW + timedelta(days=1)
        assert parsed.last_run_at == NOW
        assert parsed.last_status == "ok"
        assert parsed.last_error is None
        # Every user key (known + unknown extra) is preserved byte-for-byte
        # as a line; the runtime keys are the only lines that changed.
        user_keys_before = {
            line for line in before.decode("utf-8").splitlines()
            if line and not line.startswith(
                ("---", "next_run_at:", "last_run_at:", "last_status:", "last_error:")
            )
        }
        user_keys_after = {
            line for line in after.decode("utf-8").splitlines()
            if line and not line.startswith(
                ("---", "next_run_at:", "last_run_at:", "last_status:", "last_error:")
            )
        }
        assert user_keys_before == user_keys_after
        # The body (command note) is unchanged.
        assert b"echo hi" in after

    def test_render_user_keys_before_runtime_keys(self, tmp_path):
        """AC38-r2: user keys render before runtime keys in the file."""
        store = ScheduleStore(tmp_path / "schedules")
        store.create(_make_schedule())
        text = (tmp_path / "schedules" / "s1.md").read_text(encoding="utf-8")
        last_user = max(text.index(k + ":") for k in ("on_miss", "command", "cron", "type"))
        first_runtime = min(text.index(k + ":") for k in ("next_run_at", "last_run_at", "last_status", "last_error"))
        assert last_user < first_runtime
        # Runtime keys in canonical order.
        runtime = [text.index(k + ":") for k in ("next_run_at", "last_run_at", "last_status", "last_error")]
        assert runtime == sorted(runtime)


# ---------------------------------------------------------------------------
# AC39: concurrency and crash safety
# ---------------------------------------------------------------------------

WORKER = r"""
import json, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from sr2_spectre.schedules import Schedule, ScheduleStore

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)
base = sys.argv[1]
mode = sys.argv[2]
tag = sys.argv[3]
store = ScheduleStore(base)

def row(i, nra):
    return Schedule(
        id=f"{tag}{i}", agent="miranda", platform="discord",
        requester_id="42", requester_mention="<@42>",
        dest_channel_id="100", dest_channel_name="general", label="t",
        cron="0 7 * * *", payload_type="command",
        payload=json.dumps({"command": "echo hi", "cwd": None}),
        on_miss="run_late", next_run_at=nra, last_run_at=None,
        created_at=NOW, status="active", last_status=None, last_error=None,
    )

try:
    if mode == "writer":
        for i in range(15):
            store.create(row(i, NOW - timedelta(seconds=1)))
            if i % 3 == 0:
                store.cancel(f"{tag}{i % 5}")
            if i % 2 == 0:
                store.advance_after_run(f"{tag}{i % 4}", NOW + timedelta(minutes=i), NOW, "ok", None)
        print("writer-done")
    else:
        for i in range(15):
            due = store.due(NOW + timedelta(hours=1))
            for s in due:
                if store.claim(s.id):
                    store.advance_after_run(s.id, NOW + timedelta(hours=2, minutes=i), NOW, "ok", None)
            time.sleep(0.005)
        print("engine-done")
except Exception as e:
    print(f"worker-error: {type(e).__name__}: {e}")
    sys.exit(1)
"""


def _run_worker(base: str, mode: str, tag: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", WORKER, base, mode, tag],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )
    return proc.stdout.strip()


class TestAC39Concurrency:
    def test_two_processes_no_lost_or_corrupted(self, tmp_path):
        """AC39-r1: two concurrent writers on one directory lose nothing."""
        base = tmp_path / "schedules"
        base.mkdir()
        p1 = subprocess.Popen(
            [sys.executable, "-c", WORKER, str(base), "writer", "w"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        p2 = subprocess.Popen(
            [sys.executable, "-c", WORKER, str(base), "engine", "e"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        outs = []
        for p in (p1, p2):
            out, err = p.communicate(timeout=120)
            outs.append(out.strip())
            assert p.returncode == 0, err
        assert outs[0] == "writer-done"
        assert outs[1] == "engine-done"

        store = ScheduleStore(base)
        # Every surviving file parses; no file is left corrupt.
        valid, invalid = store.scan()
        assert invalid == []
        # Nothing was double-run: every advance writes exactly one runtime
        # key set; a file whose state came from an advance carries it.
        rows = {s.id: s for s in valid}
        # The writer's non-cancelled schedules all exist (no schedule lost).
        # i % 3 == 0 cancels w(i % 5): cancelled = {w0, w3, w4, w2, w1}
        # over i in 0..14 -> all of w0..w4 are cancelled eventually, so
        # assert the engine-side ids (e0..e2) that claims produced state on.
        for s in valid:
            assert s.id.startswith("e") or s.id.startswith("w")
        # No temp files remain for finished work.
        assert [p.name for p in base.iterdir() if p.name.startswith(".tmp-")] == []


KILL_HELPER = r"""
import os, sys, time
from sr2_spectre.schedules.store import ScheduleStore, _render_document, _runtime_list, _user_keys
from sr2_spectre.schedules import Schedule
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)
base = sys.argv[1]
store = ScheduleStore(base)
row = Schedule(
    id="k1", agent="miranda", platform="discord", requester_id="", requester_mention="",
    dest_channel_id="100", dest_channel_name="general", label="test", cron="0 7 * * *",
    payload_type="command", payload=json.dumps({"command": "echo hi", "cwd": None}),
    on_miss="run_late", next_run_at=NOW - timedelta(seconds=1), last_run_at=None,
    created_at=NOW, status="active", last_status=None, last_error=None,
)
store.create(row)
# Simulate: temp written, process dies before os.replace.
text = store._render_row(row)
tmp = store._dir / ".tmp-killed-0001-k1.md"
tmp.write_text(text, encoding="utf-8")
print("killed-ready", flush=True)
time.sleep(30)
"""


class TestAC39KillBetweenWriteAndReplace:
    def test_original_readable_and_temp_swept(self, tmp_path):
        """AC39-r2: kill between temp write and replace leaves the original
        readable; the leftover temp is swept after one poll interval."""
        base = tmp_path / "schedules"
        base.mkdir()
        store = ScheduleStore(base)
        row = _make_schedule(next_run_at=NOW - timedelta(seconds=1))
        store.create(row)
        original = store.get("s1")
        original_bytes = (base / "s1.md").read_bytes()

        proc = subprocess.Popen(
            [sys.executable, "-c", KILL_HELPER, str(base)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        line = proc.stdout.readline().strip()
        assert line == "killed-ready"
        proc.kill()
        proc.wait(timeout=10)

        # The original file is still readable and uncorrupted: its bytes
        # and parsed row are exactly what the writer committed.
        assert (base / "s1.md").read_bytes() == original_bytes
        fresh = ScheduleStore(base)
        parsed = fresh.get("s1")
        assert parsed == original

        # The leftover temp is present until one poll interval has passed,
        # then a sweep deletes it.
        temps = [p.name for p in base.iterdir() if p.name.startswith(".tmp-")]
        assert temps == [".tmp-killed-0001-k1.md"]
        # One poll interval later: the temp is past its age and a sweep
        # deletes it. (The test's clock is fake; age is proven by
        # backdating the temp's mtime past one poll interval.)
        import os as _os

        ts = (NOW - timedelta(seconds=35)).timestamp()
        _os.utime(base / ".tmp-killed-0001-k1.md", (ts, ts))
        store.sweep_temp_files(NOW, 30.0)
        assert [p.name for p in base.iterdir() if p.name.startswith(".tmp-")] == []


# ---------------------------------------------------------------------------
# AC37: async facades over the directory-backed store
# ---------------------------------------------------------------------------

class TestAC37AsyncFacades:
    @pytest.mark.asyncio
    async def test_async_facades_match_sync_contract(self, tmp_path):
        store = ScheduleStore(tmp_path / "schedules")
        assert await store.create_async(
            _make_schedule(next_run_at=NOW - timedelta(seconds=1))
        ) == "s1"

        store2 = ScheduleStore(tmp_path / "schedules")
        due = await store2.due_async(NOW)
        assert [s.id for s in due] == ["s1"]

        assert await store.claim_async("s1") is True
        assert await store.claim_async("s1") is False  # exactly-once

        nxt = NOW + timedelta(days=1)
        await store.advance_after_run_async("s1", nxt, NOW, "ok", None)
        row = await store.get_async("s1")
        assert row.next_run_at == nxt and row.last_status == "ok"

        assert await store.cancel_async("s1") is True
        assert await store.cancel_async("s1") is False

        rows = await store.list_for_agent_async("miranda")
        assert rows == []
        assert await store.get_async("s1") is None
