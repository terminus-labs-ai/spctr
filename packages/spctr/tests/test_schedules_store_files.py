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

from spctr.schedules import Schedule, ScheduleStore

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
            parsed.next_run_at, parsed.last_run_at, parsed.created_at,
            parsed.status, parsed.last_status, parsed.last_error,
        ) == (
            "s1", row.agent, row.platform, row.requester_id,
            row.requester_mention, row.dest_channel_id,
            row.dest_channel_name, row.label, row.cron,
            row.payload_type, row.payload, row.on_miss,
            row.next_run_at, row.last_run_at, row.created_at,
            row.status, row.last_status, row.last_error,
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

    def test_created_at_survives_claim_and_advance(self, tmp_path):
        """AC38: created_at is persisted and stable across runtime rewrites."""
        store = ScheduleStore(tmp_path / "schedules")
        row = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            created_at=NOW - timedelta(days=7),
        )
        sid = store.create(row)
        store.claim(sid)
        store.advance_after_run(sid, NOW + timedelta(days=1), NOW, "ok", None)
        parsed = store.get(sid)
        assert parsed.created_at == NOW - timedelta(days=7)
        assert parsed.created_at != parsed.next_run_at
        # On disk too: the created_at key survived both rewrites.
        text = (tmp_path / "schedules" / f"{sid}.md").read_text(encoding="utf-8")
        assert "created_at" in text

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
from spctr.schedules import Schedule, ScheduleStore

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
        ids = []
        for i in range(15):
            ids.append(store.create(row(i, NOW - timedelta(seconds=1))))
            if i % 3 == 0:
                store.cancel(ids[i // 3])
            if i % 2 == 0:
                store.advance_after_run(ids[(i // 2) % len(ids)], NOW + timedelta(minutes=i), NOW, "ok", None)
        print("writer-done " + ",".join(ids))
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
        assert outs[0].startswith("writer-done")
        assert outs[1] == "engine-done"

        store = ScheduleStore(base)
        # Every surviving file parses; no file is left corrupt.
        valid, invalid = store.scan()
        assert invalid == []
        # No schedule lost and none extra: exactly the schedules the
        # writer created and did NOT cancel survive — the engine only
        # claims/advances.  ids[0..4] is the cancelled set (i % 3 == 0
        # cancels ids[i // 3]); a cancelled stem may be reused by a later
        # create, so compare against the writer's own id list.
        created = outs[0].split(" ", 1)[1].split(",")
        assert len(created) == 15
        cancelled = set(created[:5])
        assert {s.id for s in valid} == set(created[5:]) - cancelled
        assert len(valid) == 10
        # Every mutation landed as a complete file: each surviving file
        # re-parses identically through a fresh store.
        fresh = ScheduleStore(base)
        for s in valid:
            assert fresh.get(s.id) == s
        # No temp files remain for finished work.
        assert [p.name for p in base.iterdir() if p.name.startswith(".tmp-")] == []


KILL_HELPER = r"""
import os, sys, time
from spctr.schedules.store import ScheduleStore, _render_document, _runtime_list, _user_keys
from spctr.schedules import Schedule
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
        # The leftover temp is not mistaken for a schedule: the scan shows
        # the two committed files as valid and reports nothing invalid.
        valid_now, invalid_now = fresh.scan()
        assert {s.id for s in valid_now} == {"s1", "s2"}
        assert invalid_now == []

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
# AC37: sequential s<N> assignment; async facades over the directory store
# ---------------------------------------------------------------------------

class TestAC37IdAssignment:
    def test_create_ignores_placeholder_id_and_never_overwrites(self, tmp_path):
        """Two creates with the same placeholder id produce s1 and s2."""
        store = ScheduleStore(tmp_path / "schedules")
        assert store.create(_make_schedule(id="s0")) == "s1"
        assert store.create(_make_schedule(id="s0")) == "s2"
        assert (tmp_path / "schedules" / "s1.md").is_file()
        assert (tmp_path / "schedules" / "s2.md").is_file()
        assert store.get("s1") is not None
        assert store.get("s2") is not None

    def test_create_continues_after_highest_existing_stem(self, tmp_path):
        """A hand-authored s7.md moves the next create to s8."""
        store = ScheduleStore(tmp_path / "schedules")
        _write_schedule_file(
            tmp_path / "schedules", "s7.md",
            USER_BASE + "next_run_at: '2026-10-21T07:00:00+00:00'\n",
            "notes",
        )
        assert store.create(_make_schedule(id="s0")) == "s8"
        # Non-s<N> files do not affect numbering.
        _write_schedule_file(tmp_path / "schedules", "standup.md", USER_BASE, "notes")
        assert store.create(_make_schedule(id="s0")) == "s9"


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


# ---------------------------------------------------------------------------
# AC37-r2: custom schedules.store_path template honored end to end
# ---------------------------------------------------------------------------

class TestAC37CustomStorePath:
    def test_custom_store_path_template(self, tmp_path):
        """AC37-r2: resolved_store_path honors a custom {agent} template."""
        from spctr.config import SchedulesConfig, resolve_sr2_home

        cfg = SchedulesConfig(store_path=str(tmp_path / "custom" / "{agent}"))
        resolved = cfg.resolved_store_path("miranda")
        assert resolved == str(tmp_path / "custom" / "miranda")

        store = ScheduleStore(resolved)
        assert store.create(_make_schedule()) == "s1"
        assert (tmp_path / "custom" / "miranda" / "s1.md").is_file()
        assert store.get("s1") is not None
        assert [s.id for s in store.list_for_agent("miranda")] == ["s1"]

        # Nothing was written under the default state/{agent}/schedules/.
        state_root = Path(resolve_sr2_home()) / "state"
        assert not (state_root / "miranda" / "schedules").exists()
        leftovers = list(state_root.rglob("*.md")) if state_root.exists() else []
        assert leftovers == []

    def test_tilde_in_store_path_expanded(self):
        """AC37-r2: a '~' store_path resolves under the user's home."""
        from spctr.config import SchedulesConfig

        cfg = SchedulesConfig(store_path="~/sched-x/{agent}")
        resolved = cfg.resolved_store_path("miranda")
        assert resolved == str(Path.home() / "sched-x" / "miranda")
        assert not resolved.startswith("~")


# ---------------------------------------------------------------------------
# Scan primitive: first-seen files, paused files, invalid files (FR39/40/44)
# ---------------------------------------------------------------------------

class TestScanPrimitive:
    def test_first_seen_no_next_run_at_is_valid_and_not_due(self, tmp_path):
        """FR39: a file with no next_run_at is a live schedule the engine
        schedules on its next poll — not an invalid file."""
        store = ScheduleStore(tmp_path / "schedules")
        _write_schedule_file(tmp_path / "schedules", "new.md", USER_BASE, "notes")
        valid, invalid = store.scan()
        assert invalid == []
        assert [s.id for s in valid] == ["new"]
        assert valid[0].next_run_at is None
        assert valid[0].status == "active"
        # Not due and not claimable until the engine stores an occurrence.
        assert store.due(NOW) == []
        assert store.claim("new") is False
        store.advance_after_run("new", NOW + timedelta(days=1), NOW, "ok", None)
        assert store.get("new").next_run_at == NOW + timedelta(days=1)
        assert store.due(NOW) == []
        assert [s.id for s in store.due(NOW + timedelta(days=2))] == ["new"]

    def test_closing_fence_at_eof_without_newline(self, tmp_path):
        """A closing --- fence at EOF (no trailing newline) parses."""
        store = ScheduleStore(tmp_path / "schedules")
        (tmp_path / "schedules" / "eof.md").write_text(
            "---\n" + USER_BASE + "next_run_at: '2026-10-21T07:00:00+00:00'\n---",
            encoding="utf-8",
        )
        row = store.get("eof")
        assert row is not None
        assert row.id == "eof"
        assert json.loads(row.payload)["command"] == "echo hi"

    def test_unquoted_yaml_timestamp_accepted(self, tmp_path):
        """A hand-authored unquoted ISO timestamp parses as a datetime."""
        store = ScheduleStore(tmp_path / "schedules")
        _write_schedule_file(
            tmp_path / "schedules", "ts.md",
            USER_BASE + "next_run_at: 2026-10-21T07:00:00+00:00\n",
            "notes",
        )
        row = store.get("ts")
        assert row is not None
        assert row.next_run_at == datetime(2026, 10, 21, 7, 0, tzinfo=UTC)

    def test_enabled_false_pauses(self, tmp_path):
        """FR44: enabled: false is valid, never due/claimed, cancellable."""
        store = ScheduleStore(tmp_path / "schedules")
        _write_schedule_file(
            tmp_path / "schedules", "parked.md",
            USER_BASE + "enabled: false\nnext_run_at: '2026-10-19T07:00:00+00:00'\n",
            "notes",
        )
        valid, invalid = store.scan()
        assert invalid == []
        assert valid[0].status == "paused"
        # Past due but paused: never due, never claimed.
        assert store.due(NOW) == []
        assert store.claim("parked") is False
        # Hidden from the default list; listed as paused when asked.
        assert store.list_for_agent("miranda") == []
        assert [
            s.id for s in store.list_for_agent("miranda", statuses=("paused",))
        ] == ["parked"]
        # A paused schedule is still a schedule: cancel removes it.
        assert store.cancel("parked") is True
        assert store.get("parked") is None

    def test_invalid_files_reported_with_reasons(self, tmp_path):
        """FR40: every invalid shape lands in scan()'s invalid list."""
        store = ScheduleStore(tmp_path / "schedules")
        base = tmp_path / "schedules"
        fence = "---\n{keys}\n---\n\nnotes\n"
        good_keys = (
            "cron: 0 7 * * *\ntype: command\ncommand: x\nplatform: discord\n"
            "next_run_at: '2026-10-21T07:00:00+00:00'\n"
        )
        bad = {
            "BAD.md": fence.format(keys=good_keys),
            "nofences.md": "just text, no frontmatter\n",
            "badyaml.md": "---\ncron: [unclosed\n---\n",
            "nocron.md": fence.format(
                keys="type: command\ncommand: x\nplatform: discord\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badcron.md": fence.format(
                keys="cron: 0 7 *** *\ntype: command\ncommand: x\nplatform: discord\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badtype.md": fence.format(
                keys="cron: 0 7 * * *\ntype: rocket\nplatform: discord\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badonmiss.md": fence.format(
                keys="cron: 0 7 * * *\ntype: command\ncommand: x\nplatform: discord\n"
                "on_miss: maybe\nnext_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badenabled.md": fence.format(
                keys="cron: 0 7 * * *\ntype: command\ncommand: x\nplatform: discord\n"
                "enabled: banana\nnext_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badplatform.md": fence.format(
                keys="cron: 0 7 * * *\ntype: command\ncommand: x\nplatform: telegraph\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "noplatform.md": fence.format(
                keys="cron: 0 7 * * *\ntype: command\ncommand: x\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
        }
        for name, text in bad.items():
            (base / name).write_text(text, encoding="utf-8")
        (base / "big.md").write_text(
            fence.format(keys=good_keys) + "x" * (64 * 1024), encoding="utf-8"
        )
        (base / "bigbody.md").write_text(
            "---\ncron: 0 7 * * *\ntype: agent\nplatform: discord\n---\n\n"
            + "x" * 4001,
            encoding="utf-8",
        )
        valid, invalid = store.scan()
        assert valid == []
        reasons = {e.filename: e.reason for e in invalid}
        assert set(reasons) == set(bad) | {"big.md", "bigbody.md"}
        assert reasons["BAD.md"] == "invalid id stem"
        assert reasons["noplatform.md"] == "platform required"
        assert "cron" in reasons["nocron.md"]
        assert "cron" in reasons["badcron.md"]
        assert "on_miss" in reasons["badonmiss.md"]
        assert "enabled" in reasons["badenabled.md"]
        assert "platform" in reasons["badplatform.md"]
        assert "64 KiB" in reasons["big.md"]
        assert "4000" in reasons["bigbody.md"]

    def test_invalid_file_does_not_block_valid_sibling(self, tmp_path):
        """FR40: an invalid file is skipped entirely; siblings unaffected."""
        store = ScheduleStore(tmp_path / "schedules")
        (tmp_path / "schedules" / "broken.md").write_text(
            "no fences here", encoding="utf-8"
        )
        store.create(_make_schedule(next_run_at=NOW - timedelta(seconds=1)))
        valid, invalid = store.scan()
        assert [s.id for s in valid] == ["s1"]
        assert [e.filename for e in invalid] == ["broken.md"]
        assert [s.id for s in store.due(NOW)] == ["s1"]
        assert store.claim("s1") is True
        # The invalid file is never due and never claimed; cancel still
        # removes it (cancel is file deletion).
        assert store.claim("broken") is False
        assert [s.id for s in store.due(NOW + timedelta(days=30))] == []
        assert store.cancel("broken") is True
