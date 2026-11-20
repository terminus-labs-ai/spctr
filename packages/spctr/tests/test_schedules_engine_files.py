"""Engine hot path over the markdown store (spc-995, FR39–FR42, FR44).

Real files in ``tmp_path``, the real ``ScheduleStore``, an injected
clock and real shell commands (each writes a marker file, so a fire is
observable on disk).  Agent payloads use a fake dispatch runner.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from spctr.schedules import AgentResult, ScheduleEngine, ScheduleStore

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 10, 20, 12, 0, 0, tzinfo=UTC)
TOMORROW_7 = datetime(2026, 10, 21, 7, 0, 0, tzinfo=UTC)
POLL = 30.0


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class Harness:
    def __init__(self, tmp_path: Path, platform: str = "discord") -> None:
        self.dir = tmp_path / "schedules"
        self.store = ScheduleStore(self.dir, agent="miranda")
        self.clock = Clock(NOW)
        self.sent: list[tuple[str, str]] = []
        self.instructions: list[str] = []
        self.tmp = tmp_path
        self.on_agent_run = None
        self._platform = platform
        self.engine = self._new_engine()

    def _new_engine(self) -> ScheduleEngine:
        return ScheduleEngine(
            self.store, send=self._send, poll_seconds=POLL, tz=UTC,
            agent_name="test", platform=self._platform, clock=self.clock,
            agent_runner=self._runner,
        )

    def restart(self) -> None:
        """Swap in a fresh store and engine over the same directory, the way
        a process restart leaves them: nothing held in-process survives."""
        self.store = ScheduleStore(self.dir, agent="miranda")
        self.engine = self._new_engine()

    async def _send(self, channel_id: str, content: str) -> None:
        self.sent.append((channel_id, content))

    async def _runner(self, target, instruction, *, timeout, agent, clock):
        self.instructions.append(instruction)
        if self.on_agent_run is not None:
            self.on_agent_run()
        return AgentResult(
            status="done", message=f"posted: {instruction}",
            exit_code=0, duration_s=0.0, log_path="",
        )

    def marker(self, name: str) -> Path:
        return self.tmp / f"{name}.count"

    def write(self, stem: str, lines: list[str], body: str = "") -> Path:
        path = self.dir / f"{stem}.md"
        text = "---\n" + "\n".join(lines) + "\n---\n"
        if body:
            text += "\n" + body
        path.write_text(text, encoding="utf-8")
        return path

    def command_file(self, stem: str, *extra: str, cron: str = "0 7 * * *",
                     marker: str | None = None, platform: str = "discord") -> Path:
        m = self.marker(marker or stem)
        return self.write(stem, [
            f"cron: \"{cron}\"",
            "type: command",
            f"command: \"printf x >> {m}\"",
            f"platform: {platform}",
            "channel: \"100\"",
            *extra,
        ])

    def fired(self, name: str) -> int:
        m = self.marker(name)
        return len(m.read_text()) if m.exists() else 0

    async def poll_at(self, when: datetime) -> None:
        self.clock.now = when
        await self.engine._poll_once()


def _replace_line(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


# ---------------------------------------------------------------------------
# AC41: discovery, first-sight advance, fresh reads, deletion
# ---------------------------------------------------------------------------

class TestAC41Discovery:
    async def test_r1_dropped_file_scheduled_never_retro_fires(self, tmp_path):
        h = Harness(tmp_path)
        h.command_file("digest")

        await h.poll_at(NOW)
        assert h.fired("digest") == 0
        row = h.store.get("digest")
        assert row.next_run_at == TOMORROW_7
        assert row.last_run_at is None and row.last_status is None

        await h.poll_at(NOW + timedelta(seconds=30))
        await h.poll_at(TOMORROW_7 - timedelta(seconds=1))
        assert h.fired("digest") == 0

        await h.poll_at(TOMORROW_7)
        assert h.fired("digest") == 1
        row = h.store.get("digest")
        assert row.next_run_at == TOMORROW_7 + timedelta(days=1)
        assert row.last_run_at == TOMORROW_7 and row.last_status == "ok"

        await h.poll_at(TOMORROW_7 + timedelta(seconds=30))
        assert h.fired("digest") == 1

    async def test_r1_other_platform_file_left_untouched(self, tmp_path):
        h = Harness(tmp_path, platform="discord")
        path = h.command_file("zfile", platform="zulip")
        before = path.read_bytes()
        await h.poll_at(NOW)
        assert path.read_bytes() == before

    async def test_r2_command_edit_runs_fresh_content(self, tmp_path):
        h = Harness(tmp_path)
        path = h.command_file("job", marker="old")
        await h.poll_at(NOW)
        await h.poll_at(TOMORROW_7)
        assert h.fired("old") == 1

        _replace_line(path, str(h.marker("old")), str(h.marker("new")))
        await h.poll_at(TOMORROW_7 + timedelta(days=1))
        assert h.fired("old") == 1
        assert h.fired("new") == 1

    async def test_r2_agent_body_edit_runs_fresh_prompt(self, tmp_path):
        h = Harness(tmp_path)
        lines = ["cron: \"0 7 * * *\"", "type: agent", "platform: discord",
                 "channel: \"100\""]
        path = h.write("brief", lines, body="first prompt\n")
        await h.poll_at(NOW)
        await h.poll_at(TOMORROW_7)
        _replace_line(path, "first prompt", "second prompt")
        await h.poll_at(TOMORROW_7 + timedelta(days=1))
        assert h.instructions == ["first prompt\n", "second prompt\n"]
        assert h.sent == [("100", "posted: first prompt\n"),
                          ("100", "posted: second prompt\n")]

    async def test_cron_edit_reschedules_within_one_poll(self, tmp_path):
        h = Harness(tmp_path)
        path = h.command_file("job")
        await h.poll_at(NOW)
        assert h.store.get("job").next_run_at == TOMORROW_7

        _replace_line(path, "0 7 * * *", "0 13 * * *")
        await h.poll_at(NOW + timedelta(seconds=30))
        assert h.store.get("job").next_run_at == datetime(2026, 10, 20, 13, 0, tzinfo=UTC)
        assert h.fired("job") == 0
        await h.poll_at(datetime(2026, 10, 20, 13, 0, tzinfo=UTC))
        assert h.fired("job") == 1

    async def test_r3_deleted_file_never_fires(self, tmp_path):
        h = Harness(tmp_path)
        path = h.command_file("gone")
        await h.poll_at(NOW)
        path.unlink()
        await h.poll_at(TOMORROW_7)
        await h.poll_at(TOMORROW_7 + timedelta(days=1))
        assert h.fired("gone") == 0
        assert not path.exists()


# ---------------------------------------------------------------------------
# AC42: invalid-file skip, ERROR once per content hash, sibling fires
# ---------------------------------------------------------------------------

PAST = "next_run_at: \"2026-10-20T11:00:00+00:00\""


class TestAC42Invalid:
    async def test_r1_each_invalid_case(self, tmp_path, caplog):
        h = Harness(tmp_path)
        base = ["type: command", "platform: discord", "channel: \"100\"", PAST]

        def cmd(name: str) -> str:
            return f"command: \"printf x >> {h.marker(name)}\""

        cases = {
            "unparseable": (["cron: [unclosed", *base, cmd("unparseable")], "",
                            "frontmatter does not parse"),
            "missing-cron": ([*base, cmd("missing-cron")], "",
                             "missing required key 'cron'"),
            "unknown-type": (["cron: \"* * * * *\"", "type: shell", "platform: discord",
                              PAST, cmd("unknown-type")], "", "unknown type 'shell'"),
            "bad-enabled": (["cron: \"* * * * *\"", *base, "enabled: maybe",
                             cmd("bad-enabled")], "", "unknown enabled 'maybe'"),
            "bad-on-miss": (["cron: \"* * * * *\"", *base, "on_miss: later",
                             cmd("bad-on-miss")], "", "unknown on_miss 'later'"),
            "bad-cron": (["cron: \"61 * * * *\"", *base, cmd("bad-cron")], "",
                         "invalid cron expression '61 * * * *'"),
            "big-prompt": (["cron: \"* * * * *\"", "type: agent", "platform: discord",
                            PAST], "p" * 4001,
                           "agent instruction exceeds 4000 characters"),
            "huge": (["cron: \"* * * * *\"", *base, cmd("huge")], "n" * (65 * 1024),
                     "file exceeds 64 KiB"),
        }
        before = {}
        for stem, (lines, body, _reason) in cases.items():
            before[stem] = h.write(stem, lines, body=body).read_bytes()
        h.command_file("sibling", PAST, cron="* * * * *")

        with caplog.at_level(logging.INFO):
            for i in range(3):
                await h.poll_at(NOW + timedelta(seconds=i))

        # The valid sibling fired (first-seen past next_run_at, run_late).
        assert h.fired("sibling") == 1
        # No invalid file ran or was claimed or rewritten.
        for stem in cases:
            assert h.fired(stem) == 0, stem
            assert (h.dir / f"{stem}.md").read_bytes() == before[stem], stem
        assert h.instructions == []

        # Store state reports each one as INVALID with its reason.
        _valid, invalid = h.store.scan()
        reasons = {e.filename: e.reason for e in invalid}
        assert set(reasons) == {f"{s}.md" for s in cases}
        for stem, (_l, _b, reason) in cases.items():
            assert reason in reasons[f"{stem}.md"], (stem, reasons[f"{stem}.md"])

        def errors_for(stem: str) -> list[logging.LogRecord]:
            return [
                r for r in caplog.records
                if r.levelno == logging.ERROR
                and r.name == "spctr.schedules.engine"
                and f"file={stem}.md " in r.getMessage()
            ]

        for stem, (_l, _b, reason) in cases.items():
            errs = errors_for(stem)
            assert len(errs) == 1, (stem, [r.getMessage() for r in errs])
            assert reason in errs[0].getMessage()

        # New content (still invalid) logs once more; same content does not.
        _replace_line(h.dir / "bad-cron.md", "61 * * * *", "62 * * * *")
        with caplog.at_level(logging.INFO):
            await h.poll_at(NOW + timedelta(seconds=10))
            await h.poll_at(NOW + timedelta(seconds=11))
        assert len(errors_for("bad-cron")) == 2
        assert len(errors_for("unparseable")) == 1

    async def test_fixed_file_becomes_valid_next_poll(self, tmp_path):
        h = Harness(tmp_path)
        path = h.command_file("fixme", cron="61 7 * * *")
        await h.poll_at(NOW)
        assert h.store.get("fixme") is None
        _replace_line(path, "61 7 * * *", "0 7 * * *")
        await h.poll_at(NOW + timedelta(seconds=30))
        assert h.store.get("fixme").next_run_at == TOMORROW_7


# ---------------------------------------------------------------------------
# AC43: file deleted mid-payload
# ---------------------------------------------------------------------------

class TestAC43DeletedMidRun:
    async def test_r1_no_write_back_post_lands_one_log(self, tmp_path, caplog):
        h = Harness(tmp_path)
        path = h.write("brief", ["cron: \"0 7 * * *\"", "type: agent",
                                 "platform: discord", "channel: \"100\""],
                       body="summarize\n")
        await h.poll_at(NOW)
        h.on_agent_run = path.unlink

        with caplog.at_level(logging.INFO):
            await h.poll_at(TOMORROW_7)

        assert h.instructions == ["summarize\n"]
        assert h.sent == [("100", "posted: summarize\n")]
        assert not path.exists()
        assert sorted(p.name for p in h.dir.iterdir()) == [".lock"]
        gone = [r for r in caplog.records if "file gone" in r.getMessage()]
        assert len(gone) == 1
        assert "id=brief" in gone[0].getMessage()


# ---------------------------------------------------------------------------
# AC45: enabled: false pauses; resume advances silently
# ---------------------------------------------------------------------------

class TestAC45Paused:
    async def test_r1_paused_past_due_never_due_or_caught_up(self, tmp_path):
        h = Harness(tmp_path)
        path = h.command_file("p", PAST, "enabled: false")
        before = path.read_bytes()

        await h.engine.start()
        await asyncio.sleep(0)
        await h.engine.stop()
        await h.poll_at(NOW + timedelta(seconds=30))
        await h.poll_at(TOMORROW_7)

        assert h.fired("p") == 0
        assert path.read_bytes() == before  # never advanced
        rows = h.store.list_for_agent("miranda", ("active", "paused"))
        assert [(r.id, r.status) for r in rows] == [("p", "paused")]
        assert h.store.due(TOMORROW_7) == []

    async def test_r2_resume_advances_silently(self, tmp_path):
        h = Harness(tmp_path)
        path = h.command_file("p", PAST, "enabled: false")
        await h.poll_at(NOW)

        _replace_line(path, "enabled: false", "enabled: true")
        await h.poll_at(NOW + timedelta(seconds=30))
        assert h.fired("p") == 0 and h.sent == []
        row = h.store.get("p")
        assert row.status == "active"
        assert row.next_run_at == TOMORROW_7
        assert row.last_run_at is None and row.last_status is None

        await h.poll_at(NOW + timedelta(seconds=60))
        assert h.fired("p") == 0
        await h.poll_at(TOMORROW_7)
        assert h.fired("p") == 1

    async def test_r3_cron_edit_while_paused_applies_on_resume(self, tmp_path):
        h = Harness(tmp_path)
        path = h.command_file("p", PAST, "enabled: false")
        await h.poll_at(NOW)
        _replace_line(path, "0 7 * * *", "30 9 * * *")
        await h.poll_at(NOW + timedelta(seconds=30))
        assert h.store.get("p").status == "paused"
        assert "2026-10-20T11:00:00+00:00" in path.read_text()

        _replace_line(path, "enabled: false", "enabled: true")
        await h.poll_at(NOW + timedelta(seconds=60))
        assert h.store.get("p").next_run_at == datetime(2026, 10, 21, 9, 30, tzinfo=UTC)
        assert h.fired("p") == 0

    async def test_r4_resume_while_engine_stopped_follows_fr39(
        self, tmp_path, caplog
    ):
        """FR44/FR39 boundary (spc-1093): a flip the engine never observed
        is ordinary FR39 catch-up, not a silent resume-advance.

        Pausing leaves no trace in the file beyond ``enabled: false``, so a
        file that reads enabled with a past ``next_run_at`` is
        indistinguishable from FR39's hand-edit escape hatch. The silent
        advance covers flips the running engine saw (``test_r2``); this row
        follows FR30 and fires once, marked late.
        """
        h = Harness(tmp_path)
        path = h.command_file("p", PAST, "enabled: false")
        await h.poll_at(NOW)  # the engine observes the pause, then stops
        _replace_line(path, "enabled: false", "enabled: true")
        h.restart()

        with caplog.at_level(logging.INFO):
            await h.engine.start()
            await asyncio.sleep(0)
            await h.engine.stop()
        assert h.fired("p") == 1
        assert any(r.getMessage() == "schedule fired late: agent=test id=p"
                   for r in caplog.records)
        row = h.store.get("p")
        assert row.status == "active"
        assert row.next_run_at == TOMORROW_7 and row.last_status == "ok"

        await h.poll_at(NOW + timedelta(seconds=30))
        assert h.fired("p") == 1  # one catch-up fire, never a second

    async def test_r5_on_miss_skip_resume_while_stopped_never_fires(self, tmp_path):
        """``on_miss: skip`` is how a schedule keeps a stop-plus-pause from
        queueing a run at all (FR44 as amended by spc-1093)."""
        h = Harness(tmp_path)
        path = h.command_file("p", PAST, "enabled: false", "on_miss: skip")
        await h.poll_at(NOW)
        _replace_line(path, "enabled: false", "enabled: true")
        h.restart()

        await h.engine.start()
        await asyncio.sleep(0)
        await h.engine.stop()
        assert h.fired("p") == 0 and h.sent == []
        row = h.store.get("p")
        assert row.status == "active"
        assert row.next_run_at == TOMORROW_7 and row.last_status == "skipped"


    async def test_r6_on_miss_skip_paused_past_due_never_advanced(self, tmp_path):
        """FR44 for ``on_miss: skip`` (spc-1094): a paused file that a poll
        sees first with a past ``next_run_at`` must stay byte-identical.

        The store's claim and reschedule refuse paused rows, but the
        first-sight ``on_miss: skip`` advance goes through
        ``advance_after_run``, which does not check status; only the
        engine's active gate keeps it off paused files.  Covered both at
        start() and on a later poll, like ``test_r1``.
        """
        h = Harness(tmp_path)
        path = h.command_file("p", PAST, "enabled: false", "on_miss: skip")
        before = path.read_bytes()

        await h.engine.start()
        await asyncio.sleep(0)
        await h.engine.stop()
        await h.poll_at(NOW + timedelta(seconds=30))
        await h.poll_at(TOMORROW_7)

        assert h.fired("p") == 0
        assert path.read_bytes() == before  # never advanced
        row = h.store.get("p")
        assert row.status == "paused"
        assert row.next_run_at == datetime(2026, 10, 20, 11, 0, tzinfo=UTC)
        assert row.last_run_at is None and row.last_status is None
        assert h.store.due(TOMORROW_7) == []


# ---------------------------------------------------------------------------
# FR39 second paragraph: first seen with a past next_run_at -> FR30 policy
# ---------------------------------------------------------------------------

class TestFR30FirstSeenPast:
    async def test_r1_startup_run_late_fires_once_late(self, tmp_path, caplog):
        h = Harness(tmp_path)
        h.command_file("late", "next_run_at: \"2026-10-19T07:00:00+00:00\"")
        with caplog.at_level(logging.INFO):
            await h.engine.start()
            await asyncio.sleep(0)
            await h.engine.stop()
            await h.poll_at(NOW + timedelta(seconds=30))
        assert h.fired("late") == 1
        assert any(r.getMessage() == "schedule fired late: agent=test id=late"
                   for r in caplog.records)
        row = h.store.get("late")
        assert row.next_run_at == TOMORROW_7 and row.last_status == "ok"

    async def test_r1_startup_skip_advances_without_firing(self, tmp_path):
        h = Harness(tmp_path)
        h.command_file("skipme", PAST, "on_miss: skip")
        await h.engine.start()
        await asyncio.sleep(0)
        await h.engine.stop()
        assert h.fired("skipme") == 0
        row = h.store.get("skipme")
        assert row.next_run_at == TOMORROW_7 and row.last_status == "skipped"

    async def test_r1_dropped_mid_run_follows_on_miss(self, tmp_path):
        h = Harness(tmp_path)
        await h.poll_at(NOW)  # engine running, directory empty
        h.command_file("skipme", PAST, "on_miss: skip")
        h.command_file("runme", PAST)
        await h.poll_at(NOW + timedelta(seconds=30))
        assert h.fired("skipme") == 0
        assert h.store.get("skipme").last_status == "skipped"
        assert h.store.get("skipme").next_run_at == TOMORROW_7
        assert h.fired("runme") == 1
        await h.poll_at(NOW + timedelta(seconds=60))
        assert h.fired("runme") == 1


# ---------------------------------------------------------------------------
# FR41 last sentence: leftover temp sweep at scan time
# ---------------------------------------------------------------------------

class TestTempSweep:
    async def test_old_temp_swept_fresh_kept(self, tmp_path):
        h = Harness(tmp_path)
        old = h.dir / ".tmp-1-aaaa-x.md"
        fresh = h.dir / ".tmp-2-bbbb-y.md"
        for p, age in ((old, 2 * POLL), (fresh, POLL / 2)):
            p.write_text("partial", encoding="utf-8")
            ts = NOW.timestamp() - age
            os.utime(p, (ts, ts))
        await h.poll_at(NOW)
        assert not old.exists()
        assert fresh.exists()
        assert (h.dir / ".lock").exists()


# ---------------------------------------------------------------------------
# Store hook: reschedule is compare-and-set and never touches paused files
# ---------------------------------------------------------------------------

class TestStoreReschedule:
    def test_reschedule_guards(self, tmp_path):
        h = Harness(tmp_path)
        h.command_file("a")
        paused = h.command_file("b", "enabled: false")
        paused_before = paused.read_bytes()

        assert h.store.reschedule("a", NOW, TOMORROW_7) is False  # stale view
        assert h.store.get("a").next_run_at is None
        assert h.store.reschedule("a", None, TOMORROW_7) is True
        assert h.store.get("a").next_run_at == TOMORROW_7
        assert h.store.get("a").last_run_at is None
        assert h.store.reschedule("b", None, TOMORROW_7) is False
        assert paused.read_bytes() == paused_before
        assert h.store.reschedule("missing", None, TOMORROW_7) is False
        assert not (h.dir / "missing.md").exists()
