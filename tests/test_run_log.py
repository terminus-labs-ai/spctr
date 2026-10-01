"""Unit tests for sr2_spectre.run_log (spc-107, spec: live-session-log.md).

Covers SessionLog / SessionLogManager behaviour: path form, envelope,
per-line flush, 16 KiB previews, key redaction, file modes, retention
sweeps with cross-process locking, and warn-don't-raise failure handling.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import select
import stat
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sr2_spectre.run_log import SessionLog, SessionLogManager

KIB16 = 16 * 1024
SECRET_KEYS = ["api_key", "authorization", "token", "password", "secret", "cookie"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "sr2home"
    h.mkdir()
    monkeypatch.setenv("SR2_HOME", str(h))
    return h


def _sessions_dir(home: Path) -> Path:
    return home / "logs" / "sessions"


def _lines(path: Path) -> list[dict]:
    """Read the file through an independent handle, as `tail -f` would."""
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    assert raw == "" or raw.endswith("\n"), "every event must be a complete line"
    return [json.loads(line) for line in raw.splitlines()]


def _walk(obj):
    """Yield (key, value) for every mapping entry, recursively (key None for list items)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield None, v
            yield from _walk(v)


def _strings(obj) -> list[str]:
    out = [obj] if isinstance(obj, str) else []
    out += [v for _, v in _walk(obj) if isinstance(v, str)]
    return out


def _age(path: Path, hours: float) -> None:
    t = time.time() - hours * 3600
    os.utime(path, (t, t))


def _stale_file(home: Path, name: str, hours: float = 25) -> Path:
    d = _sessions_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text('{"event": "old"}\n')
    _age(p, hours)
    return p


def _run_py(code: str, home: Path, **kw) -> subprocess.Popen:
    env = dict(os.environ, SR2_HOME=str(home))
    return subprocess.Popen(
        [sys.executable, "-c", code], env=env, text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, **kw,
    )


def _cleanup_in_other_process(home: Path) -> list[str]:
    code = (
        "import json\n"
        "from sr2_spectre.run_log import SessionLogManager\n"
        "print(json.dumps([str(p) for p in SessionLogManager().cleanup_once()]))\n"
    )
    out, _ = _run_py(code, home).communicate(timeout=30)
    return json.loads(out.strip().splitlines()[-1])


def _warned(caplog) -> bool:
    return any(r.levelno >= logging.WARNING for r in caplog.records)


def _break_writes(path: Path) -> None:
    """Point every descriptor this process holds on *path* at /dev/full (ENOSPC)."""
    target = os.path.realpath(path)
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == target:
                full = os.open("/dev/full", os.O_WRONLY)
                os.dup2(full, int(fd))
                os.close(full)
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Path form, uniqueness, permissions (FR1, FR12)
# ---------------------------------------------------------------------------

class TestPaths:
    def test_path_lives_under_sr2_home_logs_sessions(self, home):
        log = SessionLogManager().open_session("edi-default", "edi")
        try:
            assert isinstance(log, SessionLog)
            assert log.path.parent == _sessions_dir(home)
            assert log.path.suffix == ".jsonl"
            assert log.path.name.startswith(str(datetime.now(timezone.utc).year))
            assert "edi-default" in log.path.name
            assert log.path.exists()
        finally:
            log.close()

    def test_same_session_id_gets_unique_paths(self, home):
        mgr = SessionLogManager()
        a = mgr.open_session("same", "edi")
        b = mgr.open_session("same", "edi")
        try:
            assert a.path != b.path
            assert a.path.exists() and b.path.exists()
        finally:
            a.close()
            b.close()

    def test_unsafe_session_id_is_sanitized_in_name_but_kept_in_events(self, home):
        raw_id = "../chan/x y:*"
        log = SessionLogManager().open_session(raw_id, "edi")
        try:
            log.append("probe", {})
            assert log.path.parent == _sessions_dir(home)
            assert "/" not in log.path.name
            assert _lines(log.path)[-1]["session_id"] == raw_id
        finally:
            log.close()

    def test_directory_is_0700_and_file_is_0600_regardless_of_umask(self, home):
        old = os.umask(0)
        try:
            log = SessionLogManager().open_session("perm", "edi")
            log.append("probe", {})
        finally:
            os.umask(old)
        try:
            assert stat.S_IMODE(_sessions_dir(home).stat().st_mode) == 0o700
            assert stat.S_IMODE(log.path.stat().st_mode) == 0o600
        finally:
            log.close()


# ---------------------------------------------------------------------------
# Envelope, ordering, flush (FR3, FR4)
# ---------------------------------------------------------------------------

class TestEnvelope:
    def test_each_line_is_a_full_envelope(self, home):
        log = SessionLogManager().open_session("edi-default", "edi")
        try:
            log.set_interface("single_shot")
            log.append("session.start", {"k": "v"})
            log.append("turn.start")
            before = datetime.now(timezone.utc)
            events = _lines(log.path)
        finally:
            log.close()
        assert [e["event"] for e in events][-2:] == ["session.start", "turn.start"]
        for e in events[-2:]:
            assert e["schema_version"] == 1
            assert e["session_id"] == "edi-default"
            assert e["interface"] == "single_shot"
            assert isinstance(e["data"], dict)
            assert isinstance(e["elapsed_ms"], (int, float)) and e["elapsed_ms"] >= 0
            ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            assert ts.utcoffset() == timedelta(0)
            assert abs((before - ts).total_seconds()) < 60
        assert events[-2]["data"] == {"k": "v"}
        assert events[-1]["data"] == {}

    def test_sequence_strictly_increases_and_elapsed_never_decreases(self, home):
        log = SessionLogManager().open_session("seq", "edi")
        try:
            for i in range(20):
                log.append("tick", {"i": i})
            events = _lines(log.path)
        finally:
            log.close()
        seqs = [e["sequence"] for e in events]
        assert all(isinstance(s, int) for s in seqs)
        assert all(b > a for a, b in zip(seqs, seqs[1:]))
        elapsed = [e["elapsed_ms"] for e in events]
        assert all(b >= a for a, b in zip(elapsed, elapsed[1:]))

    def test_each_append_is_visible_to_a_concurrent_reader_immediately(self, home):
        log = SessionLogManager().open_session("live", "edi")
        try:
            with open(log.path, encoding="utf-8") as reader:
                reader.read()
                log.append("one", {})
                first = reader.read()
                log.append("two", {})
                second = reader.read()
            assert json.loads(first)["event"] == "one"
            assert json.loads(second)["event"] == "two"
        finally:
            log.close()

    def test_set_interface_applies_to_later_events(self, home):
        log = SessionLogManager().open_session("iface", "edi")
        try:
            log.set_interface("repl")
            log.append("a")
            log.set_interface("discord")
            log.append("b")
            events = _lines(log.path)
        finally:
            log.close()
        assert events[-2]["interface"] == "repl"
        assert events[-1]["interface"] == "discord"


# ---------------------------------------------------------------------------
# Truncation and redaction (FR10, FR11)
# ---------------------------------------------------------------------------

class TestContentBounds:
    def _append_one(self, data) -> dict:
        log = SessionLogManager().open_session("bounds", "edi")
        try:
            log.append("probe", data)
            return _lines(log.path)[-1]
        finally:
            log.close()

    def test_small_text_is_kept_verbatim_without_truncation_flag(self, home):
        line = self._append_one({"input": "hello world"})
        assert "hello world" in _strings(line["data"])
        assert not any(k == "truncated" and v is True for k, v in _walk(line["data"]))

    @pytest.mark.parametrize("original", ["a" * 40_000, "é" * 10_000])
    def test_long_text_is_capped_at_16kib_with_size_and_flag(self, home, original):
        line = self._append_one({"nested": {"content": original}})
        strings = _strings(line["data"])
        assert all(len(s.encode("utf-8")) <= KIB16 for s in strings)
        preview = max(strings, key=len)
        assert preview.startswith(original[:100])
        assert len(preview.encode("utf-8")) >= 15 * 1024
        pairs = list(_walk(line["data"]))
        assert ("truncated", True) in pairs
        original_size = len(original.encode("utf-8"))
        assert any(v == original_size and not isinstance(v, bool) for _, v in pairs)

    @pytest.mark.parametrize("key", SECRET_KEYS + ["API_KEY", "Authorization", "Cookie"])
    def test_secret_keys_are_redacted_case_insensitively(self, home, key):
        line = self._append_one({key: "s3cr3t-value", "keep": "visible"})
        assert key in line["data"]
        assert "s3cr3t-value" not in json.dumps(line)
        assert line["data"]["keep"] == "visible"

    def test_non_exact_secret_like_keys_stay_visible(self, home):
        data = {k: f"v-{k}" for k in
                ("max_tokens", "input_tokens", "output_tokens", "tokens", "token_count")}
        line = self._append_one(data)
        for k, v in data.items():
            assert line["data"][k] == v

    def test_redaction_is_recursive_through_dicts_and_lists(self, home):
        data = {
            "args": {
                "headers": {"Authorization": "Bearer AAA"},
                "items": [{"password": "BBB", "name": "ok"}],
                "Token": {"inner": "CCC"},
            }
        }
        line = self._append_one(data)
        text = json.dumps(line)
        for secret in ("AAA", "BBB", "CCC"):
            assert secret not in text
        assert "ok" in _strings(line["data"])

    def test_unserializable_values_do_not_raise_or_corrupt_the_file(self, home):
        log = SessionLogManager().open_session("odd", "edi")
        try:
            log.append("odd", {"obj": object(), "path": Path("/x"), "raw": b"\x00\x01"})
            log.append("after", {})
            events = _lines(log.path)
        finally:
            log.close()
        assert events[-1]["event"] == "after"


# ---------------------------------------------------------------------------
# Retention and locking (FR13, FR14)
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_deletes_only_stale_unlocked_logs(self, home):
        stale = _stale_file(home, "20200101T000000Z-old-aaaa.jsonl", hours=25)
        fresh = _stale_file(home, "20200101T000000Z-new-bbbb.jsonl", hours=23)
        other = _stale_file(home, "notes.txt", hours=100)
        deleted = SessionLogManager().cleanup_once()
        assert not stale.exists()
        assert fresh.exists()
        assert other.exists()
        assert [Path(p) for p in deleted] == [stale]

    def test_now_parameter_sets_the_reference_time(self, home):
        recent = _stale_file(home, "20200101T000000Z-r-cccc.jsonl", hours=1)
        mgr = SessionLogManager()
        assert mgr.cleanup_once(now=datetime.now(timezone.utc)) == []
        assert recent.exists()
        mgr.cleanup_once(now=datetime.now(timezone.utc) + timedelta(hours=25))
        assert not recent.exists()

    def test_missing_directory_is_not_an_error(self, home):
        assert SessionLogManager().cleanup_once() == []

    def test_log_held_by_another_live_process_survives(self, home):
        holder = _run_py(
            "import os, sys, time\n"
            "from sr2_spectre.run_log import SessionLogManager\n"
            "log = SessionLogManager().open_session('held', 'edi')\n"
            "log.append('x', {})\n"
            "t = time.time() - 48 * 3600\n"
            "os.utime(log.path, (t, t))\n"
            "print(log.path, flush=True)\n"
            "sys.stdin.readline()\n",
            home,
        )
        try:
            ready, _, _ = select.select([holder.stdout], [], [], 30)
            assert ready, "holder process did not report its log path"
            path = Path(holder.stdout.readline().strip())
            assert path.exists()
            assert SessionLogManager().cleanup_once() == []
            assert path.exists()
        finally:
            holder.stdin.close()
            holder.wait(timeout=30)
        SessionLogManager().cleanup_once()
        assert not path.exists()

    def test_open_log_survives_cleanup_in_same_and_other_process(self, home):
        mgr = SessionLogManager()
        log = mgr.open_session("mine", "edi")
        try:
            log.append("x", {})
            _age(log.path, 48)
            assert log.path not in [Path(p) for p in mgr.cleanup_once()]
            assert log.path.exists()
            # The in-process sweep must not have dropped the writer's lock.
            assert _cleanup_in_other_process(home) == []
            assert log.path.exists()
        finally:
            log.close()

    def test_close_releases_the_lock(self, home):
        log = SessionLogManager().open_session("closed", "edi")
        log.append("x", {})
        log.close()
        _age(log.path, 48)
        assert _cleanup_in_other_process(home) == [str(log.path)]

    def test_cleanup_failure_warns_instead_of_raising(self, home, caplog):
        if os.geteuid() == 0:
            pytest.skip("root ignores directory write permission")
        stale = _stale_file(home, "20200101T000000Z-x-dddd.jsonl")
        d = _sessions_dir(home)
        d.chmod(0o500)
        try:
            with caplog.at_level(logging.WARNING):
                SessionLogManager().cleanup_once()
        finally:
            d.chmod(0o700)
        assert stale.exists()
        assert _warned(caplog)

    def test_default_sweep_interval_is_24_hours(self):
        param = inspect.signature(SessionLogManager).parameters["cleanup_interval"]
        assert param.default == 24 * 60 * 60


class TestManagerLifecycle:
    async def test_start_sweeps_then_repeats_and_aclose_stops_it(self, home):
        first = _stale_file(home, "20200101T000000Z-a-0001.jsonl")
        mgr = SessionLogManager(cleanup_interval=0.05)
        await mgr.start()
        try:
            assert not first.exists(), "startup sweep must run before start() returns"
            second = _stale_file(home, "20200101T000000Z-b-0002.jsonl")
            for _ in range(100):
                if not second.exists():
                    break
                await asyncio.sleep(0.02)
            assert not second.exists(), "periodic sweep did not run"
        finally:
            await mgr.aclose()
        third = _stale_file(home, "20200101T000000Z-c-0003.jsonl")
        await asyncio.sleep(0.3)
        assert third.exists(), "periodic sweep still running after aclose()"

    async def test_aclose_closes_open_logs_and_releases_locks(self, home):
        mgr = SessionLogManager()
        await mgr.start()
        log = mgr.open_session("s", "edi")
        log.append("x", {})
        await mgr.aclose()
        _age(log.path, 48)
        assert _cleanup_in_other_process(home) == [str(log.path)]

    async def test_aclose_is_safe_without_start_and_twice(self, home):
        mgr = SessionLogManager()
        mgr.open_session("s", "edi")
        await mgr.aclose()
        await mgr.aclose()

    async def test_start_with_unusable_home_warns_and_does_not_raise(
        self, tmp_path, monkeypatch, caplog
    ):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        monkeypatch.setenv("SR2_HOME", str(blocker))
        mgr = SessionLogManager()
        with caplog.at_level(logging.WARNING):
            await mgr.start()
            await mgr.aclose()
        assert _warned(caplog)


# ---------------------------------------------------------------------------
# SR2_SESSION_LOG_DIR override (obsidian-fa6t)
# ---------------------------------------------------------------------------

class TestDirectoryOverride:
    def test_unset_or_empty_keeps_the_sr2_home_default(self, home, monkeypatch):
        monkeypatch.delenv("SR2_SESSION_LOG_DIR", raising=False)
        assert SessionLogManager().directory == _sessions_dir(home)
        monkeypatch.setenv("SR2_SESSION_LOG_DIR", "")
        assert SessionLogManager().directory == _sessions_dir(home)

    def test_path_moves_logs_out_of_sr2_home(self, home, tmp_path, monkeypatch):
        target = tmp_path / "elsewhere" / "sessions"
        monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(target))
        log = SessionLogManager().open_session("s", "edi")
        try:
            assert log.path.parent == target
            assert not (home / "logs").exists()
        finally:
            log.close()

    @pytest.mark.parametrize("value", ["off", "OFF", " off "])
    async def test_off_disables_without_touching_disk_or_warning(
        self, home, monkeypatch, caplog, value
    ):
        monkeypatch.setenv("SR2_SESSION_LOG_DIR", value)
        mgr = SessionLogManager()
        assert mgr.enabled is False
        with caplog.at_level(logging.WARNING):
            await mgr.start()
            await mgr.aclose()
        assert not _warned(caplog)
        assert not (home / "logs").exists()
        with pytest.raises(OSError):
            mgr.open_session("s", "edi")

    def test_enabled_by_default(self, home):
        assert SessionLogManager().enabled is True


# ---------------------------------------------------------------------------
# Append failures (FR15)
# ---------------------------------------------------------------------------

class TestAppendFailures:
    def test_write_failure_warns_and_does_not_raise(self, home, caplog):
        log = SessionLogManager().open_session("full", "edi")
        log.append("before", {})
        _break_writes(log.path)
        with caplog.at_level(logging.WARNING):
            log.append("during", {"x": 1})
            log.append("again", {"x": 2})
            log.close()
        assert _warned(caplog)

    def test_append_after_close_does_not_raise(self, home):
        log = SessionLogManager().open_session("late", "edi")
        log.close()
        log.append("late", {})
