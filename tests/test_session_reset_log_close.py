"""A replaced Session's log closes at the swap, not at cyclic GC (obsidian-ckst).

When an Agent replaces its Session — REPL/TUI ``/reset`` via
``Agent.new_session()``, or Discord via the ``session_id`` setter — the
outgoing Session's log must be closed right away: its file descriptor is
released and its exclusive ``flock`` dropped, so retention cleanup can treat
the file as no longer live. This must not depend on the garbage collector, so
every test here runs with ``gc`` disabled and never calls ``gc.collect()``.

The replacement Session's log stays open, locked and writable; the old log's
contents are left intact.
"""

from __future__ import annotations

import fcntl
import gc
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from sr2.protocols.llm import StreamEvent
from sr2_spectre.run_log import SessionLogManager
from tests.test_session_reset_log import _files, _new_files, _repl_ctx
from tests.test_session_run_log import ScriptedLLM, _agent, _events, _runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_cyclic_gc(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep cyclic GC from closing a leaked log and passing a test by accident.

    Automatic collection is disabled, and an explicit ``gc.collect()`` from
    anywhere fails the test: closing the log must not rely on the collector.
    """

    def _forbidden_collect(*_args, **_kwargs):
        raise AssertionError("fix must not rely on gc.collect")

    monkeypatch.setattr(gc, "collect", _forbidden_collect)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@pytest.fixture
def log_dir() -> Path:
    """The session-log directory under the conftest-isolated SR2_HOME."""
    return SessionLogManager().directory


def _open_in_process(path: Path) -> bool:
    """True if this process still has a file descriptor open on ``path``."""
    target = os.path.realpath(path)
    fd_dir = Path("/proc/self/fd")
    for entry in fd_dir.iterdir():
        try:
            if os.path.realpath(os.readlink(entry)) == target:
                return True
        except OSError:
            continue
    return False


def _lockable(path: Path) -> bool:
    """True if a fresh non-blocking exclusive flock on ``path`` succeeds."""
    with open(path, "rb") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
        return True


def _age(path: Path, hours: float = 48) -> None:
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def _assert_released(path: Path) -> None:
    assert not _open_in_process(path), f"replaced Session still holds an fd on {path}"
    assert _lockable(path), f"replaced Session still holds the flock on {path}"


def _assert_live(path: Path) -> None:
    assert _open_in_process(path), f"current Session's log {path} is not open"
    assert not _lockable(path), f"current Session's log {path} is not locked"


# ---------------------------------------------------------------------------
# AC: after new_session(), the previous log handle is closed without GC
# ---------------------------------------------------------------------------

class TestNewSessionClosesPreviousLog:
    def test_previous_log_fd_and_lock_released_immediately(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        _assert_live(first)

        agent.new_session()

        _assert_released(first)

    def test_retention_cleanup_can_remove_the_replaced_log(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)

        agent.new_session()
        (second,) = _new_files(log_dir, [first])
        _age(first)
        _age(second)

        deleted = SessionLogManager().cleanup_once()

        assert first in deleted
        assert not first.exists()
        assert second.exists(), "cleanup removed the live replacement log"
        assert second not in deleted

    def test_every_replaced_log_is_released_across_repeated_resets(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        agent.new_session()
        agent.new_session()

        files = _files(log_dir)
        assert len(files) == 3
        live = [f for f in files if _open_in_process(f)]
        assert len(live) == 1, f"expected only the current log open, got {live}"
        for f in files:
            if f not in live:
                _assert_released(f)


# ---------------------------------------------------------------------------
# AC: same via the session_id setter (Discord replacement path)
# ---------------------------------------------------------------------------

class TestSessionIdSetterClosesPreviousLog:
    def test_previous_log_fd_and_lock_released_immediately(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        _assert_live(first)

        agent.session_id = "channel-7"

        _assert_released(first)

    async def test_discord_channel_switch_releases_previous_log(self, log_dir):
        from sr2_spectre.interfaces.discord.session_map import ChannelSession
        from tests.test_session_reset_log import _start_discord

        agent = _agent()
        iface = await _start_discord(agent)
        (first,) = _files(log_dir)
        _assert_live(first)

        iface._restore_history(ChannelSession(channel_id=42, session_id="discord-42"))

        _assert_released(first)


# ---------------------------------------------------------------------------
# AC: the replacement log stays open and writable; old contents intact
# ---------------------------------------------------------------------------

class TestReplacementLogStaysLive:
    async def test_replacement_log_is_open_locked_and_writable(self, log_dir):
        agent = _agent(ScriptedLLM([StreamEvent(type="text", text="ok")]))
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)

        agent.new_session()
        (second,) = _new_files(log_dir, [first])
        _assert_live(second)
        before = _events(second)

        await agent.handle_user_message("after reset")

        after = _events(second)
        assert after[: len(before)] == before
        assert any(e["event"].startswith("turn.") for e in after[len(before):]), after
        _assert_live(second)

    def test_setter_replacement_log_is_open_and_locked(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)

        agent.session_id = "channel-7"

        (second,) = _new_files(log_dir, [first])
        _assert_live(second)

    async def test_old_log_contents_are_intact_after_replacement(self, log_dir):
        agent = _agent(ScriptedLLM([StreamEvent(type="text", text="before")]))
        agent.set_run_context(_repl_ctx())
        await agent.handle_user_message("before reset")
        (first,) = _files(log_dir)
        raw_before = first.read_bytes()
        assert raw_before, "precondition: the first log has content"

        agent.new_session()

        assert first.read_bytes() == raw_before


# ---------------------------------------------------------------------------
# Sibling Sessions under one Runtime keep their own logs (spec FR13, NFR
# "Concurrent Sessions write separate files")
# ---------------------------------------------------------------------------

class TestSiblingSessionsStayLive:
    def test_concurrent_sessions_each_hold_a_live_log(self, log_dir):
        runtime = _runtime(ScriptedLLM())
        s1 = runtime.new_session(frame_id="a")
        s1.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        s2 = runtime.new_session(frame_id="b")
        s2.set_run_context(_repl_ctx())
        (second,) = _new_files(log_dir, [first])

        _assert_live(first)
        _assert_live(second)

    def test_agent_reset_leaves_a_sibling_sessions_log_live(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        # Agent exposes no public Runtime accessor; a sibling Session must
        # come from the same Runtime the Agent replaces Sessions under.
        sibling = agent._runtime.new_session(frame_id="sibling")
        sibling.set_run_context(_repl_ctx())
        (sibling_log,) = _new_files(log_dir, [first])

        agent.new_session()

        _assert_released(first)
        _assert_live(sibling_log)

    def test_setter_replacement_leaves_a_sibling_sessions_log_live(self, log_dir):
        agent = _agent()
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        sibling = agent._runtime.new_session(frame_id="sibling")
        sibling.set_run_context(_repl_ctx())
        (sibling_log,) = _new_files(log_dir, [first])

        agent.session_id = "channel-7"

        _assert_released(first)
        _assert_live(sibling_log)


# ---------------------------------------------------------------------------
# AC: replacing a Session that never opened a log does not raise
# ---------------------------------------------------------------------------

class TestReplacingSessionWithoutLog:
    def test_new_session_without_run_context_does_not_raise(self, log_dir):
        agent = _agent()
        agent.new_session()
        agent.new_session("again")

        assert _files(log_dir) == []
        assert agent.session_id == "again"

    def test_setter_without_run_context_does_not_raise(self, log_dir):
        agent = _agent()
        agent.session_id = "channel-7"

        assert _files(log_dir) == []
        assert agent.session_id == "channel-7"
