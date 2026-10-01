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

import asyncio
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
from tests.test_session_run_log import (
    ScriptedLLM,
    _agent,
    _events,
    _register,
    _runtime,
    _tool_call,
)


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


# ---------------------------------------------------------------------------
# Replacement while a turn is in flight (Discord replaces the Session on every
# message while earlier turns still run as concurrent tasks). The outgoing log
# stays open and locked until that turn ends, records the rest of the turn
# through its terminal event, and only then is released. Spec FR5, FR13.
# ---------------------------------------------------------------------------

class GatedLLM:
    """LLMCallable whose first stream() blocks mid-response until released.

    ``entered`` is set once the model has started streaming. After ``gate``
    is set the stream either finishes normally or raises ``fail_with``.
    """

    model = "test-model"

    def __init__(self, fail_with: BaseException | None = None) -> None:
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()
        self._fail_with = fail_with
        self._calls = 0

    async def complete(self, request):
        raise NotImplementedError

    async def stream(self, request):
        self._calls += 1
        if self._calls == 1:
            yield StreamEvent(type="text", text="partial ")
            self.entered.set()
            await self.gate.wait()
            if self._fail_with is not None:
                raise self._fail_with
            yield StreamEvent(type="text", text="rest")
        else:
            yield StreamEvent(type="text", text="later")
        yield StreamEvent(type="end")


async def _consume(agent) -> list:
    return [ev async for ev in agent.stream_message("in flight")]


async def _start_turn(agent, llm: GatedLLM) -> asyncio.Task:
    task = asyncio.create_task(_consume(agent))
    await asyncio.wait_for(llm.entered.wait(), timeout=5)
    return task


def _names(path: Path) -> list[str]:
    return [e["event"] for e in _events(path)]


def _replace_by_new_session(agent) -> None:
    agent.new_session("replacement")


def _replace_by_setter(agent) -> None:
    agent.session_id = "replacement"


REPLACERS = pytest.mark.parametrize(
    "replace", [_replace_by_new_session, _replace_by_setter], ids=["new_session", "setter"],
)


class TestReplacementDuringTurn:
    @REPLACERS
    async def test_old_log_stays_open_and_locked_while_turn_runs(self, log_dir, replace):
        llm = GatedLLM()
        agent = _agent(llm)
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        task = await _start_turn(agent, llm)

        replace(agent)
        try:
            _assert_live(first)
        finally:
            llm.gate.set()
            await asyncio.wait_for(task, timeout=5)

    @REPLACERS
    async def test_old_log_records_rest_of_turn_then_is_released(self, log_dir, replace):
        llm = GatedLLM()
        agent = _agent(llm)
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        task = await _start_turn(agent, llm)
        replace(agent)
        at_swap = _names(first)
        assert "turn.start" in at_swap
        assert "turn.complete" not in at_swap

        llm.gate.set()
        await asyncio.wait_for(task, timeout=5)

        names = _names(first)
        assert names[: len(at_swap)] == at_swap, "old log rewritten"
        tail = names[len(at_swap):]
        assert "model.end" in tail, f"model.end dropped after swap: {names}"
        assert tail[-1] == "turn.complete", f"turn did not finish in old log: {names}"
        _assert_released(first)

    @REPLACERS
    async def test_in_flight_turn_does_not_write_to_replacement_log(self, log_dir, replace):
        llm = GatedLLM()
        agent = _agent(llm)
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        task = await _start_turn(agent, llm)
        replace(agent)
        (second,) = _new_files(log_dir, [first])

        llm.gate.set()
        await asyncio.wait_for(task, timeout=5)

        assert _names(second) == ["session.start"]
        _assert_live(second)

    @REPLACERS
    async def test_turn_error_is_logged_then_old_log_is_released(self, log_dir, replace):
        llm = GatedLLM(fail_with=RuntimeError("model blew up"))
        agent = _agent(llm)
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        task = await _start_turn(agent, llm)
        replace(agent)

        llm.gate.set()
        with pytest.raises(RuntimeError, match="model blew up"):
            await asyncio.wait_for(task, timeout=5)

        names = _names(first)
        assert names[-1] == "turn.error", f"turn.error not last in old log: {names}"
        _assert_released(first)

    @REPLACERS
    async def test_turn_cancel_is_logged_then_old_log_is_released(self, log_dir, replace):
        llm = GatedLLM()
        agent = _agent(llm)
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        task = await _start_turn(agent, llm)
        replace(agent)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        names = _names(first)
        assert names[-1] == "turn.cancel", f"turn.cancel not last in old log: {names}"
        _assert_released(first)


class TestDeferredCloseIsSafe:
    async def test_second_replacement_while_close_pending(self, log_dir):
        llm = GatedLLM()
        agent = _agent(llm)
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        task = await _start_turn(agent, llm)

        agent.session_id = "second"
        (second,) = _new_files(log_dir, [first])
        agent.new_session("third")
        (third,) = _new_files(log_dir, [first, second])

        _assert_live(first)
        _assert_released(second)
        _assert_live(third)

        llm.gate.set()
        await asyncio.wait_for(task, timeout=5)

        assert _names(first)[-1] == "turn.complete"
        _assert_released(first)
        _assert_live(third)

    async def test_close_twice_mid_turn_defers_once_and_does_not_raise(self, log_dir):
        llm = GatedLLM()
        runtime = _runtime(llm)
        session = runtime.new_session(frame_id="a")
        session.set_run_context(_repl_ctx())
        (log,) = _files(log_dir)
        sibling = runtime.new_session(frame_id="b")
        sibling.set_run_context(_repl_ctx())
        (sibling_log,) = _new_files(log_dir, [log])

        async def consume():
            return [ev async for ev in session.stream_message("in flight")]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(llm.entered.wait(), timeout=5)

        session.close()
        session.close()
        _assert_live(log)

        llm.gate.set()
        await asyncio.wait_for(task, timeout=5)

        assert _names(log)[-1] == "turn.complete"
        _assert_released(log)
        session.close()
        _assert_live(sibling_log)

    def test_close_twice_with_no_turn_closes_now_and_does_not_raise(self, log_dir):
        runtime = _runtime(ScriptedLLM())
        session = runtime.new_session(frame_id="a")
        session.set_run_context(_repl_ctx())
        (log,) = _files(log_dir)
        sibling = runtime.new_session(frame_id="b")
        sibling.set_run_context(_repl_ctx())
        (sibling_log,) = _new_files(log_dir, [log])

        session.close()
        _assert_released(log)
        session.close()

        _assert_released(log)
        _assert_live(sibling_log)

    async def test_sibling_log_stays_live_through_deferred_close(self, log_dir):
        llm = GatedLLM()
        agent = _agent(llm)
        agent.set_run_context(_repl_ctx())
        (first,) = _files(log_dir)
        sibling = agent._runtime.new_session(frame_id="sibling")
        sibling.set_run_context(_repl_ctx())
        (sibling_log,) = _new_files(log_dir, [first])
        task = await _start_turn(agent, llm)

        agent.new_session()
        _assert_live(sibling_log)

        llm.gate.set()
        await asyncio.wait_for(task, timeout=5)

        _assert_released(first)
        _assert_live(sibling_log)


class TwoTurnLLM:
    """LLMCallable gating each of its first two stream() calls separately.

    ``entered[i]`` is set once call ``i`` has started streaming; it then
    blocks until ``gate[i]`` is set.
    """

    model = "test-model"

    def __init__(self) -> None:
        self.entered = [asyncio.Event(), asyncio.Event()]
        self.gate = [asyncio.Event(), asyncio.Event()]
        self._calls = 0

    async def complete(self, request):
        raise NotImplementedError

    async def stream(self, request):
        i = self._calls
        self._calls += 1
        yield StreamEvent(type="text", text=f"turn {i} ")
        if i < 2:
            self.entered[i].set()
            await self.gate[i].wait()
        yield StreamEvent(type="end")


class TestDeferredCloseWithQueuedTurn:
    async def test_close_waits_for_every_in_flight_turn(self, log_dir):
        llm = TwoTurnLLM()
        runtime = _runtime(llm)
        session = runtime.new_session(frame_id="a")
        session.set_run_context(_repl_ctx())
        (log,) = _files(log_dir)

        async def consume(text):
            return [ev async for ev in session.stream_message(text)]

        turn1 = asyncio.create_task(consume("one"))
        await asyncio.wait_for(llm.entered[0].wait(), timeout=5)
        turn2 = asyncio.create_task(consume("two"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert _names(log).count("turn.start") == 2, "precondition: turn 2 queued"

        session.close()
        _assert_live(log)

        llm.gate[0].set()
        await asyncio.wait_for(turn1, timeout=5)
        await asyncio.wait_for(llm.entered[1].wait(), timeout=5)
        try:
            _assert_live(log)
        finally:
            llm.gate[1].set()
            await asyncio.wait_for(turn2, timeout=5)

        names = _names(log)
        assert names.count("turn.complete") == 2, names
        first_done = names.index("turn.complete")
        assert names[-1] == "turn.complete", names
        assert names[first_done + 1:].count("model.start") == 1, (
            f"turn 2's model events missing after turn 1 completed: {names}"
        )
        assert "model.end" in names[first_done + 1:], names
        _assert_released(log)


class TestDeferredCloseWithAbandonedStream:
    async def test_aclose_of_in_flight_stream_releases_log(self, log_dir):
        llm = ScriptedLLM(
            [_tool_call("t1", "noop")],
            [StreamEvent(type="text", text="never reached")],
        )
        runtime = _runtime(llm)
        _register(runtime, "noop", lambda **_: "ok")
        session = runtime.new_session(frame_id="a")
        session.set_run_context(_repl_ctx())
        (log,) = _files(log_dir)

        gen = session.stream_message("abandoned")
        try:
            await asyncio.wait_for(gen.__anext__(), timeout=5)
            session.close()
            _assert_live(log)
        finally:
            await gen.aclose()

        _assert_released(log)
