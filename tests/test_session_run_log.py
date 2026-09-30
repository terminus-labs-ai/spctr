"""Runtime/Session wiring for the live session log (spc-107).

Drives a real SR2 turn loop with a scripted LLM so tool events come from the
actual Session tool-executor seam. Interfaces are exercised only through the
shared run-context seam they all use.
"""

from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from sr2.protocols.llm import StreamEvent
from sr2_spectre.agent import Agent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.runtime import Runtime

KIB16 = 16 * 1024
SYSTEM_MARKER = "SYSTEM-PROMPT-MARKER-7f3a"
INTERFACES_DIR = Path(__file__).resolve().parents[1] / "src" / "sr2_spectre" / "interfaces"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "sr2home"
    h.mkdir()
    monkeypatch.setenv("SR2_HOME", str(h))
    monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
    return h


def _config() -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="edi"),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={"layers": [
            {"name": "system", "target": "system",
             "resolvers": [{"type": "static", "config": {"text": SYSTEM_MARKER}}]},
            {"name": "tools", "target": "tools", "resolvers": [],
             "tool_providers": [{"type": "spectre_tools"}]},
            {"name": "conversation", "target": "messages",
             "resolvers": [{"type": "session"}, {"type": "input"}]},
        ]},
        provenance_store_path="",
        memory_store_dsn="",
    )


class ScriptedLLM:
    """LLMCallable whose successive stream() calls replay scripted rounds."""

    model = "test-model"

    def __init__(self, *rounds) -> None:
        self._rounds = list(rounds)

    async def complete(self, request):
        raise NotImplementedError

    async def stream(self, request):
        round_ = self._rounds.pop(0) if self._rounds else [StreamEvent(type="text", text="done")]
        if isinstance(round_, Exception):
            raise round_
        for ev in round_:
            yield ev
        yield StreamEvent(type="end")


def _tool_call(tool_id: str, name: str, **args) -> StreamEvent:
    return StreamEvent(type="tool_use", tool_use_id=tool_id, tool_name=name, tool_input=args)


def _runtime(llm: ScriptedLLM) -> Runtime:
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=llm):
        return Runtime(_config())


def _agent(llm: ScriptedLLM | None = None) -> Agent:
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=llm or ScriptedLLM()):
        return Agent(config=_config(), session_id="edi-default")


def _register(runtime: Runtime, name: str, fn) -> None:
    runtime.registry.register(name, f"{name} tool", {"type": "object"}, fn)


def _ctx(interface: str = "single_shot") -> RunContext:
    return RunContext(interface=interface, mode=RunMode.HEADLESS, source=None)


def _log_files(home: Path) -> list[Path]:
    return sorted((home / "logs" / "sessions").glob("*.jsonl"))


def _only_log(home: Path) -> Path:
    files = _log_files(home)
    assert len(files) == 1, files
    return files[0]


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _named(events: list[dict], name: str) -> list[dict]:
    return [e for e in events if e["event"] == name]


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield None, v
            yield from _walk(v)


def _strings(obj) -> list[str]:
    return [v for _, v in _walk(obj) if isinstance(v, str)]


def _for_tool(events: list[dict], name: str, tool_id: str) -> list[dict]:
    return [e for e in _named(events, name) if e["data"].get("tool_use_id") == tool_id]


def _warned_about(caplog, text: str) -> bool:
    return any(
        r.levelno >= logging.WARNING and text in r.getMessage() for r in caplog.records
    )


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


def _cleanup_in_other_process(home: Path) -> list[str]:
    code = (
        "import json\n"
        "from sr2_spectre.run_log import SessionLogManager\n"
        "print(json.dumps([str(p) for p in SessionLogManager().cleanup_once()]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=dict(os.environ, SR2_HOME=str(home)),
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# AC 1 — every interface, one shared seam
# ---------------------------------------------------------------------------

async def _start_single_shot(agent):
    from sr2_spectre.interfaces.single_shot import SingleShotInterface
    await SingleShotInterface(prompt="x").start(agent)


async def _start_repl(agent):
    from sr2_spectre.interfaces.repl import REPLInterface
    await REPLInterface(console=Console(file=io.StringIO())).start(agent)


async def _start_tui(agent):
    from sr2_spectre.interfaces.tui import TUIInterface
    await TUIInterface().start(agent)


async def _start_discord(agent):
    from sr2_spectre.interfaces.discord.config import DiscordConfig
    from sr2_spectre.interfaces.discord.interface import DiscordInterface

    adapter = AsyncMock()
    adapter.bot_id = 12345
    adapter.bot_mentions = []
    adapter.set_message_handler = MagicMock()
    adapter.set_slash_handler = MagicMock()
    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter",
        return_value=adapter,
    ):
        await DiscordInterface(config=DiscordConfig()).start(agent)


class TestSessionStart:
    @pytest.mark.parametrize("interface,starter", [
        ("single_shot", _start_single_shot),
        ("repl", _start_repl),
        ("tui", _start_tui),
        ("discord", _start_discord),
    ])
    async def test_run_context_announces_path_and_records_interface(
        self, home, capsys, interface, starter
    ):
        agent = _agent()
        await starter(agent)
        path = _only_log(home)
        starts = _named(_events(path), "session.start")
        assert starts, "session.start not recorded"
        assert starts[0]["interface"] == interface
        assert starts[0]["session_id"] == "edi-default"
        out = capsys.readouterr()
        assert str(path) in out.err
        assert str(path) not in out.out

    def test_each_session_gets_its_own_file(self, home):
        runtime = _runtime(ScriptedLLM())
        a = runtime.new_session("same-frame")
        b = runtime.new_session("same-frame")
        a.set_run_context(_ctx("repl"))
        b.set_run_context(_ctx("repl"))
        files = _log_files(home)
        assert len(files) == 2
        for f in files:
            assert _named(_events(f), "session.start")

    async def test_repeated_run_context_records_one_start_and_one_announcement(
        self, home, capsys
    ):
        runtime = _runtime(ScriptedLLM([StreamEvent(type="text", text="done")]))
        session = runtime.new_session("f")
        session.set_run_context(_ctx("discord"))
        session.set_run_context(RunContext(
            interface="discord", mode=RunMode.INTERACTIVE, source=None, area="sr2-spectre",
        ))
        session.set_run_context(_ctx("repl"))
        path = _only_log(home)
        await session.handle_user_message("after the context changes")
        events = _events(path)
        assert len(_named(events, "session.start")) == 1
        assert capsys.readouterr().err.count(str(path)) == 1
        later = [e for e in events if e["event"] != "session.start"]
        assert later and later[-1]["interface"] == "repl"

    def test_discord_session_rebuild_opens_a_new_log(self, home):
        agent = _agent()
        agent.set_run_context(_ctx("discord"))
        agent.session_id = "channel-42"
        agent.set_run_context(_ctx("discord"))
        files = _log_files(home)
        assert len(files) == 2
        ids = {_named(_events(f), "session.start")[0]["session_id"] for f in files}
        assert ids == {"edi-default", "channel-42"}

    def test_discarded_session_releases_its_log_while_runtime_lives(self, home):
        agent = _agent()
        agent.set_run_context(_ctx("discord"))
        (first,) = _log_files(home)
        agent.session_id = "channel-42"
        agent.set_run_context(_ctx("discord"))
        gc.collect()
        (current,) = [p for p in _log_files(home) if p != first]
        t = time.time() - 48 * 3600
        for p in (first, current):
            os.utime(p, (t, t))
        deleted = _cleanup_in_other_process(home)
        assert str(first) in deleted
        assert str(current) not in deleted
        assert current.exists()

    def test_interfaces_contain_no_logging_code(self):
        for src in INTERFACES_DIR.rglob("*.py"):
            text = src.read_text(encoding="utf-8")
            assert "run_log" not in text and "SessionLog" not in text, src


# ---------------------------------------------------------------------------
# AC 2 / AC 3 — live turn and tool events
# ---------------------------------------------------------------------------

class TestTurnEvents:
    async def test_completed_turn_records_input_preview_and_duration(self, home):
        runtime = _runtime(ScriptedLLM([StreamEvent(type="text", text="done")]))
        session = runtime.new_session("f")
        session.set_run_context(_ctx())
        result = await session.handle_user_message("please summarise the notes")
        assert result.text == "done"
        events = _events(_only_log(home))
        names = [e["event"] for e in events]
        assert names.index("session.start") < names.index("turn.start") < names.index("turn.complete")
        assert any("please summarise the notes" in s for s in _strings(_named(events, "turn.start")[0]["data"]))
        assert _named(events, "turn.complete")[0]["data"]["duration_ms"] >= 0
        assert all(e["interface"] == "single_shot" for e in events)

    async def test_turn_error_is_recorded_and_the_error_still_propagates(self, home):
        runtime = _runtime(ScriptedLLM(RuntimeError("llm down")))
        session = runtime.new_session("f")
        session.set_run_context(_ctx())
        with pytest.raises(RuntimeError, match="llm down"):
            await session.handle_user_message("hi")
        events = _events(_only_log(home))
        errors = _named(events, "turn.error")
        assert errors and not _named(events, "turn.complete")
        assert errors[0]["data"]["duration_ms"] >= 0
        assert any("llm down" in s for s in _strings(errors[0]["data"]))

    async def test_turn_cancel_is_recorded_and_cancellation_propagates(self, home):
        started = asyncio.Event()

        async def hang(**_):
            started.set()
            await asyncio.Event().wait()

        runtime = _runtime(ScriptedLLM([_tool_call("tu_h", "hang")]))
        _register(runtime, "hang", hang)
        session = runtime.new_session("f")
        session.set_run_context(_ctx())
        task = asyncio.create_task(session.handle_user_message("go"))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        cancels = _named(_events(_only_log(home)), "turn.cancel")
        assert cancels and cancels[0]["data"]["duration_ms"] >= 0


class TestLiveToolEvents:
    async def test_tool_events_are_readable_while_the_turn_is_running(self, home):
        release = asyncio.Event()
        slow_started = asyncio.Event()

        async def fast(**_):
            return "fast-result"

        async def slow(**_):
            slow_started.set()
            await release.wait()
            return "slow-result"

        runtime = _runtime(ScriptedLLM(
            [_tool_call("tu_fast", "fast"), _tool_call("tu_slow", "slow")],
            [StreamEvent(type="text", text="done")],
        ))
        _register(runtime, "fast", fast)
        _register(runtime, "slow", slow)
        session = runtime.new_session("f")
        session.set_run_context(_ctx())
        path = _only_log(home)

        task = asyncio.create_task(session.handle_user_message("run both"))
        try:
            await asyncio.wait_for(slow_started.wait(), 5)
            await _wait_for(lambda: _for_tool(_events(path), "tool.complete", "tu_fast"))
            mid = _events(path)
            assert _named(mid, "turn.start")
            assert _for_tool(mid, "tool.start", "tu_fast")
            assert _for_tool(mid, "tool.start", "tu_slow")
            assert not _for_tool(mid, "tool.complete", "tu_slow")
            assert not _named(mid, "turn.complete")
        finally:
            release.set()
        assert (await task).text == "done"

        final = _events(path)
        seqs = [e["sequence"] for e in final]
        assert all(b > a for a, b in zip(seqs, seqs[1:]))
        names = [e["event"] for e in final]
        assert names[-1] == "turn.complete"
        slow_done = final.index(_for_tool(final, "tool.complete", "tu_slow")[0])
        assert slow_done < names.index("turn.complete")

    async def test_tool_event_fields_redaction_and_truncation(self, home):
        async def fetch(**_):
            await asyncio.sleep(0.05)
            return "r" * 40_000

        runtime = _runtime(ScriptedLLM([_tool_call(
            "tu_1", "fetch", path="notes/a.txt", api_key="SEKRET-1", blob="z" * 20_000,
        )]))
        _register(runtime, "fetch", fetch)
        session = runtime.new_session("f")
        session.set_run_context(_ctx())
        await session.handle_user_message("fetch it")
        events = _events(_only_log(home))

        (start,) = _for_tool(events, "tool.start", "tu_1")
        assert start["data"]["name"] == "fetch"
        assert "notes/a.txt" in _strings(start["data"])
        assert ("truncated", True) in list(_walk(start["data"]))

        (done,) = _for_tool(events, "tool.complete", "tu_1")
        data = done["data"]
        assert data["name"] == "fetch"
        assert data["is_error"] is False
        assert data["result_bytes"] == 40_000
        assert data["duration_ms"] >= 40
        assert any(s.startswith("rrrr") for s in _strings(data))
        assert ("truncated", True) in list(_walk(data))

        text = _only_log(home).read_text(encoding="utf-8")
        assert "SEKRET-1" not in text
        for e in events:
            assert all(len(s.encode("utf-8")) <= KIB16 for s in _strings(e["data"]))

    async def test_failing_tool_records_error_status(self, home):
        async def broken(**_):
            raise ValueError("boom-detail")

        runtime = _runtime(ScriptedLLM([_tool_call("tu_x", "broken")]))
        _register(runtime, "broken", broken)
        session = runtime.new_session("f")
        session.set_run_context(_ctx())
        result = await session.handle_user_message("try")
        assert result.text == "done"
        (done,) = _for_tool(_events(_only_log(home)), "tool.complete", "tu_x")
        assert done["data"]["is_error"] is True
        assert done["data"]["duration_ms"] >= 0
        assert any("boom-detail" in s for s in _strings(done["data"]))

    async def test_compiled_prompt_is_not_logged(self, home):
        runtime = _runtime(ScriptedLLM([_tool_call("tu_n", "noop")]))
        _register(runtime, "noop", lambda **_: "ok")
        session = runtime.new_session("f")
        session.set_run_context(_ctx())
        await session.handle_user_message("hello")
        text = _only_log(home).read_text(encoding="utf-8")
        assert '"tool.complete"' in text
        assert SYSTEM_MARKER not in text


# ---------------------------------------------------------------------------
# AC 8 — Runtime owns the sweep and releases locks on shutdown
# ---------------------------------------------------------------------------

class TestRuntimeRetention:
    async def test_initialize_sweeps_and_aclose_releases_session_logs(self, home):
        logs = home / "logs" / "sessions"
        logs.mkdir(parents=True)
        stale, fresh = logs / "20200101T000000Z-old-0001.jsonl", logs / "20200101T000000Z-new-0002.jsonl"
        for p, hours in ((stale, 25), (fresh, 1)):
            p.write_text("{}\n")
            t = time.time() - hours * 3600
            os.utime(p, (t, t))

        runtime = _runtime(ScriptedLLM())
        session = runtime.new_session("f")
        session.set_run_context(_ctx())
        new = [p for p in _log_files(home) if p not in (stale, fresh)]
        assert len(new) == 1, new
        live = new[0]
        await runtime.initialize()
        try:
            assert not stale.exists()
            assert fresh.exists()
            t = time.time() - 48 * 3600
            os.utime(live, (t, t))
            assert str(live) not in _cleanup_in_other_process(home)
        finally:
            await runtime.aclose()
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        assert pending == []
        t = time.time() - 48 * 3600
        os.utime(live, (t, t))
        assert str(live) in _cleanup_in_other_process(home)


# ---------------------------------------------------------------------------
# AC 10 — log failures never change the agent result or exit
# ---------------------------------------------------------------------------

class TestLogFailuresAreHarmless:
    async def test_append_failure_mid_session_keeps_the_result(self, home, caplog):
        runtime = _runtime(ScriptedLLM([_tool_call("tu_n", "noop")]))
        _register(runtime, "noop", lambda **_: "ok")
        session = runtime.new_session("f")
        session.set_run_context(_ctx())
        path = _only_log(home)
        _break_writes(path)
        with caplog.at_level(logging.WARNING):
            result = await session.handle_user_message("hello")
            await runtime.aclose()
        assert result.text == "done"
        assert result.tool_calls_executed == 1
        assert _warned_about(caplog, str(path))

    async def test_unusable_log_directory_keeps_single_shot_output(
        self, tmp_path, monkeypatch, capsys, caplog
    ):
        from sr2_spectre.interfaces.single_shot import SingleShotInterface

        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        monkeypatch.setenv("SR2_HOME", str(blocker))
        monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)

        agent = _agent(ScriptedLLM([StreamEvent(type="text", text="the answer")]))
        iface = SingleShotInterface(prompt="question")
        with caplog.at_level(logging.WARNING):
            await agent.initialize()
            try:
                await iface.start(agent)
                await iface.run(agent)
                await iface.stop()
            finally:
                await agent.aclose()
        assert capsys.readouterr().out == "the answer\n"
        assert _warned_about(caplog, str(blocker))
