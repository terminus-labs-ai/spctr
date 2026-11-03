"""REPL rendering and Ctrl-C behavior for active handoffs."""

from __future__ import annotations

import asyncio
import os
import re
import signal
from types import SimpleNamespace

import pytest
from rich.console import Console

from spctr.config import SpctrConfig, load_config
from spctr.events import (
    AgentDone,
    AgentHandoffEnd,
    AgentHandoffStart,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
)
from spctr.interfaces.repl import REPLInterface
from spctr.session import HandoffHandle


class _Sink:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, text: str) -> int:
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        pass


def _config(**overrides: str) -> SpctrConfig:
    return SpctrConfig.model_validate({
        "agent": {"name": "caller"},
        "models": {"default": {"model": "test", "base_url": "http://test"}},
        "pipeline": {"layers": []},
        **overrides,
    })


def _interface() -> tuple[REPLInterface, _Sink]:
    sink = _Sink()
    return REPLInterface(Console(file=sink, force_terminal=True, width=100)), sink


def _output(sink: _Sink) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", "".join(sink.chunks))


class _EventAgent:
    def __init__(self, events, config: SpctrConfig | None = None) -> None:
        self.events = events
        self.config = config or _config()
        self.session_id = "caller-frame"
        self.history = []
        self._session = SimpleNamespace(active_handoff=None)

    async def stream_message(self, text):
        for event in self.events:
            yield event


@pytest.mark.asyncio
async def test_handoff_stream_renders_takeover_events_return_and_footer_in_order():
    agent = _EventAgent([
        AgentHandoffStart(frame_id="fid", agent="role-c"),
        AgentTextDelta(text="hi", frame_id="fid"),
        AgentToolStart(name="terminal", frame_id="fid"),
        AgentToolResult(name="terminal", frame_id="fid"),
        AgentHandoffEnd(
            frame_id="fid", agent="role-c", status="done", duration_s=1412.3
        ),
        AgentDone(tool_calls_executed=1),
    ])
    interface, sink = _interface()

    await interface._stream_turn(agent, "delegate")

    output = _output(sink)
    expected = [
        "→ role-c took over · fid",
        "hi",
        "⚙ terminal",
        "✓ terminal done",
        "← role-c · DONE · 23m 32s",
    ]
    positions = [output.index(value) for value in expected]
    assert positions == sorted(positions)
    assert re.search(
        r"— session caller-f · 0 msgs · 1 tools · ~0 tok est\. —", output
    )


@pytest.mark.asyncio
async def test_handoff_cancelled_return_uses_uppercase_status_and_duration():
    agent = _EventAgent([
        AgentHandoffStart(frame_id="fid", agent="role-c"),
        AgentHandoffEnd(
            frame_id="fid", agent="role-c", status="cancelled", duration_s=41
        ),
        AgentDone(),
    ])
    interface, sink = _interface()

    await interface._stream_turn(agent, "delegate")

    assert "← role-c · CANCELLED · 0m 41s" in _output(sink)


async def _cancel_active_handoff(config: SpctrConfig) -> tuple[str, int]:
    release = asyncio.Event()
    started = asyncio.Event()
    calls = 0
    session = SimpleNamespace(active_handoff=None)

    def cancel() -> None:
        nonlocal calls
        calls += 1
        session.active_handoff = None
        release.set()

    class Agent(_EventAgent):
        async def stream_message(self, text):
            started.set()
            await release.wait()
            yield AgentDone()

    agent = Agent([], config)
    agent._session = session
    session.active_handoff = HandoffHandle("fid", "role-c", 0.0, cancel)
    interface, sink = _interface()
    turn = asyncio.create_task(interface._run_turn(agent, "delegate"))
    await asyncio.wait_for(started.wait(), timeout=2)
    os.kill(os.getpid(), signal.SIGINT)
    await asyncio.wait_for(turn, timeout=2)
    return _output(sink), calls


@pytest.mark.asyncio
async def test_first_ctrl_c_cancels_only_active_handoff_and_turn_completes():
    output, calls = await _cancel_active_handoff(_config())

    assert calls == 1
    assert "⏹ Stopping role-c…" in output
    assert "Turn stopped." not in output
    assert "— session caller-f" in output


@pytest.mark.asyncio
async def test_yaml_overrides_all_handoff_templates_and_defaults_remain(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """agent: {name: caller}
models:
  default: {model: test, base_url: http://test}
pipeline: {layers: []}
repl_handoff_start: "START {agent} {frame_id}"
repl_handoff_end: "END {agent} {status} {duration}"
repl_handoff_stop: "STOP {agent}"
""",
        encoding="utf-8",
    )
    config = load_config(path)
    defaults = _config()
    assert defaults.repl_handoff_start == "→ {agent} took over · {frame_id}"
    assert defaults.repl_handoff_end == "← {agent} · {status} · {duration}"
    assert defaults.repl_handoff_stop == "⏹ Stopping {agent}…"

    agent = _EventAgent([
        AgentHandoffStart(frame_id="fid", agent="role-c"),
        AgentHandoffEnd(
            frame_id="fid", agent="role-c", status="done", duration_s=61
        ),
        AgentDone(),
    ], config)
    interface, sink = _interface()
    await interface._stream_turn(agent, "delegate")
    output = _output(sink)
    assert "START role-c fid" in output
    assert "END role-c DONE 1m 1s" in output

    stop_output, calls = await _cancel_active_handoff(config)
    assert calls == 1
    assert "STOP role-c" in stop_output
