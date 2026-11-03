"""Handoff tool: isolated target frame, forwarded events, and envelopes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sr2.protocols.llm import StreamEvent
from spctr.agent import Agent as RealAgent
from spctr.config import SpctrConfig
from spctr.events import (
    AgentDone,
    AgentHandoffEnd,
    AgentHandoffStart,
    AgentTextDelta,
    AgentThinkingDelta,
    AgentToolResult,
    AgentToolStart,
)
from spctr.runtime import Runtime
from spctr.tools.builtins.handoff import HandoffTool, bind_caller
from tests.conftest import _surviving_pids


PIPELINE = {
    "layers": [
        {"name": "system", "target": "system", "resolvers": []},
        {"name": "tools", "target": "tools", "resolvers": [],
         "tool_providers": [{"type": "spctr_tools"}]},
        {"name": "conversation", "target": "messages",
         "resolvers": [{"type": "session"}, {"type": "input"}]},
    ]
}


class ScriptedLLM:
    model = "test-model"

    def __init__(self, *rounds):
        self.rounds = list(rounds)

    async def complete(self, request):
        raise NotImplementedError

    async def stream(self, request):
        for event in self.rounds.pop(0):
            yield event
        yield StreamEvent(type="end", meta={"finish_reason": "stop"})


class FakeTarget:
    events = [AgentTextDelta(text="hi\nVERDICT: DONE"), AgentDone()]
    error: Exception | None = None
    instances: list["FakeTarget"] = []

    def __init__(self, config, session_id):
        self.config = config
        self.session_id = session_id
        self._session = SimpleNamespace(is_handoff_target=False)
        self.registry = SimpleNamespace(
            list_names=lambda: [tool.name for tool in config.agent.tools]
        )
        self.instances.append(self)

    async def initialize(self):
        pass

    def set_run_context(self, context):
        self.context = context

    async def stream_message(self, prompt):
        for event in self.events:
            yield event
        if self.error:
            raise self.error

    async def aclose(self):
        pass


class Caller:
    def __init__(self, *, target=False, active=None):
        self.is_handoff_target = target
        self.active_handoff = active
        self.events = []
        self.logs = []

    def emit_handoff_event(self, event):
        self.events.append(event)

    def _log_event(self, event, data):
        self.logs.append((event, data))


class PrefixAgent:
    """Real target Agent with one already-produced text event for timeout proof."""

    def __init__(self, config, session_id):
        self.inner = RealAgent(config, session_id)
        self._session = self.inner._session

    async def initialize(self):
        await self.inner.initialize()

    def set_run_context(self, context):
        self.inner.set_run_context(context)

    async def stream_message(self, prompt):
        yield AgentTextDelta(text="started")
        async for event in self.inner.stream_message(prompt):
            yield event

    async def aclose(self):
        await self.inner.aclose()


def _model_config(name="agent", tools=None) -> SpctrConfig:
    return SpctrConfig.model_validate({
        "agent": {
            "name": name,
            "default_skills": False,
            "tools": tools or [],
            "skills": [],
            "skills_dirs": [],
            "mcp_servers": [],
        },
        "models": {"default": {"model": "test-model", "base_url": "http://test"}},
        "active_model": "default",
        "pipeline": PIPELINE,
        "provenance_store_path": "",
        "memory_store_dsn": "",
    })


def _write_roles(root: Path, *, target_tools="", invalid=False) -> None:
    agents = root / "agents"
    agents.mkdir(parents=True)
    (root / "config.yaml").write_text(
        """agent:
  tool_result_max_bytes: 4321
  tools:
    - name: terminal
      class_path: spctr.tools.builtins.terminal.TerminalTool
models:
  default:
    model: test-model
    base_url: http://test
active_model: default
provenance_store_path: ''
memory_store_dsn: ''
pipeline:
  layers:
    - {name: system, target: system, resolvers: []}
    - name: tools
      target: tools
      resolvers: []
      tool_providers: [{type: spctr_tools}]
    - name: conversation
      target: messages
      resolvers: [{type: session}, {type: input}]
""",
        encoding="utf-8",
    )
    role = "agent: []\n" if invalid else f"""agent:
  name: role-b
  default_skills: false
  tools:
{target_tools or '    - name: file_read\n      class_path: spctr.tools.builtins.file_read.FileReadTool'}
  skills: []
  skills_dirs: []
  mcp_servers: []
"""
    (agents / "role-b.yaml").write_text(role, encoding="utf-8")


def _payload(result: AgentToolResult) -> dict:
    return json.loads(result.content)


async def _call(tool: HandoffTool, caller: Caller, **kwargs) -> dict:
    with bind_caller(caller):
        return json.loads(await tool(**kwargs))


async def test_caller_tool_loop_forwards_target_events_and_returns_envelope(
    tmp_path, monkeypatch
):
    roles = tmp_path / "roles"
    _write_roles(roles)
    monkeypatch.chdir(tmp_path)
    FakeTarget.instances.clear()
    FakeTarget.events = [
        AgentThinkingDelta(text="working"),
        AgentTextDelta(text=f"hi\nVERDICT: DONE\nWROTE {tmp_path / 'missing.md'}"),
        AgentToolStart(tool_id="t", name="file_read", input={"path": "x"}),
        AgentToolResult(tool_id="t", name="file_read", content="x"),
        AgentDone(),
    ]
    FakeTarget.error = None
    caller_llm = ScriptedLLM(
        [StreamEvent(type="tool_use", tool_use_id="h1", tool_name="handoff",
                     tool_input={"agent": "role-b", "prompt": "smoke", "frame_id": "x"})],
        [StreamEvent(type="text", text="caller continues")],
    )
    config = _model_config("caller", [{
        "name": "handoff",
        "class_path": "spctr.tools.builtins.handoff.HandoffTool",
        "config": {"agents_root": str(roles)},
    }])

    with patch("spctr.live_llm.LiteLLMCallable", return_value=caller_llm), patch(
        "spctr.tools.builtins.handoff.Agent", FakeTarget
    ):
        runtime = Runtime(config)
        session = runtime.new_session("caller-frame")
        events = [event async for event in session.stream_message("delegate")]

    start = next(i for i, event in enumerate(events) if isinstance(event, AgentHandoffStart))
    end = next(i for i, event in enumerate(events) if isinstance(event, AgentHandoffEnd))
    forwarded = events[start + 1:end]
    assert [type(event) for event in forwarded] == [
        AgentThinkingDelta, AgentTextDelta, AgentToolStart, AgentToolResult
    ]
    assert all(event.frame_id == "x" for event in forwarded)
    assert events[start] == AgentHandoffStart(frame_id="x", agent="role-b")
    assert events[end].status == "done" and events[end].duration_s > 0
    assert next(event for event in events[end + 1:] if isinstance(event, AgentTextDelta)) == (
        AgentTextDelta(text="caller continues")
    )

    result = next(event for event in events if isinstance(event, AgentToolResult)
                  and event.name == "handoff")
    envelope = _payload(result)
    assert set(envelope) == {
        "agent", "frame_id", "status", "exit_code", "duration_s", "message",
        "wrote", "wrote_missing",
    }
    assert (envelope["agent"], envelope["frame_id"], envelope["status"],
            envelope["exit_code"]) == ("role-b", "x", "done", 0)
    assert envelope["wrote"] == [str(tmp_path / "missing.md")]
    assert envelope["wrote_missing"] == envelope["wrote"]
    assert FakeTarget.instances[0].session_id == "x"
    assert FakeTarget.instances[0].registry.list_names() == ["terminal", "file_read"]
    assert FakeTarget.instances[0].config.agent.tool_result_max_bytes == 4321
    assert "handoff" not in FakeTarget.instances[0].registry.list_names()
    assert all("hi" not in repr(message) for message in session.history)


async def test_verdict_classification_and_configurable_messages(tmp_path):
    roles = tmp_path / "roles"
    _write_roles(roles)
    caller = Caller(target=True)
    tool = HandoffTool(
        str(roles), description="delegate", agent_description="pick one",
        nested_error="no recursion",
    )
    assert tool.description == "delegate"
    assert tool.input_schema["properties"]["agent"]["description"] == "pick one"
    assert (await _call(tool, caller, agent="role-b", prompt="x"))["message"] == "no recursion"

    caller = Caller()
    FakeTarget.events = [AgentTextDelta(text="VERDICT: ESCALATION: tests contradict spec")]
    with patch("spctr.tools.builtins.handoff.Agent", FakeTarget):
        envelope = await _call(tool, caller, agent="role-b", prompt="x")
    assert (envelope["status"], envelope["exit_code"]) == ("escalation", 3)
    assert re.fullmatch(r"handoff-role-b-\d+", envelope["frame_id"])


async def test_target_exception_returns_error_and_caller_continues(tmp_path):
    roles = tmp_path / "roles"
    _write_roles(roles)
    FakeTarget.events = []
    FakeTarget.error = RuntimeError("boom")
    caller_llm = ScriptedLLM(
        [StreamEvent(type="tool_use", tool_use_id="h1", tool_name="handoff",
                     tool_input={"agent": "role-b", "prompt": "smoke"})],
        [StreamEvent(type="text", text="still running")],
    )
    config = _model_config("caller", [{
        "name": "handoff",
        "class_path": "spctr.tools.builtins.handoff.HandoffTool",
        "config": {"agents_root": str(roles)},
    }])
    with patch("spctr.live_llm.LiteLLMCallable", return_value=caller_llm), patch(
        "spctr.tools.builtins.handoff.Agent", FakeTarget
    ):
        session = Runtime(config).new_session("caller-frame")
        events = [event async for event in session.stream_message("delegate")]
    FakeTarget.error = None

    result = next(event for event in events if isinstance(event, AgentToolResult)
                  and event.name == "handoff")
    envelope = _payload(result)
    assert envelope["status"] == "error" and envelope["exit_code"] == 5
    assert "boom" in envelope["message"]
    assert any(isinstance(event, AgentTextDelta) and event.text == "still running"
               for event in events)
    assert isinstance(events[-1], AgentDone)
    assert session.active_handoff is None


async def test_missing_invalid_nested_concurrent_and_target_error(tmp_path):
    roles = tmp_path / "roles"
    _write_roles(roles)
    tool = HandoffTool(str(roles), run_error="target failed: {error}")

    FakeTarget.instances.clear()
    missing = await _call(tool, Caller(), agent="nope", prompt="x")
    assert missing["status"] == "error" and str(roles / "agents/nope.yaml") in missing["message"]
    assert FakeTarget.instances == []

    (roles / "agents/role-b.yaml").write_text("agent: []\n", encoding="utf-8")
    invalid = await _call(tool, Caller(), agent="role-b", prompt="x")
    assert invalid["status"] == "error" and "role-b.yaml" in invalid["message"]
    assert FakeTarget.instances == []

    assert (await _call(tool, Caller(target=True), agent="role-b", prompt="x"))[
        "message"] == "nested handoff refused"
    assert (await _call(tool, Caller(active=object()), agent="role-b", prompt="x"))[
        "message"] == "handoff already active"

    _write_roles(tmp_path / "other")
    error_tool = HandoffTool(str(tmp_path / "other"), run_error="target failed: {error}")
    FakeTarget.events = []
    FakeTarget.error = RuntimeError("boom")
    with patch("spctr.tools.builtins.handoff.Agent", FakeTarget):
        failed = await _call(error_tool, Caller(), agent="role-b", prompt="x")
    FakeTarget.error = None
    assert failed["status"] == "error" and failed["exit_code"] == 5
    assert failed["message"] == "target failed: boom"


@pytest.mark.parametrize("explicit_log_dir", [True, False])
async def test_target_log_uses_process_directory(
    tmp_path, monkeypatch, explicit_log_dir
):
    roles = tmp_path / "roles"
    _write_roles(roles)
    process_home = tmp_path / "process-home"
    process_home.mkdir()
    monkeypatch.setenv("SR2_HOME", str(process_home))
    if explicit_log_dir:
        log_dir = tmp_path / "explicit-logs"
        monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(log_dir))
    else:
        monkeypatch.delenv("SR2_SESSION_LOG_DIR", raising=False)
        log_dir = process_home / "logs/sessions"
    caller = Caller()
    llm = ScriptedLLM([StreamEvent(type="text", text="VERDICT: DONE")])

    with patch("spctr.live_llm.LiteLLMCallable", return_value=llm):
        envelope = await _call(
            HandoffTool(str(roles)), caller, agent="role-b", prompt="smoke",
            frame_id="target-frame",
        )

    assert envelope["status"] == "done"
    assert list(log_dir.glob("*target-frame*.jsonl"))
    assert not list((roles / "logs/sessions").glob("*.jsonl"))
    assert [name for name, _ in caller.logs] == ["handoff.start", "handoff.end"]
    assert caller.logs[0][1] == {"frame_id": "target-frame", "agent": "role-b"}
    assert caller.logs[1][1]["status"] == "done"


async def test_timeout_keeps_partial_text_and_kills_terminal_process(tmp_path, monkeypatch):
    roles = tmp_path / "roles"
    terminal = """    - name: terminal
      class_path: spctr.tools.builtins.terminal.TerminalTool
      config: {timeout: 30}
"""
    _write_roles(roles, target_tools=terminal)
    monkeypatch.setenv("SR2_HOME", str(tmp_path / "process-home"))
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", "off")
    pid_file = tmp_path / "child.pid"
    marker = f"handoff-timeout-{tmp_path.name}"
    llm = ScriptedLLM([
        StreamEvent(type="tool_use", tool_use_id="t1", tool_name="terminal",
                    tool_input={"command": f"echo $$ > {pid_file}; sleep 10 # {marker}"}),
    ])

    with patch("spctr.live_llm.LiteLLMCallable", return_value=llm), patch(
        "spctr.tools.builtins.handoff.Agent", PrefixAgent
    ):
        envelope = await _call(
            HandoffTool(str(roles)), Caller(), agent="role-b", prompt="run",
            timeout=0.2,
        )

    assert envelope["status"] == "timeout" and envelope["exit_code"] == 4
    assert envelope["message"] == "started"
    assert int(pid_file.read_text(encoding="utf-8")) > 0
    assert _surviving_pids(marker) == []
