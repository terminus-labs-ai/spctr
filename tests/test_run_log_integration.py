"""Model-call, retry, pipeline and compaction capture in the session log (spc-108).

Drives real Runtime/Session/SR2 turns. Fakes live only at the provider
boundary (the LiteLLM callable) and the tool boundary (registered tools).

Pinned names: ``model.start`` opens a model call; any other ``model.*`` event
up to the next ``model.start`` belongs to that call, and the last one is its
terminal record. Turn and tool names come from the existing session log.
SR2 retries and pipeline/compaction records are located by content, not name.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sr2.models import TokenUsage
from sr2.pipeline.tracing import CollectingTracer
from sr2.protocols.llm import StreamEvent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.runtime import Runtime

SYSTEM_MARKER = "SYSTEM-PROMPT-MARKER-91c2"
PROFILE = "deepthink"
CONFIG_MODEL = "cfg-model-a"
RESOLVED_MODEL = "openai/resolved-model-a"


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


def _config(*, active: str = PROFILE, system_resolvers=None, system_transformers=None):
    system_layer = {
        "name": "system", "target": "system",
        "resolvers": system_resolvers or [
            {"type": "static", "config": {"text": SYSTEM_MARKER}},
        ],
    }
    if system_transformers:
        system_layer["transformers"] = system_transformers
    return SpectreConfig(
        agent=AgentConfig(name="edi"),
        models={
            "default": ModelConfig(model="cfg-default", base_url="http://test:8000"),
            PROFILE: ModelConfig(model=CONFIG_MODEL, base_url="http://test:8001"),
            "altprofile": ModelConfig(model="cfg-model-b", base_url="http://test:8002"),
        },
        active_model=active,
        pipeline={"layers": [
            system_layer,
            {"name": "tools", "target": "tools", "resolvers": [],
             "tool_providers": [{"type": "spectre_tools"}]},
            {"name": "conversation", "target": "messages",
             "resolvers": [{"type": "session"}, {"type": "input"}]},
        ]},
        provenance_store_path="",
        memory_store_dsn="",
    )


COMPACTION = [{
    "type": "compaction",
    "subscriptions": [{"event": "tool_result", "phase": "completed"}],
    "max_executions": 5,
}]


class Call:
    """One scripted provider stream."""

    def __init__(self, *events, finish="stop", usage=(0, 0), error: Exception | None = None):
        self.events = list(events)
        self.finish = finish
        self.usage = usage
        self.error = error


class FakeProvider:
    """LLMCallable standing in for LiteLLMCallable; records what it was sent."""

    def __init__(self, *calls: Call, model: str = RESOLVED_MODEL, home: Path | None = None):
        self.model = model
        self._calls = list(calls)
        self._home = home
        self.requests = []
        self.log_seen_mid_stream: list[str] = []

    async def complete(self, request):
        raise NotImplementedError

    async def stream(self, request):
        self.requests.append(request)
        call = self._calls.pop(0) if self._calls else Call(StreamEvent(type="text", text="done"))
        for ev in call.events:
            yield ev
        if self._home is not None:
            self.log_seen_mid_stream.append(_read_all(self._home))
        if call.error is not None:
            raise call.error
        yield StreamEvent(type="usage", usage=TokenUsage(
            input_tokens=call.usage[0], output_tokens=call.usage[1]))
        yield StreamEvent(type="end", meta={"finish_reason": call.finish})


def _think(text):
    return StreamEvent(type="thinking", text=text)


def _text(text):
    return StreamEvent(type="text", text=text)


def _tool(tool_id, name, **args):
    return StreamEvent(type="tool_use", tool_use_id=tool_id, tool_name=name, tool_input=args)


def _runtime(provider, config=None) -> Runtime:
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=provider):
        return Runtime(config or _config())


def _session(runtime, tracer=None):
    session = runtime.new_session("frame-1", tracer=tracer)
    session.set_run_context(RunContext(interface="single_shot", mode=RunMode.HEADLESS, source=None))
    return session


def _register(runtime, name, fn):
    runtime.registry.register(name, f"{name} tool", {"type": "object"}, fn)


def _log_path(home: Path) -> Path:
    files = sorted((home / "logs" / "sessions").glob("*.jsonl"))
    assert len(files) == 1, files
    return files[0]


def _read_all(home: Path) -> str:
    return "".join(p.read_text(encoding="utf-8")
                   for p in (home / "logs" / "sessions").glob("*.jsonl"))


def _events(home: Path) -> list[dict]:
    events = [json.loads(line) for line in _log_path(home).read_text().splitlines()]
    seqs = [e["sequence"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    return events


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield None, v
            yield from _walk(v)


def _strings(event) -> list[str]:
    return [v for _, v in _walk(event["data"]) if isinstance(v, str)]


def _has_text(event, needle: str) -> bool:
    return any(needle in s for s in _strings(event))


def _numbers(event, key_part: str) -> list[float]:
    return [v for k, v in _walk(event["data"])
            if isinstance(k, str) and key_part in k.lower()
            and isinstance(v, (int, float)) and not isinstance(v, bool)]


def _ints(event) -> set[int]:
    return {v for _, v in _walk(event["data"]) if isinstance(v, int) and not isinstance(v, bool)}


def _model_calls(events) -> list[list[dict]]:
    """Group model.* events into calls, each opened by model.start."""
    calls: list[list[dict]] = []
    for e in events:
        if not e["event"].startswith("model.") or _has_text(e, "empty_response"):
            continue
        if e["event"] == "model.start":
            calls.append([e])
        else:
            assert calls, f"model event before any model.start: {e}"
            calls[-1].append(e)
    return calls


def _first(events, name, **match) -> dict:
    for e in events:
        if e["event"] == name and all(e["data"].get(k) == v for k, v in match.items()):
            return e
    raise AssertionError(f"no {name} {match}")


def _seq(e) -> int:
    return e["sequence"]


# ---------------------------------------------------------------------------
# AC 5 — every model call
# ---------------------------------------------------------------------------

class TestModelCall:
    async def test_start_records_profile_resolved_model_structure_and_estimate(self, home):
        provider = FakeProvider(Call(_text("hello"), usage=(10, 2)))
        runtime = _runtime(provider)
        _register(runtime, "alpha", lambda **kw: "")
        _register(runtime, "beta", lambda **kw: "")
        session = _session(runtime)
        await session.handle_user_message("hi there")
        (call,) = _model_calls(_events(home))
        start = call[0]
        assert _has_text(start, PROFILE)
        assert _has_text(start, RESOLVED_MODEL)
        request = provider.requests[0]
        assert len(request.messages) in set(_numbers(start, "message"))
        assert len(request.tools or []) >= 2
        assert len(request.tools) != len(request.messages)
        assert len(request.tools) in set(_numbers(start, "tool"))
        assert any(n > 0 for n in _numbers(start, "token"))
        log_text = _read_all(home)
        assert SYSTEM_MARKER not in log_text

    async def test_text_and_reasoning_progress_is_written_while_streaming(self, home):
        provider = FakeProvider(
            Call(_think("REASONING-chunk-alpha"), _text("TEXT-chunk-omega"), usage=(50, 7)),
            home=home,
        )
        session = _session(_runtime(provider))
        await session.handle_user_message("go")
        (mid_stream,) = provider.log_seen_mid_stream
        assert "REASONING-chunk-alpha" in mid_stream
        assert "TEXT-chunk-omega" in mid_stream
        (call,) = _model_calls(_events(home))
        live = [json.loads(line) for line in mid_stream.splitlines()]
        live_start = max(i for i, e in enumerate(live) if e["event"] == "model.start")
        live_progress = [e for e in live[live_start + 1:] if e["event"].startswith("model.")]
        assert any(n > 0 for e in live_progress for n in _numbers(e, "token"))
        progress = call[1:-1]
        assert any(_has_text(e, "REASONING-chunk-alpha") for e in progress)
        assert any(_has_text(e, "TEXT-chunk-omega") for e in progress)
        assert any(n > 0 for e in progress for n in _numbers(e, "token"))

    async def test_end_records_usage_finish_reason_and_duration(self, home):
        provider = FakeProvider(Call(_text("answer"), finish="length", usage=(4321, 987)))
        session = _session(_runtime(provider))
        await session.handle_user_message("go")
        (call,) = _model_calls(_events(home))
        end = call[-1]
        assert end["event"] != "model.start"
        assert {4321, 987} <= _ints(end)
        assert _has_text(end, "length")
        durations = _numbers(end, "duration")
        assert durations and all(d >= 0 for d in durations)

    async def test_every_iteration_is_a_separate_recorded_call(self, home):
        provider = FakeProvider(
            Call(_tool("t1", "echo", x=1), finish="tool_calls", usage=(100, 11)),
            Call(_text("final"), finish="stop", usage=(140, 22)),
        )
        runtime = _runtime(provider)
        _register(runtime, "echo", lambda **kw: "echoed")
        await _session(runtime).handle_user_message("go")
        calls = _model_calls(_events(home))
        assert len(calls) == 2
        assert _has_text(calls[0][-1], "tool_calls") and {100, 11} <= _ints(calls[0][-1])
        assert _has_text(calls[1][-1], "stop") and {140, 22} <= _ints(calls[1][-1])
        assert len(provider.requests[1].messages) in set(_numbers(calls[1][0], "message"))

    async def test_provider_error_is_recorded_on_the_call_and_still_raised(self, home):
        provider = FakeProvider(Call(_text("partial"), error=RuntimeError("provider exploded 503")))
        session = _session(_runtime(provider))
        with pytest.raises(RuntimeError, match="provider exploded 503"):
            await session.handle_user_message("go")
        events = _events(home)
        (call,) = _model_calls(events)
        end = call[-1]
        assert end["event"] != "model.start"
        assert _has_text(end, "provider exploded 503")
        assert _numbers(end, "duration")
        assert _seq(end) < _seq(_first(events, "turn.error"))

    async def test_records_follow_a_hot_retarget(self, home):
        runtime = _runtime(FakeProvider(Call(_text("one"))))
        session = _session(runtime)
        await session.handle_user_message("first")
        second = FakeProvider(Call(_text("two")), model="openai/resolved-model-b")
        with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=second):
            runtime.apply_config(_config(active="altprofile"))
        await session.handle_user_message("second")
        first_call, second_call = _model_calls(_events(home))
        assert _has_text(first_call[0], RESOLVED_MODEL)
        assert _has_text(second_call[0], "openai/resolved-model-b")
        assert _has_text(second_call[0], "altprofile")
        assert len(second.requests) == 1


# ---------------------------------------------------------------------------
# AC 6 — retries, pipeline firings/errors, compaction, placed in order
# ---------------------------------------------------------------------------

class TestOrderedActivity:
    async def test_empty_response_retry_sits_between_the_two_calls(self, home):
        provider = FakeProvider(
            Call(_think("only thinking"), finish="stop", usage=(30, 3)),
            Call(_text("recovered"), finish="stop", usage=(30, 4)),
        )
        session = _session(_runtime(provider))
        result = await session.handle_user_message("go")
        assert result.text == "recovered"
        events = _events(home)
        retries = [e for e in events if _has_text(e, "empty_response")]
        assert len(retries) == 1
        first, second = _model_calls(events)
        assert _seq(first[-1]) < _seq(retries[0]) < _seq(second[0])
        assert _seq(_first(events, "turn.start")) < _seq(retries[0]) < _seq(_first(events, "turn.complete"))

    async def test_pipeline_firings_are_logged_before_the_model_call(self, home):
        session = _session(_runtime(FakeProvider(Call(_text("ok")))))
        await session.handle_user_message("go")
        events = _events(home)
        (call,) = _model_calls(events)
        turn_start = _seq(_first(events, "turn.start"))
        firings = [e for e in events
                   if turn_start < _seq(e) < _seq(call[0]) and "static" in _strings(e)]
        assert firings, "static resolver firing not logged before the model call"
        assert any("system" in _strings(e) for e in firings)

    async def test_session_tracer_composes_with_an_explicit_tracer(self, home):
        tracer = CollectingTracer()
        session = _session(_runtime(FakeProvider(Call(_text("ok")))), tracer=tracer)
        await session.handle_user_message("go")
        assert any(r.component_name == "static" for r in tracer.get_trace())
        assert tracer.compiled_request is not None
        assert any("static" in _strings(e) for e in _events(home))

    async def test_failed_pipeline_component_is_logged_before_turn_error(self, home):
        config = _config(system_resolvers=[
            {"type": "static", "config": {"text": SYSTEM_MARKER}},
            {"type": "markdown_file",
             "config": {"path": "/nonexistent-spc108/{area}.md", "on_missing": "error"}},
        ])
        provider = FakeProvider(Call(_text("never")))
        session = _session(_runtime(provider, config))
        with pytest.raises(FileNotFoundError):
            await session.handle_user_message("go")
        events = _events(home)
        failures = [e for e in events
                    if not e["event"].startswith("turn.")
                    and "markdown_file" in _strings(e) and _has_text(e, "supplies no area")]
        assert failures, "failed pipeline firing not logged"
        assert _seq(failures[0]) < _seq(_first(events, "turn.error"))
        assert provider.requests == []

    async def test_compaction_firing_is_placed_between_tool_and_next_model_call(self, home):
        config = _config(system_transformers=COMPACTION)
        provider = FakeProvider(
            Call(_tool("t1", "echo", x=1), finish="tool_calls"),
            Call(_text("final")),
        )
        runtime = _runtime(provider, config)
        _register(runtime, "echo", lambda **kw: "echoed")
        await _session(runtime).handle_user_message("go")
        events = _events(home)
        compactions = [e for e in events if "compaction" in _strings(e)]
        assert compactions, "compaction activity not logged"
        tool_done = _first(events, "tool.complete", tool_use_id="t1")
        _, second = _model_calls(events)
        assert any(_seq(tool_done) < _seq(c) < _seq(second[0]) for c in compactions)


# ---------------------------------------------------------------------------
# AC 9 — reconstruct a multi-iteration tool run from the log alone
# ---------------------------------------------------------------------------

class TestReconstruction:
    async def test_log_alone_reconstructs_a_multi_iteration_tool_run(self, home):
        provider = FakeProvider(
            Call(_think("plan: read then grep"), _text("Looking."),
                 _tool("a1", "read_file", path="notes.md"),
                 _tool("a2", "broken", api_key="sk-live-123"),
                 finish="tool_calls", usage=(200, 30)),
            Call(_tool("b1", "grep", pattern="TODO"), finish="tool_calls", usage=(260, 12)),
            Call(_text("Found 2 TODOs."), finish="stop", usage=(300, 9)),
        )
        runtime = _runtime(provider)
        _register(runtime, "read_file", lambda **kw: "notes body")
        _register(runtime, "grep", lambda **kw: "TODO one\nTODO two")

        def _broken(**kw):
            raise ValueError("disk on fire")
        _register(runtime, "broken", _broken)

        await _session(runtime).handle_user_message("count the TODOs")
        events = _events(home)

        # Timeline from the log only.
        timeline = []
        for e in events:
            name = e["event"]
            if name in ("turn.start", "turn.complete"):
                timeline.append(name)
            elif name == "model.start":
                timeline.append("model")
            elif name == "tool.start":
                timeline.append(f"tool:{e['data']['name']}")
        assert timeline[0] == "turn.start" and timeline[-1] == "turn.complete"
        body = timeline[1:-1]
        assert body[0] == "model"
        assert sorted(body[1:3]) == ["tool:broken", "tool:read_file"]
        assert body[3:] == ["model", "tool:grep", "model"]

        calls = _model_calls(events)
        assert len(calls) == 3
        assert any(_has_text(e, "plan: read then grep") for e in calls[0])
        assert any(_has_text(e, "Found 2 TODOs.") for e in calls[2])
        for call, reason in zip(calls, ("tool_calls", "tool_calls", "stop")):
            assert _has_text(call[-1], reason)

        # Each tool ran after the call that requested it and before the next.
        for tool_id, call_index in (("a1", 0), ("a2", 0), ("b1", 1)):
            start = _first(events, "tool.start", tool_use_id=tool_id)
            done = _first(events, "tool.complete", tool_use_id=tool_id)
            assert _seq(calls[call_index][-1]) < _seq(start) < _seq(done)
            assert _seq(done) < _seq(calls[call_index + 1][0])

        assert _has_text(_first(events, "tool.complete", tool_use_id="b1"), "TODO two")
        broken_done = _first(events, "tool.complete", tool_use_id="a2")
        assert broken_done["data"]["is_error"] is True
        assert "sk-live-123" not in _read_all(home)
        assert _has_text(_first(events, "turn.start"), "count the TODOs")
