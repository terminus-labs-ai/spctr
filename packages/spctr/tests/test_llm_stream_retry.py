"""Retry of a streamed model call that dies mid-stream (obsidian-d6bi).

The only mocked boundary is ``sr2_spectre.live_llm.LiteLLMCallable`` (plus
``asyncio.sleep`` in the backoff test). Retry sits per Session, over LiveLLM.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import litellm.exceptions as lx
import pytest
import yaml
from pydantic import ValidationError
from sr2.protocols.llm import CompletionRequest, Message, StreamEvent, TextBlock

from sr2_spectre.config import (
    AgentConfig, ModelConfig, SpectreConfig, load_config, load_resolved_config,
)
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.live_llm import LiveLLM, ModelCallGuardError, RetryingLLM
from sr2_spectre.runtime import Runtime

MODEL = "openai/retry-test-model"
REQUEST = CompletionRequest(messages=[Message(role="user", content=[TextBlock(text="same-prompt")])])


def _midstream(msg="engine closed the connection"):
    return lx.MidStreamFallbackError(msg, model=MODEL, llm_provider="openai")


def _apiconn():
    return lx.APIConnectionError("connection reset", llm_provider="openai", model=MODEL)


class _SubConn(lx.APIConnectionError):
    pass


def _text(t):
    return StreamEvent(type="text", text=t)


class FakeInner:
    """Stand-in for LiteLLMCallable. Each attempt is (events, exc_or_None)."""

    def __init__(self, *attempts, **kwargs) -> None:
        self.model = kwargs.get("model", MODEL)
        self.attempts = list(attempts)
        self.requests: list = []

    async def stream(self, request):
        self.requests.append(request)
        events, exc = self.attempts.pop(0)
        for ev in events:
            yield ev
        if exc is not None:
            raise exc

    async def complete(self, request):
        return object()


def _cfg(retries=2, backoff=0.0) -> ModelConfig:
    return ModelConfig(model=MODEL, stream_retries=retries, stream_retry_backoff_seconds=backoff)


def _retrying(cfg: ModelConfig, fake: FakeInner) -> RetryingLLM:
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=fake):
        live = LiveLLM(cfg)
    return RetryingLLM(live, lambda: live.model_config)


async def _collect(llm) -> list:
    return [e async for e in llm.stream(REQUEST)]


OK = ([_text("good"), StreamEvent(type="end")], None)
RETRYABLE = [_midstream, _apiconn, lambda: _SubConn("x", llm_provider="openai", model=MODEL)]


class TestRetry:
    @pytest.mark.parametrize("make", RETRYABLE)
    @pytest.mark.parametrize("n_events", [0, 1, 3])
    async def test_recovers_after_error_at_any_point(self, make, n_events):
        fake = FakeInner(([_text("p")] * n_events, make()), OK)
        events = await _collect(_retrying(_cfg(), fake))
        assert [e.text for e in events if e.type == "text"] == ["good"]
        assert len(fake.requests) == 2

    async def test_failed_attempt_events_never_yielded(self):
        partial = [_text("PARTIAL"), StreamEvent(type="thinking", text="PT"),
                   StreamEvent(type="tool_use", tool_use_id="t", tool_name="x")]
        fake = FakeInner((partial, _midstream()), OK)
        events = await _collect(_retrying(_cfg(), fake))
        assert events == OK[0]

    async def test_every_attempt_uses_same_request(self):
        fake = FakeInner(([], _midstream()), ([_text("a")], _apiconn()), OK)
        await _collect(_retrying(_cfg(), fake))
        assert len(fake.requests) == 3
        assert all(r == REQUEST for r in fake.requests)

    async def test_success_first_try_is_unchanged(self):
        events = [_text("a"), StreamEvent(type="thinking", text="b"), _text("c"),
                  StreamEvent(type="end")]
        fake = FakeInner((events, None))
        assert await _collect(_retrying(_cfg(), fake)) == events
        assert len(fake.requests) == 1


class TestBounded:
    @pytest.mark.parametrize("retries", [0, 1, 2, 4])
    async def test_attempts_are_retries_plus_one_and_last_error_propagates(self, retries):
        errors = [_midstream(f"m{i}") for i in range(retries)] + [_apiconn()]
        fake = FakeInner(*[([_text("p")], e) for e in errors], OK)
        with pytest.raises(lx.APIConnectionError) as info:
            await _collect(_retrying(_cfg(retries=retries), fake))
        assert info.value is errors[-1]
        assert len(fake.requests) == retries + 1
        assert len(fake.attempts) == 1  # the spare success was never reached

    async def test_exhausted_same_type_propagates(self):
        fake = FakeInner(*[([], _midstream("boom"))] * 3)
        with pytest.raises(lx.MidStreamFallbackError):
            await _collect(_retrying(_cfg(), fake))
        assert len(fake.requests) == 3

    async def test_config_change_applies_to_next_call(self):
        a = FakeInner(([], _midstream()), OK)
        b = FakeInner(([], _midstream()), OK)
        with patch("sr2_spectre.live_llm.LiteLLMCallable", side_effect=[a, b]):
            live = LiveLLM(_cfg(retries=0))
            llm = RetryingLLM(live, lambda: live.model_config)
            with pytest.raises(lx.MidStreamFallbackError):
                await _collect(llm)
            assert live.retarget(_cfg(retries=2))
        assert [e.text for e in await _collect(llm) if e.type == "text"] == ["good"]
        assert len(b.requests) == 2


class TestBackoff:
    async def _delays(self, backoff, n_fail):
        fake = FakeInner(*[([], _midstream())] * n_fail, OK)
        with patch("asyncio.sleep", new=AsyncMock()) as sleep:
            await _collect(_retrying(_cfg(retries=3, backoff=backoff), fake))
        return [c.args[0] for c in sleep.await_args_list]

    async def test_exponential_schedule(self):
        assert await self._delays(2.0, 3) == [2.0, 4.0, 8.0]

    async def test_custom_base(self):
        assert await self._delays(0.5, 2) == [0.5, 1.0]

    async def test_zero_backoff_does_not_wait(self):
        assert [d for d in await self._delays(0.0, 3) if d > 0] == []


class TestNotRetried:
    @pytest.mark.parametrize("make", [
        lambda: ModelCallGuardError("stall", 5.0, 5.1, MODEL),
        asyncio.CancelledError,
        lambda: ValueError("bad"),
        lambda: lx.BadRequestError("bad", model=MODEL, llm_provider="openai"),
        lambda: lx.AuthenticationError("no", model=MODEL, llm_provider="openai"),
    ])
    @pytest.mark.parametrize("n_events", [0, 2])
    async def test_propagates_after_one_attempt(self, make, n_events):
        exc = make()
        fake = FakeInner(([_text("p")] * n_events, exc), OK)
        with pytest.raises(type(exc)) as info:
            await _collect(_retrying(_cfg(), fake))
        assert info.value is exc
        assert len(fake.requests) == 1


_PIPELINE = {"layers": [{"name": "system", "target": "system",
                         "resolvers": [{"type": "static", "config": {"text": "hi"}}]}]}


class TestConfigFields:
    def test_defaults(self):
        cfg = ModelConfig(model=MODEL)
        assert cfg.stream_retries == 2
        assert cfg.stream_retry_backoff_seconds == 2.0

    def test_accepts_zero(self):
        cfg = _cfg(retries=0, backoff=0)
        assert (cfg.stream_retries, cfg.stream_retry_backoff_seconds) == (0, 0.0)

    @pytest.mark.parametrize("field,value", [
        ("stream_retries", -1), ("stream_retry_backoff_seconds", -0.5)])
    def test_rejects_negative(self, field, value):
        with pytest.raises(ValidationError):
            ModelConfig(model=MODEL, **{field: value})

    def test_load_from_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump({"agent": {"name": "t"}, "pipeline": _PIPELINE, "models": {
            "default": {"model": MODEL, "stream_retries": 5, "stream_retry_backoff_seconds": 0.25}}}))
        m = load_config(str(path)).models["default"]
        assert (m.stream_retries, m.stream_retry_backoff_seconds) == (5, 0.25)

    def test_merge_through_tiers(self, tmp_path):
        home, cwd = tmp_path / "home", tmp_path / "proj"
        home.mkdir(), cwd.mkdir()
        (home / "config.yaml").write_text(yaml.dump({"models": {"default": {
            "model": MODEL, "stream_retries": 4, "stream_retry_backoff_seconds": 9}}}))
        (cwd / ".spectre.yaml").write_text(yaml.dump(
            {"models": {"default": {"stream_retry_backoff_seconds": 1.5}}}))
        top = tmp_path / "agent.yaml"
        top.write_text(yaml.dump({"agent": {"name": "t"}, "pipeline": _PIPELINE,
                                  "models": {"default": {"stream_retries": 0}}}))
        merged = load_resolved_config(top, cwd=cwd, env={"SR2_HOME": str(home)})
        m = load_config(merged).models["default"]
        assert (m.stream_retries, m.stream_retry_backoff_seconds) == (0, 1.5)

    def test_not_forwarded_to_litellm(self):
        with patch("sr2_spectre.live_llm.LiteLLMCallable") as mock:
            LiveLLM(_cfg(retries=3, backoff=1.0))
        assert "stream_retries" not in mock.call_args.kwargs
        assert "stream_retry_backoff_seconds" not in mock.call_args.kwargs


# --- Session-level: real Runtime/Session turn ---------------------------------

def _config() -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="edi"),
        models={"default": _cfg(retries=2, backoff=0.0).model_copy(update={"base_url": "http://t:1"})},
        pipeline={"layers": [
            {"name": "system", "target": "system",
             "resolvers": [{"type": "static", "config": {"text": "sys"}}]},
            {"name": "tools", "target": "tools", "resolvers": [],
             "tool_providers": [{"type": "spectre_tools"}]},
            {"name": "conversation", "target": "messages",
             "resolvers": [{"type": "session"}, {"type": "input"}]},
        ]},
        provenance_store_path="", memory_store_dsn="",
    )


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "sr2home"
    h.mkdir()
    monkeypatch.setenv("SR2_HOME", str(h))
    monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
    monkeypatch.delenv("SR2_SESSION_LOG_DIR", raising=False)
    return h


def _session(fake: FakeInner, ran: list | None = None):
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=fake):
        runtime = Runtime(_config())
    if ran is not None:
        async def ghost(**_):
            ran.append("ghost")
            return "ghost-result"
        runtime.registry.register("ghost", "ghost tool", {"type": "object"}, ghost)
    session = runtime.new_session("f")
    session.set_run_context(RunContext(interface="single_shot", mode=RunMode.HEADLESS, source=None))
    return session


FAILED = [_text("PARTIAL-TEXT"), StreamEvent(type="thinking", text="PARTIAL-THINK"),
          StreamEvent(type="tool_use", tool_use_id="tu_g", tool_name="ghost", tool_input={})]


class TestSessionTurn:
    async def test_dropped_stream_then_success_completes_turn(self, home):
        fake = FakeInner((FAILED, _midstream()), ([_text("final answer")], None))
        result = await _session(fake).handle_user_message("go")
        assert result.text == "final answer"
        assert len(fake.requests) == 2 and fake.requests[0] == fake.requests[1]

    async def test_dropped_stream_then_success_with_logging_off(self, home, monkeypatch):
        monkeypatch.setenv("SR2_SESSION_LOG_DIR", "off")
        fake = FakeInner((FAILED, _midstream()), ([_text("final answer")], None))
        result = await _session(fake).handle_user_message("go")
        assert result.text == "final answer"
        assert len(fake.requests) == 2

    async def test_retry_is_per_call_not_per_turn(self, home):
        ran: list = []
        call_ghost = StreamEvent(type="tool_use", tool_use_id="tu_g", tool_name="ghost", tool_input={})
        fake = FakeInner(
            ([call_ghost], None),
            ([_text("PARTIAL")], _midstream()),
            ([_text("final answer")], None),
        )
        result = await _session(fake, ran).handle_user_message("go")
        assert ran == ["ghost"]
        assert len(fake.requests) == 3
        assert fake.requests[1] == fake.requests[2]
        assert fake.requests[2] != fake.requests[0]
        assert result.text == "final answer"

    async def test_no_partial_output_or_tool_execution(self, home):
        ran: list = []
        fake = FakeInner((FAILED, _midstream()), ([_text("final answer")], None))
        session = _session(fake, ran)
        result = await session.handle_user_message("go")
        assert ran == []
        assert result.text == "final answer"
        dump = result.text + repr(session.history)
        assert "PARTIAL-TEXT" not in dump and "PARTIAL-THINK" not in dump
        assert "ghost" not in dump

    async def test_exhausted_retries_fail_turn_as_before(self, home):
        fake = FakeInner(*[(FAILED, _midstream())] * 3)
        with pytest.raises(lx.MidStreamFallbackError):
            await _session(fake).handle_user_message("go")
        assert len(fake.requests) == 3

    async def test_each_attempt_is_logged_as_its_own_model_call(self, home):
        fake = FakeInner((FAILED, _midstream()), ([_text("final answer")], None))
        await _session(fake).handle_user_message("go")
        (path,) = (home / "logs" / "sessions").glob("*.jsonl")
        events = [json.loads(line) for line in path.read_text().splitlines()]
        calls = [e for e in events if e["event"] in ("model.start", "model.end", "model.error")]
        assert [e["event"] for e in calls] == [
            "model.start", "model.error", "model.start", "model.end"]
        assert calls[1]["data"]["error_type"] == "MidStreamFallbackError"
