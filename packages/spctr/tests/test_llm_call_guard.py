"""Runtime guard on Spectre's single LLM call path (obsidian-5a9o).

Covers the default output-token cap injected into ``LiteLLMCallable``, the
``call_timeout_seconds`` / ``stall_timeout_seconds`` model fields, and the
duration/stall aborts raised as ``ModelCallGuardError`` by ``LiveLLM``.
The only mocked boundary is ``sr2_spectre.live_llm.LiteLLMCallable``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError
from sr2.protocols.llm import CompletionRequest, StreamEvent

from sr2_spectre.config import ModelConfig, load_config, load_resolved_config
from sr2_spectre.live_llm import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    LiveLLM,
    LoggedLLM,
    ModelCallGuardError,
)

MODEL = "openai/guard-test-model"
REQUEST = CompletionRequest(messages=[])
SLOW = 10.0  # how long a "hung" inner call would take on its own
MARGIN = 1.5  # generous abort tolerance past the limit


class FakeInner:
    """Stand-in for LiteLLMCallable: sleeps ``gaps[i]`` before event i."""

    def __init__(self, gaps=(), complete_delay=0.0, **kwargs) -> None:
        self.kwargs = kwargs
        self.model = kwargs.get("model", MODEL)
        self.gaps = list(gaps)
        self.complete_delay = complete_delay
        self.events = [StreamEvent(type="text", text=f"e{i}") for i in range(len(self.gaps))]
        self.response = object()
        self.closed = False
        self.finished = False

    async def stream(self, request):
        try:
            for gap, event in zip(self.gaps, self.events):
                await asyncio.sleep(gap)
                yield event
            self.finished = True
        finally:
            self.closed = True

    async def complete(self, request):
        await asyncio.sleep(self.complete_delay)
        return self.response


def _cfg(call=None, stall=None, **extra) -> ModelConfig:
    return ModelConfig(model=MODEL, call_timeout_seconds=call, stall_timeout_seconds=stall, **extra)


def _live(cfg: ModelConfig, *fakes: FakeInner) -> LiveLLM:
    """LiveLLM whose successive LiteLLMCallable constructions return *fakes*."""
    with patch("sr2_spectre.live_llm.LiteLLMCallable", side_effect=list(fakes)):
        return LiveLLM(cfg)


async def _drain(agen, sink: list):
    async for event in agen:
        sink.append(event)


async def _expect_guard(coro, kind: str, limit: float) -> ModelCallGuardError:
    started = time.monotonic()
    with pytest.raises(ModelCallGuardError) as info:
        await coro
    took = time.monotonic() - started
    assert info.value.kind == kind
    assert info.value.limit_seconds == pytest.approx(limit)
    assert took < limit + MARGIN  # aborted near the limit, not when inner finished
    return info.value


# --- AC1: default output-token cap --------------------------------------------

def _built_kwargs(cfg: ModelConfig) -> dict:
    with patch("sr2_spectre.live_llm.LiteLLMCallable") as mock:
        LiveLLM(cfg)
    return mock.call_args.kwargs


class TestDefaultMaxTokens:
    def test_constant_value(self):
        assert DEFAULT_MAX_OUTPUT_TOKENS == 32768

    def test_injected_when_params_lack_both_keys(self):
        cfg = ModelConfig(model=MODEL, base_url="http://x", api_key="k", params={"temperature": 0.2})
        assert _built_kwargs(cfg) == {
            "model": MODEL, "base_url": "http://x", "api_key": "k",
            "temperature": 0.2, "max_tokens": 32768,
        }

    @pytest.mark.parametrize("params", [
        {"max_tokens": 4096},
        {"max_tokens": None},
        {"max_completion_tokens": 1000},
        {"max_completion_tokens": None},
    ])
    def test_explicit_value_forwarded_unchanged(self, params):
        kwargs = _built_kwargs(ModelConfig(model=MODEL, params=params))
        assert kwargs == {"model": MODEL, "base_url": None, **params}

    def test_retarget_injects_default(self):
        with patch("sr2_spectre.live_llm.LiteLLMCallable") as mock:
            llm = LiveLLM(ModelConfig(model=MODEL, params={"max_tokens": 10}))
            assert llm.retarget(ModelConfig(model="openai/other"))
        assert mock.call_args.kwargs == {"model": "openai/other", "base_url": None, "max_tokens": 32768}


# --- AC2: ModelConfig guard fields --------------------------------------------

_PIPELINE = {"layers": [{"name": "system", "target": "system",
                         "resolvers": [{"type": "static", "config": {"text": "hi"}}]}]}


class TestGuardFields:
    def test_defaults(self):
        cfg = ModelConfig(model=MODEL)
        assert cfg.call_timeout_seconds == 900.0
        assert cfg.stall_timeout_seconds == 600.0

    def test_accepts_positive_and_none(self):
        cfg = _cfg(call=1.5, stall=None)
        assert cfg.call_timeout_seconds == 1.5
        assert cfg.stall_timeout_seconds is None
        assert ModelConfig(model=MODEL, call_timeout_seconds=None).call_timeout_seconds is None
        assert ModelConfig(model=MODEL, stall_timeout_seconds=3).stall_timeout_seconds == 3.0

    @pytest.mark.parametrize("field", ["call_timeout_seconds", "stall_timeout_seconds"])
    @pytest.mark.parametrize("value", [0, 0.0, -1, -0.5])
    def test_rejects_non_positive(self, field, value):
        with pytest.raises(ValidationError):
            ModelConfig(model=MODEL, **{field: value})

    def test_load_from_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump({
            "agent": {"name": "t"}, "pipeline": _PIPELINE,
            "models": {"default": {"model": MODEL, "call_timeout_seconds": 120,
                                   "stall_timeout_seconds": None}},
        }))
        m = load_config(str(path)).models["default"]
        assert m.call_timeout_seconds == 120.0
        assert m.stall_timeout_seconds is None

    def test_merge_through_tiers(self, tmp_path):
        home, cwd = tmp_path / "home", tmp_path / "proj"
        home.mkdir(), cwd.mkdir()
        (home / "config.yaml").write_text(yaml.dump(
            {"models": {"default": {"model": MODEL, "call_timeout_seconds": 300,
                                    "stall_timeout_seconds": 200}}}))
        (cwd / ".spectre.yaml").write_text(yaml.dump(
            {"models": {"default": {"stall_timeout_seconds": 50}}}))
        top = tmp_path / "agent.yaml"
        top.write_text(yaml.dump({"agent": {"name": "t"}, "pipeline": _PIPELINE,
                                  "models": {"default": {"call_timeout_seconds": None}}}))
        merged = load_resolved_config(top, cwd=cwd, env={"SR2_HOME": str(home)})
        m = load_config(merged).models["default"]
        assert m.call_timeout_seconds is None
        assert m.stall_timeout_seconds == 50.0

    def test_not_forwarded_to_litellm(self):
        kwargs = _built_kwargs(_cfg(call=5.0, stall=4.0))
        assert "call_timeout_seconds" not in kwargs
        assert "stall_timeout_seconds" not in kwargs


# --- AC3-AC5: stream guards ----------------------------------------------------

class TestStreamGuards:
    async def test_duration_aborts_while_events_flow(self):
        fake = FakeInner(gaps=[0.05] * 200)
        got: list = []
        await _expect_guard(_drain(_live(_cfg(call=0.3, stall=5.0), fake).stream(REQUEST), got),
                            "duration", 0.3)
        assert got and got == fake.events[: len(got)]
        assert fake.closed and not fake.finished

    async def test_duration_when_silent_and_budget_smaller_than_stall(self):
        fake = FakeInner(gaps=[SLOW])
        await _expect_guard(_drain(_live(_cfg(call=0.2, stall=5.0), fake).stream(REQUEST), []),
                            "duration", 0.2)
        assert fake.closed and not fake.finished

    async def test_stall_before_first_event(self):
        fake = FakeInner(gaps=[SLOW, 0])
        await _expect_guard(_drain(_live(_cfg(call=None, stall=0.2), fake).stream(REQUEST), []),
                            "stall", 0.2)
        assert fake.closed and not fake.finished

    async def test_stall_after_events_delivers_prior_events(self):
        fake = FakeInner(gaps=[0, 0, SLOW])
        got: list = []
        await _expect_guard(_drain(_live(_cfg(call=5.0, stall=0.2), fake).stream(REQUEST), got),
                            "stall", 0.2)
        assert got == fake.events[:2]
        assert fake.closed and not fake.finished

    async def test_stall_window_resets_on_each_event(self):
        fake = FakeInner(gaps=[0.1] * 8)  # 0.8 s total, never 0.4 s silent
        got: list = []
        await _drain(_live(_cfg(call=None, stall=0.4), fake).stream(REQUEST), got)
        assert got == fake.events


# --- AC6: complete guard -------------------------------------------------------

class TestCompleteGuard:
    async def test_duration_aborts_complete(self):
        fake = FakeInner(complete_delay=SLOW)
        await _expect_guard(_live(_cfg(call=0.2, stall=None), fake).complete(REQUEST), "duration", 0.2)

    async def test_stall_does_not_apply_to_complete(self):
        fake = FakeInner(complete_delay=0.3)
        assert await _live(_cfg(call=None, stall=0.05), fake).complete(REQUEST) is fake.response


# --- AC7: disabled guards / normal calls unchanged -----------------------------

class TestNoBehaviourChange:
    @pytest.mark.parametrize("call,stall", [(None, None), (None, 5.0), (5.0, None), (900.0, 600.0)])
    async def test_stream_passthrough(self, call, stall):
        fake = FakeInner(gaps=[0, 0.05, 0])
        got: list = []
        await _drain(_live(_cfg(call=call, stall=stall), fake).stream(REQUEST), got)
        assert len(got) == 3 and all(a is b for a, b in zip(got, fake.events))
        assert fake.finished

    async def test_complete_passthrough(self):
        fake = FakeInner(complete_delay=0.05)
        assert await _live(_cfg(call=5.0, stall=0.01), fake).complete(REQUEST) is fake.response


# --- AC8: error surface ----------------------------------------------------------

class _RecordingLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict]] = []

    def append(self, event, data=None):
        self.entries.append((event, dict(data or {})))


class TestErrorSurface:
    async def test_raised_error_fields_and_message(self):
        fake = FakeInner(gaps=[SLOW])
        exc = await _expect_guard(_drain(_live(_cfg(call=None, stall=0.2), fake).stream(REQUEST), []),
                                  "stall", 0.2)
        assert isinstance(exc, RuntimeError)
        assert isinstance(exc.elapsed_seconds, float) and exc.elapsed_seconds >= 0.15
        assert isinstance(exc.limit_seconds, float)
        assert "stall" in str(exc) and "0.2" in str(exc)

    @pytest.mark.parametrize("kind,call,stall,streamed", [
        ("duration", 0.2, None, True),
        ("stall", None, 0.2, True),
        ("duration", 0.2, None, False),
    ])
    async def test_one_warning_logged(self, caplog, kind, call, stall, streamed):
        llm = _live(_cfg(call=call, stall=stall), FakeInner(gaps=[SLOW], complete_delay=SLOW))
        caplog.set_level(logging.WARNING, logger="sr2_spectre.live_llm")
        with pytest.raises(ModelCallGuardError):
            await (_drain(llm.stream(REQUEST), []) if streamed else llm.complete(REQUEST))
        warnings = [r for r in caplog.records
                    if r.name == "sr2_spectre.live_llm" and r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert kind in msg and "0.2" in msg and "guard-test-model" in msg

    async def test_logged_llm_stream_terminal_entry_is_model_error(self):
        log = _RecordingLog()
        logged = LoggedLLM(_live(_cfg(call=None, stall=0.2), FakeInner(gaps=[0, SLOW])),
                           lambda: log, lambda: "p")
        with pytest.raises(ModelCallGuardError):
            await _drain(logged.stream(REQUEST), [])
        event, data = log.entries[-1]
        assert event == "model.error" and data["error_type"] == "ModelCallGuardError"

    async def test_logged_llm_complete_terminal_entry_is_model_error(self):
        log = _RecordingLog()
        logged = LoggedLLM(_live(_cfg(call=0.2), FakeInner(complete_delay=SLOW)),
                           lambda: log, lambda: "p")
        with pytest.raises(ModelCallGuardError):
            await logged.complete(REQUEST)
        event, data = log.entries[-1]
        assert event == "model.error" and data["error_type"] == "ModelCallGuardError"


# --- AC9: live reload ------------------------------------------------------------

@pytest.fixture
def fakes(monkeypatch) -> list:
    """Queue of fakes handed out by every LiteLLMCallable build, retargets included."""
    queue: list = []
    monkeypatch.setattr("sr2_spectre.live_llm.LiteLLMCallable", lambda **kw: queue.pop(0))
    return queue


class TestLiveReload:
    async def test_next_call_uses_relaxed_limits(self, fakes):
        relaxed = FakeInner(gaps=[0.5, 0])
        fakes += [FakeInner(), relaxed]
        llm = LiveLLM(_cfg(call=None, stall=0.2))
        assert llm.retarget(_cfg(call=None, stall=None))
        got: list = []
        await _drain(llm.stream(REQUEST), got)
        assert got == relaxed.events

    async def test_next_call_uses_stricter_limits(self, fakes):
        fakes += [FakeInner(), FakeInner(gaps=[SLOW])]
        llm = LiveLLM(_cfg(call=None, stall=None))
        assert llm.retarget(_cfg(call=None, stall=0.2))
        await _expect_guard(_drain(llm.stream(REQUEST), []), "stall", 0.2)

    async def test_next_complete_uses_new_duration(self, fakes):
        fakes += [FakeInner(), FakeInner(complete_delay=SLOW)]
        llm = LiveLLM(_cfg(call=None))
        assert llm.retarget(_cfg(call=0.2))
        await _expect_guard(llm.complete(REQUEST), "duration", 0.2)

    async def test_in_flight_keeps_strict_limits(self, fakes):
        fake = FakeInner(gaps=[0, 0.6, 0])
        fakes += [fake, FakeInner()]
        llm = LiveLLM(_cfg(call=None, stall=0.2))
        agen = llm.stream(REQUEST)
        assert await agen.__anext__() is fake.events[0]
        assert llm.retarget(_cfg(call=None, stall=None))
        with pytest.raises(ModelCallGuardError) as info:
            await _drain(agen, [])
        assert info.value.kind == "stall"

    async def test_in_flight_keeps_relaxed_limits(self, fakes):
        fake = FakeInner(gaps=[0, 0.5, 0])
        fakes += [fake, FakeInner()]
        llm = LiveLLM(_cfg(call=None, stall=None))
        agen = llm.stream(REQUEST)
        assert await agen.__anext__() is fake.events[0]
        assert llm.retarget(_cfg(call=None, stall=0.1))
        rest: list = []
        await _drain(agen, rest)
        assert rest == fake.events[1:]
