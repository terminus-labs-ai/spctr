"""obsidian-la2z: LiveLLM/LoggedLLM record the served model in the session log.

sr2's LiteLLMCallable reports the backend's real model id on the terminal end
event (``meta["served_model"]``, alongside ``finish_reason``); LiveLLM is a
pass-through and LoggedLLM must write the value into the ``model.end`` entry
next to ``usage``. Absence is not an error: a stream that carries neither
usage nor a model logs both as null and the end event keeps working for
callers that read only finish_reason.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from sr2.models import Message, TextBlock, TokenUsage
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent

from spctr.config import ModelConfig
from spctr.live_llm import LiveLLM, LoggedLLM

REQUEST = CompletionRequest(
    messages=[Message(role="user", content=[TextBlock(text="hi")])]
)


class FakeInner:
    """Stand-in for LiteLLMCallable yielding a scripted event sequence."""

    model = "openai/pool-name"

    def __init__(self, events) -> None:
        self._events = list(events)

    async def stream(self, request):
        for event in self._events:
            yield event

    async def complete(self, request):  # pragma: no cover - unused here
        raise NotImplementedError


class RecordingLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict]] = []

    def append(self, event, data=None):
        self.entries.append((event, dict(data or {})))

    def named(self, name: str) -> list[dict]:
        return [data for event, data in self.entries if event == name]


def _live_llm(inner: FakeInner) -> LiveLLM:
    cfg = ModelConfig(model="pool-name", base_url="http://test:8000")
    with patch("spctr.live_llm.LiteLLMCallable", return_value=inner):
        return LiveLLM(cfg)


async def _recorded_end(events) -> dict:
    log = RecordingLog()
    llm = LoggedLLM(_live_llm(FakeInner(events)), lambda: log, lambda: "p")
    got = [e async for e in llm.stream(REQUEST)]
    assert got == events, "the stream must pass through unchanged"
    (end,) = log.named("model.end")
    return end


async def _drain(llm, expected):
    got = [e async for e in llm.stream(REQUEST)]
    assert got == expected


USAGE = TokenUsage(input_tokens=14, output_tokens=1)


class TestServedModelInTheSessionLog:
    async def test_model_end_records_served_model_and_usage(self):
        end = await _recorded_end([
            StreamEvent(type="text", text="ok"),
            StreamEvent(type="usage", usage=USAGE),
            StreamEvent(
                type="end",
                meta={"finish_reason": "stop", "served_model": "qwen3.8-flash-next"},
            ),
        ])
        assert end["finish_reason"] == "stop"
        assert end["served_model"] == "qwen3.8-flash-next"
        assert end["usage"] == USAGE.model_dump()
        assert end["usage"]["input_tokens"] == 14
        assert end["usage"]["output_tokens"] == 1
        assert end["duration_ms"] >= 0

    async def test_stream_without_usage_or_model_logs_nulls_and_does_not_crash(self):
        end = await _recorded_end([
            StreamEvent(type="text", text="ok"),
            StreamEvent(type="end", meta={"finish_reason": "stop", "served_model": None}),
        ])
        assert end["usage"] is None
        assert end["served_model"] is None
        assert end["finish_reason"] == "stop"

    async def test_end_event_without_served_model_key_logs_null(self):
        """A callable that predates the served-model field must still log cleanly."""
        end = await _recorded_end([
            StreamEvent(type="end", meta={"finish_reason": "stop"}),
        ])
        assert end["finish_reason"] == "stop"
        assert end["served_model"] is None
        assert end["usage"] is None

    async def test_end_event_with_no_meta_at_all(self):
        """Callers that read only finish_reason keep working; nothing crashes."""
        end = await _recorded_end([StreamEvent(type="end")])
        assert end["finish_reason"] is None
        assert end["served_model"] is None
        assert end["usage"] is None


def _complete_response(served_model: str | None) -> CompletionResponse:
    return CompletionResponse(
        id="resp-1",
        content=[TextBlock(text="ok")],
        stop_reason="stop",
        usage=USAGE,
        served_model=served_model,
    )


class TestCompleteServedModelInTheSessionLog:
    """spc-1007: the non-streaming path logs served_model in model.end too."""

    async def _recorded_end(self, served_model: str | None) -> dict:
        response = _complete_response(served_model)

        class FakeCompleteInner(FakeInner):
            async def complete(self, request):
                return response

        log = RecordingLog()
        llm = LoggedLLM(_live_llm(FakeCompleteInner([])), lambda: log, lambda: "p")
        await llm.complete(REQUEST)
        (end,) = log.named("model.end")
        return end

    async def test_complete_logs_served_model_from_inner_response(self):
        end = await self._recorded_end("qwen-real")
        assert end["served_model"] == "qwen-real"
        assert end["finish_reason"] == "stop"
        assert end["usage"] == USAGE.model_dump()
        assert end["duration_ms"] >= 0

    async def test_complete_without_served_model_logs_null(self):
        end = await self._recorded_end(None)
        assert end["served_model"] is None
        assert end["finish_reason"] == "stop"


class TestLiveLLMPassThrough:
    async def test_live_llm_forwards_end_event_metadata_untouched(self):
        events = [
            StreamEvent(
                type="end",
                meta={"finish_reason": "stop", "served_model": "qwen3.8-flash-next"},
            ),
        ]
        await _drain(_live_llm(FakeInner(events)), events)
