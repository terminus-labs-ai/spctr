"""SR2 in-band pipeline error events reach the session log (obsidian-jnnl.1).

SR2 yields ``StreamEvent(type="error", errors=[...])`` for non-fatal bus
errors that have no failing pipeline component behind them (sync/async
subscriber errors, ``bus_drained`` callback errors). The turn still
completes. The Session must record those errors in its ordered log.

Errors are triggered through SR2's public event bus (``session.sr2.bus``
``subscribe``), on a real Runtime/Session/SR2 turn. The only fake is the
provider-level LLM callable. The new log event may be named
``pipeline.error`` or ``sr2.error``; it is located by name and content.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sr2.protocols.llm import StreamEvent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.runtime import Runtime

ERROR_EVENTS = {"pipeline.error", "sr2.error"}
FINAL_TEXT = "all finished"


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
             "resolvers": [{"type": "static", "config": {"text": "system prompt"}}]},
            {"name": "tools", "target": "tools", "resolvers": [],
             "tool_providers": [{"type": "spectre_tools"}]},
            {"name": "conversation", "target": "messages",
             "resolvers": [{"type": "session"}, {"type": "input"}]},
        ]},
        provenance_store_path="",
        memory_store_dsn="",
    )


class ScriptedLLM:
    """Provider-boundary fake: successive stream() calls replay scripted rounds."""

    model = "test-model"

    def __init__(self, *rounds) -> None:
        self._rounds = list(rounds)

    async def complete(self, request):
        raise NotImplementedError

    async def stream(self, request):
        round_ = self._rounds.pop(0) if self._rounds else [StreamEvent(type="text", text="done")]
        for ev in round_:
            yield ev
        yield StreamEvent(type="end", meta={"finish_reason": "stop"})


def _tool_round() -> list[StreamEvent]:
    return [StreamEvent(type="tool_use", tool_use_id="tu-1", tool_name="echo",
                        tool_input={"value": "x"})]


def _text_round() -> list[StreamEvent]:
    return [StreamEvent(type="text", text=FINAL_TEXT)]


def _session(*rounds):
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=ScriptedLLM(*rounds)):
        runtime = Runtime(_config())

    async def echo(value: str = "") -> str:
        return f"echo:{value}"

    runtime.registry.register("echo", "echo tool", {"type": "object"}, echo)
    session = runtime.new_session("frame-1")
    session.set_run_context(
        RunContext(interface="single_shot", mode=RunMode.HEADLESS, source=None)
    )
    return session


def _raising(message: str):
    def callback(event) -> None:
        raise RuntimeError(message)
    return callback


def _async_raising(message: str):
    async def callback(event) -> None:
        raise RuntimeError(message)
    return callback


def _log_files(home: Path) -> list[Path]:
    return sorted((home / "logs" / "sessions").glob("*.jsonl"))


def _events(home: Path) -> list[dict]:
    files = _log_files(home)
    assert len(files) == 1, files
    return _read_events(files[0])


def _read_events(path: Path) -> list[dict]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    seqs = [e["sequence"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    return events


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _strings(v)]
    return []


def _error_events_with(events: list[dict], needle: str) -> list[dict]:
    return [
        e for e in events
        if e["event"] in ERROR_EVENTS and any(needle in s for s in _strings(e["data"]))
    ]


def _one(events: list[dict], name: str) -> dict:
    matches = [e for e in events if e["event"] == name]
    assert len(matches) == 1, [e["event"] for e in events]
    return matches[0]


# ---------------------------------------------------------------------------
# AC1 / AC2 — in-band errors are logged in sequence, turn result unchanged
# ---------------------------------------------------------------------------

class TestInBandErrorLogging:
    async def test_sync_subscriber_error_is_logged_between_turn_start_and_complete(self, home):
        session = _session(_text_round())
        session.sr2.bus.subscribe("user_input", _raising("sync-boom-7c1"))

        result = await session.handle_user_message("go")

        assert result.text == FINAL_TEXT
        events = _events(home)
        logged = _error_events_with(events, "sync-boom-7c1")
        assert logged, [(e["event"], e["data"]) for e in events]
        start = _one(events, "turn.start")["sequence"]
        complete = _one(events, "turn.complete")["sequence"]
        assert all(start < e["sequence"] < complete for e in logged)

    async def test_bus_drained_callback_error_is_logged_before_turn_complete(self, home):
        session = _session(_text_round())
        session.sr2.bus.subscribe("bus_drained", _async_raising("drained-boom-4e9"))

        result = await session.handle_user_message("go")

        assert result.text == FINAL_TEXT
        events = _events(home)
        logged = _error_events_with(events, "drained-boom-4e9")
        assert logged, [(e["event"], e["data"]) for e in events]
        complete = _one(events, "turn.complete")["sequence"]
        assert all(e["sequence"] < complete for e in logged)

    async def test_tool_iteration_error_leaves_turn_result_unchanged(self, home):
        baseline = await _session(_tool_round(), _text_round()).handle_user_message("go")
        baseline_logs = set(_log_files(home))

        session = _session(_tool_round(), _text_round())
        session.sr2.bus.subscribe("tool_result", _raising("tool-result-boom-2b8"))
        result = await session.handle_user_message("go")

        assert baseline.tool_calls_executed == 1
        assert (result.text, result.tool_calls_executed) == (
            baseline.text, baseline.tool_calls_executed)
        new_logs = set(_log_files(home)) - baseline_logs
        assert len(new_logs) == 1, new_logs
        events = _read_events(new_logs.pop())
        logged = _error_events_with(events, "tool-result-boom-2b8")
        assert logged, [(e["event"], e["data"]) for e in events]
        complete = _one(events, "turn.complete")
        assert complete["data"]["tool_calls"] == 1
        assert all(e["sequence"] < complete["sequence"] for e in logged)
        assert not any(e["event"] == "turn.error" for e in events)

    async def test_multiple_errors_are_all_preserved(self, home):
        session = _session(_text_round())
        session.sr2.bus.subscribe("user_input", _raising("first-boom-a11"))
        session.sr2.bus.subscribe("user_input", _raising("second-boom-b22"))

        await session.handle_user_message("go")

        events = _events(home)
        assert _error_events_with(events, "first-boom-a11")
        assert _error_events_with(events, "second-boom-b22")

    async def test_clean_turn_logs_no_error_event(self, home):
        session = _session(_tool_round(), _text_round())

        result = await session.handle_user_message("go")

        assert result.text == FINAL_TEXT
        events = _events(home)
        assert [e for e in events if e["event"] in ERROR_EVENTS] == []
        assert _one(events, "turn.complete")

