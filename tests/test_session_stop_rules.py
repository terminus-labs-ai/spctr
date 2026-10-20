"""SR2 stop-rule stream events reach the session log; limit notice is configurable (spc-966).

SR2 yields ``StreamEvent(type="repeat_guard", meta=...)`` when its consecutive
repeat guard alters or stops the run, and ``StreamEvent(type="final_answer",
meta=...)`` when the forced final answer fires. The Session must record those
in its ordered log as ``tool.repeat_guard`` and ``sr2.final_answer``. A config
with both ``pipeline`` blocks must reach SR2's ``PipelineConfig`` unchanged,
and ``agent.tool_limit_notice`` must replace the hard-coded limit notice.

Turns run through a real Runtime/Session/SR2 with only the provider-level LLM
faked; for the guard events SR2's ``turn`` is stubbed to yield the scripted
StreamEvents. The session log JSONL is read back and events located by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sr2.config.models import ToolLoopLimitError
from sr2.protocols.llm import StreamEvent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig, load_config
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.runtime import Runtime


# ---------------------------------------------------------------------------
# Helpers (same shape as tests/test_session_inband_errors.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "sr2home"
    h.mkdir()
    monkeypatch.setenv("SR2_HOME", str(h))
    # SR2_SESSION_LOG_DIR (if exported) takes precedence over SR2_HOME in
    # SessionLogManager.directory; drop it so logs land under the fixture home.
    monkeypatch.delenv("SR2_SESSION_LOG_DIR", raising=False)
    monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
    return h


def _config(**agent_kwargs) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="edi", **agent_kwargs),
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


class NullLLM:
    """Provider-boundary fake: never called — the SR2 turn below is stubbed."""

    model = "test-model"

    async def complete(self, request):
        raise NotImplementedError

    async def stream(self, request):
        raise NotImplementedError
        yield  # pragma: no cover — keeps this an async generator


def _session(config: SpectreConfig | None = None):
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=NullLLM()):
        runtime = Runtime(config or _config())
    session = runtime.new_session("frame-1")
    session.set_run_context(
        RunContext(interface="single_shot", mode=RunMode.HEADLESS, source=None)
    )
    return session


def _stub_turn(session, events):
    """Replace this session's SR2 turn with one replaying scripted StreamEvents."""

    async def fake_turn(*args, **kwargs):
        for ev in events:
            yield ev

    session.sr2.turn = fake_turn


def _log_files(home: Path) -> list[Path]:
    return sorted((home / "logs" / "sessions").glob("*.jsonl"))


def _read_events(path: Path) -> list[dict]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    seqs = [e["sequence"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    return events


def _one(events: list[dict], name: str) -> dict:
    matches = [e for e in events if e["event"] == name]
    assert len(matches) == 1, [e["event"] for e in events]
    return matches[0]


# ---------------------------------------------------------------------------
# AC7-r1 / AC7-r2 — the two new stream events are logged with their meta
# ---------------------------------------------------------------------------

class TestStopRuleEventLogging:
    async def test_repeat_guard_event_is_logged_with_its_meta(self, home):
        meta = {
            "call": 2,
            "first_call": 1,
            "identical_calls": 2,
            "same_result_repeats": 1,
            "action": "note",
        }
        session = _session()
        _stub_turn(session, [
            StreamEvent(type="repeat_guard", meta=meta),
            StreamEvent(type="end"),
        ])

        await session.handle_user_message("go")

        events = _read_events(_log_files(home)[0])
        logged = _one(events, "tool.repeat_guard")
        assert logged["data"] == meta
        complete = _one(events, "turn.complete")
        assert logged["sequence"] < complete["sequence"]

    async def test_final_answer_event_is_logged_with_its_meta(self, home):
        meta = {"reason": "repeat", "fallback_used": False}
        session = _session()
        _stub_turn(session, [
            StreamEvent(type="final_answer", meta=meta),
            StreamEvent(type="end"),
        ])

        await session.handle_user_message("go")

        events = _read_events(_log_files(home)[0])
        logged = _one(events, "sr2.final_answer")
        assert logged["data"] == {"reason": "repeat", "fallback_used": False}
        complete = _one(events, "turn.complete")
        assert logged["sequence"] < complete["sequence"]


# ---------------------------------------------------------------------------
# AC7-r3 — both pipeline blocks pass through Spectre config unchanged
# ---------------------------------------------------------------------------

class TestPipelineBlockPassthrough:
    def test_custom_final_answer_and_repeat_guard_reach_pipeline_config(self):
        raw = {
            "agent": {"name": "edi"},
            "models": {"default": {"model": "test-model"}},
            "pipeline": {
                "layers": [
                    {"name": "system", "target": "system",
                     "resolvers": [{"type": "static", "config": {"text": "s"}}]},
                ],
                "final_answer": {"enabled": True, "instruction": "X {reason}"},
                "repeat_guard": {
                    "enabled": True,
                    "notes_before_stop": 3,
                    "note": "N",
                },
            },
        }

        config = load_config(raw)

        assert config.pipeline.final_answer.enabled is True
        assert config.pipeline.final_answer.instruction == "X {reason}"
        assert config.pipeline.repeat_guard.enabled is True
        assert config.pipeline.repeat_guard.notes_before_stop == 3
        assert config.pipeline.repeat_guard.note == "N"


# ---------------------------------------------------------------------------
# AC7-r4 / AC7-r5 — the limit notice is configurable
# ---------------------------------------------------------------------------

class TestToolLimitNotice:
    async def test_custom_notice_replaces_the_limit_text(self, home):
        session = _session(_config(tool_limit_notice="LIMIT"))

        async def raising_turn(*args, **kwargs):
            raise ToolLoopLimitError("too many tool iterations")
            yield  # pragma: no cover — keeps this an async generator

        session.sr2.turn = raising_turn

        result = await session.handle_user_message("go")

        assert result.text == "LIMIT"

    async def test_default_notice_is_the_historical_text(self, home):
        session = _session()

        async def raising_turn(*args, **kwargs):
            raise ToolLoopLimitError("too many tool iterations")
            yield  # pragma: no cover — keeps this an async generator

        session.sr2.turn = raising_turn

        result = await session.handle_user_message("go")

        assert result.text == "Tool iteration limit reached; stopping."
