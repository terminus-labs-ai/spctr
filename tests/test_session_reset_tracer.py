"""Replacement Sessions keep the ``--trace`` tracer (obsidian-nbsl).

``cli.py`` builds ``Agent(tracer=CollectingTracer())`` for ``--trace`` and
prints ``tracer.get_trace()`` at shutdown. When the Agent replaces its Session
— REPL/TUI ``/reset`` via ``Agent.new_session()``, or Discord via the
``session_id`` setter — turns run on the replacement must still reach that same
tracer, or they are missing from the printed trace.

Each test drives a real SR2 turn loop with a scripted LLM and observes the
tracer only through its public ``get_trace()`` / ``compiled_request``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sr2.pipeline.tracing import CollectingTracer
from sr2.protocols.llm import StreamEvent
from sr2_spectre.agent import Agent
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.run_log import SessionLogManager
from tests.test_session_run_log import ScriptedLLM, _config, _events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_memory_dsn(monkeypatch):
    monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)


def _text(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text", text=text)]


def _agent(*rounds, tracer=None) -> Agent:
    llm = ScriptedLLM(*rounds)
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=llm):
        return Agent(config=_config(), session_id="edi-default", tracer=tracer)


def _static_firings(tracer: CollectingTracer) -> list:
    """Firings of the config's ``static`` system resolver — one per turn."""
    return [r for r in tracer.get_trace() if r.component_name == "static"]


def _repl_ctx() -> RunContext:
    return RunContext(
        interface="repl", mode=RunMode.INTERACTIVE, source="/work/proj", area="proj",
    )


def _log_files() -> list[Path]:
    d = SessionLogManager().directory
    return sorted(d.glob("*.jsonl")) if d.exists() else []


# ---------------------------------------------------------------------------
# AC1 — Agent.new_session() keeps the tracer
# ---------------------------------------------------------------------------

class TestNewSessionKeepsTracer:
    async def test_turn_after_new_session_appears_in_trace(self):
        tracer = CollectingTracer()
        agent = _agent(_text("ok"), tracer=tracer)

        agent.new_session()
        await agent.handle_user_message("after reset")

        assert tracer.get_trace(), "post-reset turn produced no firings in the tracer"
        assert _static_firings(tracer), "post-reset turn's pipeline firings missing"

    async def test_turn_after_new_session_updates_compiled_request(self):
        tracer = CollectingTracer()
        agent = _agent(_text("ok"), tracer=tracer)

        agent.new_session()
        await agent.handle_user_message("after reset")

        assert tracer.compiled_request is not None, (
            "post-reset compile not reported to the tracer"
        )

    async def test_trace_spans_turns_before_and_after_new_session(self):
        """The shutdown trace holds both the pre-reset and post-reset turns."""
        tracer = CollectingTracer()
        agent = _agent(_text("one"), _text("two"), tracer=tracer)

        await agent.handle_user_message("before reset")
        before = len(_static_firings(tracer))
        assert before >= 1, "precondition: pre-reset turn is traced"

        agent.new_session()
        await agent.handle_user_message("after reset")

        assert len(_static_firings(tracer)) > before, (
            "post-reset turn missing from the trace"
        )

    async def test_new_session_with_explicit_id_keeps_tracer(self):
        tracer = CollectingTracer()
        agent = _agent(_text("ok"), tracer=tracer)

        agent.new_session("fresh-frame")
        await agent.handle_user_message("after reset")

        assert _static_firings(tracer)

    async def test_tracer_survives_repeated_new_session(self):
        tracer = CollectingTracer()
        agent = _agent(_text("one"), _text("two"), tracer=tracer)

        agent.new_session()
        await agent.handle_user_message("first reset")
        after_first = len(_static_firings(tracer))
        assert after_first >= 1

        agent.new_session()
        await agent.handle_user_message("second reset")

        assert len(_static_firings(tracer)) > after_first


# ---------------------------------------------------------------------------
# AC2 — the session_id setter keeps the tracer
# ---------------------------------------------------------------------------

class TestSessionIdSetterKeepsTracer:
    async def test_turn_after_setter_appears_in_trace(self):
        tracer = CollectingTracer()
        agent = _agent(_text("ok"), tracer=tracer)

        agent.session_id = "channel-7"
        await agent.handle_user_message("after setter")

        assert _static_firings(tracer), "turn after session_id setter missing from trace"
        assert tracer.compiled_request is not None

    async def test_trace_spans_turns_before_and_after_setter(self):
        tracer = CollectingTracer()
        agent = _agent(_text("one"), _text("two"), tracer=tracer)

        await agent.handle_user_message("before")
        before = len(_static_firings(tracer))
        assert before >= 1

        agent.session_id = "channel-7"
        await agent.handle_user_message("after")

        assert len(_static_firings(tracer)) > before


# ---------------------------------------------------------------------------
# AC3 — no tracer: replacement still works
# ---------------------------------------------------------------------------

class TestNoTracerUnchanged:
    async def test_new_session_without_tracer_runs_a_turn(self):
        agent = _agent(_text("ok"))

        agent.new_session("fresh")
        result = await agent.handle_user_message("after reset")

        assert agent.session_id == "fresh"
        assert "ok" in result.text

    async def test_setter_without_tracer_runs_a_turn(self):
        agent = _agent(_text("ok"))

        agent.session_id = "channel-7"
        result = await agent.handle_user_message("after setter")

        assert agent.session_id == "channel-7"
        assert "ok" in result.text

    async def test_tracer_does_not_cross_agents(self):
        """Replacing an untraced Agent's Session never routes into another Agent's tracer."""
        tracer = CollectingTracer()
        traced = _agent(_text("a"), tracer=tracer)
        plain = _agent(_text("b"))

        plain.new_session()
        plain.session_id = "x"
        await plain.handle_user_message("untraced turn")

        assert tracer.get_trace() == [], "untraced Agent's turn leaked into another tracer"

        # Positive control: the tracer is live — the traced Agent's own turn reaches it.
        await traced.handle_user_message("traced turn")
        assert _static_firings(tracer)


# ---------------------------------------------------------------------------
# AC4 — run-context carry-over (obsidian-gb5j) still holds alongside a tracer
# ---------------------------------------------------------------------------

class TestTracerWithRunContext:
    async def test_reset_turn_reaches_both_tracer_and_new_log(self):
        tracer = CollectingTracer()
        agent = _agent(_text("ok"), tracer=tracer)
        ctx = _repl_ctx()
        agent.set_run_context(ctx)
        (first,) = _log_files()

        agent.new_session()
        await agent.handle_user_message("after reset")

        assert agent.run_context == ctx
        new = [p for p in _log_files() if p != first]
        assert len(new) == 1, "reset Session opened no log of its own"
        names = [e["event"] for e in _events(new[0])]
        assert any(n.startswith("turn.") for n in names), names
        assert _static_firings(tracer), "post-reset turn missing from the tracer"
