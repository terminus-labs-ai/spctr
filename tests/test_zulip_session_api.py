"""Agent.create_session — independent Sessions through the public API (spc-118, AC15).

The Zulip interface needs one independent Session per channel/topic
conversation. The only public seam for that is
``Agent.create_session(frame_id)``: it must delegate to the existing
Runtime seam and return a *real* Session — its own history, its own SR2,
the Agent's tracer, the Runtime's shared registry/LLM/memory — without
touching the facade's current conversation. ``Agent.new_session()`` keeps
its existing contract: reset the facade's single conversation, return
None.

Everything here runs against the real Agent / Runtime / Session / SR2
pipeline and a real tool executor. The only boundary mocked is the LLM
itself: ``LiteLLMCallable`` construction is patched so the turns below
never contact a model endpoint, and ``ScriptedLLM`` replays deterministic
stream events.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import patch

import pytest

from sr2.pipeline.tracing import CollectingTracer
from sr2.protocols.llm import CompletionRequest, StreamEvent
from sr2_spectre.agent import Agent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.session import Session

SYSTEM_MARKER = "SYSTEM-PROMPT-MARKER-7f3a"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch) -> None:
    """Isolate SR2_HOME and disable inherited persistence/pointer state."""
    monkeypatch.setenv("SR2_HOME", str(tmp_path / "sr2home"))
    monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
    monkeypatch.delenv("SR2_AREA", raising=False)
    monkeypatch.delenv("SR2_ACTIVE_MODEL_FILE", raising=False)


def _config() -> SpectreConfig:
    """Minimal real pipeline: static system marker + shared tool provider.

    The marker is a synthetic string so every test can see, through the
    tracer's compiled request or through a tool's view of the compiled
    context, exactly which conversation history a model call received —
    which is what makes history independence provable.
    """
    return SpectreConfig(
        agent=AgentConfig(name="miranda"),
        models={
            "default": ModelConfig(
                model="test-model", base_url="http://localhost:9999"
            )
        },
        pipeline={
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [
                        {"type": "static", "config": {"text": SYSTEM_MARKER}}
                    ],
                },
                {
                    "name": "tools",
                    "target": "tools",
                    "resolvers": [],
                    "tool_providers": [{"type": "spectre_tools"}],
                },
                {
                    "name": "conversation",
                    "target": "messages",
                    "resolvers": [{"type": "session"}, {"type": "input"}],
                },
            ]
        },
        provenance_store_path="",
        memory_store_dsn="",
    )


class ScriptedLLM:
    """LLM boundary double: replays scripted stream rounds, records requests.

    Stands in for the LiteLLM endpoint only — the pipeline, tool loop and
    session state around it are all real. ``requests`` captures the compiled
    request of each model call, in order.
    """

    model = "test-model"

    def __init__(self, *rounds) -> None:
        self._rounds: list = list(rounds)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> Any:
        raise NotImplementedError

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        round_ = (
            self._rounds.pop(0)
            if self._rounds
            else [StreamEvent(type="text", text="done")]
        )
        if isinstance(round_, Exception):
            raise round_
        for ev in round_:
            yield ev
        yield StreamEvent(type="end")


def _text(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text", text=text)]


def _tool_call(tool_id: str, name: str, **args) -> StreamEvent:
    return StreamEvent(
        type="tool_use", tool_use_id=tool_id, tool_name=name, tool_input=args
    )


def _agent(
    llm: ScriptedLLM,
    tracer: CollectingTracer | None = None,
    session_id: str = "miranda-default",
) -> Agent:
    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=llm):
        return Agent(config=_config(), session_id=session_id, tracer=tracer)


def _ctx(interface: str = "single_shot", area: str | None = None) -> RunContext:
    return RunContext(interface=interface, mode=RunMode.HEADLESS, source=None, area=area)


def _request_strings(request: CompletionRequest) -> list[str]:
    """All text in a compiled request (system + messages), flattened."""
    out: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, str):
            out.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(request.system or [])
    walk([m.model_dump() for m in request.messages])
    return out


def _log_files(home: Path) -> list[Path]:
    d = home / "logs" / "sessions"
    return sorted(d.glob("*.jsonl")) if d.exists() else []


def _home() -> Path:
    return Path(os.environ["SR2_HOME"])


def _named(events: list[dict], name: str) -> list[dict]:
    return [e for e in events if e["event"] == name]


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# AC15 — create_session returns an independent real Session
# ---------------------------------------------------------------------------


class TestCreateSessionReturnsRealSession:
    async def test_returns_a_real_session_object(self):
        """The returned object is a real Session, not a double or the facade's."""
        agent = _agent(ScriptedLLM())
        session = agent.create_session("zulip-42-deploy")

        assert isinstance(session, Session)
        assert session.frame_id == "zulip-42-deploy"
        assert session is not agent._session

    async def test_returns_a_new_session_per_call(self):
        s1 = _agent(ScriptedLLM()).create_session("frame-a")
        s2 = _agent(ScriptedLLM()).create_session("frame-b")
        assert s1 is not s2
        assert s1.frame_id != s2.frame_id

    async def test_session_carries_the_agents_tracer(self):
        """A turn on the returned Session reaches the Agent's own tracer."""
        tracer = CollectingTracer()
        agent = _agent(ScriptedLLM(_text("hi")), tracer=tracer)
        session = agent.create_session("traced-frame")

        await session.handle_user_message("hello")

        assert tracer.get_trace(), "turn on created Session produced no tracer firings"
        assert tracer.compiled_request is not None, (
            "turn on created Session never compiled a request for the tracer"
        )


# ---------------------------------------------------------------------------
# AC15 — same Runtime / shared config behavior
# ---------------------------------------------------------------------------


class TestSharedRuntime:
    def test_created_session_shares_the_agents_runtime(self):
        """The created Session is wired to the Agent's Runtime shared stores —
        not a second Runtime — with the behavior proof in the sibling test."""
        agent = _agent(ScriptedLLM())
        session = agent.create_session("shared")

        assert session._memory_store is agent._runtime._memory_store, (
            "created Session must share the Runtime's memory store, not its own"
        )

    async def test_tool_registered_on_agent_reaches_created_session(self):
        """Shared registry (the public 'same Runtime' behavior): a tool
        registered on the Agent's public registry is usable by a Session the
        Agent created — proving both hang off the one shared Runtime."""
        llm = ScriptedLLM([_tool_call("tu_ping", "ping")], _text("done"))
        agent = _agent(llm)
        session = agent.create_session("tool-frame")

        async def ping(**_):
            return "pong"

        agent.registry.register("ping", "pings", {"type": "object"}, ping)

        result = await session.handle_user_message("ping it")

        assert result.text == "done"
        assert result.tool_calls_executed == 1, (
            "the shared-registry tool did not execute on the created Session"
        )
        # The tool actually ran through the shared registry, so its output
        # entered this session's compiled context for the follow-up call.
        assert any("pong" in s for s in _request_strings(llm.requests[-1])), (
            "tool result from the shared registry did not reach the created Session"
        )

    def test_sessions_share_the_runtime_llm_handle(self):
        """One model path per Runtime: the facade and every created Session
        are handed the same LiveLLM (each still keeps its own SR2)."""
        agent = _agent(ScriptedLLM(_text("a"), _text("b")))
        facade = agent._session
        created = agent.create_session("other")

        assert facade.sr2 is not created.sr2, "each Session needs its own SR2"
        # The documented invariant: one LLM path, retargetable, per Runtime.
        assert (
            facade._llm is agent._runtime.llm
            and created._llm is agent._runtime.llm
        ), "facade and created Session must share the Runtime's single LLM handle"


# ---------------------------------------------------------------------------
# AC15 — independent histories
# ---------------------------------------------------------------------------


class TestIndependentHistories:
    async def test_created_session_history_does_not_contain_facade_history(self):
        llm = ScriptedLLM(_text("facade reply"))
        agent = _agent(llm)
        await agent.handle_user_message("facade question")
        facade_before = [m for m in agent.history]
        assert len(facade_before) == 2

        session = agent.create_session("fresh-topic")
        assert session.history == [], (
            "a created Session starts with a clean history, not the facade's"
        )

    async def test_facade_history_unchanged_by_create_session(self):
        """create_session() must not reset or append to the facade conversation."""
        llm = ScriptedLLM(_text("r1"), _text("r2"))
        agent = _agent(llm)
        await agent.handle_user_message("first")
        facade_history = list(agent.history)
        facade_id_before = agent.session_id
        facade_obj = agent._session

        agent.create_session("topic-a")
        agent.create_session("topic-b")

        assert agent.history == facade_history, (
            "create_session() must leave the facade's conversation untouched"
        )
        assert agent.session_id == facade_id_before
        assert agent._session is facade_obj, "create_session() must not replace the facade"

    async def test_two_created_sessions_keep_separate_histories(self):
        """Separate calls ⇒ separate transcripts in both directions."""
        llm = ScriptedLLM(_text("ra"), _text("rb"), _text("ra2"))
        agent = _agent(llm)
        a = agent.create_session("topic-a")
        b = agent.create_session("topic-b")

        await a.handle_user_message("alpha question")
        await b.handle_user_message("beta question")

        a_user_texts = [
            blk.text
            for m in a.history
            if m.role == "user"
            for blk in m.content
            if hasattr(blk, "text")
        ]
        b_user_texts = [
            blk.text
            for m in b.history
            if m.role == "user"
            for blk in m.content
            if hasattr(blk, "text")
        ]
        assert a_user_texts == ["alpha question"]
        assert b_user_texts == ["beta question"]

        # A second turn in A: A keeps its full transcript; B still only sees
        # its own.
        await a.handle_user_message("alpha follow-up")
        a_user_texts2 = [
            blk.text
            for m in a.history
            if m.role == "user"
            for blk in m.content
            if hasattr(blk, "text")
        ]
        b_user_texts2 = [
            blk.text
            for m in b.history
            if m.role == "user"
            for blk in m.content
            if hasattr(blk, "text")
        ]
        assert a_user_texts2 == ["alpha question", "alpha follow-up"]
        assert b_user_texts2 == ["beta question"]

    async def test_history_independence_seen_by_the_model(self):
        """Proven through the compiled request, not just list identity."""
        llm = ScriptedLLM(_text("one"), _text("two"))
        agent = _agent(llm)
        a = agent.create_session("topic-a")
        b = agent.create_session("topic-b")

        await a.handle_user_message("ALPHA-MARKER-111")
        await b.handle_user_message("BETA-MARKER-222")

        a_req = llm.requests[0]
        b_req = llm.requests[1]
        a_strings = _request_strings(a_req)
        b_strings = _request_strings(b_req)

        assert any("ALPHA-MARKER-111" in s for s in a_strings)
        assert not any("ALPHA-MARKER-111" in s for s in b_strings)
        assert any("BETA-MARKER-222" in s for s in b_strings)
        assert not any("BETA-MARKER-222" in s for s in a_strings)

    async def test_facade_history_reaches_only_facade_model_calls(self):
        llm = ScriptedLLM(_text("f"), _text("c"))
        agent = _agent(llm)
        created = agent.create_session("topic-x")

        await created.handle_user_message("CREATED-MARKER")
        await agent.handle_user_message("FACADE-MARKER")

        created_req = llm.requests[0]
        facade_req = llm.requests[1]
        assert not any("FACADE-MARKER" in s for s in _request_strings(created_req))
        assert not any("CREATED-MARKER" in s for s in _request_strings(facade_req))
        assert any("FACADE-MARKER" in s for s in _request_strings(facade_req))


# ---------------------------------------------------------------------------
# AC15 — facade preserved
# ---------------------------------------------------------------------------


class TestFacadePreserved:
    async def test_facade_turn_after_create_session_continues_own_conversation(
        self,
    ):
        llm = ScriptedLLM(_text("r1"), _text("r2"))
        agent = _agent(llm)
        await agent.handle_user_message("setup question")
        agent.create_session("topic")
        await agent.handle_user_message("follow-up question")

        assert agent.session_id == "miranda-default"
        user_texts = [
            blk.text
            for m in agent.history
            if m.role == "user"
            for blk in m.content
            if hasattr(blk, "text")
        ]
        assert user_texts == ["setup question", "follow-up question"]

        # The facade's second model call saw its own first turn as history.
        facade_req = llm.requests[1]
        assert any("setup question" in s for s in _request_strings(facade_req))

    async def test_created_session_turn_does_not_perturb_facade(self):
        """A turn on a created Session must not alter facade state at all."""
        llm = ScriptedLLM(_text("r1"), _text("r2"))
        agent = _agent(llm)
        await agent.handle_user_message("setup")
        snapshot_history = [m.model_dump() for m in agent.history]
        snapshot_ctx = agent.run_context
        facade = agent._session
        facade_sr2 = facade.sr2

        created = agent.create_session("topic")
        await created.handle_user_message("on the side")

        assert [m.model_dump() for m in agent.history] == snapshot_history
        assert agent.run_context is snapshot_ctx
        assert agent._session is facade
        assert facade.sr2 is facade_sr2


# ---------------------------------------------------------------------------
# AC15 — new_session() keeps its existing contract
# ---------------------------------------------------------------------------


class TestNewSessionCompatibility:
    async def test_new_session_returns_none_and_resets_facade(self):
        llm = ScriptedLLM(_text("r1"), _text("r2"))
        agent = _agent(llm)
        await agent.handle_user_message("before reset")
        assert len(agent.history) == 2

        result = agent.new_session()

        assert result is None
        assert agent.history == []
        # The facade keeps moving under the same identity.
        await agent.handle_user_message("after reset")
        user_texts = [
            blk.text
            for m in agent.history
            if m.role == "user"
            for blk in m.content
            if hasattr(blk, "text")
        ]
        assert user_texts == ["after reset"]

    def test_new_session_with_explicit_id_resets_and_returns_none(self):
        agent = _agent(ScriptedLLM(_text("r1"), _text("r2")))
        assert agent.session_id == "miranda-default"

        assert agent.new_session("explicit-frame") is None
        assert agent.session_id == "explicit-frame"
        assert agent.history == []

    async def test_new_session_resets_even_after_create_session_calls(self):
        """The two seams do not interfere in either order."""
        llm = ScriptedLLM(_text("r1"), _text("r2"), _text("r3"))
        agent = _agent(llm)
        created = agent.create_session("topic")
        await created.handle_user_message("topic work")

        assert agent.new_session() is None
        assert agent.history == []
        # ...and a Session created after the reset still gets a fresh history.
        late = agent.create_session("late-topic")
        assert late is not created
        assert late.history == []

    async def test_new_session_closes_facade_log_not_created_session_log(self):
        """Reset closes the replaced facade's log file only; the created
        Session keeps writing to its own independent log."""
        llm = ScriptedLLM(_text("r1"), _text("r2"), _text("r3"))
        agent = _agent(llm)
        agent.set_run_context(_ctx())
        (facade_log,) = _log_files(_home())

        created = agent.create_session("topic")
        created.set_run_context(_ctx("repl"))
        created_logs = [
            p for p in _log_files(_home()) if p != facade_log
        ]
        assert len(created_logs) == 1, "created Session did not open its own log"

        agent.new_session()
        await agent.handle_user_message("after reset")

        # The created session log is still open: another turn writes to it.
        await created.handle_user_message("still alive")
        created_events = _events(created_logs[0])
        assert any(
            "still alive" in json.dumps(e.get("data", {}))
            for e in _named(created_events, "turn.start")
        ), "created Session's log stopped after the facade reset"
        assert _named(created_events, "turn.complete")


# ---------------------------------------------------------------------------
# Failure matrix — a created Session runs a real, self-contained turn
# ---------------------------------------------------------------------------


class TestCreatedSessionTurnBehavior:
    async def test_text_turn_completes_on_created_session(self):
        llm = ScriptedLLM(_text("created reply"))
        agent = _agent(llm)
        session = agent.create_session("topic")

        result = await session.handle_user_message("what is up")

        assert result.text == "created reply"
        assert result.tool_calls_executed == 0
        roles = [m.role for m in session.history]
        assert roles == ["user", "assistant"]

    async def test_tool_loop_runs_on_created_session_via_shared_registry(self):
        llm = ScriptedLLM(
            [_tool_call("tu_echo", "echo", text="hello")], _text("wrapped")
        )
        agent = _agent(llm)
        session = agent.create_session("tool-topic")

        async def echo(text: str = ""):
            return f"echoed:{text}"

        agent.registry.register("echo", "echoes", {"type": "object"}, echo)

        result = await session.handle_user_message("echo hello")

        assert result.text == "wrapped"
        assert result.tool_calls_executed == 1
        # Spectre keeps the authoritative transcript: the user turn plus the
        # final assistant reply. The tool round-trip is proven through the
        # compiled request, not the history list.
        roles = [m.role for m in session.history]
        assert roles == ["user", "assistant"]
        # The shared-registry tool actually ran, and its output reached the
        # session's compiled context for the follow-up model call.
        assert any(
            "echoed:hello" in s for s in _request_strings(llm.requests[-1])
        ), "shared-registry tool result did not reach the created Session's context"

    async def test_created_session_llm_failure_propagates_and_preserves_state(self):
        llm = ScriptedLLM(RuntimeError("endpoint down"))
        agent = _agent(llm)
        session = agent.create_session("failing-topic")

        with pytest.raises(RuntimeError, match="endpoint down"):
            await session.handle_user_message("do something")

        # No fabricated assistant reply on failure.
        roles = [m.role for m in session.history]
        assert roles == ["user"]
        # The session is still usable: the next scripted round succeeds.
        llm._rounds.append(_text("recovered"))
        result = await session.handle_user_message("retry it")
        assert result.text == "recovered"

    async def test_created_session_run_context_is_independent(self):
        llm = ScriptedLLM(_text("r1"), _text("r2"))
        agent = _agent(llm)
        created = agent.create_session("area-topic")

        agent.set_run_context(_ctx(area="facade-area"))
        created.set_run_context(_ctx(interface="zulip", area="other-area"))

        assert agent.run_context is not created.run_context
        assert created.run_context.area == "other-area"
        assert agent.run_context.area == "facade-area"
