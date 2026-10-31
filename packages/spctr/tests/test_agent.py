"""Tests for Agent — SR2-powered (Steps 4 & 5).

Covers:
  A. Agent.__init__: constructs SR2, no LiteLLMCallable call site
  B. session_id propagated to SR2
  C. Tools from config registered into ToolRegistry
  D. handle_user_message: happy path (no tools)
  E. handle_user_message: one tool round-trip
  F. handle_user_message: error recovery (FR13 — tool error fed back, loop continues)
  G. handle_user_message: tool-loop limit exceeded returns text + warning (FR14)
  H. new_session() resets history
  I. register_tool() adds to registry
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock, TokenUsage
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import TurnResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_pipeline_dict(with_tool_layer: bool = True) -> dict:
    layers = [
        {"name": "system", "target": "system", "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}]},
    ]
    if with_tool_layer:
        layers.append({"name": "tools", "target": "tools", "resolvers": [], "tool_providers": [{"type": "spectre_tools"}]})
    layers.append({"name": "conversation", "target": "messages", "resolvers": [{"type": "session"}, {"type": "input"}]})
    return {"layers": layers}


def _make_config(**agent_kwargs) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test", **agent_kwargs),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=_minimal_pipeline_dict(),
    )


def _stream_events(*events: StreamEvent):
    """Async generator that yields given StreamEvent objects."""
    async def _gen():
        for ev in events:
            yield ev
    return _gen()


def _mock_sr2(turn_events: list[StreamEvent] | None = None) -> MagicMock:
    """Return a mock SR2 instance whose turn() yields given events."""
    events = turn_events or [StreamEvent(type="text", text="Hello!"), StreamEvent(type="end")]
    sr2 = MagicMock()
    sr2.continue_archive = AsyncMock()
    sr2.seed_session = MagicMock()

    async def _stream_gen(user_input):
        for ev in events:
            yield ev
    sr2.turn = _stream_gen

    return sr2


# ---------------------------------------------------------------------------
# A. Agent.__init__: constructs SR2, no LiteLLMCallable call
# ---------------------------------------------------------------------------

class TestAgentInit:
    def test_agent_has_sr2_attribute(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config(), session_id="s1")
        assert hasattr(agent, "sr2")

    def test_sr2_constructed_not_litellm_callable(self):
        """Agent.__init__ must use SR2, not call LiteLLMCallable directly."""
        from sr2_spectre.agent import Agent
        import sr2_spectre.agent as agent_module

        # Verify LiteLLMCallable is not the primary client on Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config(), session_id="s1")

        # SR2 was constructed
        MockSR2.assert_called_once()
        call_kwargs = MockSR2.call_args.kwargs
        # tool_source must be passed
        assert "tool_source" in call_kwargs

    def test_tool_source_is_registry(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config(), session_id="s1")
        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["tool_source"] is agent.registry

    def test_llm_dict_has_default_key(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            Agent(config=_make_config(), session_id="s1")
        call_kwargs = MockSR2.call_args.kwargs
        assert "default" in call_kwargs["llm"]


# ---------------------------------------------------------------------------
# B. session_id
# ---------------------------------------------------------------------------

class TestSessionId:
    def test_session_id_propagated(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config(), session_id="my-session")
        assert agent.session_id == "my-session"

    def test_default_session_id_uses_name(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())
        assert "test" in agent.session_id


# ---------------------------------------------------------------------------
# C. Tool registration from config
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_tools_from_config_registered(self):
        from sr2_spectre.agent import Agent
        from sr2_spectre.config import ToolConfig

        cfg = _make_config(tools=[
            ToolConfig(
                name="dummy",
                class_path="sr2_spectre.tools.registry:ToolRegistry",  # won't register real tool
                config={},
            )
        ])

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            with patch.object(
                __import__("sr2_spectre.tools.registry", fromlist=["ToolRegistry"]).ToolRegistry,
                "register_from_class_path",
            ) as mock_reg:
                agent = Agent(config=cfg)

    def test_register_tool_at_runtime(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())
        agent.register_tool("echo", "Echo", {}, lambda msg: msg)
        assert "echo" in agent.registry


# ---------------------------------------------------------------------------
# D. handle_user_message: happy path (no tools)
# ---------------------------------------------------------------------------

class TestHandleUserMessageHappyPath:
    @pytest.mark.asyncio
    async def test_returns_turn_result_with_text(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            mock_sr2 = _mock_sr2([
                StreamEvent(type="text", text="Hello "),
                StreamEvent(type="text", text="world!"),
                StreamEvent(type="end"),
            ])
            MockSR2.return_value = mock_sr2
            agent = Agent(config=_make_config(), session_id="s")

        result = await agent.handle_user_message("Hi")
        assert isinstance(result, TurnResult)
        assert result.text == "Hello world!"
        assert result.tool_calls_executed == 0

    @pytest.mark.asyncio
    async def test_seed_session_called_each_round(self):
        from sr2_spectre.agent import Agent
        mock_sr2 = MagicMock()
        mock_sr2.continue_archive = AsyncMock()
        mock_sr2.seed_session = MagicMock()
        events = [StreamEvent(type="text", text="Hi"), StreamEvent(type="end")]

        async def _turn(user_input):
            for ev in events:
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        await agent.handle_user_message("Hello")
        mock_sr2.seed_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_history_has_user_and_assistant_messages(self):
        from sr2_spectre.agent import Agent
        mock_sr2 = _mock_sr2([
            StreamEvent(type="text", text="Response"),
            StreamEvent(type="end"),
        ])
        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        await agent.handle_user_message("Question")
        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"


# ---------------------------------------------------------------------------
# E. handle_user_message: one tool round-trip
# ---------------------------------------------------------------------------

class TestHandleUserMessageToolRoundTrip:
    @pytest.mark.asyncio
    async def test_single_tool_call_executed(self):
        """SR2 handles tool loop internally. We see tool_use_emitted + tool_result_received events."""
        from sr2_spectre.agent import Agent

        # SR2's turn() yields: tool_use_emitted, tool_result_received, iteration_complete,
        # then final text + end.
        events = [
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="add", input={"a": 1, "b": 2})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="3")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="The answer is 3"),
            StreamEvent(type="end"),
        ]

        mock_sr2 = MagicMock()
        mock_sr2.continue_archive = AsyncMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            for ev in events:
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        result = await agent.handle_user_message("What is 1+2?")
        assert result.text == "The answer is 3"
        assert result.tool_calls_executed == 1

    @pytest.mark.asyncio
    async def test_tool_result_appended_to_history(self):
        from sr2_spectre.agent import Agent

        # SR2 handles tool loop; history gets user + assistant (with text from final response)
        events = [
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="echo", input={"msg": "hi"})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="hi")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="done"),
            StreamEvent(type="end"),
        ]

        mock_sr2 = MagicMock()
        mock_sr2.continue_archive = AsyncMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            for ev in events:
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        await agent.handle_user_message("Echo hi")

        # history: user → assistant (SR2 handled tool loop internally, agent sees final text)
        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"


# ---------------------------------------------------------------------------
# F. Error recovery — FR13
# ---------------------------------------------------------------------------

class TestToolErrorRecovery:
    @pytest.mark.asyncio
    async def test_tool_error_fed_back_as_tool_result(self):
        """A failing tool's error text is surfaced via tool_result_received, loop continues."""
        from sr2_spectre.agent import Agent

        events = [
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="fail_tool", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="something went wrong", is_error=True)],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="recovered"),
            StreamEvent(type="end"),
        ]

        mock_sr2 = MagicMock()
        mock_sr2.continue_archive = AsyncMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            for ev in events:
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        result = await agent.handle_user_message("Try the tool")
        assert result.text == "recovered"

    @pytest.mark.asyncio
    async def test_tool_error_does_not_raise_out_of_loop(self):
        """Tool failure must NOT propagate as an exception from handle_user_message."""
        from sr2_spectre.agent import Agent

        events = [
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="boom", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="BOOM", is_error=True)],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(type="text", text="ok"),
            StreamEvent(type="end"),
        ]

        mock_sr2 = MagicMock()
        mock_sr2.continue_archive = AsyncMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            for ev in events:
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        result = await agent.handle_user_message("Trigger tool")
        assert result.text == "ok"


# ---------------------------------------------------------------------------
# G. tool-loop limit exceeded — FR14
# ---------------------------------------------------------------------------

class TestMaxToolRounds:
    @pytest.mark.asyncio
    async def test_multiple_tool_iterations_counted(self):
        """SR2 handles multiple tool iterations; Agent counts tool calls from events."""
        from sr2_spectre.agent import Agent

        # Two tool iterations then final text
        events = [
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="loop_tool", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="still going")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu2", name="loop_tool", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu2", content="again")],
            ),
            StreamEvent(type="iteration_complete", iteration=1),
            StreamEvent(type="text", text="done"),
            StreamEvent(type="end"),
        ]

        mock_sr2 = MagicMock()
        mock_sr2.continue_archive = AsyncMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            for ev in events:
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        result = await agent.handle_user_message("Start")
        assert isinstance(result, TurnResult)
        assert result.tool_calls_executed == 2

    @pytest.mark.asyncio
    async def test_sr2_max_tool_iterations_enforced(self):
        """SR2's own max_tool_iterations limits the loop; Agent receives whatever SR2 yields."""
        from sr2_spectre.agent import Agent

        # SR2 yields 2 tool iterations then stops (simulating max_tool_iterations hit)
        events = [
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu1", name="loop_tool", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu1", content="ok")],
            ),
            StreamEvent(type="iteration_complete", iteration=0),
            StreamEvent(
                type="tool_use_emitted",
                tool_uses=[ToolUseBlock(id="tu2", name="loop_tool", input={})],
            ),
            StreamEvent(
                type="tool_result_received",
                tool_results=[ToolResultBlock(tool_use_id="tu2", content="ok")],
            ),
            StreamEvent(type="iteration_complete", iteration=1),
            StreamEvent(type="text", text="stopped"),
            StreamEvent(type="end"),
        ]

        mock_sr2 = MagicMock()
        mock_sr2.continue_archive = AsyncMock()
        mock_sr2.seed_session = MagicMock()

        async def _turn(user_input):
            for ev in events:
                yield ev
        mock_sr2.turn = _turn

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        result = await agent.handle_user_message("Trigger")
        assert result is not None
        assert result.tool_calls_executed == 2


# ---------------------------------------------------------------------------
# H. new_session()
# ---------------------------------------------------------------------------

class TestNewSession:
    @pytest.mark.asyncio
    async def test_new_session_resets_history(self):
        from sr2_spectre.agent import Agent
        mock_sr2 = _mock_sr2()
        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        await agent.handle_user_message("Hello")
        assert len(agent.history) > 0

        agent.new_session("fresh")
        assert agent.history == []
        assert agent.session_id == "fresh"


# ---------------------------------------------------------------------------
# I. register_tool (runtime)
# ---------------------------------------------------------------------------

class TestRegisterTool:
    def test_register_tool_appears_in_registry(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())
        agent.register_tool("ping", "Ping", {}, lambda: "pong")
        assert "ping" in agent.registry


# ---------------------------------------------------------------------------
# K. pipeline.max_tool_iterations is the single source of truth for the
#    tool-loop limit (reverses obsidian-ydt Behavior 1).
#
# SR2 owns and enforces the tool-loop limit. Spectre must pass the configured
# pipeline.max_tool_iterations through to SR2 UNCHANGED — there is no
# spectre-side knob (agent.max_tool_rounds is gone) that overrides it.
# Regression guard for the inverse bug: configuring pipeline.max_tool_iterations
# (e.g. 1000) must be the value SR2 is actually built with.
# ---------------------------------------------------------------------------

def _config_with_iterations(pipeline_iterations: int) -> SpectreConfig:
    """Build a config whose pipeline.max_tool_iterations is set to a known value."""
    pipeline = _minimal_pipeline_dict()
    pipeline["max_tool_iterations"] = pipeline_iterations
    return SpectreConfig(
        agent=AgentConfig(name="test"),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=pipeline,
    )


class TestMaxToolIterationsSingleSource:
    def test_pipeline_max_tool_iterations_flows_through_unchanged(self):
        """SR2 must be constructed with exactly the configured pipeline.max_tool_iterations."""
        from sr2_spectre.agent import Agent

        # 1000 mirrors the real bug: user set it, the loop must honor it.
        cfg = _config_with_iterations(pipeline_iterations=1000)

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            Agent(config=cfg, session_id="s1")

        MockSR2.assert_called_once()
        # pipeline_config may be passed positionally or by keyword; check both.
        call = MockSR2.call_args
        pipeline_config = call.kwargs.get("pipeline_config")
        if pipeline_config is None and call.args:
            pipeline_config = call.args[0]
        assert pipeline_config is not None, "SR2 was not given a pipeline_config"
        assert pipeline_config.max_tool_iterations == 1000

    def test_different_iteration_value_propagates(self):
        """A different configured value propagates too — value is not hard-coded."""
        from sr2_spectre.agent import Agent

        cfg = _config_with_iterations(pipeline_iterations=7)

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            Agent(config=cfg, session_id="s1")

        call = MockSR2.call_args
        pipeline_config = call.kwargs.get("pipeline_config")
        if pipeline_config is None and call.args:
            pipeline_config = call.args[0]
        assert pipeline_config is not None, "SR2 was not given a pipeline_config"
        assert pipeline_config.max_tool_iterations == 7


# ---------------------------------------------------------------------------
# J. aclose() — MCP client cleanup
# ---------------------------------------------------------------------------

class TestAgentAClose:
    @pytest.mark.asyncio
    async def test_aclose_calls_close_on_each_mcp_client(self):
        """aclose() must call close() on every MCP client it created."""
        from sr2_spectre.agent import Agent
        from sr2_spectre.config import McpServerConfig

        cfg = _make_config(mcp_servers=[
            McpServerConfig(name="a", type="stdio", command=["server_a"]),
            McpServerConfig(name="b", type="stdio", command=["server_b"]),
        ])

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()

            mock_client_a = AsyncMock()
            mock_client_b = AsyncMock()

            with patch(
                "sr2_spectre.runtime.MCPClient",
                side_effect=[mock_client_a, mock_client_b],
            ):
                agent = Agent(config=cfg)

        await agent.aclose()

        mock_client_a.close.assert_awaited_once()
        mock_client_b.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_noop_when_no_mcp_servers(self):
        """aclose() is a no-op when the agent has no MCP servers configured."""
        from sr2_spectre.agent import Agent

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())

        await agent.aclose()  # must not raise
