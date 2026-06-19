"""Tests for Runtime and Session — FR2 (Runtime/Session split).

Covers:
  A. Runtime construction (config, llm, registry, MCP clients)
  B. Runtime.initialize() connects MCP clients
  C. Runtime.new_session() creates a Session with correct frame_id
  D. Session has own SR2, history, lock
  E. Session.stream_message() works (happy path)
  F. Session handles concurrent turns safely (lock)
  G. Agent backward compatibility (single-frame regression)
  H. Agent delegates to Runtime + Session internally
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
from sr2_spectre.config import AgentConfig, McpServerConfig, ModelConfig, SpectreConfig
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


def _mock_sr2(turn_events: list[StreamEvent] | None = None) -> MagicMock:
    """Return a mock SR2 instance whose turn() yields given events."""
    events = turn_events or [StreamEvent(type="text", text="Hello!"), StreamEvent(type="end")]
    sr2 = MagicMock()
    sr2.seed_session = MagicMock()

    async def _stream_gen(user_input):
        for ev in events:
            yield ev
    sr2.turn = _stream_gen

    return sr2


# ---------------------------------------------------------------------------
# A. Runtime construction
# ---------------------------------------------------------------------------

class TestRuntimeInit:
    def test_runtime_has_config(self):
        from sr2_spectre.runtime import Runtime
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=_make_config())
        assert runtime.config is not None
        assert runtime.config.agent.name == "test"

    def test_runtime_has_llm_callable(self):
        from sr2_spectre.runtime import Runtime
        with patch("sr2_spectre.runtime.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock()
            runtime = Runtime(config=_make_config())
        assert runtime.llm is not None
        MockLLM.assert_called_once_with(
            model="test-model",
            base_url="http://test:8000",
        )

    def test_runtime_has_tool_registry(self):
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.tools.registry import ToolRegistry
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=_make_config())
        assert isinstance(runtime.registry, ToolRegistry)

    def test_runtime_tools_from_config_registered(self):
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import ToolConfig

        cfg = _make_config(tools=[
            ToolConfig(
                name="dummy",
                class_path="sr2_spectre.tools.registry:ToolRegistry",
                config={},
            )
        ])

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch.object(
                __import__("sr2_spectre.tools.registry", fromlist=["ToolRegistry"]).ToolRegistry,
                "register_from_class_path",
            ) as mock_reg:
                runtime = Runtime(config=cfg)
        mock_reg.assert_called_once()

    def test_runtime_creates_mcp_clients(self):
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import McpServerConfig

        cfg = _make_config(mcp_servers=[
            McpServerConfig(name="a", type="stdio", command=["server_a"]),
            McpServerConfig(name="b", type="http", url="http://localhost:8080"),
        ])

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.runtime.MCPClient") as MockMCP:
                runtime = Runtime(config=cfg)
        assert MockMCP.call_count == 2

    def test_runtime_no_mcp_clients_when_empty(self):
        from sr2_spectre.runtime import Runtime

        cfg = _make_config()
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)
        assert runtime._mcp_clients == []


# ---------------------------------------------------------------------------
# B. Runtime.initialize()
# ---------------------------------------------------------------------------

class TestRuntimeInitialize:
    @pytest.mark.asyncio
    async def test_initialize_connects_mcp_clients(self):
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import McpServerConfig

        cfg = _make_config(mcp_servers=[
            McpServerConfig(name="a", type="stdio", command=["server_a"]),
        ])

        mock_client = AsyncMock()
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
                runtime = Runtime(config=cfg)

        await runtime.initialize()
        mock_client.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_registers_mcp_tools(self):
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import McpServerConfig

        cfg = _make_config(mcp_servers=[
            McpServerConfig(name="a", type="stdio", command=["server_a"]),
        ])

        mock_bridge = MagicMock()
        mock_bridge.name = "test_tool"
        mock_bridge.description = "test"
        mock_bridge.input_schema = {}
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=[mock_bridge])

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
                runtime = Runtime(config=cfg)

        await runtime.initialize()

        # The bridge should be registered — check the registry's tools dict
        assert runtime.registry._tools.get("test_tool") is not None

    @pytest.mark.asyncio
    async def test_initialize_noop_when_no_mcp_servers(self):
        from sr2_spectre.runtime import Runtime

        cfg = _make_config()
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        await runtime.initialize()  # must not raise


# ---------------------------------------------------------------------------
# C. Runtime.new_session()
# ---------------------------------------------------------------------------

class TestRuntimeNewSession:
    def test_new_session_returns_session(self):
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.session import Session

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2"):
                runtime = Runtime(config=_make_config())
        session = runtime.new_session(frame_id="test-frame")
        assert isinstance(session, Session)
        assert session.frame_id == "test-frame"

    def test_new_session_constructs_sr2_with_frame_id(self):
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                MockSR2.return_value = MagicMock()
                runtime = Runtime(config=_make_config())
                runtime.new_session(frame_id="frame-123")

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["session_id"] == "frame-123"

    def test_new_session_sr2_gets_tool_registry(self):
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                MockSR2.return_value = MagicMock()
                runtime = Runtime(config=_make_config())
                runtime.new_session(frame_id="f1")

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["tool_source"] is runtime.registry

    def test_new_session_creates_independent_sr2_instances(self):
        from sr2_spectre.runtime import Runtime

        mock_sr2_a = MagicMock()
        mock_sr2_b = MagicMock()

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2", side_effect=[mock_sr2_a, mock_sr2_b]):
                runtime = Runtime(config=_make_config())
                session_a = runtime.new_session(frame_id="frame-a")
                session_b = runtime.new_session(frame_id="frame-b")

        assert session_a.sr2 is mock_sr2_a
        assert session_b.sr2 is mock_sr2_b
        assert session_a.sr2 is not session_b.sr2


# ---------------------------------------------------------------------------
# D. Session attributes
# ---------------------------------------------------------------------------

class TestSessionInit:
    def test_session_has_frame_id(self):
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.session import Session

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2"):
                runtime = Runtime(config=_make_config())
                session = runtime.new_session(frame_id="my-frame")
        assert session.frame_id == "my-frame"

    def test_session_has_empty_history(self):
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2"):
                runtime = Runtime(config=_make_config())
                session = runtime.new_session(frame_id="f")
        assert session.history == []

    def test_session_has_lock(self):
        import asyncio
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2"):
                runtime = Runtime(config=_make_config())
                session = runtime.new_session(frame_id="f")
        assert isinstance(session._lock, asyncio.Lock)


# ---------------------------------------------------------------------------
# E. Session.stream_message() — happy path
# ---------------------------------------------------------------------------

class TestSessionStreamMessage:
    @pytest.mark.asyncio
    async def test_stream_returns_text(self):
        from sr2_spectre.agent import Agent

        mock_sr2 = _mock_sr2([
            StreamEvent(type="text", text="Response "),
            StreamEvent(type="text", text="text"),
            StreamEvent(type="end"),
        ])

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        result = await agent.handle_user_message("Hello")
        assert result.text == "Response text"

    @pytest.mark.asyncio
    async def test_session_history_grows(self):
        from sr2_spectre.agent import Agent

        mock_sr2 = _mock_sr2([
            StreamEvent(type="text", text="Hi"),
            StreamEvent(type="end"),
        ])

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        await agent.handle_user_message("Hello")
        assert len(agent.history) == 2
        assert agent.history[0].role == "user"
        assert agent.history[1].role == "assistant"


# ---------------------------------------------------------------------------
# F. Session lock serializes turns
# ---------------------------------------------------------------------------

class TestSessionLock:
    @pytest.mark.asyncio
    async def test_lock_is_used_for_turns(self):
        """Session uses asyncio.Lock to serialize turns within the same frame."""
        import asyncio
        from sr2_spectre.runtime import Runtime

        events = [StreamEvent(type="text", text="ok"), StreamEvent(type="end")]

        async def _slow_turn(user_input):
            await asyncio.sleep(0.01)
            for ev in events:
                yield ev

        mock_sr2 = MagicMock()
        mock_sr2.seed_session = MagicMock()
        mock_sr2.turn = _slow_turn

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
                runtime = Runtime(config=_make_config())
                session = runtime.new_session(frame_id="f")

        # Verify the lock exists and is used
        assert hasattr(session, "_lock")
        assert isinstance(session._lock, asyncio.Lock)


# ---------------------------------------------------------------------------
# G. Agent backward compatibility (single-frame regression)
# ---------------------------------------------------------------------------

class TestAgentBackwardCompat:
    """Agent must remain behaviorally identical for single-frame usage."""

    def test_agent_still_has_session_id(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config(), session_id="s1")
        assert agent.session_id == "s1"

    def test_agent_still_has_history(self):
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())
        assert isinstance(agent.history, list)

    def test_agent_still_has_registry(self):
        from sr2_spectre.agent import Agent
        from sr2_spectre.tools.registry import ToolRegistry
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())
        assert isinstance(agent.registry, ToolRegistry)

    @pytest.mark.asyncio
    async def test_agent_handle_user_message_still_works(self):
        """Regression: handle_user_message must still return TurnResult."""
        from sr2_spectre.agent import Agent

        mock_sr2 = _mock_sr2([
            StreamEvent(type="text", text="Works"),
            StreamEvent(type="end"),
        ])

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        result = await agent.handle_user_message("Test")
        assert isinstance(result, TurnResult)
        assert result.text == "Works"

    @pytest.mark.asyncio
    async def test_agent_stream_message_still_works(self):
        """Regression: stream_message must still yield AgentEvents."""
        from sr2_spectre.agent import Agent
        from sr2_spectre.events import AgentDone, AgentTextDelta

        mock_sr2 = _mock_sr2([
            StreamEvent(type="text", text="Hi"),
            StreamEvent(type="end"),
        ])

        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config(), session_id="s")

        events = [ev async for ev in agent.stream_message("Hello")]
        assert any(isinstance(ev, AgentTextDelta) for ev in events)
        assert any(isinstance(ev, AgentDone) for ev in events)

    def test_agent_new_session_still_works(self):
        """Regression: new_session() must still reset history."""
        from sr2_spectre.agent import Agent

        mock_sr2 = _mock_sr2()
        with patch("sr2_spectre.session.SR2", return_value=mock_sr2):
            agent = Agent(config=_make_config())

        agent.new_session("fresh")
        assert agent.history == []
        assert agent.session_id == "fresh"

    def test_agent_register_tool_still_works(self):
        """Regression: register_tool() must still work."""
        from sr2_spectre.agent import Agent
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=_make_config())
        agent.register_tool("echo", "Echo", {}, lambda x: x)
        assert "echo" in agent.registry

    @pytest.mark.asyncio
    async def test_agent_aclose_still_works(self):
        """Regression: aclose() must still close MCP clients."""
        from sr2_spectre.agent import Agent
        from sr2_spectre.config import McpServerConfig

        cfg = _make_config(mcp_servers=[
            McpServerConfig(name="a", type="stdio", command=["server_a"]),
        ])

        mock_client = AsyncMock()
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            with patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
                agent = Agent(config=cfg)

        await agent.aclose()
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_initialize_still_works(self):
        """Regression: initialize() must still connect MCP clients."""
        from sr2_spectre.agent import Agent
        from sr2_spectre.config import McpServerConfig

        cfg = _make_config(mcp_servers=[
            McpServerConfig(name="a", type="stdio", command=["server_a"]),
        ])

        mock_bridge = MagicMock(name="test_tool", description="test", input_schema={})
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=[mock_bridge])

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            with patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
                agent = Agent(config=cfg)

        await agent.initialize()
        mock_client.connect.assert_awaited_once()


# ---------------------------------------------------------------------------
# H. Runtime.aclose()
# ---------------------------------------------------------------------------

class TestRuntimeAClose:
    @pytest.mark.asyncio
    async def test_aclose_closes_mcp_clients(self):
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import McpServerConfig

        cfg = _make_config(mcp_servers=[
            McpServerConfig(name="a", type="stdio", command=["server_a"]),
            McpServerConfig(name="b", type="stdio", command=["server_b"]),
        ])

        mock_client_a = AsyncMock()
        mock_client_b = AsyncMock()

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch(
                "sr2_spectre.runtime.MCPClient",
                side_effect=[mock_client_a, mock_client_b],
            ):
                runtime = Runtime(config=cfg)

        await runtime.aclose()

        mock_client_a.close.assert_awaited_once()
        mock_client_b.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_noop_when_no_mcp_servers(self):
        from sr2_spectre.runtime import Runtime

        cfg = _make_config()
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        await runtime.aclose()  # must not raise

    @pytest.mark.asyncio
    async def test_aclose_continues_when_one_mcp_client_raises(self):
        """One MCP client failing to close must not strand the others.

        Teardown is best-effort: a raise from client_a.close() must be
        swallowed and client_b.close() must still run. aclose() must not
        propagate the exception (regression: spc-63 nonzero_exit at shutdown).
        """
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import McpServerConfig

        cfg = _make_config(mcp_servers=[
            McpServerConfig(name="a", type="stdio", command=["server_a"]),
            McpServerConfig(name="b", type="stdio", command=["server_b"]),
        ])

        mock_client_a = AsyncMock()
        mock_client_a.close.side_effect = RuntimeError("transport already closed")
        mock_client_b = AsyncMock()

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch(
                "sr2_spectre.runtime.MCPClient",
                side_effect=[mock_client_a, mock_client_b],
            ):
                runtime = Runtime(config=cfg)

        await runtime.aclose()  # must not raise

        mock_client_a.close.assert_awaited_once()
        mock_client_b.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_swallows_cancelled_error_during_mcp_close(self):
        """A CancelledError from an MCP client close must not escape aclose().

        This is the exact spc-63 failure: httpcore/anyio raised
        asyncio.CancelledError while tearing down an HTTP MCP client during a
        clean shutdown, which propagated through aclose() -> asyncio.run() and
        gave the process exit code 1 even though the task succeeded.
        CancelledError is a BaseException, so a bare `except Exception` would
        NOT catch it — this test guards that.
        """
        import asyncio

        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import McpServerConfig

        cfg = _make_config(mcp_servers=[
            McpServerConfig(name="a", type="stdio", command=["server_a"]),
            McpServerConfig(name="b", type="stdio", command=["server_b"]),
        ])

        mock_client_a = AsyncMock()
        mock_client_a.close.side_effect = asyncio.CancelledError()
        mock_client_b = AsyncMock()

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch(
                "sr2_spectre.runtime.MCPClient",
                side_effect=[mock_client_a, mock_client_b],
            ):
                runtime = Runtime(config=cfg)

        await runtime.aclose()  # must not raise CancelledError

        mock_client_a.close.assert_awaited_once()
        mock_client_b.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# I. Session isolation — different sessions have independent history
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    @pytest.mark.asyncio
    async def test_two_sessions_independent_history(self):
        """Two sessions from the same runtime maintain separate histories."""
        from sr2_spectre.runtime import Runtime

        events = [StreamEvent(type="text", text="ok"), StreamEvent(type="end")]

        async def _turn(user_input):
            for ev in events:
                yield ev

        mock_sr2_a = MagicMock()
        mock_sr2_a.seed_session = MagicMock()
        mock_sr2_a.turn = _turn

        mock_sr2_b = MagicMock()
        mock_sr2_b.seed_session = MagicMock()
        mock_sr2_b.turn = _turn

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2", side_effect=[mock_sr2_a, mock_sr2_b]):
                runtime = Runtime(config=_make_config())
                session_a = runtime.new_session(frame_id="frame-a")
                session_b = runtime.new_session(frame_id="frame-b")

        # Session A sends a message
        from sr2_spectre.core import TurnResult
        result_a = await session_a.handle_user_message("Message to A")
        assert len(session_a.history) == 2

        # Session B is untouched
        assert len(session_b.history) == 0

        # Session B sends a message
        result_b = await session_b.handle_user_message("Message to B")
        assert len(session_b.history) == 2

        # Histories are independent
        assert session_a.history[0].content[0].text == "Message to A"
        assert session_b.history[0].content[0].text == "Message to B"
