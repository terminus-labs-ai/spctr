"""Tests for MCP config models and Agent MCP wiring.

Covers:
  Config:
    1. McpServerConfig parses a stdio server config
    2. McpServerConfig parses an http server config
    3. McpServerConfig defaults (args, env, url, command)
    4. AgentConfig accepts mcp_servers list
    5. SpectreConfig round-trips through load_config() with mcp_servers

  Agent wiring:
    6. Agent.__init__ with one stdio mcp_server creates one MCPClient
    7. Agent.__init__ with one http mcp_server creates one MCPClient
    8. Agent.__init__ with no mcp_servers never constructs an MCPClient
    9. Agent.initialize() connects each MCPClient and registers bridges into self.registry
   10. Agent.initialize() with a failing server logs a warning and continues
   11. Agent.initialize() with no mcp_servers completes without error

  Smoke (live server):
   12. beads-mcp stdio server starts, returns tools, MCPClient connects
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import shutil

from sr2_spectre.config import AgentConfig, McpServerConfig, ModelConfig, SpectreConfig


def _which(name: str) -> bool:
    """Check if a binary is available on PATH or in the project .venv."""
    if shutil.which(name) is not None:
        return True
    # Check the venv bin directory (for tests run via venv)
    import os, sys
    # sys.executable lives inside .venv/bin/python, so its parent is .venv/bin
    venv_bin = os.path.dirname(sys.executable)
    return os.path.isfile(os.path.join(venv_bin, name))


# ---------------------------------------------------------------------------
# Shared test helpers (mirrors test_agent.py conventions)
# ---------------------------------------------------------------------------

def _minimal_pipeline_dict() -> dict:
    return {
        "layers": [
            {
                "name": "system",
                "target": "system",
                "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
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
    }


def _make_config(**agent_kwargs) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test", **agent_kwargs),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=_minimal_pipeline_dict(),
    )


def _make_mock_bridge(name: str) -> MagicMock:
    """Return a mock MCPToolBridge with the given tool name."""
    bridge = MagicMock()
    bridge.name = name
    bridge.description = f"Tool {name}"
    bridge.input_schema = {"type": "object", "properties": {}}
    # Make asyncio.iscoroutinefunction() return True so ToolRegistry sets is_async=True
    bridge._is_coroutine = AsyncMock._is_coroutine
    return bridge


# ---------------------------------------------------------------------------
# 1-3. McpServerConfig
# ---------------------------------------------------------------------------

class TestMcpServerConfig:
    def test_stdio_server_config(self):
        """Req 1: McpServerConfig parses a stdio server config."""
        cfg = McpServerConfig(
            name="my-stdio",
            type="stdio",
            command=["python", "-m", "my_server"],
            args=["--debug"],
            env={"MY_VAR": "value"},
        )
        assert cfg.name == "my-stdio"
        assert cfg.type == "stdio"
        assert cfg.command == ["python", "-m", "my_server"]
        assert cfg.args == ["--debug"]
        assert cfg.env == {"MY_VAR": "value"}

    def test_http_server_config(self):
        """Req 2: McpServerConfig parses an http server config."""
        cfg = McpServerConfig(
            name="my-http",
            type="http",
            url="http://localhost:8080/sse",
        )
        assert cfg.name == "my-http"
        assert cfg.type == "http"
        assert cfg.url == "http://localhost:8080/sse"

    def test_defaults(self):
        """Req 3: McpServerConfig defaults: args=[], env={}, url='', command=[]."""
        cfg = McpServerConfig(name="minimal", type="stdio")
        assert cfg.args == []
        assert cfg.env == {}
        assert cfg.url == ""
        assert cfg.command == []


# ---------------------------------------------------------------------------
# 4. AgentConfig accepts mcp_servers
# ---------------------------------------------------------------------------

class TestAgentConfigMcpServers:
    def test_accepts_mcp_servers_list(self):
        """Req 4: AgentConfig accepts mcp_servers list of McpServerConfig."""
        cfg = AgentConfig(
            mcp_servers=[
                McpServerConfig(name="srv1", type="stdio", command=["my-server"]),
                McpServerConfig(name="srv2", type="http", url="http://host/sse"),
            ]
        )
        assert len(cfg.mcp_servers) == 2
        assert cfg.mcp_servers[0].name == "srv1"
        assert cfg.mcp_servers[1].name == "srv2"

    def test_mcp_servers_default_empty(self):
        """mcp_servers defaults to empty list when not supplied."""
        cfg = AgentConfig()
        assert cfg.mcp_servers == []


# ---------------------------------------------------------------------------
# 5. SpectreConfig round-trips through load_config() with mcp_servers
# ---------------------------------------------------------------------------

class TestLoadConfigWithMcpServers:
    def test_load_config_with_mcp_servers(self, tmp_path):
        """Req 5: SpectreConfig round-trips through load_config() with mcp_servers."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "agent:\n"
            "  name: wired-agent\n"
            "  mcp_servers:\n"
            "    - name: glyph\n"
            "      type: stdio\n"
            "      command: [python, -m, glyph_server]\n"
            "      args: [--port, '9000']\n"
            "      env:\n"
            "        GLYPH_DB: /data/glyph.db\n"
            "    - name: galaxy\n"
            "      type: http\n"
            "      url: http://localhost:7777/sse\n"
            "\n"
            "models:\n"
            "  default:\n"
            "    model: openai/qwen3:27b\n"
            "    base_url: http://localhost:11438/v1\n"
            "\n"
            "pipeline:\n"
            "  layers:\n"
            "    - name: system\n"
            "      target: system\n"
            "      resolvers:\n"
            "        - type: static\n"
            "          config:\n"
            "            text: You are a helpful assistant.\n"
            "    - name: conversation\n"
            "      target: messages\n"
            "      resolvers:\n"
            "        - type: session\n"
            "        - type: input\n"
        )

        from sr2_spectre.config import load_config
        cfg = load_config(str(config_file))

        assert cfg.agent.name == "wired-agent"
        assert len(cfg.agent.mcp_servers) == 2

        stdio_srv = cfg.agent.mcp_servers[0]
        assert stdio_srv.name == "glyph"
        assert stdio_srv.type == "stdio"
        assert stdio_srv.command == ["python", "-m", "glyph_server"]
        assert stdio_srv.args == ["--port", "9000"]
        assert stdio_srv.env == {"GLYPH_DB": "/data/glyph.db"}

        http_srv = cfg.agent.mcp_servers[1]
        assert http_srv.name == "galaxy"
        assert http_srv.type == "http"
        assert http_srv.url == "http://localhost:7777/sse"


# ---------------------------------------------------------------------------
# 6-8. Agent.__init__ MCPClient instantiation
# ---------------------------------------------------------------------------

class TestAgentInitMcpClients:
    def test_stdio_mcp_server_creates_one_client(self):
        """Req 6: __init__ with one stdio mcp_server config creates one MCPClient."""
        from sr2_spectre.agent import Agent

        cfg = _make_config(
            mcp_servers=[
                McpServerConfig(
                    name="stdio-srv",
                    type="stdio",
                    command=["my-server"],
                    args=["--flag"],
                    env={"K": "V"},
                )
            ]
        )

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient") as MockMCPClient:
            MockSR2.return_value = MagicMock()
            MockMCPClient.return_value = MagicMock()

            agent = Agent(config=cfg)

        MockMCPClient.assert_called_once_with(
            server_type="stdio",
            command=["my-server"],
            args=["--flag"],
            env={"K": "V"},
        )

    def test_http_mcp_server_creates_one_client(self):
        """Req 7: __init__ with one http mcp_server config creates one MCPClient."""
        from sr2_spectre.agent import Agent

        cfg = _make_config(
            mcp_servers=[
                McpServerConfig(
                    name="http-srv",
                    type="http",
                    url="http://localhost:9000/sse",
                )
            ]
        )

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient") as MockMCPClient:
            MockSR2.return_value = MagicMock()
            MockMCPClient.return_value = MagicMock()

            agent = Agent(config=cfg)

        MockMCPClient.assert_called_once_with(
            server_type="http",
            url="http://localhost:9000/sse",
        )

    def test_no_mcp_servers_creates_no_clients(self):
        """Req 8: __init__ with no mcp_servers never constructs an MCPClient."""
        from sr2_spectre.agent import Agent

        cfg = _make_config()  # no mcp_servers

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient") as MockMCPClient:
            MockSR2.return_value = MagicMock()

            agent = Agent(config=cfg)

        MockMCPClient.assert_not_called()


# ---------------------------------------------------------------------------
# 9-11. Agent.initialize()
# ---------------------------------------------------------------------------

class TestAgentInitialize:
    async def test_initialize_connects_client_and_registers_bridges(self):
        """Req 9: initialize() connects each MCPClient and registers bridges into self.registry."""
        from sr2_spectre.agent import Agent

        bridge = _make_mock_bridge("list_files")
        mock_client = MagicMock()
        mock_client.connect = AsyncMock(return_value=[bridge])

        cfg = _make_config(
            mcp_servers=[
                McpServerConfig(name="fs-server", type="stdio", command=["fs-server"])
            ]
        )

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        await agent.initialize()

        mock_client.connect.assert_awaited_once()
        assert "list_files" in agent.registry

    async def test_initialize_failing_server_logs_warning_and_continues(self, caplog):
        """Req 10: initialize() with a failing server logs a warning and the remaining server still connects."""
        from sr2_spectre.agent import Agent
        from sr2_spectre.mcp.client import MCPConnectionError

        bridge = _make_mock_bridge("working_tool")

        failing_client = MagicMock()
        failing_client.connect = AsyncMock(side_effect=MCPConnectionError("refused"))

        ok_client = MagicMock()
        ok_client.connect = AsyncMock(return_value=[bridge])

        cfg = _make_config(
            mcp_servers=[
                McpServerConfig(name="bad-server", type="stdio", command=["bad"]),
                McpServerConfig(name="ok-server", type="stdio", command=["ok"]),
            ]
        )

        client_instances = [failing_client, ok_client]
        client_iter = iter(client_instances)

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient", side_effect=lambda **kw: next(client_iter)):
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        with caplog.at_level(logging.WARNING, logger="sr2_spectre.agent"):
            await agent.initialize()

        # Warning was logged for the failing server
        assert any("bad-server" in record.message or "refused" in record.message
                   for record in caplog.records
                   if record.levelno >= logging.WARNING)

        # The working server's tool was still registered
        assert "working_tool" in agent.registry

    async def test_initialize_no_mcp_servers_is_noop(self):
        """Req 11: initialize() with no mcp_servers completes without error."""
        from sr2_spectre.agent import Agent

        cfg = _make_config()  # no mcp_servers

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        # Must not raise
        await agent.initialize()

        # Only load_skill (auto-injected) should be present; no MCP tools added
        assert "load_skill" in agent.registry
        assert len(agent.registry) == 1

    async def test_initialize_multiple_bridges_all_registered(self):
        """initialize() registers ALL bridges returned by a single client."""
        from sr2_spectre.agent import Agent

        bridge_a = _make_mock_bridge("tool_alpha")
        bridge_b = _make_mock_bridge("tool_beta")

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(return_value=[bridge_a, bridge_b])

        cfg = _make_config(
            mcp_servers=[
                McpServerConfig(name="multi-server", type="http", url="http://host/sse")
            ]
        )

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        await agent.initialize()

        assert "tool_alpha" in agent.registry
        assert "tool_beta" in agent.registry

    async def test_initialize_does_not_raise_on_mcp_connection_error(self):
        """MCPConnectionError must never propagate out of initialize()."""
        from sr2_spectre.agent import Agent
        from sr2_spectre.mcp.client import MCPConnectionError

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(side_effect=MCPConnectionError("timeout"))

        cfg = _make_config(
            mcp_servers=[
                McpServerConfig(name="flaky", type="http", url="http://flaky/sse")
            ]
        )

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        # Must not raise — swallowed as warning
        await agent.initialize()


# ---------------------------------------------------------------------------
# 12. Smoke test: live beads-mcp connection
# ---------------------------------------------------------------------------

class TestBeadsMcpSmoke:
    @pytest.mark.skipif(
        not _which("beads-mcp"),
        reason="beads-mcp not installed",
    )
    async def test_beads_mcp_connects_and_returns_tools(self):
        """Smoke: connect to the live beads-mcp stdio server and verify tools exist."""
        import os, sys

        from sr2_spectre.mcp.client import MCPClient

        beads_bin = shutil.which("beads-mcp") or os.path.join(os.path.dirname(sys.executable), "beads-mcp")
        assert os.path.isfile(beads_bin), f"beads-mcp not found at {beads_bin}"

        client = MCPClient(
            server_type="stdio",
            command=[beads_bin],
            env={"BEADS_WORKING_DIR": "/data/obsidian"},
        )
        try:
            bridges = await client.connect()
            # beads-mcp v1.x exposes multiple tools; verify at least a few exist
            names = {b.name for b in bridges}
            assert len(names) >= 3, f"Expected at least 3 tools from beads-mcp, got: {names}"
            # Verify core tracking verbs are present
            assert "list" in names or "search" in names, f"Expected list/search tool in {names}"
        finally:
            await client.close()
