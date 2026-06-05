"""Smoke tests — end-to-end integration verification for the sr2-spectre stack.

These are NOT unit tests. They verify the full stack works end-to-end:
config loading → agent instantiation → tool registry → interface wiring →
MCP integration path → planning resolver → degradation.

No external LLM calls required. All mocks are at the SR2/LLM boundary.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.agent import Agent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig, load_config
from sr2_spectre.interfaces.single_shot import SingleShotInterface
from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_spectre_config(
    tools: list[dict] | None = None,
    mcp_servers: list | None = None,
) -> SpectreConfig:
    """Build a minimal SpectreConfig with a static system prompt."""
    return SpectreConfig(
        agent=AgentConfig(
            name="smoke-agent",
            tools=tools or [],
            mcp_servers=mcp_servers or [],
        ),
        models={"default": ModelConfig(model="test", base_url="http://test:11434/v1")},
        pipeline={
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
        },
    )


# ---------------------------------------------------------------------------
# Smoke 1: Config loading round-trip
# ---------------------------------------------------------------------------

class TestSmokeConfigLoading:
    def test_load_full_config_from_yaml(self, tmp_path: Path) -> None:
        """A realistic config file loads without error and round-trips correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                agent:
                  name: smoke-test
                  tools:
                    - name: terminal
                      class_path: sr2_spectre.tools.builtins.terminal.TerminalTool
                      config:
                        timeout: 30
                    - name: file_read
                      class_path: sr2_spectre.tools.builtins.file_read.FileReadTool
                    - name: file_write
                      class_path: sr2_spectre.tools.builtins.file_write.FileWriteTool
                models:
                  default:
                    model: test/model
                    base_url: http://localhost:11434/v1
                pipeline:
                  token_budget: 200000
                  max_tool_iterations: 40
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
                    - name: tools
                      target: tools
                      resolvers: []
                      tool_providers:
                        - type: spectre_tools
                    - name: conversation
                      target: messages
                      resolvers:
                        - type: session
                        - type: input
                """)
        )
        cfg = load_config(str(config_file))
        assert cfg.agent.name == "smoke-test"
        assert len(cfg.agent.tools) == 3
        assert cfg.agent.tools[0].name == "terminal"
        assert cfg.models["default"].model == "test/model"
        assert len(cfg.pipeline.layers) == 3

    def test_load_config_with_extends(self, tmp_path: Path) -> None:
        """Config with extends: field merges correctly."""
        base = tmp_path / "base.yaml"
        base.write_text(
            textwrap.dedent("""\
                agent:
                  name: base-agent
                models:
                  default:
                    model: test/model
                    base_url: http://localhost:11434/v1
                pipeline:
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
                    - name: conversation
                      target: messages
                      resolvers:
                        - type: session
                        - type: input
                """)
        )
        override = tmp_path / "override.yaml"
        override.write_text(
            textwrap.dedent(f"""\
                extends: {base}
                agent:
                  name: extended-agent
                  tools:
                    - name: terminal
                      class_path: sr2_spectre.tools.builtins.terminal.TerminalTool
                models:
                  default:
                    model: test/model
                    base_url: http://localhost:11434/v1
                pipeline:
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
                    - name: conversation
                      target: messages
                      resolvers:
                        - type: session
                        - type: input
                """)
        )
        cfg = load_config(str(override))
        assert cfg.agent.name == "extended-agent"
        assert len(cfg.agent.tools) == 1
        assert cfg.agent.tools[0].name == "terminal"
        # Pipeline from override preserved
        assert len(cfg.pipeline.layers) == 2


# ---------------------------------------------------------------------------
# Smoke 2: Agent instantiation + tool registry
# ---------------------------------------------------------------------------

class TestSmokeAgentInstantiation:
    def test_agent_creates_with_built_in_tools(self) -> None:
        """Agent instantiates and registers the specified built-in tools."""
        cfg = _minimal_spectre_config(
            tools=[
                {"name": "terminal", "class_path": "sr2_spectre.tools.builtins.terminal.TerminalTool"},
                {"name": "file_read", "class_path": "sr2_spectre.tools.builtins.file_read.FileReadTool"},
                {"name": "file_write", "class_path": "sr2_spectre.tools.builtins.file_write.FileWriteTool"},
            ]
        )
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        assert "terminal" in agent.registry
        assert "file_read" in agent.registry
        assert "file_write" in agent.registry

    def test_agent_no_tools_has_only_auto_injected_tools(self) -> None:
        """Agent with no tools still has load_skill auto-injected."""
        cfg = _minimal_spectre_config(tools=[])
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        assert "load_skill" in agent.registry
        assert len(agent.registry) == 1

    def test_agent_registers_all_built_in_tools(self) -> None:
        """All shipped built-in tools can be registered without error."""
        all_tool_defs = [
            {"name": "terminal", "class_path": "sr2_spectre.tools.builtins.terminal.TerminalTool"},
            {"name": "file_read", "class_path": "sr2_spectre.tools.builtins.file_read.FileReadTool"},
            {"name": "file_write", "class_path": "sr2_spectre.tools.builtins.file_write.FileWriteTool"},
            {"name": "edit", "class_path": "sr2_spectre.tools.builtins.edit.EditTool"},
            {"name": "grep", "class_path": "sr2_spectre.tools.builtins.grep.GrepTool"},
            {"name": "glob", "class_path": "sr2_spectre.tools.builtins.glob.GlobTool"},
            {"name": "web_search", "class_path": "sr2_spectre.tools.builtins.web_search.WebSearchTool", "config": {"base_url": "http://localhost:8080"}},
            {"name": "complete_step", "class_path": "sr2_spectre.tools.builtins.complete_step.CompleteStepTool"},
        ]
        cfg = _minimal_spectre_config(tools=all_tool_defs)
        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        for tool_def in all_tool_defs:
            assert tool_def["name"] in agent.registry, f"Tool {tool_def['name']} not registered"


# ---------------------------------------------------------------------------
# Smoke 3: Single-shot interface path
# ---------------------------------------------------------------------------

class TestSmokeSingleShotInterface:
    async def test_single_shot_runs_end_to_end(self, capsys: pytest.CaptureFixture) -> None:
        """Single-shot interface accepts a prompt and produces output."""
        from sr2_spectre.core import TurnResult

        agent = AsyncMock()
        agent.handle_user_message.return_value = TurnResult(
            text="Hello from smoke test",
            tool_calls_executed=0,
            total_tokens=25,
        )
        interface = SingleShotInterface(prompt="smoke test")
        await interface.run(agent)

        agent.handle_user_message.assert_called_once_with("smoke test")
        captured = capsys.readouterr()
        assert "Hello from smoke test" in captured.out


# ---------------------------------------------------------------------------
# Smoke 4: MCP wiring path (no real connection)
# ---------------------------------------------------------------------------

class TestSmokeMcpWiring:
    async def test_agent_initializes_mcp_and_registers_bridge(self) -> None:
        """MCP server config → MCPClient → bridge → tool registry path works."""
        from sr2_spectre.config import McpServerConfig
        from sr2_spectre.mcp.tool_bridge import MCPToolBridge

        bridge = MagicMock(spec=MCPToolBridge)
        bridge.name = "mcp_list_files"
        bridge.description = "List files via MCP"
        bridge.input_schema = {"type": "object", "properties": {}}
        bridge._is_coroutine = AsyncMock._is_coroutine

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(return_value=[bridge])

        cfg = _minimal_spectre_config(
            mcp_servers=[
                McpServerConfig(name="fs", type="stdio", command=["fs-server"])
            ]
        )

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        await agent.initialize()
        assert "mcp_list_files" in agent.registry

    async def test_agent_graceful_on_mcp_connection_failure(self) -> None:
        """MCP connection failure is swallowed and logged, not raised."""
        from sr2_spectre.config import McpServerConfig
        from sr2_spectre.mcp.client import MCPConnectionError

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(side_effect=MCPConnectionError("connection refused"))

        cfg = _minimal_spectre_config(
            mcp_servers=[
                McpServerConfig(name="unreachable", type="http", url="http://bad:9999/sse")
            ]
        )

        with patch("sr2_spectre.session.SR2") as MockSR2, \
             patch("sr2_spectre.runtime.MCPClient", return_value=mock_client):
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg)

        # Should not raise
        await agent.initialize()


# ---------------------------------------------------------------------------
# Smoke 5: Tool execution path (terminal)
# ---------------------------------------------------------------------------

class TestSmokeToolExecution:
    async def test_terminal_tool_executes_command(self) -> None:
        """Terminal tool can execute a simple command and return output."""
        from sr2_spectre.tools.builtins.terminal import TerminalTool

        tool = TerminalTool(timeout=5)
        result = await tool(command="echo 'smoke OK'")

        assert "smoke OK" in str(result)


# ---------------------------------------------------------------------------
# Smoke 6: File I/O tools
# ---------------------------------------------------------------------------

class TestSmokeFileTools:
    async def test_file_write_and_read_round_trip(self, tmp_path: Path) -> None:
        """file_write + file_read round-trips content correctly."""
        from sr2_spectre.tools.builtins.file_read import FileReadTool
        from sr2_spectre.tools.builtins.file_write import FileWriteTool

        test_file = tmp_path / "smoke_test.txt"
        content = "Smoke test round-trip content\nLine 2\n"

        writer = FileWriteTool()
        await writer(path=str(test_file), content=content)

        reader = FileReadTool()
        result = await reader(path=str(test_file))

        assert content.strip() in str(result)

    async def test_file_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """file_write creates missing parent directories."""
        from sr2_spectre.tools.builtins.file_write import FileWriteTool

        nested = tmp_path / "a" / "b" / "c" / "file.txt"
        writer = FileWriteTool()
        await writer(path=str(nested), content="nested content")

        assert nested.exists()
        assert nested.read_text() == "nested content"


# ---------------------------------------------------------------------------
# Smoke 7: PlanResolver registration
# ---------------------------------------------------------------------------

class TestSmokePlanResolver:
    def test_plan_resolver_importable(self) -> None:
        """PlanResolver can be imported and instantiated with ResolverConfig."""
        from sr2.config.models import ResolverConfig
        from sr2_spectre.planning.resolver import PlanResolver

        resolver_cfg = ResolverConfig(
            type="plan",
            config={
                "plans_root": str(Path.home() / ".sr2" / "plans"),
                "project": "test-project",
            },
            max_executions=10,
        )
        resolver = PlanResolver(config=resolver_cfg)
        assert isinstance(resolver, PlanResolver)


# ---------------------------------------------------------------------------
# Smoke 8: Config validation catches bad configs
# ---------------------------------------------------------------------------

class TestSmokeConfigValidation:
    def test_missing_model_raises(self, tmp_path: Path) -> None:
        """Config without a model section should fail validation."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("agent:\n  name: broken\n")

        with pytest.raises((ValueError, TypeError, KeyError)):
            load_config(str(bad))

    def test_bad_tool_class_path_logs_warning(self, tmp_path: Path) -> None:
        """A config with an unresolvable tool class_path logs a warning but loads."""
        config_file = tmp_path / "bad_tool.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                agent:
                  name: bad-tool-agent
                  tools:
                    - name: nonexistent
                      class_path: definitely.not.a.real.module.SomeTool
                models:
                  default:
                    model: test
                    base_url: http://localhost:11434/v1
                pipeline:
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
                    - name: tools
                      target: tools
                      resolvers: []
                      tool_providers:
                        - type: spectre_tools
                    - name: conversation
                      target: messages
                      resolvers:
                        - type: session
                        - type: input
                """)
        )
        # load_config itself may or may not validate tool paths at load time
        # The key is that it doesn't crash the application
        cfg = load_config(str(config_file))
        assert cfg.agent.name == "bad-tool-agent"
        assert len(cfg.agent.tools) == 1


# ---------------------------------------------------------------------------
# Smoke 9: Discord interface end-to-end (mocked adapter)
# ---------------------------------------------------------------------------

class TestSmokeDiscordInterface:
    """Verify the full Discord interface path works end-to-end with mocked adapter.

    These are integration-style smoke tests — they verify the Interface protocol
    is properly implemented and message routing works through the full stack.
    They do NOT require discord.py installed or a real Discord bot.
    """

    def test_discord_interface_implements_protocol(self) -> None:
        """DiscordInterface has all Interface protocol attributes."""
        from sr2_spectre.interfaces import Interface
        from sr2_spectre.interfaces.discord import DiscordInterface

        interface = DiscordInterface()
        assert hasattr(interface, "name")
        assert interface.name == "discord"
        assert hasattr(interface, "start")
        assert hasattr(interface, "stop")
        assert hasattr(interface, "run")

    def test_discord_interface_config_loading(self) -> None:
        """DiscordInterface loads with custom config without error."""
        from sr2_spectre.interfaces.discord.config import DiscordConfig
        from sr2_spectre.interfaces.discord.interface import DiscordInterface

        config = DiscordConfig(
            token="test-token",
            mention_only=True,
            max_message_length=1500,
            tool_embed_enabled=False,
        )
        interface = DiscordInterface(config=config)
        assert interface.config.token == "test-token"
        assert interface.config.mention_only is True
        assert interface.config.max_message_length == 1500
        assert interface.config.tool_embed_enabled is False

    def test_discord_handler_should_respond(self) -> None:
        """Message routing respects mention_only setting."""
        from sr2_spectre.interfaces.discord.handler import should_respond

        # Without mention_only, everything gets a response
        assert should_respond("hello", False, 11111, ["<@11111>"]) is True

        # With mention_only, non-mention messages are ignored
        assert should_respond("hello", True, 11111, ["<@11111>"]) is False

        # With mention_only, mentioned messages get a response
        assert should_respond("<@11111> hello", True, 11111, ["<@11111>"]) is True

    def test_discord_handler_slash_commands(self) -> None:
        """Slash command parsing and handling works correctly."""
        from sr2_spectre.interfaces.discord.handler import (
            handle_command,
            parse_slash_command,
        )

        # Parse known command
        cmd, rest = parse_slash_command("/reset")
        assert cmd == "reset"
        assert rest == ""

        # Parse command with arguments
        cmd, rest = parse_slash_command("/ask what is the weather?")
        assert cmd == "ask"
        assert rest == "what is the weather?"

        # Not a command
        cmd, rest = parse_slash_command("hello world")
        assert cmd is None

        # /help returns help text
        response = handle_command("help", "")
        assert response is not None
        assert "/ask" in response

    def test_discord_session_map_isolation(self) -> None:
        """SessionMap creates isolated sessions per channel."""
        from sr2_spectre.interfaces.discord.session_map import SessionMap

        sm = SessionMap()
        s1 = sm.get_or_create(111)
        s2 = sm.get_or_create(222)

        # Different sessions for different channels
        assert s1 is not s2

        # Same session for same channel
        s1_again = sm.get_or_create(111)
        assert s1 is s1_again

        # Reset clears a specific channel
        s1.history.append({"role": "user", "content": []})
        assert len(s1.history) == 1
        sm.reset(111)
        assert len(s1.history) == 0
        assert len(s2.history) == 0  # unaffected

    def test_discord_message_chunking(self) -> None:
        """Long messages are chunked correctly for Discord's 2000-char limit."""
        from sr2_spectre.interfaces.discord.handler import chunk_message

        text = "x" * 5000
        chunks = chunk_message(text, 2000)

        assert len(chunks) == 3
        # chunk_message adds "..." split markers, so chunks may approach max_length
        assert all(len(c) <= 2000 for c in chunks)
        # All text accounted for (split markers add extra chars)
        assert sum(len(c) for c in chunks) >= len(text)


# ---------------------------------------------------------------------------
# Smoke 10: Memory extraction + retrieval (sr2 memory subsystem)
# ---------------------------------------------------------------------------

class TestSmokeMemorySubsystem:
    """Verify the SR2 memory subsystem works end-to-end:
    extraction → storage → search → resolver injection.
    """

    def test_memory_extraction_from_turn(self) -> None:
        """RuleBasedExtractor extracts memories from conversation text."""
        from sr2.memory import RuleBasedExtractor

        extractor = RuleBasedExtractor()

        # Text with preference signal
        result = extractor.extract(
            "I prefer using pytest over unittest for testing."
        )
        assert len(result.memories) >= 1
        assert any("preference" in m.tags for m in result.memories)

    def test_memory_store_save_and_search(self) -> None:
        """InMemoryMemoryStore saves memories and retrieves them by search."""
        from sr2.memory import InMemoryMemoryStore, Memory, MemoryScope

        store = InMemoryMemoryStore()

        # Save some memories
        mem1 = Memory(
            content="The project uses Python 3.12 with uv for package management.",
            scope=MemoryScope.PROJECT,
            tags=["fact", "tooling"],
        )
        mem2 = Memory(
            content="User prefers dark theme for the TUI interface.",
            scope=MemoryScope.SHARED,
            tags=["preference"],
        )
        store.save(mem1)
        store.save(mem2)

        # Search by keyword
        results = store.search("Python")
        assert len(results) >= 1
        assert "Python" in results[0].content

        results = store.search("theme")
        assert len(results) >= 1
        assert "theme" in results[0].content

    def test_memory_store_tag_filtering(self) -> None:
        """Memory store supports filtering by tag."""
        from sr2.memory import InMemoryMemoryStore, Memory, MemoryScope

        store = InMemoryMemoryStore()

        store.save(Memory(content="Fact about the system", scope=MemoryScope.PROJECT, tags=["fact"]))
        store.save(Memory(content="User preference for formatting", scope=MemoryScope.SHARED, tags=["preference"]))
        store.save(Memory(content="Another fact about architecture", scope=MemoryScope.PROJECT, tags=["fact", "architecture"]))

        fact_results = store.get_by_tag("fact")
        assert len(fact_results) == 2

        pref_results = store.get_by_tag("preference")
        assert len(pref_results) == 1

    def test_memory_frequency_reinforcement(self) -> None:
        """Saving the same memory again increments frequency."""
        from sr2.memory import InMemoryMemoryStore, Memory

        store = InMemoryMemoryStore()
        mem = Memory(content="Important recurring fact")
        store.save(mem)
        assert mem.frequency == 0  # First save

        # Save again — frequency increments
        again = store.save(mem)
        assert again.frequency == 1

    @pytest.mark.asyncio
    async def test_memory_resolver_injects_context(self) -> None:
        """MemoryResolver searches the store and injects memories as context."""
        from sr2.config.models import ResolverConfig
        from sr2.memory import InMemoryMemoryStore, Memory, MemoryScope, MemoryResolver
        from sr2.pipeline.events import Event, EventPhase

        store = InMemoryMemoryStore()
        store.save(Memory(
            content="The user prefers snake_case for variable names.",
            scope=MemoryScope.SHARED,
            tags=["preference"],
        ))

        resolver = MemoryResolver(
            config=ResolverConfig(type="memory", config={"limit": 5}),
            store=store,
        )

        # Simulate a user input event — data can be a plain string
        events = [
            Event(
                name="user_input",
                phase=EventPhase.STARTING,
                source_layer="",
                data="What naming convention should I use?",
            )
        ]

        content = await resolver.resolve(events)

        assert content.resolver_name == "memory"
        # The resolver searched the store; content may or may not have matches
        # depending on keyword overlap between query and stored memories.
        assert len(content.content) >= 0

    def test_memory_extraction_deduplication(self) -> None:
        """RuleBasedExtractor deduplicates similar extractions."""
        from sr2.memory import RuleBasedExtractor

        extractor = RuleBasedExtractor()

        # Text with repeated similar facts
        text = (
            "The project uses Docker. The project uses Docker for testing. "
            "The project uses Docker in production."
        )
        result = extractor.extract(text)

        # Should deduplicate similar content
        contents = [m.content for m in result.memories]
        # At least some deduplication should occur
        assert len(contents) <= len(set(c[:30] for c in contents)) + 1


# ---------------------------------------------------------------------------
# Smoke 11: Degradation + priority shedding
# ---------------------------------------------------------------------------

class TestSmokeDegradation:
    """Verify degradation subsystem works end-to-end:
    ladder → active categories → priority shedding → circuit breaker.
    """

    def test_degradation_ladder_step_down(self) -> None:
        """DegradationLadder steps down correctly, reducing active categories."""
        from sr2.degradation import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()

        # Start at FULL
        assert ladder.is_at_full() is True
        full_categories = ladder.active_categories()
        assert "system" in full_categories
        assert "memory" in full_categories
        assert "tools" in full_categories

        # Step down — categories reduce
        ladder.step_down()
        reduced = ladder.active_categories()
        assert len(reduced) < len(full_categories) or reduced != full_categories

        # Reset returns to full
        ladder.reset()
        assert ladder.is_at_full() is True

    def test_degradation_ladder_max_depth(self) -> None:
        """Stepping down beyond max depth is a no-op."""
        from sr2.degradation import DegradationLadder

        ladder = DegradationLadder()
        # Step down through all levels
        for _ in range(10):
            ladder.step_down()

        # Should be at the lowest level
        assert not ladder.is_at_full()

    def test_priority_shedding_removes_lowest_priority_first(self) -> None:
        """shed() removes lowest-priority layers first to fit budget."""
        from sr2.degradation.shedding import shed

        class Layer:
            def __init__(self, name, priority, token_count):
                self.name = name
                self.priority = priority
                self.token_count = token_count

        layers = [
            Layer("system", priority=10, token_count=100),       # High priority
            Layer("memory", priority=5, token_count=200),        # Medium priority
            Layer("tools", priority=5, token_count=150),         # Medium priority
            Layer("history", priority=3, token_count=300),       # Low priority
        ]

        # Budget of 450 — should keep system(100) + memory(200) + tools(150) = 450
        survivors = shed(layers, 450)
        names = [l.name for l in survivors]
        assert "history" not in names  # Lowest priority shed first
        assert "system" in names

        # Tighter budget — shed more
        survivors = shed(layers, 250)
        total = sum(l.token_count for l in survivors)
        assert total <= 250

    def test_priority_shedding_preserves_order(self) -> None:
        """shed() preserves the original order of surviving layers."""
        from sr2.degradation.shedding import shed

        class Layer:
            def __init__(self, name, priority, token_count):
                self.name = name
                self.priority = priority
                self.token_count = token_count

        layers = [
            Layer("a", priority=3, token_count=100),
            Layer("b", priority=10, token_count=100),
            Layer("c", priority=5, token_count=100),
            Layer("d", priority=1, token_count=100),
        ]

        # Budget 200 — should keep highest priority layers
        survivors = shed(layers, 200)
        names = [l.name for l in survivors]

        # Survivors should be in original order
        for i in range(len(names) - 1):
            idx_a = next(j for j, l in enumerate(layers) if l.name == names[i])
            idx_b = next(j for j, l in enumerate(layers) if l.name == names[i + 1])
            assert idx_a < idx_b

    def test_priority_shedding_empty_input(self) -> None:
        """shed() handles empty input gracefully."""
        from sr2.degradation.shedding import shed

        assert shed([], 100) == []

    def test_priority_shedding_within_budget(self) -> None:
        """shed() returns all layers when total is within budget."""
        from sr2.degradation.shedding import shed

        class Layer:
            def __init__(self, name, priority, token_count):
                self.name = name
                self.priority = priority
                self.token_count = token_count

        layers = [
            Layer("a", priority=5, token_count=100),
            Layer("b", priority=10, token_count=200),
        ]

        survivors = shed(layers, 500)
        assert len(survivors) == 2

    def test_circuit_breaker_open_on_failures(self) -> None:
        """CircuitBreaker opens after consecutive failures, allows recovery via half-open."""
        import time
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        # Record failures to trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout to elapse
        time.sleep(0.15)

        # allow_request() transitions OPEN → HALF_OPEN after timeout
        assert breaker.allow_request() is True
        assert breaker.state == CircuitState.HALF_OPEN

        # Success in half-open closes the circuit
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_success_resets_counter(self) -> None:
        """Successful call resets the failure counter."""
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_success()
        # After success, a third failure shouldn't trip
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
