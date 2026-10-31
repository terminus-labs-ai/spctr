"""Tests for SpectreToolProvider (Step 2).

Covers:
  A. build() reads tool_registry from deps.tool_source
  B. provide() returns ToolDefinition list from the registry
  C. Protocol attributes (subscriptions, max_executions, execution_count)
  D. execution_count incremented inside provide()
  E. Entry-point discovery: spectre_tools appears in _TOOL_PROVIDERS
  F. End-to-end: SpectreToolProvider injects tools into CompletionRequest.tools
"""

from __future__ import annotations

import importlib.metadata
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sr2.models import ToolDefinition
from sr2.pipeline.events import EventSubscription
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent, TextBlock
from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for name in names:
        reg.register(
            name=name,
            description=f"Tool {name}",
            input_schema={"type": "object", "properties": {}},
            fn=lambda **kw: "result",
        )
    return reg


def _make_deps(registry: ToolRegistry | None = None) -> Any:
    from sr2.pipeline.dependencies import Dependencies
    return Dependencies(tool_source=registry or _make_registry())


def _make_config(max_executions: int = 1) -> Any:
    from sr2.config.models import ToolProviderConfig
    return ToolProviderConfig(type="spectre_tools", max_executions=max_executions)


# ---------------------------------------------------------------------------
# A. build()
# ---------------------------------------------------------------------------

class TestBuild:
    def test_build_returns_provider_instance(self):
        from sr2_spectre.providers import SpectreToolProvider
        provider = SpectreToolProvider.build(_make_config(), _make_deps())
        assert isinstance(provider, SpectreToolProvider)

    def test_build_reads_tool_registry_from_tool_source(self):
        from sr2_spectre.providers import SpectreToolProvider
        reg = _make_registry("search")
        provider = SpectreToolProvider.build(_make_config(), _make_deps(reg))
        assert provider._registry is reg

    def test_build_missing_tool_registry_raises(self):
        from sr2.pipeline.dependencies import Dependencies
        from sr2_spectre.providers import SpectreToolProvider
        deps = Dependencies(tool_source=None)
        with pytest.raises(RuntimeError, match="tool_source"):
            SpectreToolProvider.build(_make_config(), deps)


# ---------------------------------------------------------------------------
# B. provide()
# ---------------------------------------------------------------------------

class TestProvide:
    @pytest.mark.asyncio
    async def test_provide_returns_tool_definitions(self):
        from sr2_spectre.providers import SpectreToolProvider
        reg = _make_registry("search", "write")
        provider = SpectreToolProvider.build(_make_config(), _make_deps(reg))
        result = await provider.provide(events=[])
        assert len(result) == 2
        assert all(isinstance(d, ToolDefinition) for d in result)
        names = {d.name for d in result}
        assert names == {"search", "write"}

    @pytest.mark.asyncio
    async def test_provide_empty_registry(self):
        from sr2_spectre.providers import SpectreToolProvider
        provider = SpectreToolProvider.build(_make_config(), _make_deps(_make_registry()))
        result = await provider.provide(events=[])
        assert result == []

    @pytest.mark.asyncio
    async def test_provide_definition_fields_match_registry(self):
        from sr2_spectre.providers import SpectreToolProvider
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        reg = ToolRegistry()
        reg.register(name="lookup", description="Look something up", input_schema=schema, fn=lambda q: q)
        provider = SpectreToolProvider.build(_make_config(), _make_deps(reg))
        result = await provider.provide(events=[])
        assert result[0].name == "lookup"
        assert result[0].description == "Look something up"
        assert result[0].input_schema == schema


# ---------------------------------------------------------------------------
# C. Protocol attributes
# ---------------------------------------------------------------------------

class TestProtocolAttributes:
    def test_has_subscriptions(self):
        from sr2_spectre.providers import SpectreToolProvider
        provider = SpectreToolProvider.build(_make_config(), _make_deps())
        assert hasattr(provider, "subscriptions")
        assert isinstance(provider.subscriptions, list)

    def test_subscriptions_include_turn_start(self):
        from sr2_spectre.providers import SpectreToolProvider
        provider = SpectreToolProvider.build(_make_config(), _make_deps())
        event_names = [s.event_name for s in provider.subscriptions]
        assert "turn_start" in event_names

    def test_has_max_executions(self):
        from sr2_spectre.providers import SpectreToolProvider
        provider = SpectreToolProvider.build(_make_config(max_executions=3), _make_deps())
        assert provider.max_executions == 3

    def test_has_execution_count_zero(self):
        from sr2_spectre.providers import SpectreToolProvider
        provider = SpectreToolProvider.build(_make_config(), _make_deps())
        assert provider.execution_count == 0


# ---------------------------------------------------------------------------
# D. execution_count incremented by provide()
# ---------------------------------------------------------------------------

class TestExecutionCount:
    @pytest.mark.asyncio
    async def test_execution_count_incremented_after_provide(self):
        from sr2_spectre.providers import SpectreToolProvider
        provider = SpectreToolProvider.build(_make_config(), _make_deps())
        assert provider.execution_count == 0
        await provider.provide(events=[])
        assert provider.execution_count == 1

    @pytest.mark.asyncio
    async def test_execution_count_increments_each_call(self):
        from sr2_spectre.providers import SpectreToolProvider
        provider = SpectreToolProvider.build(_make_config(max_executions=5), _make_deps())
        for i in range(3):
            await provider.provide(events=[])
        assert provider.execution_count == 3


# ---------------------------------------------------------------------------
# E. Entry-point discovery
# ---------------------------------------------------------------------------

class TestEntryPointDiscovery:
    def test_spectre_tools_entry_point_registered(self):
        """spectre_tools appears in importlib entry points after uv sync."""
        eps = importlib.metadata.entry_points(group="sr2.tool_providers")
        names = [ep.name for ep in eps]
        assert "spectre_tools" in names, (
            f"spectre_tools not found in sr2.tool_providers entry points. "
            f"Found: {names}. Did you run 'uv sync'?"
        )

    def test_spectre_tools_loads_correct_class(self):
        from sr2_spectre.providers import SpectreToolProvider
        eps = importlib.metadata.entry_points(group="sr2.tool_providers")
        ep = next(ep for ep in eps if ep.name == "spectre_tools")
        cls = ep.load()
        assert cls is SpectreToolProvider


# ---------------------------------------------------------------------------
# F. End-to-end: tools injected into CompletionRequest
# ---------------------------------------------------------------------------

class _MockLLMRecorder:
    def __init__(self) -> None:
        self.last_request: CompletionRequest | None = None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        from sr2.models import TokenUsage
        self.last_request = request
        return CompletionResponse(
            id="mock",
            content=[TextBlock(text="done")],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest):
        self.last_request = request
        yield StreamEvent(type="text", text="done")
        yield StreamEvent(type="end")


class TestEndToEnd:
    @pytest.fixture(autouse=False)
    def reset_registries(self):
        from sr2.orchestrator import reset_discovery as _reset_discovery
        _reset_discovery()
        yield
        _reset_discovery()

    def _ep_side_effect(self, registry: ToolRegistry):
        from sr2_spectre.providers import SpectreToolProvider
        from sr2.pipeline.resolvers.static import StaticResolver

        def _side(group: str):
            if group == "sr2.resolvers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = "static"
                ep.load.return_value = StaticResolver
                ep.dist = MagicMock(); ep.dist.name = "sr2"
                return [ep]
            if group == "sr2.tool_providers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = "spectre_tools"
                ep.load.return_value = SpectreToolProvider
                ep.dist = MagicMock(); ep.dist.name = "sr2-spectre"
                return [ep]
            return []
        return _side

    @pytest.mark.asyncio
    async def test_spectre_tools_injected_into_request(self, reset_registries):
        """SpectreToolProvider on a TOOLS layer → tools in CompletionRequest."""
        from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig, ToolProviderConfig
        from sr2.orchestrator import SR2

        reg = _make_registry("web_search")
        mock_llm = _MockLLMRecorder()

        pipeline = PipelineConfig(layers=[
            LayerConfig(name="system", target="system", resolvers=[
                ResolverConfig(type="static", config={"text": "You are helpful."})
            ]),
            LayerConfig(name="tools", target="tools", resolvers=[], tool_providers=[
                ToolProviderConfig(type="spectre_tools")
            ]),
        ])

        with patch("sr2.plugins.registry.entry_points", side_effect=self._ep_side_effect(reg)):
            sr2 = SR2(
                pipeline_config=pipeline,
                llm={"default": mock_llm},
                token_counter=CharacterTokenCounter(),
                tool_source=reg,
            )
            stream = sr2.turn([TextBlock(text="hello")])
            async for _ in stream:
                pass

        tools = mock_llm.last_request.tools
        assert tools is not None and len(tools) == 1
        assert tools[0].name == "web_search"

    @pytest.mark.asyncio
    async def test_spectre_tools_present_on_turn_two(self, reset_registries):
        """Tools appear in CompletionRequest on both turn 1 and turn 2."""
        from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig, ToolProviderConfig
        from sr2.orchestrator import SR2

        reg = _make_registry("calc")
        mock_llm = _MockLLMRecorder()

        pipeline = PipelineConfig(layers=[
            LayerConfig(name="system", target="system", resolvers=[
                ResolverConfig(type="static", config={"text": "You are helpful."})
            ]),
            LayerConfig(name="tools", target="tools", resolvers=[], tool_providers=[
                ToolProviderConfig(type="spectre_tools")
            ]),
        ])

        with patch("sr2.plugins.registry.entry_points", side_effect=self._ep_side_effect(reg)):
            sr2 = SR2(
                pipeline_config=pipeline,
                llm={"default": mock_llm},
                token_counter=CharacterTokenCounter(),
                tool_source=reg,
            )
            stream = sr2.turn([TextBlock(text="first")])
            async for _ in stream:
                pass
            turn1_tools = mock_llm.last_request.tools

            stream = sr2.turn([TextBlock(text="second")])
            async for _ in stream:
                pass
            turn2_tools = mock_llm.last_request.tools

        assert turn1_tools is not None and len(turn1_tools) == 1
        assert turn2_tools is not None and len(turn2_tools) == 1
