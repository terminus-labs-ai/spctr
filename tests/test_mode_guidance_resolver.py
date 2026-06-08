"""Tests for ModeGuidanceResolver: mode-aware guidance injection via resolver."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sr2.config.models import ResolverConfig
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import RunContext, RunMode


def _minimal_pipeline_dict() -> dict:
    """Build a minimal pipeline config dict for SpectreConfig construction."""
    return {
        "layers": [
            {
                "name": "system",
                "target": "system",
                "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
            },
            {
                "name": "conversation",
                "target": "messages",
                "resolvers": [{"type": "session"}, {"type": "input"}],
            },
        ],
    }


def _make_config() -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test"),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=_minimal_pipeline_dict(),
    )


# ---------------------------------------------------------------------------
# Resolver unit tests — build() + resolve() with controlled deps
# ---------------------------------------------------------------------------


class TestModeGuidanceResolver:
    """ModeGuidanceResolver returns the correct guidance per mode."""

    def _make_resolver_config(self) -> ResolverConfig:
        return ResolverConfig(
            type="mode_guidance",
            config={},
            max_executions=1,
        )

    async def test_headless_guidance(self):
        """When mode=headless, resolver returns headless guidance block."""
        from sr2_spectre.pipeline.mode_guidance_resolver import (
            ModeGuidanceResolver,
            _HEADLESS_GUIDANCE,
        )

        deps = Dependencies(
            run_context_provider=lambda: {"mode": "headless", "source": "single_shot"}
        )
        resolver = ModeGuidanceResolver.build(self._make_resolver_config(), deps)

        result = await resolver.resolve([
            Event(name="turn_start", phase=EventPhase.STARTING, source_layer="engine")
        ])

        assert len(result.content) == 1
        assert result.content[0].text == _HEADLESS_GUIDANCE
        assert "headless" in result.content[0].text.lower()
        assert "clarifying questions" in result.content[0].text
        assert result.source_layer == "mode_guidance"

    async def test_interactive_guidance(self):
        """When mode=interactive, resolver returns interactive guidance block."""
        from sr2_spectre.pipeline.mode_guidance_resolver import (
            ModeGuidanceResolver,
            _INTERACTIVE_GUIDANCE,
        )

        deps = Dependencies(
            run_context_provider=lambda: {"mode": "interactive", "source": "tui"}
        )
        resolver = ModeGuidanceResolver.build(self._make_resolver_config(), deps)

        result = await resolver.resolve([
            Event(name="turn_start", phase=EventPhase.STARTING, source_layer="engine")
        ])

        assert len(result.content) == 1
        assert result.content[0].text == _INTERACTIVE_GUIDANCE
        assert "interactive" in result.content[0].text.lower()
        assert "clarifying questions" in result.content[0].text

    async def test_fallback_when_no_provider(self):
        """When run_context_provider is None, resolver returns fallback guidance."""
        from sr2_spectre.pipeline.mode_guidance_resolver import (
            ModeGuidanceResolver,
            _FALLBACK_GUIDANCE,
        )

        deps = Dependencies()  # run_context_provider defaults to None
        resolver = ModeGuidanceResolver.build(self._make_resolver_config(), deps)

        result = await resolver.resolve([
            Event(name="turn_start", phase=EventPhase.STARTING, source_layer="engine")
        ])

        assert len(result.content) == 1
        assert result.content[0].text == _FALLBACK_GUIDANCE

    async def test_fallback_when_provider_returns_none(self):
        """When provider returns None, resolver returns fallback guidance."""
        from sr2_spectre.pipeline.mode_guidance_resolver import (
            ModeGuidanceResolver,
            _FALLBACK_GUIDANCE,
        )

        deps = Dependencies(run_context_provider=lambda: None)
        resolver = ModeGuidanceResolver.build(self._make_resolver_config(), deps)

        result = await resolver.resolve([
            Event(name="turn_start", phase=EventPhase.STARTING, source_layer="engine")
        ])

        assert result.content[0].text == _FALLBACK_GUIDANCE

    async def test_fallback_on_unknown_mode(self):
        """When mode has an unknown value, resolver returns fallback guidance."""
        from sr2_spectre.pipeline.mode_guidance_resolver import (
            ModeGuidanceResolver,
            _FALLBACK_GUIDANCE,
        )

        deps = Dependencies(
            run_context_provider=lambda: {"mode": "batch", "source": "cron"}
        )
        resolver = ModeGuidanceResolver.build(self._make_resolver_config(), deps)

        result = await resolver.resolve([
            Event(name="turn_start", phase=EventPhase.STARTING, source_layer="engine")
        ])

        assert result.content[0].text == _FALLBACK_GUIDANCE

    async def test_execution_count_increments(self):
        """Resolver execution_count increments on each resolve()."""
        from sr2_spectre.pipeline.mode_guidance_resolver import ModeGuidanceResolver

        deps = Dependencies(
            run_context_provider=lambda: {"mode": "headless"}
        )
        resolver = ModeGuidanceResolver.build(self._make_resolver_config(), deps)

        assert resolver.execution_count == 0
        await resolver.resolve([])
        assert resolver.execution_count == 1
        await resolver.resolve([])
        assert resolver.execution_count == 2

    async def test_subscriptions_default(self):
        """Resolver subscribes to turn_start by default."""
        from sr2_spectre.pipeline.mode_guidance_resolver import ModeGuidanceResolver

        deps = Dependencies()
        resolver = ModeGuidanceResolver.build(self._make_resolver_config(), deps)

        assert any(
            s.event_name == "turn_start"
            for s in resolver.subscriptions
        )

    async def test_guidance_blocks_are_different(self):
        """Headless and interactive guidance blocks have different content."""
        from sr2_spectre.pipeline.mode_guidance_resolver import (
            _HEADLESS_GUIDANCE,
            _INTERACTIVE_GUIDANCE,
        )

        assert _HEADLESS_GUIDANCE != _INTERACTIVE_GUIDANCE
        assert "headless" in _HEADLESS_GUIDANCE.lower()
        assert "interactive" in _INTERACTIVE_GUIDANCE.lower()
        # Headless says "do not ask" — interactive says "ask clarifying"
        assert "do not ask" in _HEADLESS_GUIDANCE.lower()
        assert "ask clarifying" in _INTERACTIVE_GUIDANCE.lower()


# ---------------------------------------------------------------------------
# Session wiring test — run_context_provider flows through SR2
# ---------------------------------------------------------------------------


class TestSessionWiresRunContextProvider:
    """Session passes RunContext through to SR2's run_context_provider."""

    def test_session_provider_reads_run_context(self):
        """Session's run_context_provider returns data from set_run_context()."""
        from sr2_spectre.session import Session
        from sr2_spectre.tools.registry import ToolRegistry

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            registry = ToolRegistry()
            session = Session(
                frame_id="test-frame",
                config=_make_config(),
                llm=MagicMock(),
                registry=registry,
            )

            # SR2 was constructed with a run_context_provider callback
            call_kwargs = MockSR2.call_args.kwargs
            provider = call_kwargs.get("run_context_provider")
            assert provider is not None, "run_context_provider must be passed to SR2"

            # Before set_run_context, provider returns None
            assert provider() is None

            # After set_run_context, provider returns the mode dict
            session.set_run_context(
                RunContext(
                    interface="single_shot",
                    mode=RunMode.HEADLESS,
                    source=None,
                )
            )
            ctx = provider()
            assert ctx is not None
            assert ctx["mode"] == "headless"

            # Change mode and verify the callback reads the updated value
            session.set_run_context(
                RunContext(
                    interface="tui",
                    mode=RunMode.INTERACTIVE,
                    source="/home/user",
                )
            )
            ctx = provider()
            assert ctx["mode"] == "interactive"
            assert ctx["source"] == "/home/user"
