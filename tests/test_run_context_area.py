"""RunContext.area and the run-context provider's three-state area key.

Spec: ``specs/channel-area-injection.md`` (bead spc-48).

  AC 7  — the provider dict omits ``area`` entirely when ``RunContext.area``
          is ``None``, and includes it (including as ``""``) when it is set.
  AC 12 — TUI and single-shot runs produce no ``area`` key and resolve as
          before. The REPL (``interfaces/repl.py``) landed after the spec;
          obsidian-7qdu extended it to stamp the launch-directory area, so
          it now behaves like the other area-resolving interfaces.

The three states the rest of the pipeline reads (FR 10):
  key absent      -> this interface does not resolve areas
  key == ""       -> explicitly no area; consumers must not fall through
  key non-empty   -> use it as the area name
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import RunContext, RunMode


def _minimal_pipeline_dict() -> dict:
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


def _agent_and_provider():
    """A real Agent with SR2 stubbed, plus the provider it hands SR2.

    The returned callable is the exact object resolvers receive as
    ``Dependencies.run_context_provider``.
    """
    from sr2_spectre.agent import Agent

    with patch("sr2_spectre.session.SR2") as MockSR2:
        MockSR2.return_value = MagicMock()
        agent = Agent(config=_make_config())
        provider = MockSR2.call_args.kwargs["run_context_provider"]
    return agent, provider


# ---------------------------------------------------------------------------
# RunContext.area field (FR 9)
# ---------------------------------------------------------------------------

class TestRunContextAreaField:
    def test_area_defaults_to_none(self) -> None:
        ctx = RunContext(interface="tui", mode=RunMode.INTERACTIVE, source="/tmp")
        assert ctx.area is None

    def test_area_carries_a_name(self) -> None:
        ctx = RunContext(
            interface="discord",
            mode=RunMode.INTERACTIVE,
            source=None,
            area="fractured-roots",
        )
        assert ctx.area == "fractured-roots"

    def test_empty_area_is_distinct_from_none(self) -> None:
        """An empty string (explicit no-area) and None (no resolution) differ."""
        no_resolution = RunContext(
            interface="discord", mode=RunMode.INTERACTIVE, source=None, area=None
        )
        explicit_none = RunContext(
            interface="discord", mode=RunMode.INTERACTIVE, source=None, area=""
        )
        assert explicit_none.area == ""
        assert explicit_none != no_resolution


# ---------------------------------------------------------------------------
# AC 7 — provider dict
# ---------------------------------------------------------------------------

class TestProviderAreaKey:
    def test_provider_omits_area_when_none(self) -> None:
        agent, provider = _agent_and_provider()
        agent.set_run_context(
            RunContext(interface="discord", mode=RunMode.INTERACTIVE, source=None)
        )
        out = provider()
        assert out is not None
        assert "area" not in out

    @pytest.mark.parametrize("area", ["", "fractured-roots"])
    def test_provider_includes_area_when_set(self, area: str) -> None:
        """Both an explicit no-area and a named area reach consumers."""
        agent, provider = _agent_and_provider()
        agent.set_run_context(
            RunContext(
                interface="discord", mode=RunMode.INTERACTIVE, source=None, area=area
            )
        )
        assert provider()["area"] == area

    def test_existing_keys_survive(self) -> None:
        """mode and source keep their current meaning alongside area."""
        agent, provider = _agent_and_provider()
        agent.set_run_context(
            RunContext(
                interface="tui",
                mode=RunMode.HEADLESS,
                source="/home/x",
                area="alpha",
            )
        )
        out = provider()
        assert out["mode"] == "headless"
        assert out["source"] == "/home/x"

    def test_provider_returns_none_without_a_run_context(self) -> None:
        _agent, provider = _agent_and_provider()
        assert provider() is None


# ---------------------------------------------------------------------------
# AC 12 — TUI and single-shot are untouched
# ---------------------------------------------------------------------------

class TestNonDiscordInterfaces:
    async def test_tui_sets_no_area(self) -> None:
        from sr2_spectre.interfaces.tui import TUIInterface

        agent, provider = _agent_and_provider()
        await TUIInterface().start(agent)

        assert agent.run_context.area is None
        assert "area" not in provider()
        assert agent.run_context.source == os.getcwd()

    async def test_repl_stamps_derived_area(self, monkeypatch) -> None:
        """The REPL derives its area from the launch directory (obsidian-7qdu)."""
        from sr2_spectre.interfaces.repl import REPLInterface

        monkeypatch.setattr(
            "sr2_spectre.interfaces.repl.derive_area", lambda: "fractured-roots"
        )
        agent, provider = _agent_and_provider()
        await REPLInterface().start(agent)

        assert agent.run_context.area == "fractured-roots"
        assert provider()["area"] == "fractured-roots"
        assert agent.run_context.source == os.getcwd()

    async def test_single_shot_sets_no_area(self) -> None:
        from sr2_spectre.interfaces.single_shot import SingleShotInterface

        agent, provider = _agent_and_provider()
        await SingleShotInterface(prompt="test").start(agent)

        assert agent.run_context.area is None
        assert "area" not in provider()
        assert agent.run_context.source is None
