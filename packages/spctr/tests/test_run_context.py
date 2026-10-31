"""Tests for RunContext, RunMode, and the run-context seam through Agent/Session/Interfaces."""
from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.config import AgentConfig, ModelConfig
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


def _make_config() -> "SpectreConfig":  # type: ignore[name-defined]
    from sr2_spectre.config import SpectreConfig

    return SpectreConfig(
        agent=AgentConfig(name="test"),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=_minimal_pipeline_dict(),
    )


# ---------------------------------------------------------------------------
# RunContext dataclass
# ---------------------------------------------------------------------------

def test_run_context_dataclass_fields() -> None:
    """RunContext must have exactly interface, mode, source."""
    ctx = RunContext(interface="tui", mode=RunMode.INTERACTIVE, source="/tmp")
    assert ctx.interface == "tui"
    assert ctx.mode == RunMode.INTERACTIVE
    assert ctx.source == "/tmp"


def test_run_context_frozen() -> None:
    """RunContext must be frozen (immutable)."""
    ctx = RunContext(interface="single_shot", mode=RunMode.HEADLESS, source=None)
    with pytest.raises(FrozenInstanceError):
        ctx.interface = "tui"  # type: ignore[frozen-instantiation]


def test_run_context_equality() -> None:
    """Frozen dataclasses support value equality."""
    a = RunContext(interface="tui", mode=RunMode.INTERACTIVE, source="/home")
    b = RunContext(interface="tui", mode=RunMode.INTERACTIVE, source="/home")
    c = RunContext(interface="tui", mode=RunMode.HEADLESS, source="/home")
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# RunMode enum
# ---------------------------------------------------------------------------

def test_run_mode_enum_values() -> None:
    """RunMode must have INTERACTIVE and HEADLESS."""
    assert RunMode.INTERACTIVE == "interactive"
    assert RunMode.HEADLESS == "headless"


def test_run_mode_str_comparison() -> None:
    """RunMode should compare equal to its string value (StrEnum)."""
    assert RunMode.INTERACTIVE == "interactive"
    assert RunMode.HEADLESS == "headless"


# ---------------------------------------------------------------------------
# Agent run_context delegation (real Agent construction)
# ---------------------------------------------------------------------------

def test_agent_run_context_defaults_none() -> None:
    """A fresh Agent must have run_context == None."""
    from sr2_spectre.agent import Agent

    with patch("sr2_spectre.session.SR2") as MockSR2:
        MockSR2.return_value = MagicMock()
        agent = Agent(config=_make_config())

    assert agent.run_context is None


def test_agent_set_run_context() -> None:
    """Agent.set_run_context must store and return the context."""
    from sr2_spectre.agent import Agent

    with patch("sr2_spectre.session.SR2") as MockSR2:
        MockSR2.return_value = MagicMock()
        agent = Agent(config=_make_config())

    ctx = RunContext(interface="tui", mode=RunMode.INTERACTIVE, source="/home/user")
    agent.set_run_context(ctx)
    assert agent.run_context is ctx
    assert agent.run_context.interface == "tui"
    assert agent.run_context.mode == RunMode.INTERACTIVE
    assert agent.run_context.source == "/home/user"


def test_agent_set_run_context_overwrites() -> None:
    """Calling set_run_context twice must replace the previous context."""
    from sr2_spectre.agent import Agent

    with patch("sr2_spectre.session.SR2") as MockSR2:
        MockSR2.return_value = MagicMock()
        agent = Agent(config=_make_config())

    ctx_a = RunContext(interface="single_shot", mode=RunMode.HEADLESS, source=None)
    ctx_b = RunContext(interface="tui", mode=RunMode.INTERACTIVE, source="/tmp")
    agent.set_run_context(ctx_a)
    assert agent.run_context.mode == RunMode.HEADLESS
    agent.set_run_context(ctx_b)
    assert agent.run_context.mode == RunMode.INTERACTIVE


# ---------------------------------------------------------------------------
# Interface integration — SingleShotInterface sets headless context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_shot_sets_headless_context() -> None:
    """SingleShotInterface.start() must set a headless RunContext."""
    from sr2_spectre.agent import Agent
    from sr2_spectre.interfaces.single_shot import SingleShotInterface

    with patch("sr2_spectre.session.SR2") as MockSR2:
        MockSR2.return_value = MagicMock()
        agent = Agent(config=_make_config())

    iface = SingleShotInterface(prompt="test")
    await iface.start(agent)

    assert agent.run_context is not None
    assert agent.run_context.interface == "single_shot"
    assert agent.run_context.mode == RunMode.HEADLESS
    assert agent.run_context.source is None


# ---------------------------------------------------------------------------
# Interface integration — TUIInterface sets interactive context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tui_sets_interactive_context() -> None:
    """TUIInterface.start() must set an interactive RunContext with cwd as source."""
    from sr2_spectre.agent import Agent
    from sr2_spectre.interfaces.tui import TUIInterface

    with patch("sr2_spectre.session.SR2") as MockSR2:
        MockSR2.return_value = MagicMock()
        agent = Agent(config=_make_config())

    iface = TUIInterface()
    await iface.start(agent)

    assert agent.run_context is not None
    assert agent.run_context.interface == "tui"
    assert agent.run_context.mode == RunMode.INTERACTIVE
    assert agent.run_context.source == os.getcwd()


# ---------------------------------------------------------------------------
# Interface integration — DiscordInterface sets interactive context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discord_sets_interactive_context() -> None:
    """DiscordInterface.start() must set an interactive RunContext."""
    from sr2_spectre.agent import Agent
    from sr2_spectre.interfaces.discord.config import DiscordConfig
    from sr2_spectre.interfaces.discord.interface import DiscordInterface

    with patch("sr2_spectre.session.SR2") as MockSR2:
        MockSR2.return_value = MagicMock()
        agent = Agent(config=_make_config())

    iface = DiscordInterface(config=DiscordConfig())

    # Patch the adapter so start() doesn't try to connect to Discord.
    mock_adapter = AsyncMock()
    mock_adapter.bot_id = 12345
    mock_adapter.bot_mentions = []
    mock_adapter.set_message_handler = MagicMock()  # not async — avoids RuntimeWarning
    with patch.object(
        iface.__class__, "_process_message", new=AsyncMock()
    ), patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter",
        return_value=mock_adapter,
    ):
        await iface.start(agent)

    assert agent.run_context is not None
    assert agent.run_context.interface == "discord"
    assert agent.run_context.mode == RunMode.INTERACTIVE
    assert agent.run_context.source is None


# ---------------------------------------------------------------------------
# New session preserves run_context (independent session, same runtime)
# ---------------------------------------------------------------------------

def test_new_session_preserves_run_context() -> None:
    """Calling new_session() carries the run context to the replacement Session.

    The new Session has fresh history but inherits the run context already in
    force, so interfaces that reset (REPL/TUI /reset) need not re-set it.
    """
    from sr2_spectre.agent import Agent

    with patch("sr2_spectre.session.SR2") as MockSR2:
        MockSR2.return_value = MagicMock()
        agent = Agent(config=_make_config())

    ctx = RunContext(interface="tui", mode=RunMode.INTERACTIVE, source="/home")
    agent.set_run_context(ctx)
    assert agent.run_context is not None

    # new_session creates a new Session that inherits the run context
    agent.new_session("fresh")
    assert agent.run_context == ctx
