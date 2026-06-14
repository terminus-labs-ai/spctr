"""Tests for auto_thread feature — channel thread creation and routing.

Covers:
1.  DiscordConfig.auto_thread field
2.  SessionMap parent->thread linking
3.  DiscordBotAdapter.create_thread and is_thread_channel
4.  DiscordInterface._resolve_target_channel logic
5.  Full flow: message in parent -> thread created -> replies in thread
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from sr2_spectre.events import AgentDone, AgentTextDelta
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.interface import DiscordInterface
from sr2_spectre.interfaces.discord.session_map import SessionMap


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestAutoThreadConfig:
    def test_auto_thread_defaults_to_false(self) -> None:
        config = DiscordConfig()
        assert config.auto_thread is False

    def test_auto_thread_can_be_enabled(self) -> None:
        config = DiscordConfig(auto_thread=True)
        assert config.auto_thread is True

    def test_full_config_with_auto_thread(self) -> None:
        config = DiscordConfig(
            token="bot-token",
            mention_only=True,
            auto_thread=True,
        )
        assert config.auto_thread is True
        assert config.token == "bot-token"
        assert config.mention_only is True


# ---------------------------------------------------------------------------
# SessionMap threading
# ---------------------------------------------------------------------------

class TestSessionMapThreading:
    def test_get_thread_for_parent_returns_none_by_default(self) -> None:
        sm = SessionMap()
        assert sm.get_thread_for_parent(123) is None

    def test_link_and_get_parent_thread(self) -> None:
        sm = SessionMap()
        sm.link_parent_thread(123, 999)
        assert sm.get_thread_for_parent(123) == 999

    def test_different_parents_different_threads(self) -> None:
        sm = SessionMap()
        sm.link_parent_thread(100, 901)
        sm.link_parent_thread(200, 902)
        assert sm.get_thread_for_parent(100) == 901
        assert sm.get_thread_for_parent(200) == 902

    def test_overwrite_parent_thread_link(self) -> None:
        sm = SessionMap()
        sm.link_parent_thread(123, 999)
        sm.link_parent_thread(123, 998)
        assert sm.get_thread_for_parent(123) == 998

    def test_clear_removes_parent_thread_links(self) -> None:
        sm = SessionMap()
        sm.link_parent_thread(123, 999)
        sm.clear()
        assert sm.get_thread_for_parent(123) is None


# ---------------------------------------------------------------------------
# Adapter — create_thread
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Adapter — create_thread (real discord.py offline)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("discord", reason="discord.py not installed"),
    reason="discord.py not installed",
)
class TestAdapterCreateThreadOffline:
    """Tests using real discord.py types (offline, no network)."""

    def test_is_thread_channel_returns_true_for_thread(self) -> None:
        import discord
        from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter

        adapter = DiscordBotAdapter(DiscordConfig(token="fake"))
        mock_thread = MagicMock(spec=discord.Thread)
        assert adapter.is_thread_channel(mock_thread) is True

    def test_is_thread_channel_returns_false_for_text_channel(self) -> None:
        import discord
        from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter

        adapter = DiscordBotAdapter(DiscordConfig(token="fake"))
        mock_channel = MagicMock(spec=discord.TextChannel)
        assert adapter.is_thread_channel(mock_channel) is False


# ---------------------------------------------------------------------------
# Interface — auto_thread flow
# ---------------------------------------------------------------------------

def _make_mock_agent(events: list | None = None) -> MagicMock:
    """Create a mock Agent with configurable stream events."""
    agent = MagicMock()
    agent.history = []
    agent.session_id = "discord-test"
    if events is None:
        events = [AgentTextDelta(text="OK"), AgentDone()]

    async def _stream(text: str) -> Any:
        for ev in events:
            yield ev

    agent.stream_message = _stream
    return agent


def _make_mock_message(
    content: str = "hello",
    channel_id: int = 12345,
    message_id: int = 77777,
    is_thread: bool = False,
) -> MagicMock:
    """Create a mock discord.Message."""
    message = MagicMock()
    message.content = content
    message.id = message_id

    channel = MagicMock()
    channel.id = channel_id
    message.channel = channel

    # Mock is_thread_channel behavior
    channel._is_thread = is_thread
    message.channel._is_thread = is_thread

    author = MagicMock()
    author.id = 99999
    message.author = author

    return message


def _make_mock_adapter() -> MagicMock:
    """Create a fully async-compatible mock adapter."""
    mock_adapter = MagicMock()
    mock_adapter.bot_id = 11111
    mock_adapter.bot_mentions = ["<@11111>"]
    mock_adapter.start = AsyncMock()
    mock_adapter.stop = AsyncMock()
    mock_adapter.send_message = AsyncMock(return_value=MagicMock(id=888))
    mock_adapter.edit_message = AsyncMock()
    mock_adapter.send_embed = AsyncMock()
    mock_adapter.set_message_handler = MagicMock()
    mock_adapter.create_thread = AsyncMock()
    mock_adapter.is_thread_channel = MagicMock()
    return mock_adapter


@pytest.mark.asyncio
async def test_auto_thread_disabled_returns_same_channel() -> None:
    """When auto_thread=False, target channel is the message's channel."""
    interface = DiscordInterface(DiscordConfig(auto_thread=False))
    agent = _make_mock_agent()

    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter"
    ) as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter
        await interface.start(agent)

        msg = _make_mock_message(channel_id=123)
        target = await interface._resolve_target_channel(
            msg, msg.channel.id, msg.channel
        )
        assert target == 123
        assert not mock_adapter.create_thread.called


@pytest.mark.asyncio
async def test_auto_thread_enabled_creates_thread_for_parent_channel() -> None:
    """When auto_thread=True and message is in a parent channel, create a thread."""
    interface = DiscordInterface(DiscordConfig(auto_thread=True))
    agent = _make_mock_agent()

    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter"
    ) as MockAdapter:
        mock_adapter = _make_mock_adapter()
        mock_adapter.is_thread_channel.return_value = False
        mock_adapter.create_thread.return_value = 99999
        MockAdapter.return_value = mock_adapter
        await interface.start(agent)

        msg = _make_mock_message(
            content="Help me with factorio",
            channel_id=123,
            message_id=777,
        )
        target = await interface._resolve_target_channel(
            msg, msg.channel.id, msg.channel
        )
        assert target == 99999

        # Verify thread creation was called with correct args
        mock_adapter.create_thread.assert_called_once()
        call_kwargs = mock_adapter.create_thread.call_args
        assert call_kwargs[0][0] == 123  # channel_id
        assert call_kwargs[0][1] == "Help me with factorio"  # name
        assert call_kwargs[0][2] == 777  # message_id

        # Verify parent->thread link was stored
        assert interface._session_map.get_thread_for_parent(123) == 99999


@pytest.mark.asyncio
async def test_auto_thread_existing_thread_reused() -> None:
    """When a thread already exists for the parent, reuse it instead of creating a new one."""
    interface = DiscordInterface(DiscordConfig(auto_thread=True))
    agent = _make_mock_agent()

    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter"
    ) as MockAdapter:
        mock_adapter = _make_mock_adapter()
        mock_adapter.is_thread_channel.return_value = False
        MockAdapter.return_value = mock_adapter
        await interface.start(agent)

        # Pre-link a thread
        interface._session_map.link_parent_thread(123, 88888)

        msg = _make_mock_message(channel_id=123)
        target = await interface._resolve_target_channel(
            msg, msg.channel.id, msg.channel
        )
        assert target == 88888
        # Should NOT create a new thread
        assert not mock_adapter.create_thread.called


@pytest.mark.asyncio
async def test_auto_thread_message_inside_thread_returns_thread_id() -> None:
    """When message is already inside a thread, use the thread ID directly."""
    interface = DiscordInterface(DiscordConfig(auto_thread=True))
    agent = _make_mock_agent()

    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter"
    ) as MockAdapter:
        mock_adapter = _make_mock_adapter()
        mock_adapter.is_thread_channel.return_value = True
        MockAdapter.return_value = mock_adapter
        await interface.start(agent)

        msg = _make_mock_message(channel_id=99999)
        target = await interface._resolve_target_channel(
            msg, msg.channel.id, msg.channel
        )
        assert target == 99999
        assert not mock_adapter.create_thread.called


@pytest.mark.asyncio
async def test_auto_thread_fallback_on_create_failure() -> None:
    """When thread creation fails, fall back to parent channel."""
    interface = DiscordInterface(DiscordConfig(auto_thread=True))
    agent = _make_mock_agent()

    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter"
    ) as MockAdapter:
        mock_adapter = _make_mock_adapter()
        mock_adapter.is_thread_channel.return_value = False
        mock_adapter.create_thread.return_value = None  # Failure
        MockAdapter.return_value = mock_adapter
        await interface.start(agent)

        msg = _make_mock_message(channel_id=123)
        target = await interface._resolve_target_channel(
            msg, msg.channel.id, msg.channel
        )
        # Falls back to parent channel
        assert target == 123


@pytest.mark.asyncio
async def test_full_flow_reply_routed_to_thread() -> None:
    """End-to-end: message in parent -> thread created -> reply sent to thread."""
    interface = DiscordInterface(DiscordConfig(auto_thread=True))
    agent = _make_mock_agent([AgentTextDelta(text="Answer!"), AgentDone()])

    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter"
    ) as MockAdapter:
        mock_adapter = _make_mock_adapter()
        mock_adapter.is_thread_channel.return_value = False
        mock_adapter.create_thread.return_value = 55555
        MockAdapter.return_value = mock_adapter
        await interface.start(agent)

        msg = _make_mock_message(
            content="What's 2+2?",
            channel_id=100,
            message_id=200,
        )
        await interface._process_message(msg)

        # Thread should have been created
        mock_adapter.create_thread.assert_called_once_with(
            100, "What's 2+2?", 200
        )

        # Thinking message should be sent to the thread (55555), not parent (100)
        send_calls = mock_adapter.send_message.call_args_list
        assert len(send_calls) >= 1
        # First send_message call should target the thread
        first_call = send_calls[0]
        assert first_call[0][0] == 55555

        # Edit should also target the thread
        edit_calls = mock_adapter.edit_message.call_args_list
        assert len(edit_calls) >= 1
        first_edit = edit_calls[0]
        assert first_edit[0][0] == 55555
