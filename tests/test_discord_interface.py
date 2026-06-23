"""Tests for DiscordInterface — Interface protocol implementation.

Tests the interface lifecycle and message routing logic using mocked
adapter objects. Does NOT require discord.py installed.

Covers:
1.  Interface protocol (name, start, stop, run)
2.  Message routing (mention filter, channel filter)
3.  Slash command processing (/reset, /help, /status, /ask)
4.  Agent stream integration (text delta collection, history management)
5.  Session isolation per channel
6.  Error handling during agent streaming
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.events import AgentDone, AgentTextDelta, AgentToolResult, AgentToolStart
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.interface import DiscordInterface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_agent(events: list | None = None) -> MagicMock:
    """Create a mock Agent with configurable stream events."""
    agent = MagicMock()
    agent.history = []
    agent.session_id = "discord-test"

    if events is None:
        events = [AgentTextDelta(text="Hello!"), AgentDone(tool_calls_executed=0)]

    async def _stream(text: str) -> Any:
        for ev in events:
            yield ev

    agent.stream_message = _stream
    return agent


def _make_mock_message(
    content: str = "hello",
    channel_id: int = 12345,
) -> MagicMock:
    """Create a mock discord.Message."""
    message = MagicMock()
    message.content = content

    channel = MagicMock()
    channel.id = channel_id
    message.channel = channel

    author = MagicMock()
    author.id = 99999
    message.author = author

    return message


def _make_mock_adapter(is_thread: bool = False) -> MagicMock:
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
    mock_adapter.is_thread_channel = MagicMock(return_value=is_thread)

    # channel_typing returns an async context manager (no-op in tests)
    typing_ctx = AsyncMock()
    typing_ctx.__aenter__ = AsyncMock()
    typing_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_adapter.channel_typing = MagicMock(return_value=typing_ctx)

    return mock_adapter


# ---------------------------------------------------------------------------
# Interface protocol
# ---------------------------------------------------------------------------

class TestInterfaceProtocol:
    def test_name_attribute(self) -> None:
        interface = DiscordInterface()
        assert interface.name == "discord"

    def test_default_config(self) -> None:
        interface = DiscordInterface()
        assert isinstance(interface.config, DiscordConfig)
        assert interface.config.token == ""

    def test_custom_config(self) -> None:
        config = DiscordConfig(token="test-token")
        interface = DiscordInterface(config=config)
        assert interface.config.token == "test-token"


@pytest.mark.asyncio
async def test_start_initializes_adapter() -> None:
    """start() creates adapter, sets handler, calls adapter.start()."""
    config = DiscordConfig(token="test")
    interface = DiscordInterface(config=config)
    agent = _make_mock_agent()

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        MockAdapter.assert_called_once_with(config)
        mock_adapter.start.assert_called_once()


@pytest.mark.asyncio
async def test_stop_cleans_up() -> None:
    """stop() calls adapter.stop() and clears sessions."""
    config = DiscordConfig(token="test")
    interface = DiscordInterface(config=config)

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(_make_mock_agent())
        await interface.stop()

        mock_adapter.stop.assert_called_once()
        assert interface._session_map.active() == []


# ---------------------------------------------------------------------------
# Message routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ignores_empty_messages() -> None:
    """Messages with empty content are ignored."""
    interface = DiscordInterface(DiscordConfig(token="test"))
    agent = _make_mock_agent()

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="")
        await interface._process_message(msg)

        # send_message should NOT have been called
        assert not mock_adapter.send_message.called


@pytest.mark.asyncio
async def test_responds_to_regular_message() -> None:
    """Non-mention messages trigger agent when mention_only is False."""
    interface = DiscordInterface(DiscordConfig(mention_only=False))
    agent = _make_mock_agent([AgentTextDelta(text="Hi!"), AgentDone()])

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="hello bot")
        await interface._process_message(msg)

        assert mock_adapter.send_message.called
        assert mock_adapter.edit_message.called


@pytest.mark.asyncio
async def test_mention_filter_blocks_non_mentions() -> None:
    """With mention_only=True, messages without mentions are ignored."""
    interface = DiscordInterface(DiscordConfig(mention_only=True))
    agent = _make_mock_agent()

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="hello bot")  # No mention
        await interface._process_message(msg)

        assert not mock_adapter.send_message.called


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slash_resets_channel_session() -> None:
    """/reset clears the channel's conversation history."""
    interface = DiscordInterface(DiscordConfig())
    agent = _make_mock_agent()

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        # First populate history
        session = interface._session_map.get_or_create(12345)
        session.history.append({"role": "user", "content": []})
        assert len(session.history) == 1

        msg = _make_mock_message(content="/reset", channel_id=12345)
        await interface._process_message(msg)

        assert session.history == []


@pytest.mark.asyncio
async def test_slash_help_sends_help_text() -> None:
    """/help sends the help text to the channel."""
    interface = DiscordInterface(DiscordConfig())
    agent = _make_mock_agent()

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="/help")
        await interface._process_message(msg)

        call_args = mock_adapter.send_message.call_args
        assert "/ask" in call_args[0][1]


@pytest.mark.asyncio
async def test_slash_hb_posts_probe_without_agent() -> None:
    """/hb runs the Harbinger probe and posts it, bypassing the agent."""
    interface = DiscordInterface(DiscordConfig())
    agent = _make_mock_agent()

    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter"
    ) as MockAdapter, patch(
        "sr2_spectre.interfaces.discord.interface.probe_harbinger_status",
        new=AsyncMock(return_value="```\nLive slots: busy=1\n```"),
    ) as mock_probe:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="/hb", channel_id=777)
        await interface._process_message(msg)

        mock_probe.assert_awaited_once()
        sent = mock_adapter.send_message.call_args[0][1]
        assert "Live slots: busy=1" in sent
        # Bypasses the LLM: the agent path would first post "⏳ Thinking…"
        # and send multiple messages; the probe posts exactly one.
        assert mock_adapter.send_message.call_count == 1
        thinking_sent = any(
            "Thinking" in c.args[1] for c in mock_adapter.send_message.call_args_list
        )
        assert not thinking_sent


# ---------------------------------------------------------------------------
# Agent stream integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_collects_text_and_sends_final() -> None:
    """Text deltas are collected, final response edits the thinking message."""
    interface = DiscordInterface(DiscordConfig())
    events = [
        AgentTextDelta(text="Hello"),
        AgentTextDelta(text=", "),
        AgentTextDelta(text="world!"),
        AgentDone(tool_calls_executed=0),
    ]
    agent = _make_mock_agent(events)

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="say hello", channel_id=54321)
        await interface._process_message(msg)

        # Final edit should contain the complete response
        edit_calls = mock_adapter.edit_message.call_args_list
        assert len(edit_calls) > 0
        final_content = edit_calls[-1][0][2]
        assert "Hello, world!" in final_content


@pytest.mark.asyncio
async def test_history_isolated_per_channel() -> None:
    """Each channel maintains its own conversation history."""
    interface = DiscordInterface(DiscordConfig())
    agent = _make_mock_agent([AgentTextDelta(text="OK"), AgentDone()])

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        # Message in channel 1
        await interface._process_message(_make_mock_message(content="msg1", channel_id=1))
        s1 = interface._session_map.get_or_create(1)
        assert len(s1.history) == 2  # user + assistant

        # Message in channel 2
        await interface._process_message(_make_mock_message(content="msg2", channel_id=2))
        s2 = interface._session_map.get_or_create(2)
        assert len(s2.history) == 2

        # Channel 1 should still have its own history
        assert s1.history[0]["content"][0]["text"] == "msg1"
        assert s2.history[0]["content"][0]["text"] == "msg2"


@pytest.mark.asyncio
async def test_long_response_chunked() -> None:
    """Responses over max_message_length are split into multiple messages."""
    interface = DiscordInterface(DiscordConfig(max_message_length=100))
    long_text = "x" * 250
    agent = _make_mock_agent([AgentTextDelta(text=long_text), AgentDone()])

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="tell me something long")
        await interface._process_message(msg)

        # Should have at least an edit + extra send messages
        assert mock_adapter.send_message.call_count >= 1
        assert mock_adapter.edit_message.call_count >= 1


@pytest.mark.asyncio
async def test_agent_error_handled_gracefully() -> None:
    """Agent errors don't crash the interface; error message is sent."""
    async def _failing_stream(text: str) -> Any:
        yield AgentTextDelta(text="partial")
        raise RuntimeError("LLM is down")

    agent = MagicMock()
    agent.history = []
    agent.session_id = "test"
    agent.stream_message = _failing_stream

    interface = DiscordInterface(DiscordConfig())

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="hello")
        await interface._process_message(msg)

        # Should have edited the thinking message with error
        edit_calls = mock_adapter.edit_message.call_args_list
        assert len(edit_calls) > 0
        final_content = edit_calls[-1][0][2]
        assert "Error" in final_content


@pytest.mark.asyncio
async def test_slash_ask_routes_to_agent() -> None:
    """/ask with content routes through the agent."""
    interface = DiscordInterface(DiscordConfig())
    agent = _make_mock_agent([AgentTextDelta(text="Answer"), AgentDone()])

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="/ask what is the weather?")
        await interface._process_message(msg)

        # send_message should have been called (thinking message + final)
        assert mock_adapter.send_message.called


@pytest.mark.asyncio
async def test_slash_ask_resolves_thread_channel() -> None:
    """/ask resolves the target channel through _resolve_target_channel,
    so it routes to the thread when auto_thread is enabled.

    Regression: previously /ask sent to the raw message channel (parent),
    bypassing thread resolution. This caused messages to split across
    parent and thread.
    """
    parent_id = 11111
    thread_id = 22222
    interface = DiscordInterface(DiscordConfig(auto_thread=True))
    agent = _make_mock_agent([AgentTextDelta(text="Answer"), AgentDone()])

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter(is_thread=False)
        MockAdapter.return_value = mock_adapter
        # Pre-link the parent to a thread
        mock_adapter.create_thread = AsyncMock(return_value=thread_id)

        await interface.start(agent)

        msg = _make_mock_message(content="/ask what is the weather?", channel_id=parent_id)
        await interface._process_message(msg)

        # The parent-channel mention creates a fresh thread; send_message
        # should target that THREAD, not the parent.
        send_calls = mock_adapter.send_message.call_args_list
        for call in send_calls:
            target_channel = call[0][0]
            assert target_channel == thread_id, (
                f"/ask sent to channel {target_channel} instead of thread {thread_id}"
            )


@pytest.mark.asyncio
async def test_tool_events_merged_into_thinking_message() -> None:
    """Tool start/result events accumulate in the thinking message itself,
    not a separate tool-log message.

    Regression: previously each tool event sent a separate embed message,
    then a separate tool-log message. Now everything lives in one message.
    """
    interface = DiscordInterface(DiscordConfig(tool_embed_enabled=True))
    events = [
        AgentToolStart(tool_id="t1", name="grep", input={"pattern": "test"}),
        AgentToolResult(tool_id="t1", name="grep", content="found it", is_error=False),
        AgentToolStart(tool_id="t2", name="file_read", input={"path": "x.py"}),
        AgentToolResult(tool_id="t2", name="file_read", content="contents", is_error=False),
        AgentTextDelta(text="Done"),
        AgentDone(),
    ]
    agent = _make_mock_agent(events)

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="search for something")
        await interface._process_message(msg)

        # send_embed should NOT be called (no embeds, plain text in thinking msg)
        assert not mock_adapter.send_embed.called

        # Only ONE send_message call: the "⏳ Thinking..." placeholder.
        # Tool lines are flushed via edit_message, not separate sends.
        send_calls = [c for c in mock_adapter.send_message.call_args_list]
        assert mock_adapter.send_message.call_count == 1
        assert send_calls[0][0][1] == "⏳ Thinking..."

        # The thinking message was edited to include tool lines
        edit_calls = mock_adapter.edit_message.call_args_list
        tool_edits = [c for c in edit_calls if "▶" in c[0][2]]
        assert len(tool_edits) > 0, "Expected at least one edit containing tool lines"

        # Verify tool lines accumulated (last tool edit should have all events)
        last_tool_content = tool_edits[-1][0][2]
        assert "`grep`" in last_tool_content
        assert "`file_read`" in last_tool_content
        assert "✓" in last_tool_content  # tool success marker


@pytest.mark.asyncio
async def test_interleaved_text_and_tools_do_not_clobber() -> None:
    """In-progress edits show tool lines AND streamed text together.

    Regression (the "edited back and forth" bug): the tool path rendered
    "tool lines + Thinking..." while the text path rendered "text + ...",
    each on the SAME thinking message. Interleaved events made the message
    flip between tool-only and text-only content. Now both funnel through a
    single renderer, so any in-progress edit carrying the streamed text must
    also still carry the accumulated tool lines.
    """
    interface = DiscordInterface(DiscordConfig(tool_embed_enabled=True))
    events = [
        AgentToolStart(tool_id="t1", name="grep", input={"pattern": "x"}),
        AgentToolResult(tool_id="t1", name="grep", content="hit", is_error=False),
        AgentTextDelta(text="Answer"),
        AgentDone(),
    ]
    agent = _make_mock_agent(events)

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="search then answer", channel_id=4242)
        await interface._process_message(msg)

        edit_contents = [c[0][2] for c in mock_adapter.edit_message.call_args_list]

        # Every IN-PROGRESS edit (marked with "...") that shows the streamed
        # text must also retain the tool line — never one without the other.
        in_progress_with_text = [c for c in edit_contents if "Answer..." in c]
        assert in_progress_with_text, "Expected an in-progress edit carrying the streamed text"
        for content in in_progress_with_text:
            assert "`grep`" in content, (
                f"In-progress edit dropped tool lines: {content!r} "
                f"(the back-and-forth clobber bug)"
            )


@pytest.mark.asyncio
async def test_tool_log_suppressed_when_disabled() -> None:
    """When tool_embed_enabled=False, no tool log messages are sent."""
    interface = DiscordInterface(DiscordConfig(tool_embed_enabled=False))
    events = [
        AgentToolStart(tool_id="t1", name="search", input={"q": "test"}),
        AgentToolResult(tool_id="t1", name="search", content="result", is_error=False),
        AgentTextDelta(text="Done"),
        AgentDone(),
    ]
    agent = _make_mock_agent(events)

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="search for something")
        await interface._process_message(msg)

        # Only the "⏳ Thinking..." message should be sent (no tool log)
        send_calls = mock_adapter.send_message.call_args_list
        thinking_sends = [c for c in send_calls if c[0][1] == "⏳ Thinking..."]
        assert len(thinking_sends) == 1
        # No additional messages beyond thinking + final answer path
        non_thinking_sends = [c for c in send_calls if c[0][1] != "⏳ Thinking..."]
        assert len(non_thinking_sends) == 0


# ---------------------------------------------------------------------------
# Streaming edit race condition
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_edit_does_not_overwrite_final_message() -> None:
    """The last streaming edit (with '...') must not overwrite the finalized message.

    Regression test: when the stream ends quickly, a pending ensure_future
    from _maybe_edit_stream could resolve after finalization and overwrite
    the clean final text with the '...' version.
    """
    import asyncio

    interface = DiscordInterface(
        DiscordConfig(edit_stream_interval=0.1, max_message_length=2000)
    )
    events = [
        AgentTextDelta(text="Hey"),
        AgentDone(tool_calls_executed=0),
    ]
    agent = _make_mock_agent(events)

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        msg = _make_mock_message(content="heya", channel_id=54321)
        await interface._process_message(msg)

        # The LAST edit_message call must be the clean final text (no "...")
        edit_calls = mock_adapter.edit_message.call_args_list
        assert len(edit_calls) > 0
        final_content = edit_calls[-1][0][2]
        assert final_content == "Hey", (
            f"Final message was '{final_content}' — the '...' streaming edit "
            f"overwrote the clean final text (race condition not fixed)"
        )
        assert "..." not in final_content


@pytest.mark.asyncio
async def test_cancel_pending_stream_edit_clears_future() -> None:
    """_cancel_pending_stream_edit cancels the tracked future."""
    import asyncio

    interface = DiscordInterface()

    # Simulate a pending future
    future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    interface._pending_stream_edit = future

    interface._cancel_pending_stream_edit()

    assert future.cancelled()
    assert interface._pending_stream_edit is None


# ---------------------------------------------------------------------------
# Thread-aware mention bypass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mention_bypassed_in_thread_with_active_session() -> None:
    """Inside a thread where the agent has an active session, skip the mention check.

    When auto_thread is enabled and mention_only is True, the first message
    in a parent channel must mention the bot. But once inside a thread with
    an active session, follow-up messages should not require a mention.
    """
    thread_id = 99999
    interface = DiscordInterface(DiscordConfig(mention_only=True))
    agent = _make_mock_agent([AgentTextDelta(text="Follow-up answer"), AgentDone()])

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter(is_thread=True)
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        # Pre-create the session so the thread is recognized as "active"
        interface._session_map.get_or_create(thread_id)

        # Message WITHOUT mention in the thread
        msg = _make_mock_message(content="what about the other thing?", channel_id=thread_id)
        await interface._process_message(msg)

        # Should have responded (mention bypassed because active thread session)
        assert mock_adapter.send_message.called


@pytest.mark.asyncio
async def test_mention_still_required_in_thread_without_session() -> None:
    """A thread with no active session still requires a mention.

    If the bot hasn't started a conversation in a thread (no session exists),
    mention_only still applies.
    """
    thread_id = 77777
    interface = DiscordInterface(DiscordConfig(mention_only=True))
    agent = _make_mock_agent()

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter(is_thread=True)
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        # No session created for this thread_id
        msg = _make_mock_message(content="hello", channel_id=thread_id)
        await interface._process_message(msg)

        # Should NOT respond — no session, still needs mention
        assert not mock_adapter.send_message.called


@pytest.mark.asyncio
async def test_mention_still_required_in_parent_channel() -> None:
    """Parent channels always require a mention when mention_only is True,
    regardless of whether a thread session exists for them.
    """
    parent_id = 11111
    thread_id = 22222
    interface = DiscordInterface(DiscordConfig(mention_only=True))
    agent = _make_mock_agent()

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        # Parent channel is NOT a thread
        mock_adapter = _make_mock_adapter(is_thread=False)
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        # A parent channel with no mention must not respond, regardless of
        # any prior thread activity.
        interface._session_map.get_or_create(thread_id)

        msg = _make_mock_message(content="no mention here", channel_id=parent_id)
        await interface._process_message(msg)

        assert not mock_adapter.send_message.called


@pytest.mark.asyncio
async def test_mention_bypass_not_applied_when_mention_only_false() -> None:
    """When mention_only is False, the bypass is irrelevant — all messages respond."""
    interface = DiscordInterface(DiscordConfig(mention_only=False))
    agent = _make_mock_agent([AgentTextDelta(text="OK"), AgentDone()])

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter(is_thread=False)
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        # No session, not a thread, mention_only=False → still responds
        msg = _make_mock_message(content="anything at all", channel_id=55555)
        await interface._process_message(msg)

        assert mock_adapter.send_message.called


# ---------------------------------------------------------------------------
# /status — real session info
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slash_status_renders_session_info() -> None:
    """/status renders real per-channel session info (session_id + message count).

    Regression: previously /status returned a dead stub string that the
    interface never rendered. Now it shows actual session data.
    """
    channel_id = 77777
    interface = DiscordInterface(DiscordConfig())
    agent = _make_mock_agent()

    with patch("sr2_spectre.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter

        await interface.start(agent)

        # Pre-populate the session with some history
        session = interface._session_map.get_or_create(channel_id)
        session.history.append({"role": "user", "content": [{"type": "text", "text": "hello"}]})
        session.history.append({"role": "assistant", "content": [{"type": "text", "text": "hi"}]})

        msg = _make_mock_message(content="/status", channel_id=channel_id)
        await interface._process_message(msg)

        # The status response should contain the session ID and message count
        sent = mock_adapter.send_message.call_args[0][1]
        assert "discord-77777" in sent
        assert "2" in sent  # 2 messages in history
        assert "Session" in sent
        assert "Messages" in sent
