"""DiscordInterface — the Interface protocol implementation for Discord.

Implements the Interface protocol (name, start, stop, run) to connect
SR2 Spectre to Discord via discord.py.

Architecture:
- DiscordInterface: Implements the Interface protocol. Controls lifecycle.
- DiscordBotAdapter: Wraps discord.py Bot client (start/stop/send/edit).
- SessionMap: Maps channel_id → conversation history (session-per-channel).
- MessageHandler: Pure logic for routing, commands, chunking.

Data flow:
1. User sends message in Discord channel
2. Adapter receives discord.Message → passes to Interface
3. Interface checks: mention filter, channel filter, slash commands
4. For regular messages: Agent.stream_message() with channel-specific history
5. Response streamed via message edits (configurable)
6. Tool execution shown as embeds (configurable)
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import AgentDone, AgentTextDelta, AgentToolResult, AgentToolStart
from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.handler import (
    chunk_message,
    handle_command,
    parse_slash_command,
    should_respond,
)
from sr2_spectre.interfaces.discord.session_map import SessionMap

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent

logger = logging.getLogger(__name__)


class DiscordInterface:
    """Discord interface for SR2 Spectre.

    Implements the Interface protocol. Each Discord channel maintains
    its own conversation history (session-per-channel isolation).
    """
    name = "discord"

    def __init__(self, config: DiscordConfig | None = None) -> None:
        self.config = config or DiscordConfig()
        self._agent: Agent | None = None
        self._adapter: DiscordBotAdapter | None = None
        self._session_map = SessionMap()
        self._running = False

    async def start(self, agent: "Agent") -> None:
        """Initialize the Discord interface and start the bot.

        Args:
            agent: The Spectre Agent instance.
        """
        self._agent = agent
        self._adapter = DiscordBotAdapter(self.config)
        self._running = True

        # Set interactive run context for Discord
        agent.set_run_context(RunContext(
            interface="discord",
            mode=RunMode.INTERACTIVE,
            source=None,  # channel-specific source set per-message in handler
        ))

        # Set up the message handler
        async def _handle_message(message: Any) -> None:
            await self._process_message(message)

        self._adapter.set_message_handler(_handle_message)

        await self._adapter.start()

    async def stop(self) -> None:
        """Stop the Discord bot and clean up sessions."""
        self._running = False
        if self._adapter is not None:
            await self._adapter.stop()
        self._session_map.clear()

    async def run(self, agent: "Agent") -> None:
        """Run the Discord bot loop.

        This blocks until stop() is called. The adapter handles the
        actual discord.py event loop internally.
        """
        if self._adapter is None:
            await self.start(agent)

        # Run the bot — this blocks until the bot is stopped
        await self._adapter.run()

    async def _resolve_target_channel(
        self,
        message: Any,
        channel_id: int,
        channel_obj: Any,
    ) -> int:
        """Resolve the target channel ID for the response.

        When auto_thread is disabled, returns the message's channel ID
        (the identity path).

        When auto_thread is enabled:
        - If the message is already in a thread, return the thread ID.
        - If the message is in a parent channel and no thread session
          exists yet, create a thread and return the thread ID.
        - If the message is in a parent channel and a thread session
          already exists, return the existing thread ID.

        Args:
            message: discord.Message object.
            channel_id: The message's channel ID.
            channel_obj: The message's channel object.

        Returns:
            The channel ID to send responses to.
        """
        if not self.config.auto_thread:
            return channel_id

        # If we're already inside a thread, use it as-is
        if self._adapter and self._adapter.is_thread_channel(channel_obj):
            return channel_id

        # We're in a parent channel — check if there's an existing thread
        # for this channel's session
        existing = self._session_map.get_thread_for_parent(channel_id)
        if existing is not None:
            return existing

        # No thread yet — create one anchored on the user's message
        message_id = getattr(message, "id", None)
        if message_id is None or self._adapter is None:
            return channel_id

        # Build thread name from the first line of the message
        content = getattr(message, "content", "")
        first_line = content.strip().split("\n")[0][:70]
        thread_name = first_line or "SR2 conversation"

        thread_id = await self._adapter.create_thread(
            channel_id, thread_name, message_id
        )
        if thread_id is not None:
            self._session_map.link_parent_thread(channel_id, thread_id)
            return thread_id

        # Thread creation failed — fall back to parent channel
        logger.warning(
            "Thread creation failed, falling back to parent channel %d",
            channel_id,
        )
        return channel_id

    async def _process_message(self, message: Any) -> None:
        """Process an incoming Discord message.

        Extracts channel_id, content, and author from the discord.Message
        object, then routes through the handler logic.

        When auto_thread is enabled, determines the effective channel ID:
        if the message is already in a thread, use the thread ID;
        if the message is in a parent channel, create a thread and route
        to the thread ID.

        Args:
            message: discord.Message object from discord.py.
        """
        if self._adapter is None or self._agent is None:
            return

        # Extract plain data from discord.Message
        channel_obj = getattr(message, "channel", None)
        if channel_obj is None:
            logger.warning("Could not extract channel from message")
            return

        channel_id = getattr(channel_obj, "id", None)
        if channel_id is None:
            logger.warning("Could not extract channel_id from message")
            return

        content = getattr(message, "content", "")
        if not content:
            return

        bot_id = self._adapter.bot_id
        bot_mentions = self._adapter.bot_mentions

        # Check if we should respond to this message
        if not should_respond(content, self.config.mention_only, bot_id, bot_mentions):
            return

        # Parse slash commands
        command, rest = parse_slash_command(content)

        if command is not None:
            await self._handle_command(command, rest, channel_id)
            return

        # Determine effective channel for threading
        target_channel_id = await self._resolve_target_channel(
            message, channel_id, channel_obj
        )

        # Regular message — process through the agent
        await self._process_through_agent(content, target_channel_id)

    async def _handle_command(
        self,
        command: str,
        rest: str,
        channel_id: int,
    ) -> None:
        """Process a slash command.

        Args:
            command: Command name (already lowercase).
            rest: Remainder of message content.
            channel_id: Discord channel ID.
        """
        response = handle_command(command, rest)

        if command == "reset":
            self._session_map.reset(channel_id)

        if response is not None:
            if self._adapter is not None:
                chunks = chunk_message(response, self.config.max_message_length)
                for chunk in chunks:
                    await self._adapter.send_message(channel_id, chunk)

        # /ask with content — process through agent
        if command == "ask" and rest.strip():
            await self._process_through_agent(rest, channel_id)

    async def _process_through_agent(
        self,
        content: str,
        channel_id: int,
    ) -> None:
        """Route a message through the Agent and stream the response.

        Orchestrates the full flow: session setup, agent call with stream
        rendering, history management, and final message delivery.

        Args:
            content: The user's message text.
            channel_id: Discord channel ID.
        """
        if self._adapter is None or self._agent is None:
            return

        session = self._session_map.get_or_create(channel_id)

        # Append user message to channel history
        session.history.append({"role": "user", "content": [{"type": "text", "text": content}]})

        # Send initial placeholder message for streaming
        thinking_msg = await self._adapter.send_message(channel_id, "⏳ Thinking...")
        thinking_id = getattr(thinking_msg, "id", None) if thinking_msg else None
        session.pending_message_id = thinking_id

        # Restore channel history into the agent
        self._restore_history(session)

        # Drive the agent and render stream events to Discord
        response_parts, stream_error = await self._drive_agent_stream(
            content, channel_id, thinking_id
        )

        if stream_error is not None:
            session.pending_message_id = None
            return

        # Finalize: set the complete response
        final_text = "".join(response_parts)
        if not final_text:
            final_text = "(No response generated)"

        # Append assistant response to channel history
        session.history.append(
            {"role": "assistant", "content": [{"type": "text", "text": final_text}]}
        )

        # Send final message(s) — chunked if too long
        if thinking_id is not None and self._adapter:
            chunks = chunk_message(final_text, self.config.max_message_length)
            if len(chunks) == 1:
                # Single chunk — edit the thinking message
                await self._adapter.edit_message(channel_id, thinking_id, chunks[0])
            else:
                # Multiple chunks — edit first, send rest as new messages
                await self._adapter.edit_message(channel_id, thinking_id, chunks[0])
                for chunk in chunks[1:]:
                    await self._adapter.send_message(channel_id, chunk)

        session.pending_message_id = None

    async def _drive_agent_stream(
        self,
        content: str,
        channel_id: int,
        thinking_id: int | None,
    ) -> tuple[list[str], Exception | None]:
        """Drive the agent's message stream and render events to Discord.

        Handles the event loop: collects text deltas, sends stream edits,
        renders tool start/result embeds. Separates the Discord rendering
        concern from the orchestration in _process_through_agent.

        Args:
            content: The user's message text.
            channel_id: Discord channel ID.
            thinking_id: ID of the thinking placeholder message.

        Returns:
            Tuple of (collected response text parts, error or None).
            If error is not None, the thinking message was already updated
            with the error and no finalization should occur.
        """
        response_parts: list[str] = []
        last_edit_time: float | None = None
        loop = asyncio.get_event_loop()

        try:
            async for event in self._agent.stream_message(content):
                if isinstance(event, AgentTextDelta):
                    response_parts.append(event.text)
                    self._maybe_edit_stream(
                        channel_id, thinking_id,
                        "".join(response_parts),
                        last_edit_time, loop,
                    )
                    last_edit_time = loop.time()

                elif isinstance(event, AgentToolStart):
                    await self._render_tool_start(event, channel_id)

                elif isinstance(event, AgentToolResult):
                    await self._render_tool_result(event, channel_id)

                # AgentDone is handled after the loop

        except Exception as exc:
            logger.error("Agent stream error: %s", exc)
            error_msg = f"Error: {exc}"
            if thinking_id is not None and self._adapter:
                await self._adapter.edit_message(channel_id, thinking_id, error_msg)
            return response_parts, exc

        return response_parts, None

    async def _render_tool_start(
        self,
        event: AgentToolStart,
        channel_id: int,
    ) -> None:
        """Render a tool-start event as a Discord embed.

        Only sends an embed when tool_embed_enabled is True in the config.

        Args:
            event: The AgentToolStart event.
            channel_id: Discord channel ID.
        """
        if not self.config.tool_embed_enabled or self._adapter is None:
            return

        embed = {
            "title": f"🔧 {event.name}",
            "description": "Running...",
            "color": 16753920,  # Yellow
        }
        await self._adapter.send_embed(channel_id, embed)

    async def _render_tool_result(
        self,
        event: AgentToolResult,
        channel_id: int,
    ) -> None:
        """Render a tool-result event as a Discord embed.

        Only sends an embed when tool_embed_enabled is True in the config.

        Args:
            event: The AgentToolResult event.
            channel_id: Discord channel ID.
        """
        if not self.config.tool_embed_enabled or self._adapter is None:
            return

        from sr2_spectre.interfaces.discord.handler import build_tool_embed

        tool_name = event.name or "tool"
        status = "failed" if event.is_error else "completed"
        embed = build_tool_embed(
            tool_name, status, error="Error" if event.is_error else None
        )
        await self._adapter.send_embed(channel_id, embed)

    def _maybe_edit_stream(
        self,
        channel_id: int,
        message_id: int | None,
        current_text: str,
        last_edit_time: float | None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Edit the streaming message if enough time has passed.

        Uses asyncio.ensure_future to avoid blocking the event loop.

        Args:
            channel_id: Discord channel ID.
            message_id: ID of the message being edited.
            current_text: Current accumulated response text.
            last_edit_time: Time of the last edit (None for first edit).
            loop: Current asyncio event loop.
        """
        if message_id is None or self._adapter is None:
            return

        if self.config.edit_stream_interval <= 0:
            return  # Streaming edits disabled

        now = loop.time()
        if last_edit_time is None or (now - last_edit_time) >= self.config.edit_stream_interval:
            truncated = current_text + "..."
            truncated = truncated[: self.config.max_message_length]
            asyncio.ensure_future(
                self._adapter.edit_message(channel_id, message_id, truncated)
            )

    def _restore_history(self, session: "ChannelSession") -> None:
        """Restore channel history into the agent's history list.

        Replaces the agent's current history with the channel's
        conversation history, so the agent has context for this
        specific channel's conversation.

        Args:
            session: The ChannelSession containing serialized history.
        """
        from sr2.models import Message, TextBlock

        # Agent.history is a getter-only property (delegates to Session) — never
        # assign to it. Setting session_id rebuilds a fresh Session (empty
        # history) under the shared Runtime, which both resets history and
        # points the agent at this channel's frame.
        self._agent.session_id = session.session_id

        for entry in session.history:
            role = entry.get("role", "user")
            blocks = []
            for block in entry.get("content", []):
                if block.get("type") == "text":
                    blocks.append(TextBlock(text=block.get("text", "")))
            if blocks:
                self._agent.history.append(Message(role=role, content=blocks))
