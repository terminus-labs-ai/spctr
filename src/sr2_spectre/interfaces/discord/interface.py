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
7. Local image paths in responses are uploaded and replaced with Discord URLs
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import AgentDone, AgentTextDelta, AgentToolResult, AgentToolStart
from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.config_source import (
    DiscordConfigProvider,
    DiscordConfigSource,
    DiscordConfigView,
)
from sr2_spectre.interfaces.discord.handler import (
    CommandContext,
    chunk_message,
    handle_command,
    parse_slash_command,
    probe_harbinger_status,
    render_model_command,
    resolve_area,
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

    def __init__(
        self,
        config: DiscordConfig | None = None,
        config_source: DiscordConfigProvider | None = None,
    ) -> None:
        """
        Args:
            config: A fixed config. Ignored when config_source is given.
            config_source: Live config source, re-read on every inbound
                message so config edits apply without a process restart.
                A ``DiscordConfigView`` additionally carries the whole
                SpectreConfig, letting model, endpoint, pipeline and tool
                edits reach the running agent — see ``_apply_agent_config``.
                When omitted, *config* is frozen for the life of the process.
        """
        self._config_source = config_source or DiscordConfigSource.static(config)
        self._agent: Agent | None = None
        self._adapter: DiscordBotAdapter | None = None
        self._session_map = SessionMap()
        self._running = False
        self._pending_stream_edit: asyncio.Future[None] | None = None
        # Per-channel tool activity lines — prepended to the thinking message
        # during streaming, then replaced by the final response on completion.
        self._tool_lines: dict[int, list[str]] = {}
        # Per-channel accumulated streamed text. Shared state so the unified
        # progress renderer can show tool lines AND text together — without it,
        # the tool path and text path clobber each other's content on the same
        # message (the "edited back and forth" bug).
        self._stream_text: dict[int, str] = {}
        # In-flight agent runs, keyed by channel, so /stop can cancel the
        # current turn. Set while a run is active, removed when it settles.
        self._active_runs: dict[int, asyncio.Task[None]] = {}

    @property
    def config(self) -> DiscordConfig:
        """The Discord config in force, as of the last reload.

        Reads through to the shared config source, so handling of a message
        sees the values the adapter loaded when that message arrived.
        """
        return self._config_source.current

    async def start(self, agent: "Agent") -> None:
        """Initialize the Discord interface and start the bot.

        Args:
            agent: The Spectre Agent instance.
        """
        self._agent = agent
        self._adapter = DiscordBotAdapter(self._config_source)
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

        # Wire native slash commands (must be set before adapter.start(), which
        # syncs the command tree in on_ready).
        self._adapter.set_slash_handler(self._handle_slash)

        await self._adapter.start()

    def _apply_agent_config(self) -> None:
        """Hand the agent the config the adapter just reloaded.

        Only meaningful on the ``DiscordConfigView`` path, where the reload
        covered the whole SpectreConfig. With a plain ``DiscordConfigSource``
        (tests, embedders) there is no agent config to apply and this is a
        no-op.
        """
        if self._agent is None or not isinstance(self._config_source, DiscordConfigView):
            return
        self._agent.apply_config(self._config_source.source.current)

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
        - If the message is already in a thread, return the thread ID
          (continuation of that conversation).
        - If the message is in a parent channel, ALWAYS create a fresh
          thread and return its ID. Each parent-channel mention is a new
          conversation. (Reusing one thread per parent funnels distinct
          topics into the first thread — the wrong-topic bug, obsidian-a3q.)

        Args:
            message: discord.Message object.
            channel_id: The message's channel ID.
            channel_obj: The message's channel object.

        Returns:
            The channel ID to send responses to.
        """
        logger.info(
            "resolve_target: auto_thread=%s, channel_id=%d, channel_type=%s",
            self.config.auto_thread,
            channel_id,
            type(channel_obj).__name__,
        )
        if not self.config.auto_thread:
            return channel_id

        # If we're already inside a thread, use it as-is (continuation)
        if self._adapter and self._adapter.is_thread_channel(channel_obj):
            return channel_id

        # We're in a parent channel — every mention starts a NEW thread.
        message_id = getattr(message, "id", None)
        if message_id is None or self._adapter is None:
            return channel_id

        # Build thread name from the first line of the message,
        # stripping bot mentions so "<@123456> deploy prod" → "deploy prod"
        content = getattr(message, "content", "")
        if self._adapter:
            for mention in self._adapter.bot_mentions:
                content = content.replace(mention, "")
        first_line = content.strip().split("\n")[0][:70]
        thread_name = first_line or "SR2 conversation"

        thread_id = await self._adapter.create_thread(
            channel_id, thread_name, message_id
        )
        if thread_id is not None:
            return thread_id

        # Thread creation failed — fall back to parent channel
        logger.warning(
            "Thread creation failed, falling back to parent channel %d",
            channel_id,
        )
        return channel_id

    def _stamp_area(self, channel_obj: Any) -> None:
        """Derive this message's area and stamp it on the agent's RunContext.

        Runs after ``should_respond`` and after ``_apply_agent_config()``, so
        the ``channel_areas`` map in force is the one this message loaded.
        Uses the message's own channel — not any thread later created for the
        reply — so a fresh auto-created thread never shifts the area.

        Discord always resolves an area: "no area" reaches ``RunContext`` as
        ``""``, never ``None`` — ``None`` means "this interface does not do
        areas" and would let ``PlanResolver`` fall through to
        ``SR2_PROJECT``/cwd, the silent-substitution bug this feature exists
        to remove.
        """
        if self._adapter is None or self._agent is None:
            return

        area_channel_id, area_channel_name = self._adapter.area_channel(channel_obj)
        channel_areas = self.config.channel_areas
        is_override = (
            area_channel_id is not None and str(area_channel_id) in channel_areas
        )
        resolved = resolve_area(area_channel_id, area_channel_name, channel_areas)

        self._agent.set_run_context(RunContext(
            interface="discord",
            mode=RunMode.INTERACTIVE,
            source=None,
            area=resolved if resolved is not None else "",
        ))

        if resolved is None:
            logger.info("area=none (channel=%s)", area_channel_id)
        elif is_override:
            logger.info(
                "area=%s (channel_areas override, channel=%s)",
                resolved, area_channel_id,
            )
        else:
            logger.info(
                "area=%s (derived from channel name, channel=%s)",
                resolved, area_channel_id,
            )

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

        # The adapter reloaded the config when this message arrived; push the
        # non-Discord half of that load into the agent before handling it.
        self._apply_agent_config()

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

        # Skip mention check if we're inside a thread where the agent
        # already has an active session — the conversation was started by
        # a mention (or mention_only was off), no need to mention again.
        in_active_thread = (
            self._adapter
            and self._adapter.is_thread_channel(channel_obj)
            and channel_id in self._session_map.active()
        )

        if not should_respond(
            content, self.config.mention_only and not in_active_thread, bot_id, bot_mentions
        ):
            return

        self._stamp_area(channel_obj)

        # Parse slash commands
        command, rest = parse_slash_command(content)

        if command is not None:
            target = await self._resolve_target_channel(
                message, channel_id, channel_obj
            )
            await self._handle_command(command, rest, target)
            return

        # Determine effective channel for threading
        target_channel_id = await self._resolve_target_channel(
            message, channel_id, channel_obj
        )

        # Regular message — process through the agent (cancellable via /stop)
        await self._run_agent(content, target_channel_id)

    def _command_context(self, channel_id: int) -> CommandContext:
        """Build a CommandContext, enriched with live model info when available.

        The whole-config path (DiscordConfigView) carries models and the
        active-model pointer, so /status can show what the agent is talking
        to. Other paths (tests, embedders) leave those fields None.
        """
        session = self._session_map.get_or_create(channel_id)
        active_model: str | None = None
        model_label: str | None = None
        if isinstance(self._config_source, DiscordConfigView):
            cfg = self._config_source.source.current
            active_model = cfg.active_model
            mc = cfg.active_model_config
            model_label = mc.model if mc.base_url is None else f"{mc.model} @ {mc.base_url}"
        return CommandContext(
            channel_id=channel_id,
            session_id=session.session_id,
            message_count=len(session.history),
            active_model=active_model,
            model_label=model_label,
        )

    async def _run_agent(self, content: str, channel_id: int) -> None:
        """Run one agent turn as a cancellable task so /stop can abort it.

        The turn runs as a tracked ``asyncio.Task`` keyed by channel. If /stop
        cancels it, ``CancelledError`` (a BaseException, so the stream loop's
        ``except Exception`` never swallows it) unwinds ``_process_through_agent``
        without finalizing; we then render the partial output plus a stop note.
        """
        task = asyncio.ensure_future(
            self._process_through_agent(content, channel_id)
        )
        self._active_runs[channel_id] = task
        try:
            await task
        except asyncio.CancelledError:
            await self._finalize_cancelled(channel_id)
        finally:
            if self._active_runs.get(channel_id) is task:
                del self._active_runs[channel_id]

    async def _finalize_cancelled(self, channel_id: int) -> None:
        """Render a stopped turn: partial streamed text + a stop marker.

        Keeps channel history paired (the user turn was already appended) by
        recording the partial text as the assistant turn, so the next message
        does not see two user turns in a row.
        """
        self._cancel_pending_stream_edit()
        session = self._session_map.get_or_create(channel_id)
        thinking_id = session.pending_message_id
        partial = self._stream_text.get(channel_id, "")
        note = "⏹ Stopped."
        display = f"{partial}\n\n{note}" if partial.strip() else note
        if thinking_id is not None and self._adapter:
            await self._adapter.edit_message(channel_id, thinking_id, display)
        session.history.append(
            {"role": "assistant",
             "content": [{"type": "text", "text": partial or "(stopped)"}]}
        )
        session.pending_message_id = None

    async def _stop_command(self, channel_id: int) -> str:
        """Handle /stop (and its /cancel alias) — cancel this channel's run."""
        task = self._active_runs.get(channel_id)
        if task is None or task.done():
            return "Nothing is running in this channel."
        task.cancel()
        return "⏹ Stopping the current run."

    async def _model_command(self, text: str) -> str:
        """Handle /model — list models or switch the active one.

        Reads the live model list from the whole-config source and persists a
        switch by writing the model name to the writable pointer file
        (``$SR2_ACTIVE_MODEL_FILE``) — the only writable surface, since the
        config itself is mounted read-only. The next message's config reload
        retargets the LLM. Returns the text to send back.
        """
        cfg = None
        if isinstance(self._config_source, DiscordConfigView):
            cfg = self._config_source.source.current
        if cfg is None:
            return "⚠ Model switching isn't available in this configuration."

        pointer = os.environ.get("SR2_ACTIVE_MODEL_FILE")
        response, selection = render_model_command(
            text, set(cfg.models), cfg.active_model, pointer is not None
        )
        if selection is not None and pointer:
            try:
                Path(pointer).write_text(f"{selection}\n")
            except OSError as exc:
                logger.warning("Failed to write model pointer %s: %s", pointer, exc)
                return f"⚠ Could not persist the switch: {exc}"
        return response

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
        # /hb — probe Harbinger via the CLI, bypassing the LLM entirely.
        if command == "hb":
            output = await probe_harbinger_status()
            if self._adapter is not None:
                for chunk in chunk_message(output, self.config.max_message_length):
                    await self._adapter.send_message(channel_id, chunk)
            return

        # /model — list or switch the active model. Writes the pointer file;
        # the next message's reload retargets the LLM.
        if command == "model":
            response = await self._model_command(rest)
            if self._adapter is not None:
                for chunk in chunk_message(response, self.config.max_message_length):
                    await self._adapter.send_message(channel_id, chunk)
            return

        # /stop (+ /cancel alias) — cancel the current run in this channel.
        if command in ("stop", "cancel"):
            response = await self._stop_command(channel_id)
            if self._adapter is not None:
                await self._adapter.send_message(channel_id, response)
            return

        ctx = self._command_context(channel_id)
        response = handle_command(command, rest, ctx)

        if command == "reset":
            self._session_map.reset(channel_id)

        if response is not None:
            if self._adapter is not None:
                chunks = chunk_message(response, self.config.max_message_length)
                for chunk in chunks:
                    await self._adapter.send_message(channel_id, chunk)

        # /ask with content — process through agent
        if command == "ask" and rest.strip():
            await self._run_agent(rest, channel_id)

    async def _handle_slash(
        self,
        name: str,
        text: str,
        interaction: Any,
    ) -> None:
        """Dispatch a native Discord slash command.

        Mirrors the text-prefix path (``_handle_command``) but responds through
        the interaction instead of a channel message. Slash commands arrive on
        the interactions gateway, so they bypass the on_message mention/channel
        filters — the channel allowlist is re-applied here to match the message
        path.

        Args:
            name: Command name (without leading /), already lowercase.
            text: Command argument text (only ``/ask`` uses it).
            interaction: The discord.py Interaction (treated opaquely; all
                discord.py calls route through the adapter).
        """
        if self._adapter is None or self._agent is None:
            return

        # The adapter reloaded config when the interaction arrived; push the
        # non-Discord half into the agent, mirroring the message path.
        self._apply_agent_config()

        channel_obj = getattr(interaction, "channel", None)
        channel_id = getattr(interaction, "channel_id", None)
        if channel_id is None and channel_obj is not None:
            channel_id = getattr(channel_obj, "id", None)
        if channel_id is None:
            await self._adapter.interaction_send(
                interaction, "⚠ Could not resolve this channel."
            )
            return

        # Re-apply the channel allowlist the message path enforces in the
        # adapter's dispatch_message — slash commands skip that filter.
        channels = self.config.channels
        if channels and channel_id not in channels:
            await self._adapter.interaction_send(
                interaction, "⚠ Commands aren't enabled in this channel."
            )
            return

        if channel_obj is not None:
            self._stamp_area(channel_obj)

        # /hb — probe Harbinger via the CLI, bypassing the LLM. Can take a few
        # seconds, so defer first to keep the interaction alive.
        if name == "hb":
            await self._adapter.interaction_defer(interaction)
            output = await probe_harbinger_status()
            for chunk in chunk_message(output, self.config.max_message_length):
                await self._adapter.interaction_send(interaction, chunk)
            return

        # /model — list or switch the active model.
        if name == "model":
            response = await self._model_command(text)
            for chunk in chunk_message(response, self.config.max_message_length):
                await self._adapter.interaction_send(interaction, chunk)
            return

        # /stop (+ /cancel alias) — cancel the current run in this channel.
        if name in ("stop", "cancel"):
            response = await self._stop_command(channel_id)
            await self._adapter.interaction_send(interaction, response)
            return

        # /ask — run the agent loop. Acknowledge with the prompt as a visible
        # anchor (slash input is otherwise hidden from other members), then
        # stream into the channel exactly like a normal message.
        if name == "ask":
            prompt = text.strip()
            if not prompt:
                await self._adapter.interaction_send(
                    interaction, "⚠ `/ask` needs a message. Try `/ask <your question>`."
                )
                return
            await self._adapter.interaction_send(interaction, f"💭 {prompt}")
            await self._run_agent(prompt, channel_id)
            return

        # Text-producing commands: /reset, /status, /help.
        ctx = self._command_context(channel_id)
        response = handle_command(name, text, ctx)

        if name == "reset":
            self._session_map.reset(channel_id)

        if response is not None:
            for chunk in chunk_message(response, self.config.max_message_length):
                await self._adapter.interaction_send(interaction, chunk)
        else:
            await self._adapter.interaction_send(interaction, "Done.")

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

        # Reset tool lines and streamed text for this channel — each agent
        # turn gets a fresh log.
        self._tool_lines[channel_id] = []
        self._stream_text[channel_id] = ""

        # Restore channel history into the agent
        self._restore_history(session)

        # Show typing indicator while calling the agent (per-message, visible to Discord users).
        # The typing context manager auto-clears when the first message edit/send happens.
        async with self._adapter.channel_typing(channel_id):
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

        # Cancel any pending streaming edit — it would overwrite our
        # finalized message with the "..." version (race condition).
        self._cancel_pending_stream_edit()

        # Detect local image paths in the final response and upload them
        # to Discord before sending the text.
        if self._adapter:
            final_text = await self._upload_images_in_text(final_text, channel_id)

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
        """Render a tool-start event by appending a line to the thinking message.

        Tool activity lines accumulate in the thinking message itself so
        everything stays in one message — no separate tool-log to scroll past.

        Args:
            event: The AgentToolStart event.
            channel_id: Discord channel ID.
        """
        if not self.config.tool_embed_enabled:
            return

        line = f"▶ `{event.name}`"
        self._tool_lines.setdefault(channel_id, []).append(line)
        self._flush_tool_lines(channel_id)


    async def _render_tool_result(
        self,
        event: AgentToolResult,
        channel_id: int,
    ) -> None:
        """Render a tool-result event by appending a line to the thinking message.

        Tool activity lines accumulate in the thinking message itself so
        everything stays in one message — no separate tool-log to scroll past.

        Args:
            event: The AgentToolResult event.
            channel_id: Discord channel ID.
        """
        if not self.config.tool_embed_enabled:
            return

        icon = "✖" if event.is_error else "✓"
        tool_name = event.name or "tool"
        line = f"{icon} `{tool_name}`"
        self._tool_lines.setdefault(channel_id, []).append(line)
        self._flush_tool_lines(channel_id)

    def _flush_tool_lines(self, channel_id: int) -> None:
        """Re-render the thinking message after a tool event.

        Delegates to the unified progress renderer so tool lines and the
        accumulated streamed text are shown TOGETHER in one message — they
        no longer clobber each other.

        Args:
            channel_id: Discord channel ID.
        """
        session = self._session_map.get_or_create(channel_id)
        self._render_progress(channel_id, session.pending_message_id)

    def _render_progress(
        self,
        channel_id: int,
        thinking_id: int | None,
    ) -> None:
        """Render combined progress (tool lines + streamed text) to the thinking message.

        Single source of truth for the in-progress message. Both the tool
        path and the text-delta path funnel through here, so the message
        always shows everything accumulated so far instead of flipping
        between tool-only and text-only content.

        The edit is scheduled on the tracked ``_pending_stream_edit`` future
        so finalization (or a newer progress render) can cancel a stale edit
        before it lands — the same race guard the text path already used.

        Args:
            channel_id: Discord channel ID.
            thinking_id: ID of the thinking placeholder message.
        """
        if thinking_id is None or self._adapter is None:
            return

        lines = self._tool_lines.get(channel_id, [])
        text = self._stream_text.get(channel_id, "")

        parts: list[str] = []
        if lines:
            parts.append("\n".join(lines))
        # Show the streamed text so far with an in-progress marker; fall back
        # to the placeholder when no text has arrived yet.
        parts.append(text + "..." if text else "⏳ Thinking...")

        content = "\n\n".join(parts)
        content = content[: self.config.max_message_length]

        # Supersede any pending edit — this render reflects the latest state.
        self._cancel_pending_stream_edit()
        self._pending_stream_edit = asyncio.ensure_future(
            self._adapter.edit_message(channel_id, thinking_id, content)
        )

    async def _upload_images_in_text(
        self, text: str, channel_id: int
    ) -> str:
        """Detect local image paths in response text and upload to Discord.

        Handles two patterns:
        1. Markdown images: ![alt](/path/to/image.png)
        2. Plain text paths: "sitting at /tmp/spectre_images/spectre.png"

        Uploads each image via send_image() and replaces the reference
        with "[📎 image attached]" so the text remains clean.

        Args:
            text: The final response text.
            channel_id: Discord channel ID.

        Returns:
            Text with image references replaced by attachment markers.
        """
        if self._adapter is None:
            return text

        # Pattern 1: Markdown images ![alt](/path)
        md_pattern = re.compile(r"!\[([^\]]*)\]\((/[^\)]+)\)")
        for m in list(md_pattern.finditer(text)):
            img_path = m.group(2)
            if await self._try_upload(img_path, channel_id):
                text = text.replace(m.group(0), "📎 [image attached]")

        # Pattern 2: Plain text paths (common image extensions)
        plain_pattern = re.compile(
            r"(/(?:\w+/)*\.(?:png|jpg|jpeg|gif|webp|bmp))",
            re.IGNORECASE,
        )
        for m in list(plain_pattern.finditer(text)):
            img_path = m.group(1)
            if await self._try_upload(img_path, channel_id):
                text = text.replace(m.group(0), "📎 [image attached]")

        return text

    async def _try_upload(self, image_path: str, channel_id: int) -> bool:
        """Try to upload a single image to Discord.

        Args:
            image_path: Absolute path to the image file.
            channel_id: Discord channel ID.

        Returns:
            True if the upload succeeded, False otherwise.
        """
        if self._adapter is None:
            return False
        try:
            from pathlib import Path

            path = Path(image_path)
            if not path.exists():
                logger.warning("Image path does not exist: %s", image_path)
                return False
            await self._adapter.send_image(channel_id, path)
            logger.info("Uploaded image to Discord: %s", image_path)
            return True
        except Exception:
            logger.exception("Failed to upload image: %s", image_path)
            return False

    def _cancel_pending_stream_edit(self) -> None:
        """Cancel any pending streaming edit future to prevent it from
        overwriting a finalized message.
        """
        if self._pending_stream_edit is not None and not self._pending_stream_edit.done():
            self._pending_stream_edit.cancel()
            self._pending_stream_edit = None

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
        The pending future is tracked so it can be cancelled before
        finalization or when a newer edit supersedes it.

        Args:
            channel_id: Discord channel ID.
            message_id: ID of the message being edited.
            current_text: Current accumulated response text.
            last_edit_time: Time of the last edit (None for first edit).
            loop: Current asyncio event loop.
        """
        if message_id is None or self._adapter is None:
            return

        # Always record the latest text so a tool flush in between throttled
        # edits still renders the up-to-date streamed text.
        self._stream_text[channel_id] = current_text

        if self.config.edit_stream_interval <= 0:
            return  # Streaming edits disabled

        now = loop.time()
        if last_edit_time is None or (now - last_edit_time) >= self.config.edit_stream_interval:
            # Render combined progress (tool lines + this text) through the
            # single tracked future.
            self._render_progress(channel_id, message_id)

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
