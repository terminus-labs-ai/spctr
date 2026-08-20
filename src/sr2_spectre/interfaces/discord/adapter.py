"""Discord adapter — bridges discord.py objects to handler types.

This layer contains all discord.py imports. The handler and session_map
modules are engine-independent and testable without discord.py installed.

The adapter:
- Wraps the discord.py bot client lifecycle (start/stop/reconnect)
- Converts discord.Message objects to plain Python types
- Sends messages/embeds through the discord.py API
- Handles message edits for streaming simulation
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.config_source import (
    DiscordConfigProvider,
    DiscordConfigSource,
)

logger = logging.getLogger(__name__)

# discord.py is optional — import lazily so tests can run without it
_discord: Any = None


def _import_discord() -> Any:
    """Import discord.py, raising ImportError if unavailable."""
    global _discord
    if _discord is None:
        try:
            import discord as _discord
        except ImportError:
            raise ImportError(
                "discord.py is required for the Discord interface. "
                "Install it with: pip install discord"
            )
    return _discord


class DiscordBotAdapter:
    """Wraps discord.py's Bot client for the Spectre interface.

    Manages bot lifecycle (login, run, close) and provides methods
    for sending messages, editing messages, and sending embeds.

    The adapter is designed so the Interface can control the bot
    lifecycle without the bot controlling the interface.
    """

    def __init__(self, config: DiscordConfig | DiscordConfigProvider) -> None:
        """
        Args:
            config: Either a live config provider — a DiscordConfigSource, or
                a DiscordConfigView onto the whole SpectreConfig, both re-read
                on every message — or a plain DiscordConfig, which is wrapped
                in a static source that never changes.
        """
        self._config_source = (
            config
            if isinstance(config, DiscordConfigProvider)
            else DiscordConfigSource.static(config)
        )
        self._bot: Any = None
        self._running = False
        self._task: asyncio.Task | None = None
        # Set via set_message_handler(); read by the on_message closure built
        # in start(). MUST live here, not in start() — start() runs AFTER the
        # interface wires the handler, so resetting it there drops every
        # message (handler clobbered back to None).
        self._on_message_handler: Any = None
        # Native Discord slash commands (app_commands). The tree is built in
        # start(); the handler is set via set_slash_handler() BEFORE start(),
        # same lifecycle contract as the message handler above.
        self._tree: Any = None
        self._slash_handler: Any = None

    @property
    def config(self) -> DiscordConfig:
        """The Discord config in force, as of the last reload."""
        return self._config_source.current

    @property
    def bot_id(self) -> int | None:
        """Return the bot's Discord user ID, or None if not connected."""
        if self._bot is None:
            return None
        user = getattr(self._bot, "user", None)
        if user is not None:
            return getattr(user, "id", None)
        return None

    @property
    def bot_mentions(self) -> list[str] | None:
        """Return pre-rendered mention strings for the bot."""
        if self._bot is None:
            return None
        user = getattr(self._bot, "user", None)
        if user is not None and hasattr(user, "mention"):
            return [user.mention, f"<@!{user.id}>" if hasattr(user, 'id') else None]
        return None

    async def start(self) -> None:
        """Start the discord.py bot client.

        Creates the Bot instance and begins the connection.
        Blocks until the bot is ready or an error occurs.
        """
        if not self.config.token:
            raise ValueError("Discord bot token is required. Set discord.token in config.")

        discord = _import_discord()

        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content

        self._bot = discord.Client(intents=intents)
        self._running = True

        self._register_slash_commands()

        @self._bot.event
        async def on_ready() -> None:
            user = self._bot.user
            logger.info("Discord bot logged in as %s (ID: %s)", user.name, user.id)
            await self._sync_slash_commands()

        @self._bot.event
        async def on_message(message: Any) -> None:
            await self.dispatch_message(message)

    async def dispatch_message(self, message: Any) -> None:
        """Filter one inbound message and hand it to the message handler.

        This is the process's single entry point for a Discord message, so it
        is where the config is re-read: the filters below, and the handler
        downstream, run against the config this message just loaded.
        """
        config = self._config_source.reload()

        # Skip bot's own messages
        if self._bot is not None and message.author == self._bot.user:
            return

        # Skip DMs if channels are configured (server-only mode)
        if config.channels and not hasattr(message, "channel"):
            return

        # Channel filter
        if config.channels and message.channel.id not in config.channels:
            return

        if self._on_message_handler is not None:
            await self._on_message_handler(message)

    def set_message_handler(self, handler: Any) -> None:
        """Set the message handler callback for incoming messages.

        The handler should be an async function that accepts a
        discord.Message object.
        """
        self._on_message_handler = handler

    def set_slash_handler(self, handler: Any) -> None:
        """Set the async callback for native slash-command interactions.

        Signature: ``async (name: str, text: str, interaction) -> None``.
        Set this BEFORE start(): the command callbacks built in
        _register_slash_commands() read it at invocation time, and the
        on_ready sync makes the commands live.
        """
        self._slash_handler = handler

    # ------------------------------------------------------------------
    # Native Discord slash commands (app_commands)
    # ------------------------------------------------------------------

    def _register_slash_commands(self) -> None:
        """Build the app_commands tree and register the built-in commands.

        Each callback delegates to the interface's slash handler (set via
        set_slash_handler). Descriptions come from the engine-independent
        handler registry so the slash and text-prefix command sets stay in
        sync. A plain discord.Client has no command tree of its own, so one
        is constructed here and synced per-guild in on_ready.
        """
        discord = _import_discord()
        from sr2_spectre.interfaces.discord.handler import get_registered_commands

        tree = discord.app_commands.CommandTree(self._bot)
        self._tree = tree
        registry = get_registered_commands()

        def _desc(name: str, fallback: str) -> str:
            cmd = registry.get(name)
            return cmd.description if cmd is not None else fallback

        async def _dispatch(name: str, text: str, interaction: Any) -> None:
            if self._slash_handler is None:
                logger.warning("Slash command /%s fired with no handler set", name)
                return
            await self._slash_handler(name, text, interaction)

        @tree.command(name="ask", description="Send a message to the agent")
        @discord.app_commands.describe(text="What to ask the agent")
        async def _ask(interaction: Any, text: str) -> None:  # noqa: ANN001
            await _dispatch("ask", text, interaction)

        @tree.command(
            name="reset",
            description=_desc("reset", "Start a new conversation in this channel"),
        )
        async def _reset(interaction: Any) -> None:  # noqa: ANN001
            await _dispatch("reset", "", interaction)

        @tree.command(
            name="status",
            description=_desc("status", "Show current session info"),
        )
        async def _status(interaction: Any) -> None:  # noqa: ANN001
            await _dispatch("status", "", interaction)

        @tree.command(
            name="help",
            description=_desc("help", "Show available commands"),
        )
        async def _help(interaction: Any) -> None:  # noqa: ANN001
            await _dispatch("help", "", interaction)

        @tree.command(
            name="hb",
            description="Probe Harbinger: live slots, run outcomes, done & blocked beads",
        )
        async def _hb(interaction: Any) -> None:  # noqa: ANN001
            await _dispatch("hb", "", interaction)

        @tree.command(
            name="model",
            description=_desc(
                "model", "List models, or switch with /model <name>"
            ),
        )
        @discord.app_commands.describe(name="Model to switch to (omit to list)")
        async def _model(interaction: Any, name: str = "") -> None:  # noqa: ANN001
            await _dispatch("model", name, interaction)

        @tree.command(
            name="stop",
            description=_desc(
                "stop", "Stop the agent's current run in this channel"
            ),
        )
        async def _stop(interaction: Any) -> None:  # noqa: ANN001
            await _dispatch("stop", "", interaction)

        @tree.command(
            name="cancel",
            description="Alias for /stop — cancel the current run",
        )
        async def _cancel(interaction: Any) -> None:  # noqa: ANN001
            await _dispatch("cancel", "", interaction)

    async def _sync_slash_commands(self) -> None:
        """Sync slash commands to every connected guild for instant availability.

        Per-guild sync propagates immediately; a global sync can take up to an
        hour to appear in clients. Guilds joined after startup pick the commands
        up on the next restart. Never raises — a failed sync is logged and the
        bot keeps running (text-prefix commands still work).

        The GLOBAL command scope is then cleared: our commands are registered
        per-guild above, so any global commands present belong to a previous
        registration under this same application (e.g. a prior bot identity).
        Pushing an empty global set deletes those leftovers so only the intended
        commands remain. Global deletions can take up to an hour to disappear
        from clients; guild commands update instantly.
        """
        if self._tree is None:
            return
        try:
            total = 0
            guilds = list(getattr(self._bot, "guilds", []))
            for guild in guilds:
                self._tree.copy_global_to(guild=guild)
                synced = await self._tree.sync(guild=guild)
                total += len(synced)
            # Clear stale global commands AFTER the guild loop, so copy_global_to
            # above still sees our command set. clear_commands empties the tree's
            # global scope; the empty sync() deletes them on Discord's side.
            self._tree.clear_commands(guild=None)
            cleared = await self._tree.sync()
            logger.info(
                "Synced %d slash command(s) across %d guild(s); global scope now "
                "holds %d command(s)",
                total, len(guilds), len(cleared),
            )
        except Exception as exc:
            logger.error("Slash command sync failed: %s", exc)

    async def interaction_defer(self, interaction: Any) -> None:
        """Acknowledge a slash interaction so Discord does not time it out.

        Discord requires an initial response within ~3 seconds; deferring buys
        up to 15 minutes for a followup. No-op if already responded. Never
        raises.
        """
        try:
            if not interaction.response.is_done():
                await interaction.response.defer()
        except Exception as exc:
            logger.error("Failed to defer interaction: %s", exc)

    async def interaction_send(self, interaction: Any, content: str) -> None:
        """Send content back to a slash interaction.

        Uses the initial response the first time and followups thereafter, so a
        handler may call this repeatedly (e.g. for chunked output). Never
        raises.
        """
        try:
            if interaction.response.is_done():
                await interaction.followup.send(content)
            else:
                await interaction.response.send_message(content)
        except Exception as exc:
            logger.error("Failed to respond to interaction: %s", exc)

    async def run(self) -> None:
        """Run the bot until stopped."""
        if self._bot is None:
            raise RuntimeError("Call start() before run()")
        await self._bot.start(self.config.token)

    async def stop(self) -> None:
        """Stop the bot client gracefully."""
        self._running = False
        if self._bot is not None:
            await self._bot.close()

    async def send_message(
        self,
        channel_id: int,
        content: str,
    ) -> Any:
        """Send a message to a channel by ID.

        Returns the discord.Message object, or None if the channel
        couldn't be resolved.
        """
        if self._bot is None:
            logger.error("Bot not initialized — cannot send message")
            return None

        channel = self._bot.get_channel(channel_id)
        if channel is None:
            # Try fetching the channel
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except Exception as exc:
                logger.error("Could not fetch channel %d: %s", channel_id, exc)
                return None

        try:
            message = await channel.send(content)
            return message
        except Exception as exc:
            logger.error("Failed to send message to channel %d: %s", channel_id, exc)
            return None

    async def edit_message(
        self,
        channel_id: int,
        message_id: int,
        content: str,
    ) -> Any:
        """Edit an existing message.

        Used for streaming simulation — progressively updating a message
        as the agent generates text.
        """
        if self._bot is None:
            return None

        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except Exception as exc:
                logger.error("Could not fetch channel %d: %s", channel_id, exc)
                return None

        try:
            message = await channel.fetch_message(message_id)
            await message.edit(content=content)
            return message
        except Exception as exc:
            logger.error("Failed to edit message %d in channel %d: %s", message_id, channel_id, exc)
            return None

    async def send_embed(
        self,
        channel_id: int,
        embed_dict: dict,
    ) -> Any:
        """Send a message with an embed.

        Args:
            channel_id: Discord channel ID.
            embed_dict: Embed dict compatible with discord.Embed.from_dict().

        Returns:
            The discord.Message object, or None on failure.
        """
        if self._bot is None:
            return None

        discord = _import_discord()
        embed = discord.Embed.from_dict(embed_dict)

        return await self.send_embed_raw(channel_id, embed)

    async def send_embed_raw(
        self,
        channel_id: int,
        embed: Any,
    ) -> Any:
        """Send a message with a pre-built discord.Embed object."""
        if self._bot is None:
            return None

        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except Exception as exc:
                logger.error("Could not fetch channel %d: %s", channel_id, exc)
                return None

        try:
            message = await channel.send(embed=embed)
            return message
        except Exception as exc:
            logger.error("Failed to send embed to channel %d: %s", channel_id, exc)
            return None

    async def create_thread(
        self,
        channel_id: int,
        name: str,
        message_id: int,
    ) -> int | None:
        """Create a public thread from an existing message.

        Creates a public thread anchored on the given message in the
        parent channel. Returns the thread's channel ID, or None on failure.

        Args:
            channel_id: Parent channel ID.
            name: Thread name (Discord limit: 100 chars).
            message_id: ID of the message to anchor the thread on.

        Returns:
            Thread channel ID, or None if creation failed.
        """
        if self._bot is None:
            return None

        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except Exception as exc:
                logger.error("Could not fetch channel %d: %s", channel_id, exc)
                return None

        try:
            # Trim to Discord's 100-char thread name limit
            thread_name = name[:100]

            # create_thread expects a Snowflake (object with .id attribute),
            # not a bare int. Fetch the actual message from the channel.
            message = await channel.fetch_message(message_id)

            thread = await channel.create_thread(
                name=thread_name,
                message=message,
                auto_archive_duration=1440,  # 24 hours
            )
            thread_id = getattr(thread, "id", None)
            if thread_id is not None:
                logger.info(
                    "Created thread %s (ID: %d) in channel %d",
                    thread_name, thread_id, channel_id,
                )
            return thread_id
        except Exception as exc:
            logger.error(
                "Failed to create thread in channel %d: %s", channel_id, exc
            )
            return None

    async def send_image(
        self,
        channel_id: int,
        image_path: str,
        caption: str = "",
    ) -> Any:
        """Send an image file to a Discord channel.

        Args:
            channel_id: Discord channel ID.
            image_path: Absolute path to the image file.
            caption: Optional text to send with the image.

        Returns:
            The discord.Message object, or None on failure.
        """
        if self._bot is None:
            logger.error("Bot not initialized — cannot send image")
            return None

        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except Exception as exc:
                logger.error("Could not fetch channel %d: %s", channel_id, exc)
                return None

        try:
            discord = _import_discord()
            file = discord.File(image_path)
            message = await channel.send(content=caption, file=file)
            return message
        except Exception as exc:
            logger.error(
                "Failed to send image to channel %d: %s", channel_id, exc
            )
            return None

    @contextlib.asynccontextmanager
    async def channel_typing(
        self,
        channel_id: int,
    ) -> AsyncIterator[None]:
        """Hold the typing indicator in a channel for the duration of a block.

        Discord's typing indicator shows "Bot is typing..." and lasts only
        ~10 seconds per trigger, so it must be held open by an async context
        manager that refreshes it. Use as::

            async with adapter.channel_typing(channel_id):
                await do_agent_work()

        If the bot is not connected or the channel cannot be resolved, the
        block still runs — just without a typing indicator.

        Args:
            channel_id: Discord channel ID.
        """
        if self._bot is None:
            yield
            return

        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except Exception as exc:
                logger.error("Could not fetch channel %d for typing: %s", channel_id, exc)
                yield
                return

        async with channel.typing():
            yield

    def is_thread_channel(self, channel: Any) -> bool:
        """Check if a discord.py channel object is a Thread.

        Args:
            channel: A discord.py channel object.

        Returns:
            True if the channel is a Discord thread.
        """
        discord = _import_discord()
        return isinstance(channel, discord.Thread)

    def area_channel(self, channel: Any) -> tuple[int | None, str | None]:
        """Return (channel_id, channel_name) of the area-bearing channel.

        For a Thread, this is its parent channel. For a regular text channel,
        the channel itself. Returns (None, None) for a DM, an orphaned thread,
        or any channel whose name cannot be read.
        """
        discord = _import_discord()
        if isinstance(channel, discord.Thread):
            channel = channel.parent
            if channel is None:
                return None, None

        name = getattr(channel, "name", None)
        if name is None:
            return None, None

        return getattr(channel, "id", None), name
