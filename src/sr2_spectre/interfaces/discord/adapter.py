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
import logging
from typing import TYPE_CHECKING, Any

from sr2_spectre.interfaces.discord.config import DiscordConfig

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

    def __init__(self, config: DiscordConfig) -> None:
        self.config = config
        self._bot: Any = None
        self._running = False
        self._task: asyncio.Task | None = None
        # Set via set_message_handler(); read by the on_message closure built
        # in start(). MUST live here, not in start() — start() runs AFTER the
        # interface wires the handler, so resetting it there drops every
        # message (handler clobbered back to None).
        self._on_message_handler: Any = None

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

        @self._bot.event
        async def on_ready() -> None:
            user = self._bot.user
            logger.info("Discord bot logged in as %s (ID: %s)", user.name, user.id)

        @self._bot.event
        async def on_message(message: Any) -> None:
            # Skip bot's own messages
            if message.author == self._bot.user:
                return

            # Skip DMs if channels are configured (server-only mode)
            if self.config.channels and not hasattr(message, "channel"):
                return

            # Channel filter
            if self.config.channels and message.channel.id not in self.config.channels:
                return

            if self._on_message_handler is not None:
                await self._on_message_handler(message)

    def set_message_handler(self, handler: Any) -> None:
        """Set the message handler callback for incoming messages.

        The handler should be an async function that accepts a
        discord.Message object.
        """
        self._on_message_handler = handler

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

    def is_thread_channel(self, channel: Any) -> bool:
        """Check if a discord.py channel object is a Thread.

        Args:
            channel: A discord.py channel object.

        Returns:
            True if the channel is a Discord thread.
        """
        discord = _import_discord()
        return isinstance(channel, discord.Thread)
