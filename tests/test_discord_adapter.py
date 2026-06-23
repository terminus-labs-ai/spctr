"""Tests for DiscordBotAdapter — the only module that touches discord.py.

Unlike the other discord tests (which mock the bot and run without discord.py
installed), these construct the REAL discord client offline. This is the
coverage that was missing: the adapter previously called ``discord.Bot(...)``
(a py-cord API absent from discord.py), which every mock-based test sailed past
because the bot was never actually built. ``adapter.start()`` constructs the
client and registers event handlers but does NOT open a network connection
(that happens in ``run()``), so it is fully offline-testable.

Guarded with ``importorskip`` so the suite still passes where discord.py is not
installed.
"""
from __future__ import annotations

import pytest

discord = pytest.importorskip("discord")

from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter
from sr2_spectre.interfaces.discord.config import DiscordConfig


def _adapter(**overrides) -> DiscordBotAdapter:
    cfg = DiscordConfig(token="fake-token-for-offline-construction", **overrides)
    return DiscordBotAdapter(cfg)


async def test_start_builds_a_real_discord_client_offline() -> None:
    """start() must construct an actual discord.py client without connecting.

    This is the regression guard for the discord.Bot -> discord.Client fix:
    with the old py-cord call this raised AttributeError at construction.
    """
    adapter = _adapter()
    try:
        await adapter.start()
        assert isinstance(adapter._bot, discord.Client)
    finally:
        await adapter.stop()


async def test_start_enables_message_content_intent() -> None:
    """The bot must request the message_content intent (needed to read text)."""
    adapter = _adapter()
    try:
        await adapter.start()
        assert adapter._bot.intents.message_content is True
    finally:
        await adapter.stop()


async def test_start_without_token_raises() -> None:
    """An empty token is a configuration error, surfaced before any connect."""
    adapter = DiscordBotAdapter(DiscordConfig(token=""))
    with pytest.raises(ValueError, match="token is required"):
        await adapter.start()


async def test_bot_id_is_none_before_connection() -> None:
    """bot_id resolves from the connected user; offline it is None."""
    adapter = _adapter()
    try:
        await adapter.start()
        assert adapter.bot_id is None
    finally:
        await adapter.stop()


async def test_start_preserves_a_handler_set_beforehand() -> None:
    """start() must NOT clobber a handler wired before it.

    The interface calls set_message_handler() and THEN adapter.start(); a
    stray ``self._on_message_handler = None`` inside start() silently dropped
    every inbound message. Regression guard: the handler survives start().
    """
    adapter = _adapter()

    async def handler(_message) -> None:  # pragma: no cover - identity check only
        pass

    adapter.set_message_handler(handler)
    try:
        await adapter.start()
        assert adapter._on_message_handler is handler
    finally:
        await adapter.stop()


class _RecordingTyping:
    """Stand-in for discord.py's channel.typing() async context manager."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "_RecordingTyping":
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.exited = True


class _FakeChannel:
    def __init__(self) -> None:
        self._typing = _RecordingTyping()

    def typing(self) -> _RecordingTyping:
        return self._typing


async def test_channel_typing_is_usable_as_async_context_manager() -> None:
    """channel_typing must be entered with ``async with`` and hold typing for
    the whole block.

    Regression: channel_typing was ``async def ... return channel.typing()``,
    so calling it produced a *coroutine* — ``async with`` on it raised because
    a coroutine has no ``__aenter__``. The interface wrapped the agent turn in
    ``async with self._adapter.channel_typing(...)`` and the typing indicator
    never appeared on Discord. The contract: calling channel_typing(id) yields
    an async context manager that enters and exits channel.typing().
    """
    adapter = _adapter()
    fake_channel = _FakeChannel()

    class _FakeBot:
        def get_channel(self, _cid: int) -> _FakeChannel:
            return fake_channel

    adapter._bot = _FakeBot()  # type: ignore[assignment]

    async with adapter.channel_typing(123):
        # Inside the block, typing must be active.
        assert fake_channel._typing.entered is True
        assert fake_channel._typing.exited is False

    # After the block, typing must have been released.
    assert fake_channel._typing.exited is True
