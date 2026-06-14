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
