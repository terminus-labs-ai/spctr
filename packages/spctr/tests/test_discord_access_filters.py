"""Behavioral coverage for the adapter-owned Discord access boundary."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter
from sr2_spectre.interfaces.discord.config import DiscordConfig


def _message(
    *,
    user_id: int | None = 1,
    guild_id: int | None = 2,
    channel_id: int | None = 3,
    parent_id: int | None = None,
) -> SimpleNamespace:
    event = SimpleNamespace()
    if user_id is not None:
        event.author = SimpleNamespace(id=user_id)
    if guild_id is not None:
        event.guild = SimpleNamespace(id=guild_id)
    if channel_id is not None:
        parent = SimpleNamespace(id=parent_id) if parent_id is not None else None
        event.channel = SimpleNamespace(id=channel_id, parent=parent)
    return event


def _interaction(
    *,
    user_id: int | None = 1,
    guild_id: int | None = 2,
    channel_id: int | None = 3,
    parent_id: int | None = None,
) -> SimpleNamespace:
    event = SimpleNamespace()
    if user_id is not None:
        event.user = SimpleNamespace(id=user_id)
    event.guild_id = guild_id
    event.channel_id = channel_id
    if parent_id is not None:
        event.channel = SimpleNamespace(
            id=channel_id,
            parent=SimpleNamespace(id=parent_id),
        )
    return event


@pytest.mark.parametrize("event", [_message(), _interaction()])
def test_empty_access_filters_allow_every_event(event: SimpleNamespace) -> None:
    assert DiscordBotAdapter(DiscordConfig()).access_allowed(event)


@pytest.mark.parametrize(
    ("event", "allowed"),
    [
        (_message(user_id=7), True),
        (_message(user_id=8), False),
        (_message(user_id=None), False),
        (_interaction(user_id=7), True),
        (_interaction(user_id=8), False),
        (_interaction(user_id=None), False),
    ],
)
def test_user_filter_supports_message_and_interaction_shapes(
    event: SimpleNamespace, allowed: bool
) -> None:
    assert DiscordBotAdapter(DiscordConfig(users=[7])).access_allowed(event) is allowed


@pytest.mark.parametrize(
    ("event", "allowed"),
    [
        (_message(guild_id=2), True),
        (_message(guild_id=9), False),
        (_message(guild_id=None), True),
        (_interaction(guild_id=2), True),
        (_interaction(guild_id=9), False),
        (_interaction(guild_id=None), True),
    ],
)
def test_guild_filter_allows_direct_messages_but_rejects_other_guilds(
    event: SimpleNamespace, allowed: bool
) -> None:
    assert DiscordBotAdapter(DiscordConfig(guilds=[2])).access_allowed(event) is allowed


@pytest.mark.parametrize(
    ("event", "allowed"),
    [
        (_message(channel_id=3), True),
        (_message(channel_id=4), False),
        (_message(channel_id=3, parent_id=9), True),
        (_message(channel_id=4, parent_id=3), True),
        (_message(channel_id=4, parent_id=9), False),
    ],
)
def test_channel_filter_accepts_direct_or_parent_channel(
    event: SimpleNamespace, allowed: bool
) -> None:
    assert DiscordBotAdapter(DiscordConfig(channels=[3])).access_allowed(event) is allowed


def test_direct_messages_still_obey_user_and_channel_filters() -> None:
    adapter = DiscordBotAdapter(DiscordConfig(users=[1], channels=[3]))
    assert adapter.access_allowed(_message(user_id=1, guild_id=None, channel_id=3))
    assert not adapter.access_allowed(_message(user_id=9, guild_id=None, channel_id=3))
    assert not adapter.access_allowed(_message(user_id=1, guild_id=None, channel_id=None))


def test_configured_filters_are_conjunctive() -> None:
    adapter = DiscordBotAdapter(DiscordConfig(users=[1], guilds=[2], channels=[3]))
    assert adapter.access_allowed(_interaction(user_id=1, guild_id=2, channel_id=3))
    assert not adapter.access_allowed(_interaction(user_id=9, guild_id=2, channel_id=3))
    assert not adapter.access_allowed(_interaction(user_id=1, guild_id=9, channel_id=3))
    assert not adapter.access_allowed(_interaction(user_id=1, guild_id=2, channel_id=9))


def test_interaction_id_fallbacks_and_parent_thread_are_supported() -> None:
    adapter = DiscordBotAdapter(DiscordConfig(users=[1], guilds=[2], channels=[3]))
    assert adapter.access_allowed(_interaction(user_id=1, guild_id=2, channel_id=3))
    assert adapter.access_allowed(_interaction(user_id=1, guild_id=2, channel_id=9, parent_id=3))
    assert not adapter.access_allowed(_interaction(user_id=1, guild_id=2, channel_id=9, parent_id=8))


def test_denial_logs_filter_and_observed_ids_once(caplog: pytest.LogCaptureFixture) -> None:
    adapter = DiscordBotAdapter(DiscordConfig(users=[7]))
    event = _interaction(user_id=8, guild_id=2, channel_id=3, parent_id=4)

    with caplog.at_level(logging.INFO):
        assert not adapter.access_allowed(event)

    denied = [record.getMessage() for record in caplog.records if "access denied" in record.getMessage()]
    assert denied == [
        "Discord access denied: filter=users user=8 guild=2 channel=3 parent=4"
    ]
    caplog.clear()
    assert DiscordBotAdapter(DiscordConfig(users=[8])).access_allowed(event)
    assert not [record for record in caplog.records if "access denied" in record.getMessage()]


async def test_dispatch_only_calls_handler_for_allowed_messages() -> None:
    adapter = DiscordBotAdapter(DiscordConfig(users=[1], guilds=[2], channels=[3]))
    handler = AsyncMock()
    adapter.set_message_handler(handler)

    await adapter.dispatch_message(_message(user_id=9, guild_id=2, channel_id=3))
    await adapter.dispatch_message(_message(user_id=1, guild_id=2, channel_id=3))

    handler.assert_awaited_once()
