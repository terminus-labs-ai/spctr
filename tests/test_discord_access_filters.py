"""Tests for dispatch_message access control — obsidian-0mlp.

Covers the two gaps this bead closes:

1. Guild and user allowlists (``DiscordConfig.guilds`` / ``DiscordConfig.users``).
   Previously the channel allowlist was the ONLY access control, so a bot
   invited to another server answered there.

2. ``channels`` + ``auto_thread`` silent drop. Replies are routed into a
   thread whose id is a fresh snowflake — never the parent channel id — so a
   follow-up message in that thread failed the ``channel.id in channels``
   check and was dropped with no log line: the bot answered once, then went
   quiet. A message whose channel is a thread must pass when its PARENT is
   in the allowlist.

These tests drive ``dispatch_message`` directly (no bot required), the same
way the live-reload tests in test_discord_adapter.py do.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter
from sr2_spectre.interfaces.discord.config import DiscordConfig


def _message(
    channel_id: int,
    *,
    author_id: int = 42,
    guild_id: int | None = 1000,
    parent_id: int | None = None,
) -> SimpleNamespace:
    """Build a duck-typed message: channel (optionally a thread) + author + guild.

    ``parent_id`` set means the channel is a thread (it has a ``.parent``).
    ``guild_id=None`` means a DM (no guild attribute), mirroring discord.py.
    """
    channel = SimpleNamespace(id=channel_id)
    if parent_id is not None:
        channel.parent = SimpleNamespace(id=parent_id)
    if guild_id is None:
        return SimpleNamespace(
            author=SimpleNamespace(id=author_id),
            channel=channel,
            content="hello",
        )
    return SimpleNamespace(
        author=SimpleNamespace(id=author_id),
        channel=channel,
        content="hello",
        guild=SimpleNamespace(id=guild_id),
    )


def _adapter_with_handler(config: DiscordConfig, seen: list[int]) -> DiscordBotAdapter:
    async def _handler(message) -> None:
        seen.append(message.channel.id)

    adapter = DiscordBotAdapter(config)
    adapter.set_message_handler(_handler)
    return adapter


# ---------------------------------------------------------------------------
# channels + auto_thread: thread membership via parent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_followup_in_auto_created_thread_reaches_handler() -> None:
    """THE regression: auto_thread routes replies into a thread whose id is
    not the parent's id. The follow-up in that thread must not be dropped.
    """
    config = DiscordConfig(channels=[111])
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    # First message in the parent channel: allowed.
    await adapter.dispatch_message(_message(111))
    # Reply routed into a fresh thread (id 999 — minted by Discord).
    await adapter.dispatch_message(_message(999, parent_id=111))

    assert seen == [111, 999]


@pytest.mark.asyncio
async def test_thread_whose_parent_is_not_allowed_is_dropped() -> None:
    config = DiscordConfig(channels=[111])
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    # Thread in channel 222 — 222 is not in the allowlist.
    await adapter.dispatch_message(_message(999, parent_id=222))

    assert seen == []


@pytest.mark.asyncio
async def test_thread_drop_logs_a_reason_line() -> None:
    """Dropped messages must leave a trace — the bug was *silent*.

    dispatch_message logs a debug line naming the filter and the channel id
    for every drop.
    """
    config = DiscordConfig(channels=[111])
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    with _capture_logs("sr2_spectre.interfaces.discord.adapter") as records:
        await adapter.dispatch_message(_message(999, parent_id=222))

    assert seen == []
    joined = " ".join(r.getMessage() for r in records)
    assert "not in channels allowlist" in joined
    assert "999" in joined  # the channel id that got dropped


@pytest.mark.asyncio
async def test_allowed_thread_logs_nothing_dropped() -> None:
    config = DiscordConfig(channels=[111])
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    with _capture_logs("sr2_spectre.interfaces.discord.adapter") as records:
        await adapter.dispatch_message(_message(999, parent_id=111))

    assert seen == [999]
    assert records == []


class _capture_logs:
    """Capture log records from a logger for the duration of the block."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._records: list = []

    def __enter__(self):
        logger = logging.getLogger(self._name)
        self._records = []

        def _handler(record: logging.LogRecord) -> None:
            self._records.append(record)

        handler = logging.Handler()
        handler.emit = _handler
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        self._restore = (logger, handler, old_level)
        return self._records

    def __exit__(self, *exc: object) -> None:
        logger, handler, old_level = self._restore
        logger.removeHandler(handler)
        logger.setLevel(old_level)


# ---------------------------------------------------------------------------
# guilds allowlist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guild_allowlist_drops_other_servers() -> None:
    config = DiscordConfig(guilds=[1000])
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    await adapter.dispatch_message(_message(111, guild_id=1000))  # ours
    await adapter.dispatch_message(_message(111, guild_id=2000))  # invited elsewhere

    assert seen == [111]


@pytest.mark.asyncio
async def test_guild_allowlist_drop_logs_a_reason_line() -> None:
    config = DiscordConfig(guilds=[1000])
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    with _capture_logs("sr2_spectre.interfaces.discord.adapter") as records:
        await adapter.dispatch_message(_message(111, guild_id=2000))

    joined = " ".join(r.getMessage() for r in records)
    assert "not in guilds allowlist" in joined
    assert "2000" in joined


@pytest.mark.asyncio
async def test_guild_allowlist_ignores_dms() -> None:
    """DMs have no guild — a guild allowlist must not be an excuse to
    silently ignore private messages.
    """
    config = DiscordConfig(guilds=[1000])
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    await adapter.dispatch_message(_message(333, guild_id=None))

    assert seen == [333]


@pytest.mark.asyncio
async def test_empty_guild_list_means_all_guilds() -> None:
    config = DiscordConfig()  # guilds=[]
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    await adapter.dispatch_message(_message(111, guild_id=1000))
    await adapter.dispatch_message(_message(111, guild_id=2000))

    assert seen == [111, 111]


# ---------------------------------------------------------------------------
# users allowlist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_allowlist_drops_other_users() -> None:
    config = DiscordConfig(users=[42])
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    await adapter.dispatch_message(_message(111, author_id=42))
    await adapter.dispatch_message(_message(111, author_id=999))

    assert seen == [111]


@pytest.mark.asyncio
async def test_user_allowlist_applies_regardless_of_guild_or_channel() -> None:
    """A user allowlist is user-scoped: being in the right channel of the
    wrong server (or right server, wrong user) still gets dropped when the
    user isn't listed.
    """
    config = DiscordConfig(guilds=[1000], channels=[111], users=[42])
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    # Right server, right channel, wrong user.
    await adapter.dispatch_message(_message(111, author_id=999, guild_id=1000))
    # Right user, wrong server.
    await adapter.dispatch_message(_message(222, author_id=42, guild_id=2000))
    # Right user, right server, right channel.
    await adapter.dispatch_message(_message(111, author_id=42, guild_id=1000))

    assert seen == [111]


@pytest.mark.asyncio
async def test_empty_user_list_means_all_users() -> None:
    config = DiscordConfig()  # users=[]
    seen: list[int] = []
    adapter = _adapter_with_handler(config, seen)

    await adapter.dispatch_message(_message(111, author_id=1))
    await adapter.dispatch_message(_message(111, author_id=2))

    assert seen == [111, 111]


# ---------------------------------------------------------------------------
# config model
# ---------------------------------------------------------------------------


def test_guilds_and_users_default_to_empty() -> None:
    config = DiscordConfig()
    assert config.guilds == []
    assert config.users == []


def test_guilds_and_users_accept_ids() -> None:
    config = DiscordConfig(guilds=[1, 2], users=[7, 8])
    assert config.guilds == [1, 2]
    assert config.users == [7, 8]
