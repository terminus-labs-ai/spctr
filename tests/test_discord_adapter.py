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

from types import SimpleNamespace

import pytest

discord = pytest.importorskip("discord")

from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.config_source import DiscordConfigSource


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


# --- Live config reload -------------------------------------------------
#
# dispatch_message() is the process's single entry point for an inbound
# Discord message, so it owns the per-message config reload. These tests
# drive it directly (no bot required) with a source whose loader is under
# test control.

def _message(channel_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        author=SimpleNamespace(id=999),
        channel=SimpleNamespace(id=channel_id),
        content="hello",
    )


async def test_dispatch_reloads_the_config_for_every_message() -> None:
    calls: list[int] = []

    def _load() -> DiscordConfig:
        calls.append(1)
        return DiscordConfig(token="t")

    adapter = DiscordBotAdapter(DiscordConfigSource(_load, DiscordConfig(token="t")))
    adapter.set_message_handler(lambda message: _noop())

    for _ in range(3):
        await adapter.dispatch_message(_message(1))

    assert len(calls) == 3


async def _noop() -> None:
    return None


async def test_channel_filter_uses_the_reloaded_channels() -> None:
    """Adding a channel to the config file must take effect without a restart."""
    configs = [
        DiscordConfig(token="t", channels=[111]),
        DiscordConfig(token="t", channels=[111, 222]),
    ]
    seen: list[int] = []

    async def _handler(message) -> None:
        seen.append(message.channel.id)

    adapter = DiscordBotAdapter(DiscordConfigSource(lambda: configs.pop(0), configs[0]))
    adapter.set_message_handler(_handler)

    await adapter.dispatch_message(_message(222))   # loads config #1: filtered out
    await adapter.dispatch_message(_message(222))   # loads config #2: allowed

    assert seen == [222]


async def test_config_property_reads_through_to_the_source() -> None:
    source = DiscordConfigSource(
        lambda: DiscordConfig(token="t", mention_only=True),
        DiscordConfig(token="t"),
    )
    adapter = DiscordBotAdapter(source)

    assert adapter.config.mention_only is False
    source.reload()
    assert adapter.config.mention_only is True


async def test_a_plain_config_still_works() -> None:
    """Callers that pass a DiscordConfig get a source that never changes."""
    adapter = DiscordBotAdapter(DiscordConfig(token="t", mention_only=True))

    assert adapter.config.mention_only is True
    assert adapter._config_source.reload().mention_only is True


# --- Native slash commands (app_commands) -------------------------------
#
# start() builds a real discord.app_commands.CommandTree offline. These
# assert the tree is populated with the expected built-in commands; the
# on_ready sync itself needs a live gateway and is not exercised here.

async def test_start_registers_native_slash_commands() -> None:
    """start() builds a command tree with the built-in slash commands."""
    adapter = _adapter()
    try:
        await adapter.start()
        assert adapter._tree is not None
        names = {cmd.name for cmd in adapter._tree.get_commands()}
        assert names == {"ask", "reset", "status", "help", "hb", "model", "stop", "cancel"}
    finally:
        await adapter.stop()


async def test_ask_slash_command_declares_a_text_argument() -> None:
    """/ask must expose a `text` parameter so Discord prompts for input."""
    adapter = _adapter()
    try:
        await adapter.start()
        ask = next(c for c in adapter._tree.get_commands() if c.name == "ask")
        assert "text" in {p.name for p in ask.parameters}
    finally:
        await adapter.stop()


async def test_model_slash_command_declares_a_name_argument() -> None:
    """/model must expose an optional `name` parameter to pick a model."""
    adapter = _adapter()
    try:
        await adapter.start()
        model = next(c for c in adapter._tree.get_commands() if c.name == "model")
        assert "name" in {p.name for p in model.parameters}
    finally:
        await adapter.stop()


async def test_slash_callback_delegates_to_the_slash_handler() -> None:
    """Invoking a tree command must call the handler with (name, text, interaction)."""
    adapter = _adapter()
    calls: list[tuple[str, str, object]] = []

    async def _handler(name: str, text: str, interaction: object) -> None:
        calls.append((name, text, interaction))

    adapter.set_slash_handler(_handler)
    try:
        await adapter.start()
        reset = next(c for c in adapter._tree.get_commands() if c.name == "reset")
        sentinel = object()
        await reset.callback(sentinel)
        assert calls == [("reset", "", sentinel)]
    finally:
        await adapter.stop()


async def test_slash_command_with_no_handler_is_a_noop() -> None:
    """A command firing before the handler is wired must not raise."""
    adapter = _adapter()
    try:
        await adapter.start()
        ask = next(c for c in adapter._tree.get_commands() if c.name == "ask")
        await ask.callback(object(), "hello")  # no handler set — must not raise
    finally:
        await adapter.stop()


class _FakeResponse:
    def __init__(self, done: bool) -> None:
        self._done = done
        self.sent: list[str] = []

    def is_done(self) -> bool:
        return self._done

    async def send_message(self, content: str) -> None:
        self.sent.append(content)

    async def defer(self) -> None:
        self._done = True


class _FakeFollowup:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, content: str) -> None:
        self.sent.append(content)


class _FakeInteraction:
    def __init__(self, done: bool = False) -> None:
        self.response = _FakeResponse(done)
        self.followup = _FakeFollowup()


async def test_interaction_send_uses_initial_response_then_followups() -> None:
    """First interaction_send is the initial response; later ones are followups."""
    adapter = _adapter()
    interaction = _FakeInteraction(done=False)

    await adapter.interaction_send(interaction, "first")
    assert interaction.response.sent == ["first"]

    interaction.response._done = True
    await adapter.interaction_send(interaction, "second")
    assert interaction.followup.sent == ["second"]


async def test_interaction_defer_marks_the_response_done() -> None:
    """interaction_defer acknowledges an un-answered interaction exactly once."""
    adapter = _adapter()
    interaction = _FakeInteraction(done=False)

    await adapter.interaction_defer(interaction)
    assert interaction.response.is_done() is True

    # Deferring again is a no-op (already done) and must not raise.
    await adapter.interaction_defer(interaction)


async def test_sync_clears_global_scope_after_syncing_guilds() -> None:
    """_sync_slash_commands syncs per-guild, THEN clears the global scope.

    Ordering matters: copy_global_to must run while the tree still holds our
    commands, and the global scope is cleared afterwards to delete leftover
    commands from a prior registration under the same application (the hermes
    global commands bug). The final global sync must push the emptied set.
    """
    adapter = _adapter()
    calls: list[tuple[str, object]] = []
    guild = SimpleNamespace(id=42)

    class _FakeTree:
        def copy_global_to(self, guild: object) -> None:
            calls.append(("copy", guild))

        async def sync(self, guild: object = None) -> list:
            calls.append(("sync", guild))
            return []

        def clear_commands(self, guild: object = None) -> None:
            calls.append(("clear", guild))

    adapter._tree = _FakeTree()
    adapter._bot = SimpleNamespace(guilds=[guild])

    await adapter._sync_slash_commands()

    # Guild sync happens before the global clear...
    assert calls.index(("copy", guild)) < calls.index(("clear", None))
    assert calls.index(("sync", guild)) < calls.index(("clear", None))
    # ...and the global scope is cleared, then synced empty last.
    assert ("clear", None) in calls
    assert calls[-1] == ("sync", None)
