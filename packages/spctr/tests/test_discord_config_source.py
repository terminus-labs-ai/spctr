"""Tests for DiscordConfigSource — the live config reload path.

The Discord bot runs unattended, so the reload has to be both fresh (an edit
applies on the next message) and forgiving (a broken file does not take the
bot down).
"""
from __future__ import annotations

import logging

import pytest

from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.config_source import DiscordConfigSource


def test_current_starts_at_the_initial_config() -> None:
    initial = DiscordConfig(token="tok", mention_only=True)
    source = DiscordConfigSource(loader=lambda: DiscordConfig(), initial=initial)

    assert source.current is initial


def test_reload_picks_up_changed_values() -> None:
    loaded = [
        DiscordConfig(mention_only=False),
        DiscordConfig(mention_only=True, channels=[42]),
    ]
    source = DiscordConfigSource(loader=lambda: loaded.pop(0), initial=loaded[0])

    assert source.reload().mention_only is False
    assert source.reload().mention_only is True
    assert source.current.channels == [42]


def test_reload_returns_the_config_now_in_force() -> None:
    source = DiscordConfigSource(
        loader=lambda: DiscordConfig(max_message_length=500),
        initial=DiscordConfig(max_message_length=2000),
    )

    returned = source.reload()

    assert returned is source.current
    assert returned.max_message_length == 500


def test_loader_failure_keeps_the_last_good_config() -> None:
    """A malformed or mid-save config file must not kill the bot."""
    def _boom() -> DiscordConfig:
        raise ValueError("bad yaml")

    initial = DiscordConfig(mention_only=True, channels=[7])
    source = DiscordConfigSource(loader=_boom, initial=initial)

    assert source.reload() == initial
    assert source.current == initial


def test_repeated_failures_log_once(caplog: pytest.LogCaptureFixture) -> None:
    """One log line per distinct error, not one per message."""
    def _boom() -> DiscordConfig:
        raise ValueError("bad yaml")

    source = DiscordConfigSource(loader=_boom, initial=DiscordConfig())

    with caplog.at_level(logging.WARNING, logger="sr2_spectre.interfaces.discord.config_source"):
        for _ in range(5):
            source.reload()

    failures = [r for r in caplog.records if "reload failed" in r.message]
    assert len(failures) == 1


def test_reload_recovers_after_a_failure() -> None:
    results: list[object] = [
        ValueError("bad yaml"),
        DiscordConfig(mention_only=True),
    ]

    def _load() -> DiscordConfig:
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[return-value]

    source = DiscordConfigSource(loader=_load, initial=DiscordConfig())

    assert source.reload().mention_only is False
    assert source.reload().mention_only is True


def test_token_is_pinned_to_the_startup_value() -> None:
    """A new token needs a fresh gateway login, so reload must not apply it."""
    source = DiscordConfigSource(
        loader=lambda: DiscordConfig(token="rotated", mention_only=True),
        initial=DiscordConfig(token="original"),
    )

    reloaded = source.reload()

    assert reloaded.token == "original"
    assert reloaded.mention_only is True


def test_an_unset_token_is_adopted_from_the_file() -> None:
    """Nothing to protect until the bot has a token to connect with."""
    source = DiscordConfigSource(
        loader=lambda: DiscordConfig(token="from-file"),
        initial=DiscordConfig(token=""),
    )

    assert source.reload().token == "from-file"


def test_pinned_token_change_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    source = DiscordConfigSource(
        loader=lambda: DiscordConfig(token="rotated"),
        initial=DiscordConfig(token="original"),
    )

    with caplog.at_level(logging.WARNING, logger="sr2_spectre.interfaces.discord.config_source"):
        for _ in range(3):
            source.reload()

    warnings = [r for r in caplog.records if "cannot be applied" in r.message]
    assert len(warnings) == 1


def test_reload_does_not_log_secret_values(caplog: pytest.LogCaptureFixture) -> None:
    source = DiscordConfigSource(
        loader=lambda: DiscordConfig(token="super-secret", channels=[1]),
        initial=DiscordConfig(token="super-secret"),
    )

    with caplog.at_level(logging.INFO, logger="sr2_spectre.interfaces.discord.config_source"):
        source.reload()

    assert "super-secret" not in caplog.text
    assert "channels" in caplog.text


def test_static_source_never_changes() -> None:
    config = DiscordConfig(mention_only=True)
    source = DiscordConfigSource.static(config)

    assert source.reload() is config
    assert source.current is config


def test_static_source_defaults_to_an_empty_config() -> None:
    source = DiscordConfigSource.static(None)

    assert source.current == DiscordConfig()
