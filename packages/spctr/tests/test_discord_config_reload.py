"""End-to-end: editing the config file changes bot behaviour without a restart.

The Discord bot is long-lived and often runs where the operator cannot reach
it, so a config edit must take effect on the next message. These tests drive
the real loader built by the CLI against a real file on disk.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr2_spectre.cli import build_discord_config_source
from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter

_BASE = """\
agent:
  name: edi
models:
  default:
    model: test
    base_url: http://localhost:11434/v1
pipeline:
  layers:
    - name: system
      target: system
      resolvers:
        - type: static
          config:
            text: You are helpful.
"""


def _write_config(path: Path, discord_block: str) -> None:
    path.write_text(_BASE + textwrap.dedent(discord_block))


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config file on disk, with the other resolution tiers pointed at tmp."""
    home = tmp_path / "sr2home"
    home.mkdir()
    monkeypatch.setenv("SR2_HOME", str(home))

    path = tmp_path / "edi.yaml"
    _write_config(path, """\
        discord:
          token: bot-token
          channels: [111]
          mention_only: false
    """)
    return path


def _source(config_path: Path):
    source = build_discord_config_source(config_path, config_path.parent)
    return source


def test_edited_values_apply_on_the_next_reload(config_path: Path) -> None:
    source = _source(config_path)
    assert source.reload().mention_only is False

    _write_config(config_path, """\
        discord:
          token: bot-token
          channels: [111, 222]
          mention_only: true
          max_message_length: 900
    """)

    reloaded = source.reload()
    assert reloaded.mention_only is True
    assert reloaded.channels == [111, 222]
    assert reloaded.max_message_length == 900


def test_the_startup_token_survives_a_reload(config_path: Path) -> None:
    source = build_discord_config_source(
        config_path, config_path.parent, initial=None
    )
    source.reload()  # picks up bot-token from the file

    _write_config(config_path, """\
        discord:
          token: rotated-token
    """)

    assert source.reload().token == "bot-token"


def test_a_broken_config_file_keeps_the_last_good_config(config_path: Path) -> None:
    source = _source(config_path)
    good = source.reload()

    config_path.write_text("agent: [this is not: valid yaml\n")

    assert source.reload() == good


def test_a_removed_config_file_keeps_the_last_good_config(config_path: Path) -> None:
    source = _source(config_path)
    good = source.reload()

    config_path.unlink()

    assert source.reload() == good


def test_a_dropped_discord_block_falls_back_to_defaults(config_path: Path) -> None:
    """Removing the block is a real edit, not an error — defaults take over."""
    source = _source(config_path)
    source.reload()

    _write_config(config_path, "")

    assert source.reload().channels == []


async def test_a_newly_allowed_channel_is_answered_without_a_restart(
    config_path: Path,
) -> None:
    """The behaviour the reload exists for, exercised through the adapter."""
    source = _source(config_path)
    adapter = DiscordBotAdapter(source)

    answered: list[int] = []

    async def _handler(message) -> None:
        answered.append(message.channel.id)

    adapter.set_message_handler(_handler)

    message = SimpleNamespace(
        author=SimpleNamespace(id=7),
        channel=SimpleNamespace(id=222),
        content="hello",
    )

    await adapter.dispatch_message(message)
    assert answered == []  # 222 is not in the configured channels yet

    _write_config(config_path, """\
        discord:
          token: bot-token
          channels: [111, 222]
    """)

    await adapter.dispatch_message(message)
    assert answered == [222]
