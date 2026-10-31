"""Tests for DiscordConfig — configuration model.

Covers:
1.  Default values
2.  Custom values
3.  Validation
"""
from __future__ import annotations

import pytest

from sr2_spectre.interfaces.discord.config import DiscordConfig


class TestDiscordConfig:
    def test_default_values(self) -> None:
        config = DiscordConfig()
        assert config.token == ""
        assert config.channels == []
        assert config.mention_only is False
        assert config.max_message_length == 2000
        assert config.edit_stream_interval == 1.0
        assert config.tool_embed_enabled is True

    def test_custom_token(self) -> None:
        config = DiscordConfig(token="my-secret-token")
        assert config.token == "my-secret-token"

    def test_custom_channels(self) -> None:
        config = DiscordConfig(channels=[123, 456, 789])
        assert config.channels == [123, 456, 789]

    def test_mention_only_true(self) -> None:
        config = DiscordConfig(mention_only=True)
        assert config.mention_only is True

    def test_custom_max_message_length(self) -> None:
        config = DiscordConfig(max_message_length=1000)
        assert config.max_message_length == 1000

    def test_streaming_disabled(self) -> None:
        config = DiscordConfig(edit_stream_interval=0)
        assert config.edit_stream_interval == 0

    def test_tool_embeds_disabled(self) -> None:
        config = DiscordConfig(tool_embed_enabled=False)
        assert config.tool_embed_enabled is False

    def test_full_config(self) -> None:
        config = DiscordConfig(
            token="bot-token-123",
            channels=[111, 222],
            mention_only=True,
            max_message_length=1500,
            edit_stream_interval=0.5,
            tool_embed_enabled=False,
        )
        assert config.token == "bot-token-123"
        assert config.channels == [111, 222]
        assert config.mention_only is True
        assert config.max_message_length == 1500
        assert config.edit_stream_interval == 0.5
        assert config.tool_embed_enabled is False
