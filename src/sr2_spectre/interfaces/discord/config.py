"""Discord interface configuration model.

DiscordConfig is a pydantic model that holds all Discord-specific
configuration. It is intended to be nested under agent.discord in the
SpectreConfig (or loaded as a standalone config block).
"""
from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env(raw: str) -> str:
    """Resolve ${VAR} tokens in a string using os.environ.

    Unresolved tokens are left as-is (non-fatal for token strings).
    """
    def _replace(match: re.Match) -> str:
        name = match.group(1)
        return os.environ.get(name, match.group(0))

    return _VAR_RE.sub(_replace, raw)


class DiscordConfig(BaseModel):
    """Configuration for the Discord interface.

    Attributes:
        token: Discord bot token (required for a real connection; can be
               empty for tests that mock the bot client).
        channels: List of channel IDs to monitor. Empty list means all
                  text channels the bot has access to.
        mention_only: If True, only respond when the bot is mentioned
                      (via @BotName or <@BotID>). If False, respond to
                      every message in configured channels.
        max_message_length: Maximum length of a single Discord message
                            sent by the bot. Discord hard limit is 2000.
        edit_stream_interval: Seconds between message edits when
                              simulating streaming output. Set to 0 to
                              disable streaming edits (send one final
                              message instead).
        tool_embed_enabled: Whether to send tool execution updates as
                            Discord embeds.
        auto_thread: If True, automatically create a channel thread when
                     a new conversation starts in a parent channel and
                     route all replies into that thread instead of the
                     parent channel. Defaults to False.
    """
    token: str = ""
    channels: list[int] = Field(default_factory=list)
    mention_only: bool = False
    max_message_length: int = 2000
    edit_stream_interval: float = 1.0
    tool_embed_enabled: bool = True
    auto_thread: bool = False

    @field_validator("token", mode="before")
    @classmethod
    def _resolve_token_env(cls, v: str) -> str:
        """Resolve ${VAR} tokens in the bot token."""
        if isinstance(v, str):
            return _resolve_env(v)
        return v
