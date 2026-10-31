"""Discord interface — full discord.py bot integration.

An Interface (front door) that connects SR2 Spectre to Discord.
Features:
- Respond to mentions or all messages in configured channels
- Session-per-channel (isolated conversation history per channel)
- Message edit simulation for streaming output
- Tool execution embeds
- Slash commands (/ask, /reset, /status, /help, /hb) — registered as native
  Discord application commands (autocomplete UI, no @mention needed) and also
  accepted as text-prefix commands for backwards compatibility
- Error handling and reconnection support
- Config re-read from disk on every message (edits apply without a restart)
"""
from __future__ import annotations

from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.config_source import DiscordConfigSource
from sr2_spectre.interfaces.discord.interface import DiscordInterface

__all__ = ["DiscordConfig", "DiscordConfigSource", "DiscordInterface"]
