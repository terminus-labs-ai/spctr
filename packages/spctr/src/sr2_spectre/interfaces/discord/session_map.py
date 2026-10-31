"""Session-per-channel management for the Discord interface.

DiscordInterface needs to maintain separate conversation histories per
channel (and optionally per thread). This module provides a SessionMap
that maps Discord channel IDs to Agent session IDs and tracks active
sessions.

The design mirrors the Agent's new_session() pattern: each channel gets
its own session_id, which the Agent uses to isolate history. Since the
Agent currently owns a single history list, the Discord interface
maintains a mapping from channel_id -> history snapshot, and restores
the correct history before calling stream_message().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChannelSession:
    """Tracks the conversation state for a single Discord channel/thread.

    Attributes:
        channel_id: Discord channel ID.
        session_id: Spectre session ID (derived from channel_id).
        history: List of serialized Message dicts representing the
                 conversation history for this channel.
        pending_message: Discord message ID being edited for streaming
                         (None when not in an active turn).
    """
    channel_id: int
    session_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    pending_message_id: int | None = None

    @staticmethod
    def session_id_for(channel_id: int) -> str:
        """Generate a deterministic session ID from a Discord channel ID."""
        return f"discord-{channel_id}"


class SessionMap:
    """Maps Discord channel IDs to ChannelSession objects.

    Provides O(1) lookup by channel_id. Thread-safe for single-threaded
    async use (discord.py runs in a single thread per bot instance).

    When auto_thread is enabled, each parent-channel mention spawns its own
    fresh thread (one conversation per thread). The interface does NOT keep a
    parent->thread link — reusing a single thread per parent funnels distinct
    topics into the first thread (the wrong-topic bug, obsidian-a3q).
    """

    def __init__(self) -> None:
        self._sessions: dict[int, ChannelSession] = {}

    def get_or_create(self, channel_id: int) -> ChannelSession:
        """Get the session for a channel, creating one if it doesn't exist."""
        if channel_id not in self._sessions:
            self._sessions[channel_id] = ChannelSession(
                channel_id=channel_id,
                session_id=ChannelSession.session_id_for(channel_id),
            )
        return self._sessions[channel_id]

    def reset(self, channel_id: int) -> None:
        """Reset conversation history for a channel (keep the session alive)."""
        session = self.get_or_create(channel_id)
        session.history = []
        session.pending_message_id = None

    def active(self) -> list[int]:
        """Return list of channel IDs with active sessions."""
        return list(self._sessions.keys())

    def clear(self) -> None:
        """Clear all sessions (called on shutdown)."""
        self._sessions.clear()
