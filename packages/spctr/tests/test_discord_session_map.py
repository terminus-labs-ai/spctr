"""Tests for SessionMap — session-per-channel management.

Covers:
1.  get_or_create() — creates new sessions for unknown channels
2.  session_id_for() — deterministic session ID generation
3.  reset() — clears history without removing the session
4.  active() — lists active channel IDs
5.  clear() — removes all sessions
"""
from __future__ import annotations

import pytest

from sr2_spectre.interfaces.discord.session_map import ChannelSession, SessionMap


class TestChannelSession:
    def test_session_id_for_deterministic(self) -> None:
        """session_id_for produces the same ID for the same channel."""
        assert ChannelSession.session_id_for(123) == "discord-123"
        assert ChannelSession.session_id_for(456) == "discord-456"
        assert ChannelSession.session_id_for(123) == "discord-123"

    def test_default_values(self) -> None:
        session = ChannelSession(channel_id=123, session_id="discord-123")
        assert session.history == []
        assert session.pending_message_id is None


class TestSessionMap:
    def test_get_or_create_new(self) -> None:
        """Creates a new session for an unknown channel."""
        sm = SessionMap()
        session = sm.get_or_create(999)
        assert session.channel_id == 999
        assert session.session_id == "discord-999"

    def test_get_or_create_existing(self) -> None:
        """Returns the same session object for repeated calls."""
        sm = SessionMap()
        s1 = sm.get_or_create(123)
        s2 = sm.get_or_create(123)
        assert s1 is s2

    def test_different_channels_different_sessions(self) -> None:
        sm = SessionMap()
        s1 = sm.get_or_create(123)
        s2 = sm.get_or_create(456)
        assert s1 is not s2
        assert s1.session_id != s2.session_id

    def test_reset_clears_history(self) -> None:
        sm = SessionMap()
        session = sm.get_or_create(123)
        session.history = [{"role": "user", "content": []}]
        session.pending_message_id = 987
        sm.reset(123)
        assert session.history == []
        assert session.pending_message_id is None

    def test_reset_nonexistent_creates_session(self) -> None:
        """Resetting an unknown channel creates the session first."""
        sm = SessionMap()
        sm.reset(999)
        session = sm.get_or_create(999)
        assert session.history == []

    def test_active_lists_channels(self) -> None:
        sm = SessionMap()
        sm.get_or_create(1)
        sm.get_or_create(2)
        sm.get_or_create(3)
        assert sorted(sm.active()) == [1, 2, 3]

    def test_active_empty_initially(self) -> None:
        sm = SessionMap()
        assert sm.active() == []

    def test_clear_removes_all(self) -> None:
        sm = SessionMap()
        sm.get_or_create(1)
        sm.get_or_create(2)
        sm.clear()
        assert sm.active() == []

    def test_independent_histories(self) -> None:
        """Each channel maintains independent history."""
        sm = SessionMap()
        s1 = sm.get_or_create(1)
        s2 = sm.get_or_create(2)
        s1.history.append({"role": "user", "text": "hello channel 1"})
        s2.history.append({"role": "user", "text": "hello channel 2"})
        assert len(s1.history) == 1
        assert len(s2.history) == 1
        assert s1.history[0]["text"] == "hello channel 1"
        assert s2.history[0]["text"] == "hello channel 2"
