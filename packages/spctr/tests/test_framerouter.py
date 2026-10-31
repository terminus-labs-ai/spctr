"""Tests for FrameRouter — origin binding + ambient frame minting (FR3).

Covers:
- Fresh origin mints a convo:<ULID> ambient frame
- Second message on same origin routes to same frame
- bind() rebinds atomically (detach old, attach new)
- list_frames() returns correct info
- resolve() returns correct Session per origin
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from sr2_spectre.framerouter import FrameInfo, FrameRouter


@pytest.fixture
def runtime():
    """Mock Runtime that tracks created sessions."""
    runtime = MagicMock()
    runtime._created_sessions = []

    def _new_session(frame_id: str, **kwargs):
        session = MagicMock()
        session.frame_id = frame_id
        runtime._created_sessions.append(session)
        return session

    runtime.new_session = _new_session
    return runtime


# ---------------------------------------------------------------------------
# Ambient frame minting (FR1 + FR3)
# ---------------------------------------------------------------------------

class TestAmbientFrameMinting:
    """Fresh origin mints a convo:<ULID> frame."""

    def test_resolve_new_origin_mints_ambient_frame(self, runtime):
        router = FrameRouter(runtime)

        session = router.resolve("origin-a")

        assert session is not None
        assert session.frame_id.startswith("convo:")
        assert len(runtime._created_sessions) == 1
        assert runtime._created_sessions[0] is session

    def test_second_message_same_origin_returns_same_session(self, runtime):
        router = FrameRouter(runtime)

        session_a = router.resolve("origin-a")
        session_b = router.resolve("origin-a")

        assert session_a is session_b
        assert len(runtime._created_sessions) == 1

    def test_different_origins_get_different_frames(self, runtime):
        router = FrameRouter(runtime)

        session_a = router.resolve("origin-a")
        session_b = router.resolve("origin-b")

        assert session_a is not session_b
        assert session_a.frame_id != session_b.frame_id
        assert len(runtime._created_sessions) == 2

    def test_ambient_frame_id_format(self, runtime):
        """convo:<ULID> — ULID is 26 Base32 chars."""
        router = FrameRouter(runtime)
        session = router.resolve("origin-x")

        # convo:<26-char ULID>
        pattern = r"^convo:[A-HJKMNP-TV-Z0-9]{26}$"
        assert re.match(pattern, session.frame_id)


# ---------------------------------------------------------------------------
# Binding (FR3)
# ---------------------------------------------------------------------------

class TestBinding:
    """Origin → frame_id binding map."""

    def test_resolve_populates_binding(self, runtime):
        router = FrameRouter(runtime)
        session = router.resolve("origin-a")

        assert router.get_binding("origin-a") == session.frame_id

    def test_unbound_origin_returns_none(self, runtime):
        router = FrameRouter(runtime)
        assert router.get_binding("nonexistent") is None


# ---------------------------------------------------------------------------
# Atomic rebind (FR9 — handoff prep)
# ---------------------------------------------------------------------------

class TestAtomicRebind:
    """bind(origin, frame_id) detaches old origin, attaches new."""

    def test_rebind_detaches_old_origin(self, runtime):
        router = FrameRouter(runtime)
        session_a = router.resolve("origin-a")

        router.bind("origin-b", session_a.frame_id)

        # origin-a no longer bound to session_a
        assert router.get_binding("origin-a") is None
        # origin-b now bound to session_a
        assert router.get_binding("origin-b") == session_a.frame_id

    def test_rebind_unknown_frame_raises(self, runtime):
        router = FrameRouter(runtime)

        with pytest.raises(KeyError, match="frame_id.*not found"):
            router.bind("origin-x", "convo:nonexistent")

    def test_bind_existing_frame_to_new_origin_keeps_session(self, runtime):
        """Rebinding doesn't destroy the Session — it moves it."""
        router = FrameRouter(runtime)
        session = router.resolve("origin-a")
        original_frame_id = session.frame_id

        router.bind("origin-b", original_frame_id)

        assert router.get_binding("origin-b") == original_frame_id
        assert session in router._sessions.values()


# ---------------------------------------------------------------------------
# list_frames (FR9 — handoff UI)
# ---------------------------------------------------------------------------

class TestListFrames:
    """list_frames() returns FrameInfo with origin + frame_id."""

    def test_empty_router_returns_empty_list(self, runtime):
        router = FrameRouter(runtime)
        assert router.list_frames() == []

    def test_list_shows_all_frames_with_origin(self, runtime):
        router = FrameRouter(runtime)
        session_a = router.resolve("origin-a")
        session_b = router.resolve("origin-b")

        frames = router.list_frames()

        assert len(frames) == 2
        frame_map = {f.frame_id: f for f in frames}

        assert frame_map[session_a.frame_id].origin == "origin-a"
        assert frame_map[session_b.frame_id].origin == "origin-b"

    def test_frame_info_type(self, runtime):
        router = FrameRouter(runtime)
        router.resolve("origin-x")

        frames = router.list_frames()
        assert isinstance(frames[0], FrameInfo)
        assert hasattr(frames[0], "frame_id")
        assert hasattr(frames[0], "origin")


# ---------------------------------------------------------------------------
# Concurrent frame isolation (acceptance criteria)
# ---------------------------------------------------------------------------

class TestConcurrentIsolation:
    """Two frames don't see each other's history."""

    def test_frames_have_independent_sessions(self, runtime):
        """Each frame gets its own Session object."""
        router = FrameRouter(runtime)
        s_a = router.resolve("origin-a")
        s_b = router.resolve("origin-b")

        assert s_a is not s_b
        assert s_a.frame_id != s_b.frame_id
