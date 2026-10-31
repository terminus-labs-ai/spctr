"""FrameRouter — origin binding + ambient frame minting (FR3).

Holds two maps:
- origin → frame_id (mutable current binding)
- frame_id → Session (durable per-frame state)

On an inbound message from an origin with no bound frame, mints a new
ambient frame (convo:<ULID>), creates its Session, and binds it.
"""

from __future__ import annotations

from dataclasses import dataclass

import ulid

from sr2_spectre.runtime import Runtime


@dataclass(frozen=True)
class FrameInfo:
    """Information about an active frame for handoff UI."""
    frame_id: str
    origin: str | None  # None if frame exists but has no current binding


class FrameRouter:
    """Routes origins to per-frame Sessions, minting ambient frames as needed.

    Two maps:
    - _binding: origin → frame_id (mutable)
    - _sessions: frame_id → Session (durable)

    Ambient frames are minted as ``convo:<ULID>`` on first contact from
    an unbound origin.
    """

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._binding: dict[str, str] = {}  # origin -> frame_id
        self._sessions: dict[str, object] = {}  # frame_id -> Session

    def resolve(self, origin: str) -> object:
        """Return the Session for *origin*, minting an ambient frame if unbound.

        If *origin* has no binding, creates a new ``convo:<ULID>`` frame,
        builds its Session from the Runtime, and binds it.
        """
        frame_id = self._binding.get(origin)
        if frame_id is not None:
            return self._sessions[frame_id]

        # Mint ambient frame
        frame_id = f"convo:{ulid.ULID()}"
        session = self._runtime.new_session(frame_id=frame_id)
        self._sessions[frame_id] = session
        self._binding[origin] = frame_id
        return session

    def get_binding(self, origin: str) -> str | None:
        """Return the frame_id bound to *origin*, or None."""
        return self._binding.get(origin)

    def bind(self, origin: str, frame_id: str) -> None:
        """Bind *origin* to an existing frame, detaching the old binding.

        Atomic rebind: if *frame_id* was previously bound to another origin,
        that origin is detached first. The original origin (if any) is also
        detached.

        Raises KeyError if *frame_id* is not a known session.
        """
        if frame_id not in self._sessions:
            raise KeyError(f"frame_id {frame_id!r} not found in router sessions")

        # Detach frame from any prior origin (at-most-one-binding)
        old_origin = self._find_origin_by_frame(frame_id)
        if old_origin is not None:
            del self._binding[old_origin]

        # Detach origin from any prior frame
        if origin in self._binding:
            del self._binding[origin]

        # Attach
        self._binding[origin] = frame_id

    def list_frames(self) -> list[FrameInfo]:
        """Return info for all active frames with their current origin binding."""
        # Build reverse map: frame_id -> origin (last one wins if multiple,
        # though at-most-one-binding means this shouldn't happen)
        frame_to_origin: dict[str, str] = {}
        for origin, frame_id in self._binding.items():
            frame_to_origin[frame_id] = origin

        return [
            FrameInfo(frame_id=frame_id, origin=frame_to_origin.get(frame_id))
            for frame_id in self._sessions
        ]

    def _find_origin_by_frame(self, frame_id: str) -> str | None:
        """Find which origin (if any) is bound to *frame_id*."""
        for origin, fid in self._binding.items():
            if fid == frame_id:
                return origin
        return None
