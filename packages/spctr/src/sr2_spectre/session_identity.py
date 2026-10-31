"""Current session identity — per-turn ContextVar (spc-137).

Leaf module: imports nothing from ``sr2_spectre`` so any module (session,
tools) can import it without cycles.

One process hosts many Sessions, so a process-wide environment variable
cannot name *the* session. A ContextVar names it per task: ``Session``
sets it at the start of every turn, and tools read it at call time. The
SR2 pipeline awaits the tool executor inline in the turn flow, so a set
at turn start is visible to the tool calls that turn makes.

Mirrors the documented ``turn_scope`` idiom in
``sr2_spectre.reminders.tool``: consumers must treat a ``ContextVar.set``
inside a function's own frame as making that function a coroutine whose
context attaches to whoever awaits it (PEP 550) — the set/reset here
lives in ``Session.stream_message``, which is an async generator started
by the turn's own task.
"""
from __future__ import annotations

from contextvars import ContextVar

# This turn's session identity: the session log stem when logging is on,
# otherwise the frame_id. None outside a session turn — consumers (the
# terminal tool) then behave exactly as if this feature did not exist.
current_session_id: ContextVar[str | None] = ContextVar(
    "spectre_session_id", default=None
)
