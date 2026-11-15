"""Shared turn-scope helper for the reminder test modules (spc-1073).

``test_reminders_due_time.py``, ``test_reminders_resolver_seam.py`` and
``test_reminders_zulip.py`` each needed the same ``_in_turn`` turn-scope
context manager (introduced by spc-170 when the reminder tests moved onto
the task-local ``turn_scope``); this module is the single definition.
Import it as a plain top-level module, as ``zulip_e2e_support`` is:
``tests/__init__.py`` puts the tests directory on ``sys.path`` via pytest's
rootdir-based ``consider_namespace_packages``/rootdir insertion, and the
sibling test modules import it the same way.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from spctr.reminders import ReminderOrigin
from spctr.reminders.tool import TurnScope, turn_scope


@contextmanager
def _in_turn(origin: ReminderOrigin) -> Iterator[None]:
    """This task's ``turn_scope`` set to *origin*, reset in ``finally``.

    The ContextVar is read at call time inside the tool's coroutine, so
    wrapping the awaited tool call keeps the scope out of every other test.
    """
    token = turn_scope.set(TurnScope(origin=origin))
    try:
        yield
    finally:
        turn_scope.reset(token)
