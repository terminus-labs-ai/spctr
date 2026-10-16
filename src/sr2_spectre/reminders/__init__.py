"""Reminder feature (spc-138 epic, slice spc-139).

Per-agent one-shot reminders: a single ``reminders`` config block enables
both the ``reminder`` tool and the delivery loop (FR6). Reminders persist
in a per-agent SQLite store, deliver at due time through the owning
interface's send callable, and are at-least-once durable (FR17).

spc-139 ships create and delivery; spc-140 adds list, cancel, named
channels and the FR18 retry policy; Discord delivery arrives in spc-141.
"""
from __future__ import annotations

from sr2_spectre.reminders.loop import DeliveryLoop
from sr2_spectre.reminders.store import Reminder, ReminderStore
from sr2_spectre.reminders.tool import ReminderOrigin, ReminderTool

__all__ = [
    "DeliveryLoop",
    "Reminder",
    "ReminderOrigin",
    "ReminderStore",
    "ReminderTool",
]
