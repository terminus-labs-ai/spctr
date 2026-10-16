"""Reminder feature (spc-138 epic, slice spc-139).

Per-agent one-shot reminders: a single ``reminders`` config block enables
both the ``reminder`` tool and the delivery loop (FR6). Reminders persist
in a per-agent SQLite store, deliver at due time through the owning
interface's send callable, and are at-least-once durable (FR17).

This slice ships ``create`` only (default destination); ``list``/``cancel``
and named channels arrive in spc-140, Discord delivery in spc-141.
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
