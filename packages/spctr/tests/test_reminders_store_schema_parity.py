"""Schema-parity tests for the reminders store (spc-145).

AC1: the single ordered source (_COLUMN_DEFS) defines all seventeen
columns exactly once; the rendered CREATE TABLE body matches the live
schema.

AC2: a live schema-parity check via PRAGMA table_info — on a fresh store
the name set equals the single source; on a legacy fixture file pinned
to the spc-139-era DDL (missing next_attempt_at), migration to full
parity occurs when the store opens.

These tests are C-owned on a direct-profile bead; the acceptance
criteria are the definition of done.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from spctr.reminders.store import _COLUMNS, _COLUMN_DEFS, ReminderStore

# The spc-139-era DDL, recovered verbatim from git history
# (commit ffab217). It has 16 columns; next_attempt_at was added by spc-157.
_SPC139_DDL = """
CREATE TABLE IF NOT EXISTS reminders (
  id TEXT PRIMARY KEY,
  agent TEXT NOT NULL,
  platform TEXT NOT NULL,
  requester_id TEXT NOT NULL,
  requester_mention TEXT NOT NULL,
  origin_channel_id TEXT NOT NULL,
  dest_channel_id TEXT NOT NULL,
  dest_channel_name TEXT NOT NULL,
  dest_topic TEXT NOT NULL,
  text TEXT NOT NULL,
  due_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  delivered_at TEXT NULL,
  status TEXT NOT NULL CHECK (status IN
    ('pending', 'sending', 'delivered', 'failed', 'cancelled')),
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT NULL
);
"""


def test_ac1_column_defs_are_single_source():
    """AC1: _COLUMN_DEFS defines all 17 columns exactly once, in order.
    _COLUMNS is derived from it and matches the same name set."""
    assert len(_COLUMN_DEFS) == 17
    names = [entry.name for entry in _COLUMN_DEFS]
    assert len(names) == len(set(names)), "duplicate column names in _COLUMN_DEFS"
    assert names == list(_COLUMNS), "_COLUMNS must be derived from _COLUMN_DEFS in the same order"


def test_ac1_create_table_body_matches_live_schema(tmp_path: Path):
    """AC1: the rendered CREATE TABLE body produces exactly the columns
    in _COLUMN_DEFS — no more, no less, same order."""
    store = ReminderStore(tmp_path / "reminders.sqlite")
    with store._connect() as conn:
        rows = conn.execute("PRAGMA table_info(reminders)").fetchall()
    live_names = [r["name"] for r in rows]
    expected_names = [entry.name for entry in _COLUMN_DEFS]
    assert live_names == expected_names, (
        f"Live schema columns {live_names} != _COLUMN_DEFS {expected_names}"
    )


def test_ac2_fresh_store_schema_matches_source(tmp_path: Path):
    """AC2(a): on a fresh store, the live schema name set equals
    set(_COLUMNS) (which is derived from _COLUMN_DEFS)."""
    store = ReminderStore(tmp_path / "reminders.sqlite")
    with store._connect() as conn:
        rows = conn.execute("PRAGMA table_info(reminders)").fetchall()
    live_names = {r["name"] for r in rows}
    assert live_names == set(_COLUMNS), (
        f"Fresh store columns {live_names} != set(_COLUMNS) {set(_COLUMNS)}"
    )


def test_ac2_legacy_fixture_migrates_to_full_parity(tmp_path: Path):
    """AC2(b): opening a store over a legacy fixture file pinned to the
    spc-139-era DDL (missing next_attempt_at) migrates it to full parity.
    The migration is additive: existing rows keep NULL and stay
    deliverable."""
    db_path = tmp_path / "legacy_reminders.sqlite"
    # Create the legacy fixture: spc-139 DDL (16 columns, no next_attempt_at).
    conn = sqlite3.connect(str(db_path))
    conn.execute(_SPC139_DDL)
    conn.commit()
    conn.close()

    # Verify the fixture really is missing the retry column.
    conn = sqlite3.connect(str(db_path))
    fixture_names = {
        row[1] for row in conn.execute("PRAGMA table_info(reminders)")
    }
    conn.close()
    assert "next_attempt_at" not in fixture_names, (
        "Fixture DDL must be the spc-139 era (no next_attempt_at)"
    )
    assert len(fixture_names) == 16, f"Expected 16 columns, got {len(fixture_names)}"

    # Open the store: migration should add the missing column.
    store = ReminderStore(db_path)
    with store._connect() as conn:
        rows = conn.execute("PRAGMA table_info(reminders)").fetchall()
    migrated_names = {r["name"] for r in rows}
    assert migrated_names == set(_COLUMNS), (
        f"Migrated schema {migrated_names} != set(_COLUMNS) {set(_COLUMNS)}"
    )
    assert len(migrated_names) == 17, f"Expected 17 columns after migration, got {len(migrated_names)}"
