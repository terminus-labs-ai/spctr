"""Reminders store column dedup (spc-144, spc-165).

Direct-profile unit tests for the spc-144 AC subset (R1, R2) and the
spc-165 value-binding fix. The store refactor names the reminder column
set once and maps rows to ``Reminder`` by column name; behavior must not
change.

- R1: one module-level column definition; ``due()``, ``get()`` and
  ``create()`` derive from it; row mapping is by name, not index.
- R2: a full round trip (create → get → due) preserves every ``Reminder``
  field, including create()'s fixed 'pending'/0 status and the timestamp
  parsing (due_at, created_at, delivered_at).

Everything is a temp-file SQLite database under tmp_path — no network,
no real waiting, no ~/.sr2 live state.
"""
from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from spctr import reminders
from spctr.reminders.store import (
    _COLUMNS,
    _RETRY_COLUMN,
    Reminder,
    ReminderStore,
)

UTC = timezone.utc

NOW = datetime(2026, 10, 15, 22, 0, 0, tzinfo=UTC)

# spc-139-era DDL (16 columns, no retry column), the legacy file shape
# ``_migrate`` is meant to upgrade in place. Pinned here independently of
# ``_SCHEMA`` so the fixture cannot silently become the current schema.
_LEGACY_DDL = """
CREATE TABLE reminders (
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

# The columns the spc-139 file has, in DDL order — literally named, so the
# fixture above never derives the legacy shape from the current source.
_LEGACY_INSERT_COLUMNS = (
    "id",
    "agent",
    "platform",
    "requester_id",
    "requester_mention",
    "origin_channel_id",
    "dest_channel_id",
    "dest_channel_name",
    "dest_topic",
    "text",
    "due_at",
    "created_at",
    "delivered_at",
    "status",
    "attempts",
    "last_error",
)


def _legacy_file(db_path: Path, reminder_id: str = "r1") -> Reminder:
    """Write a legacy-schema store file holding one pending row.

    The INSERT names the sixteen spc-139 columns literally (no
    ``_RETRY_COLUMN``), so the fixture is independent of the source of
    truth: a column added to ``_COLUMN_DEFS`` without this fixture being
    taught about it fails the test loudly rather than silently building a
    current-schema file. The expected read-back is the ``Reminder`` the row
    was built from — stated here, never taken from the code under test.
    """
    row = _row(id=reminder_id)
    values = []
    for name in _LEGACY_INSERT_COLUMNS:
        field = getattr(row, name)
        if isinstance(field, datetime):
            field = field.astimezone(UTC).isoformat()
        values.append(field)
    conn = sqlite3.connect(str(db_path))
    conn.execute(_LEGACY_DDL)
    conn.execute(
        f"INSERT INTO reminders ({', '.join(_LEGACY_INSERT_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(_LEGACY_INSERT_COLUMNS))})",
        values,
    )
    conn.commit()
    conn.close()
    assert row.next_attempt_at is None, (
        "the fixture's expected read-back assumes a fresh row schedules no retry"
    )
    return row


class _AlterFailureInjector:
    """Raise once on the retry-column ``ALTER TABLE`` of the first pass.

    Installs a ``sqlite3.connect`` wrapper for the duration of its use, and
    removes only its own wrapper on exit (a later wrapper, e.g. r4's spy,
    stays chained). A connection ``factory`` is passed through, so the
    injection connection still maps rows by column name.
    """

    FAILURE = "injected: database is locked (spc-1075)"

    def __init__(self):
        self.alter_seen = 0
        self._real_connect = None

    def __enter__(self):
        self._real_connect = sqlite3.connect
        real = self._real_connect
        outer = self

        class Injected(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                if isinstance(sql, str) and "ALTER TABLE" in sql.upper():
                    outer.alter_seen += 1
                    raise sqlite3.OperationalError(outer.FAILURE)
                return super().execute(sql, *args, **kwargs)

        def connect(path, *args, **kwargs):
            kwargs.setdefault("factory", Injected)
            return real(path, *args, **kwargs)

        sqlite3.connect = connect
        return outer

    def __exit__(self, *exc_info):
        if sqlite3.connect is not self._real_connect:
            sqlite3.connect = self._real_connect
        return False


def _row(**overrides) -> Reminder:
    """A fully populated pending reminder row (all seventeen fields)."""
    base = dict(
        id="r1",
        agent="miranda",
        platform="zulip",
        requester_id="8",
        requester_mention="@8",
        origin_channel_id="42",
        dest_channel_id="42",
        dest_channel_name="spctr",
        dest_topic="reminders",
        text="look into https://example.com/x",
        due_at=NOW + timedelta(minutes=60),
        created_at=NOW,
        delivered_at=None,
        status="pending",
        attempts=0,
        last_error=None,
    )
    base.update(overrides)
    return Reminder(**base)


@pytest.fixture
def store(tmp_path: Path) -> ReminderStore:
    return ReminderStore(tmp_path / "reminders.sqlite")


# ---------------------------------------------------------------------------
# R1 — the column names appear once; due()/get()/create() derive from it
# ---------------------------------------------------------------------------


def test_r1_single_column_definition():
    """R1: store.py names the column set once as a module-level tuple.
    Every ``_COLUMNS`` entry is a real ``Reminder`` field (a column the
    store persists must exist to map into), and every ``Reminder`` field
    without a default is persisted (a required field may never be
    silently dropped). Tuple equality is deliberately gone: a new
    nullable column (defaulted on the dataclass) must not break R1."""
    store_mod = reminders.store
    assert isinstance(store_mod._COLUMNS, tuple)
    field_names = {f.name for f in dataclasses.fields(Reminder)}
    assert set(store_mod._COLUMNS) <= field_names
    for field in dataclasses.fields(Reminder):
        if field.default is dataclasses.MISSING:
            assert field.name in store_mod._COLUMNS


def test_r1_migration_probe_does_not_fire_per_operation(tmp_path: Path):
    """spc-163 (spc-248: migration moved off construction to the first
    ``_connect()``): the schema-migration probe (``PRAGMA table_info``)
    runs exactly once per store instance — on the first ``_connect()``,
    which happens during the first operation — and never on a later
    connection. The spy below sees only the statements an operation
    actually needs; a migration smuggled into a per-operation ``_connect``
    body would add a ``PRAGMA table_info`` line to every store call and
    couple this column-dedup file to the migration's concerns."""
    record_sql: list[str] = []

    class SpyConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            record_sql.append(sql)
            return super().execute(sql, *args, **kwargs)

    # Build the store with the real connect: construction is now
    # side-effect-free, so migration rides on the first _connect() —
    # warm the store up before installing the spy so it falls outside
    # the per-operation window. (A bare get here, not the spy: the warm-up
    # connection is the migration connection and _migrate reads rows by
    # name, which the spy's row factory would break.)
    store = ReminderStore(tmp_path / "reminders.sqlite")
    store.get("r-none")

    real_connect = sqlite3.connect

    def spy_connect(path, *args, **kwargs):
        kwargs["factory"] = SpyConnection
        return real_connect(path, *args, **kwargs)

    sqlite3.connect = spy_connect
    try:
        store.create(_row())
        store.get("r1")
        store.due(now=NOW + timedelta(hours=1))
    finally:
        sqlite3.connect = real_connect

    assert [s for s in record_sql if "table_info" in s.lower()] == []


# ---------------------------------------------------------------------------
# R2 — round trip create → get → due preserves every field
# ---------------------------------------------------------------------------


def test_r2_round_trip_preserves_every_field(store):
    """R1 + R2: a reminder written by create() reads back through get()
    and due() with every Reminder field preserved: timestamps parsed
    (due_at, created_at, delivered_at), status/attempts the fixed
    'pending'/0 of create(). This round trip is also the R1 proof that
    due()/get()/create() stay consistent with ``_COLUMNS`` — no literal
    SQL text, no sqlite3.connect monkeypatch: the behaviour itself is
    what survives a new nullable column."""
    created_id = store.create(_row())
    assert created_id == "r1"

    future = NOW + timedelta(hours=1)
    by_id = store.get(created_id)
    in_due = store.due(now=future)

    assert [r.id for r in in_due] == [created_id]
    for got in (by_id, in_due[0]):
        # Field-by-field equality over all seventeen dataclass fields.
        assert dataclasses.asdict(got) == dataclasses.asdict(_row())
        assert isinstance(got.due_at, datetime)
        assert isinstance(got.created_at, datetime)
        assert got.due_at == NOW + timedelta(minutes=60)
        assert got.created_at == NOW
        assert got.delivered_at is None
        assert got.status == "pending"
        assert got.attempts == 0
        assert got.last_error is None


def test_r2_pending_statuses_default(store):
    """R2: due() with its default statuses returns pending rows; a row
    claimed to 'sending' is not due until 'sending' is asked for."""
    store.create(_row())
    assert store.claim("r1") is True
    future = NOW + timedelta(hours=1)
    assert store.due(now=future) == []
    assert [r.id for r in store.due(now=future, statuses=("sending",))] == ["r1"]


def test_r2_round_trip_after_delivered(store):
    """R2: delivered_at parses back as a datetime and last_error survives
    a failed attempt, mapped by column name."""
    due_at = NOW + timedelta(minutes=30)
    store.create(_row(due_at=due_at))
    store.mark_failed_attempt("r1", "boom")
    assert store.get("r1").attempts == 1
    assert store.get("r1").last_error == "boom"

    delivered_at = NOW + timedelta(minutes=31)
    store.claim("r1")
    store.mark_delivered("r1", delivered_at)

    got = store.get("r1")
    assert got.status == "delivered"
    assert got.delivered_at == delivered_at
    # Past-due delivered rows only surface when 'delivered' is requested.
    in_due = store.due(now=delivered_at, statuses=("delivered",))[0]
    assert dataclasses.asdict(in_due) == dataclasses.asdict(got)


# ---------------------------------------------------------------------------
# R3 — create() binds next_attempt_at (spc-165)
# ---------------------------------------------------------------------------


def test_r3_create_preserves_next_attempt_at(store):
    """spc-165: create() must persist a supplied next_attempt_at so that
    an import or restore of an in-flight reminder keeps its retry
    schedule instead of silently dropping it."""
    retry_at = NOW + timedelta(minutes=5)
    row = _row(next_attempt_at=retry_at)
    created_id = store.create(row)
    got = store.get(created_id)
    assert got.next_attempt_at == retry_at


def test_r3_create_next_attempt_at_defaults_to_none(store):
    """spc-165: when next_attempt_at is not supplied (default None),
    create() writes NULL and the round-trip returns None."""
    created_id = store.create(_row())
    got = store.get(created_id)
    assert got.next_attempt_at is None


# ---------------------------------------------------------------------------
# R4 — _row_to_reminder covers every column in _COLUMNS (spc-165)
# ---------------------------------------------------------------------------


def test_r4_row_to_reminder_completeness(store):
    """spc-165: _row_to_reminder must map every _COLUMNS entry to a
    Reminder field; adding a column without updating the mapping must
    fail here, not silently drop data."""
    created_id = store.create(_row())
    # Read the raw row to confirm all columns are present.
    with store._connect() as conn:
        raw = conn.execute(
            "SELECT * FROM reminders WHERE id = ?", (created_id,)
        ).fetchone()
    raw_keys = set(raw.keys())
    assert raw_keys == set(_COLUMNS), (
        f"DB columns and _COLUMNS diverge: extra={raw_keys - set(_COLUMNS)}, "
        f"missing={set(_COLUMNS) - raw_keys}"
    )
    # The mapping produces a Reminder with every field populated.
    got = store.get(created_id)
    field_names = {f.name for f in dataclasses.fields(Reminder)}
    assert field_names == set(_COLUMNS), (
        f"_COLUMNS and Reminder fields diverge: "
        f"extra={set(_COLUMNS) - field_names}, "
        f"missing={field_names - set(_COLUMNS)}"
    )


# ---------------------------------------------------------------------------
# R5 — construction is side-effect-free (spc-248)
# ---------------------------------------------------------------------------


def test_r5_construction_writes_nothing_until_first_operation(tmp_path: Path):
    """spc-248: constructing a store for a path under a fresh tmp_path
    creates neither the parent directory nor the database file. The first
    operation (create here) triggers mkdir + migration + file creation,
    and the row then round-trips through get() and due()."""
    db_path = tmp_path / "sub" / "reminders.sqlite"
    store = ReminderStore(db_path)

    assert not (tmp_path / "sub").exists()
    assert not db_path.exists()

    created_id = store.create(_row())
    assert created_id == "r1"

    assert (tmp_path / "sub").is_dir()
    assert db_path.is_file()

    assert dataclasses.asdict(store.get("r1")) == dataclasses.asdict(_row())
    assert [r.id for r in store.due(now=NOW + timedelta(hours=1))] == ["r1"]


# ---------------------------------------------------------------------------
# spc-1075 — a failed initialization pass must not mark the store ready
# ---------------------------------------------------------------------------


def _fail_first_migration(db_path: Path):
    """Build a store over a legacy file whose first migration pass is injected to raise.

    A store built over an *existing* file runs its migration pass during
    construction (spc-248), and that first attempt is the one that must
    fail. The instance is captured before the pass runs so it survives the
    raise: the object handed back is the one whose first pass failed, with
    the injection already removed and no pass retried since.
    """
    store = ReminderStore.__new__(ReminderStore)
    with _AlterFailureInjector() as injector:
        with pytest.raises(sqlite3.OperationalError) as failure:
            store.__init__(db_path)
            # Unreachable: construction is where the first pass runs.
            raise AssertionError("the injected first pass did not raise")
    assert injector.alter_seen == 1, (
        "the injection never hit the retry-column ALTER: "
        f"{injector.alter_seen} ALTERs seen"
    )
    assert _AlterFailureInjector.FAILURE in str(failure.value)
    return store


def test_failed_first_migration_pass_leaves_the_store_not_ready(tmp_path: Path):
    """r1 (AC1075-a): with the first ``ALTER TABLE ... ADD COLUMN
    next_attempt_at`` injected to raise, the first operation raises and the
    instance is left NOT ready.

    ``_ready`` is checked immediately after the failure, and
    ``_initializing`` must be back to False: the guard resets
    unconditionally, so the only thing that may stay set after a failed
    pass is nothing at all — the next ``_connect()`` has to be able to
    retry the pass.
    """
    db_path = tmp_path / "reminders.sqlite"
    _legacy_file(db_path)
    store = _fail_first_migration(db_path)

    assert store._ready is False
    assert store._initializing is False
    # And the file on disk is still the unmigrated legacy schema.
    assert _retry_column_present(db_path) is False


def test_retry_after_failed_pass_succeeds_on_the_same_instance(tmp_path: Path):
    """r2 (AC1075-a): with the injection removed, the NEXT operation on the
    SAME instance retries the pass and succeeds; the store then works.

    Against the pre-fix implementation this fails twice over: ``_ready``
    stays True after the failed pass, so the retry never happens and both
    ``store.get`` calls below die with
    ``OperationalError: no such column: next_attempt_at``.
    """
    db_path = tmp_path / "reminders.sqlite"
    _legacy_file(db_path)
    store = _fail_first_migration(db_path)
    assert store._ready is False

    # Injection gone: this operation performs the pass that failed.
    retried = store.get("r1")
    assert store._ready is True
    assert retried is not None and retried.id == "r1"

    # Normal operation afterwards: a write and a read round-trip.
    assert store.create(_row(id="ignored")) == "r2"
    assert dataclasses.asdict(store.get("r2")) == dataclasses.asdict(
        _row(id="r2")
    )


def test_retry_after_failed_pass_adds_the_column_additively(tmp_path: Path):
    """r3 (AC1075-b): the retried pass completes the additive migration —
    ``PRAGMA table_info(reminders)`` gains ``next_attempt_at``, and the
    pre-existing legacy row keeps NULL there and still maps to a readable
    ``Reminder`` (fields and types, not just a row that comes back)."""
    db_path = tmp_path / "reminders.sqlite"
    expected = _legacy_file(db_path)
    store = _fail_first_migration(db_path)

    assert store.get("r1") is not None
    names = _table_info_names(db_path)
    assert _RETRY_COLUMN in names, f"retried pass did not add the column: {names}"

    # Additive: the legacy columns keep their order, the new one is last.
    assert list(names[: len(_LEGACY_INSERT_COLUMNS)]) == list(
        _LEGACY_INSERT_COLUMNS
    )

    got = store.get("r1")
    assert got.next_attempt_at is None
    assert dataclasses.asdict(got) == dataclasses.asdict(expected)
    assert isinstance(got.due_at, datetime) and isinstance(got.created_at, datetime)
    assert got.due_at == NOW + timedelta(minutes=60)
    assert got.delivered_at is None
    # A legacy row is due on its own due time, as the migration promises.
    assert [r.id for r in store.due(now=NOW + timedelta(hours=1))] == ["r1"]
    assert store.due(now=NOW) == []


def test_retry_after_failed_pass_reaches_the_retry_schedule(tmp_path: Path):
    """r3 (AC1075-b), the column's purpose: after the retried pass a legacy
    row can take a retry schedule through ``due_with_retry`` (which writes
    ``next_attempt_at``) and read it back.

    The pre-fix implementation has no column to write to, so the failure
    mode this pins — a store permanently wedged on an unmigrated file —
    raises here instead of scheduling.
    """
    db_path = tmp_path / "reminders.sqlite"
    _legacy_file(db_path)
    store = _fail_first_migration(db_path)
    assert store._ready is False

    retry_at = NOW + timedelta(minutes=5)
    store.due_with_retry("r1", "first send failed", retry_at)
    got = store.get("r1")
    assert got.next_attempt_at == retry_at
    assert got.attempts == 1
    assert got.last_error == "first send failed"
    # The retry schedule, not due_at, decides when the next attempt is due.
    assert [r.id for r in store.due(now=retry_at)] == ["r1"]
    assert store.due(now=retry_at - timedelta(minutes=1)) == []


def test_fresh_path_migrates_exactly_once_across_two_operations(
    tmp_path: Path,
):
    """r4 (AC1075-c): on the success path a fresh path still initializes
    exactly once per instance — across two operations the parent mkdir runs
    once and the migration pass (``_migrate``) runs once, because the
    ``_initializing`` guard lets the pass's own inner ``_connect()`` through
    without re-entry and ``_ready`` then short-circuits every later call.

    Re-entering the pass would double the mkdir and the migration; a
    migration smuggled back into the per-operation ``_connect()`` would make
    ``migrate`` 2. Connections are counted too, as context for the single
    migration pass (one for the pass, one per operation).
    """
    record: list[str] = []

    real_mkdir = Path.mkdir

    def spy_mkdir(self, *args, **kwargs):
        record.append("mkdir")
        return real_mkdir(self, *args, **kwargs)
    real_connect = sqlite3.connect

    def spy_connect(path, *args, **kwargs):
        record.append("connect")
        return real_connect(path, *args, **kwargs)

    store = ReminderStore(tmp_path / "reminders.sqlite")
    real_migrate = ReminderStore._migrate

    def spy_migrate(conn):
        record.append("migrate")
        return real_migrate(conn)

    Path.mkdir = spy_mkdir
    sqlite3.connect = spy_connect
    ReminderStore._migrate = staticmethod(spy_migrate)
    try:
        store.get("r-none")
        store.get("r-none")
    finally:
        Path.mkdir = real_mkdir
        sqlite3.connect = real_connect
        ReminderStore._migrate = real_migrate

    assert record.count("mkdir") == 1, record
    assert record.count("migrate") == 1, record
    # One connection for the pass itself plus one per operation: the pass is
    # not re-entered from inside a later operation's connection.
    assert record == ["mkdir", "connect", "migrate", "connect", "connect"], record

    # After the guard is restored, a second connection opens no second pass
    # and the store is usable.
    assert store.create(_row()) == "r1"
    assert dataclasses.asdict(store.get("r1")) == dataclasses.asdict(_row())


def _table_info_names(db_path: Path) -> tuple[str, ...]:
    """Column names of ``reminders``, in schema order, straight from the
    file (no store involved, so no migration can ride along)."""
    conn = sqlite3.connect(str(db_path))
    try:
        return tuple(
            row[1] for row in conn.execute("PRAGMA table_info(reminders)")
        )
    finally:
        conn.close()


def _retry_column_present(db_path: Path) -> bool:
    return _RETRY_COLUMN in _table_info_names(db_path)
