"""REPL unique session keys and /resume listing + takeover (spc-132, FR 9-12).

AC10 — unique keys: a REPL start on the constructed default identity mints a
fresh ``<name>-repl-<YYYYmmdd-HHMMSS>`` key (collision-guarded against
already-archived keys via ``list_sessions``); an explicit ``--session-id``
survives; ``/reset`` mints a new key and the old key's archive stays frozen.
AC12 — takeover: ``/resume <n|key>`` restores the target's resumable history
through the increment-1 reconstruction (interrupted marker included), adopts
the key as the agent's identity, re-seeds the next turn's ``turn_seq`` to the
archived max, and the next turn archives under the adopted key — the run's
previous key receives nothing new.  Unknown keys, an empty archive, a restore
failure and a missing store all error without switching.
AC13 — listing: bare ``/resume`` lists up to 15 archived sessions
newest-first as ``<idx>  <YYYY-MM-DD HH:MM>  <turns>  <session-id>`` with the
current key marked; out-of-range indexes error without switching; exact keys
need no prior listing.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sr2.models import Message, TextBlock
from sr2.pipeline.provenance import (
    ContextLifecycle,
    ContextScope,
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
    SessionSummary,
    TURN_ARCHIVE_LAYER,
)
from sr2_spectre.agent import Agent
from sr2_spectre.config import SpectreConfig
from sr2_spectre.interfaces import repl as repl_mod
from sr2_spectre.interfaces.repl import (
    REPLInterface,
    _needs_repl_key,
    _mint_repl_key,
    render_replay_items,
    render_session_listing,
    resolve_resume_target,
)
from sr2_spectre.resume import load_resumable_history
from sr2_spectre.runtime import Runtime

KEY_RE = re.compile(r"^miranda-repl-\d{8}-\d{6}$")

BASE = datetime(2026, 10, 30, 12, 0, 0)
MARKER = "[This turn was interrupted by a restart before it finished.]"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _config() -> SpectreConfig:
    return SpectreConfig(
        agent={"name": "miranda"},
        models={"default": {"model": "m", "base_url": "http://x/v1"}},
        pipeline={"layers": []},
    )


def _entry(session_id: str, turn: int, seq: int, content, ts: datetime) -> Entry:
    return Entry(
        id=f"{session_id}-{turn}-{seq}",
        content=content,
        sources=(),
        origin=EntryOrigin(kind="resolver", name="turn_archivist"),
        layer="turn_archive",
        session_id=session_id,
        created_at=ts,
        lifecycle=ContextLifecycle.ARCHIVE,
        scope=ContextScope(owner_id=session_id),
        meta={"turn_seq": turn, "seq": seq},
    )


def _msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def _completed_turn_entries(session_id: str, turns: int, base: datetime = BASE) -> list[Entry]:
    """One user + one assistant entry per turn, one minute apart (naive local)."""
    entries: list[Entry] = []
    for turn in range(turns):
        ts = base + timedelta(minutes=turn * 2)
        entries.append(_entry(session_id, turn, 0, _msg("user", f"u{turn}"), ts))
        entries.append(_entry(session_id, turn, 1, _msg("assistant", f"a{turn}"), ts))
    return entries


async def _seeded_store(fixtures: list[tuple[str, int, datetime]]) -> InMemoryProvenanceStore:
    """InMemory store seeded with completed turns (user+assistant per turn)."""
    store = InMemoryProvenanceStore()
    for session_id, turns, base in fixtures:
        await store.write_batch(_completed_turn_entries(session_id, turns, base))
    return store


def _capturing_iface(tmp_path, monkeypatch) -> tuple[REPLInterface, list[str]]:
    """REPLInterface with a line-capturing console and sandboxed prompt history."""
    from rich.console import Console

    lines: list[str] = []

    class _Sink:
        def write(self, s: str) -> int:
            for part in s.split("\n"):
                if part:
                    lines.append(part)
            return len(s)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(repl_mod, "_history_file", lambda: tmp_path / "hist")
    iface = REPLInterface(console=Console(file=_Sink(), force_terminal=False, width=120))
    return iface, lines


def _plain(lines: list[str]) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(lines))


def _rows(lines: list[str]) -> list[str]:
    """The numbered listing rows (``<idx>  <YYYY-MM-DD HH:MM> ...``)."""
    return [ln.strip() for ln in lines if re.match(
        r"^\s*\d{1,2}  \d{4}-\d{2}-\d{2} \d{2}:\d{2}", _plain([ln]))]


def _local(ts: datetime) -> str:
    return ts.astimezone().strftime("%Y-%m-%d %H:%M")


def _engine_stub() -> MagicMock:
    """A minimal SR2 stand-in for takeover tests.

    Mirrors the shape the takeover re-seat walks on the real object
    (``SR2._engine.seed_turn_seq``) and records the seeded turn numbers, so
    the adopted Session's next-turn numbering is observable without a real
    pipeline.
    """
    engine = MagicMock()
    sr2 = MagicMock()
    sr2.seeded_seq: list[int] = []
    sr2._seq = -1
    sr2.turn_seq = -1

    def _seed(turn_seq: int) -> None:
        sr2.seeded_seq.append(turn_seq)
        sr2._seq = turn_seq

    engine.seed_turn_seq = _seed
    sr2._engine = engine
    return sr2


class _TakeoverAgent:
    """Stand-in facade for takeover tests, exposing the ``/resume`` surface.

    ``session_id`` assignment stands in for ``Agent.session_id``'s setter: it
    replaces the internal Session, which starts with fresh history and its own
    SR2. Here the replacement SR2 is a stub engine recording the continuation
    re-seat, so the takeover's per-identity state is observable without a
    pipeline.
    """

    def __init__(self, session_id: str, store) -> None:
        self.config = _config()
        self.provenance_store = store
        self.history: list = []
        self.sr2 = _engine_stub()
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        # Like Agent._replace_session: a fresh Session (fresh history, fresh
        # SR2 with its own archive continuation) under the new identity.
        self._session_id = value
        self.history = []
        self.sr2 = _engine_stub()


async def _fake_turn(agent, store) -> None:
    """Advance the stub engine one turn and archive it under the current key.

    Mirrors the shipped increment-1 behavior: the turn runs at the seeded
    turn number + 1 and the archivist writes user+assistant entries for the
    adopted key at that turn_seq.
    """
    engine = agent.sr2
    turn = engine._seq + 1
    key = agent.session_id
    ts = BASE + timedelta(minutes=300)
    await store.write_batch([
        _entry(key, turn, 0, _msg("user", "next"), ts),
        _entry(key, turn, 1, _msg("assistant", "done"), ts),
    ])
    engine._seq = turn
    engine.turn_seq = turn


# ---------------------------------------------------------------------------
# Facade passthroughs (the surface /resume consumes)
# ---------------------------------------------------------------------------


def test_agent_provenance_store_passthrough_is_none_when_unconfigured():
    agent = Agent(_config(), session_id="custom")
    assert agent.provenance_store is None


def test_runtime_provenance_store_property_tracks_initialize(tmp_path):
    config = _config()
    config.provenance_store_path = str(tmp_path / "prov.db")
    runtime = Runtime(config)
    assert runtime.provenance_store is None

    async def scenario():
        await runtime.initialize()
        try:
            assert runtime.provenance_store is not None
            session = runtime.new_session("f1")
            assert session._provenance_store is runtime.provenance_store
        finally:
            await runtime.aclose()

    asyncio.run(scenario())
    assert runtime.provenance_store is None


def test_agent_restore_history_passthrough_delegates_to_session():
    agent = Agent(_config(), session_id="custom")
    session = agent._session

    async def scenario():
        result = await agent.restore_history()
        return result, session._resume_result

    result, session_result = asyncio.run(scenario())
    assert result is session_result  # same object: a pure passthrough


# ---------------------------------------------------------------------------
# Pure decision logic
# ---------------------------------------------------------------------------


def test_mint_key_shape_and_local_time():
    stamp = _mint_repl_key("miranda", datetime(2026, 10, 30, 9, 7, 3))
    assert stamp == "miranda-repl-20261030-090703"
    assert KEY_RE.match(stamp)


def test_needs_repl_key_detects_only_default_identity():
    assert _needs_repl_key("miranda-default", "miranda")
    assert not _needs_repl_key("miranda-repl-20261030-090703", "miranda")
    assert not _needs_repl_key("custom-id", "miranda")
    # Another agent's default is not this agent's constructed identity.
    assert not _needs_repl_key("edie-default", "miranda")


def test_listing_rows_shape_and_marker():
    # The store supplies rows newest-activity-first; the helper renders them
    # in the given order, numbering from 1.
    summaries = [
        SessionSummary("new", 2, datetime(2026, 10, 30, 12, 0)),
        SessionSummary("cur", 1, datetime(2026, 10, 30, 11, 30)),
        SessionSummary("old", 3, datetime(2026, 10, 30, 11, 4)),
    ]
    lines, rows = render_session_listing(summaries, "cur", 2)
    assert rows == [
        "1  2026-10-30 12:00  2  new",
        "2  2026-10-30 11:30  1  cur  → current",
    ]
    # The 15-row cap is applied by the helper itself.
    _lines, capped = render_session_listing(
        [SessionSummary(f"s{i}", 1, datetime(2026, 10, 30, 12, i)) for i in range(20)],
        "none",
        15,
    )
    assert len(capped) == 15


def test_listing_converts_utc_last_activity_to_local():
    # A UTC-aware last_activity renders in local time (the InMemory path).
    _lines, rows = render_session_listing(
        [SessionSummary("u", 1, datetime(2026, 10, 30, 13, 0, tzinfo=timezone.utc))],
        "other",
        15,
    )
    assert rows == [f"1  {_local(datetime(2026, 10, 30, 13, 0))}  1  u"]


def test_resolve_resume_target_index_and_exact():
    summaries = [
        SessionSummary("a", 1, BASE),
        SessionSummary("b", 1, BASE),
        SessionSummary("c", 1, BASE),
    ]
    assert resolve_resume_target(summaries, "1") == ("a", None)
    assert resolve_resume_target(summaries, "3") == ("c", None)
    # Out of range: error, no session id.
    sid, err = resolve_resume_target(summaries, "99")
    assert sid is None and "index 99" in err and "Run /resume" in err
    # The listing cap bounds the indexable rows.
    sid, err = resolve_resume_target(summaries, "4", limit=2)
    assert sid is None and "index 4" in err
    # Exact key: resolves directly, listed or not (the archive read decides).
    assert resolve_resume_target(summaries, "b") == ("b", None)
    assert resolve_resume_target([], "exact-key") == ("exact-key", None)
    assert resolve_resume_target(None, "exact-key") == ("exact-key", None)
    assert resolve_resume_target(summaries, "ghost") == ("ghost", None)


# ---------------------------------------------------------------------------
# AC10 — unique keys
# ---------------------------------------------------------------------------


def test_ac10_c1_start_mints_unique_repl_key_per_run(tmp_path, monkeypatch):
    """AC10-c1: a start on <name>-default switches to a key matching the
    unique-repl pattern, and two starts produce different keys."""
    agent = Agent(_config())  # constructed identity: miranda-default
    store = asyncio.run(_seeded_store([]))
    agent._runtime._provenance_store = store
    iface, _lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface.start(agent)
        key1 = agent.session_id
        assert KEY_RE.match(key1)

        # A second start (after a hypothetical return to the default
        # identity) mints its own key; the archived collision guard keeps
        # them apart even inside the same second.
        await store.write_batch(_completed_turn_entries(key1, 1, BASE))
        agent.session_id = "miranda-default"
        await iface.start(agent)
        key2 = agent.session_id
        assert KEY_RE.match(key2)
        assert key2 != key1

    asyncio.run(scenario())


def test_ac10_c1_collision_guard_moves_past_archived_stamp(tmp_path, monkeypatch):
    """A minted key already present in the archive is rejected: the start
    moves to a different key (sub-second restarts cannot share an archive)."""
    # Pre-archive the key this second's mint would produce (both spellings:
    # aware, as fresh writes carry it, and naive, as the SQLite round-trip
    # reads it back) so the guard sees the collision whichever store path
    # the summary came through.
    stamp_now = _mint_repl_key("miranda")
    agent = Agent(_config())
    iface, _lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        store = await _seeded_store([])
        # Pre-archive the key this second's mint would produce (both
        # spellings: aware, as fresh writes carry it, and naive, as the
        # SQLite round-trip reads it back) so the guard sees the collision
        # whichever store path the summary came through.
        for ts in (datetime.now(), datetime.now().replace(tzinfo=None)):
            await store.write_batch(_completed_turn_entries(stamp_now, 1, ts))
        agent._runtime._provenance_store = store
        await iface.start(agent)
        key = agent.session_id
        assert KEY_RE.match(key)
        assert key != stamp_now

    asyncio.run(scenario())


def test_ac10_c2_explicit_session_id_survives_start(tmp_path, monkeypatch):
    agent = Agent(_config(), session_id="custom-id")
    agent._runtime._provenance_store = asyncio.run(_seeded_store([]))
    iface, _ = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface.start(agent)
        assert agent.session_id == "custom-id"

    asyncio.run(scenario())


def test_ac10_c3_reset_mints_new_key_and_freezes_old_archive(tmp_path, monkeypatch):
    """AC10-c3: /reset mid-run moves to a new unique key; the old key's
    archive (two completed turns) is read, never written to."""
    agent = Agent(_config())
    store = asyncio.run(_seeded_store([("miranda-default", 2, BASE)]))
    agent._runtime._provenance_store = store
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface.start(agent)
        minted = agent.session_id

        # Archive a turn under the minted key, as the run would before a
        # reset: the reset must move off this exact key even when the fresh
        # mint lands on the same second.
        await store.write_batch(_completed_turn_entries(minted, 1, datetime.now()))
        await iface._cmd_reset(agent)
        return minted, agent.session_id

    minted, after_reset = asyncio.run(scenario())
    assert KEY_RE.match(minted)
    assert KEY_RE.match(after_reset)
    assert after_reset != minted
    assert "New session started" in _plain(lines)

    # The old key receives no further archive entries: every turn_archive
    # entry still belongs to the pre-reset session, nothing new anywhere.
    old = [e for e in store._store.values() if e.session_id == "miranda-default"]
    assert len(old) == 4  # 2 turns x (user, assistant), unchanged
    assert all(e.layer == TURN_ARCHIVE_LAYER for e in old)
    new_entries = [e for e in store._store.values() if e.session_id == after_reset]
    assert new_entries == []
    assert iface._listing == []  # the stale listing is dropped on reset


def test_ac10_c3_reset_on_non_default_key_falls_back_to_default(tmp_path, monkeypatch):
    """Outside AC10-c3's scope: a run on an explicit --session-id keeps the
    shipped /reset semantics (back to <name>-default), which the REPL only
    overrides for keys it minted itself."""
    agent = Agent(_config(), session_id="explicit-choice")
    iface, _ = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface.start(agent)
        await iface._cmd_reset(agent)
        assert agent.session_id == "miranda-default"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# AC12 — takeover
# ---------------------------------------------------------------------------


def test_ac12_c1_takeover_by_index_continues_archived_turn_seq(tmp_path, monkeypatch):
    """AC12-c1: key P (one turn) archived; fresh run R; /resume 1 → P.

    History equals load_resumable_history(P); session_id is P; the next turn
    archives under P at turn_seq 1 (archived max 0 + 1) and R receives no
    archive entries.
    """
    P, R = "miranda-repl-20261030-110000", "miranda-repl-20261030-120000"
    store = asyncio.run(_seeded_store([(P, 1, BASE)]))  # turn 0: u0 / a0
    agent = _TakeoverAgent(R, store)
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface._cmd_resume(agent, None)  # bare listing primes the cache
        await iface._cmd_resume(agent, "1")  # index resolves against it
        await _fake_turn(agent, store)  # one turn under the adopted key

    asyncio.run(scenario())

    expected = asyncio.run(load_resumable_history(store, P, _config().agent.resume))
    # load_resumable_history is read *after* the scenario: it now covers the
    # resumed turn too ([u0, a0, next, done]); the takeover itself restored
    # the pre-takeover archive.
    assert expected.history == [
        _msg("user", "u0"), _msg("assistant", "a0"),
        _msg("user", "next"), _msg("assistant", "done"),
    ]
    assert expected.interrupted is False

    assert agent.session_id == P
    # The history adopted at takeover time equals the reconstruction of P as
    # it stood then: the archived turn 0, before the resumed run's own turn.
    assert list(agent.history) == [_msg("user", "u0"), _msg("assistant", "a0")]

    # One console line reports the adopted key and restored count.
    report = [ln for ln in lines if "Resumed" in ln]
    assert len(report) == 1
    assert P in report[0] and "2 messages restored" in report[0]

    # Continuation re-seed: the archived max (0), so the next turn ran at 1.
    assert agent.sr2.seeded_seq == [0]
    assert agent.sr2.turn_seq == 1

    # Archive continuity: P's archive holds exactly turn 0 (the two seeded
    # entries) plus turn 1 (the two the resumed run wrote under P); R has
    # nothing at all.
    p_archived = [
        e
        for e in store._store.values()
        if e.session_id == P and e.layer == TURN_ARCHIVE_LAYER
    ]
    assert sorted({e.meta["turn_seq"] for e in p_archived}) == [0, 1]
    assert sorted((e.meta["turn_seq"], e.meta["seq"]) for e in p_archived) == [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ]
    assert [e for e in store._store.values() if e.session_id == R] == []


def test_ac12_c2_interrupted_target_restores_marker_and_reports(tmp_path, monkeypatch):
    """AC12-c2: a target whose last turn ends at a user message restores with
    the configured interrupted marker, and the console line reports it."""
    P = "miranda-repl-20261030-110000"
    store = asyncio.run(_seeded_store([(P, 1, BASE)]))  # turn 0 complete
    agent = _TakeoverAgent("miranda-default", store)
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        # Turn 1 ends at a user message → interrupted.
        await store.write_batch([
            _entry(P, 1, 0, _msg("user", "cut off here"), BASE + timedelta(minutes=120)),
        ])
        await iface._cmd_resume(agent, P)  # exact key, no prior listing

    asyncio.run(scenario())

    assert agent.session_id == P
    assert list(agent.history) == [
        _msg("user", "u0"), _msg("assistant", "a0"),
        _msg("user", "cut off here"), _msg("assistant", MARKER),
    ]
    # The continuation re-seat is the archived max (1) → the next turn is 2.
    assert agent.sr2.seeded_seq == [1]

    report = [ln for ln in lines if "Resumed" in ln]
    assert len(report) == 1
    assert P in report[0] and "interrupted" in report[0]


def test_ac12_c3_unknown_key_and_missing_store_switch_nothing(tmp_path, monkeypatch):
    """AC12-c3: /resume 99 (no listing → error, nothing read) and /resume
    with no store both error, leave the identity and history unchanged, and
    read no archive after the failure (no get_session at all)."""
    store = MagicMock(spec=InMemoryProvenanceStore)
    store.list_sessions = AsyncMock(return_value=[])
    store.get_session = AsyncMock(return_value=[])
    agent = _TakeoverAgent("miranda-repl-20261030-120000", store)
    agent.history = [_msg("user", "live")]
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface._cmd_resume(agent, "99")  # nothing listed: error, no read
        agent.provenance_store = None  # then the missing-store path
        await iface._cmd_resume(agent, "99")
        await iface._cmd_resume(agent, None)

    asyncio.run(scenario())

    assert agent.session_id == "miranda-repl-20261030-120000"
    assert list(agent.history) == [_msg("user", "live")]
    assert any("index 99" in ln for ln in lines)
    assert any("No provenance store" in ln for ln in lines)
    # One listing fetch (the index call, nothing listed yet); with an empty
    # store the index resolves to nothing and no archive read ever runs.
    store.list_sessions.assert_awaited_once()
    store.get_session.assert_not_awaited()


def test_takeover_restore_failure_switches_nothing(tmp_path, monkeypatch):
    """A restore failure (unreadable archive) must not adopt the key or touch
    the running conversation (FR 11: a key that cannot be read prints an
    error and does not switch)."""
    P = "miranda-repl-20261030-110000"
    store = MagicMock(spec=InMemoryProvenanceStore)
    store.list_sessions = AsyncMock(return_value=[SessionSummary(P, 1, BASE)])
    store.get_session = AsyncMock(side_effect=OSError("unreadable"))
    agent = _TakeoverAgent("miranda-default", store)
    agent.history = [_msg("user", "live")]
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface._cmd_resume(agent, "1")  # listed, but the read fails

    asyncio.run(scenario())

    assert agent.session_id == "miranda-default"
    assert list(agent.history) == [_msg("user", "live")]
    assert any("restore failed" in ln and P in ln for ln in lines)
    store.get_session.assert_awaited_once_with(P)


def test_takeover_unknown_key_switches_nothing(tmp_path, monkeypatch):
    """A key the store has never archived: an index outside the listing
    errors with nothing read (FR 11)."""
    P = "miranda-repl-20261030-110000"
    store = asyncio.run(_seeded_store([]))  # empty
    agent = _TakeoverAgent("miranda-default", store)
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface._cmd_resume(agent, "1")  # nothing listed

    asyncio.run(scenario())

    assert agent.session_id == "miranda-default"
    assert agent.history == []
    assert any("index 1" in ln for ln in lines)
    # No switch: the engine was never re-seeded.
    assert agent.sr2.seeded_seq == []


# ---------------------------------------------------------------------------
# AC13 — listing command
# ---------------------------------------------------------------------------


def test_ac13_c1_bare_resume_lists_15_newest_first_with_marker(tmp_path, monkeypatch):
    """AC13-c1: 16 archived sessions → 15 rows, newest first, in the
    <idx>  <YYYY-MM-DD HH:MM>  <turns>  <session-id> shape, current marked."""
    # One shared activity instant for all 16: list_sessions then orders
    # session id descending (its documented tie-break), independent of
    # the wall clock — expected rows are exact.
    fixtures = [(f"sess-{i:02d}", 1, BASE) for i in range(16)]
    store = asyncio.run(_seeded_store(fixtures))
    current = "sess-15"  # the newest session is the current key
    agent = _TakeoverAgent(current, store)
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface._cmd_resume(agent, None)

    asyncio.run(scenario())

    rows = _rows(lines)
    assert len(rows) == 15
    assert rows[0] == "1  " + _local(BASE) + "  1  sess-15  → current"
    assert rows[1] == "2  " + _local(BASE) + "  1  sess-14"
    assert rows[14] == "15  " + _local(BASE) + "  1  sess-01"
    # The 16th (oldest) session is not listed.
    assert all("sess-00" not in row for row in rows)
    # The listing cache is what the index form resolves against.
    assert [s.session_id for s in iface._listing] == [f"sess-{i:02d}" for i in range(15, 0, -1)]


def test_ac13_current_key_outside_top15_is_not_marked(tmp_path, monkeypatch):
    """The marker lands only on a row that is actually listed."""
    # One shared activity instant for all 16: list_sessions then orders
    # session id descending (its documented tie-break), independent of
    # the wall clock — expected rows are exact.
    fixtures = [(f"sess-{i:02d}", 1, BASE) for i in range(16)]
    store = asyncio.run(_seeded_store(fixtures))
    agent = _TakeoverAgent("sess-00", store)  # oldest: not in the top 15
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface._cmd_resume(agent, None)

    asyncio.run(scenario())
    assert "→ current" not in _plain(lines)


def test_ac13_c2_exact_session_id_works_without_prior_listing(tmp_path, monkeypatch):
    """AC13-c2: /resume <exact-key> with no bare listing beforehand still
    takes the key over (resolved as an exact id, not an index)."""
    P = "miranda-repl-20261030-110000"
    store = asyncio.run(_seeded_store([(P, 1, BASE)]))
    agent = _TakeoverAgent("miranda-default", store)
    iface, lines = _capturing_iface(tmp_path, monkeypatch)
    assert iface._listing == []  # no bare /resume has been issued

    async def scenario():
        await iface._cmd_resume(agent, P)

    asyncio.run(scenario())

    expected = asyncio.run(load_resumable_history(store, P, _config().agent.resume))
    assert agent.session_id == P
    assert list(agent.history) == expected.history
    assert any("Resumed" in ln and P in ln for ln in lines)


def test_ac13_out_of_range_index_errors_without_switching(tmp_path, monkeypatch):
    """/resume 99: out of range errors and does not switch, even with a
    prior listing."""
    P = "miranda-repl-20261030-110000"
    store = asyncio.run(_seeded_store([(P, 1, BASE)]))
    agent = _TakeoverAgent("miranda-default", store)
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface._cmd_resume(agent, None)  # prime the listing (one row)
        await iface._cmd_resume(agent, "99")  # out of range

    asyncio.run(scenario())

    assert agent.session_id == "miranda-default"
    assert agent.history == []
    assert any("index 99" in ln for ln in lines)
    assert not any("Resumed" in ln for ln in lines)
    assert agent.sr2.seeded_seq == []


# ---------------------------------------------------------------------------
# /resume joins _COMMANDS and _HELP
# ---------------------------------------------------------------------------


def test_resume_in_command_list_and_help():
    from sr2_spectre.interfaces.repl import _COMMANDS, _HELP

    assert "/resume" in _COMMANDS
    assert "/resume" in _HELP


# ---------------------------------------------------------------------------
# AC15 — replay of the restored conversation at takeover (increment 3)
# ---------------------------------------------------------------------------


def _replay_block(lines: list[str], report_index: int) -> list[str]:
    """The printed lines after the takeover line, trailing padding stripped."""
    return [ln.rstrip() for ln in lines[report_index + 1:]]


def test_ac15_c1_takeover_replays_history_in_order_after_the_takeover_line(
    tmp_path, monkeypatch
):
    """AC15-c1: history [u0, a0, u1, a1] replays as You:/u0, a0, You:/u1, a1,
    in order, strictly after the takeover line; nothing replays before it."""
    P = "miranda-repl-20261030-110000"
    store = asyncio.run(_seeded_store([(P, 2, BASE)]))  # turns 0,1 → u0/a0, u1/a1
    agent = _TakeoverAgent("miranda-default", store)
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface._cmd_resume(agent, P)

    asyncio.run(scenario())

    reports = [i for i, ln in enumerate(lines) if "Resumed" in ln]
    assert len(reports) == 1  # one takeover line; no replay above it
    report = reports[0]
    assert P in lines[report]
    assert _replay_block(lines, report) == ["You: u0", "a0", "You: u1", "a1"]


def test_ac15_c2_user_markup_escaped_verbatim_assistant_renders_markdown(
    tmp_path, monkeypatch
):
    """AC15-c2: user text ``[dim]literal[/dim] bold **text**`` prints with the
    literal bracket characters (markup escaped, never styled, no exception);
    the assistant message goes through the Markdown path."""
    P = "miranda-repl-20261030-110000"
    store = asyncio.run(_seeded_store([]))
    agent = _TakeoverAgent("miranda-default", store)
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await store.write_batch([
            _entry(P, 0, 0, _msg("user", "[dim]literal[/dim] bold **text**"), BASE),
            _entry(P, 0, 1, _msg("assistant", "**hello** world"), BASE),
        ])
        await iface._cmd_resume(agent, P)

    asyncio.run(scenario())  # a markup error here would escape as MarkupError

    report = next(ln for ln in lines if "Resumed" in ln)
    # Verbatim, brackets and all, in order — never styled on.
    assert "You: [dim]literal[/dim] bold **text**" in lines
    # Markdown rendering strips the emphasis markers: the source spelling is
    # gone, the rendered text is present.
    assert "**hello** world" not in _plain(lines)
    assert any(ln.startswith("hello world") for ln in _replay_block(lines, lines.index(report)))


def test_ac15_c3_interrupted_target_replays_marker_as_final_assistant_message(
    tmp_path, monkeypatch
):
    """AC15-c3: an interrupted target's replay ends with the marker line,
    after the last user message (no special-casing — it rides in history)."""
    P = "miranda-repl-20261030-110000"
    store = asyncio.run(_seeded_store([(P, 1, BASE)]))  # turn 0 complete
    agent = _TakeoverAgent("miranda-default", store)
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        # Turn 1 ends at a user message → the reconstruction appends the marker.
        await store.write_batch([
            _entry(P, 1, 0, _msg("user", "cut off here"), BASE + timedelta(minutes=120)),
        ])
        await iface._cmd_resume(agent, P)

    asyncio.run(scenario())

    report = next(ln for ln in lines if "Resumed" in ln)
    block = _replay_block(lines, lines.index(report))
    assert block == [
        "You: u0",
        "a0",
        "You: cut off here",
        MARKER,
    ]


def test_ac15_c4_failed_and_empty_takeovers_print_no_replay(tmp_path, monkeypatch):
    """AC15-c4: an unknown index, a restore-failed target and an empty
    archive each print only their existing line — no ``You:`` label, no
    history text."""
    P = "miranda-repl-20261030-110000"

    async def run_case(body):
        store = MagicMock(spec=InMemoryProvenanceStore)
        agent = _TakeoverAgent("miranda-default", store)
        agent.history = [_msg("user", "live")]
        iface, lines = _capturing_iface(tmp_path, monkeypatch)
        await body(store, agent, iface)
        return agent, lines

    # Unknown index: nothing listed, error line only, no archive read.
    async def unknown(store, agent, iface):
        store.list_sessions = AsyncMock(return_value=[])
        store.get_session = AsyncMock(return_value=[])
        await iface._cmd_resume(agent, "99")

    agent, lines = asyncio.run(run_case(unknown))
    assert any("index 99" in ln for ln in lines)
    assert not any("You:" in ln for ln in lines)
    assert agent.session_id == "miranda-default"

    # Restore failure: the archive read raises; only the error line appears,
    # and none of the target's content could leak into a replay.
    async def failed(store, agent, iface):
        store.list_sessions = AsyncMock(return_value=[SessionSummary(P, 1, BASE)])
        store.get_session = AsyncMock(side_effect=OSError("unreadable"))
        await iface._cmd_resume(agent, "1")

    agent, lines = asyncio.run(run_case(failed))
    assert any("restore failed" in ln and P in ln for ln in lines)
    assert not any("You:" in ln for ln in lines)
    assert agent.session_id == "miranda-default"
    assert list(agent.history) == [_msg("user", "live")]

    # Empty archive under an exact key: the "no archived conversation" line
    # only — and definitively no replay, since an empty history is the very
    # case a per-item loop vacuously satisfies.
    async def empty(store, agent, iface):
        store.list_sessions = AsyncMock(return_value=[])
        store.get_session = AsyncMock(return_value=[])
        await iface._cmd_resume(agent, P)

    agent, lines = asyncio.run(run_case(empty))
    assert any("No archived conversation" in ln and P in ln for ln in lines)
    assert lines == [ln for ln in lines if "No archived conversation" in ln]
    assert agent.session_id == "miranda-default"


def test_ac15_reset_does_not_replay(tmp_path, monkeypatch):
    """/reset starts fresh: the notice prints, no ``You:`` label or replay."""
    agent = Agent(_config())
    agent.history[:] = [_msg("user", "old"), _msg("assistant", "prior")]
    iface, lines = _capturing_iface(tmp_path, monkeypatch)

    async def scenario():
        await iface._cmd_reset(agent)

    asyncio.run(scenario())
    assert any("New session started" in ln for ln in lines)
    assert not any("You:" in ln for ln in lines)
    assert not any("old" in ln or "prior" in ln for ln in lines)


def test_render_replay_items_pure_shape():
    """The helper is pure and item-per-message: a user message yields one
    Text whose plain text is the ``You:`` label plus the verbatim body (its
    bracket markup survives as characters, no span styled it on), an
    assistant message yields a Markdown renderable of its text."""
    from rich.markdown import Markdown
    from rich.text import Text as RichText

    items = render_replay_items([
        _msg("user", "[dim]literal[/dim] bold **text**"),
        _msg("assistant", "# Head"),
    ])
    assert len(items) == 2
    user_item, assistant_item = items
    assert isinstance(user_item, RichText)
    # Pinned in full: the exact bracket characters, in order, unescaped.
    assert user_item.plain == "You: [dim]literal[/dim] bold **text**"
    assert "[dim]" in user_item.plain and "[/dim]" in user_item.plain
    # No span applied a dim style to the bracketed fragment — it is literal.
    assert all("dim" not in str(span.style) for span in user_item._spans)
    assert isinstance(assistant_item, Markdown)
    assert render_replay_items([]) == []

