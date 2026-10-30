"""Archive reconstruction, interrupted turns and durable rewind ranges."""

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.provenance import (
    ContextLifecycle, ContextScope, Entry, EntryOrigin, InMemoryProvenanceStore,
)
from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore
from sr2.protocols.llm import StreamEvent

from sr2_spectre.config import AgentConfig, ResumeConfig, SpectreConfig
from sr2_spectre.resume import ResumeResult, load_resumable_history, write_rewind
from sr2_spectre.session import Session
from sr2_spectre.tools.registry import ToolRegistry


def message(role, text):
    return Message(role=role, content=[TextBlock(text=text)])


def archive(turn, seq, content, session_id="s"):
    return Entry(
        id=f"{session_id}-{turn}-{seq}", content=content, sources=(),
        origin=EntryOrigin(kind="resolver", name="turn_archivist"),
        layer="turn_archive", session_id=session_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        lifecycle=ContextLifecycle.ARCHIVE, scope=ContextScope(owner_id=session_id),
        meta={"turn_seq": turn, "seq": seq},
    )


TOOL = ToolUseBlock(id="call", name="example", input={})
RESULT = ToolResultBlock(tool_use_id="call", content="tool output")


def completed_archive():
    return [
        archive(0, 0, message("user", "u0")),
        archive(0, 1, message("assistant", "A")),
        archive(0, 2, TOOL), archive(0, 3, RESULT),
        archive(0, 4, message("assistant", "B")),
        archive(1, 0, message("user", "u1")),
        archive(1, 1, message("assistant", "C")),
    ]


@pytest.mark.parametrize("shuffled", [False, True], ids=["AC2-c1", "AC2-c3"])
async def test_completed_history_matches_session(shuffled):
    entries = completed_archive()
    if shuffled:
        entries = [entries[i] for i in [6, 2, 4, 0, 5, 3, 1]]
    store = InMemoryProvenanceStore()
    await store.write_batch(entries)
    await store.write(archive(99, 0, message("user", "other"), "other-session"))
    await store.write(replace(archive(99, 1, message("user", "unrelated")), layer="session"))
    store.get_session = AsyncMock(wraps=store.get_session)
    store.write = AsyncMock(wraps=store.write)
    result = await load_resumable_history(store, "s", ResumeConfig())
    expected = [message("user", "u0"), message("assistant", "AB"),
                message("user", "u1"), message("assistant", "C")]
    assert result == ResumeResult(expected, interrupted=False, restore_failed=False)
    store.get_session.assert_awaited_once_with("s")
    store.write.assert_not_awaited()

    # Independently run Session's accumulation of the same text/tool events.
    async def turn(user_input):
        if user_input[0].text == "u0":
            yield StreamEvent(type="text", text="A")
            yield StreamEvent(type="tool_use_emitted", tool_uses=[TOOL])
            yield StreamEvent(type="tool_result_received", tool_results=[RESULT])
            yield StreamEvent(type="text", text="B")
        else:
            yield StreamEvent(type="text", text="C")
    engine = MagicMock()
    engine.continue_archive = AsyncMock()
    engine.turn = turn
    config = SpectreConfig(agent={}, models={"default": {"model": "test"}}, pipeline={"layers": []})
    with patch("sr2_spectre.session.SR2", return_value=engine):
        session = Session("s", config, MagicMock(), ToolRegistry())
    await session.handle_user_message("u0")
    await session.handle_user_message("u1")
    assert result.history == session.history == expected


async def test_empty_archive():
    """AC2-c2: a new session is not a failed restore."""
    result = await load_resumable_history(InMemoryProvenanceStore(), "s", ResumeConfig())
    assert result == ResumeResult([], interrupted=False, restore_failed=False)


async def test_nonlast_turn_without_assistant_has_empty_content():
    """AC2-c4: only the highest archived turn can be marked interrupted."""
    store = InMemoryProvenanceStore()
    await store.write_batch([
        archive(0, 0, message("user", "silent")),
        archive(1, 0, message("user", "next")),
        archive(1, 1, message("assistant", "done")),
    ])
    result = await load_resumable_history(store, "s", ResumeConfig())
    assert result == ResumeResult([
        message("user", "silent"), Message(role="assistant", content=[]),
        message("user", "next"), message("assistant", "done"),
    ], interrupted=False, restore_failed=False)


@pytest.mark.parametrize("tail", [[], [message("assistant", "partial"), TOOL],
                                  [message("assistant", "partial"), TOOL, RESULT]],
                         ids=["AC3-c1", "AC3-c2", "AC3-c3"])
async def test_interrupted_tail_uses_configured_marker_without_execution(tail):
    entries = [archive(7, 0, message("user", "request"))]
    entries.extend(archive(7, seq, content) for seq, content in enumerate(tail, 1))
    # The loader has only the store capability: any write or execution would fail.
    store = MagicMock(spec=["get_session"], get_session=AsyncMock(return_value=list(reversed(entries))))
    result = await load_resumable_history(store, "s", ResumeConfig(interrupted_marker="cut off"))
    assert result == ResumeResult([message("user", "request"), message("assistant", "cut off")],
                                 interrupted=True, restore_failed=False)
    store.get_session.assert_awaited_once_with("s")


@pytest.mark.parametrize("turn_count,bounds,kept", [
    (4, [(2, 3)], [0, 1, 2]),
    (5, [(-1, 2)], [3, 4]),
    (4, [(-1, 1), (2, 3)], [2]),
    (5, [(2, 3)], [0, 1, 2, 4]),
    (5, [(-1, 4)], []),
], ids=["AC4-c1", "AC4-c2", "AC4-c3", "later-turn", "full-reset"])
async def test_rewind_ranges_survive_reopen_and_keep_later_turns(tmp_path, turn_count, bounds, kept):
    path = str(tmp_path / "provenance.db")
    store = SQLiteProvenanceStore(db_path=path)
    await store.connect()
    try:
        for turn in range(turn_count):
            await store.write(archive(turn, 0, message("user", f"u{turn}")))
            await store.write(archive(turn, 1, message("assistant", f"a{turn}")))
        for keep, through in bounds:
            await write_rewind(store, "s", keep, through)
        await write_rewind(store, "other-session", -1, 99)
    finally:
        await store.close()
    reopened = SQLiteProvenanceStore(db_path=path)
    await reopened.connect()
    try:
        result = await load_resumable_history(reopened, "s", ResumeConfig())
        expected = [item for turn in kept for item in
                    [message("user", f"u{turn}"), message("assistant", f"a{turn}")]]
        assert result == ResumeResult(expected, interrupted=False, restore_failed=False)
        assert len(await reopened.get_session("s", layer="turn_archive")) == turn_count * 2
    finally:
        await reopened.close()


@pytest.mark.parametrize("keep,action", [(1, "retry"), (-1, "reset")])
async def test_write_rewind_metadata_scope_and_append_only(keep, action):
    """AC4-c4: bounds, action and owner survive as distinct archive entries."""
    store = InMemoryProvenanceStore()
    await write_rewind(store, "s", keep, 3)
    entries = await store.get_session("s")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.layer == "session_rewind"
    assert entry.session_id == "s"
    assert entry.meta == {"keep_through_turn_seq": keep, "through_turn_seq": 3}
    assert entry.content == TextBlock(text=action)
    assert entry.lifecycle == ContextLifecycle.ARCHIVE
    assert entry.scope == ContextScope(owner_id="s")
    await write_rewind(store, "s", keep, 3)
    assert len(await store.get_session("s")) == 2
    assert await store.get(entry.id) == entry


async def test_rewound_interrupted_tail_does_not_mark_previous_turn():
    store = InMemoryProvenanceStore()
    await store.write_batch([archive(0, 0, message("user", "silent")),
                             archive(1, 0, message("user", "interrupted"))])
    await write_rewind(store, "s", 0, 1)
    result = await load_resumable_history(store, "s", ResumeConfig())
    assert result == ResumeResult([message("user", "silent"), Message(role="assistant", content=[])],
                                 interrupted=False, restore_failed=False)


@pytest.mark.parametrize("absent", [False, True])
async def test_restore_failure_returns_empty_history_and_logs_session(absent, caplog):
    store = None if absent else MagicMock(spec=["get_session"], get_session=AsyncMock())
    if store is not None:
        store.get_session.side_effect = OSError("unreadable")
    result = await load_resumable_history(store, "failed-session", ResumeConfig())
    assert result == ResumeResult([], interrupted=False, restore_failed=True)
    assert "failed-session" in caplog.text


@pytest.mark.parametrize("absent", [False, True])
async def test_disabled_resume_does_not_read_even_rewinds(absent):
    store = None if absent else MagicMock(spec=["get_session"], get_session=AsyncMock())
    result = await load_resumable_history(store, "s", ResumeConfig(enabled=False))
    assert result == ResumeResult([], interrupted=False, restore_failed=False)
    if store is not None:
        store.get_session.assert_not_awaited()


async def test_malformed_archive_metadata_fails_without_partial_history(caplog):
    store = InMemoryProvenanceStore()
    await store.write_batch(completed_archive())
    await store.write(replace(archive(9, 0, message("user", "bad")), meta={"turn_seq": True}))
    result = await load_resumable_history(store, "s", ResumeConfig())
    assert result == ResumeResult([], interrupted=False, restore_failed=True)
    assert "session=s" in caplog.text


def test_resume_config_defaults_and_overrides():
    defaults = AgentConfig().resume
    assert defaults.enabled is True
    assert defaults.interrupted_marker == "[This turn was interrupted by a restart before it finished.]"
    assert defaults.restore_failed_notice == "Earlier context for this conversation could not be restored."
    config = SpectreConfig.model_validate({
        "agent": {"resume": {"enabled": False, "interrupted_marker": "interrupted",
                             "restore_failed_notice": "unavailable"}},
        "models": {"default": {"model": "test"}}, "pipeline": {"layers": []},
    })
    assert config.agent.resume == ResumeConfig(enabled=False, interrupted_marker="interrupted",
                                               restore_failed_notice="unavailable")
