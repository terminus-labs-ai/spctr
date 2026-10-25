"""Agent payload: dispatch, post result, failure line (spc-187).

AC32–AC35. Injected clock, fake agent runner (returns canned envelopes),
fake send seam. No real dispatch, no network.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from sr2_spectre.schedules import ScheduleStore, Schedule, ScheduleEngine
from sr2_spectre.schedules.agent import AgentResult, split_message
from sr2_spectre.tools.schedule_tool import (
    AGENT_INSTRUCTION_MAX,
    ScheduleOrigin,
    ScheduleTool,
    ScheduleTurnScope,
    schedule_turn_scope,
)

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)


def _make_schedule(**kwargs) -> Schedule:
    defaults = dict(
        id="s0",
        agent="miranda",
        platform="discord",
        requester_id="42",
        requester_mention="<@42>",
        dest_channel_id="100",
        dest_channel_name="general",
        label="test",
        cron="0 7 * * *",
        payload_type="command",
        payload=json.dumps({"command": "echo hi", "cwd": None}),
        on_miss="run_late",
        next_run_at=NOW + timedelta(hours=1),
        last_run_at=None,
        created_at=NOW,
        status="active",
        last_status=None,
        last_error=None,
    )
    defaults.update(kwargs)
    return Schedule(**defaults)


def _agent_payload(agent: str = "miranda", instruction: str = "Summarize beads") -> str:
    return json.dumps({"agent": agent, "instruction": instruction})


# ---------------------------------------------------------------------------
# AC32: create with agent payload stores instruction and target agent
# ---------------------------------------------------------------------------

class TestAC32CreateAgentPayload:
    """AC32: create with an agent payload stores the instruction and the target agent (default: the creating agent)."""

    @pytest.fixture
    def tool_and_store(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        tool = ScheduleTool(
            store=store,
            timezone="America/Phoenix",
            agent_name="miranda",
            origin_getter=lambda: None,
            clock=lambda: NOW,
        )
        return tool, store

    @pytest.mark.asyncio
    async def test_agent_payload_stores_instruction_and_agent(self, tool_and_store):
        tool, store = tool_and_store
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(
                action="create",
                cron="0 7 * * *",
                type="agent",
                instruction="Summarize the beads closed this week",
            )
            assert "Schedule s1 set" in result
            row = store.get("s1")
            assert row is not None
            assert row.payload_type == "agent"
            payload = json.loads(row.payload)
            assert payload["instruction"] == "Summarize the beads closed this week"
            # Default agent is the creating agent
            assert payload["agent"] == "miranda"
        finally:
            schedule_turn_scope.set(None)

    @pytest.mark.asyncio
    async def test_agent_payload_explicit_target_agent(self, tool_and_store):
        tool, store = tool_and_store
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(
                action="create",
                cron="0 9 * * 1",
                type="agent",
                instruction="Do the thing",
                agent="edi",
            )
            assert "Schedule s1 set" in result
            row = store.get("s1")
            payload = json.loads(row.payload)
            assert payload["agent"] == "edi"
            assert payload["instruction"] == "Do the thing"
        finally:
            schedule_turn_scope.set(None)

    @pytest.mark.asyncio
    async def test_agent_payload_requires_instruction(self, tool_and_store):
        tool, store = tool_and_store
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(
                action="create", cron="0 7 * * *", type="agent", instruction=""
            )
            assert "instruction" in result.lower()
            rows = store.list_for_agent("miranda")
            assert len(rows) == 0
        finally:
            schedule_turn_scope.set(None)

    @pytest.mark.asyncio
    async def test_agent_payload_instruction_too_long(self, tool_and_store):
        tool, store = tool_and_store
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            long_instruction = "x" * (AGENT_INSTRUCTION_MAX + 1)
            result = await tool(
                action="create", cron="0 7 * * *", type="agent", instruction=long_instruction
            )
            assert "exceeds" in result.lower()
            rows = store.list_for_agent("miranda")
            assert len(rows) == 0
        finally:
            schedule_turn_scope.set(None)

    @pytest.mark.asyncio
    async def test_agent_payload_missing_instruction(self, tool_and_store):
        tool, store = tool_and_store
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(
                action="create", cron="0 7 * * *", type="agent"
            )
            assert "instruction" in result.lower()
            rows = store.list_for_agent("miranda")
            assert len(rows) == 0
        finally:
            schedule_turn_scope.set(None)


# ---------------------------------------------------------------------------
# split_message unit tests
# ---------------------------------------------------------------------------

class TestSplitMessage:
    def test_short_text_single_chunk(self):
        assert split_message("hello") == ["hello"]

    def test_exact_limit(self):
        text = "x" * 2000
        chunks = split_message(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_at_newline(self):
        # Build text with a newline well before the limit
        line1 = "a" * 500
        line2 = "b" * 1600
        text = line1 + "\n" + line2  # 500 + 1 + 1600 = 2101 > 2000
        chunks = split_message(text)
        assert len(chunks) == 2
        # First chunk ends at the newline (includes it)
        assert chunks[0] == line1 + "\n"
        assert chunks[1] == line2

    def test_splits_at_last_newline_before_limit(self):
        # Multiple newlines; picks the last one at or before 2000
        text = ("a" * 900 + "\n") + ("b" * 900 + "\n") + ("c" * 300)
        # Total: 901 + 901 + 300 = 2102 > 2000
        chunks = split_message(text)
        assert len(chunks) == 2
        # Last newline at or before 2000 is at position 1801 (after second \n)
        assert chunks[0].endswith("\n")
        assert len(chunks[0]) <= 2000

    def test_force_split_no_newlines(self):
        text = "x" * 3000
        chunks = split_message(text)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2000
        assert len(chunks[1]) == 1000

    def test_empty_string(self):
        assert split_message("") == []

    def test_multiple_splits(self):
        # Build 5000 chars with newlines every 400 chars
        text = "\n".join(["x" * 399 for _ in range(13)])  # 13*400 - 1 = 5199 chars
        chunks = split_message(text)
        assert all(len(c) <= 2000 for c in chunks)
        assert "".join(chunks) == text


# ---------------------------------------------------------------------------
# AC33: Due agent schedule runs dispatch, posts final message verbatim,
#       split at 2000 chars; only done posts content
# ---------------------------------------------------------------------------

class TestAC33AgentFiresAndPosts:
    @pytest.mark.asyncio
    async def test_done_posts_message_verbatim(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            return AgentResult(
                status="done",
                message="Here is your summary of beads closed this week.",
                exit_code=0,
                duration_s=5.2,
                log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload_type="agent",
            payload=_agent_payload(instruction="Summarize beads closed"),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )
        await engine._poll_once()

        # Posted the message verbatim
        assert len(sent) == 1
        assert sent[0][0] == "100"  # dest_channel_id
        assert sent[0][1] == "Here is your summary of beads closed this week."
        # Schedule advanced
        row = store.get("s1")
        assert row.next_run_at > NOW
        assert row.last_status == "ok"

    @pytest.mark.asyncio
    async def test_done_splits_at_2000(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        long_message = "line\n" * 500  # 2500 chars
        assert len(long_message) > 2000

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            return AgentResult(
                status="done",
                message=long_message,
                exit_code=0,
                duration_s=10.0,
                log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload_type="agent",
            payload=_agent_payload(),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )
        await engine._poll_once()

        # Multiple chunks posted
        assert len(sent) > 1
        # Each chunk at most 2000
        for _, content in sent:
            assert len(content) <= 2000
        # Reassembled equals original
        assert "".join(c for _, c in sent) == long_message

    @pytest.mark.asyncio
    async def test_done_empty_message_posts_nothing(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            return AgentResult(
                status="done", message="", exit_code=0,
                duration_s=1.0, log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload_type="agent",
            payload=_agent_payload(),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )
        await engine._poll_once()

        # Empty message posts nothing
        assert len(sent) == 0
        row = store.get("s1")
        assert row.last_status == "ok"

    @pytest.mark.asyncio
    async def test_runner_receives_correct_args(self, tmp_path):
        """The runner gets the target agent, instruction, and timeout."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []
        captured_args: dict[str, Any] = {}

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            captured_args["agent_name"] = agent_name
            captured_args["instruction"] = instruction
            captured_args.update(kw)
            return AgentResult(
                status="done", message="ok", exit_code=0,
                duration_s=1.0, log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload_type="agent",
            payload=_agent_payload(agent="edi", instruction="Run diagnostics"),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            agent_timeout=300.0,
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )
        await engine._poll_once()

        assert captured_args["agent_name"] == "edi"
        assert captured_args["instruction"] == "Run diagnostics"
        assert captured_args["timeout"] == 300.0


# ---------------------------------------------------------------------------
# AC34: Non-done status posts ⚠️ with status; schedule stays active
# ---------------------------------------------------------------------------

class TestAC34AgentFailurePostsWarning:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["failed", "needs-input", "escalation", "timeout", "error"])
    async def test_non_done_posts_warning(self, tmp_path, status):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            return AgentResult(
                status=status,
                message="partial output" if status != "timeout" else "",
                exit_code=1,
                duration_s=2.0,
                log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload_type="agent",
            payload=_agent_payload(),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )
        await engine._poll_once()

        # Posted one ⚠️ message
        assert len(sent) == 1
        assert "⚠️" in sent[0][1]
        assert status in sent[0][1]

        # Schedule stays active with next_run advanced
        row = store.get("s1")
        assert row.status == "active"
        assert row.next_run_at > NOW
        assert row.last_status == "failed"

    @pytest.mark.asyncio
    async def test_warning_includes_schedule_id_and_label(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            return AgentResult(
                status="failed", message="", exit_code=1,
                duration_s=1.0, log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload_type="agent",
            payload=_agent_payload(),
            label="weekly summary",
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )
        await engine._poll_once()

        assert len(sent) == 1
        assert "s1" in sent[0][1]
        assert "weekly summary" in sent[0][1]

    @pytest.mark.asyncio
    async def test_non_done_does_not_post_message_content(self, tmp_path):
        """Only status=done posts the message field; other statuses post only the ⚠️ line."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            return AgentResult(
                status="failed",
                message="This should NOT appear in the channel",
                exit_code=1,
                duration_s=1.0,
                log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload_type="agent",
            payload=_agent_payload(),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )
        await engine._poll_once()

        assert len(sent) == 1
        assert "This should NOT appear" not in sent[0][1]


# ---------------------------------------------------------------------------
# AC35: run_now on an agent payload posts the result to the destination
# ---------------------------------------------------------------------------

class TestAC35RunNowAgent:
    @pytest.mark.asyncio
    async def test_run_now_posts_done_message(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            return AgentResult(
                status="done",
                message="Manual run result here",
                exit_code=0,
                duration_s=3.0,
                log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW + timedelta(days=1),  # future — not due
            payload_type="agent",
            payload=_agent_payload(instruction="Do manual thing"),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )

        row = store.get("s1")
        await engine.execute_now(row)

        # Posted the result
        assert len(sent) == 1
        assert sent[0][1] == "Manual run result here"
        # Schedule advanced (next cron after now)
        updated = store.get("s1")
        assert updated.next_run_at > NOW
        assert updated.last_status == "ok"

    @pytest.mark.asyncio
    async def test_run_now_posts_failure_warning(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            return AgentResult(
                status="timeout", message="", exit_code=None,
                duration_s=600.0, log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW + timedelta(days=1),
            payload_type="agent",
            payload=_agent_payload(),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )

        row = store.get("s1")
        await engine.execute_now(row)

        # Posted ⚠️ warning
        assert len(sent) == 1
        assert "⚠️" in sent[0][1]
        assert "timeout" in sent[0][1]
        # Schedule stays active
        updated = store.get("s1")
        assert updated.status == "active"
        assert updated.next_run_at > NOW

    @pytest.mark.asyncio
    async def test_run_now_via_tool(self, tmp_path):
        """run_now through the tool dispatches to engine.execute_now."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        async def fake_runner(agent_name: str, instruction: str, **kw) -> AgentResult:
            return AgentResult(
                status="done", message="tool run result", exit_code=0,
                duration_s=1.0, log_path="/tmp/test.log",
            )

        sched = _make_schedule(
            next_run_at=NOW + timedelta(days=1),
            payload_type="agent",
            payload=_agent_payload(),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
            agent_runner=fake_runner,
        )

        tool = ScheduleTool(
            store=store,
            timezone="America/Phoenix",
            agent_name="miranda",
            origin_getter=lambda: None,
            clock=lambda: NOW,
            engine=engine,
        )
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(action="run_now", id="s1")
            assert "executed" in result.lower()
        finally:
            schedule_turn_scope.set(None)

        assert len(sent) == 1
        assert sent[0][1] == "tool run result"


# ---------------------------------------------------------------------------
# AgentResult property tests
# ---------------------------------------------------------------------------

class TestAgentResult:
    def test_done_is_success(self):
        r = AgentResult(status="done", message="hi", exit_code=0, duration_s=1.0, log_path="")
        assert r.success

    def test_failed_is_not_success(self):
        r = AgentResult(status="failed", message="", exit_code=1, duration_s=1.0, log_path="")
        assert not r.success

    def test_timeout_error_line(self):
        r = AgentResult(status="timeout", message="", exit_code=None, duration_s=600.0, log_path="")
        assert "timeout" in r.error_line

    def test_error_line_contains_status(self):
        r = AgentResult(status="escalation", message="", exit_code=3, duration_s=5.0, log_path="")
        assert "escalation" in r.error_line
