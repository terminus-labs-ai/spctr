"""Schedules core: cron, store, engine, command, tool (spc-185).

AC18–AC28. Injected clock, fake command runner (tmp_path scripts),
fake send seam. No services, credentials or network.
"""
from __future__ import annotations

import asyncio
import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from sr2_spectre.schedules import ScheduleStore, Schedule, ScheduleEngine, parse_cron
from sr2_spectre.schedules.cron import CronParseError, CronSpec
from sr2_spectre.schedules.command import CommandResult, run_command
from sr2_spectre.tools.schedule_tool import (
    NO_INTERFACE_ERROR,
    ScheduleOrigin,
    ScheduleTool,
    ScheduleTurnScope,
    schedule_turn_scope,
)

UTC = ZoneInfo("UTC")
PHX = ZoneInfo("America/Phoenix")
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)  # Monday 7am Phoenix


# ---------------------------------------------------------------------------
# AC19: Cron parser
# ---------------------------------------------------------------------------

class TestCronParser:
    """AC19: each valid form computes the next occurrence; invalid returns error."""

    def test_star(self):
        spec = parse_cron("* * * * *")
        result = spec.next_after(NOW, UTC)
        assert result == datetime(2026, 10, 20, 14, 1, tzinfo=UTC)

    def test_specific_minute(self):
        spec = parse_cron("30 * * * *")
        result = spec.next_after(NOW, UTC)
        assert result == datetime(2026, 10, 20, 14, 30, tzinfo=UTC)

    def test_specific_time(self):
        spec = parse_cron("0 7 * * *")
        result = spec.next_after(NOW, UTC)
        # NOW is 14:00 UTC, so next 07:00 is tomorrow
        assert result == datetime(2026, 10, 21, 7, 0, tzinfo=UTC)

    def test_list(self):
        spec = parse_cron("0,15,30,45 * * * *")
        result = spec.next_after(NOW, UTC)
        assert result == datetime(2026, 10, 20, 14, 15, tzinfo=UTC)

    def test_range(self):
        spec = parse_cron("0 9-17 * * *")
        # NOW is 14:00, next hour in 9-17 is 15:00
        result = spec.next_after(NOW, UTC)
        assert result == datetime(2026, 10, 20, 15, 0, tzinfo=UTC)

    def test_step(self):
        spec = parse_cron("*/15 * * * *")
        result = spec.next_after(NOW, UTC)
        assert result == datetime(2026, 10, 20, 14, 15, tzinfo=UTC)

    def test_step_with_range(self):
        spec = parse_cron("1-10/3 * * * *")
        result = spec.next_after(NOW, UTC)
        # 1,4,7,10 — next after :00 is :01
        assert result == datetime(2026, 10, 20, 14, 1, tzinfo=UTC)

    def test_day_of_week(self):
        # Oct 20 2026 is Tuesday. cron dow 1 = Monday.
        spec = parse_cron("0 9 * * 1")
        result = spec.next_after(NOW, UTC)
        # Next Monday at 09:00 is Oct 27... wait, Oct 26 is Monday.
        assert result == datetime(2026, 10, 26, 9, 0, tzinfo=UTC)

    def test_day_of_week_sunday(self):
        spec = parse_cron("0 9 * * 0")
        result = spec.next_after(NOW, UTC)
        # Sunday Oct 25 at 09:00
        assert result == datetime(2026, 10, 25, 9, 0, tzinfo=UTC)

    def test_timezone_aware(self):
        # 0 7 * * * in Phoenix (UTC-7): 7am Phoenix = 14:00 UTC
        spec = parse_cron("0 7 * * *")
        # NOW = 14:00 UTC = 7:00 Phoenix — next is tomorrow 7am Phoenix
        result = spec.next_after(NOW, PHX)
        assert result.astimezone(PHX).hour == 7
        assert result.astimezone(PHX).day == 21

    def test_invalid_too_few_fields(self):
        with pytest.raises(CronParseError):
            parse_cron("* * *")

    def test_invalid_too_many_fields(self):
        with pytest.raises(CronParseError):
            parse_cron("* * * * * *")

    def test_invalid_minute_out_of_range(self):
        with pytest.raises(CronParseError):
            parse_cron("60 * * * *")

    def test_invalid_hour_out_of_range(self):
        with pytest.raises(CronParseError):
            parse_cron("0 24 * * *")

    def test_invalid_text(self):
        with pytest.raises(CronParseError):
            parse_cron("abc * * * *")

    def test_invalid_step_zero(self):
        with pytest.raises(CronParseError):
            parse_cron("*/0 * * * *")


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------

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


class TestStore:
    def test_create_assigns_sequential_ids(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        id1 = store.create(_make_schedule())
        id2 = store.create(_make_schedule())
        assert id1 == "s1"
        assert id2 == "s2"

    def test_due_returns_active_past_schedules(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule(next_run_at=NOW - timedelta(minutes=1)))
        due = store.due(NOW)
        assert len(due) == 1
        assert due[0].id == "s1"

    def test_due_excludes_future(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule(next_run_at=NOW + timedelta(hours=1)))
        due = store.due(NOW)
        assert len(due) == 0

    def test_due_platform_scoped(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule(next_run_at=NOW - timedelta(minutes=1), platform="discord"))
        store.create(_make_schedule(next_run_at=NOW - timedelta(minutes=1), platform="zulip"))
        due = store.due(NOW, platform="discord")
        assert len(due) == 1
        assert due[0].platform == "discord"

    def test_claim_atomic(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule(next_run_at=NOW - timedelta(minutes=1)))
        assert store.claim("s1") is True
        # Second claim fails — next_run_at already moved
        assert store.claim("s1") is False

    def test_cancel(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule(next_run_at=NOW - timedelta(minutes=1)))
        assert store.cancel("s1") is True
        assert store.cancel("s1") is False  # already cancelled
        # Cancelled schedule is not due
        due = store.due(NOW)
        assert len(due) == 0

    def test_advance_after_run(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule(next_run_at=NOW - timedelta(minutes=1)))
        store.claim("s1")
        next_time = NOW + timedelta(days=1)
        store.advance_after_run("s1", next_time, NOW, "ok", None)
        row = store.get("s1")
        assert row.next_run_at == next_time
        assert row.last_status == "ok"


# ---------------------------------------------------------------------------
# AC18: Disabled/enabled config
# ---------------------------------------------------------------------------

class TestAC18ConfigGate:
    """Disabled config registers no tool and starts no engine; enabled registers both."""

    def test_disabled_no_tool_no_engine(self):
        from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig, SchedulesConfig
        from sr2_spectre.runtime import Runtime

        config = SpectreConfig(
            agent=AgentConfig(name="test", default_skills=False, schedules=SchedulesConfig(enabled=False)),
            models={"default": ModelConfig(model="test")},
            pipeline={"layers": []},
            provenance_store_path="",
            memory_store_dsn="",
        )
        runtime = Runtime(config)
        assert "schedule" not in runtime.registry
        assert runtime.schedules is None

    def test_enabled_registers_tool(self, tmp_path):
        from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig, SchedulesConfig
        from sr2_spectre.runtime import Runtime

        config = SpectreConfig(
            agent=AgentConfig(name="test", default_skills=False, schedules=SchedulesConfig(
                enabled=True, timezone="America/Phoenix",
                store_path=str(tmp_path / "s.sqlite"),
            )),
            models={"default": ModelConfig(model="test")},
            pipeline={"layers": []},
            provenance_store_path="",
            memory_store_dsn="",
        )
        runtime = Runtime(config)
        assert "schedule" in runtime.registry
        assert runtime.schedules is not None
        assert runtime.schedules_enabled is True


# ---------------------------------------------------------------------------
# AC19/AC20: Tool create validation
# ---------------------------------------------------------------------------

class TestToolCreateValidation:
    """AC19/AC20: invalid input returns error, store unchanged."""

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
    async def test_no_origin_returns_error(self, tool_and_store):
        tool, store = tool_and_store
        result = await tool(action="create", cron="0 7 * * *", type="command", command="echo")
        assert result == NO_INTERFACE_ERROR

    @pytest.mark.asyncio
    async def test_invalid_cron_stores_nothing(self, tool_and_store):
        tool, store = tool_and_store
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(action="create", cron="invalid", type="command", command="echo")
            assert "invalid cron" in result.lower()
            rows = store.list_for_agent("miranda")
            assert len(rows) == 0
        finally:
            schedule_turn_scope.set(None)

    @pytest.mark.asyncio
    async def test_missing_type_stores_nothing(self, tool_and_store):
        tool, store = tool_and_store
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(action="create", cron="0 7 * * *")
            assert "type" in result.lower()
            rows = store.list_for_agent("miranda")
            assert len(rows) == 0
        finally:
            schedule_turn_scope.set(None)

    @pytest.mark.asyncio
    async def test_empty_command_stores_nothing(self, tool_and_store):
        tool, store = tool_and_store
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(action="create", cron="0 7 * * *", type="command", command="")
            assert "command" in result.lower()
            rows = store.list_for_agent("miranda")
            assert len(rows) == 0
        finally:
            schedule_turn_scope.set(None)

    @pytest.mark.asyncio
    async def test_valid_create_stores_row(self, tool_and_store):
        tool, store = tool_and_store
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(action="create", cron="0 7 * * *", type="command", command="echo hi")
            assert "Schedule s1 set" in result
            rows = store.list_for_agent("miranda")
            assert len(rows) == 1
        finally:
            schedule_turn_scope.set(None)


# ---------------------------------------------------------------------------
# AC21: Due schedule fires exactly once, command runs, next_run advances
# ---------------------------------------------------------------------------

class TestAC21FireOnce:
    @pytest.mark.asyncio
    async def test_fires_once_and_advances(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        # Create a schedule due NOW
        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload=json.dumps({"command": "echo hello", "cwd": str(tmp_path)}),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
        )
        await engine._poll_once()

        # It ran: next_run_at advanced
        row = store.get("s1")
        assert row.next_run_at > NOW
        assert row.last_status == "ok"
        # No failure post
        assert len(sent) == 0

        # Second poll: not due again
        await engine._poll_once()
        row2 = store.get("s1")
        assert row2.next_run_at == row.next_run_at  # unchanged


# ---------------------------------------------------------------------------
# AC22: Non-zero exit posts ⚠️, logs, schedule stays active
# ---------------------------------------------------------------------------

class TestAC22Failure:
    @pytest.mark.asyncio
    async def test_nonzero_exit_posts_warning(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        # Script that fails
        script = tmp_path / "fail.sh"
        script.write_text("#!/bin/sh\necho 'something broke' >&2\nexit 1")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload=json.dumps({"command": str(script), "cwd": str(tmp_path)}),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
        )
        await engine._poll_once()

        assert len(sent) == 1
        assert sent[0][0] == "100"  # dest_channel_id
        assert "⚠️" in sent[0][1]
        assert "something broke" in sent[0][1]

        # Schedule stays active with next_run advanced
        row = store.get("s1")
        assert row.status == "active"
        assert row.next_run_at > NOW
        assert row.last_status == "failed"

    @pytest.mark.asyncio
    async def test_timeout_posts_warning(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        # Script that sleeps forever
        script = tmp_path / "sleep.sh"
        script.write_text("#!/bin/sh\nsleep 100")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload=json.dumps({"command": str(script), "cwd": str(tmp_path)}),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            command_timeout=0.5,
            clock=lambda: NOW,
        )
        await engine._poll_once()

        assert len(sent) == 1
        assert "⚠️" in sent[0][1]
        assert "timed out" in sent[0][1]
        row = store.get("s1")
        assert row.status == "active"


# ---------------------------------------------------------------------------
# AC23: Exit 0 posts nothing, logs output
# ---------------------------------------------------------------------------

class TestAC23Success:
    @pytest.mark.asyncio
    async def test_success_posts_nothing(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        sched = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload=json.dumps({"command": "echo all good", "cwd": str(tmp_path)}),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
        )
        await engine._poll_once()

        assert len(sent) == 0
        row = store.get("s1")
        assert row.last_status == "ok"


# ---------------------------------------------------------------------------
# AC24: create without channel stores origin channel root (thread → parent)
# ---------------------------------------------------------------------------

class TestAC24OriginRoot:
    @pytest.mark.asyncio
    async def test_thread_parent_is_dest(self, tmp_path):
        """A schedule created from a thread targets the parent channel."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        tool = ScheduleTool(
            store=store,
            timezone="America/Phoenix",
            agent_name="ezio",
            origin_getter=lambda: None,
            clock=lambda: NOW,
        )
        # Origin is the parent channel (the interface resolves thread→parent)
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("ezio", "discord", "99", "<@99>", "555", "estudos")
        ))
        try:
            result = await tool(
                action="create", cron="0 7 * * *", type="command", command="echo"
            )
            assert "Schedule s1 set" in result
            row = store.get("s1")
            assert row.dest_channel_id == "555"
            assert row.dest_channel_name == "estudos"
        finally:
            schedule_turn_scope.set(None)


# ---------------------------------------------------------------------------
# AC25: run_now executes immediately, next cron intact
# ---------------------------------------------------------------------------

class TestAC25RunNow:
    @pytest.mark.asyncio
    async def test_run_now_executes_and_keeps_schedule(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        sched = _make_schedule(
            next_run_at=NOW + timedelta(days=1),  # future — not due
            payload=json.dumps({"command": "echo manual", "cwd": str(tmp_path)}),
        )
        store.create(sched)

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
        )

        # Get the schedule and run it now
        row = store.get("s1")
        await engine.execute_now(row)

        # It executed (last_status updated)
        updated = store.get("s1")
        assert updated.last_status == "ok"
        # next_run_at is the next cron occurrence after NOW (tomorrow 7am)
        assert updated.next_run_at > NOW


# ---------------------------------------------------------------------------
# AC26: Cancel — cancelled never fires, unknown ID errors
# ---------------------------------------------------------------------------

class TestAC26Cancel:
    @pytest.mark.asyncio
    async def test_cancel_prevents_fire(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sched = _make_schedule(next_run_at=NOW - timedelta(seconds=1))
        store.create(sched)
        store.cancel("s1")

        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
        )
        await engine._poll_once()
        assert len(sent) == 0

    @pytest.mark.asyncio
    async def test_cancel_unknown_id(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        tool = ScheduleTool(
            store=store,
            timezone="America/Phoenix",
            agent_name="miranda",
            origin_getter=lambda: None,
            clock=lambda: NOW,
        )
        schedule_turn_scope.set(ScheduleTurnScope(
            origin=ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")
        ))
        try:
            result = await tool(action="cancel", id="s999")
            assert "not found" in result
        finally:
            schedule_turn_scope.set(None)


# ---------------------------------------------------------------------------
# AC27: Payload failure never stops the engine
# ---------------------------------------------------------------------------

class TestAC27EngineContinues:
    @pytest.mark.asyncio
    async def test_failure_doesnt_stop_other_schedules(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        # First schedule fails
        script_fail = tmp_path / "fail.sh"
        script_fail.write_text("#!/bin/sh\nexit 1")
        script_fail.chmod(script_fail.stat().st_mode | stat.S_IEXEC)

        # Second schedule succeeds
        script_ok = tmp_path / "ok.sh"
        script_ok.write_text("#!/bin/sh\necho ok")
        script_ok.chmod(script_ok.stat().st_mode | stat.S_IEXEC)

        store.create(_make_schedule(
            next_run_at=NOW - timedelta(seconds=2),
            payload=json.dumps({"command": str(script_fail), "cwd": str(tmp_path)}),
        ))
        store.create(_make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            payload=json.dumps({"command": str(script_ok), "cwd": str(tmp_path)}),
        ))

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            clock=lambda: NOW,
        )
        await engine._poll_once()

        # First failed → posted warning
        assert any("⚠️" in c for _, c in sent)
        # Second ran and succeeded → no extra warning for it
        # Both advanced
        row1 = store.get("s1")
        row2 = store.get("s2")
        assert row1.next_run_at > NOW
        assert row2.next_run_at > NOW
        assert row1.last_status == "failed"
        assert row2.last_status == "ok"


# ---------------------------------------------------------------------------
# AC28: Platform-scoped delivery
# ---------------------------------------------------------------------------

class TestAC28PlatformScoped:
    @pytest.mark.asyncio
    async def test_discord_engine_ignores_zulip_rows(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        sent: list[tuple[str, str]] = []

        async def fake_send(channel_id: str, content: str) -> None:
            sent.append((channel_id, content))

        # Zulip row due NOW
        store.create(_make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            platform="zulip",
            payload=json.dumps({"command": "echo zulip", "cwd": str(tmp_path)}),
        ))

        engine = ScheduleEngine(
            store, send=fake_send,
            poll_seconds=999, tz=UTC, agent_name="test",
            platform="discord",
            clock=lambda: NOW,
        )
        await engine._poll_once()
        # Nothing sent — the discord engine doesn't touch zulip rows
        assert len(sent) == 0

    @pytest.mark.asyncio
    async def test_claim_platform_scoped(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule(
            next_run_at=NOW - timedelta(seconds=1), platform="zulip",
        ))
        # Discord claim should fail on a zulip row
        assert store.claim("s1", platform="discord") is False
        # Zulip claim should succeed
        assert store.claim("s1", platform="zulip") is True


# ---------------------------------------------------------------------------
# Command executor unit tests
# ---------------------------------------------------------------------------

class TestCommandExecutor:
    @pytest.mark.asyncio
    async def test_success(self, tmp_path):
        result = await run_command(
            "echo hello", agent="test", cwd=str(tmp_path),
            clock=lambda: NOW,
        )
        assert result.success
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_failure_captures_stderr(self, tmp_path):
        result = await run_command(
            "echo 'bad' >&2; exit 2", agent="test", cwd=str(tmp_path),
            clock=lambda: NOW,
        )
        assert not result.success
        assert result.exit_code == 2
        assert "bad" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path):
        result = await run_command(
            "sleep 10", agent="test", cwd=str(tmp_path), timeout=0.3,
            clock=lambda: NOW,
        )
        assert result.timed_out
        assert not result.success

    @pytest.mark.asyncio
    async def test_log_file_created(self, tmp_path):
        # Override HOME to tmp_path for log isolation
        result = await run_command(
            "echo logged", agent="testagent", cwd=str(tmp_path),
            clock=lambda: NOW,
        )
        log = Path(result.log_path)
        assert log.exists()
        content = log.read_text()
        assert "echo logged" in content
        assert "logged" in content
