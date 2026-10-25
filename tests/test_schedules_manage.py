"""Schedules management: list, named channels, startup catch-up (spc-186).

AC29–AC31 (FR31, FR9–FR11, FR30). Injected clock, fake command payloads
(tmp_path scripts), fake resolver and send seams, and one real
DiscordInterface._start_schedules wiring check with a mocked adapter.
No services, credentials or network.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from sr2_spectre.reminders import tool as reminders_tool
from sr2_spectre.reminders.tool import ChannelMatch
from sr2_spectre.schedules import Schedule, ScheduleEngine, ScheduleStore
from sr2_spectre.tools.schedule_tool import (
    CROSS_PLATFORM_ERROR,
    MALFORMED_RESOLVER_ERROR,
    ScheduleOrigin,
    ScheduleTool,
    ScheduleTurnScope,
    schedule_turn_scope,
)

UTC = ZoneInfo("UTC")
PHX = ZoneInfo("America/Phoenix")
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)  # Monday 7am Phoenix


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


@pytest.fixture(autouse=True)
def _clean_scopes():
    """Turn scopes are ContextVars; start and end each test empty."""
    schedule_turn_scope.set(None)
    reminders_tool.turn_scope.set(None)
    yield
    schedule_turn_scope.set(None)
    reminders_tool.turn_scope.set(None)


def _open_origin_scope(
    agent: str = "miranda",
    platform: str = "discord",
    channel_id: str = "100",
    channel_name: str = "general",
) -> None:
    schedule_turn_scope.set(ScheduleTurnScope(
        origin=ScheduleOrigin(
            agent, platform, "42", "<@42>", channel_id, channel_name
        )
    ))


def _make_tool(store: ScheduleStore, resolver: Any = None) -> ScheduleTool:
    tool = ScheduleTool(
        store=store,
        timezone="America/Phoenix",
        agent_name="miranda",
        origin_getter=lambda: None,
        clock=lambda: NOW,
    )
    if resolver is not None:
        tool.set_channel_resolver(resolver)
    return tool


# ---------------------------------------------------------------------------
# AC29: list per FR31
# ---------------------------------------------------------------------------

class TestAC29List:
    """list returns this agent's active schedules sorted by next run with
    ID, local next run (weekday, date, time, zone), cron, destination,
    payload summary (first 40 characters) and last status."""

    async def test_line_fields_exact(self, tmp_path):
        """One row renders the whole FR31 line exactly."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule(
            next_run_at=datetime(2026, 10, 21, 14, 0, tzinfo=UTC),
            cron="0 7 * * *",
            payload=json.dumps({"command": "echo hello world", "cwd": None}),
            last_status="ok",
        ))
        tool = _make_tool(store)
        _open_origin_scope()

        out = await tool(action="list")
        # 2026-10-21 14:00 UTC == Wed 2026-10-21 07:00 in Phoenix; %Z is
        # the zone's local abbreviation (MST).
        assert out == (
            "s1 Wed 2026-10-21 07:00 MST cron=0 7 * * * "
            "#general command: echo hello world [ok]"
        )

    async def test_sorted_by_next_run(self, tmp_path):
        """Rows appear in next-run order, not ID or creation order.

        Payloads are swapped against ID order, so a row-order check
        (or any sort key other than next run) fails the test.
        """
        store = ScheduleStore(tmp_path / "test.sqlite")
        # Created first, due LATER; its summary says "later".
        store.create(_make_schedule(
            next_run_at=datetime(2026, 10, 23, 14, 0, tzinfo=UTC),
            payload=json.dumps({"command": "later", "cwd": None}),
        ))
        store.create(_make_schedule(
            next_run_at=datetime(2026, 10, 22, 14, 0, tzinfo=UTC),
            payload=json.dumps({"command": "sooner", "cwd": None}),
        ))
        tool = _make_tool(store)
        _open_origin_scope()

        out = await tool(action="list")
        lines = out.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("s2 Thu 2026-10-22 07:00 MST")
        assert "sooner" in lines[0]
        assert lines[1].startswith("s1 Fri 2026-10-23 07:00 MST")
        assert "later" in lines[1]

    async def test_payload_summary_truncated_to_40(self, tmp_path):
        """The summary is the first 40 characters, no more."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        long_command = "A" * 40 + "B" * 20
        store.create(_make_schedule(
            payload=json.dumps({"command": long_command, "cwd": None}),
        ))
        tool = _make_tool(store)
        _open_origin_scope()

        out = await tool(action="list")
        assert "A" * 40 in out
        assert "B" not in out  # character 41+ must not leak into the summary

    async def test_last_status_shown_or_never_run(self, tmp_path):
        """A run row shows its last status; a never-run row says so."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule(last_status=None))
        store.create(_make_schedule(
            next_run_at=NOW + timedelta(hours=2), last_status="failed",
        ))
        tool = _make_tool(store)
        _open_origin_scope()

        out = await tool(action="list")
        lines = out.splitlines()
        assert lines[0].endswith("[never run]")
        assert lines[1].endswith("[failed]")

    async def test_empty_returns_no_schedules(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        tool = _make_tool(store)
        _open_origin_scope()
        assert await tool(action="list") == "No schedules."

    async def test_only_own_active_rows(self, tmp_path):
        """Another agent's rows and cancelled rows never appear."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        store.create(_make_schedule())  # mine, active
        store.create(_make_schedule(agent="other"))  # another agent
        store.create(_make_schedule(status="cancelled"))  # cancelled
        tool = _make_tool(store)
        _open_origin_scope()

        out = await tool(action="list")
        lines = out.splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("s1 ")
        assert "other" not in out


# ---------------------------------------------------------------------------
# AC30: create with channel per FR10/FR11
# ---------------------------------------------------------------------------

class TestAC30NamedChannel:
    """create with `channel` resolves a Discord text channel; zero
    matches, multiple matches or a cross-platform name errors and stores
    nothing."""

    async def test_resolved_destination_stored(self, tmp_path):
        """A unique name stores the resolved id and name, not the origin."""
        store = ScheduleStore(tmp_path / "test.sqlite")

        async def resolver(platform: str, name: str, scope: Any = None):
            assert platform == "discord"
            assert name == "random"  # '#' stripped by the tool
            return [ChannelMatch(channel_id=777, channel_name="random")]

        tool = _make_tool(store, resolver=resolver)
        _open_origin_scope(channel_id="100", channel_name="general")

        result = await tool(
            action="create", cron="0 7 * * *", type="command",
            command="echo hi", channel="#random",
        )
        assert "Schedule s1 set" in result
        assert "#random" in result
        row = store.get("s1")
        assert row.dest_channel_id == "777"
        assert row.dest_channel_name == "random"

    async def test_name_without_hash_prefix_works(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")

        async def resolver(platform: str, name: str, scope: Any = None):
            assert name == "random"
            return [ChannelMatch(channel_id=777, channel_name="random")]

        tool = _make_tool(store, resolver=resolver)
        _open_origin_scope()
        result = await tool(
            action="create", cron="0 7 * * *", type="command",
            command="echo hi", channel="random",
        )
        assert "Schedule s1 set" in result
        assert store.get("s1").dest_channel_id == "777"

    async def test_zero_matches_stores_nothing(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")

        async def resolver(platform: str, name: str, scope: Any = None):
            return []

        tool = _make_tool(store, resolver=resolver)
        _open_origin_scope()
        result = await tool(
            action="create", cron="0 7 * * *", type="command",
            command="echo hi", channel="nope",
        )
        assert result == "Error: channel 'nope' not found on this platform"
        assert store.list_for_agent("miranda") == []

    async def test_multiple_matches_stores_nothing(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")

        async def resolver(platform: str, name: str, scope: Any = None):
            return [
                ChannelMatch(channel_id=1, channel_name="dup"),
                ChannelMatch(channel_id=2, channel_name="dup"),
            ]

        tool = _make_tool(store, resolver=resolver)
        _open_origin_scope()
        result = await tool(
            action="create", cron="0 7 * * *", type="command",
            command="echo hi", channel="dup",
        )
        assert result == "Error: channel 'dup' is ambiguous on this platform"
        assert store.list_for_agent("miranda") == []

    async def test_cross_platform_name_refused(self, tmp_path):
        """AC30 (FR11): 'zulip#general' from a Discord turn is refused with
        the exact error, nothing stored and no lookup attempted."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        calls: list = []

        async def spy(platform: str, name: str, scope: Any = None):
            calls.append((platform, name))
            return [ChannelMatch(channel_id=1, channel_name="general")]

        tool = _make_tool(store, resolver=spy)
        _open_origin_scope()
        result = await tool(
            action="create", cron="0 7 * * *", type="command",
            command="echo hi", channel="zulip#general",
        )
        assert result == "Error: Cross-platform schedules are not supported"
        assert CROSS_PLATFORM_ERROR == result
        assert store.list_for_agent("miranda") == []
        assert calls == []  # refused before any lookup

    async def test_same_platform_prefix_is_accepted(self, tmp_path):
        """FR11: naming THIS platform is not cross-platform; the name
        resolves after the prefix is stripped."""
        store = ScheduleStore(tmp_path / "test.sqlite")

        async def resolver(platform: str, name: str, scope: Any = None):
            assert name == "general"
            return [ChannelMatch(channel_id=777, channel_name="general")]

        tool = _make_tool(store, resolver=resolver)
        _open_origin_scope()
        result = await tool(
            action="create", cron="0 7 * * *", type="command",
            command="echo hi", channel="discord#general",
        )
        assert "Schedule s1 set" in result
        assert store.get("s1").dest_channel_id == "777"

    async def test_resolver_absent_stores_nothing(self, tmp_path):
        """FR10: a named create with no registered resolver errors and
        stores nothing."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        tool = _make_tool(store)  # no resolver
        _open_origin_scope()
        result = await tool(
            action="create", cron="0 7 * * *", type="command",
            command="echo hi", channel="random",
        )
        assert result.startswith("Error:")
        assert store.list_for_agent("miranda") == []

    async def test_malformed_resolver_result_stores_nothing(self, tmp_path):
        """FR10 guard: a resolver returning the wrong shape is a tool
        error, not an exception, and stores nothing."""
        store = ScheduleStore(tmp_path / "test.sqlite")

        async def resolver(platform: str, name: str, scope: Any = None):
            return [{"id": 5, "name": "raw-dict"}]

        tool = _make_tool(store, resolver=resolver)
        _open_origin_scope()
        result = await tool(
            action="create", cron="0 7 * * *", type="command",
            command="echo hi", channel="x",
        )
        assert result == "Error: channel lookup returned an unusable result"
        assert result == MALFORMED_RESOLVER_ERROR
        assert store.list_for_agent("miranda") == []

    async def test_create_without_channel_uses_origin(self, tmp_path):
        """Omitting `channel` is untouched behavior: the origin is stored
        even with a resolver registered."""
        store = ScheduleStore(tmp_path / "test.sqlite")

        async def resolver(platform: str, name: str, scope: Any = None):
            raise AssertionError("resolver must not be called")

        tool = _make_tool(store, resolver=resolver)
        _open_origin_scope(channel_id="100", channel_name="general")
        result = await tool(
            action="create", cron="0 7 * * *", type="command",
            command="echo hi",
        )
        assert "Schedule s1 set" in result
        row = store.get("s1")
        assert row.dest_channel_id == "100"
        assert row.dest_channel_name == "general"


# ---------------------------------------------------------------------------
# AC31: startup catch-up per FR30
# ---------------------------------------------------------------------------

def _make_engine(store: ScheduleStore, sent: list, tmp_path: Path) -> ScheduleEngine:
    async def fake_send(channel_id: str, content: str) -> None:
        sent.append((channel_id, content))

    return ScheduleEngine(
        store, send=fake_send,
        poll_seconds=999, tz=UTC, agent_name="test",
        platform="discord", clock=lambda: NOW,
    )


class TestAC31StartupCatchUp:
    """A schedule overdue at start fires once for the most recent missed
    occurrence, marked late in the log, and advances past now; on_miss:
    skip advances without firing; never more than one catch-up per
    schedule per start."""

    async def test_overdue_fires_once_late_and_advances(self, tmp_path, caplog):
        store = ScheduleStore(tmp_path / "test.sqlite")
        marker = tmp_path / "fired.count"
        store.create(_make_schedule(
            next_run_at=NOW - timedelta(hours=2),  # missed occurrence
            payload=json.dumps(
                {"command": f"printf x >> {marker}", "cwd": str(tmp_path)}
            ),
        ))
        sent: list = []
        engine = _make_engine(store, sent, tmp_path)

        with caplog.at_level(logging.INFO, logger="sr2_spectre.schedules.engine"):
            await engine.start()
            await asyncio.sleep(0)  # let the (fast-forwarded) poll task begin
            await engine.stop()

        # Fired: the payload really ran.
        assert marker.exists() and marker.read_text() == "x"
        # Marked late in the log.
        assert any(
            "schedule fired late" in rec.getMessage() and "id=s1" in rec.getMessage()
            for rec in caplog.records
        )
        assert not any(
            rec.getMessage() == "schedule fired: agent=test id=s1"
            for rec in caplog.records
        )
        # Advanced past now to the next cron occurrence after NOW
        # (cron "0 7 * * *" in UTC: tomorrow 07:00).
        row = store.get("s1")
        assert row.next_run_at == datetime(2026, 10, 21, 7, 0, tzinfo=UTC)
        assert row.last_status == "ok"
        assert row.last_run_at == NOW
        # No failure post for a successful run.
        assert sent == []

    async def test_never_more_than_one_catch_up_per_start(self, tmp_path, caplog):
        """Even a **/min schedule (next occurrence within the process
        lifetime) fires exactly once per start."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        marker = tmp_path / "fired.count"
        store.create(_make_schedule(
            cron="* * * * *",
            next_run_at=NOW - timedelta(hours=2),
            payload=json.dumps(
                {"command": f"printf x >> {marker}", "cwd": str(tmp_path)}
            ),
        ))
        sent: list = []
        engine = _make_engine(store, sent, tmp_path)

        with caplog.at_level(logging.INFO, logger="sr2_spectre.schedules.engine"):
            await engine.start()
            await asyncio.sleep(0.4)  # real seconds: a */min poll could refire
            await engine.stop()

        assert marker.read_text() == "x"  # exactly one payload run
        fires = [
            r for r in caplog.records
            if "schedule fired" in r.getMessage() and "id=s1" in r.getMessage()
        ]
        assert len(fires) == 1

    async def test_skip_advances_without_firing(self, tmp_path, caplog):
        store = ScheduleStore(tmp_path / "test.sqlite")
        marker = tmp_path / "fired.count"
        store.create(_make_schedule(
            on_miss="skip",
            next_run_at=NOW - timedelta(hours=2),
            payload=json.dumps(
                {"command": f"printf x >> {marker}", "cwd": str(tmp_path)}
            ),
        ))
        sent: list = []
        engine = _make_engine(store, sent, tmp_path)

        with caplog.at_level(logging.INFO, logger="sr2_spectre.schedules.engine"):
            await engine.start()
            await asyncio.sleep(0.1)
            await engine.stop()

        assert not marker.exists()  # payload never ran
        assert sent == []  # no posts
        row = store.get("s1")
        assert row.next_run_at > NOW  # advanced past now
        assert row.last_status == "skipped"
        assert any(
            "skipped" in r.getMessage() and "id=s1" in r.getMessage()
            for r in caplog.records
        )

    async def test_second_start_does_not_catch_up_again(self, tmp_path):
        """After a start's catch-up advanced the row past now, a new
        engine (new process) does not fire it again."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        marker = tmp_path / "fired.count"
        payload = json.dumps(
            {"command": f"printf x >> {marker}", "cwd": str(tmp_path)}
        )
        store.create(_make_schedule(
            next_run_at=NOW - timedelta(hours=2), payload=payload,
        ))
        sent: list = []

        first = _make_engine(store, sent, tmp_path)
        await first.start()
        await asyncio.sleep(0)
        await first.stop()
        assert marker.read_text() == "x"

        second = _make_engine(store, sent, tmp_path)
        await second.start()
        await asyncio.sleep(0.1)
        await second.stop()
        assert marker.read_text() == "x"  # still exactly one run

    async def test_future_row_untouched_by_catch_up(self, tmp_path):
        store = ScheduleStore(tmp_path / "test.sqlite")
        marker = tmp_path / "fired.count"
        future = datetime(2026, 10, 21, 14, 0, tzinfo=UTC)
        store.create(_make_schedule(
            next_run_at=future,
            payload=json.dumps(
                {"command": f"printf x >> {marker}", "cwd": str(tmp_path)}
            ),
        ))
        sent: list = []
        engine = _make_engine(store, sent, tmp_path)

        await engine.start()
        await asyncio.sleep(0.1)
        await engine.stop()

        assert not marker.exists()
        assert sent == []
        row = store.get("s1")
        assert row.next_run_at == future
        assert row.last_status is None
        assert row.last_run_at is None

    async def test_catch_up_platform_scoped(self, tmp_path):
        """A discord engine's catch-up never touches another platform's
        overdue rows (spc-176 scoping, same as the poll)."""
        store = ScheduleStore(tmp_path / "test.sqlite")
        marker = tmp_path / "fired.count"
        store.create(_make_schedule(
            platform="zulip",
            next_run_at=NOW - timedelta(hours=2),
            payload=json.dumps(
                {"command": f"printf x >> {marker}", "cwd": str(tmp_path)}
            ),
        ))
        sent: list = []
        engine = _make_engine(store, sent, tmp_path)

        await engine.start()
        await asyncio.sleep(0.1)
        await engine.stop()

        assert not marker.exists()
        row = store.get("s1")
        assert row.next_run_at == NOW - timedelta(hours=2)
        assert row.last_status is None


# ---------------------------------------------------------------------------
# Wiring (orchestrator finding 3): _start_schedules seats the resolver
# ---------------------------------------------------------------------------

class TestDiscordWiring:
    """The Discord interface seats its guild-scoped resolver on the
    schedule tool through the post-construction seam, and the tool then
    resolves through the REAL interface resolver with the reminder turn
    scope as the guild source."""

    async def test_start_schedules_seats_the_real_resolver(self, tmp_path):
        import discord

        from sr2_spectre.interfaces.discord.config import DiscordConfig
        from sr2_spectre.interfaces.discord.interface import DiscordInterface

        def channel(cls, name: str, id_: int) -> MagicMock:
            # spec= makes the mock pass the adapter's real isinstance
            # check; the Thread mock proves threads never resolve (the
            # AC15 rule the real adapter enforces).
            c = MagicMock(spec=cls)
            c.name = name
            c.id = id_
            return c

        interface = DiscordInterface(config=DiscordConfig(token="test"))
        adapter = MagicMock()
        # Assign the real isinstance rule: a plain MagicMock child would
        # answer "sendable" for every channel, and the Thread mock below
        # would resolve unchecked.
        import sr2_spectre.interfaces.discord.adapter as adapter_mod
        real_is_text = adapter_mod.DiscordBotAdapter.is_sendable_text_channel
        # A function stored on a MagicMock attribute is called plain (no
        # descriptor binding), so the real unbound check gets its one arg.
        adapter.is_sendable_text_channel = lambda ch: real_is_text(adapter, ch)
        adapter.guild_channels.return_value = [
            channel(discord.TextChannel, "random", 777),
            channel(discord.TextChannel, "general", 100),
            channel(discord.Thread, "random", 888),
        ]
        interface._adapter = adapter

        store = ScheduleStore(tmp_path / "test.sqlite")
        tool = ScheduleTool(
            store=store,
            timezone="America/Phoenix",
            agent_name="miranda",
            origin_getter=lambda: None,
            clock=lambda: NOW,
        )
        agent = MagicMock()
        agent.schedules_stamped = 0
        agent.schedules = MagicMock(
            store=store,
            tool=tool,
            poll_seconds=999.0,
            tz=UTC,
            agent_name="miranda",
            command_timeout=60.0,
        )

        interface._start_schedules(agent)
        try:
            # The engine start task was launched; the tool carries the seam.
            # (Bound-method comparison pins name, instance and function:
            #  tool._schedule_channel_resolver == interface._resolve_channel.)
            assert tool._schedule_channel_resolver == interface._resolve_channel
            assert tool._schedule_channel_resolver.__self__ is interface
            assert (
                getattr(tool._schedule_channel_resolver, "spectre_platform", None)
                == "discord"
            )

            # And create resolves end-to-end through the real interface
            # resolver: guild from the reminder turn scope (finding 4).
            _open_origin_scope(channel_id="100", channel_name="general")
            reminders_tool.turn_scope.set(reminders_tool.TurnScope(
                origin=reminders_tool.ReminderOrigin(
                    "miranda", "discord", "42", "<@42>", "100", "general", ""
                ),
                guild_id=555,
            ))
            result = await tool(
                action="create", cron="0 7 * * *", type="command",
                command="echo hi", channel="#random",
            )
            assert "Schedule s1 set" in result
            row = store.get("s1")
            assert row.dest_channel_id == "777"
            assert row.dest_channel_name == "random"
            adapter.guild_channels.assert_called_once_with(555)
        finally:
            await interface._stop_schedules()
            schedule_turn_scope.set(None)
            reminders_tool.turn_scope.set(None)
