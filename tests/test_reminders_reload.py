"""agent.reminders follows live config reload (spc-143).

The reload gap: ``Runtime.apply_config`` never re-ran ``_reminders_for``
and the Zulip delivery loop started only in ``ZulipInterface.start()``, so
the FR6 switch could drift — enabling at runtime did nothing, disabling
left the loop delivering.

These tests drive ``Runtime.apply_config`` and the real ZulipInterface
with fakes at the same seams spc-139's tests use:

- the clock is an injected callable returning a controlled UTC datetime
  (no real waiting on reminder timing; polling intervals are real but
  tiny);
- the Zulip send boundary is a recording fake client (no server, no
  credentials, no network);
- the reminder store is a per-test SQLite file under ``tmp_path`` —
  never ``~/.sr2`` live state.

A ``FakeAgent`` facade mirrors the attributes the interface reads off a
real ``Agent`` (``config``, ``reminders``, ``reminders_seq``,
``_runtime``); the Runtime under test is real, built with the documented
``sr2_spectre.live_llm.LiteLLMCallable`` seam patched.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from sr2_spectre.config import (
    AgentConfig,
    ModelConfig,
    SpectreConfig,
)
from sr2_spectre.reminders import Reminder, ReminderStore

UTC = ZoneInfo("UTC")

# Fixed "now" so no test ever waits on the wall clock.
NOW = datetime(2026, 10, 15, 22, 0, 0, tzinfo=UTC)


def _clock_at(moment: datetime):
    """A clock frozen at *moment*."""
    return lambda: moment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(store: Path, **reminders: Any) -> SpectreConfig:
    """A minimal real SpectreConfig with one reminders block."""
    agent: dict[str, Any] = {"name": "miranda", "default_skills": False}
    if reminders:
        agent["reminders"] = dict(reminders)
    return SpectreConfig(
        agent=AgentConfig(**agent),
        models={"default": ModelConfig(model="test-model")},
        pipeline={"layers": []},
        provenance_store_path="",
        memory_store_dsn="",
    )


def _enabled(store: Path, **overrides: Any) -> SpectreConfig:
    block: dict[str, Any] = {
        "enabled": True,
        "timezone": "America/Phoenix",
        "store_path": str(store),
        "poll_seconds": 60,
    }
    block.update(overrides)
    return _config(store, **block)


def _runtime(config: SpectreConfig):
    """A real Runtime with the documented LLM seam patched (no model)."""
    from sr2_spectre.runtime import Runtime

    with patch("sr2_spectre.live_llm.LiteLLMCallable"):
        return Runtime(config=config)


class FakeAgent:
    """Facade stand-in for what the Zulip interface reads off an Agent:
    ``_runtime`` (via ``_runtime_of``), ``reminders``, ``reminders_seq``
    and ``config``. ``config`` tracks the Runtime across reloads, like the
    real ``Agent.apply_config`` facade does."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def config(self) -> Any:
        return self._runtime.config

    @property
    def reminders(self) -> Any:
        return self._runtime.reminders

    @property
    def reminders_seq(self) -> int:
        return self._runtime.reminders_seq

    @property
    def reminders_stamped(self) -> int:
        return self._runtime.reminders_stamped

    def create_session(self, frame_id: str):
        """Same delegation the real Agent facade does."""
        return self._runtime.new_session(frame_id=frame_id)


def _make_interface(store_dir: Path, poll_seconds: float = 0.01):
    """A ZulipInterface with a fake client: no server, no credentials, no
    network (same shape as spc-139's test helper).

    ``poll_seconds`` overrides each built delivery loop's real sleep
    interval (tiny — injected clocks mean no reminder timing depends on
    it), so loop churn between tests stays cheap.
    """
    import sr2_spectre.interfaces.zulip.interface as iface_mod
    from sr2_spectre.interfaces.zulip.client import ZulipCredentials
    from sr2_spectre.interfaces.zulip.config import ZulipConfig

    config = ZulipConfig(
        zuliprc=str(store_dir / "zuliprc"),
        users=[8],
        channels=[42],
        channel_areas={"42": ""},
    )

    class _FakeClient:
        def __init__(self) -> None:
            self.sends: list[dict] = []

        async def register(self):
            return {
                "queue_id": "q-test",
                "last_event_id": 0,
                "realm_users": [{"user_id": 9, "is_bot": True}],
                "realm_bots": [],
            }

        async def aclose(self):
            pass

        async def get_topic_messages(self, channel_id, topic):
            # Cold topic: only the trigger itself (excluded by id), so the
            # probe admits the Session.
            return {"messages": []}

        async def send_message(self, **kwargs):
            self.sends.append(kwargs)
            return 1

    iface_mod.load_zulip_credentials = lambda path: ZulipCredentials(
        email="bot@test.local", api_key="k", site="http://127.0.0.1:1"
    )
    iface_mod.ZulipClient = lambda **kw: _FakeClient()
    interface = iface_mod.ZulipInterface(config)

    real_start_reminders = interface._start_reminders

    def _start(agent: Any) -> None:
        real_start_reminders(agent)
        if interface._reminders_loop is not None:
            interface._reminders_loop._poll_seconds = poll_seconds

    interface._start_reminders = _start
    return interface


async def _pending_row(
    store_path: Path, text: str, reminder_id: str = "r1", due_at: datetime | None = None
) -> None:
    """Insert one pending row directly (bypasses tool validation).

    ``due_at`` defaults to 10 minutes before NOW (delivered on the first
    poll of a loop frozen at NOW); pass an explicit future time for rows
    that must stay pending.
    """
    store = ReminderStore(store_path)
    store.create(
        Reminder(
            id=reminder_id,
            agent="miranda",
            platform="zulip",
            requester_id="8",
            requester_mention="@8",
            origin_channel_id="42",
            dest_channel_id="42",
            dest_channel_name="sr2",
            dest_topic="reminders",
            text=text,
            due_at=due_at if due_at is not None else NOW - timedelta(minutes=10),
            created_at=NOW,
            delivered_at=None,
            status="pending",
            attempts=0,
            last_error=None,
        )
    )


def _statuses(store_path: Path) -> dict[str, str]:
    """id -> status for every row (compare pending rows across a disable)."""
    store = ReminderStore(store_path)
    conn = store._connect()
    try:
        rows = conn.execute("SELECT id, status FROM reminders").fetchall()
    finally:
        conn.close()
    return {row["id"]: row["status"] for row in rows}


async def _deliveries(interface: Any, attempts: int = 10) -> list[dict]:
    """Let the running loop poll a few times (tiny sleeps, injected
    clock); return whatever the fake client received."""
    for _ in range(attempts):
        if interface._client.sends:
            break
        await asyncio.sleep(0.01)
    return interface._client.sends


async def _drive_message(interface: Any, agent: Any, message_id: int) -> None:
    """One message through the real admit+turn path, with a stub Session.

    The per-message config reload itself runs upstream in the real
    Session.handle_user_message; the reload under test is performed
    explicitly by the caller before this, so the stub turn only verifies
    the interface's own per-message sync fires inside the turn.
    """
    message = {
        "type": "stream",
        "stream_id": 42,
        "id": message_id,
        "subject": "plan",
        "content": "hello",
        "sender_id": 8,
    }
    assert interface._is_allowed_message(message)
    entry = await interface._ensure_session(
        "zulip:miranda:http://127.0.0.1:1#42#plan",
        42,
        "plan",
        message_id,
        {message_id},
    )
    assert entry is not None
    session, _seen = entry

    async def handle(text: str):
        return type("Result", (), {"text": ""})()

    session.handle_user_message = handle
    try:
        await interface._run_turn(session, 42, "plan", message)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# R1 — disabled -> enabled: tool registered, loop started, no restart
# ---------------------------------------------------------------------------


class TestR1EnableOnReload:
    async def test_reload_enable_registers_tool_and_starts_loop(self, tmp_path):
        """R1: reloading a config that switches reminders from disabled to
        enabled registers the reminder tool and starts the Zulip delivery
        loop without restarting the interface — and a row already due at
        that moment is delivered through the interface's own send path."""
        store_file = tmp_path / "r.sqlite"
        interface = _make_interface(tmp_path)
        runtime = _runtime(_config(store_file))  # no block = disabled
        agent = FakeAgent(runtime)
        assert "reminder" not in runtime.registry
        assert runtime.reminders is None

        await interface.start(agent)
        try:
            assert interface._reminders_loop is None  # nothing to run yet

            # Operator flips enabled: true between messages.
            runtime.apply_config(_enabled(store_file, poll_seconds=30))
            assert "reminder" in runtime.registry
            await interface._sync_reminders()
            loop = interface._reminders_loop
            assert loop is not None
            assert loop._store is runtime.reminders.store
            assert loop._tz is runtime.reminders.tz

            # The started loop delivers: a pending due row reaches the
            # fake client with the FR13 shape. (The helper's tiny
            # poll_seconds override applies to loops started on reload
            # too, so this stays a fast real-poll test, not a wait.)
            await _pending_row(store_file, "ship it")
            loop._clock = _clock_at(NOW)
            sends = await _deliveries(interface)
            assert sends, "the reloaded loop never delivered"
            assert sends[0]["to"] == 42
            assert sends[0]["topic"] == "reminders"
            assert "@8 Reminder: ship it" in sends[0]["content"]
        finally:
            await interface.stop()

    async def test_running_interface_starts_loop_on_next_message(self, tmp_path):
        """R1 through the message path: the loop is None after start(), a
        reload-then-message leaves it running (sync fires in _run_turn)."""
        store_file = tmp_path / "r.sqlite"
        interface = _make_interface(tmp_path)
        runtime = _runtime(_config(store_file))
        agent = FakeAgent(runtime)

        await interface.start(agent)
        try:
            assert interface._reminders_loop is None
            runtime.apply_config(_enabled(store_file, poll_seconds=30))
            # Nothing synced yet — the loop only appears via a message.
            assert interface._reminders_loop is None
            await _drive_message(interface, agent, message_id=101)
            assert interface._reminders_loop is not None
        finally:
            await interface.stop()


# ---------------------------------------------------------------------------
# R2 — enabled -> disabled: tool unregistered, loop stopped, rows intact
# ---------------------------------------------------------------------------


class TestR2DisableOnReload:
    async def test_reload_disable_unregisters_tool_and_stops_loop(self, tmp_path):
        """R2: reloading a config that switches reminders from enabled to
        disabled unregisters the tool and stops the loop; pending rows are
        left intact."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        agent = FakeAgent(runtime)
        assert "reminder" in runtime.registry
        interface = _make_interface(tmp_path)

        await interface.start(agent)
        try:
            assert interface._reminders_loop is not None
            # A pending row a year ahead: the running loop must never
            # touch it, and it must survive the disable exactly as saved.
            await _pending_row(
                store_file, "survivor", due_at=NOW + timedelta(days=365)
            )
            before = _statuses(store_file)
            assert before == {"r1": "pending"}

            applied = runtime.apply_config(_config(store_file))
            assert "agent.reminders" in applied
            assert "reminder" not in runtime.registry
            assert runtime.reminders is None

            await interface._sync_reminders()
            assert interface._reminders_loop is None

            # A stopped loop delivers nothing further...
            for _ in range(5):
                await asyncio.sleep(0.01)
            assert interface._client.sends == []
            # ...and the pending row is intact.
            assert _statuses(store_file) == before
        finally:
            await interface.stop()


# ---------------------------------------------------------------------------
# R3 — unchanged block: no loop restart, no re-registration
# ---------------------------------------------------------------------------


class TestR3UnchangedIsQuiet:
    async def test_unchanged_block_neither_restarts_loop_nor_re_registers(
        self, tmp_path
    ):
        """R3: reloading with an unchanged reminders block neither restarts
        the loop nor re-registers the tool — every identity is preserved,
        so the running task is never touched."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        agent = FakeAgent(runtime)
        interface = _make_interface(tmp_path)

        await interface.start(agent)
        try:
            loop_before = interface._reminders_loop
            tool_before = runtime.reminders.tool
            store_before = runtime.reminders.store
            task_before = loop_before._task

            # Reload #1: whole config identical — early return.
            assert runtime.apply_config(_enabled(store_file, poll_seconds=30)) == []
            await interface._sync_reminders()

            assert interface._reminders_loop is loop_before
            assert runtime.reminders.tool is tool_before
            assert runtime.reminders.store is store_before
            assert loop_before._task is task_before  # same asyncio task

            # Reload #2: the block stays equal while another field moves,
            # so the reload runs its full body — reminders must still be
            # left exactly alone (no re-registration, no restart).
            other = _enabled(store_file, poll_seconds=30)
            other = other.model_copy(
                update={"models": {"default": ModelConfig(model="other-model")}}
            )
            assert runtime.apply_config(other) == ["models"]
            await interface._sync_reminders()

            assert interface._reminders_loop is loop_before
            assert runtime.reminders.tool is tool_before
            assert runtime.reminders.store is store_before
            assert loop_before._task is task_before
        finally:
            await interface.stop()

    async def test_no_op_reload_with_other_fields_leaves_reminders_alone(
        self, tmp_path
    ):
        """R3: a reload that changes something else (the model) applies it
        but the reminders branch stays untouched — no restart, no
        re-register, and it is not reported as applied."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        agent = FakeAgent(runtime)
        interface = _make_interface(tmp_path)

        await interface.start(agent)
        try:
            loop_before = interface._reminders_loop
            tool_before = runtime.reminders.tool

            changed = _enabled(store_file, poll_seconds=30)
            changed = changed.model_copy(
                update={
                    "models": {"default": ModelConfig(model="other-model")}
                }
            )
            applied = runtime.apply_config(changed)
            assert "agent.reminders" not in applied
            await interface._sync_reminders()

            assert interface._reminders_loop is loop_before
            assert runtime.reminders.tool is tool_before
        finally:
            await interface.stop()


# ---------------------------------------------------------------------------
# R4 — settings change while enabled takes effect without a restart
# ---------------------------------------------------------------------------


class TestR4SettingsChangeWhileEnabled:
    async def test_poll_and_timezone_change_take_effect_without_restart(
        self, tmp_path
    ):
        """R4: a reload that changes reminders settings while enabled
        (poll_seconds, timezone) takes effect without restarting the
        interface — the loop is stopped and rebuilt from the fresh handle,
        and the restarted loop delivers."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        agent = FakeAgent(runtime)
        interface = _make_interface(tmp_path, poll_seconds=0.01)

        await interface.start(agent)
        try:
            loop_before = interface._reminders_loop

            applied = runtime.apply_config(
                _enabled(store_file, poll_seconds=31, timezone="America/New_York")
            )
            assert "agent.reminders" in applied

            await interface._sync_reminders()
            loop = interface._reminders_loop
            assert loop is not None
            assert loop is not loop_before        # replaced, not reused
            assert loop_before._task is None      # old loop was stopped
            assert loop._tz.key == "America/New_York"
            assert loop._store is runtime.reminders.store
            # Nominal poll value from the fresh handle (the helper's tiny
            # override applies after, and only changes the test cadence).
            assert runtime.reminders.poll_seconds == 31

            # The re-seated loop delivers through the interface.
            await _pending_row(store_file, "after reseat")
            loop._clock = _clock_at(NOW)
            sends = await _deliveries(interface)
            assert sends, "the re-seated loop never delivered"
            assert "after reseat" in sends[0]["content"]
        finally:
            await interface.stop()


# ---------------------------------------------------------------------------
# Runtime-level surfacing (what the interface compares against)
# ---------------------------------------------------------------------------


class TestRuntimeSurfacing:
    async def test_applied_names_and_stamp_semantics(self, tmp_path):
        """apply_config reports "agent.reminders" ONLY when the block
        changed; reminders_stamped (what the interface compares) moves on
        every change — enable, settings, disable — and never on an
        unchanged reload; a disabled startup contributes no stamp."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_config(store_file))
        assert runtime.reminders_stamped == 0  # disabled startup: nothing seated
        assert runtime.reminders_seq == 0

        assert runtime.apply_config(_enabled(store_file, poll_seconds=30)) == [
            "agent.reminders"
        ]
        stamp = runtime.reminders_stamped
        assert stamp == 1
        assert runtime.reminders_seq == 1

        # Unchanged block, freshly built equal value: nothing applied,
        # nothing stamped (R3's runtime half).
        assert runtime.apply_config(_enabled(store_file, poll_seconds=30)) == []
        assert runtime.reminders_stamped == stamp

        # Settings change while enabled: reported and re-seated.
        assert runtime.apply_config(
            _enabled(store_file, poll_seconds=45)
        ) == ["agent.reminders"]
        assert runtime.reminders_stamped == stamp + 1

        # Disable: reported and stamped (the interface must stop its
        # loop), but no new seat — the handle clears instead.
        assert runtime.apply_config(_config(store_file)) == ["agent.reminders"]
        assert runtime.reminders_stamped == stamp + 2
        assert runtime.reminders_seq == stamp + 1
        assert runtime.reminders is None
        assert runtime.reminders_enabled is False

    async def test_enabled_reload_without_timezone_leaves_consistent_state(
        self, tmp_path
    ):
        """An enabled block without a timezone raises on reload (same
        contract as startup) and the Runtime keeps the previous working
        seat; the raise aborts the reload, and the last-good source makes
        the next message retry from a consistent base."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        tool_before = runtime.reminders.tool
        seq_before = runtime.reminders_seq

        # Built directly: enabled=True with the default empty timezone is
        # the exact broken state an operator's edit produces.
        broken = _enabled(store_file, poll_seconds=30).model_dump()
        broken["agent"]["reminders"] = {"enabled": True}
        broken = SpectreConfig(**broken)
        with pytest.raises(ValueError, match="timezone"):
            runtime.apply_config(broken)

        assert runtime.reminders.tool is tool_before
        assert runtime.reminders_seq == seq_before
        # _reminders_for raises before "agent.reminders" is reported; the
        # seat is untouched either way.
        # Re-raising with the same broken block (as the last-good source
        # will keep serving until the file is fixed) behaves identically.
        with pytest.raises(ValueError, match="timezone"):
            runtime.apply_config(broken)
