"""spc-159/spc-166 — one hand-off for the reminder channel resolver.

The resolver reaches ``ReminderTool`` through its constructor and through
nothing else: the owning interface registers its resolver once on the
Runtime through the public seam (``Runtime.set_reminders_channel_resolver``,
mirrored on the Agent facade), the Runtime passes it into every
ReminderTool it builds, including the one rebuilt when a live reload
re-seats reminders (spc-143) — and, since spc-166, including the tool
already seated when the resolver arrives, which the seam rebuilds through
that same construction site. No code outside ``reminders/`` reads or
assigns a private ReminderTool attribute.

The same fakes as the other reminder tests: a recording fake Zulip
client (no server, no credentials, no network), injected clocks, and a
per-test SQLite store under ``tmp_path`` — never ``~/.sr2`` live state.

Lines tagged READBACK touch ReminderTool's private state from the tests
as a fixture convenience (an origin, a frozen clock); the R1 rule binds
production code, and R1's grep covers ``src/``.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

from sr2_spectre.config import (
    AgentConfig,
    ModelConfig,
    SpectreConfig,
)
from sr2_spectre.reminders import (
    ChannelMatch,
    ReminderOrigin,
    ReminderStore,
    ReminderTool,
)
from sr2_spectre.reminders.tool import (
    NO_INTERFACE_ERROR,
    _match_has_channels,
    format_local_due,
)

UTC = ZoneInfo("UTC")

# Fixed "now" so no test ever waits on the wall clock.
NOW = datetime(2026, 10, 15, 22, 0, 0, tzinfo=UTC)  # 15:00 MST


def _clock_at(moment: datetime):
    """A clock frozen at *moment*."""
    return lambda: moment


# ---------------------------------------------------------------------------
# Helpers (same shapes as the existing reminder tests)
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
    """Facade stand-in mirroring the real Agent for the interface: the
    attributes the interface reads (``config``, ``reminders``,
    ``reminders_stamped``, ``_runtime``) plus the spc-159 seam, delegated
    to the Runtime exactly like the real facade does. ``create_session``
    returns None here; R3 drives ``_run_turn`` with a stub Session
    instead (same as spc-143's reload tests)."""

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

    def set_reminders_channel_resolver(self, resolver: Any) -> None:
        self._runtime.set_reminders_channel_resolver(resolver)


def _make_interface(store_dir: Path, poll_seconds: float = 0.01):
    """A ZulipInterface with a fake client: no server, no credentials,
    no network (same shape as the spc-139/spc-143 test helpers).

    ``poll_seconds`` overrides each built delivery loop's real sleep
    interval (tiny) so loop churn between tests stays cheap.
    """
    import sr2_spectre.interfaces.zulip.interface as iface_mod
    from sr2_spectre.interfaces.zulip.client import ZulipCredentials
    from sr2_spectre.interfaces.zulip.config import ZulipConfig

    config = ZulipConfig(
        zuliprc=str(store_dir / "zuliprc"),
        allowed_user_ids=[8],
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
                "streams": [
                    {"stream_id": 42, "name": "sr2-spectre"},
                    {"stream_id": 7, "name": "general"},
                ],
            }

        async def aclose(self):
            pass

        async def get_streams(self):
            return {
                "streams": [
                    {"stream_id": 42, "name": "sr2-spectre", "can_send": True},
                    {"stream_id": 7, "name": "general", "can_send": True},
                    {"stream_id": 8, "name": "read-only", "can_send": False},
                ]
            }

        async def get_topic_messages(self, channel_id, topic):
            # Cold topic: only the trigger itself (excluded by id).
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


def _origin(**overrides) -> ReminderOrigin:
    """A Zulip turn origin: Diego in channel 42, topic 'plan'."""
    base = dict(
        agent="miranda",
        platform="zulip",
        requester_id="8",
        requester_mention="@8",
        channel_id="42",
        channel_name="sr2-spectre",
        topic="plan",
    )
    base.update(overrides)
    return ReminderOrigin(**base)


def _tool(store: ReminderStore, **kwargs: Any) -> ReminderTool:
    """A ReminderTool on the frozen clock; every other arg as given —
    the constructor is the single documented construction path."""
    params: dict[str, Any] = dict(
        store=store,
        timezone="America/Phoenix",
        agent_name="miranda",
        origin_getter=lambda: None,
        clock=_clock_at(NOW),
    )
    params.update(kwargs)
    return ReminderTool(**params)


def _resolver(match_lists: dict, platform: str = "zulip"):
    """A fake resolver: name (lowercased) -> list of ChannelMatch built
    from (channel_id, name) tuples. The seam takes exactly the FR10
    documented match shape — asserted at yield, not at the first
    named-channel create. Carries the seam's platform marker (spc-159).
    """

    async def resolve(platform: str, name: str):  # noqa: A002 — seam signature
        matches = [
            ChannelMatch(channel_id=cid, channel_name=cname)
            for cid, cname in match_lists.get(name.lower(), [])
        ]
        assert _match_has_channels(matches), (
            "fake resolver yields non-matches — fix the fake, the seam "
            "takes the FR10 shape"
        )
        return matches

    resolve.spectre_platform = platform
    return resolve


async def _drive_zulip_turn(interface: Any, agent: Any, tool: Any) -> str:
    """One named-channel create through the REAL turn path.

    ``_run_turn`` records the per-turn origin on the Runtime, runs the
    sync, and clears the origin afterwards; the stub "model turn" is
    where the tool call happens, so the create sees a genuine Zulip
    turn origin (channel 42, sender 8 — the interface's own
    ``_reminders_origin_for``). The caller performs any reload before
    this; the sync inside re-seats the loop (spc-143).
    """
    message = {
        "type": "stream",
        "stream_id": 42,
        "id": 101,
        "subject": "plan",
        "content": "remind me",
        "sender_id": 8,
    }
    captured: list[str] = []

    class _StubSession:
        async def handle_user_message(self, text: str):
            captured.append(
                await tool(
                    action="create", text="standup", in_minutes=60,
                    channel="general",
                )
            )
            # Empty text: the interface sends no reply; the store tells all.
            return type("Result", (), {"text": ""})()

    await interface._run_turn(_StubSession(), 42, "plan", message)
    assert len(captured) == 1
    return captured[0]


# ---------------------------------------------------------------------------
# R1 — the resolver has one construction path (constructor via the seam)
# ---------------------------------------------------------------------------


class TestR1SingleConstructionPath:
    def test_no_file_outside_reminders_touches_a_private_tool_attr(self):
        """R1 (source-level): no line in src/ names a private
        ReminderTool attribute outside reminders/, except the seam's own
        ends (the setter, its delegation, the ctor-kwarg hand-off)."""
        repo_root = Path(__file__).resolve().parents[1]
        src_root = repo_root / "src" / "sr2_spectre"
        reminders_dir = src_root / "reminders"
        # The tool's private attribute names, harvested from its ctor.
        import inspect

        import sr2_spectre.reminders.tool as tool_mod

        tool_privates = {
            m.group(1)
            for m in re.finditer(
                r"self\.(_[a-z]\w*)\s*=",
                inspect.getsource(tool_mod.ReminderTool.__init__),
            )
        }
        assert "_channel_resolver" in tool_privates  # READBACK
        private_attr_use = re.compile(r"\b\w+\s*\.\s*(_[a-z]\w*)\b")
        offenders: list[str] = []
        for path in sorted(src_root.rglob("*.py")):
            if path.resolve().is_relative_to(reminders_dir):
                continue
            for n, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                # The Runtime legitimately names its own seam field
                # (the setter stores it; the ctor kwarg reads it). Any
                # OTHER mention of one of the tool's privates outside
                # reminders/ — above all the superseded
                # ``tool._channel_resolver = …`` patch — is what R1 forbids.
                if "_reminders_channel_resolver" in line:
                    continue
                for m in private_attr_use.finditer(line):
                    if m.group(1) in tool_privates:
                        offenders.append(
                            f"{path.relative_to(repo_root)}:{n}: {line.strip()}"
                        )
                        break
        assert offenders == [], offenders

    async def test_runtime_passes_the_registered_resolver_into_construction(
        self, tmp_path
    ):
        """R1: a resolver registered on the Runtime seam reaches the tool
        through its constructor and is used from there, regardless of
        whether it was registered before or after the tool was built."""
        # Order A: register first, then seat — the ctor takes the seam.
        store_file = tmp_path / "order-a.sqlite"
        runtime = _runtime(_config(store_file))  # disabled: nothing to build
        resolver = _resolver({"general": [(7, "general")]})
        runtime.set_reminders_channel_resolver(resolver)
        runtime.apply_config(_enabled(store_file, poll_seconds=30))
        tool = runtime.reminders.tool
        assert isinstance(tool, ReminderTool)
        tool._origin_getter = lambda: _origin()  # READBACK
        result = await tool(
            action="create", text="t", in_minutes=60, channel="#General"
        )
        assert row_for(store_file, result)

        # Order B: seat first (startup with an enabled block), hand off
        # when the interface starts — spc-166: registering the resolver
        # rebuilds the seated tool through the Runtime's single
        # construction site, so the current tool carries it from its
        # constructor, and every later construction (re-seats included)
        # takes the same path.
        store_b = tmp_path / "order-b.sqlite"
        runtime_b = _runtime(_enabled(store_b, poll_seconds=30))
        tool_b = runtime_b.reminders.tool
        assert isinstance(tool_b, ReminderTool)
        interface_b = _make_interface(tmp_path)
        await interface_b.start(FakeAgent(runtime_b))
        try:
            seated = runtime_b.reminders.tool
            assert seated is not tool_b  # rebuilt through the seam
            seated._origin_getter = lambda: _origin()  # READBACK
            result = await seated(
                action="create", text="t", in_minutes=60, channel="general"
            )
            assert row_for(store_b, result)
            # And the re-seat after registration builds WITH it too.
            runtime_b.apply_config(_enabled(store_b, poll_seconds=45))
            rebuilt = runtime_b.reminders.tool
            assert rebuilt is not seated
        finally:
            await interface_b.stop()

    async def test_named_create_through_the_constructor_resolver(self, tmp_path):
        """R1: a tool built with channel_resolver resolves a named create
        and stores the resolved destination — the constructor path alone
        (no Runtime, no seam) still works unchanged."""
        store = ReminderStore(tmp_path / "r.sqlite")
        tool = _tool(
            store,
            origin_getter=lambda: _origin(),
            channel_resolver=_resolver({"general": [(7, "general")]}),
        )
        result = await tool(
            action="create", text="t", in_minutes=60, channel="#General"
        )
        assert result == "Reminder r1 set for Thu 2026-10-15 16:00 MST in #general"
        row = store.get("r1")
        assert row.dest_channel_id == "7"
        assert row.dest_channel_name == "general"


def row_for(store_file, result: str):
    """True when the reply names r1 and the row stored channel 7."""
    assert result.startswith("Reminder r1 set for"), result
    row = ReminderStore(store_file).get("r1")
    return row.dest_channel_id == "7" and row.dest_channel_name == "general"


# ---------------------------------------------------------------------------
# R2 — the interface registers through the seam, never touches the tool
# ---------------------------------------------------------------------------


class TestR2InterfaceRegistersViaSeam:
    async def test_start_registers_the_interface_resolver_on_the_runtime(
        self, tmp_path
    ):
        """R2: ZulipInterface.start registers its resolver on the seam
        (never touching the tool): the tool the seam seats in its place —
        the startup one having been built with no resolver — resolves
        #General through the interface's real resolver and the fake
        client, and stores the destination."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        agent = FakeAgent(runtime)
        interface = _make_interface(tmp_path)
        tool = runtime.reminders.tool
        assert "reminder" in runtime.registry
        # No interface yet: the tool exists, no resolver is registered.
        tool._origin_getter = lambda: _origin()  # READBACK
        assert await tool(
            action="create", text="t", in_minutes=60, channel="general"
        ) == "Error: named channels are not available in this build"

        await interface.start(agent)  # the seam registration under test

        # spc-166: registration seats a rebuilt tool in the old one's
        # place; read the handle again instead of the pre-start object.
        seated = runtime.reminders.tool
        assert seated is not tool
        seated._origin_getter = lambda: _origin()  # READBACK
        result = await seated(
            action="create", text="t", in_minutes=60, channel="#General"
        )
        assert result.startswith("Reminder r1 set for"), result
        row = ReminderStore(store_file).get("r1")
        assert row.dest_channel_id == "7"
        assert row.dest_channel_name == "general"
        await interface.stop()


# ---------------------------------------------------------------------------
# R3 — after a live reload re-seat, named-channel create still resolves
# ---------------------------------------------------------------------------


class TestR3ResolverSurvivesReseat:
    async def test_named_create_after_disable_enable_reseat(self, tmp_path):
        """R3 (spc-159): a disabled→enabled live reload rebuilds the tool
        (spc-143 re-seat) and the rebuilt tool still carries the seam's
        resolver — a named-channel create from a Zulip turn origin
        resolves through the REAL ZulipInterface resolver (fake client)
        and stores the row."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_config(store_file))  # disabled at startup
        agent = FakeAgent(runtime)
        interface = _make_interface(tmp_path)
        await interface.start(agent)  # registers the resolver on the seam
        try:
            assert runtime.reminders is None

            # Operator flips enabled: true between messages.
            runtime.apply_config(_enabled(store_file, poll_seconds=30))
            handle = runtime.reminders
            assert handle is not None  # the tool was REBUILT by the re-seat
            await _stop_reshaped_loop(interface)

            # The Runtime-built tool keeps its own real clock (none was
            # injected); ``_now()`` reads exactly that clock, so the
            # expected reply matches what the tool itself will compute,
            # and the stored row proves the resolution.
            expected = (
                "Reminder r1 set for "
                f"{format_local_due(handle.tool._now() + timedelta(minutes=60), handle.tz)}"
                " in #general"
            )
            result = await _drive_zulip_turn(interface, agent, handle.tool)
            assert result == expected
            row = ReminderStore(store_file).get("r1")
            assert row.dest_channel_id == "7"
            assert row.dest_channel_name == "general"
            assert row.dest_topic == "reminders"
        finally:
            await interface.stop()

    async def test_named_create_after_changed_block_reseat(self, tmp_path):
        """R3 (spc-159): a reminders-block change while enabled rebuilds
        the tool; the rebuilt tool resolves a named create from a Zulip
        turn through the interface resolver and stores the row."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        agent = FakeAgent(runtime)
        interface = _make_interface(tmp_path)
        await interface.start(agent)
        try:
            tool_before = runtime.reminders.tool
            runtime.apply_config(_enabled(store_file, poll_seconds=45))
            handle = runtime.reminders
            assert handle.tool is not tool_before  # rebuilt
            await _stop_reshaped_loop(interface)

            expected = (
                "Reminder r1 set for "
                f"{format_local_due(handle.tool._now() + timedelta(minutes=60), handle.tz)}"
                " in #general"
            )
            result = await _drive_zulip_turn(interface, agent, handle.tool)
            assert result == expected
            row = ReminderStore(store_file).get("r1")
            assert row.dest_channel_id == "7"
            assert row.dest_channel_name == "general"
        finally:
            await interface.stop()


async def _stop_reshaped_loop(interface: Any) -> None:
    """Let the interface re-seat its delivery loop (spc-143 sync), then
    stop it so the file-backed loop never races the create under test.
    Tiny poll intervals and no due rows — no real waiting anywhere."""
    await interface._sync_reminders()
    loop = interface._reminders_loop
    if loop is not None:
        await asyncio.sleep(0.02)
        await loop.stop()


# ---------------------------------------------------------------------------
# R4 — no resolver supplied: behavior unchanged (FR21 refuses everything)
# ---------------------------------------------------------------------------


class TestR4NoResolverUnchanged:
    async def test_every_action_refuses_without_a_chat_origin(self, tmp_path):
        """R4 (FR21): a tool with no origin getter value (non-Zulip/
        non-Discord interface) returns the FR21 error for every action,
        resolver or not."""
        store = ReminderStore(tmp_path / "r.sqlite")
        with_origin = _tool(store, origin_getter=lambda: None)
        # Even with a resolver registered, the missing chat origin wins.
        with_resolver = _tool(
            store,
            origin_getter=lambda: None,
            channel_resolver=_resolver({"general": [(7, "general")]}),
        )
        for tool in (with_origin, with_resolver):
            assert await tool(action="create", text="t", in_minutes=5) == (
                NO_INTERFACE_ERROR
            )
            assert await tool(action="list") == NO_INTERFACE_ERROR
            assert await tool(action="cancel", id="r1") == NO_INTERFACE_ERROR
        assert store.count_by_status() == {}

    async def test_named_create_without_resolver_errors_stores_nothing(
        self, tmp_path
    ):
        """R4: with an origin but no resolver (never registered on the
        seam), a named-channel create errors and stores nothing — the
        FR21 guard for list/cancel is the origin, which is unchanged."""
        store = ReminderStore(tmp_path / "r.sqlite")
        tool = _tool(store, origin_getter=lambda: _origin())
        result = await tool(
            action="create", text="t", in_minutes=60, channel="general"
        )
        assert result == "Error: named channels are not available in this build"
        assert store.count_by_status() == {}

    async def test_runtime_without_registration_builds_a_resolverless_tool(
        self, tmp_path
    ):
        """R4: a Runtime nobody registered a resolver on still builds its
        tool (startup path unchanged) with no resolver."""
        store_file = tmp_path / "r.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        tool = runtime.reminders.tool
        assert isinstance(tool, ReminderTool)
        tool._origin_getter = lambda: _origin()  # READBACK
        result = await tool(
            action="create", text="t", in_minutes=60, channel="general"
        )
        assert result == "Error: named channels are not available in this build"


# ---------------------------------------------------------------------------
# R1/R2 (spc-166) — registering a resolver rebuilds the seated tool
# ---------------------------------------------------------------------------


class TestR1RegistrationRebuildsTheSeatedTool:
    async def test_r1_registration_rebuilds_through_the_construction_site(
        self, tmp_path
    ):
        """R1: with reminders enabled and the tool seated, registering a
        resolver rebuilds it through Runtime._reminders_for — the existing
        single construction site — and the handle's ``tool`` and the tool
        registered under ``reminder`` in the registry both point at the
        new one (the registry re-registration by name replaces the old
        callable, so the next lookup — the next model turn — runs it)."""
        store_file = tmp_path / "rebuild.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        seated = runtime.reminders.tool
        handle_before = runtime.reminders
        seq_before = runtime.reminders_seq
        resolver = _resolver({"general": [(7, "general")]})

        seen: list[Any] = []
        real_for = runtime._reminders_for

        def spy(config: Any) -> None:
            seen.append(config)
            real_for(config)

        runtime._reminders_for = spy  # type: ignore[method-assign]
        runtime.set_reminders_channel_resolver(resolver)

        assert seen == [runtime.config]  # rebuilt through that one site
        rebuilt = runtime.reminders.tool
        assert rebuilt is not seated
        assert runtime.reminders_seq == seq_before + 1
        # A rebuilt handle; the store is a fresh connection on the SAME
        # path, so the row the new tool writes is the loop's own table.
        assert runtime.reminders.store is not handle_before.store
        assert runtime.reminders.poll_seconds == handle_before.poll_seconds
        assert runtime.reminders.agent_name == handle_before.agent_name
        # The registry entry is the rebuilt tool too: registration by name
        # replaced the old callable, so the next lookup — the next model
        # turn — executes the tool that carries the resolver.
        assert runtime.registry._tools["reminder"].fn.__self__ is rebuilt
        rebuilt._origin_getter = lambda: _origin()  # READBACK
        result = await rebuilt(
            action="create", text="t", in_minutes=60, channel="general"
        )
        assert row_for(store_file, result)

    async def test_r1_registered_tool_carries_the_resolver_from_the_ctor(
        self, tmp_path
    ):
        """R1: the rebuild hands the resolver over through the ctor alone
        — the old seat stays resolverless, and calling the tool registered
        under ``reminder`` (what a model turn executes) resolves."""
        store_file = tmp_path / "ctor.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        seated = runtime.reminders.tool
        resolver = _resolver({"general": [(7, "general")]})
        runtime.set_reminders_channel_resolver(resolver)

        # The old tool was never touched: it still has no resolver.
        seated._origin_getter = lambda: _origin()  # READBACK
        assert await seated(
            action="create", text="t", in_minutes=60, channel="general"
        ) == "Error: named channels are not available in this build"

        spec = runtime.registry._tools["reminder"]
        tool = spec.fn.__self__  # READBACK
        assert tool is not seated
        tool._origin_getter = lambda: _origin()  # READBACK
        assert row_for(store_file, await tool(action="create", text="t", in_minutes=60, channel="general"))


class TestR2RegistrationEdges:
    def test_r2_registration_does_not_restart_the_delivery_loop(self, tmp_path):
        """R2: registering a resolver moves the tool but does NOT bump
        ``reminders_stamped`` (``reminders_seq`` is the only thing that
        moves), so an owning interface's per-message sync sees no reason
        to restart a running delivery loop."""
        store_file = tmp_path / "stamp.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        stamp_before = runtime.reminders_stamped
        runtime.set_reminders_channel_resolver(_resolver({}))
        assert runtime.reminders_stamped == stamp_before
        # The interface's steady-state check compares exactly this int.
        agent = FakeAgent(runtime)
        assert agent.reminders_stamped == stamp_before

    async def test_r2_no_seated_tool_builds_nothing(self, tmp_path):
        """R2: with reminders disabled there is no tool to rebuild —
        registration only stores the resolver and builds nothing."""
        store_file = tmp_path / "disabled.sqlite"
        runtime = _runtime(_config(store_file))  # disabled: nothing seated
        assert runtime.reminders is None
        seq_before = runtime.reminders_seq
        resolver = _resolver({"general": [(7, "general")]})
        runtime.set_reminders_channel_resolver(resolver)
        assert runtime.reminders is None  # still nothing built
        assert runtime.reminders_seq == seq_before
        assert runtime._reminders_channel_resolver is resolver  # READBACK
        # And the seat that arrives later is born with it.
        runtime.apply_config(_enabled(store_file, poll_seconds=30))
        tool = runtime.reminders.tool
        tool._origin_getter = lambda: _origin()  # READBACK
        assert row_for(store_file, await tool(action="create", text="t", in_minutes=60, channel="general"))

    def test_r2_unknown_platform_is_ignored_without_a_rebuild(self, tmp_path, caplog):
        """R2 (FR11): a resolver from an unknown platform is still ignored
        with the existing warning, and the seated tool is left alone."""
        store_file = tmp_path / "unknown.sqlite"
        runtime = _runtime(_enabled(store_file, poll_seconds=30))
        seated = runtime.reminders.tool
        seq_before = runtime.reminders_seq

        async def foreign(platform: str, name: str):  # no seam marker
            return []

        with caplog.at_level("WARNING"):
            runtime.set_reminders_channel_resolver(foreign)

        assert "Ignored a reminders channel resolver from unknown platform" in (
            caplog.text
        )
        assert runtime.reminders.tool is seated  # not rebuilt
        assert runtime.reminders_seq == seq_before
        assert runtime._reminders_channel_resolver is None  # READBACK

        class Unmarked:
            pass

        runtime.set_reminders_channel_resolver(Unmarked())
        assert runtime.reminders.tool is seated
        assert runtime._reminders_channel_resolver is None  # READBACK
