"""Schedules — dual-platform delivery parity on Zulip (spc-997, FR47).

Direct-profile unit tests for AC47, case rows AC47-r1..r6. Everything is
faked at the declared seams: the Zulip client is a recording fake (no
server, no credentials, no network), the agent dispatch runner is a fake
returning a canned envelope result, clocks are injected or pinned by
READBACK, and the store directory is a per-test temp dir. Command payloads
run real short shell commands so the FR26 forms come from the real
executor.

The seam tests drive a real Agent over a real Runtime (which builds the
``schedule`` tool) through the real ``ZulipInterface`` start and turn
paths, so the wiring between interface, tool, store and engine is the
production one (spc-140/141: fake-only tests can miss wiring shapes).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
import yaml

import spctr.schedules.engine as engine_mod
from spctr.config import SchedulesConfig
from spctr.reminders import tool as reminders_tool
from spctr.schedules import AgentResult, ScheduleEngine, ScheduleStore
from spctr.tools.schedule_tool import (
    ScheduleOrigin,
    ScheduleTool,
    ScheduleTurnScope,
    schedule_turn_scope,
)

UTC = timezone.utc
PHX = ZoneInfo("America/Phoenix")
# Monday 2030-01-07 07:00 MST: far enough ahead of the wall clock that a
# tool-created row is never due for an engine running on the real clock.
NOW = datetime(2030, 1, 7, 14, 0, 0, tzinfo=UTC)

# The Zulip streams the fake server knows (FR10 names, exact match).
STREAMS = [
    {"stream_id": 42, "name": "spctr", "can_send": True},
    {"stream_id": 7, "name": "general", "can_send": True},
    {"stream_id": 55, "name": "dev-notes", "can_send": True},
    {"stream_id": 56, "name": "dev-notes-old", "can_send": True},
]


@pytest.fixture(autouse=True)
def _clean_scopes():
    """Turn scopes are ContextVars; start and end each test empty."""
    schedule_turn_scope.set(None)
    reminders_tool.turn_scope.set(None)
    yield
    schedule_turn_scope.set(None)
    reminders_tool.turn_scope.set(None)


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    return yaml.safe_load(text[4 : text.index("\n---", 4)])


def _write_file(directory: Path, stem: str, body: str = "", **keys: Any) -> Path:
    """A hand-authored schedule file in the FR37 format."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.md"
    head = yaml.safe_dump(keys, sort_keys=False, allow_unicode=True)
    path.write_text("---\n" + head + "---\n\n" + body, encoding="utf-8")
    return path


def _agent_config(schedules: dict) -> Any:
    """A minimal real SpctrConfig for Runtime/Agent construction."""
    from spctr.config import AgentConfig, ModelConfig, SpctrConfig

    return SpctrConfig(
        agent=AgentConfig(name="miranda", default_skills=False, schedules=schedules),
        models={"default": ModelConfig(model="test-model")},
        pipeline={"layers": []},
        provenance_store_path="",
        memory_store_dsn="",
    )


def _make_agent(config: Any) -> Any:
    """A real Agent with the documented LLM seam patched (no model call)."""
    from spctr.agent import Agent

    llm = object.__new__(type("FakeLLM", (), {}))
    with patch("spctr.live_llm.LiteLLMCallable", return_value=llm):
        return Agent(config=config)


class _FakeClient:
    """The Zulip transport: registration snapshot, stream list, and a
    record of every message send (kwargs as the interface passed them)."""

    def __init__(self) -> None:
        self.sends: list[dict] = []

    async def register(self):
        return {
            "queue_id": "q-test",
            "last_event_id": 0,
            "realm_users": [{"user_id": 9, "is_bot": True}],
            "realm_bots": [],
            "streams": [
                {"stream_id": s["stream_id"], "name": s["name"]} for s in STREAMS
            ],
        }

    async def get_streams(self):
        return {"streams": [dict(s) for s in STREAMS]}

    async def send_message(self, **kwargs):
        self.sends.append(kwargs)
        return len(self.sends)

    async def update_message(self, message_id, content):
        return None

    async def aclose(self):
        pass


def _make_interface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A real ZulipInterface over a fake client (installed by monkeypatch,
    restored at teardown). Returns (interface, client)."""
    import spctr.interfaces.zulip.interface as iface_mod
    from spctr.interfaces.zulip.client import ZulipCredentials
    from spctr.interfaces.zulip.config import ZulipConfig

    client = _FakeClient()
    monkeypatch.setattr(
        iface_mod,
        "load_zulip_credentials",
        lambda path: ZulipCredentials(
            email="bot@test.local", api_key="k", site="http://127.0.0.1:1"
        ),
    )
    monkeypatch.setattr(iface_mod, "ZulipClient", lambda **kw: client)
    config = ZulipConfig(
        zuliprc=str(tmp_path / "zuliprc"),
        users=[8],
        channels=[42, 7],
        channel_areas={"42": "", "7": ""},
    )
    return iface_mod.ZulipInterface(config), client


class _ToolSession:
    """A Session whose turn calls the schedule tool instead of a model.

    The call runs inside ``ZulipInterface._run_turn``'s task, so the tool
    reads the turn scope that real path set. ``gate`` (optional) holds the
    call until released, to overlap two turns.
    """

    def __init__(self, tool: ScheduleTool, gate: asyncio.Event | None = None, **kwargs):
        self._tool = tool
        self._gate = gate
        self._kwargs = kwargs
        self.results: list[str] = []

    async def stream_message(self, content: str):
        if self._gate is not None:
            await self._gate.wait()
        self.results.append(await self._tool(**self._kwargs))
        return
        yield  # pragma: no cover - makes this an async generator


def _message(sender_id: int, topic: str = "plan") -> dict:
    return {"sender_id": sender_id, "subject": topic, "content": "schedule it"}


def _zulip_origin(default_topic: str = "schedules") -> ScheduleOrigin:
    return ScheduleOrigin(
        agent="miranda",
        platform="zulip",
        requester_id="8",
        requester_mention="@8",
        channel_id="42",
        channel_name="spctr",
        default_topic=default_topic,
    )


def _discord_origin() -> ScheduleOrigin:
    return ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")


def _tool(store: ScheduleStore, resolver: Any = None) -> ScheduleTool:
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


def _schedules_block(store_dir: Path, **extra: Any) -> dict:
    block = {
        "enabled": True,
        "timezone": "America/Phoenix",
        "store_path": str(store_dir),
        "poll_seconds": 999,
    }
    block.update(extra)
    return block


# ---------------------------------------------------------------------------
# AC47-r1 / r4: create from a Zulip turn, then list — real interface seam
# ---------------------------------------------------------------------------


class TestZulipCreateSeam:
    async def test_create_resolves_stream_and_list_names_it(self, monkeypatch, tmp_path):
        """AC47-r1 + r4 through the real interface: start() starts a
        Zulip-scoped engine and seats the stream resolver on the Runtime's
        tool; a create inside a real Zulip turn resolves "dev-notes" to
        stream 55 by exact name and the file carries ``platform: zulip``,
        ``topic: schedules`` (the config default) and the resolved ID.
        ``list`` in a Zulip turn names the stream and shows the topic."""
        store_dir = tmp_path / "schedules"
        agent = _make_agent(_agent_config(_schedules_block(store_dir)))
        try:
            assert SchedulesConfig().zulip_topic == "schedules"
            interface, client = _make_interface(monkeypatch, tmp_path)
            await interface.start(agent)
            try:
                engine = interface._schedules_engine
                assert isinstance(engine, ScheduleEngine)
                assert engine._platform == "zulip"
                assert engine._schedule_store is agent.schedules.store
                tool = agent.schedules.tool
                assert tool._engine is engine
                tool._now_fn = lambda: NOW  # READBACK: the Runtime seat runs the real clock

                session = _ToolSession(
                    tool, action="create", cron="0 7 * * *", type="command",
                    command="echo hi", channel="#Dev-Notes",
                )
                await interface._run_turn(session, 42, "plan", _message(8))
                assert session.results == [
                    "Schedule s1 set: Tue 07:00 MST in #dev-notes > schedules "
                    "(command: echo hi)"
                ]
                keys = _frontmatter(store_dir / "s1.md")
                assert keys["platform"] == "zulip"
                assert keys["channel"] == "55"
                assert keys["topic"] == "schedules"
                assert keys["requester_id"] == "8"
                row = agent.schedules.store.get("s1")
                assert (row.platform, row.dest_channel_id, row.topic) == (
                    "zulip", "55", "schedules",
                )

                listing = _ToolSession(tool, action="list")
                await interface._run_turn(listing, 42, "plan", _message(8))
                assert listing.results == [
                    "s1 Tue 2030-01-08 07:00 MST cron=0 7 * * * "
                    "#dev-notes > schedules command: echo hi [never run]"
                ]
                # The scope is the turn's own: nothing leaks after it.
                assert schedule_turn_scope.get() is None
            finally:
                await interface.stop()
            assert interface._schedules_engine is None
        finally:
            await agent.aclose()

    async def test_concurrent_turns_keep_their_own_origin(self, monkeypatch, tmp_path):
        """Delivery seam constraint: two overlapping Zulip turns in
        different streams each create with their own origin stream and
        requester (the per-turn scope, not the interface lock)."""
        store_dir = tmp_path / "schedules"
        agent = _make_agent(_agent_config(_schedules_block(store_dir)))
        try:
            interface, _client = _make_interface(monkeypatch, tmp_path)
            await interface.start(agent)
            try:
                tool = agent.schedules.tool
                tool._now_fn = lambda: NOW  # READBACK
                gate = asyncio.Event()
                kwargs = dict(action="create", cron="0 7 * * *", type="command")
                first = _ToolSession(tool, gate, label="first", command="echo 1", **kwargs)
                second = _ToolSession(tool, gate, label="second", command="echo 2", **kwargs)
                turns = asyncio.gather(
                    interface._run_turn(first, 42, "plan", _message(8)),
                    interface._run_turn(second, 7, "ops", _message(11)),
                )
                await asyncio.sleep(0.05)  # both turns now wait on the gate
                gate.set()
                await turns
                rows = {
                    json.loads(r.payload)["command"]: r
                    for r in agent.schedules.store.list_for_agent("miranda")
                }
                assert (rows["echo 1"].dest_channel_id, rows["echo 1"].requester_id) == ("42", "8")
                assert (rows["echo 2"].dest_channel_id, rows["echo 2"].requester_id) == ("7", "11")
                assert rows["echo 1"].topic == rows["echo 2"].topic == "schedules"
            finally:
                await interface.stop()
        finally:
            await agent.aclose()

    async def test_disabled_starts_no_engine(self, monkeypatch, tmp_path):
        """FR46: schedules disabled registers no tool and starts no engine."""
        agent = _make_agent(_agent_config({"enabled": False}))
        try:
            assert "schedule" not in agent.registry
            interface, _client = _make_interface(monkeypatch, tmp_path)
            await interface.start(agent)
            try:
                assert interface._schedules_engine is None
            finally:
                await interface.stop()
        finally:
            await agent.aclose()


# ---------------------------------------------------------------------------
# AC47-r1 / r2: tool-level Zulip create branch
# ---------------------------------------------------------------------------


class TestZulipCreateTool:
    async def test_explicit_topic_is_stored(self, tmp_path):
        """AC47-r2: an explicit ``topic: standup`` is the stored topic."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        tool = _tool(store)
        schedule_turn_scope.set(ScheduleTurnScope(origin=_zulip_origin()))
        result = await tool(
            action="create", cron="0 9 * * 1", type="command",
            command="echo standup", topic="standup",
        )
        assert result == (
            "Schedule s1 set: Mon 09:00 MST in #spctr > standup "
            "(command: echo standup)"
        )
        keys = _frontmatter(tmp_path / "schedules" / "s1.md")
        assert (keys["platform"], keys["channel"], keys["topic"]) == ("zulip", "42", "standup")
        assert store.get("s1").topic == "standup"

    async def test_default_topic_comes_from_the_origin(self, tmp_path):
        """AC47-r1: with no ``topic`` the origin's configured default is
        stored (here a non-default ``schedules.zulip_topic`` value)."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        tool = _tool(store)
        schedule_turn_scope.set(ScheduleTurnScope(origin=_zulip_origin("ops-board")))
        await tool(action="create", cron="0 9 * * 1", type="command", command="echo x")
        assert _frontmatter(tmp_path / "schedules" / "s1.md")["topic"] == "ops-board"

    async def test_discord_create_stores_no_topic(self, tmp_path):
        """A Discord create is unchanged: no ``topic`` key, even if given."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        tool = _tool(store)
        schedule_turn_scope.set(ScheduleTurnScope(origin=_discord_origin()))
        result = await tool(
            action="create", cron="0 9 * * 1", type="command",
            command="echo x", topic="standup",
        )
        assert result == "Schedule s1 set: Mon 09:00 MST in #general (command: echo x)"
        assert "topic" not in _frontmatter(tmp_path / "schedules" / "s1.md")
        assert store.get("s1").topic == ""


# ---------------------------------------------------------------------------
# AC47-r3: the Zulip engine posts FR26/FR27 forms to stream + topic
# ---------------------------------------------------------------------------


class TestZulipEnginePosts:
    async def test_fr26_fr27_forms_land_in_stream_and_topic(self, monkeypatch, tmp_path):
        """AC47-r3 through the real interface engine: due command and
        agent schedules fire on start (FR30 catch-up). Command exit 0
        posts nothing; non-zero and timeout post the one-line ⚠️ form;
        agent ``done`` posts the final message verbatim split at 2000
        characters; agent failure posts the ⚠️ form. Every post goes to
        the row's stream and topic; a Zulip file without ``topic`` posts
        to ``schedules.zulip_topic``; the Discord file never posts here."""
        store_dir = tmp_path / "schedules"
        past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
        common = dict(cron="0 7 * * *", platform="zulip", channel="55")
        _write_file(store_dir, "cmd-ok", label="cmd-ok", type="command",
                    command="true", topic="ops", next_run_at=past, **common)
        _write_file(store_dir, "cmd-fail", label="cmd-fail", type="command",
                    command="echo boom >&2; exit 3", topic="ops",
                    next_run_at=past, **common)
        _write_file(store_dir, "cmd-timeout", label="cmd-timeout", type="command",
                    # ``exec``: the shell becomes the sleep, so the timeout
                    # kill closes the pipes (a forked grandchild would hold
                    # them for the full sleep).
                    command="exec sleep 30", topic="ops", next_run_at=past, **common)
        _write_file(store_dir, "cmd-default", label="cmd-default", type="command",
                    command="echo nope >&2; exit 1", next_run_at=past, **common)
        _write_file(store_dir, "agent-ok", "write the report", label="agent-ok",
                    type="agent", topic="reports", next_run_at=past, **common)
        _write_file(store_dir, "agent-fail", "fail please", label="agent-fail",
                    type="agent", topic="reports", next_run_at=past, **common)
        _write_file(store_dir, "discord-row", label="discord-row", type="command",
                    command="echo discord >&2; exit 1", cron="0 7 * * *",
                    platform="discord", channel="100", next_run_at=past)

        final = "".join(chr(97 + i % 26) for i in range(2500))

        async def fake_dispatch(target, instruction, *, timeout, agent, clock):
            status = "done" if instruction == "write the report" else "failed"
            return AgentResult(
                status=status, message=final if status == "done" else "",
                exit_code=0, duration_s=0.1, log_path="",
            )

        monkeypatch.setattr(engine_mod, "run_agent_dispatch", fake_dispatch)
        agent = _make_agent(_agent_config(_schedules_block(
            store_dir, command_timeout_seconds=1, zulip_topic="digest",
        )))
        try:
            interface, client = _make_interface(monkeypatch, tmp_path)
            await interface.start(agent)
            try:
                await asyncio.wait_for(interface._schedules_start_task, 20)
            finally:
                await interface.stop()
        finally:
            await agent.aclose()

        posts = [(s["type"], s["to"], s["topic"], s["content"]) for s in client.sends]
        # Catch-up runs due rows in (next_run_at, id) order.
        assert posts == [
            ("stream", 55, "reports", "⚠️ Schedule agent-fail (agent-fail) failed: "
             "agent dispatch returned status 'failed'"),
            ("stream", 55, "reports", final[:2000]),
            ("stream", 55, "reports", final[2000:]),
            ("stream", 55, "digest", "⚠️ Schedule cmd-default (cmd-default) failed: nope"),
            ("stream", 55, "ops", "⚠️ Schedule cmd-fail (cmd-fail) failed: boom"),
            ("stream", 55, "ops", "⚠️ Schedule cmd-timeout (cmd-timeout) failed: "
             "command timed out"),
        ]
        store = ScheduleStore(store_dir, agent="miranda")
        assert store.get("cmd-ok").last_status == "ok"
        assert store.get("cmd-fail").topic == "ops"  # preserved across the rewrite
        discord = store.get("discord-row")
        assert discord.last_status is None
        assert discord.next_run_at.isoformat() == past


# ---------------------------------------------------------------------------
# AC47-r4: list from the Discord side over the same directory
# ---------------------------------------------------------------------------


class TestListFromDiscord:
    async def test_discord_list_shows_zulip_stream_and_topic(self, tmp_path):
        """AC47-r4: a Discord-side tool (no Zulip stream directory) lists a
        Zulip row with its stored stream ID and its topic; Discord rows
        keep the increment-2 form."""
        store_dir = tmp_path / "schedules"
        zulip_store = ScheduleStore(store_dir, agent="miranda")
        schedule_turn_scope.set(ScheduleTurnScope(origin=_zulip_origin()))
        await _tool(zulip_store)(
            action="create", cron="0 7 * * *", type="command",
            command="echo z", topic="standup",
        )
        schedule_turn_scope.set(ScheduleTurnScope(origin=_discord_origin()))
        discord_tool = _tool(ScheduleStore(store_dir, agent="miranda"))
        await discord_tool(action="create", cron="0 8 * * *", type="command", command="echo d")
        out = await discord_tool(action="list")
        assert out == (
            "s2 Mon 2030-01-07 08:00 MST cron=0 8 * * * "
            "#100 command: echo d [never run]\n"
            "s1 Tue 2030-01-08 07:00 MST cron=0 7 * * * "
            "#42 > standup command: echo z [never run]"
        )


# ---------------------------------------------------------------------------
# AC47-r5: two engines over one directory never cross platforms
# ---------------------------------------------------------------------------


class _Sends:
    def __init__(self) -> None:
        self.posts: list[tuple[str, str]] = []

    async def send(self, channel_id: str, content: str) -> None:
        self.posts.append((channel_id, content))


def _engine(store: ScheduleStore, sends: _Sends, platform: str) -> ScheduleEngine:
    return ScheduleEngine(
        store, sends.send, poll_seconds=999, tz=PHX, agent_name="miranda",
        command_timeout=10, platform=platform, clock=lambda: NOW,
    )


class TestTwoEnginesOneDirectory:
    async def test_due_claim_catch_up_and_recovery_are_platform_scoped(self, tmp_path):
        """AC47-r5: with a discord file and a zulip file of each kind
        (poll-due, overdue for catch-up, stuck at the claim sentinel), each
        engine's due/claim/catch-up/recovery touches only its platform."""
        store_dir = tmp_path / "schedules"
        due = (NOW - timedelta(seconds=30)).isoformat()
        overdue = (NOW - timedelta(hours=3)).isoformat()
        sentinel = "9999-12-31T23:59:59+00:00"

        def write(stem, platform, channel, next_run_at):
            _write_file(store_dir, stem, label=stem, cron="0 7 * * *",
                        type="command", command=f"echo {stem} >&2; exit 1",
                        platform=platform, channel=channel, next_run_at=next_run_at)

        write("d-poll", "discord", "100", due)
        write("z-poll", "zulip", "55", due)
        store = ScheduleStore(store_dir, agent="miranda")
        zulip_sends, discord_sends = _Sends(), _Sends()
        zulip = _engine(store, zulip_sends, "zulip")
        discord = _engine(ScheduleStore(store_dir, agent="miranda"), discord_sends, "discord")

        # Due and claim queries are scoped.
        assert [r.id for r in store.due(NOW, "zulip")] == ["z-poll"]
        assert [r.id for r in store.due(NOW, "discord")] == ["d-poll"]
        assert store.claim("d-poll", "zulip") is False
        assert store.claim("z-poll", "discord") is False
        d_bytes = (store_dir / "d-poll.md").read_bytes()

        await zulip._poll_once()
        assert zulip_sends.posts == [("55", "⚠️ Schedule z-poll (z-poll) failed: z-poll")]
        assert (store_dir / "d-poll.md").read_bytes() == d_bytes
        await discord._poll_once()
        assert discord_sends.posts == [("100", "⚠️ Schedule d-poll (d-poll) failed: d-poll")]
        assert len(zulip_sends.posts) == 1

        # Catch-up and sentinel recovery at start are scoped too.
        write("d-late", "discord", "100", overdue)
        write("z-late", "zulip", "55", overdue)
        write("d-stuck", "discord", "100", sentinel)
        write("z-stuck", "zulip", "55", sentinel)
        d_late = (store_dir / "d-late.md").read_bytes()
        d_stuck = (store_dir / "d-stuck.md").read_bytes()
        zulip_sends.posts.clear()
        await zulip.start()
        await zulip.stop()
        assert zulip_sends.posts == [("55", "⚠️ Schedule z-late (z-late) failed: z-late")]
        assert store.get("z-stuck").last_error == "recovered from crash"
        assert (store_dir / "d-late.md").read_bytes() == d_late
        assert (store_dir / "d-stuck.md").read_bytes() == d_stuck

        discord_sends.posts.clear()
        await discord.start()
        await discord.stop()
        assert discord_sends.posts == [("100", "⚠️ Schedule d-late (d-late) failed: d-late")]
        assert store.get("d-stuck").last_error == "recovered from crash"
        assert zulip_sends.posts == [("55", "⚠️ Schedule z-late (z-late) failed: z-late")]


# ---------------------------------------------------------------------------
# AC47-r6: interfaces with neither platform keep the FR34 text verbatim
# ---------------------------------------------------------------------------


class TestNeitherPlatform:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(action="create", cron="0 7 * * *", type="command", command="echo x"),
            dict(action="list"),
            dict(action="cancel", id="s1"),
            dict(action="run_now", id="s1"),
        ],
    )
    async def test_every_action_gets_the_fr34_text(self, tmp_path, kwargs):
        """AC47-r6: with no Discord or Zulip turn scope every action
        returns exactly the increment-2 FR34 error and stores nothing."""
        store_dir = tmp_path / "schedules"
        tool = _tool(ScheduleStore(store_dir, agent="miranda"))
        assert await tool(**kwargs) == "Error: Schedules need the Discord interface"
        assert list(store_dir.glob("*.md")) == []
