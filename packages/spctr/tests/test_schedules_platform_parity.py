"""spc-1098: one shared platform-engine seam for both chat interfaces.

The Zulip and Discord interfaces once carried near-identical copies of the
_start/_stop/_sync_schedules bodies. The shared construction, stop and
re-seat logic now lives in spctr.schedules.engine. These tests pin the
consolidation itself: both interfaces route through the one shared helper
with their own send seam and platform value, and the shared stop/sync
edges behave as the per-interface copies did. Behavioural coverage of the
engines themselves stays in test_schedules_{core,manage,agent,zulip}.py
and test_runtime.py.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


def _fake_schedules(tool: object | None = None) -> SimpleNamespace:
    """A schedules handle shaped like the Runtime's, with an async store."""
    store = SimpleNamespace()

    async def recover_claimed_async(*args, **kwargs):
        return None

    store.recover_claimed_async = recover_claimed_async
    return SimpleNamespace(
        store=store,
        poll_seconds=1.0,
        tz="UTC",
        agent_name="parity-test",
        command_timeout=30.0,
        agent_timeout=300.0,
        tool=tool,
    )


class TestSharedRouting:
    """Both interfaces reference the one shared helper, not private copies."""

    def test_interfaces_route_through_the_one_shared_helper(self):
        import spctr.interfaces.discord.interface as discord_mod
        import spctr.interfaces.zulip.interface as zulip_mod
        from spctr import schedules

        for name in (
            "start_platform_engine",
            "stop_platform_engine",
            "sync_platform_engine",
        ):
            assert getattr(zulip_mod, name) is getattr(schedules, name)
            assert getattr(discord_mod, name) is getattr(schedules, name)

    async def test_zulip_start_routes_with_zulip_seams(self, monkeypatch, tmp_path):
        from spctr.interfaces.zulip.config import ZulipConfig
        from spctr.interfaces.zulip.interface import ZulipInterface

        calls: dict = {}
        dummy_engine = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())

        def fake_start(schedules, *, send, platform, channel_resolver=None,
                       channel_namer=None):
            calls["send"] = send
            calls["platform"] = platform
            calls["resolver"] = channel_resolver
            calls["namer"] = channel_namer
            return dummy_engine

        monkeypatch.setattr(
            "spctr.interfaces.zulip.interface.start_platform_engine",
            fake_start,
        )
        interface = ZulipInterface(ZulipConfig(
            zuliprc=str(tmp_path / "zuliprc"),
            users=[8],
            channels=[42, 7],
            channel_areas={"42": "", "7": ""},
        ))
        interface._client = MagicMock()
        interface._client.send_message = AsyncMock()
        interface._start_schedules(
            SimpleNamespace(schedules=_fake_schedules(), schedules_stamped=7)
        )
        assert calls["platform"] == "zulip"
        assert calls["resolver"] is not None
        assert calls["namer"] is not None
        assert interface._schedules_engine is dummy_engine
        assert interface._schedules_stamp == 7

        # The send seam still posts through the interface's Zulip client
        # to the row's stream and topic.
        row = SimpleNamespace(dest_channel_id="42", topic="standup")
        await calls["send"](row, "hello")
        interface._client.send_message.assert_awaited_once_with(
            type="stream", to=42, topic="standup", content="hello"
        )

    async def test_discord_start_routes_with_discord_seams(self, monkeypatch):
        from spctr.interfaces.discord.config import DiscordConfig
        from spctr.interfaces.discord.interface import DiscordInterface

        calls: dict = {}
        dummy_engine = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())

        def fake_start(schedules, *, send, platform, channel_resolver=None,
                       channel_namer=None):
            calls["send"] = send
            calls["platform"] = platform
            calls["resolver"] = channel_resolver
            calls["namer"] = channel_namer
            return dummy_engine

        monkeypatch.setattr(
            "spctr.interfaces.discord.interface.start_platform_engine",
            fake_start,
        )
        interface = DiscordInterface(config=DiscordConfig(token="test"))
        interface._adapter = MagicMock()
        interface._adapter.try_send_message = AsyncMock(
            return_value=(None, None)
        )
        interface._start_schedules(
            SimpleNamespace(schedules=_fake_schedules(), schedules_stamped=3)
        )
        assert calls["platform"] == "discord"
        assert calls["resolver"] is not None
        assert calls["namer"] is None
        assert interface._schedules_engine is dummy_engine
        assert interface._schedules_stamp == 3

        # The send seam still posts through the interface's Discord
        # adapter, channel-scoped (no topic).
        row = SimpleNamespace(dest_channel_id="7", topic="")
        await calls["send"](row, "hi")
        interface._adapter.try_send_message.assert_awaited_once_with(7, "hi")

    async def test_start_without_a_schedules_handle_registers_nothing(
        self, tmp_path
    ):
        from spctr.interfaces.discord.config import DiscordConfig
        from spctr.interfaces.discord.interface import DiscordInterface
        from spctr.interfaces.zulip.config import ZulipConfig
        from spctr.interfaces.zulip.interface import ZulipInterface

        discord_interface = DiscordInterface(config=DiscordConfig(token="test"))
        discord_interface._adapter = MagicMock()
        zulip_interface = ZulipInterface(ZulipConfig(
            zuliprc=str(tmp_path / "zuliprc"),
            users=[8],
            channels=[42, 7],
            channel_areas={"42": "", "7": ""},
        ))
        for interface in (zulip_interface, discord_interface):
            interface._start_schedules(SimpleNamespace())
            assert interface._schedules_engine is None
            assert interface._schedules_stamp is None


class TestSharedStopAndSyncEdges:
    """The shared stop/sync bodies keep the edges the copies had."""

    async def test_stop_swallows_cancellation_and_tolerates_none(self):
        from spctr.schedules import stop_platform_engine

        calls: list = []

        class FlakyEngine:
            async def stop(self):
                calls.append("stopped")
                raise asyncio.CancelledError()

        await stop_platform_engine(FlakyEngine())
        assert calls == ["stopped"]
        await stop_platform_engine(None)
        assert calls == ["stopped"]

    async def test_sync_edges_match_the_per_interface_copies(self):
        from spctr.schedules import sync_platform_engine

        class Recorder:
            def __init__(self) -> None:
                self.events: list = []

            async def stop(self) -> None:
                self.events.append("stop")

            def start(self, agent) -> None:
                self.events.append(("start", agent))

        def run(schedules_handle, runtime_stamp, stamp, has_engine):
            recorder = Recorder()
            agent = SimpleNamespace(schedules=schedules_handle)
            runtime = SimpleNamespace(schedules_stamped=runtime_stamp)
            return recorder, agent, runtime

        # Same stamp: no-op.
        rec, agent, runtime = run(object(), 5, 5, True)
        await sync_platform_engine(
            agent, runtime=runtime, stamp=5, has_engine=True,
            stop=rec.stop, start=rec.start,
        )
        assert rec.events == []

        # Changed stamp: stop, then re-start on the live handle.
        rec, agent, runtime = run(object(), 6, 5, True)
        await sync_platform_engine(
            agent, runtime=runtime, stamp=5, has_engine=True,
            stop=rec.stop, start=rec.start,
        )
        assert rec.events == ["stop", ("start", agent)]

        # Never started (stamp None) with a live handle: start.
        rec, agent, runtime = run(object(), None, None, False)
        await sync_platform_engine(
            agent, runtime=runtime, stamp=None, has_engine=False,
            stop=rec.stop, start=rec.start,
        )
        assert rec.events == [("start", agent)]

        # Changed stamp but the handle is gone (disabled): stop only.
        rec, agent, runtime = run(None, 9, 5, True)
        await sync_platform_engine(
            agent, runtime=runtime, stamp=5, has_engine=True,
            stop=rec.stop, start=rec.start,
        )
        assert rec.events == ["stop"]

        # Never started, no handle, no runtime: nothing at all.
        rec, agent, runtime = run(None, None, None, False)
        await sync_platform_engine(
            agent, runtime=runtime, stamp=None, has_engine=False,
            stop=rec.stop, start=rec.start,
        )
        assert rec.events == []
