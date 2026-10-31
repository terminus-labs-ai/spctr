"""Zulip event queue expiry: register with a long idle timeout, re-register
on BAD_EVENT_QUEUE_ID.

The server collects a queue nobody polls for its idle timeout (10 minutes by
default, https://zulip.com/api/register-queue). Turns run inside the poll
loop, so a long turn starved the queue and every later poll failed forever.

Driven against a loopback Zulip endpoint with the REAL ``ZulipClient`` and
interfaces, a synthetic zuliprc and no realm.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web

from sr2_spectre.interfaces.zulip import ZulipConfig, ZulipDMInterface, ZulipInterface
from sr2_spectre.interfaces.zulip.client import (
    ZulipClient,
    ZulipCredentials,
    ZulipQueueExpiredError,
    ZulipTransportError,
)

DIEGO_ID = 8
BOT_ID = 9


class _Realm:
    """Loopback endpoint: each register hands out a new queue; polls on any
    queue in ``expired`` answer BAD_EVENT_QUEUE_ID."""

    def __init__(self) -> None:
        self.register_forms: list[dict[str, str]] = []
        self.polled: list[str] = []
        self.expired: set[str] = set()
        self.on_poll: Any = None

    async def register(self, request: web.Request) -> web.Response:
        self.register_forms.append(dict(await request.post()))
        return web.json_response({
            "result": "success",
            "queue_id": f"q-{len(self.register_forms)}",
            "last_event_id": 0,
            "user_id": BOT_ID,
            "full_name": "Miranda",
            "realm_users": [
                {"user_id": DIEGO_ID, "is_bot": False},
                {"user_id": BOT_ID, "is_bot": True},
            ],
        })

    async def events(self, request: web.Request) -> web.Response:
        queue_id = request.query["queue_id"]
        self.polled.append(queue_id)
        if self.on_poll is not None:
            self.on_poll(queue_id)
        if queue_id == "q-bad-request":
            return web.json_response({
                "result": "error", "msg": "Bad request", "code": "BAD_REQUEST",
            }, status=400)
        if queue_id in self.expired:
            return web.json_response({
                "result": "error",
                "msg": f"Bad event queue ID: {queue_id}",
                "code": "BAD_EVENT_QUEUE_ID",
                "queue_id": queue_id,
            }, status=400)
        return web.json_response(
            {"result": "success", "msg": "", "events": []}
        )


@pytest.fixture
async def realm(monkeypatch):
    import sr2_spectre.interfaces.zulip.client as client_mod
    import sr2_spectre.interfaces.zulip.direct as direct_mod
    import sr2_spectre.interfaces.zulip.interface as iface_mod

    # Other test files replace these module attributes without restoring them.
    for module in (iface_mod, direct_mod):
        monkeypatch.setattr(module, "ZulipClient", client_mod.ZulipClient)
        monkeypatch.setattr(module, "load_zulip_credentials",
                            client_mod.load_zulip_credentials)
    state = _Realm()
    app = web.Application()
    app.router.add_post("/api/v1/register", state.register)
    app.router.add_get("/api/v1/events", state.events)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        await runner.cleanup()


def _zuliprc(tmp_path: Path) -> str:
    path = tmp_path / "bot.zuliprc"
    path.write_text(
        "[api]\nemail = miranda-bot@zulip.example\n"
        "key = synthetic-key-queue-expiry\nsite = https://normandy.invalid\n"
    )
    return str(path)


class _FakeAgent:
    def __init__(self) -> None:
        self.config = SimpleNamespace(agent=SimpleNamespace(name="miranda"))

    def apply_config(self, config: Any) -> None:
        pass


async def test_register_asks_for_a_long_idle_queue_timeout(tmp_path, realm):
    url, state = realm
    interface = ZulipInterface(ZulipConfig(
        zuliprc=_zuliprc(tmp_path), api_url=url, users=[DIEGO_ID],
    ))
    await interface.start(_FakeAgent())
    await interface.stop()

    assert int(state.register_forms[0]["idle_queue_timeout"]) >= 3600


async def test_bad_event_queue_id_raises_queue_expired(realm):
    url, state = realm
    state.expired.add("q-gone")
    client = ZulipClient(
        credentials=ZulipCredentials(
            email="miranda-bot@zulip.example", api_key="k",
            site="https://normandy.invalid",
        ),
        config=ZulipConfig(zuliprc="unused", api_url=url),
    )
    try:
        with pytest.raises(ZulipQueueExpiredError) as caught:
            await client.get_events("q-gone", 0)
    finally:
        await client.aclose()
    # The server body (which echoes the queue ID) never reaches the message.
    assert "q-gone" not in str(caught.value)


async def test_other_http_400_stays_a_transport_error(realm):
    url, _state = realm
    client = ZulipClient(
        credentials=ZulipCredentials(
            email="miranda-bot@zulip.example", api_key="k",
            site="https://normandy.invalid",
        ),
        config=ZulipConfig(zuliprc="unused", api_url=url),
    )
    try:
        with pytest.raises(ZulipTransportError) as caught:
            await client.get_events("q-bad-request", 0)
    finally:
        await client.aclose()
    assert not isinstance(caught.value, ZulipQueueExpiredError)


@pytest.mark.parametrize("mode", ["stream", "direct"])
async def test_expired_queue_is_re_registered_and_polling_resumes(
    tmp_path, realm, mode
):
    url, state = realm
    if mode == "stream":
        interface: Any = ZulipInterface(ZulipConfig(
            zuliprc=_zuliprc(tmp_path), api_url=url, users=[DIEGO_ID],
        ))
        await interface.start(_FakeAgent())
    else:
        interface = ZulipDMInterface(ZulipConfig(
            zuliprc=_zuliprc(tmp_path), api_url=url, users=[DIEGO_ID],
            mode="direct", peer_id=DIEGO_ID,
        ))
        await interface.start(None)
    state.expired.add("q-1")

    def stop_on_new_queue(queue_id: str) -> None:
        if queue_id == "q-2":
            interface._running = False

    state.on_poll = stop_on_new_queue
    try:
        await asyncio.wait_for(interface.run(None), timeout=10)
    finally:
        await interface.stop()

    assert len(state.register_forms) == 2
    assert state.polled == ["q-1", "q-2"]
