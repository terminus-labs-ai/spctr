"""spc-1012 — Zulip channel access mirrors Discord (AC1, AC2, AC3).

Case-table rows of specs/zulip-channel-access.md, one test (or one
parametrized case) each, named ``test_<row>_...``:

- AC1: the register request of each mode, recorded by a loopback Zulip
  endpoint (real ``ZulipClient``, synthetic zuliprc, no realm).
- AC2/AC3: the stream interface's poll loop driven by a fake transport and
  a fake agent whose Sessions record every turn and the ``RunContext`` the
  turn ran under. No model, no network.

Rows use ``users: [8]`` and human sender 8.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from pydantic import ValidationError

from sr2_spectre.config import SpectreConfig
from sr2_spectre.config_source import SpectreConfigSource
from sr2_spectre.events import AgentTextDelta
from sr2_spectre.interfaces.zulip import ZulipConfig, ZulipDMInterface, ZulipInterface
from sr2_spectre.interfaces.zulip.client import ZulipCredentials
from sr2_spectre.resume import ResumeResult

DIEGO_ID = 8
BOT_ID = 9
TOPIC = "save system"


# ---------------------------------------------------------------------------
# AC1 — register request shape per mode, against a loopback endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
async def register_endpoint(monkeypatch):
    """Loopback Zulip endpoint answering POST /api/v1/register; records forms.

    Pins the REAL transport and credential loader on both interface modules:
    other test files replace those module attributes without restoring them.
    """
    import sr2_spectre.interfaces.zulip.client as client_mod
    import sr2_spectre.interfaces.zulip.direct as direct_mod
    import sr2_spectre.interfaces.zulip.interface as iface_mod

    for module in (iface_mod, direct_mod):
        monkeypatch.setattr(module, "ZulipClient", client_mod.ZulipClient)
        monkeypatch.setattr(module, "load_zulip_credentials",
                            client_mod.load_zulip_credentials)
    forms: list[dict[str, str]] = []

    async def register(request: web.Request) -> web.Response:
        forms.append(dict(await request.post()))
        return web.json_response({
            "result": "success",
            "queue_id": "q-spc-1012",
            "last_event_id": 0,
            "user_id": BOT_ID,
            "full_name": "Miranda",
            "realm_users": [
                {"user_id": DIEGO_ID, "is_bot": False},
                {"user_id": BOT_ID, "is_bot": True},
            ],
        })

    app = web.Application()
    app.router.add_post("/api/v1/register", register)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", forms
    finally:
        await runner.cleanup()


def _zuliprc(tmp_path: Path) -> str:
    path = tmp_path / "bot.zuliprc"
    path.write_text(
        "[api]\nemail = miranda-bot@zulip.example\n"
        "key = synthetic-key-spc-1012\nsite = https://normandy.invalid\n"
    )
    return str(path)


async def test_AC1_a_stream_mode_registers_all_public_streams_and_realm_user(
    tmp_path, register_endpoint
):
    url, forms = register_endpoint
    interface = ZulipInterface(ZulipConfig(
        zuliprc=_zuliprc(tmp_path), api_url=url, users=[DIEGO_ID],
    ))
    await interface.start(_FakeAgent())
    await interface.stop()

    assert len(forms) == 1
    form = forms[0]
    assert json.loads(form["event_types"]) == ["message"]
    fetched = json.loads(form["fetch_event_types"])
    assert "message" in fetched and "realm_user" in fetched
    assert form["all_public_streams"] == "true"


async def test_AC1_b_direct_mode_registers_without_all_public_streams(
    tmp_path, register_endpoint
):
    url, forms = register_endpoint
    interface = ZulipDMInterface(ZulipConfig(
        zuliprc=_zuliprc(tmp_path), api_url=url, users=[DIEGO_ID],
        mode="direct", peer_id=DIEGO_ID,
    ))
    await interface.start(None)
    await interface.stop()

    assert len(forms) == 1
    form = forms[0]
    assert "all_public_streams" not in form
    assert "realm_user" in json.loads(form["fetch_event_types"])


# ---------------------------------------------------------------------------
# AC2/AC3 harness — fake transport and agent around the real interface
# ---------------------------------------------------------------------------


class _FakeSession:
    async def restore_history(self) -> ResumeResult:
        return ResumeResult([], interrupted=False, restore_failed=False)

    def __init__(self) -> None:
        self.context: Any = None
        self.turns: list[tuple[str, str]] = []   # (content, area at turn time)

    def set_run_context(self, context: Any) -> None:
        self.context = context

    async def stream_message(self, text: str) -> Any:
        self.turns.append((text, self.context.area))
        yield AgentTextDelta(text=f"reply to {text}")

    def close(self) -> None:
        pass


class _FakeAgent:
    def __init__(self) -> None:
        self.config = SimpleNamespace(agent=SimpleNamespace(name="miranda"))
        self.sessions: dict[str, _FakeSession] = {}

    def create_session(self, frame_id: str) -> _FakeSession:
        session = _FakeSession()
        self.sessions[frame_id] = session
        return session

    def apply_config(self, config: Any) -> None:
        pass

    @property
    def turns(self) -> list[tuple[str, str]]:
        return [turn for s in self.sessions.values() for turn in s.turns]


class _FakeClient:
    """Event queue scripted as batches; each batch may carry the config the
    whole-config reload loads when that batch arrives (a disk edit)."""

    def __init__(self, interface: Any, batches: list, holder: dict) -> None:
        self._interface = interface
        self._batches = list(batches)
        self._holder = holder
        self.sends: list[dict] = []
        self.history_calls: list[tuple[int, str]] = []

    async def register(self) -> dict:
        return {"queue_id": "q", "last_event_id": 0,
                "realm_users": [{"user_id": BOT_ID, "is_bot": True}]}

    async def get_events(self, queue_id: str, last_event_id: int) -> dict:
        if not self._batches:
            self._interface._running = False
            return {"events": []}
        config, messages = self._batches.pop(0)
        if config is not None:
            self._holder["config"] = config
        return {"events": [{"id": m["id"], "type": "message", "message": m}
                           for m in messages]}

    async def get_topic_messages(self, channel_id: int, topic: str) -> dict:
        self.history_calls.append((channel_id, topic))
        return {"messages": []}

    async def send_message(self, **kwargs: Any) -> int:
        self.sends.append(kwargs)
        return len(self.sends)   # message ID = position in the ledger

    async def update_message(self, message_id: int, content: str) -> None:
        # Like a real server: the edit replaces the stored message's content,
        # so ``sends`` holds each message as the topic finally shows it.
        self.sends[message_id - 1] = {**self.sends[message_id - 1],
                                      "content": content}

    async def aclose(self) -> None:
        pass


_next_id = [100]


def _msg(stream_id: int, name: str, content: str, topic: str = TOPIC) -> dict:
    _next_id[0] += 1
    return {"id": _next_id[0], "type": "stream", "stream_id": stream_id,
            "display_recipient": name, "subject": topic,
            "sender_id": DIEGO_ID, "content": content}


def _zulip(**over: Any) -> ZulipConfig:
    return ZulipConfig(zuliprc="/unused.zuliprc", users=[DIEGO_ID], **over)


def _spectre(zulip: ZulipConfig) -> SpectreConfig:
    return SpectreConfig(agent={"name": "miranda"},
                         models={"default": {"model": "synthetic-model"}},
                         pipeline={"layers": []}, zulip=zulip)


async def _drive(monkeypatch, initial: ZulipConfig, batches: list):
    """Run the real poll loop over *batches* with a live config source."""
    import sr2_spectre.interfaces.zulip.interface as iface_mod

    holder = {"config": _spectre(initial)}
    source = SpectreConfigSource(loader=lambda: holder["config"],
                                 initial=holder["config"])
    interface = ZulipInterface(config_source=source)
    client = _FakeClient(interface, batches, holder)
    monkeypatch.setattr(iface_mod, "load_zulip_credentials",
                        lambda path: ZulipCredentials(
                            email="bot@test.local", api_key="k",
                            site="http://127.0.0.1:1"))
    monkeypatch.setattr(iface_mod, "ZulipClient", lambda **kw: client)
    agent = _FakeAgent()
    await interface.start(agent)
    try:
        await interface.run(agent)
    finally:
        await interface.stop()
    return agent, client


# ---------------------------------------------------------------------------
# AC2 — channels is a list; empty admits every delivered channel
# ---------------------------------------------------------------------------


async def test_AC2_a_empty_channels_admits_any_delivered_channel(monkeypatch):
    agent, client = await _drive(monkeypatch, _zulip(channels=[]),
                                 [(None, [_msg(77, "lab", "hello lab")])])
    assert [text for text, _area in agent.turns] == ["hello lab"]
    assert client.sends == [{"type": "stream", "to": 77, "topic": TOPIC,
                             "content": "reply to hello lab"}]


async def test_AC2_b_unlisted_channel_triggers_no_work(monkeypatch):
    agent, client = await _drive(monkeypatch, _zulip(channels=[2]),
                                 [(None, [_msg(3, "other", "not here")])])
    assert agent.turns == []
    assert agent.sessions == {}
    assert client.sends == []
    assert client.history_calls == []


async def test_AC2_c_listed_channel_runs_one_turn(monkeypatch):
    agent, client = await _drive(monkeypatch, _zulip(channels=[2]),
                                 [(None, [_msg(2, "sandbox", "listed")])])
    assert [text for text, _area in agent.turns] == ["listed"]
    assert len(client.sends) == 1 and client.sends[0]["to"] == 2


def test_AC2_d_mapping_valued_channels_fails_validation():
    with pytest.raises(ValidationError):
        _zulip(channels={2: ""})


async def test_AC2_e_channels_edit_applies_to_next_batch(monkeypatch):
    agent, client = await _drive(monkeypatch, _zulip(channels=[2]), [
        (None, [_msg(2, "two", "before edit", topic="t0")]),
        (_spectre(_zulip(channels=[3])),
         [_msg(2, "two", "in two"), _msg(3, "three", "in three")]),
    ])
    assert sorted(text for text, _area in agent.turns) == ["before edit", "in three"]
    assert [send["to"] for send in client.sends] == [2, 3]


# ---------------------------------------------------------------------------
# AC3 — area: channel_areas override, else the Discord name derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row,channel_areas,stream_id,name,expected", [
    pytest.param("AC3-a", {"2": ""}, 2, "sandbox", "", id="AC3_a-empty-override-stays-empty"),
    pytest.param("AC3-b", {"5": "harbinger"}, 5, "Fractured-Roots", "harbinger",
                 id="AC3_b-override-wins-over-name"),
    pytest.param("AC3-c", {}, 6, "Fractured-Roots!", "fractured-roots",
                 id="AC3_c-name-derived-like-discord"),
    pytest.param("AC3-d", {}, 7, "---", "", id="AC3_d-no-alphanumeric-is-empty"),
])
async def test_AC3_a_to_d_turn_area(monkeypatch, row, channel_areas, stream_id,
                                    name, expected):
    agent, _client = await _drive(
        monkeypatch, _zulip(channel_areas=channel_areas),
        [(None, [_msg(stream_id, name, row)])],
    )
    assert agent.turns == [(row, expected)]
    area = agent.turns[0][1]
    assert isinstance(area, str)   # never None


def test_AC3_e_integer_channel_areas_key_fails_validation():
    with pytest.raises(ValidationError):
        _zulip(channel_areas={2: ""})


def test_AC3_f_derivation_is_one_implementation_shared_with_discord():
    from sr2_spectre.interfaces import common
    from sr2_spectre.interfaces.discord import handler
    from sr2_spectre.interfaces.zulip import config as zulip_config

    assert zulip_config.resolve_area is common.resolve_area
    assert handler.resolve_area is common.resolve_area
    assert handler.derive_area_name is common.derive_area_name


async def test_AC3_g_channel_areas_edit_reaches_existing_topic(monkeypatch):
    agent, _client = await _drive(monkeypatch, _zulip(channel_areas={"2": ""}), [
        (None, [_msg(2, "sandbox", "first")]),
        (_spectre(_zulip(channel_areas={"2": "lab"})), [_msg(2, "sandbox", "second")]),
    ])
    assert len(agent.sessions) == 1   # the same existing topic Session
    assert agent.turns == [("first", ""), ("second", "lab")]
