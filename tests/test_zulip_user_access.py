"""spc-1013 — Zulip users list, bot/self loop guard and mention_only (AC4–AC6).

Case-table rows of specs/zulip-channel-access.md, one test (or one
parametrized case) each, named ``test_<row>_...``. Registration in every
row: own ``user_id`` 14, ``full_name`` "Miranda", ``realm_users`` = humans 8
and 9, bots 14 and 20. ``channels: []`` throughout.

- Stream rows drive the real ``ZulipInterface`` poll loop with a fake
  transport and a fake agent whose Sessions record every turn.
- Direct rows drive the real ``ZulipDMInterface`` start/poll core with a
  fake transport. No model, no network, no credentials.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from sr2_spectre.config import SpectreConfig
from sr2_spectre.interfaces.zulip import ZulipConfig, ZulipDMInterface, ZulipInterface
from sr2_spectre.interfaces.zulip.client import ZulipCredentials

OWN_ID = 14
HUMAN_A = 8
HUMAN_B = 9
OTHER_BOT = 20
CHANNEL = 2
TOPIC = "status"

REGISTRATION = {
    "queue_id": "q-spc-1013",
    "last_event_id": 0,
    "user_id": OWN_ID,
    "full_name": "Miranda",
    "realm_users": [
        {"user_id": HUMAN_A, "is_bot": False},
        {"user_id": HUMAN_B, "is_bot": False},
        {"user_id": OWN_ID, "is_bot": True},
        {"user_id": OTHER_BOT, "is_bot": True},
    ],
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self) -> None:
        self.context: Any = None
        self.turns: list[str] = []

    def set_run_context(self, context: Any) -> None:
        self.context = context

    async def handle_user_message(self, text: str) -> Any:
        self.turns.append(text)
        return SimpleNamespace(text=f"reply to {text}")

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
    def turns(self) -> list[str]:
        return [turn for s in self.sessions.values() for turn in s.turns]


class _FakeClient:
    """Event queue scripted as batches of message dicts."""

    def __init__(self, interface: Any, batches: list[list[dict]],
                 registration: dict) -> None:
        self._interface = interface
        self._batches = list(batches)
        self._registration = registration
        self.sends: list[dict] = []
        self.history_calls: list[tuple[int, str]] = []

    async def register(self) -> dict:
        return dict(self._registration)

    async def get_events(self, queue_id: str, last_event_id: int) -> dict:
        if not self._batches:
            self._interface._running = False
            return {"events": []}
        messages = self._batches.pop(0)
        return {"events": [{"id": m["id"], "type": "message", "message": m}
                           for m in messages]}

    async def get_topic_messages(self, channel_id: int, topic: str) -> dict:
        self.history_calls.append((channel_id, topic))
        return {"messages": []}

    async def send_message(self, **kwargs: Any) -> int:
        self.sends.append(kwargs)
        return 1

    async def aclose(self) -> None:
        pass


_next_id = [1000]


def _stream_msg(sender_id: int, content: str = "status?", topic: str = TOPIC) -> dict:
    _next_id[0] += 1
    return {"id": _next_id[0], "type": "stream", "stream_id": CHANNEL,
            "display_recipient": "sandbox", "subject": topic,
            "sender_id": sender_id, "content": content}


def _dm(sender_id: int, content: str = "hi") -> dict:
    _next_id[0] += 1
    return {"id": _next_id[0], "type": "private", "sender_id": sender_id,
            "content": content}


def _credentials(path: str) -> ZulipCredentials:
    return ZulipCredentials(email="miranda-bot@test.local", api_key="k",
                            site="http://127.0.0.1:1")


async def _drive_stream(monkeypatch, config: ZulipConfig,
                        batches: list[list[dict]],
                        registration: dict = REGISTRATION):
    import sr2_spectre.interfaces.zulip.interface as iface_mod

    interface = ZulipInterface(config)
    client = _FakeClient(interface, batches, registration)
    monkeypatch.setattr(iface_mod, "load_zulip_credentials", _credentials)
    monkeypatch.setattr(iface_mod, "ZulipClient", lambda **kw: client)
    agent = _FakeAgent()
    await interface.start(agent)
    try:
        await interface.run(agent)
    finally:
        await interface.stop()
    return agent, client


async def _poll_direct(monkeypatch, config: ZulipConfig, messages: list[dict]):
    """Admitted sender IDs of one DM batch through the real start/poll core."""
    import sr2_spectre.interfaces.zulip.direct as direct_mod

    interface = ZulipDMInterface(config)
    client = _FakeClient(interface, [messages], REGISTRATION)
    monkeypatch.setattr(direct_mod, "load_zulip_credentials", _credentials)
    monkeypatch.setattr(direct_mod, "ZulipClient", lambda **kw: client)
    await interface.start(None)
    try:
        admitted = await interface.poll()
    finally:
        await interface.stop()
    return [dm.sender_id for dm in admitted]


def _stream(**over: Any) -> ZulipConfig:
    return ZulipConfig(zuliprc="/unused.zuliprc", **over)


def _direct(**over: Any) -> ZulipConfig:
    return ZulipConfig(zuliprc="/unused.zuliprc", mode="direct", **over)


def _assert_no_work(agent: _FakeAgent, client: _FakeClient) -> None:
    assert agent.turns == []
    assert agent.sessions == {}
    assert client.sends == []
    assert client.history_calls == []


# ---------------------------------------------------------------------------
# AC4 — users: empty = any human, nonempty = listed IDs; allowed_user_ids fails
# ---------------------------------------------------------------------------


async def test_AC4_a_empty_users_admits_any_human_in_stream(monkeypatch):
    agent, client = await _drive_stream(monkeypatch, _stream(users=[]),
                                        [[_stream_msg(HUMAN_B, "from nine")]])
    assert agent.turns == ["from nine"]
    assert client.sends == [{"type": "stream", "to": CHANNEL, "topic": TOPIC,
                             "content": "reply to from nine"}]


async def test_AC4_b_nonempty_users_rejects_unlisted_human(monkeypatch):
    agent, client = await _drive_stream(monkeypatch, _stream(users=[HUMAN_A]),
                                        [[_stream_msg(HUMAN_B, "from nine")]])
    _assert_no_work(agent, client)


@pytest.mark.parametrize("build", [
    pytest.param(lambda: ZulipConfig(zuliprc="/x", allowed_user_ids=[HUMAN_A]),
                 id="zulip-config"),
    pytest.param(lambda: SpectreConfig(
        agent={"name": "miranda"}, models={"default": {"model": "m"}},
        pipeline={"layers": []},
        zulip={"zuliprc": "/x", "allowed_user_ids": [HUMAN_A]}),
                 id="whole-config"),
])
def test_AC4_c_allowed_user_ids_fails_naming_users(build):
    with pytest.raises(ValidationError) as excinfo:
        build()
    message = str(excinfo.value)
    assert "allowed_user_ids" in message
    assert "`users`" in message


async def test_AC4_d_direct_empty_users_admits_peer_only(monkeypatch):
    admitted = await _poll_direct(
        monkeypatch, _direct(users=[], peer_id=HUMAN_A),
        [_dm(HUMAN_A), _dm(HUMAN_B)],
    )
    assert admitted == [HUMAN_A]


async def test_AC4_e_direct_listed_peer_admitted(monkeypatch):
    admitted = await _poll_direct(
        monkeypatch, _direct(users=[HUMAN_A], peer_id=HUMAN_A), [_dm(HUMAN_A)],
    )
    assert admitted == [HUMAN_A]


# ---------------------------------------------------------------------------
# AC5 — a bot or the bot's own account never triggers, whatever users holds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row,sender,users", [
    pytest.param("AC5-a", OWN_ID, [], id="AC5_a-own-id-empty-users"),
    pytest.param("AC5-b", OTHER_BOT, [], id="AC5_b-other-bot-empty-users"),
    pytest.param("AC5-d", OWN_ID, [OWN_ID], id="AC5_d-own-id-listed"),
])
async def test_AC5_a_b_d_stream_bot_or_self_never_triggers(monkeypatch, row,
                                                          sender, users):
    agent, client = await _drive_stream(monkeypatch, _stream(users=users),
                                        [[_stream_msg(sender, row)]])
    _assert_no_work(agent, client)


async def test_AC5_own_id_never_triggers_even_unmarked_in_realm_users(monkeypatch):
    """The own-ID clause stands alone: 14 not marked is_bot still never triggers."""
    registration = dict(REGISTRATION, realm_users=[
        {"user_id": HUMAN_A, "is_bot": False},
        {"user_id": OWN_ID, "is_bot": False},
    ])
    agent, client = await _drive_stream(monkeypatch, _stream(users=[]),
                                        [[_stream_msg(OWN_ID, "my own reply")]],
                                        registration=registration)
    _assert_no_work(agent, client)


@pytest.mark.parametrize("peer_id", [OWN_ID, OTHER_BOT])
async def test_AC5_c_direct_bot_or_self_never_admitted(monkeypatch, peer_id):
    admitted = await _poll_direct(
        monkeypatch, _direct(users=[], peer_id=peer_id),
        [_dm(OWN_ID), _dm(OTHER_BOT)],
    )
    assert admitted == []


# ---------------------------------------------------------------------------
# AC6 — mention_only in stream mode
# ---------------------------------------------------------------------------


async def test_AC6_a_mention_only_unmentioned_new_topic_no_turn(monkeypatch):
    agent, client = await _drive_stream(
        monkeypatch, _stream(mention_only=True),
        [[_stream_msg(HUMAN_A, "status?")]],
    )
    _assert_no_work(agent, client)


async def test_AC6_b_mention_by_name_runs_one_turn(monkeypatch):
    agent, client = await _drive_stream(
        monkeypatch, _stream(mention_only=True),
        [[_stream_msg(HUMAN_A, "@**Miranda** status?")]],
    )
    assert agent.turns == ["@**Miranda** status?"]
    assert len(client.sends) == 1


@pytest.mark.parametrize("content", [
    "@**Miranda|14** hi", "@_**Miranda|14** hi", "@_**Miranda** hi",
])
async def test_AC6_c_id_and_silent_mentions_each_run_one_turn(monkeypatch, content):
    agent, client = await _drive_stream(
        monkeypatch, _stream(mention_only=True),
        [[_stream_msg(HUMAN_A, content, topic=f"new {content}")]],
    )
    assert agent.turns == [content]
    assert len(client.sends) == 1


@pytest.mark.parametrize("content", [
    "@**all** hi", "@**everyone** hi", "@**channel** hi", "@**stream** hi",
    "@**topic** hi",
])
async def test_AC6_d_wildcard_mention_no_turn(monkeypatch, content):
    agent, client = await _drive_stream(
        monkeypatch, _stream(mention_only=True),
        [[_stream_msg(HUMAN_A, content)]],
    )
    _assert_no_work(agent, client)


async def test_AC6_e_active_topic_needs_no_mention(monkeypatch):
    agent, client = await _drive_stream(monkeypatch, _stream(mention_only=True), [
        [_stream_msg(HUMAN_A, "@**Miranda** status?")],
        [_stream_msg(HUMAN_A, "and then?")],
    ])
    assert len(agent.sessions) == 1
    assert agent.turns == ["@**Miranda** status?", "and then?"]
    assert len(client.sends) == 2


async def test_AC6_f_name_prefix_is_not_a_mention(monkeypatch):
    agent, client = await _drive_stream(
        monkeypatch, _stream(mention_only=True),
        [[_stream_msg(HUMAN_A, "@**Mirandas** hi")]],
    )
    _assert_no_work(agent, client)


async def test_AC6_g_mention_only_defaults_false(monkeypatch):
    assert _stream().mention_only is False
    agent, client = await _drive_stream(monkeypatch, _stream(),
                                        [[_stream_msg(HUMAN_A, "status?")]])
    assert agent.turns == ["status?"]
    assert len(client.sends) == 1


async def test_AC6_h_direct_mode_ignores_mention_only(monkeypatch):
    admitted = await _poll_direct(
        monkeypatch, _direct(mention_only=True, peer_id=HUMAN_A), [_dm(HUMAN_A)],
    )
    assert admitted == [HUMAN_A]
