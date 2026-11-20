"""spc-1071 — Zulip topic-ownership routing (AC1–AC8, case table r1–r10).

Two fake agents (EDI user_id 101, Miranda user_id 102) drive the real
``ZulipInterface`` one message at a time — no concurrency harness: events
are awaited sequentially, alternating instances. Both fakes share ONE
test-owned per-topic message list, so the opener fetch sees the same
first message for every bot; ``get_topic_opener`` calls are counted and
the opener path can be made to fail. Fakes record Session probes
(``create_session``), model turns (``stream_message``) and replies so a
drop can be asserted as zero Session and zero runtime work.

Identities: human Diego user_id 5; bots 101 (EDI) and 102 (Miranda).
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from pydantic import ValidationError

from spctr.events import AgentTextDelta
from spctr.interfaces.zulip import (
    ZulipConfig,
    ZulipDMInterface,
    ZulipInterface,
)
from spctr.interfaces.zulip.client import (
    ZulipClient,
    ZulipCredentials,
    ZulipTransportError,
)
from spctr.interfaces.zulip.config import (
    bot_names_from_registration,
    mentions_other,
    mentions_other_bot,
)
from spctr.interfaces.zulip.interface import TopicOwner
from spctr.resume import ResumeResult

EDI_ID = 101
EDI_NAME = "EDI"
MIRANDA_ID = 102
MIRANDA_NAME = "Miranda"
DIEGO_ID = 5
CHANNEL = 2
TOPIC = "status"


# ---------------------------------------------------------------------------
# Harness — fake agent/session/client around the real interface
# ---------------------------------------------------------------------------


class _FakeSession:
    async def restore_history(self) -> ResumeResult:
        return ResumeResult([], interrupted=False, restore_failed=False)

    def __init__(self) -> None:
        self.context: Any = None
        self.turns: list[str] = []
        self.active_handoff: Any = None

    def set_run_context(self, context: Any) -> None:
        self.context = context

    async def stream_message(self, text: str) -> Any:
        self.turns.append(text)
        yield AgentTextDelta(text=f"reply to {text}")

    def close(self) -> None:
        pass


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.config = SimpleNamespace(agent=SimpleNamespace(name=name))
        self.sessions: dict[str, _FakeSession] = {}
        self.creations = 0

    def create_session(self, frame_id: str) -> _FakeSession:
        self.creations += 1
        session = _FakeSession()
        self.sessions[frame_id] = session
        return session

    def apply_config(self, config: Any) -> None:
        pass

    @property
    def turns(self) -> list[str]:
        return [t for s in self.sessions.values() for t in s.turns]


class _FakeClient:
    """Per-interface transport; every instance reads ONE shared topics map
    (keyed ``(channel_id, topic)``), so the opener fetch returns the same
    first message whichever bot asks."""

    def __init__(self, topics: dict, *, opener_fails: bool = False) -> None:
        self._topics = topics
        self._opener_fails = opener_fails
        self.sends: list[dict] = []
        self.opener_calls: list[tuple[int, str]] = []
        self.history_calls: list[tuple[int, str]] = []

    async def register(self) -> dict:
        return {"queue_id": "q", "last_event_id": 0}

    async def get_events(self, queue_id: str, last_event_id: int) -> dict:
        return {"events": []}

    async def get_topic_opener(self, channel_id: int, topic: str) -> dict:
        self.opener_calls.append((channel_id, topic))
        if self._opener_fails:
            raise ZulipTransportError("synthetic opener failure")
        messages = self._topics.get((channel_id, topic)) or []
        return {"messages": messages[:1]}

    async def get_topic_messages(self, channel_id: int, topic: str) -> dict:
        self.history_calls.append((channel_id, topic))
        return {"messages": []}

    async def send_message(self, **kwargs: Any) -> int:
        self.sends.append(kwargs)
        return len(self.sends)   # message ID = position in the ledger

    async def update_message(self, message_id: int, content: str) -> None:
        self.sends[message_id - 1] = {
            **self.sends[message_id - 1],
            "content": content,
        }

    async def aclose(self) -> None:
        pass


_next_id = [1000]


def _msg(content: str, sender: int = DIEGO_ID,
         channel: int = CHANNEL, topic: str = TOPIC) -> dict:
    _next_id[0] += 1
    return {"id": _next_id[0], "type": "stream", "stream_id": channel,
            "display_recipient": "sandbox", "subject": topic,
            "sender_id": sender, "content": content}


def _iface(name: str, user_id: int, client: _FakeClient,
           *, mention_only: bool = False, routing: bool = True):
    """One real ``ZulipInterface`` bound to a fake agent and transport."""
    config = ZulipConfig(
        zuliprc="/unused.zuliprc", api_url="http://zulip.test",
        users=[DIEGO_ID], mention_only=mention_only,
        topic_owner_routing=routing, edit_stream_interval=0,
    )
    iface = ZulipInterface(config)
    agent = _FakeAgent(name)
    iface._agent = agent
    iface._client = client
    iface._credentials = ZulipCredentials(
        email="b@test.local", api_key="k", site="http://zulip.test")
    iface._bot_ids = {EDI_ID, MIRANDA_ID}
    iface._bot_names = {EDI_NAME, MIRANDA_NAME}
    iface._bot_user_id = user_id
    iface._bot_full_name = name
    return iface, agent


def _pair(topics: dict, *, edi_mention_only: bool = True,
          mir_mention_only: bool = False, routing: bool = True):
    """EDI + Miranda interfaces sharing one topics map."""
    edi, edi_agent = _iface(EDI_NAME, EDI_ID, _FakeClient(topics),
                            mention_only=edi_mention_only, routing=routing)
    mir, mir_agent = _iface(MIRANDA_NAME, MIRANDA_ID, _FakeClient(topics),
                            mention_only=mir_mention_only, routing=routing)
    return edi, edi_agent, mir, mir_agent


def _key(channel: int = CHANNEL, topic: str = TOPIC) -> tuple[int, str]:
    return (channel, topic)


async def _settle(*ifaces: ZulipInterface) -> None:
    """Await every task ``_dispatch`` spawned: the per-topic chains
    (``_topic_tails``) plus the fire-and-forget ``!stop`` fast path."""
    await asyncio.gather(
        *(t for i in ifaces for t in i._topic_tails.values()),
        *(
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ),
        return_exceptions=True,
    )


# ---------------------------------------------------------------------------
# r1 — opener mentioning only EDI: EDI self + processes, Miranda other + drop
# ---------------------------------------------------------------------------


async def test_r1_edi_opener_self_for_edi_other_for_miranda():
    opener = _msg("@**EDI** status?")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)

    await edi._handle_message(opener, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert ea.turns == ["@**EDI** status?"]
    assert edi._client.opener_calls == [_key()]

    await mir._handle_message(opener, reload=False)
    assert mir._topic_owners[_key()] is TopicOwner.OTHER
    assert ma.turns == [] and ma.sessions == {} and ma.creations == 0
    assert mir._client.sends == []
    assert mir._client.opener_calls == [_key()]


# ---------------------------------------------------------------------------
# r2 — topic recorded self(EDI): untagged follow-up passes EDI's
# mention_only (no active Session yet), Miranda (other) drops
# ---------------------------------------------------------------------------


async def test_r2_self_recorded_topic_exempts_mention_only_other_drops():
    # The opener (already in the topic) tagged EDI; this event is an
    # untagged follow-up arriving before EDI built any Session.
    opener = _msg("@**EDI** status?")
    follow_up = _msg("and then?")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)   # EDI mention_only=True

    await edi._handle_message(follow_up, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert ea.turns == ["and then?"]          # exempt, no prior Session

    await mir._handle_message(follow_up, reload=False)
    assert mir._topic_owners[_key()] is TopicOwner.OTHER
    assert ma.turns == [] and ma.creations == 0


# ---------------------------------------------------------------------------
# r3 — untagged opener: both record unowned; non-mention_only bot answers;
# untagged follow-ups record nothing new
# ---------------------------------------------------------------------------


async def test_r3_untagged_opener_unowned_for_both_no_followup_claim():
    opener = _msg("anyone here?")
    follow_up = _msg("still nobody")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)

    await edi._handle_message(opener, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.UNOWNED
    assert ea.turns == []               # mention_only still gates it

    await mir._handle_message(opener, reload=False)
    assert mir._topic_owners[_key()] is TopicOwner.UNOWNED
    assert ma.turns == ["anyone here?"]

    await edi._handle_message(follow_up, reload=False)
    await mir._handle_message(follow_up, reload=False)
    # Follow-ups without realm mentions never re-record the owner.
    assert edi._topic_owners[_key()] is TopicOwner.UNOWNED
    assert mir._topic_owners[_key()] is TopicOwner.UNOWNED
    # And the recorded (even unowned) value is cached: still one fetch each.
    assert edi._client.opener_calls == [_key()]
    assert mir._client.opener_calls == [_key()]


# ---------------------------------------------------------------------------
# r4 — EDI-owned topic: Miranda mention hands it over, EDI records other
# and drops it plus the untagged follow-up; EDI mention re-claims
# ---------------------------------------------------------------------------


async def test_r4_handover_and_reclaim_sequence():
    m1 = _msg("@**EDI** first")
    m2 = _msg("@**Miranda** take this")
    m3 = _msg("and then?")
    mstop = _msg("!stop")
    m4 = _msg("@**EDI** you again")
    topics = {_key(): [m1]}
    edi, ea, mir, ma = _pair(topics)
    ec, mc = edi._client, mir._client

    # EDI-owned topic is established.
    await edi._handle_message(m1, reload=False)
    await mir._handle_message(m1, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert mir._topic_owners[_key()] is TopicOwner.OTHER
    assert ea.turns == ["@**EDI** first"] and ma.turns == []

    # A message tagging Miranda: Miranda claims self and processes it;
    # EDI records other and drops it.
    await edi._handle_message(m2, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.OTHER
    assert ea.turns == ["@**EDI** first"]          # unchanged: dropped
    await mir._handle_message(m2, reload=False)
    assert mir._topic_owners[_key()] is TopicOwner.SELF
    assert ma.turns == ["@**Miranda** take this"]

    # Untagged follow-up: EDI still drops, Miranda (owner self) continues.
    await edi._handle_message(m3, reload=False)
    await mir._handle_message(m3, reload=False)
    assert ea.turns == ["@**EDI** first"]
    assert ma.turns == ["@**Miranda** take this", "and then?"]

    # Even !stop sits behind ownership: EDI drops it with no reply.
    sends_before = len(ec.sends)
    await edi._handle_message(mstop, reload=False)
    assert len(ec.sends) == sends_before

    # A message tagging EDI re-claims the topic for EDI; Miranda records
    # other and drops.
    await edi._handle_message(m4, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert ea.turns == ["@**EDI** first", "@**EDI** you again"]
    await mir._handle_message(m4, reload=False)
    assert mir._topic_owners[_key()] is TopicOwner.OTHER
    assert ma.turns == ["@**Miranda** take this", "and then?"]


# ---------------------------------------------------------------------------
# r5 — opener mentioning a human (not a bot): both record other, both drop
# ---------------------------------------------------------------------------


async def test_r5_human_mentioned_opener_records_other_for_both():
    opener = _msg("@**Diego** heads up")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)

    await edi._handle_message(opener, reload=False)
    await mir._handle_message(opener, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.OTHER
    assert mir._topic_owners[_key()] is TopicOwner.OTHER
    assert ea.turns == [] and ma.turns == []
    assert edi._client.sends == [] and mir._client.sends == []


# ---------------------------------------------------------------------------
# r6 — plain typed @edi/@miranda is not a realm mention anywhere
# ---------------------------------------------------------------------------


async def test_r6_plain_at_typing_is_no_mention():
    opener = _msg("hey @edi and @miranda look here")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)

    await edi._handle_message(opener, reload=False)
    await mir._handle_message(opener, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.UNOWNED
    assert mir._topic_owners[_key()] is TopicOwner.UNOWNED
    assert ea.turns == []                    # mention_only: no mention
    assert ma.turns == ["hey @edi and @miranda look here"]


# ---------------------------------------------------------------------------
# r7 — every drop consumes zero Session probes and zero runtime work
# ---------------------------------------------------------------------------


async def test_r7_drop_paths_do_no_session_or_model_work():
    other_topic = _msg("@**EDI** topic")
    topics = {_key(): [other_topic]}
    edi, ea, mir, ma = _pair(topics)

    # Miranda drops every event in this EDI-owned topic.
    for msg in (other_topic, _msg("untagged one"), _msg("!stop"),
                _msg("@**Diego** human"), _msg("still nothing")):
        await mir._handle_message(msg, reload=False)

    assert ma.creations == 0                 # zero Session probes/creations
    assert ma.sessions == {}
    assert ma.turns == []                    # zero runtime/model invocations
    assert mir._client.sends == []           # zero replies (incl. indicator)
    assert mir._client.history_calls == []   # no topic-history probe either
    # Exactly one opener fetch — then cached as OTHER.
    assert mir._client.opener_calls == [_key()]


# ---------------------------------------------------------------------------
# r8 — flag off (default): pre-bead behavior, zero fetches, empty map
# ---------------------------------------------------------------------------


async def test_r8_flag_off_reproduces_pre_bead_behavior():
    opener = _msg("@**EDI** status?")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics, routing=False)

    await edi._handle_message(opener, reload=False)
    await mir._handle_message(opener, reload=False)

    # Pre-bead: BOTH agents answer a tagged topic (the live bug).
    assert ea.turns == ["@**EDI** status?"]
    assert ma.turns == ["@**EDI** status?"]
    # No opener fetch ever happens and nothing is recorded.
    assert edi._client.opener_calls == [] and mir._client.opener_calls == []
    assert edi._topic_owners == {} and mir._topic_owners == {}


async def test_r8_flag_off_untagged_behaves_as_before():
    opener = _msg("anyone?")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics, routing=False)

    await edi._handle_message(opener, reload=False)
    await mir._handle_message(opener, reload=False)
    assert ea.turns == []                    # mention_only, as before
    assert ma.turns == ["anyone?"]
    assert edi._topic_owners == {} and mir._topic_owners == {}


# ---------------------------------------------------------------------------
# r9 — strict config validation, default, and direct mode
# ---------------------------------------------------------------------------


def test_r9_topic_owner_routing_strict_bool_and_default():
    with pytest.raises(ValidationError):
        ZulipConfig(zuliprc="/x", topic_owner_routing="yes")
    with pytest.raises(ValidationError):
        ZulipConfig(zuliprc="/x", topic_owner_routing=1)
    assert ZulipConfig(zuliprc="/x").topic_owner_routing is False


async def test_r9_direct_mode_ignores_flag_no_fetches(monkeypatch):
    import spctr.interfaces.zulip.direct as direct_mod

    topics = {_key(): [_msg("unused")]}
    client = _FakeClient(topics)
    client._events = [[{"id": 1, "type": "private",
                       "sender_id": DIEGO_ID, "content": "hi"}]]

    async def _events(queue_id: str, last_event_id: int) -> dict:
        batches = client._events
        client._events = []
        return {"events": [
            {"id": m["id"], "type": "message", "message": m}
            for m in batches[0]
        ]}

    client.get_events = _events
    monkeypatch.setattr(direct_mod, "load_zulip_credentials",
                        lambda path: ZulipCredentials(
                            email="b@test.local", api_key="k",
                            site="http://127.0.0.1:1"))
    monkeypatch.setattr(direct_mod, "ZulipClient", lambda **kw: client)

    interface = ZulipDMInterface(ZulipConfig(
        zuliprc="/x", mode="direct", peer_id=DIEGO_ID,
        users=[DIEGO_ID], topic_owner_routing=True,
    ))
    await interface.start(None)
    try:
        admitted = await interface.poll()
    finally:
        await interface.stop()

    assert [dm.sender_id for dm in admitted] == [DIEGO_ID]
    assert client.opener_calls == []         # DM path never consults the flag


# ---------------------------------------------------------------------------
# r10 — fetch failure fail-open + retry; wildcard opener; bot-sent opener
# ---------------------------------------------------------------------------


async def test_r10_opener_fetch_failure_records_nothing_and_retries():
    opener = _msg("@**Miranda** hi")
    topics = {_key(): [opener]}
    failing = _FakeClient(topics, opener_fails=True)
    mir, ma = _iface(MIRANDA_NAME, MIRANDA_ID, failing)

    await mir._handle_message(opener, reload=False)
    assert mir._topic_owners == {}           # nothing recorded on failure
    assert ma.turns == ["@**Miranda** hi"]   # routes exactly as today

    second = _msg("again")
    await mir._handle_message(second, reload=False)
    # The unrecorded topic retries the fetch on the next event — and
    # still records nothing while it keeps failing.
    assert failing.opener_calls == [_key(), _key()]
    assert mir._topic_owners == {}


async def test_r10_wildcard_only_opener_is_unowned():
    opener = _msg("@**all** @**everyone** heads up")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)

    await edi._handle_message(opener, reload=False)
    await mir._handle_message(opener, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.UNOWNED
    assert mir._topic_owners[_key()] is TopicOwner.UNOWNED
    assert ea.turns == []                    # mention_only still applies
    assert ma.turns == ["@**all** @**everyone** heads up"]


async def test_r10_bot_sent_opener_mentioning_bot_is_unowned():
    # Opener authored by EDI (a bot) that mentions Miranda must never
    # establish ``other`` for Miranda.
    opener = _msg("@**Miranda** status?", sender=EDI_ID)
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)

    trigger = _msg("status?")
    await edi._handle_message(trigger, reload=False)
    await mir._handle_message(trigger, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.UNOWNED
    assert mir._topic_owners[_key()] is TopicOwner.UNOWNED
    assert ma.turns == ["status?"]


# ---------------------------------------------------------------------------
# Extra pins from the binding decisions
# ---------------------------------------------------------------------------


async def test_caching_unowned_and_owned_both_single_fetch():
    owned = _msg("@**EDI** owned")
    free = _msg("free")
    topics = {_key(): [owned], (CHANNEL, "free"): [free]}
    edi, ea, mir, ma = _pair(topics)

    for msg in (owned, _msg("one"), _msg("two")):
        await edi._handle_message(msg, reload=False)
        await mir._handle_message(msg, reload=False)
    for msg in (free, _msg("f1", topic="free"), _msg("f2", topic="free")):
        await edi._handle_message(msg, reload=False)
        await mir._handle_message(msg, reload=False)

    # One fetch per topic per process — including when the recorded
    # value is UNOWNED.
    assert edi._client.opener_calls == [_key(), (CHANNEL, "free")]
    assert mir._client.opener_calls == [_key(), (CHANNEL, "free")]
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert edi._topic_owners[(CHANNEL, "free")] is TopicOwner.UNOWNED


async def test_r4_name_id_mention_hands_over():
    """``@**Miranda|102**`` hands an EDI-owned topic to Miranda exactly
    like the bare-name form."""
    m1 = _msg("@**EDI** first")
    m2 = _msg("@**Miranda|102** take it")
    topics = {_key(): [m1]}
    edi, ea, mir, ma = _pair(topics)

    await edi._handle_message(m1, reload=False)
    await mir._handle_message(m1, reload=False)
    await edi._handle_message(m2, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.OTHER
    await mir._handle_message(m2, reload=False)
    assert mir._topic_owners[_key()] is TopicOwner.SELF
    assert ma.turns == ["@**Miranda|102** take it"]


async def test_later_human_mention_keeps_owner():
    """Diego ruling 2026-11-16: in a recorded topic a human mention —
    the quote-reply ``@_**Name|id** [said]`` attribution included —
    never changes the owner."""
    m1 = _msg("@**EDI** first")
    m2 = _msg("@**Diego** what do you think")
    m3 = _msg("@_**Diego|5** [said](#narrow/near/1):\n"
              "```quote\nx\n```\nfollow-up")
    topics = {_key(): [m1]}
    edi, ea, mir, ma = _pair(topics)

    await edi._handle_message(m1, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.SELF

    await edi._handle_message(m2, reload=False)
    await edi._handle_message(m3, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    # EDI processed all three: the human mentions never hand it over.
    assert ea.turns == [
        "@**EDI** first", "@**Diego** what do you think", m3["content"]
    ]


async def test_later_human_mention_keeps_unowned():
    opener = _msg("anyone?")
    m2 = _msg("@**Diego** look")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)

    await edi._handle_message(opener, reload=False)
    await mir._handle_message(opener, reload=False)
    await edi._handle_message(m2, reload=False)
    await mir._handle_message(m2, reload=False)
    assert edi._topic_owners[_key()] is TopicOwner.UNOWNED
    assert mir._topic_owners[_key()] is TopicOwner.UNOWNED
    assert ea.turns == []                    # mention_only still gates
    assert ma.turns == ["anyone?", "@**Diego** look"]


# ---------------------------------------------------------------------------
# The real poller path — _dispatch's !stop fast path is ownership-gated
# (defect: _dispatch answered !stop before _handle_message ever ran)
# ---------------------------------------------------------------------------


async def test_stop_via_dispatch_dropped_for_non_owner():
    """``!stop`` through ``_dispatch``: the non-owner's stop is dropped
    with no reply; the owner's stop sends the stop reply."""
    opener = _msg("@**EDI** first")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)

    edi._dispatch(opener, set(), edi._config)
    mir._dispatch(opener, set(), mir._config)
    await _settle(edi, mir)
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert mir._topic_owners[_key()] is TopicOwner.OTHER

    mir._dispatch(_msg("!stop"), set(), mir._config)
    await _settle(mir)
    assert mir._client.sends == []          # no "nothing running" reply

    before = len(edi._client.sends)
    edi._dispatch(_msg("!stop"), set(), edi._config)
    await _settle(edi)
    assert [s["content"] for s in edi._client.sends[before:]] == [
        edi._config.stop_idle_message
    ]


async def test_stop_via_dispatch_flag_off_unchanged():
    """Flag off: ``_dispatch``'s ``!stop`` fast path is the pre-bead one —
    the reply goes out, zero opener fetches, empty owner map."""
    edi, ea, mir, ma = _pair({}, routing=False)

    mir._dispatch(_msg("!stop"), set(), mir._config)
    await _settle(mir)
    assert [s["content"] for s in mir._client.sends] == [
        mir._config.stop_idle_message
    ]
    assert mir._client.opener_calls == []
    assert mir._topic_owners == {}


async def test_untagged_followup_via_dispatch_dropped_for_non_owner():
    """An untagged follow-up through ``_dispatch`` is dropped for the
    non-owner (no turn, no send) and runs the owner's turn."""
    opener = _msg("@**EDI** first")
    followup = _msg("untagged tail")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics)

    edi._dispatch(opener, set(), edi._config)
    mir._dispatch(opener, set(), mir._config)
    edi._dispatch(followup, set(), edi._config)
    mir._dispatch(followup, set(), mir._config)
    await _settle(edi, mir)

    assert mir._topic_owners[_key()] is TopicOwner.OTHER
    assert ma.turns == [] and mir._client.sends == []
    assert ea.turns == ["@**EDI** first", "untagged tail"]


async def test_unowned_topic_later_self_mention_claims_via_dispatch():
    """Unowned topic through ``_dispatch``: a message mentioning this bot
    records SELF, so the mention_only bot's later UNTAGGED follow-ups pass
    (the self-claim exemption). If the claim stopped recording in an
    unowned topic, the follow-up turn would be silently dropped and nothing
    else here would fail early."""
    opener = _msg("anyone here?")
    claim = _msg("@**EDI** take it")
    followup = _msg("and then?")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics, mir_mention_only=True)

    edi._dispatch(opener, set(), edi._config)
    mir._dispatch(opener, set(), mir._config)
    await _settle(edi, mir)
    # The untagged opener is unowned for both; mention_only gates both.
    assert edi._topic_owners[_key()] is TopicOwner.UNOWNED
    assert mir._topic_owners[_key()] is TopicOwner.UNOWNED
    assert ea.turns == [] and ma.turns == []

    edi._dispatch(claim, set(), edi._config)
    mir._dispatch(claim, set(), mir._config)
    await _settle(edi, mir)
    # The later bot mention moves the owner: EDI claims self, Miranda
    # hands it away.
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert mir._topic_owners[_key()] is TopicOwner.OTHER
    assert ea.turns == ["@**EDI** take it"]
    assert ma.turns == [] and mir._client.sends == []

    # The pin: an untagged follow-up still reaches EDI's model — the
    # recorded SELF exempts it from mention_only even though no Session
    # had existed when the claim landed.
    edi._dispatch(followup, set(), edi._config)
    mir._dispatch(followup, set(), mir._config)
    await _settle(edi, mir)
    assert ea.turns == ["@**EDI** take it", "and then?"]
    assert ma.turns == [] and mir._client.sends == []


async def test_human_opener_topic_reclaim_via_dispatch():
    """Topic whose opener mentioned a human (recorded OTHER) through
    ``_dispatch``: a later message mentioning this bot re-claims the
    topic — SELF again, its own turns admitted, the other bot back to
    dropping."""
    opener = _msg("@**Diego** heads up")
    reclaim = _msg("@**EDI** actually you")
    followup = _msg("follow-up")
    topics = {_key(): [opener]}
    edi, ea, mir, ma = _pair(topics, mir_mention_only=True)

    edi._dispatch(opener, set(), edi._config)
    mir._dispatch(opener, set(), mir._config)
    await _settle(edi, mir)
    # A human mention in the opener records OTHER for every bot.
    assert edi._topic_owners[_key()] is TopicOwner.OTHER
    assert mir._topic_owners[_key()] is TopicOwner.OTHER
    assert ea.turns == [] and ma.turns == []
    assert edi._client.sends == [] and mir._client.sends == []

    edi._dispatch(reclaim, set(), edi._config)
    mir._dispatch(reclaim, set(), mir._config)
    await _settle(edi, mir)
    # The bot mention re-claims for EDI and re-records OTHER for Miranda.
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert mir._topic_owners[_key()] is TopicOwner.OTHER
    assert ea.turns == ["@**EDI** actually you"]
    assert ma.turns == [] and mir._client.sends == []

    # Re-claimed ownership persists: the untagged follow-up runs for EDI
    # and stays dropped for Miranda.
    edi._dispatch(followup, set(), edi._config)
    mir._dispatch(followup, set(), mir._config)
    await _settle(edi, mir)
    assert ea.turns == ["@**EDI** actually you", "follow-up"]
    assert ma.turns == [] and mir._client.sends == []
    # One opener fetch per bot, cached as OTHER then re-recorded by the
    # later mention.
    assert edi._client.opener_calls == [_key()]
    assert mir._client.opener_calls == [_key()]


@pytest.mark.parametrize("content,expected", [
    pytest.param("@**Diego** hi", True, id="other-name"),
    pytest.param("@**Miranda|102** hi", True, id="other-name-id"),
    pytest.param("@_**Miranda** hi", True, id="other-silent"),
    pytest.param("@_**Miranda|102** hi", True, id="other-silent-id"),
    pytest.param("@**EDI** hi", False, id="self-name"),
    pytest.param("@**EDI|101** hi", False, id="self-name-id"),
    pytest.param("@_**EDI** hi", False, id="self-silent"),
    pytest.param("@**Anything|101** hi", True, id="stale-name-own-id"),
    pytest.param("@**all** hi", False, id="wildcard-all"),
    pytest.param("@**everyone** hi", False, id="wildcard-everyone"),
    pytest.param("@**channel** hi", False, id="wildcard-channel"),
    pytest.param("@**stream** hi", False, id="wildcard-stream"),
    pytest.param("@**topic** hi", False, id="wildcard-topic"),
    pytest.param("@*ops-team* hi", False, id="user-group"),
    pytest.param("plain @edi hi", False, id="plain-typed"),
    pytest.param("no mention at all", False, id="none"),
    pytest.param("@**EDI** and @**Miranda**", True, id="self-plus-other"),
])
def test_mentions_other_table(content, expected):
    assert mentions_other(
        content, full_name=EDI_NAME, user_id=EDI_ID) is expected


@pytest.mark.parametrize("content,expected", [
    pytest.param("@**Miranda** hi", True, id="other-bot-name"),
    pytest.param("@**Miranda|102** hi", True, id="other-bot-name-id"),
    pytest.param("@**Mira|102** hi", True, id="other-bot-by-id"),
    pytest.param("@_**Miranda** hi", True, id="other-bot-silent"),
    pytest.param("@**Diego** hi", False, id="human"),
    pytest.param("@_**Diego|5** [said](#narrow/near/1)", False,
                 id="quote-attribution"),
    pytest.param("@**EDI** hi", False, id="self-name"),
    pytest.param("@**EDI|101** hi", False, id="self-name-id"),
    pytest.param("@**all** hi", False, id="wildcard"),
    pytest.param("plain @miranda hi", False, id="plain-typed"),
    pytest.param("@**Diego** and @**Miranda**", True, id="human-plus-bot"),
])
def test_mentions_other_bot_table(content, expected):
    """Later-message rule, from EDI's view: only another BOT is a claim."""
    assert mentions_other_bot(
        content, full_name=EDI_NAME, user_id=EDI_ID,
        bot_ids={EDI_ID, MIRANDA_ID},
        bot_names={EDI_NAME, MIRANDA_NAME}) is expected


def test_bot_names_from_registration():
    """Only bot identities land in the name set — realm_users is_bot
    entries and realm_bots entries that carry a full_name."""
    registration = {
        "realm_users": [
            {"user_id": EDI_ID, "is_bot": True, "full_name": "EDI"},
            {"user_id": DIEGO_ID, "is_bot": False, "full_name": "Diego"},
            {"user_id": 103, "is_bot": True},
        ],
        "realm_bots": [
            {"bot_id": MIRANDA_ID, "full_name": "Miranda"},
            {"bot_id": 104},
        ],
    }
    assert bot_names_from_registration(registration) == {"EDI", "Miranda"}


# ---------------------------------------------------------------------------
# Client seam — get_topic_opener wire shape against a real aiohttp server
# ---------------------------------------------------------------------------


@pytest.fixture
async def zulip_endpoint():
    """Loopback Zulip server recording every request's method/path/query."""
    requests: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        requests.append({
            "method": request.method,
            "path": request.path,
            "query": dict(request.query),
        })
        return web.json_response({
            "messages": [
                {"id": 1, "sender_id": DIEGO_ID, "content": "the opener"}
            ],
            "result": "success", "msg": "",
        })

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:
        await runner.cleanup()


async def test_client_get_topic_opener_wire_shape(zulip_endpoint):
    url, requests = zulip_endpoint
    client = ZulipClient(
        credentials=ZulipCredentials(
            email="b@test.local", api_key="k", site="https://x.invalid"),
        config=ZulipConfig(zuliprc="/x", api_url=url, users=[]),
    )
    try:
        opener_resp = await client.get_topic_opener(CHANNEL, TOPIC)
        await client.get_topic_messages(CHANNEL, TOPIC)
    finally:
        await client.aclose()

    assert opener_resp["messages"][0]["id"] == 1
    opener_req, history_req = requests[0], requests[1]
    assert opener_req["method"] == "GET"
    assert opener_req["path"] == "/api/v1/messages"
    assert opener_req["query"]["anchor"] == "oldest"
    assert opener_req["query"]["num_before"] == "0"
    assert opener_req["query"]["num_after"] == "1"
    # Same narrow construction as the history read.
    assert opener_req["query"]["narrow"] == history_req["query"]["narrow"]
    assert json.loads(opener_req["query"]["narrow"]) == [
        {"operator": "channel", "operand": CHANNEL},
        {"operator": "topic", "operand": TOPIC},
    ]


async def test_topic_opener_lookup_through_real_client(zulip_endpoint):
    """Consumer-through-real-producer: ``_topic_opener`` unwraps the real
    client's ``messages[0]`` protocol answer."""
    url, _requests = zulip_endpoint
    client = ZulipClient(
        credentials=ZulipCredentials(
            email="b@test.local", api_key="k", site="https://x.invalid"),
        config=ZulipConfig(zuliprc="/x", api_url=url, users=[]),
    )
    iface, _agent = _iface(MIRANDA_NAME, MIRANDA_ID, client)
    try:
        opener = await iface._topic_opener(CHANNEL, TOPIC)
    finally:
        await client.aclose()
    assert opener == {"id": 1, "sender_id": DIEGO_ID, "content": "the opener"}


# ---------------------------------------------------------------------------
# spc-1084 — the !stop fast path races the queued first message for the
# same unrecorded topic. The opener fetch must run exactly once per topic
# (AC1), both callers must share the in-flight fetch, and the spc-1071
# routing is unchanged: the opener classification is recorded once and a
# later self-mention claim still overrides it.
# ---------------------------------------------------------------------------


class _DelayedClient(_FakeClient):
    """_FakeClient whose opener fetch holds for ``delay`` seconds, so a
    concurrent caller is guaranteed to arrive while the fetch is in
    flight (equal-latency race: both callers see the topic unrecorded)."""

    def __init__(self, topics: dict, *, delay: float = 0.03,
                 **kwargs: Any) -> None:
        super().__init__(topics, **kwargs)
        self._delay = delay

    async def get_topic_opener(self, channel_id: int, topic: str) -> dict:
        self.opener_calls.append((channel_id, topic))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._opener_fails:
            raise ZulipTransportError("synthetic opener failure")
        messages = self._topics.get((channel_id, topic)) or []
        return {"messages": messages[:1]}


async def test_ac1_concurrent_stop_shares_single_opener_fetch():
    """AC1, r1: a !stop dispatched alongside the first message in an
    unrecorded topic — one opener fetch total (previously two), the
    in-flight map is empty again once both callers settle, and the stop
    reply goes out through the fast path."""
    opener = _msg("@**EDI** status?")
    follow = _msg("and then?")
    topics = {_key(): [opener]}
    client = _DelayedClient(topics)
    edi, ea = _iface(EDI_NAME, EDI_ID, client)

    # Same-tick dispatch: the queued message and the !stop fast path both
    # find the topic unrecorded and both await the in-flight fetch.
    edi._dispatch(follow, set(), edi._config)
    edi._dispatch(_msg("!stop"), set(), edi._config)
    await _settle(edi)

    assert client.opener_calls == [_key()]              # exactly one fetch
    assert edi._topic_opener_inflight == {}             # no leaked entry
    assert edi._topic_owners[_key()] is TopicOwner.SELF  # opener class stands
    assert [s["content"] for s in client.sends if
            s["content"] == edi._config.stop_idle_message] == [
        edi._config.stop_idle_message
    ]


async def test_ac2_probe_opener_class_still_replies_and_single_fetch():
    """AC2, r1 — the G probe shape: opener mentions a human (OTHER class),
    a self-mention message and a !stop are dispatched concurrently against
    the same unrecorded topic. One opener fetch total (previously two);
    the !stop runs through the fast path (no turn queue); the opener's
    OTHER class is recorded exactly once; and a later self-mention claims
    the topic back without a second opener fetch."""
    opener = _msg("@**Diego** help")         # human mention -> OTHER
    follow = _msg("@**EDI** help")            # self claim -> SELF
    topics = {_key(): [opener]}
    client = _DelayedClient(topics)
    edi, ea = _iface(EDI_NAME, EDI_ID, client)

    # Same-tick dispatch: both callers find the topic unrecorded and both
    # await the in-flight fetch. The !stop fast path resolves first (it has
    # no turn to run), records the opener's OTHER class, and is dropped
    # silently (no reply). The queued message then claims SELF.
    edi._dispatch(follow, set(), edi._config)
    edi._dispatch(_msg("!stop"), set(), edi._config)
    await _settle(edi)

    assert client.opener_calls == [_key()]              # exactly one fetch
    assert edi._topic_opener_inflight == {}             # no leaked entry
    # The opener's OTHER class was recorded; the follow-up self-mention
    # then claimed the topic back (spc-1071 rule, unchanged).
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert ea.turns == ["@**EDI** help"]

    # A further !stop in the now-SELF topic gets its stop reply.
    before = len(client.sends)
    edi._dispatch(_msg("!stop"), set(), edi._config)
    await _settle(edi)
    assert [s["content"] for s in client.sends[before:] if
            s["content"] == edi._config.stop_idle_message] == [
        edi._config.stop_idle_message
    ]
    # No second opener fetch.
    assert client.opener_calls == [_key()]


async def test_ac2_opener_class_recorded_once_and_claim_overrides():
    """AC2 ordering: in the bare-!stop probe, the opener's OTHER class is
    recorded exactly once (not overwritten by either caller), the !stop is
    dropped silently (no reply), and a later self-mention still claims the
    topic without a second opener fetch."""
    opener = _msg("@**Diego** help")
    topics = {_key(): [opener]}
    client = _DelayedClient(topics)
    edi, ea = _iface(EDI_NAME, EDI_ID, client)

    edi._dispatch(_msg("!stop"), set(), edi._config)
    await _settle(edi)

    assert client.opener_calls == [_key()]
    assert edi._topic_owners[_key()] is TopicOwner.OTHER
    assert client.sends == []                            # dropped, no reply

    # A later self-mention claims the topic back (spc-1071 rule intact)
    # and fetches nothing further.
    edi._dispatch(_msg("@**EDI** help"), set(), edi._config)
    await _settle(edi)
    assert client.opener_calls == [_key()]
    assert edi._topic_owners[_key()] is TopicOwner.SELF
    assert ea.turns == ["@**EDI** help"]


async def test_ac1_concurrent_failure_still_single_fetch_and_retries():
    """AC1 fail-open: a failed fetch is shared by the concurrent !stop too
    (one attempt for the race), nothing is recorded, and the next event
    retries the fetch — never a lockout, never a second in-flight fetch."""
    topics = {_key(): [_msg("anyone?")]}
    client = _DelayedClient(topics, opener_fails=True)
    edi, ea = _iface(EDI_NAME, EDI_ID, client)

    edi._dispatch(_msg("first"), set(), edi._config)
    edi._dispatch(_msg("!stop"), set(), edi._config)
    await _settle(edi)

    assert client.opener_calls == [_key()]          # one attempt for the race
    assert edi._topic_owners == {}                  # failed: recorded nothing
    assert edi._topic_opener_inflight == {}

    # Next event retries (the fail-open contract, spc-1071).
    client._opener_fails = False
    edi._dispatch(_msg("second"), set(), edi._config)
    await _settle(edi)
    assert client.opener_calls == [_key(), _key()]
    assert edi._topic_owners[_key()] is TopicOwner.UNOWNED
