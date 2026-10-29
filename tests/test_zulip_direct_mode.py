"""Direct-profile cage — spc-992 (promote the DM-only Zulip channel to a
config-selectable framework interface mode).

What this file pins, seam by seam:

- ``admits_human_trigger`` — the ONE admission core (bot-never-triggers
  outranks the allowlist; strict integer membership; empty ``users`` admits
  any human, spc-1013). The stream front door ``is_configured_no_area_channel``
  and the DM wrapper ``admits_direct_message`` are tested against the SAME
  allowlist/bot case matrix: the parametrized rows below call both entry
  points, so a second drifting copy of the rule cannot pass.
- ``ZulipConfig`` mode fields — ``mode`` defaults to ``"stream"`` (existing
  configs parse unchanged), ``peer_id`` is strict (a string never loads),
  an unknown mode is rejected, and ``validate_direct_mode`` fails startup
  with an explicit error when direct mode has no peer.
- ``ZulipClient.send_direct`` — the JSON-list private destination is encoded
  on the client (form fields captured at the HTTP boundary; no real server).
- ``ZulipDMInterface`` — the promoted channel core (start/admits/poll/send/
  should_consider_beat/stop) and the Interface-protocol DM loop: one Session
  with the explicit no-area ``RunContext``, serialized turns, replies via
  the client's DM send, bot identity from registration metadata including
  the bot's own ``user_id``. No network, no model, no subprocess.
- ``cli._load_interface`` — ``--interface zulip`` selects the DM class when
  the resolved zulip block says ``mode: direct``, and the stream class for
  the default and for explicit ``mode: stream``.

Existing stream/topic behavior is caged by the pre-existing zulip files,
which this change leaves untouched.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from pydantic import ValidationError

from sr2_spectre.interfaces.zulip import (
    ZulipClient,
    ZulipConfig,
    ZulipDMInterface,
    ZulipError,
    ZulipInterface,
    admits_direct_message,
    admits_human_trigger,
    is_configured_no_area_channel,
)
from sr2_spectre.interfaces.zulip.config import (
    bot_ids_from_registration,
    validate_direct_mode,
)

# Identities mirror the spec's Dependencies (Miranda bot 9, Diego 8).
BOT_ID = 9
DIEGO_ID = 8
INTRUDER_ID = 7
STRANGER_BOT_ID = 11
PEER_ID = DIEGO_ID

ZULIPRC = "/tmp/spc-992-unused.zuliprc"        # never read: no credential I/O
STREAM_CHANNEL = 42


def _stream_config(**kw: Any) -> ZulipConfig:
    base: dict[str, Any] = {
        "zuliprc": ZULIPRC,
        "users": [DIEGO_ID],
        "channels": [STREAM_CHANNEL],
        "channel_areas": {str(STREAM_CHANNEL): ""},
    }
    base.update(kw)
    return ZulipConfig(**base)


def _direct_config(**kw: Any) -> ZulipConfig:
    base: dict[str, Any] = {
        "zuliprc": ZULIPRC,
        "users": [DIEGO_ID],
        "mode": "direct",
        "peer_id": PEER_ID,
    }
    base.update(kw)
    return ZulipConfig(**base)


# ---------------------------------------------------------------------------
# AC1: ONE admission rule — the same case matrix through both entry points.
# ---------------------------------------------------------------------------

# (channel, sender, sender_is_bot, admit?) — the security-relevant cases.
ADMISSION_MATRIX = [
    pytest.param(DIEGO_ID, False, True, id="allowed-human"),
    pytest.param(INTRUDER_ID, False, False, id="off-allowlist-human"),
    pytest.param(BOT_ID, True, False, id="allowlisted-bot-never-triggers"),
    pytest.param(STRANGER_BOT_ID, True, False, id="stranger-bot-never-triggers"),
    pytest.param("8", False, False, id="stringified-id-never-matches"),
    pytest.param(None, False, False, id="missing-id-never-matches"),
]


@pytest.mark.parametrize("sender_id,is_bot,expected", ADMISSION_MATRIX)
def test_shared_core_admits_only_allowlisted_humans(sender_id, is_bot, expected):
    assert admits_human_trigger(
        sender_id=sender_id, sender_is_bot=is_bot, users=[DIEGO_ID, BOT_ID]
    ) is expected


@pytest.mark.parametrize("sender_id,is_bot,expected", ADMISSION_MATRIX)
def test_stream_entry_point_matches_the_shared_core(sender_id, is_bot, expected):
    """The stream mode's decision on the allowlist/bot axis IS the core."""
    assert is_configured_no_area_channel(
        channel_id=STREAM_CHANNEL,
        sender_id=sender_id,
        sender_is_bot=is_bot,
        config=_stream_config(users=[DIEGO_ID, BOT_ID]),
    ) is expected


@pytest.mark.parametrize("sender_id,is_bot,expected", ADMISSION_MATRIX)
def test_direct_entry_point_matches_the_shared_core(sender_id, is_bot, expected):
    """The DM mode runs the SAME matrix: same allowlist, same bot rule.

    ``is_bot`` marks membership in the realm-metadata bot set the wrapper
    is handed; the message shape (private, integer id) is fixed here so
    only the shared policy varies.
    """
    bot_ids = {BOT_ID, STRANGER_BOT_ID} if is_bot else set()
    assert admits_direct_message(
        message={"type": "private", "id": 1, "sender_id": sender_id},
        users=[DIEGO_ID, BOT_ID],
        bot_ids=bot_ids,
        peer_id=None,
    ) is expected


def test_both_modes_cover_the_same_matrix_cases():
    """Guard for the parametrization above: both entry points run every
    case id of the shared matrix, so 'framework tests cover both modes
    against the same allowlist/bot cases' stays checkable."""
    ids = {param.id for param in ADMISSION_MATRIX}
    assert "allowed-human" in ids and "allowlisted-bot-never-triggers" in ids


def test_stream_only_rules_stay_in_stream():
    """An unconfigured channel never admits in stream mode; channel
    membership stays a stream-shape check (direct mode has no channels)."""
    assert not is_configured_no_area_channel(
        channel_id=99, sender_id=DIEGO_ID, sender_is_bot=False,
        config=_stream_config(),
    )


def test_empty_users_admits_any_human_in_both_modes():
    """spc-1013 (FR6): an empty ``users`` admits any human."""
    assert admits_human_trigger(
        sender_id=DIEGO_ID, sender_is_bot=False, users=[]
    )
    assert is_configured_no_area_channel(
        channel_id=STREAM_CHANNEL, sender_id=DIEGO_ID, sender_is_bot=False,
        config=_stream_config(users=[]),
    )
    assert admits_direct_message(
        message={"type": "private", "id": 1, "sender_id": DIEGO_ID},
        users=[], bot_ids=set(), peer_id=None,
    )


# ---------------------------------------------------------------------------
# DM-shape checks (added by the wrapper, not by the core).
# ---------------------------------------------------------------------------

def _admits(message: dict, **kw: Any) -> bool:
    params: dict[str, Any] = dict(
        users=[DIEGO_ID], bot_ids={BOT_ID}, peer_id=None
    )
    params.update(kw)
    return admits_direct_message(message=message, **params)


def test_direct_shape_stream_post_never_admits():
    assert not _admits(
        {"type": "stream", "id": 1, "sender_id": DIEGO_ID, "stream_id": STREAM_CHANNEL}
    )


def test_direct_shape_string_ids_never_admit():
    assert not _admits({"type": "private", "id": 1, "sender_id": "8"})
    assert not _admits({"type": "private", "id": "1", "sender_id": DIEGO_ID})
    assert not _admits({"type": "private", "sender_id": DIEGO_ID})


def test_direct_shape_peer_filter():
    """With a peer configured, only that peer's DMs admit — an allowlisted
    stranger still never does; without one, the allowlist alone decides."""
    stranger = {"type": "private", "id": 1, "sender_id": INTRUDER_ID}
    assert _admits(stranger, users=[DIEGO_ID, INTRUDER_ID], peer_id=None)
    assert not _admits(
        stranger, users=[DIEGO_ID, INTRUDER_ID], peer_id=DIEGO_ID
    )
    assert _admits(
        {"type": "private", "id": 2, "sender_id": DIEGO_ID},
        users=[DIEGO_ID, INTRUDER_ID], peer_id=DIEGO_ID,
    )


def test_direct_shape_bots_own_account_never_admits():
    """Her own reply never triggers, even absent from the realm bot set."""
    assert not _admits(
        {"type": "private", "id": 1, "sender_id": BOT_ID},
        users=[BOT_ID], bot_ids=set(), bot_user_id=BOT_ID, peer_id=None,
    )


def test_bot_ids_from_registration_reads_both_arrays():
    bot_ids = bot_ids_from_registration({
        "realm_users": [
            {"user_id": DIEGO_ID, "is_bot": False},
            {"user_id": BOT_ID, "is_bot": True},
            {"user_id": "12", "is_bot": True},   # malformed: not an int
        ],
        "realm_bots": [{"bot_id": STRANGER_BOT_ID}, {"bot_id": "x"}],
    })
    assert bot_ids == {BOT_ID, STRANGER_BOT_ID}


# ---------------------------------------------------------------------------
# AC2: config — mode defaults, strictness, startup errors.
# ---------------------------------------------------------------------------

def test_mode_defaults_to_stream_existing_configs_parse_unchanged():
    cfg = ZulipConfig(
        zuliprc=ZULIPRC, users=[DIEGO_ID], channels=[STREAM_CHANNEL]
    )
    assert cfg.mode == "stream"
    assert cfg.peer_id is None


def test_mode_direct_and_peer_id_parse():
    cfg = _direct_config()
    assert cfg.mode == "direct" and cfg.peer_id == DIEGO_ID


def test_string_peer_id_rejected_by_the_strict_schema():
    with pytest.raises(ValidationError):
        ZulipConfig(zuliprc=ZULIPRC, mode="direct", peer_id="8")


def test_unknown_mode_rejected():
    with pytest.raises(ValidationError):
        ZulipConfig(zuliprc=ZULIPRC, mode="carrier-pigeon")


def test_validate_direct_mode_requires_peer():
    with pytest.raises(ValueError, match="peer_id"):
        validate_direct_mode(_direct_config(peer_id=None))
    assert validate_direct_mode(_direct_config()) is None


# ---------------------------------------------------------------------------
# AC3: outbound DM send lives on the client (JSON-list `to`).
# ---------------------------------------------------------------------------


def _client_for_form_capture(monkeypatch, captured: list) -> ZulipClient:
    """A real ZulipClient whose HTTP layer answers a canned send result.

    Only the aiohttp request is substituted, at the boundary, so the form
    fields the client actually puts on the wire are what we assert.
    """

    class _Resp:
        status = 200

        async def read(self):
            return json.dumps({"result": "success", "id": 4242}).encode()

    class _Ctx:
        def __init__(self, method, url, **kwargs):
            captured.append({"method": method, "url": url, "kwargs": kwargs})

        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def request(self, method, url, **kwargs):
            return _Ctx(method, url, **kwargs)

        async def close(self):
            pass

    client = ZulipClient.__new__(ZulipClient)  # no real HTTP session
    client._origin = "http://127.0.0.1:1"
    client._timeout = 5.0
    client._event_poll_timeout = 5.0
    client._session = _Session()
    return client


def test_send_direct_encodes_json_list_destination_on_the_client():
    captured: list = []
    client = _client_for_form_capture(None, captured)
    result = asyncio.run(client.send_direct(DIEGO_ID, "hi"))
    assert result == 4242
    assert len(captured) == 1
    form = captured[0]["kwargs"]["data"]
    assert form["type"] == "private"
    assert json.loads(form["to"]) == [DIEGO_ID]
    assert form["content"] == "hi"


# ---------------------------------------------------------------------------
# AC4/AC6: ZulipDMInterface — the promoted core + the Interface DM loop.
# ---------------------------------------------------------------------------

REGISTRATION = {
    "queue_id": "q-992",
    "last_event_id": 0,
    # The register-queue contract: the bot's own user_id at top level,
    # realm_users with is_bot flags, and realm_bots.
    "user_id": BOT_ID,
    "realm_users": [
        {"user_id": DIEGO_ID, "is_bot": False},
        {"user_id": BOT_ID, "is_bot": True},
        {"user_id": STRANGER_BOT_ID, "is_bot": True},
    ],
    "realm_bots": [{"bot_id": STRANGER_BOT_ID}],
}


class FakeDMClient:
    """Records sends; replays scripted event batches; never touches sockets."""

    def __init__(self, batches: list[list[dict]] | None = None) -> None:
        self.batches = list(batches or [])
        self.sends: list[tuple[int, str]] = []
        self.register_calls = 0
        self.closed = False

    async def register(self):
        self.register_calls += 1
        return dict(REGISTRATION)

    async def get_events(self, queue_id, last_event_id=-1):
        assert queue_id == "q-992"
        if not self.batches:
            await asyncio.sleep(0.01)
            return {"events": []}
        return {"events": self.batches.pop(0)}

    async def send_direct(self, to, content):
        self.sends.append((to, content))
        return 5000 + len(self.sends)

    async def aclose(self):
        self.closed = True


def _dm_event(msg_id: int, sender_id: Any, content: str = "hey") -> dict:
    return {
        "id": msg_id,
        "type": "message",
        "message": {
            "id": 1000 + msg_id,
            "type": "private",
            "sender_id": sender_id,
            "content": content,
        },
    }


@pytest.fixture
def patched_credentials(monkeypatch):
    """No credential I/O, no real ZulipClient construction."""
    import sr2_spectre.interfaces.zulip.direct as direct_mod

    monkeypatch.setattr(
        direct_mod,
        "load_zulip_credentials",
        lambda path: type(
            "Creds", (), {"email": "bot@test.local", "api_key": "k",
                          "site": "http://127.0.0.1:1"}
        )(),
    )
    return direct_mod


def _dm_interface(patched_credentials, client: FakeDMClient, config=None):
    iface = ZulipDMInterface(config or _direct_config())
    patched_credentials.ZulipClient = lambda **kw: client
    return iface


class StubSession:
    def __init__(self, reply: str = "reply-text") -> None:
        self.reply = reply
        self.run_context: Any = None
        self.turns: list[str] = []
        self.closed = False
        self.fail = False

    def set_run_context(self, context) -> None:
        self.run_context = context

    async def handle_user_message(self, text):
        if self.fail:
            raise RuntimeError("brain down")
        from sr2_spectre.core import TurnResult
        self.turns.append(text)
        return TurnResult(text=self.reply)

    def close(self) -> None:
        self.closed = True


class StubAgent:
    def __init__(self, session: StubSession) -> None:
        self._session = session
        self.frame_ids: list[str] = []
        self.config = type(
            "Cfg", (), {"agent": type("A", (), {"name": "iris"})()}
        )()
        self.applied: list[Any] = []

    def create_session(self, frame_id: str):
        self.frame_ids.append(frame_id)
        return self._session

    def apply_config(self, cfg) -> None:
        self.applied.append(cfg)


def test_implements_the_interface_protocol():
    from sr2_spectre.interfaces import Interface

    iface = ZulipDMInterface(_direct_config())
    assert isinstance(iface, Interface)
    assert iface.name == "zulip"


def test_start_registers_queue_and_reads_bot_identity(patched_credentials):
    """Bot identity (including her OWN user_id) comes from registration —
    no extra HTTP identity call. ``_started`` marks the registered-queue
    state (the promoted channel's public readiness fact)."""
    client = FakeDMClient()
    iface = _dm_interface(patched_credentials, client)
    assert iface._started is False
    asyncio.run(iface.start(agent=None))
    assert client.register_calls == 1
    assert iface.bot_user_id == BOT_ID
    assert iface._started is True
    assert {STRANGER_BOT_ID}.issubset(iface._bot_ids)
    asyncio.run(iface.stop())
    assert iface._started is False


def test_start_without_peer_fails_explicitly(patched_credentials):
    """Even a block that reached start() without an integer peer_id (a
    caller bypassing the schema) fails startup with the declared error,
    and start() opens no transport when it does."""
    iface = _dm_interface(patched_credentials, FakeDMClient())
    iface._config = iface._config.model_copy(update={"peer_id": None})
    with pytest.raises(ValueError, match="peer_id"):
        asyncio.run(iface.start(agent=None))
    assert iface._client is None


def test_construction_without_peer_fails_explicitly():
    """Direct mode requires its peer at construction, before any I/O —
    on the fixed-config path and on the config-source path alike."""
    with pytest.raises(ValueError, match="peer_id"):
        ZulipDMInterface(_direct_config(peer_id=None))

    import sr2_spectre.interfaces.zulip.direct as direct_mod

    class _StubView:
        """Stands in for ZulipConfigView: slice in force has no peer."""

        def __init__(self, source):
            self.source = source

        @property
        def current(self):
            return _direct_config(peer_id=None)

    original = direct_mod.ZulipConfigView
    direct_mod.ZulipConfigView = _StubView
    try:
        with pytest.raises(ValueError, match="peer_id"):
            ZulipDMInterface(config_source=object())
    finally:
        direct_mod.ZulipConfigView = original


def test_admits_uses_registration_bot_metadata(patched_credentials):
    client = FakeDMClient()
    iface = _dm_interface(patched_credentials, client)
    asyncio.run(iface.start(agent=None))
    assert iface.admits({"type": "private", "id": 1, "sender_id": DIEGO_ID})
    assert not iface.admits({"type": "private", "id": 2, "sender_id": BOT_ID})
    assert not iface.admits(
        {"type": "private", "id": 3, "sender_id": STRANGER_BOT_ID}
    )
    assert not iface.admits({"type": "private", "id": 4, "sender_id": INTRUDER_ID})
    assert not iface.admits(
        {"type": "stream", "id": 5, "sender_id": DIEGO_ID, "stream_id": 1}
    )


def test_poll_returns_admitted_dmshaped_messages(patched_credentials):
    client = FakeDMClient([[
        _dm_event(1, DIEGO_ID, "hey"),
        _dm_event(2, BOT_ID, "reply"),            # own account: never
        _dm_event(3, STRANGER_BOT_ID, "bot bcast"),
        _dm_event(4, INTRUDER_ID, "hi"),
        {"id": 5, "type": "presence"},
        {"id": 6, "type": "message",
         "message": {"id": 6, "type": "private", "sender_id": "8",
                     "content": "string id"}},
    ]])
    iface = _dm_interface(patched_credentials, client)
    asyncio.run(iface.start(agent=None))
    admitted = asyncio.run(iface.poll())
    assert [(m.message_id, m.sender_id, m.text) for m in admitted] == [
        (1001, DIEGO_ID, "hey")
    ]


def test_send_targets_peer_via_client(patched_credentials):
    client = FakeDMClient()
    iface = _dm_interface(patched_credentials, client)
    asyncio.run(iface.start(agent=None))
    assert asyncio.run(iface.send("hi diego")) == 5001
    assert client.sends == [(DIEGO_ID, "hi diego")]


def test_should_consider_beat_quiet_period(patched_credentials):
    iface = ZulipDMInterface(_direct_config(), quiet_seconds=999.0)
    assert iface.should_consider_beat(None) is True
    now = time.monotonic()
    assert iface.should_consider_beat(now) is False       # mid-exchange
    assert iface.should_consider_beat(now - 1000.0) is True
    iface_fast = ZulipDMInterface(_direct_config(), quiet_seconds=0.0)
    assert iface_fast.should_consider_beat(now) is True


def _drive_run(iface, agent, until, timeout=5.0):
    """Run iface.run until *until()* is true or timeout; then cancel.

    ``run`` long-polls the (fake) queue between batches, so the wait and
    the cancellation are both bounded here.
    """

    async def _main():
        task = asyncio.ensure_future(iface.run(agent))
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if until():
                    return True
                await asyncio.sleep(0.01)
            return False
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    return asyncio.run(_main())


def test_run_handles_admitted_dm_end_to_end(patched_credentials):
    client = FakeDMClient([[
        _dm_event(1, BOT_ID, "own reply"),
        _dm_event(2, DIEGO_ID, "hey iris"),
    ]])
    iface = _dm_interface(patched_credentials, client)
    session = StubSession(reply="hey you")
    agent = StubAgent(session)
    asyncio.run(iface.start(agent=agent))
    assert _drive_run(iface, agent, lambda: len(client.sends) >= 1)
    assert agent.frame_ids and agent.frame_ids[0].startswith("zulip-dm:iris:")
    assert agent.frame_ids[0].endswith(f"#{DIEGO_ID}")
    context = session.run_context
    assert context.area == ""                 # explicit no-area, never None
    assert context.mode.name == "INTERACTIVE"
    assert session.turns == ["hey iris"]
    assert client.sends == [(DIEGO_ID, "hey you")]
    # The ignored own-account event advanced the cursor: never re-delivered.
    assert iface._last_event_id >= 2


def test_run_reuses_one_session_and_survives_turn_failure(patched_credentials):
    client = FakeDMClient([
        [_dm_event(1, DIEGO_ID, "one")],
        [_dm_event(2, DIEGO_ID, "two")],
    ])
    iface = _dm_interface(patched_credentials, client)
    session = StubSession(reply="ok")
    agent = StubAgent(session)
    asyncio.run(iface.start(agent=agent))
    assert _drive_run(iface, agent, lambda: len(client.sends) >= 1)
    session.fail = False
    assert _drive_run(iface, agent, lambda: len(client.sends) >= 2)
    assert agent.frame_ids == [agent.frame_ids[0]]      # ONE Session
    assert session.turns == ["one", "two"]
    # A failing turn: logged, not raised; the loop keeps serving.
    client.batches.append([_dm_event(3, DIEGO_ID, "three")])
    session.fail = True
    assert _drive_run(iface, agent, lambda: iface._last_event_id >= 3)
    assert len(client.sends) == 2


def test_stop_closes_session_and_transport(patched_credentials):
    client = FakeDMClient()
    iface = _dm_interface(patched_credentials, client)
    session = StubSession()
    agent = StubAgent(session)
    asyncio.run(iface.start(agent=agent))
    iface._session = session
    asyncio.run(iface.stop())
    assert session.closed and client.closed


def test_poll_transport_error_propagates(patched_credentials):
    class BoomClient(FakeDMClient):
        async def get_events(self, queue_id, last_event_id=-1):
            raise ZulipError("down")

    iface = _dm_interface(patched_credentials, BoomClient())
    asyncio.run(iface.start(agent=None))
    with pytest.raises(ZulipError):
        asyncio.run(iface.poll())


# ---------------------------------------------------------------------------
# AC5: cli registration — mode: direct selects the DM class.
# ---------------------------------------------------------------------------

def test_cli_zulip_defaults_to_stream_interface():
    from sr2_spectre.cli import _load_interface

    assert _load_interface("zulip") is ZulipInterface
    assert _load_interface("zulip", config=_stream_config()) is ZulipInterface


def test_cli_zulip_direct_mode_selects_dm_interface():
    from sr2_spectre.cli import _load_interface

    assert _load_interface("zulip", config=_direct_config()) is ZulipDMInterface


def test_cli_zulip_via_loaded_config_block_selects_dm():
    """The block as it arrives from YAML resolution drives the choice."""
    import yaml

    from sr2_spectre.cli import _load_interface

    block = yaml.safe_load(
        "zuliprc: /tmp/x\nusers: [8]\nmode: direct\npeer_id: 8\n"
    )
    cfg = ZulipConfig(**block)
    assert _load_interface("zulip", config=cfg) is ZulipDMInterface
