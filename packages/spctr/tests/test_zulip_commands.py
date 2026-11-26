"""spctr-2 — the ``!model`` command in the Zulip interface (AC1–AC5).

The real ``ZulipInterface`` runs against a fake agent whose config carries
the model profiles; a fake transport records the replies. Each command is
routed through ``_handle_message`` exactly as the poller delivers it (the
accepted pre-routing call convention, ``reload=False``), so the intercept
sits where production sits: after the ``!stop`` check, before any Session
machinery.

The expected profile listing follows the config's declaration order, as
AC3's literal message does (base.yaml order) — not REPL's sorted() form.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from spctr.events import AgentTextDelta
from spctr.interfaces.zulip import ZulipConfig, ZulipInterface
from spctr.interfaces.zulip.client import ZulipCredentials
from spctr.interfaces.zulip.interface import is_model_command, parse_model_command
from spctr.resume import ResumeResult

DIEGO_ID = 5
CHANNEL = 2
TOPIC = "status"

# The profiles the fake config declares, in declaration order — AC3's
# literal listing format against a current config.
PROFILES = [
    "default",
    "qwen3.6",
    "coding",
    "instruct",
    "default-xtx",
    "default-5090",
    "uncensored",
]
AVAILABLE = ", ".join(PROFILES)


class _FakeConfig:
    """Minimal stand-in for ``SpctrConfig``'s model-switch surface."""

    def __init__(self, active_model: str = "default") -> None:
        self.agent = SimpleNamespace(name="EDI")
        self.models = {name: SimpleNamespace(name=name) for name in PROFILES}
        self.active_model = active_model

    def model_copy(self, update: dict[str, Any]) -> "_FakeConfig":
        copy = _FakeConfig(self.active_model)
        for key, value in update.items():
            setattr(copy, key, value)
        return copy


class _FakeSession:
    async def restore_history(self) -> ResumeResult:
        return ResumeResult([], interrupted=False, restore_failed=False)

    def __init__(self) -> None:
        self.turns: list[str] = []

    def set_run_context(self, context: Any) -> None:
        pass

    async def stream_message(self, text: str) -> Any:
        self.turns.append(text)
        yield AgentTextDelta(text=f"reply to {text}")

    def close(self) -> None:
        pass


class _FakeAgent:
    def __init__(self, active_model: str = "default") -> None:
        self.config = _FakeConfig(active_model)
        self.applied: list[str] = []
        self.sessions: dict[str, _FakeSession] = {}
        self.creations = 0

    def create_session(self, frame_id: str) -> _FakeSession:
        self.creations += 1
        session = _FakeSession()
        self.sessions[frame_id] = session
        return session

    def apply_config(self, config: Any) -> None:
        self.applied.append(getattr(config, "active_model", ""))
        self.config = config

    @property
    def turns(self) -> list[str]:
        return [t for s in self.sessions.values() for t in s.turns]


class _FakeClient:
    def __init__(self) -> None:
        self.sends: list[dict] = []

    async def register(self) -> dict:
        return {"queue_id": "q", "last_event_id": 0}

    async def get_events(self, queue_id: str, last_event_id: int) -> dict:
        return {"events": []}

    async def get_topic_messages(self, channel_id: int, topic: str) -> dict:
        return {"messages": []}

    async def send_message(self, **kwargs: Any) -> int:
        self.sends.append(kwargs)
        return len(self.sends)

    async def update_message(self, message_id: int, content: str) -> None:
        self.sends[message_id - 1] = {**self.sends[message_id - 1], "content": content}

    async def aclose(self) -> None:
        pass


_next_id = [1000]


def _msg(content: str, sender: int = DIEGO_ID) -> dict:
    _next_id[0] += 1
    return {"id": _next_id[0], "type": "stream", "stream_id": CHANNEL,
            "display_recipient": "sandbox", "subject": TOPIC,
            "sender_id": sender, "content": content}


def _iface(active_model: str = "default"):
    config = ZulipConfig(
        zuliprc="/unused.zuliprc", api_url="http://zulip.test",
        users=[DIEGO_ID], edit_stream_interval=0,
    )
    iface = ZulipInterface(config)
    agent = _FakeAgent(active_model)
    iface._agent = agent
    iface._client = _FakeClient()
    iface._credentials = ZulipCredentials(
        email="b@test.local", api_key="k", site="http://zulip.test")
    iface._bot_ids = set()
    iface._bot_names = set()
    iface._bot_user_id = None
    iface._bot_full_name = None
    return iface, agent


async def _settle(iface: ZulipInterface) -> None:
    await asyncio.gather(
        *iface._topic_tails.values(), return_exceptions=True
    )


async def _send(iface: ZulipInterface, content: str) -> None:
    await iface._handle_message(_msg(content), reload=False)
    await _settle(iface)


def _reply(client: _FakeClient) -> str:
    assert len(client.sends) == 1
    return client.sends[0]["content"]


# ---------------------------------------------------------------------------
# Helpers under test
# ---------------------------------------------------------------------------


def test_is_model_command_matches_prefix_and_bare():
    assert is_model_command("!model coding")
    assert is_model_command("!model")
    assert is_model_command("  !model coding  ")
    assert is_model_command("!MODEL coding")
    assert not is_model_command("!modeling coding")
    assert not is_model_command("!modelcoding")
    assert not is_model_command("!stop")
    assert not is_model_command("regular chat")


def test_parse_model_command_extracts_and_lowers():
    assert parse_model_command("!model Coding") == "coding"
    assert parse_model_command("!MODEL  DEFAULT ") == "default"
    assert parse_model_command("!model") == ""


# ---------------------------------------------------------------------------
# AC1 — a valid profile switches the active model; no session, no turn
# ---------------------------------------------------------------------------


async def test_ac1_valid_profile_switches_model():
    iface, agent = _iface()
    await _send(iface, "!model coding")

    assert agent.config.active_model == "coding"
    assert _reply(iface._client) == "Switched to model 'coding'."
    assert agent.creations == 0 and agent.turns == []


# ---------------------------------------------------------------------------
# AC2 — !model default switches back to the default profile
# ---------------------------------------------------------------------------


async def test_ac2_default_switches_back():
    iface, agent = _iface(active_model="coding")
    await _send(iface, "!model default")

    assert agent.config.active_model == "default"
    assert _reply(iface._client) == "Switched to model 'default'."


# ---------------------------------------------------------------------------
# AC3 — an unknown profile errors with the declaration-order listing
# ---------------------------------------------------------------------------


async def test_ac3_unknown_profile_lists_available():
    iface, agent = _iface()
    await _send(iface, "!model invalid_profile")

    assert _reply(iface._client) == (
        f"Unknown model 'invalid_profile'. Available: {AVAILABLE}"
    )
    # No switch happened and no override was recorded.
    assert agent.config.active_model == "default"
    assert iface._active_model_override is None


# ---------------------------------------------------------------------------
# AC4 — the command prefix is case-insensitive
# ---------------------------------------------------------------------------


async def test_ac4_prefix_case_insensitive():
    for text in ("!Model coding", "!MODEL coding", "!MoDeL coding"):
        iface, agent = _iface()
        await _send(iface, text)
        assert agent.config.active_model == "coding", text
        assert _reply(iface._client) == "Switched to model 'coding'."


# ---------------------------------------------------------------------------
# AC5 — non-command messages are processed normally
# ---------------------------------------------------------------------------


async def test_ac5_non_commands_unchanged():
    iface, agent = _iface()

    await _send(iface, "hello there")
    assert agent.creations == 1 and agent.turns == ["hello there"]
    assert iface._client.sends, "the turn still answers"

    # A word sharing the prefix is a regular message, not the command.
    iface2, agent2 = _iface()
    await _send(iface2, "!modeling coding")
    assert agent2.creations == 1 and agent2.turns == ["!modeling coding"]
    assert agent2.config.active_model == "default"


# ---------------------------------------------------------------------------
# A bare !model lists without switching
# ---------------------------------------------------------------------------


async def test_bare_model_lists_without_switching():
    iface, agent = _iface()
    await _send(iface, "!model")

    assert _reply(iface._client) == f"Available models: {AVAILABLE}"
    assert agent.config.active_model == "default"
    assert iface._active_model_override is None


# ---------------------------------------------------------------------------
# The selection survives a whole-config reload (the on-disk active_model
# would otherwise clobber it on the next inbound batch)
# ---------------------------------------------------------------------------


async def test_model_switch_survives_config_reload():
    iface, agent = _iface()
    await _send(iface, "!model coding")
    assert agent.config.active_model == "coding"

    # A config edit reloads the whole file: its own active_model is
    # 'default' again, and the Zulip slice is a fresh read.
    disk = _FakeConfig(active_model="default")
    iface._view = SimpleNamespace(
        reload=lambda: iface._config,
        source=SimpleNamespace(current=disk),
    )
    iface._reload_config()

    assert agent.config.active_model == "coding"


async def test_reload_without_switch_is_unchanged():
    iface, agent = _iface()
    disk = _FakeConfig(active_model="coding")
    iface._view = SimpleNamespace(
        reload=lambda: iface._config,
        source=SimpleNamespace(current=disk),
    )
    iface._reload_config()

    assert agent.config.active_model == "coding"


# ---------------------------------------------------------------------------
# spc-1122 — !model reply texts come from ZulipConfig fields
# ---------------------------------------------------------------------------
CUSTOM = {
    "model_none_message": "NONE-X",
    "model_list_message": "LIST<{names}>",
    "model_unknown_message": "UNK<{name}|{names}>",
    "model_switched_message": "SW<{name}>",
}


def test_defaults_equal_previous_literals():
    config = ZulipConfig(zuliprc="/unused.zuliprc")
    assert config.model_none_message == "No models are configured."
    assert config.model_list_message == "Available models: {names}"
    assert config.model_unknown_message == (
        "Unknown model '{name}'. Available: {names}"
    )
    assert config.model_switched_message == "Switched to model '{name}'."


def _custom_iface():
    iface, agent = _iface()
    iface._config = iface._config.model_copy(update=CUSTOM)
    return iface, agent


async def test_custom_none_message():
    iface, agent = _custom_iface()
    agent.config.models = {}
    await _send(iface, "!model coding")
    assert _reply(iface._client) == "NONE-X"


async def test_custom_list_message():
    iface, _ = _custom_iface()
    await _send(iface, "!model")
    assert _reply(iface._client) == f"LIST<{AVAILABLE}>"


async def test_custom_unknown_message():
    iface, agent = _custom_iface()
    await _send(iface, "!model nope")
    assert _reply(iface._client) == f"UNK<nope|{AVAILABLE}>"
    assert agent.config.active_model == "default"


async def test_custom_switched_message():
    iface, agent = _custom_iface()
    await _send(iface, "!model coding")
    assert _reply(iface._client) == "SW<coding>"
    assert agent.config.active_model == "coding"
