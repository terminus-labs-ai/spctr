"""Reminders — Discord creation and delivery (spc-141, AC13-AC15).

Fakes stand in for discord.py (``MagicMock(spec=...)``: no gateway, no
token, no network); the clock the tool and loop run on is an injected
callable; every store path is under tmp_path. The turn path runs through
the real ``DiscordInterface``, ``Runtime`` and ``ReminderTool`` so the
tests pin the seam between them: the origin a turn records (AC13), the
guild a named channel resolves inside (AC14), the send a due reminder
takes (AC15), plus the two spec seam constraints (origin isolation,
loop start/stop).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.interface import DiscordInterface
from sr2_spectre.reminders import (
  DeliveryLoop,
  Reminder,
  ReminderStore,
  ReminderTool,
)
from sr2_spectre.reminders.tool import turn_scope

UTC = ZoneInfo("UTC")
PLACEHOLDER = "\u23f3 Thinking..."
NOW = datetime(2026, 10, 15, 22, 0, 0, tzinfo=UTC)

GUILD_ID = 700
GUILD_B = 701
PARENT_ID = 555
THREAD_ID = 999
GENERAL_ID = 70
OTHER_GUILD_ID = 808
PARENT_NAME = "estudos"
GENERAL_NAME = "general"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _text_channel(channel_id: int, name: str) -> Any:
  discord = pytest.importorskip("discord")
  channel = MagicMock(spec=discord.TextChannel)
  channel.id = channel_id
  channel.name = name
  return channel

def _thread(thread_id: int, parent: Any, name: str = "me lembra disso") -> Any:
  discord = pytest.importorskip("discord")
  thread = MagicMock(spec=discord.Thread)
  thread.id = thread_id
  thread.name = name
  thread.parent = parent
  return thread

def _message(
  channel: Any,
  author_id: int = 4242,
  guild_id: int = GUILD_ID,
  content: str = "me lembra disso amanhã às 9",
) -> Any:
  message = MagicMock()
  message.content = content
  message.id = 1234
  message.channel = channel
  message.author = MagicMock(id=author_id)
  message.guild = MagicMock(id=guild_id)
  return message

class FakeAdapter:
  """Stand-in for DiscordBotAdapter: records sends, answers channel lookups.

  ``is_sendable_text_channel`` follows the real rule (a Thread is never a
  destination); ``guild_channels`` answers from the test's guild map, which
  is what proves a lookup stayed inside the origin guild.
  """

  bot_id = 11111
  bot_mentions = ["<@11111>"]

  def __init__(self, guilds: dict[int, list[Any]] | None = None) -> None:
    import discord

    self._discord = discord
    self._guilds = guilds or {}
    self.sends: list[tuple[int, str]] = []
    self.lookups: list[int | None] = []

  async def start(self) -> None: ...
  async def stop(self) -> None: ...

  def set_message_handler(self, handler) -> None:
    self.handler = handler

  def set_slash_handler(self, handler) -> None:
    self.slash_handler = handler

  def is_thread_channel(self, channel_obj) -> bool:
    return isinstance(channel_obj, self._discord.Thread)

  def is_sendable_text_channel(self, channel) -> bool:
    return isinstance(channel, self._discord.TextChannel)

  def guild_channels(self, guild_id: int | None) -> list[Any]:
    self.lookups.append(guild_id)
    return self._guilds.get(guild_id, [])

  def area_channel(self, channel_obj):
    if isinstance(channel_obj, self._discord.Thread):
      channel_obj = channel_obj.parent
      if channel_obj is None:
        return None, None
    return getattr(channel_obj, "id", None), getattr(channel_obj, "name", None)

  def channel_typing(self, channel_id: int):
    typing_ctx = MagicMock()
    typing_ctx.__aenter__ = AsyncMock()
    typing_ctx.__aexit__ = AsyncMock(return_value=None)
    return typing_ctx

  async def send_message(self, channel_id: int, content: str):
    self.sends.append((channel_id, content))
    return MagicMock(id=500 + len(self.sends))

  async def try_send_message(self, channel_id: int, content: str):
    return await self.send_message(channel_id, content), None

  async def edit_message(self, channel_id: int, message_id: int, text: str):
    return None


# ---------------------------------------------------------------------------
# Helpers — real Runtime/Agent/Tool over a tmp_path store
# ---------------------------------------------------------------------------

def _config(store: Path, **reminders: Any) -> SpectreConfig:
  agent: dict[str, Any] = {"name": "ezio", "default_skills": False}
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
    "poll_seconds": 30,
  }
  block.update(overrides)
  return _config(store, **block)

def _runtime(config: SpectreConfig):
  from sr2_spectre.runtime import Runtime

  with patch("sr2_spectre.live_llm.LiteLLMCallable"):
    return Runtime(config=config)

async def _started(
  tmp_path: Path,
  guilds: dict[int, list[Any]] | None = None,
  enabled: bool = True,
  poll_seconds: float = 0.01,
  runtime: Any = None,
) -> tuple[DiscordInterface, FakeAdapter, Any]:
  """A real DiscordInterface started over a FakeAdapter and a real Agent.

  The loop's poll interval is shrunk after start so it yields often and no
  test waits on the wall clock. ``runtime`` swaps the Runtime the agent
  fronts (the reload tests).
  """
  from sr2_spectre.agent import Agent

  store = tmp_path / "reminders.sqlite"
  config = _enabled(store, poll_seconds=60) if enabled else _config(store)
  with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=object()):
    agent = Agent(config=config)
  if runtime is not None:
    agent._runtime = runtime  # the facade reads its Runtime through this one attribute
  adapter = FakeAdapter(guilds)
  interface = DiscordInterface(config=DiscordConfig(token="test"))
  real_start = interface._start_reminders

  def _start(agent_: Any) -> None:
    real_start(agent_)
    if interface._reminders_loop is not None:
      interface._reminders_loop._poll_seconds = poll_seconds

  interface._start_reminders = _start  # type: ignore[method-assign]
  with patch(
    "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter",
    return_value=adapter,
  ):
    await interface.start(agent)
  _close_agents.started.append(agent)
  return interface, adapter, agent

async def _stopped(interface: DiscordInterface) -> None:
  """Stop the interface; the agent itself closes in the autouse fixture."""
  await interface.stop()


@pytest.fixture(autouse=True)
async def _close_agents():
  """Every agent a test started is closed afterwards, whatever it asserts."""
  started: list = []
  _close_agents.started = started
  yield
  for agent in started:
    await agent.aclose()

async def _settle(loop: Any, polls: int = 3) -> None:
  for _ in range(polls):
    await asyncio.sleep(0.02)

def _row(due: datetime) -> Reminder:
  return Reminder(
    id="r1",
    agent="ezio",
    platform="discord",
    requester_id="4242",
    requester_mention="<@4242>",
    origin_channel_id=str(THREAD_ID),
    dest_channel_id=str(GENERAL_ID),
    dest_channel_name=GENERAL_NAME,
    dest_topic="",  # Discord has no topic (FR9)
    text="look into https://example.com/x",
    due_at=due,
    created_at=due - timedelta(minutes=30),
    delivered_at=None,
    status="pending",
    attempts=0,
    last_error=None,
  )

def _create_turn(
  agent: Any, channel_name: str | None, into: list[str] | None = None
) -> None:
  """Make the fake model turn call the Runtime's reminder tool once.

  ``into`` collects the tool's reply; the turn itself is driven by
  ``_drive_turn``, which waits until the model turn really ran.
  """
  from sr2_spectre.events import AgentTextDelta

  captured = [] if into is None else into

  async def _stream(_text: str):
    captured.append(
      await agent.reminders.tool(
        action="create", text="lembrete", in_minutes=60, channel=channel_name
      )
    )
    yield AgentTextDelta(text="ok")

  agent.stream_message = _stream

async def _drive_turn(
  interface: DiscordInterface, message: Any
) -> None:
  """One message turn through the real interface, as discord.py dispatches it.

  Waits until the fake model turn actually ran (the sent placeholder proves
  the model loop was entered) before returning.
  """
  done = asyncio.Event()
  real_send = interface._adapter.send_message

  async def _send(channel_id: int, content: str):
    msg = await real_send(channel_id, content)
    if content == "⏳ Thinking...":
      done.set()
    return msg

  interface._adapter.send_message = _send
  await asyncio.wait_for(interface._process_message(message), timeout=20)
  await asyncio.wait_for(done.wait(), timeout=20)


# ---------------------------------------------------------------------------
# AC13 — default destination: the origin channel, or a thread's parent
# ---------------------------------------------------------------------------

class TestAC13DefaultDestination:
  async def test_thread_turn_stores_the_parent_channel(self, tmp_path) -> None:
    """AC13: a create made in a thread is addressed to the thread's parent,
    mentions the requester by <@id>, and carries no topic (FR9)."""
    parent = _text_channel(PARENT_ID, PARENT_NAME)
    interface, _adapter, agent = await _started(tmp_path)
    store: ReminderStore = agent.reminders.store
    _create_turn(agent, None)
    await _drive_turn(interface, _message(_thread(THREAD_ID, parent)))
    row = store.get("r1")
    assert row.dest_channel_id == str(PARENT_ID)
    assert row.dest_channel_name == PARENT_NAME
    assert row.platform == "discord"
    assert row.requester_id == "4242"
    assert row.requester_mention == "<@4242>"

  async def test_channel_turn_stores_that_channel(self, tmp_path) -> None:
    """AC13: a create made in a text channel is addressed to that channel."""
    parent = _text_channel(PARENT_ID, PARENT_NAME)
    interface, _adapter, agent = await _started(tmp_path)
    store: ReminderStore = agent.reminders.store
    _create_turn(agent, None)
    await _drive_turn(interface, _message(parent))
    row = store.get("r1")
    assert row.dest_channel_id == str(PARENT_ID)
    assert row.dest_channel_name == PARENT_NAME


# ---------------------------------------------------------------------------
# AC14 — named channel: one match in the origin guild stores; zero or
#         multiple error and store nothing
# ---------------------------------------------------------------------------

class TestAC14NamedChannel:
  async def test_named_create_stores_the_resolved_channel(self, tmp_path) -> None:
    """AC14: inside a real turn the name matches case-insensitively among the
    ORIGIN guild's text channels — a thread-name sibling is never a match."""
    parent = _text_channel(PARENT_ID, PARENT_NAME)
    general = _text_channel(GENERAL_ID, GENERAL_NAME)
    lookalike = _thread(THREAD_ID, parent, name=GENERAL_NAME)
    interface, adapter, agent = await _started(
      tmp_path, guilds={GUILD_ID: [parent, general, lookalike]}
    )
    store: ReminderStore = agent.reminders.store
    _create_turn(agent, "general")
    await _drive_turn(interface, _message(parent))
    assert store.get("r1").dest_channel_id == str(GENERAL_ID)
    assert set(adapter.lookups) == {GUILD_ID}  # only the origin guild

  async def test_named_create_uses_the_turn_guild_from_the_context(
    self, tmp_path
  ) -> None:
    """FR10: the tool hands the resolver THIS turn's scope, set by the
    interface around the turn — guild and origin travel together."""
    parent = _text_channel(PARENT_ID, PARENT_NAME)
    interface, _adapter, agent = await _started(tmp_path, guilds={})
    seen: list[Any] = []

    class ResolverSpy:
      """Wrapper with the seam's marker: records what the tool's call carried."""

      spectre_platform = "discord"

      async def __call__(self, platform: str, name: str, scope=None):
        seen.append((platform, name, scope))
        return await interface._resolve_channel(platform, name, scope)

    agent.reminders.tool.attach_channel_resolver(ResolverSpy())
    _create_turn(agent, "#General")
    await _drive_turn(interface, _message(_thread(THREAD_ID, parent)))
    platform, name, scope = seen[0]
    assert (platform, name) == ("discord", "General")
    assert scope.guild_id == GUILD_ID
    assert scope.origin.channel_id == str(PARENT_ID)  # the thread's parent

  async def test_zero_matches_error_stores_nothing(self, tmp_path) -> None:
    """AC14: no text channel in the origin guild answers → tool error, store
    unchanged."""
    parent = _text_channel(PARENT_ID, PARENT_NAME)
    interface, _adapter, agent = await _started(tmp_path, guilds={GUILD_ID: [parent]})
    replies: list[str] = []
    _create_turn(agent, "#nope", replies)
    await _drive_turn(interface, _message(parent))
    assert replies == ["Error: channel '#nope' not found on this platform"]
    assert agent.reminders.store.count_by_status() == {}

  async def test_multiple_matches_error_stores_nothing(self, tmp_path) -> None:
    """AC14: two channels answering the same name → tool error, and the
    ambiguity is never resolved by picking one."""
    parent = _text_channel(PARENT_ID, PARENT_NAME)
    interface, _adapter, agent = await _started(
      tmp_path,
      guilds={
        GUILD_ID: [
          parent,
          _text_channel(GENERAL_ID, "General"),
          _text_channel(71, "GENERAL"),
        ]
      },
    )
    store: ReminderStore = agent.reminders.store
    _create_turn(agent, "general")
    await _drive_turn(interface, _message(parent))
    assert store.count_by_status() == {}

  async def test_a_channel_of_another_guild_is_not_matched(self, tmp_path) -> None:
    """AC14: #general in ANOTHER guild is no match for a turn in guild 700,
    and that other guild is never searched."""
    parent = _text_channel(PARENT_ID, PARENT_NAME)
    other_general = _text_channel(GENERAL_ID, GENERAL_NAME)
    interface, adapter, agent = await _started(
      tmp_path,
      guilds={GUILD_ID: [parent], OTHER_GUILD_ID: [other_general]},
    )
    store: ReminderStore = agent.reminders.store
    _create_turn(agent, "general")
    await _drive_turn(interface, _message(parent, guild_id=GUILD_ID))
    assert store.count_by_status() == {}
    assert set(adapter.lookups) == {GUILD_ID}

  async def test_named_create_works_with_a_tool_seated_before_start(
    self, tmp_path
  ) -> None:
    """The resolver reaches the tool seated BEFORE the interface started:
    start registers it on the Runtime seam AND attaches it to the seated
    tool (the Zulip wiring), with no reload in between."""
    parent = _text_channel(PARENT_ID, PARENT_NAME)
    general = _text_channel(GENERAL_ID, GENERAL_NAME)
    interface, _adapter, agent = await _started(
      tmp_path, guilds={GUILD_ID: [parent, general]}
    )
    store: ReminderStore = agent.reminders.store
    assert isinstance(agent.reminders.tool, ReminderTool)
    _create_turn(agent, "general")
    await _drive_turn(interface, _message(_thread(THREAD_ID, parent)))
    assert store.get("r1").dest_channel_id == str(GENERAL_ID)


# ---------------------------------------------------------------------------
# AC15 — a due reminder posts to the channel (never a thread) with a
#         <@id> mention and is marked delivered
# ---------------------------------------------------------------------------

class TestAC15Delivery:
  async def test_due_reminder_posts_to_the_channel_and_is_delivered(
    self, tmp_path
  ) -> None:
    """AC15: the loop sends ``<@id> Reminder: <text>`` to the destination
    CHANNEL (the row's, not the origin thread's) and marks the row delivered."""
    interface, adapter, agent = await _started(tmp_path)
    store: ReminderStore = agent.reminders.store
    store.create(_row(due=NOW - timedelta(minutes=2)))
    loop = interface._reminders_loop
    assert loop is not None
    loop._clock = lambda: NOW  # injected clock: no real waiting
    await _settle(loop)
    assert adapter.sends == [
      (GENERAL_ID, "<@4242> Reminder: look into https://example.com/x")
    ]
    row = store.get("r1")
    assert row.status == "delivered"
    assert row.delivered_at == NOW

  async def test_send_failure_leaves_the_row_pending(self, tmp_path) -> None:
    """AC15 (FR17): only an accepted send marks delivered; a failed adapter
    send leaves the row pending for the FR18 retry."""

    class FailingAdapter(FakeAdapter):
      async def try_send_message(self, channel_id: int, content: str):
        return None, RuntimeError("403 Forbidden")

    from sr2_spectre.agent import Agent

    with patch("sr2_spectre.live_llm.LiteLLMCallable", return_value=object()):
      agent = Agent(
        config=_enabled(tmp_path / "fail.sqlite", poll_seconds=60)
      )
    adapter = FailingAdapter({})
    interface = DiscordInterface(config=DiscordConfig(token="test"))
    with patch(
      "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter",
      return_value=adapter,
    ):
      await interface.start(agent)
    store: ReminderStore = agent.reminders.store
    store.create(_row(due=NOW - timedelta(minutes=2)))
    interface._reminders_loop._clock = lambda: NOW
    await _settle(interface._reminders_loop)
    row = store.get("r1")
    assert row.status == "pending"
    assert row.attempts == 1
    assert "403" in (row.last_error or "")

  async def test_a_thread_destination_is_never_posted_to(self, tmp_path) -> None:
    """AC15: the destination must be a text channel — a thread is refused at
    the adapter rather than posted to (FR9)."""
    from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter

    adapter = DiscordBotAdapter(DiscordConfig(token="fake"))
    assert adapter.is_sendable_text_channel(_text_channel(GENERAL_ID, GENERAL_NAME))
    assert not adapter.is_sendable_text_channel(_thread(THREAD_ID, None))

  async def test_delivery_goes_through_the_adapter_send_path(self) -> None:
    """AC15: the destination send is the adapter's own discord.py
    ``channel.send`` — the same path a chat reply takes."""
    from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter

    adapter = DiscordBotAdapter(DiscordConfig(token="fake"))
    channel = MagicMock()
    sent = AsyncMock(return_value=MagicMock(id=7))
    channel.send = sent
    bot = MagicMock()
    bot.get_channel = MagicMock(return_value=channel)
    adapter._bot = bot
    message, error = await adapter.try_send_message(GENERAL_ID, "hello")
    assert error is None and message.id == 7
    sent.assert_awaited_once_with("hello")


# ---------------------------------------------------------------------------
# Origin isolation — concurrent turns never record each other's origin
# ---------------------------------------------------------------------------

def _placeholder_reached(interface: DiscordInterface, seen: set, waiting: asyncio.Event):
  """Adapter.send_message wrapper: signals once the turn reached its model.

  The placeholder send is the last step before the interface enters its
  turn lock, so a signal here proves the turn is in flight.
  """
  real_send = interface._adapter.send_message

  async def _send(channel_id: int, content: str):
    if content == PLACEHOLDER:
      seen.add((turn_scope.get().origin.channel_id, turn_scope.get().guild_id))
      waiting.set()
    return await real_send(channel_id, content)

  interface._adapter.send_message = _send

class TestOriginIsolation:
  async def test_overlapping_turns_record_their_own_origin_and_guild(
    self, tmp_path
  ) -> None:
    """Spec, Delivery seam: while a turn in one channel is in flight, a turn
    in another channel records only its OWN origin and guild, and the tool
    call that stores a row reads only its own task's scope.

    The second message is dispatched while the first turn still holds the
    interface's turn lock — the shipped interface serializes the two model
    turns, and this test proves the opposite property: the scope is the
    task's own, so it is never the other turn's, locked or not.
    """
    other_parent = _text_channel(600, "other-room")
    other_general = _text_channel(70, "general")
    interface, adapter, agent = await _started(
      tmp_path,
      guilds={
        GUILD_ID: [
          _text_channel(PARENT_ID, PARENT_NAME),
          _text_channel(GENERAL_ID, GENERAL_NAME),
        ],
        GUILD_B: [other_parent, other_general],
      },
    )
    store: ReminderStore = agent.reminders.store
    entered: asyncio.Queue = asyncio.Queue()
    inside: asyncio.Queue = asyncio.Queue()
    turn_a_in_flight = asyncio.Event()
    scopes_seen: set = set()
    _placeholder_reached(interface, scopes_seen, turn_a_in_flight)

    async def _stream(_text: str):
      from sr2_spectre.events import AgentTextDelta

      scope = turn_scope.get()
      entered.put_nowait((scope.origin, scope.guild_id))
      # Hold this turn inside its model call until the other channel's
      # message has been dispatched: two turns are then open at once.
      await asyncio.wait_for(turn_a_in_flight.wait(), timeout=10)
      assert turn_scope.get() is scope  # still this task's own scope
      reply = await agent.reminders.tool(
        action="create", text="t", in_minutes=60, channel="general"
      )
      inside.put_nowait((turn_scope.get().origin, reply))
      yield AgentTextDelta(text="ok")

    agent.stream_message = _stream
    first = asyncio.ensure_future(
      interface._process_message(
        _message(_thread(THREAD_ID, _text_channel(PARENT_ID, PARENT_NAME)))
      )
    )
    origin_a, guild_a = await asyncio.wait_for(entered.get(), timeout=10)
    second = asyncio.ensure_future(
      interface._process_message(_message(other_parent, guild_id=GUILD_B))
    )
    reports = [
      await asyncio.wait_for(inside.get(), timeout=20) for _ in (0, 1)
    ]
    await asyncio.wait_for(asyncio.gather(first, second), timeout=20)

    assert (origin_a.channel_id, origin_a.channel_name, guild_a) == (
      str(PARENT_ID),
      PARENT_NAME,
      GUILD_ID,
    )
    by_channel = {o.channel_id: (o, r) for o, r in reports}
    origin_b, reply_b = by_channel["600"]
    assert not reply_b.startswith("Error")
    # Each row went to its own guild's #general: 70 in guild 701, 70 elsewhere.
    assert {store.get("r1").dest_channel_id, store.get("r2").dest_channel_id} == {
        str(GENERAL_ID),
        "70",
    }
    # Both turns' scopes were live, and the second one became live while the
    # first was still in flight.
    assert scopes_seen == {(str(PARENT_ID), GUILD_ID), ("600", GUILD_B)}
    assert turn_scope.get() is None  # both scopes closed


# ---------------------------------------------------------------------------
# Loop lifecycle — starts/stops with the interface (FR22), follows a re-seat
# ---------------------------------------------------------------------------

class TestLoopLifecycle:
  async def test_enabled_starts_and_stop_stops_the_loop(self, tmp_path) -> None:
    """FR6/FR22: an enabled agent gets the delivery loop on the Runtime's
    store at start(); stop() ends it and touches no row."""
    interface, _adapter, agent = await _started(tmp_path)
    store: ReminderStore = agent.reminders.store
    store.create(_row(due=NOW + timedelta(days=1)))
    loop = interface._reminders_loop
    assert isinstance(loop, DeliveryLoop)
    assert loop._store is agent.reminders.store
    await _stopped(interface)
    assert interface._reminders_loop is None
    assert loop._task is None
    assert store.count_by_status() == {"pending": 1}

  async def test_disabled_starts_no_loop(self, tmp_path) -> None:
    interface, _adapter, _agent = await _started(tmp_path, enabled=False)
    assert interface._reminders_loop is None

  async def test_a_live_reload_reseat_moves_the_loop(self, tmp_path) -> None:
    """spc-143: a reload that re-seats reminders is followed — the running
    loop stops and rebuilds from the new handle; a no-op reload does not."""
    store_file = tmp_path / "reload.sqlite"
    interface, _adapter, agent = await _started(
      tmp_path, runtime=_runtime(_enabled(store_file, poll_seconds=30))
    )
    loop_before = interface._reminders_loop
    assert loop_before is not None
    runtime = agent._runtime
    assert runtime.apply_config(_enabled(store_file, poll_seconds=45)) == [
      "agent.reminders"
    ]
    assert interface._reminders_loop is loop_before  # not by itself
    await interface._sync_reminders()
    loop_after = interface._reminders_loop
    assert loop_after is not None and loop_after is not loop_before
    assert loop_after._store is runtime.reminders.store
    assert loop_before._task is None
    assert runtime.apply_config(_enabled(store_file, poll_seconds=45)) == []
    await interface._sync_reminders()
    assert interface._reminders_loop is loop_after
