"""DiscordInterface — the Interface protocol implementation for Discord.

Implements the Interface protocol (name, start, stop, run) to connect
SR2 Spectre to Discord via discord.py.

Architecture:
- DiscordInterface: Implements the Interface protocol. Controls lifecycle.
- DiscordBotAdapter: Wraps discord.py Bot client (start/stop/send/edit).
- SessionMap: Maps channel_id → conversation history (session-per-channel).
- MessageHandler: Pure logic for routing, commands, chunking.

Data flow:
1. User sends message in Discord channel
2. Adapter receives discord.Message → passes to Interface
3. Interface checks: mention filter, channel filter, slash commands
4. For regular messages: Agent.stream_message() with channel-specific history
5. Response streamed via message edits (configurable)
6. Tool execution shown as embeds (configurable)
7. Local image paths in responses are uploaded and replaced with Discord URLs
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import AgentTextDelta, AgentToolResult, AgentToolStart
from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.config_source import (
  DiscordConfigProvider,
  DiscordConfigSource,
  DiscordConfigView,
)
from sr2_spectre.reminders import (
  ChannelMatch,
  DeliveryLoop,
  ReminderOrigin,
)
from sr2_spectre.reminders import tool as reminders_tool
from sr2_spectre.schedules import ScheduleEngine
from sr2_spectre.tools.schedule_tool import ScheduleOrigin, ScheduleTurnScope, schedule_turn_scope
from sr2_spectre.interfaces.discord.handler import (
  CommandContext,
  chunk_message,
  handle_command,
  parse_slash_command,
  probe_harbinger_status,
  render_model_command,
  resolve_area,
  should_respond,
  split_for_retry,
  tail_for_stream,
)
from sr2_spectre.interfaces.discord.session_map import SessionMap

if TYPE_CHECKING:
  from sr2_spectre.agent import Agent

logger = logging.getLogger(__name__)


class DiscordInterface:
  """Discord interface for SR2 Spectre.

  Implements the Interface protocol. Each Discord channel maintains
  its own conversation history (session-per-channel isolation).
  """

  name = "discord"

  def __init__(
    self,
    config: DiscordConfig | None = None,
    config_source: DiscordConfigProvider | None = None,
  ) -> None:
    """
    Args:
        config: A fixed config. Ignored when config_source is given.
        config_source: Live config source, re-read on every inbound
            message so config edits apply without a process restart.
            A ``DiscordConfigView`` additionally carries the whole
            SpectreConfig, letting model, endpoint, pipeline and tool
            edits reach the running agent — see ``_apply_agent_config``.
            When omitted, *config* is frozen for the life of the process.
    """
    self._config_source = config_source or DiscordConfigSource.static(config)
    self._agent: Agent | None = None
    self._adapter: DiscordBotAdapter | None = None
    self._session_map = SessionMap()
    self._running = False
    # The shared Agent holds one live Session, and restoring a channel
    # replaces it. Serialize restore -> stream so concurrent turns cannot
    # swap the Session under each other (obsidian-hw0f).
    self._agent_turn_lock = asyncio.Lock()
    self._pending_stream_edit: asyncio.Future[None] | None = None
    # Per-channel tool activity lines — prepended to the thinking message
    # during streaming, then replaced by the final response on completion.
    self._tool_lines: dict[int, list[str]] = {}
    # Per-channel accumulated streamed text. Shared state so the unified
    # progress renderer can show tool lines AND text together — without it,
    # the tool path and text path clobber each other's content on the same
    # message (the "edited back and forth" bug).
    self._stream_text: dict[int, str] = {}
    # In-flight agent runs, keyed by channel, so /stop can cancel the
    # current turn. Set while a run is active, removed when it settles.
    self._active_runs: dict[int, asyncio.Task[None]] = {}
    # Reminders (spc-141, FR6/FR22): the delivery loop this interface owns
    # and the reload stamp it started under (see _sync_reminders).
    self._reminders_loop: Any = None
    self._reminders_stamp: int | None = None
    self._reminders_start_task: Any = None
    # Schedules (spc-185, FR23/FR35): the engine this interface owns.
    self._schedules_engine: Any = None
    self._schedules_stamp: int | None = None
    self._schedules_start_task: Any = None

  @property
  def config(self) -> DiscordConfig:
    """The Discord config in force, as of the last reload.

    Reads through to the shared config source, so handling of a message
    sees the values the adapter loaded when that message arrived.
    """
    return self._config_source.current

  async def start(self, agent: "Agent") -> None:
    """Initialize the Discord interface and start the bot.

    Args:
        agent: The Spectre Agent instance.
    """
    self._agent = agent
    self._adapter = DiscordBotAdapter(self._config_source)
    self._running = True

    # Set interactive run context for Discord
    agent.set_run_context(
      RunContext(
        interface="discord",
        mode=RunMode.INTERACTIVE,
        source=None,  # channel-specific source set per-message in handler
      )
    )

    # Set up the message handler
    async def _handle_message(message: Any) -> None:
      await self._process_message(message)

    self._adapter.set_message_handler(_handle_message)

    # Wire native slash commands (must be set before adapter.start(), which
    # syncs the command tree in on_ready).
    self._adapter.set_slash_handler(self._handle_slash)

    await self._adapter.start()
    # Reminders (spc-141, FR6): the resolver goes on the Runtime seam at
    # start, which rebuilds the tool already seated so it carries the
    # resolver — see _wire_reminder_channel_lookup.
    self._wire_reminder_channel_lookup(agent)
    self._start_reminders(agent)
    self._start_schedules(agent)

  def _apply_agent_config(self) -> None:
    """Hand the agent the config the adapter just reloaded.

    Only meaningful on the ``DiscordConfigView`` path, where the reload
    covered the whole SpectreConfig. With a plain ``DiscordConfigSource``
    (tests, embedders) there is no agent config to apply and this is a
    no-op.
    """
    if self._agent is None or not isinstance(self._config_source, DiscordConfigView):
      return
    self._agent.apply_config(self._config_source.source.current)

  # ---- reminders (spc-141, FR6/FR9/FR10/FR22) -------------------------------

  def _runtime_of(self, agent: Any) -> Any:
    return getattr(agent, "_runtime", None)

  def _wire_reminder_channel_lookup(self, agent: Any) -> None:
    """Hand this interface's named-channel resolver to the Runtime (spc-159).

    Registered once, at start, through the Runtime's public seam, which
    passes it into every ReminderTool it builds (a live-reload re-seat is
    born with it) and rebuilds the seat that already exists through that
    same construction site (spc-166). The Zulip interface wires the same
    way, for the same reason: the Runtime builds its tool before any
    interface exists to register a resolver.
    """
    registrar = getattr(agent, "set_reminders_channel_resolver", None)
    if registrar is not None:
      registrar(self._resolve_channel)

  async def _resolve_channel(
    self, platform: str, name: str, scope: Any = None
  ) -> list[Any]:
    """FR10: exact case-insensitive match among the ORIGIN guild's channels.

    Only postable text channels match — never a thread, which a reminder
    must never be posted to (AC15). ``scope`` is this turn's reminder scope,
    handed to every Discord resolver by the tool; its guild scopes the
    search, since the origin alone names no guild. No guild or no match
    answers "no match": the tool errors and stores nothing.

    ``spectre_platform`` is the seam's marker that this resolver resolves
    Discord names (spc-159, FR11).
    """
    if platform != "discord" or self._adapter is None or scope is None:
      return []
    guild_id = getattr(scope, "guild_id", None)
    if guild_id is None:
      return []
    matches = []
    for channel in self._adapter.guild_channels(guild_id):
      if (
        self._adapter.is_sendable_text_channel(channel)
        and getattr(channel, "name", None)
        and str(channel.name).lower() == name.lower()
      ):
        matches.append(
          ChannelMatch(channel_id=channel.id, channel_name=channel.name)
        )
    return matches

  # The seam's platform marker (spc-159): set on the underlying function,
  # so every bound ``interface._resolve_channel`` carries it.
  _resolve_channel.spectre_platform = "discord"

  def _reminders_origin_for(
    self,
    channel_obj: Any,
    area: tuple[int | None, str | None],
    message: Any,
  ) -> ReminderOrigin:
    """The origin of THIS turn, built from this message alone.

    Everything is read from the message being handled — its own channel
    object, its own author, the area derived from its own channel — so two
    concurrent turns cannot read each other's (spec, Delivery seam).

    FR9: the destination defaults to the origin's channel ROOT — for a
    message in a thread, that thread's parent channel — so the reminder
    lands where the conversation lives, not in a thread.
    """
    channel_id, channel_name = area
    # The area-bearing channel — a thread's parent. The fallbacks address a
    # channel with no area (a DM) by its own ID rather than an empty string.
    root_id = channel_id if channel_id is not None else getattr(channel_obj, "id", None)
    root_name = (
      channel_name
      if channel_name is not None
      else str(getattr(channel_obj, "name", "") or "")
    )
    author_id = getattr(getattr(message, "author", None), "id", None)
    if author_id is None:
      author_id = getattr(message, "author_id", None)
    return ReminderOrigin(
      agent=self._agent.config.agent.name,
      platform="discord",
      requester_id=str(author_id) if author_id is not None else "",
      # FR13: Discord mentions a user by ID.
      requester_mention=f"<@{author_id}>" if author_id is not None else "",
      channel_id=str(root_id) if root_id is not None else "",
      channel_name=root_name,
      # Discord has no topic; a reminder posts to the channel (FR9).
      topic="",
    )

  # The scope is set and reset in these two helpers, never inline: a
  # ``ContextVar.set`` in a function's own frame makes that function a
  # coroutine whose context attaches to whoever awaits it (PEP 550), which
  # would run a turn against the caller's context instead of the task's.
  def _reminders_scope_open(self, message: Any, channel_obj: Any, area: Any) -> None:
    reminders_tool.turn_scope.set(
      reminders_tool.TurnScope(
        origin=self._reminders_origin_for(channel_obj, area, message),
        guild_id=getattr(getattr(message, "guild", None), "id", None),
      )
    )

  def _reminders_scope_close(self) -> None:
    # A turn's scope was set on entry, so a value is always there; this
    # leaves the task's context as it was before the turn.
    if reminders_tool.turn_scope.get() is not None:
      reminders_tool.turn_scope.set(None)

  def _reminders_turn_open(
    self,
    message: Any,
    channel_obj: Any,
    area: tuple[int | None, str | None],
  ) -> str:
    """Open this turn's reminder scope: its origin and its guild (spc-141).

    A task runs one message and a context is the task's own, so two turns
    in two channels share nothing set here — the isolation the spec's
    Delivery seam demands, with no reliance on the turn lock. The Runtime's
    origin slot is deliberately not written: one shared value is exactly
    what an overlapping turn could overwrite.
    """
    self._reminders_scope_open(message, channel_obj, area)
    return "open"

  def _reminders_turn_close(self, token: Any) -> None:
    """Undo the scope _reminders_turn_open set, in this task only."""
    if token is None:
      return
    self._reminders_scope_close()

  def _start_reminders(self, agent: Any) -> None:
    """Start the delivery loop when this agent has a reminders handle.

    Disabled is None and no loop is built (FR6). The loop claims due rows
    from the Runtime's store and posts them through this interface's
    adapter, so a reminder travels the same send path as a reply. The stamp
    recorded here is what ``_sync_reminders`` compares against afterwards.
    """
    reminders = getattr(agent, "reminders", None)
    if reminders is None:
      return
    if not asyncio.iscoroutinefunction(
      getattr(reminders.store, "reset_sending_to_pending_async", None)
    ):
      # A handle without a usable store is not a reminders handle (a test
      # stand-in): building a loop on it would fail inside the loop's task.
      return

    async def send(channel_id: str, topic: str, content: str) -> None:
      # ``topic`` is Zulip's (FR9); a Discord reminder posts to its channel.
      _message, error = await self._adapter.try_send_message(
        int(channel_id), content
      )
      if error is not None:
        # Raised, not swallowed: a row is marked delivered only once the
        # platform accepted the send (FR17), and a failure feeds FR18.
        raise error

    loop = DeliveryLoop(
      reminders.store,
      send=send,
      poll_seconds=reminders.poll_seconds,
      tz=reminders.tz,
      agent_name=reminders.agent_name,
      # spc-176: this loop serves only Discord — one agent's Zulip loop
      # shares the store and must not touch these rows.
      platform="discord",
    )
    self._reminders_loop = loop
    self._reminders_stamp = getattr(agent, "reminders_stamped", 0)
    self._reminders_start_task = asyncio.ensure_future(loop.start())

  async def _stop_reminders(self) -> None:
    """End the delivery loop; pending rows are left intact (FR22)."""
    loop = self._reminders_loop
    self._reminders_loop = None
    self._reminders_stamp = None
    if loop is None:
      return
    try:
      await loop.stop()
    except asyncio.CancelledError:
      pass

  async def _sync_reminders(self) -> None:
    """Re-seat the delivery loop after a live config reload (spc-143).

    Runs once per message, after ``_apply_agent_config``, so the tool this
    turn's model calls is the re-seated one. Steady state costs one int
    compare. Once the Runtime's stamp moves, re-assert from the current
    handle: enable starts, disable stops with pending rows intact, a
    settings change restarts on the fresh settings.
    """
    runtime = self._runtime_of(self._agent)
    stamp = getattr(runtime, "reminders_stamped", None) if runtime else None
    if self._reminders_stamp is not None:
      if stamp == self._reminders_stamp:
        return
      await self._stop_reminders()
    handle = getattr(self._agent, "reminders", None)
    if self._reminders_loop is None and handle is not None:
      self._start_reminders(self._agent)

  # ---- schedules (spc-185, FR23/FR35) ---------------------------------------

  def _start_schedules(self, agent: Any) -> None:
    """Start the schedule engine when this agent has a schedules handle."""
    schedules = getattr(agent, "schedules", None)
    if schedules is None:
      return
    if not asyncio.iscoroutinefunction(
      getattr(schedules.store, "recover_claimed_async", None)
    ):
      return

    async def send(channel_id: str, content: str) -> None:
      _message, error = await self._adapter.try_send_message(
        int(channel_id), content
      )
      if error is not None:
        raise error

    engine = ScheduleEngine(
      schedules.store,
      send=send,
      poll_seconds=schedules.poll_seconds,
      tz=schedules.tz,
      agent_name=schedules.agent_name,
      command_timeout=schedules.command_timeout,
      platform="discord",
    )
    # Wire the engine into the tool so run_now works.
    if hasattr(schedules, "tool") and schedules.tool is not None:
      schedules.tool._engine = engine
    self._schedules_engine = engine
    self._schedules_stamp = getattr(agent, "schedules_stamped", 0)
    self._schedules_start_task = asyncio.ensure_future(engine.start())

  async def _stop_schedules(self) -> None:
    """End the engine; unrun rows are left intact (FR35)."""
    engine = self._schedules_engine
    self._schedules_engine = None
    self._schedules_stamp = None
    if engine is None:
      return
    try:
      await engine.stop()
    except asyncio.CancelledError:
      pass

  async def _sync_schedules(self) -> None:
    """Re-seat the engine after a live config reload (FR23)."""
    runtime = self._runtime_of(self._agent)
    stamp = getattr(runtime, "schedules_stamped", None) if runtime else None
    if self._schedules_stamp is not None:
      if stamp == self._schedules_stamp:
        return
      await self._stop_schedules()
    handle = getattr(self._agent, "schedules", None)
    if self._schedules_engine is None and handle is not None:
      self._start_schedules(self._agent)

  # ---- schedules turn scope ------------------------------------------------

  def _schedules_origin_for(
    self,
    channel_obj: Any,
    area: tuple[int | None, str | None],
    message: Any,
  ) -> ScheduleOrigin:
    """The schedule origin of THIS turn (same logic as reminders)."""
    channel_id, channel_name = area
    root_id = channel_id if channel_id is not None else getattr(channel_obj, "id", None)
    root_name = (
      channel_name
      if channel_name is not None
      else str(getattr(channel_obj, "name", "") or "")
    )
    author_id = getattr(getattr(message, "author", None), "id", None)
    if author_id is None:
      author_id = getattr(message, "author_id", None)
    return ScheduleOrigin(
      agent=self._agent.config.agent.name,
      platform="discord",
      requester_id=str(author_id) if author_id is not None else "",
      requester_mention=f"<@{author_id}>" if author_id is not None else "",
      channel_id=str(root_id) if root_id is not None else "",
      channel_name=root_name,
    )

  def _schedules_turn_open(
    self,
    message: Any,
    channel_obj: Any,
    area: tuple[int | None, str | None],
  ) -> str:
    schedule_turn_scope.set(
      ScheduleTurnScope(origin=self._schedules_origin_for(channel_obj, area, message))
    )
    return "open"

  def _schedules_turn_close(self, token: Any) -> None:
    if token is None:
      return
    if schedule_turn_scope.get() is not None:
      schedule_turn_scope.set(None)

  async def stop(self) -> None:
    """Stop the Discord bot and clean up sessions."""
    self._running = False
    await self._stop_reminders()
    await self._stop_schedules()
    if self._adapter is not None:
      await self._adapter.stop()
    self._session_map.clear()

  async def run(self, agent: "Agent") -> None:
    """Run the Discord bot loop.

    This blocks until stop() is called. The adapter handles the
    actual discord.py event loop internally.
    """
    if self._adapter is None:
      await self.start(agent)

    # Run the bot — this blocks until the bot is stopped
    await self._adapter.run()

  async def _resolve_target_channel(
    self,
    message: Any,
    channel_id: int,
    channel_obj: Any,
  ) -> int:
    """Resolve the target channel ID for the response.

    When auto_thread is disabled, returns the message's channel ID
    (the identity path).

    When auto_thread is enabled:
    - If the message is already in a thread, return the thread ID
      (continuation of that conversation).
    - If the message is in a parent channel, ALWAYS create a fresh
      thread and return its ID. Each parent-channel mention is a new
      conversation. (Reusing one thread per parent funnels distinct
      topics into the first thread — the wrong-topic bug, obsidian-a3q.)

    Args:
        message: discord.Message object.
        channel_id: The message's channel ID.
        channel_obj: The message's channel object.

    Returns:
        The channel ID to send responses to.
    """
    logger.info(
      "resolve_target: auto_thread=%s, channel_id=%d, channel_type=%s",
      self.config.auto_thread,
      channel_id,
      type(channel_obj).__name__,
    )
    if not self.config.auto_thread:
      return channel_id

    # If we're already inside a thread, use it as-is (continuation)
    if self._adapter and self._adapter.is_thread_channel(channel_obj):
      return channel_id

    # We're in a parent channel — every mention starts a NEW thread.
    message_id = getattr(message, "id", None)
    if message_id is None or self._adapter is None:
      return channel_id

    # Build thread name from the first line of the message,
    # stripping bot mentions so "<@123456> deploy prod" → "deploy prod"
    content = getattr(message, "content", "")
    if self._adapter:
      for mention in self._adapter.bot_mentions:
        content = content.replace(mention, "")
    first_line = content.strip().split("\n")[0][:70]
    thread_name = first_line or "SR2 conversation"

    thread_id = await self._adapter.create_thread(channel_id, thread_name, message_id)
    if thread_id is not None:
      return thread_id

    # Thread creation failed — fall back to parent channel
    logger.warning(
      "Thread creation failed, falling back to parent channel %d",
      channel_id,
    )
    return channel_id

  def _stamp_area(self, channel_obj: Any) -> None:
    """Derive this message's area and stamp it on the agent's RunContext.

    Runs after ``should_respond`` and after ``_apply_agent_config()``, so
    the ``channel_areas`` map in force is the one this message loaded.
    Uses the message's own channel — not any thread later created for the
    reply — so a fresh auto-created thread never shifts the area.

    Discord always resolves an area: "no area" reaches ``RunContext`` as
    ``""``, never ``None`` — ``None`` means "this interface does not do
    areas" and would let ``PlanResolver`` fall through to
    ``SR2_PROJECT``/cwd, the silent-substitution bug this feature exists
    to remove.
    """
    if self._adapter is None or self._agent is None:
      return

    area_channel_id, area_channel_name = self._adapter.area_channel(channel_obj)
    resolved, provenance = resolve_area(
      area_channel_id, area_channel_name, self.config.channel_areas
    )

    self._agent.set_run_context(
      RunContext(
        interface="discord",
        mode=RunMode.INTERACTIVE,
        source=None,
        area=resolved if resolved is not None else "",
      )
    )

    if provenance == "override" and resolved is None:
      logger.info(
        "area=none (channel_areas override, channel=%s)",
        area_channel_id,
      )
    elif resolved is None:
      logger.info("area=none (channel=%s)", area_channel_id)
    elif provenance == "override":
      logger.info(
        "area=%s (channel_areas override, channel=%s)",
        resolved,
        area_channel_id,
      )
    else:
      logger.info(
        "area=%s (derived from channel name, channel=%s)",
        resolved,
        area_channel_id,
      )

  async def _process_message(self, message: Any) -> None:
    """Process an incoming Discord message.

    Extracts channel_id, content, and author from the discord.Message
    object, then routes through the handler logic.

    When auto_thread is enabled, determines the effective channel ID:
    if the message is already in a thread, use the thread ID;
    if the message is in a parent channel, create a thread and route
    to the thread ID.

    Args:
        message: discord.Message object from discord.py.
    """
    if self._adapter is None or self._agent is None:
      return

    # The adapter reloaded the config when this message arrived; push the
    # non-Discord half of that load into the agent before handling it.
    self._apply_agent_config()

    # Extract plain data from discord.Message
    channel_obj = getattr(message, "channel", None)
    if channel_obj is None:
      logger.warning("Could not extract channel from message")
      return

    channel_id = getattr(channel_obj, "id", None)
    if channel_id is None:
      logger.warning("Could not extract channel_id from message")
      return

    content = getattr(message, "content", "")
    if not content:
      return

    bot_id = self._adapter.bot_id
    bot_mentions = self._adapter.bot_mentions

    # Skip mention check if we're inside a thread where the agent
    # already has an active session — the conversation was started by
    # a mention (or mention_only was off), no need to mention again.
    in_active_thread = (
      self._adapter
      and self._adapter.is_thread_channel(channel_obj)
      and channel_id in self._session_map.active()
    )

    if not should_respond(
      content, self.config.mention_only and not in_active_thread, bot_id, bot_mentions
    ):
      return

    self._stamp_area(channel_obj)

    # Reminders (spc-141): carry this turn's origin and guild, and follow a
    # live-reload re-seat (spc-143) before this turn's tool can create a row.
    reminders_token = self._reminders_turn_open(
      message, channel_obj, self._adapter.area_channel(channel_obj)
    )
    schedules_token = self._schedules_turn_open(
      message, channel_obj, self._adapter.area_channel(channel_obj)
    )
    try:
      await self._sync_reminders()
      await self._sync_schedules()
      # Parse slash commands
      command, rest = parse_slash_command(content)

      if command is not None:
        target = await self._resolve_target_channel(
          message, channel_id, channel_obj
        )
        await self._handle_command(command, rest, target)
        return

      # Determine effective channel for threading
      target_channel_id = await self._resolve_target_channel(
        message, channel_id, channel_obj
      )

      # Regular message — process through the agent (cancellable via /stop)
      await self._run_agent(content, target_channel_id)
    finally:
      self._schedules_turn_close(schedules_token)
      self._reminders_turn_close(reminders_token)

  def _command_context(self, channel_id: int) -> CommandContext:
    """Build a CommandContext, enriched with live model info when available.

    The whole-config path (DiscordConfigView) carries models and the
    active-model pointer, so /status can show what the agent is talking
    to. Other paths (tests, embedders) leave those fields None.
    """
    session = self._session_map.get_or_create(channel_id)
    active_model: str | None = None
    model_label: str | None = None
    if isinstance(self._config_source, DiscordConfigView):
      cfg = self._config_source.source.current
      active_model = cfg.active_model
      mc = cfg.active_model_config
      model_label = mc.model if mc.base_url is None else f"{mc.model} @ {mc.base_url}"
    rc = self._agent.run_context if self._agent is not None else None
    area = rc.area if rc is not None and isinstance(rc.area, str) else None
    return CommandContext(
      channel_id=channel_id,
      session_id=session.session_id,
      message_count=len(session.history),
      active_model=active_model,
      model_label=model_label,
      area=area,
    )

  async def _run_agent(self, content: str, channel_id: int) -> None:
    """Run one agent turn as a cancellable task so /stop can abort it.

    The turn runs as a tracked ``asyncio.Task`` keyed by channel. If /stop
    cancels it, ``CancelledError`` (a BaseException, so the stream loop's
    ``except Exception`` never swallows it) unwinds ``_process_through_agent``
    without finalizing; we then render the partial output plus a stop note.

    The task is born from the CALLER'S context (spc-141): a task copies
    the context current where it is created, so the reminder turn scope
    set around ``_process_message`` reaches the tool the model calls
    over here.
    """
    task = asyncio.ensure_future(self._process_through_agent(content, channel_id))
    self._active_runs[channel_id] = task
    try:
      await task
    except asyncio.CancelledError:
      await self._finalize_cancelled(channel_id)
    finally:
      if self._active_runs.get(channel_id) is task:
        del self._active_runs[channel_id]

  async def _finalize_cancelled(self, channel_id: int) -> None:
    """Render a stopped turn: partial streamed text + a stop marker.

    Keeps channel history paired (the user turn was already appended) by
    recording the partial text as the assistant turn, so the next message
    does not see two user turns in a row.
    """
    self._cancel_pending_stream_edit()
    session = self._session_map.get_or_create(channel_id)
    thinking_id = session.pending_message_id
    partial = self._stream_text.get(channel_id, "")
    note = "⏹ Stopped."
    display = f"{partial}\n\n{note}" if partial.strip() else note
    if thinking_id is not None and self._adapter:
      await self._adapter.edit_message(channel_id, thinking_id, display)
    session.history.append(
      {
        "role": "assistant",
        "content": [{"type": "text", "text": partial or "(stopped)"}],
      }
    )
    session.pending_message_id = None

  async def _stop_command(self, channel_id: int) -> str:
    """Handle /stop (and its /cancel alias) — cancel this channel's run."""
    task = self._active_runs.get(channel_id)
    if task is None or task.done():
      return "Nothing is running in this channel."
    task.cancel()
    return "⏹ Stopping the current run."

  async def _model_command(self, text: str) -> str:
    """Handle /model — list models or switch the active one.

    Reads the live model list from the whole-config source and persists a
    switch by writing the model name to the writable pointer file
    (``$SR2_ACTIVE_MODEL_FILE``) — the only writable surface, since the
    config itself is mounted read-only. The next message's config reload
    retargets the LLM. Returns the text to send back.
    """
    cfg = None
    if isinstance(self._config_source, DiscordConfigView):
      cfg = self._config_source.source.current
    if cfg is None:
      return "⚠ Model switching isn't available in this configuration."

    pointer = os.environ.get("SR2_ACTIVE_MODEL_FILE")
    response, selection = render_model_command(
      text, set(cfg.models), cfg.active_model, pointer is not None
    )
    if selection is not None and pointer:
      try:
        Path(pointer).write_text(f"{selection}\n")
      except OSError as exc:
        logger.warning("Failed to write model pointer %s: %s", pointer, exc)
        return f"⚠ Could not persist the switch: {exc}"
    return response

  async def _handle_command(
    self,
    command: str,
    rest: str,
    channel_id: int,
  ) -> None:
    """Process a slash command.

    Args:
        command: Command name (already lowercase).
        rest: Remainder of message content.
        channel_id: Discord channel ID.
    """
    # /hb — probe Harbinger via the CLI, bypassing the LLM entirely.
    if command == "hb":
      output = await probe_harbinger_status()
      if self._adapter is not None:
        for chunk in chunk_message(output, self.config.max_message_length):
          await self._adapter.send_message(channel_id, chunk)
      return

    # /model — list or switch the active model. Writes the pointer file;
    # the next message's reload retargets the LLM.
    if command == "model":
      response = await self._model_command(rest)
      if self._adapter is not None:
        for chunk in chunk_message(response, self.config.max_message_length):
          await self._adapter.send_message(channel_id, chunk)
      return

    # /stop (+ /cancel alias) — cancel the current run in this channel.
    if command in ("stop", "cancel"):
      response = await self._stop_command(channel_id)
      if self._adapter is not None:
        await self._adapter.send_message(channel_id, response)
      return

    # /retry — re-run the last user message, dropping the turn it replaces.
    if command == "retry":
      if self._adapter is not None:
        session = self._session_map.get_or_create(channel_id)
        recalled = split_for_retry(session.history)
        if recalled is None:
          await self._adapter.send_message(
            channel_id,
            "Nothing to retry — no earlier message in this channel.",
          )
        else:
          text, trimmed = recalled
          session.history[:] = trimmed
          await self._run_agent(text, channel_id)
      return

    ctx = self._command_context(channel_id)
    response = handle_command(command, rest, ctx)

    if command == "reset":
      self._session_map.reset(channel_id)

    if response is not None:
      if self._adapter is not None:
        chunks = chunk_message(response, self.config.max_message_length)
        for chunk in chunks:
          await self._adapter.send_message(channel_id, chunk)

    # /ask with content — process through agent
    if command == "ask" and rest.strip():
      await self._run_agent(rest, channel_id)

  async def _handle_slash(
    self,
    name: str,
    text: str,
    interaction: Any,
  ) -> None:
    """Dispatch a native Discord slash command.

    Mirrors the text-prefix path (``_handle_command``) but responds through
    the interaction instead of a channel message. Slash commands arrive on
    the interactions gateway, so they bypass the on_message mention/channel
    filters — the channel allowlist is re-applied here to match the message
    path.

    Args:
        name: Command name (without leading /), already lowercase.
        text: Command argument text (only ``/ask`` uses it).
        interaction: The discord.py Interaction (treated opaquely; all
            discord.py calls route through the adapter).
    """
    if self._adapter is None or self._agent is None:
      return

    if not self._adapter.access_allowed(interaction):
      await self._adapter.interaction_send(
        interaction,
        "⚠ Commands aren't enabled for this user, server, or channel.",
      )
      return

    # The adapter reloaded config when the interaction arrived; push the
    # non-Discord half into the agent, mirroring the message path.
    self._apply_agent_config()

    channel_obj = getattr(interaction, "channel", None)
    channel_id = getattr(interaction, "channel_id", None)
    if channel_id is None and channel_obj is not None:
      channel_id = getattr(channel_obj, "id", None)
    if channel_id is None:
      await self._adapter.interaction_send(
        interaction, "⚠ Could not resolve this channel."
      )
      return

    if channel_obj is not None:
      self._stamp_area(channel_obj)

    # Reminders (spc-141): this interaction's turn carries its own origin
    # and guild, and follows a live-reload re-seat, exactly like the
    # message path.
    reminders_token = self._reminders_turn_open(
      interaction,
      channel_obj,
      self._adapter.area_channel(channel_obj) if channel_obj else (None, None),
    )
    schedules_token = self._schedules_turn_open(
      interaction,
      channel_obj,
      self._adapter.area_channel(channel_obj) if channel_obj else (None, None),
    )
    try:
      await self._sync_reminders()
      await self._sync_schedules()
      await self._dispatch_slash(name, text, interaction, channel_id, channel_obj)
    finally:
      self._schedules_turn_close(schedules_token)
      self._reminders_turn_close(reminders_token)

  async def _dispatch_slash(
    self,
    name: str,
    text: str,
    interaction: Any,
    channel_id: int,
    channel_obj: Any,
  ) -> None:
    """The slash command bodies, running inside the turn's reminder scope."""
    # /hb — probe Harbinger via the CLI, bypassing the LLM. Can take a few
    # seconds, so defer first to keep the interaction alive.
    if name == "hb":
      await self._adapter.interaction_defer(interaction)
      output = await probe_harbinger_status()
      for chunk in chunk_message(output, self.config.max_message_length):
        await self._adapter.interaction_send(interaction, chunk)
      return

    # /model — list or switch the active model.
    if name == "model":
      response = await self._model_command(text)
      for chunk in chunk_message(response, self.config.max_message_length):
        await self._adapter.interaction_send(interaction, chunk)
      return

    # /stop (+ /cancel alias) — cancel the current run in this channel.
    if name in ("stop", "cancel"):
      response = await self._stop_command(channel_id)
      await self._adapter.interaction_send(interaction, response)
      return

    # /retry — re-run the last user message. Ack with the recalled prompt as
    # a visible anchor, then stream into the channel like a normal message.
    if name == "retry":
      session = self._session_map.get_or_create(channel_id)
      recalled = split_for_retry(session.history)
      if recalled is None:
        await self._adapter.interaction_send(
          interaction,
          "Nothing to retry — no earlier message in this channel.",
        )
        return
      text, trimmed = recalled
      session.history[:] = trimmed
      await self._adapter.interaction_send(interaction, f"🔁 {text}")
      await self._run_agent(text, channel_id)
      return

    # /ask — run the agent loop. Acknowledge with the prompt as a visible
    # anchor (slash input is otherwise hidden from other members), then
    # stream into the channel exactly like a normal message.
    if name == "ask":
      prompt = text.strip()
      if not prompt:
        await self._adapter.interaction_send(
          interaction, "⚠ `/ask` needs a message. Try `/ask <your question>`."
        )
        return
      await self._adapter.interaction_send(interaction, f"💭 {prompt}")
      await self._run_agent(prompt, channel_id)
      return

    # Text-producing commands: /reset, /status, /help.
    ctx = self._command_context(channel_id)
    response = handle_command(name, text, ctx)

    if name == "reset":
      self._session_map.reset(channel_id)

    if response is not None:
      for chunk in chunk_message(response, self.config.max_message_length):
        await self._adapter.interaction_send(interaction, chunk)
    else:
      await self._adapter.interaction_send(interaction, "Done.")

  async def _process_through_agent(
    self,
    content: str,
    channel_id: int,
  ) -> None:
    """Route a message through the Agent and stream the response.

    Orchestrates the full flow: session setup, agent call with stream
    rendering, history management, and final message delivery.

    Args:
        content: The user's message text.
        channel_id: Discord channel ID.
    """
    if self._adapter is None or self._agent is None:
      return

    session = self._session_map.get_or_create(channel_id)

    # Append user message to channel history
    session.history.append(
      {"role": "user", "content": [{"type": "text", "text": content}]}
    )

    # Send initial placeholder message for streaming
    thinking_msg = await self._adapter.send_message(channel_id, "⏳ Thinking...")
    thinking_id = getattr(thinking_msg, "id", None) if thinking_msg else None
    session.pending_message_id = thinking_id

    # Reset tool lines and streamed text for this channel — each agent
    # turn gets a fresh log.
    self._tool_lines[channel_id] = []
    self._stream_text[channel_id] = ""

    async with self._agent_turn_lock:
      # Restore channel history into the agent
      self._restore_history(session)

      # Show typing indicator while calling the agent (per-message, visible to Discord users).
      # The typing context manager auto-clears when the first message edit/send happens.
      async with self._adapter.channel_typing(channel_id):
        # Drive the agent and render stream events to Discord
        response_parts, stream_error = await self._drive_agent_stream(
          content, channel_id, thinking_id
        )

    if stream_error is not None:
      session.pending_message_id = None
      return

    # Finalize: set the complete response
    final_text = "".join(response_parts)
    if not final_text:
      final_text = "(No response generated)"

    # Append assistant response to channel history
    session.history.append(
      {"role": "assistant", "content": [{"type": "text", "text": final_text}]}
    )

    # Cancel any pending streaming edit — it would overwrite our
    # finalized message with the "..." version (race condition).
    self._cancel_pending_stream_edit()

    # Detect local image paths in the final response and upload them
    # to Discord before sending the text.
    if self._adapter:
      final_text = await self._upload_images_in_text(final_text, channel_id)

    # Send final message(s) — chunked if too long
    if thinking_id is not None and self._adapter:
      chunks = chunk_message(final_text, self.config.max_message_length)
      if len(chunks) == 1:
        # Single chunk — edit the thinking message
        await self._adapter.edit_message(channel_id, thinking_id, chunks[0])
      else:
        # Multiple chunks — edit first, send rest as new messages
        await self._adapter.edit_message(channel_id, thinking_id, chunks[0])
        for chunk in chunks[1:]:
          await self._adapter.send_message(channel_id, chunk)

    session.pending_message_id = None

  async def _drive_agent_stream(
    self,
    content: str,
    channel_id: int,
    thinking_id: int | None,
  ) -> tuple[list[str], Exception | None]:
    """Drive the agent's message stream and render events to Discord.

    Handles the event loop: collects text deltas, sends stream edits,
    renders tool start/result embeds. Separates the Discord rendering
    concern from the orchestration in _process_through_agent.

    Args:
        content: The user's message text.
        channel_id: Discord channel ID.
        thinking_id: ID of the thinking placeholder message.

    Returns:
        Tuple of (collected response text parts, error or None).
        If error is not None, the thinking message was already updated
        with the error and no finalization should occur.
    """
    response_parts: list[str] = []
    last_edit_time: float | None = None
    loop = asyncio.get_event_loop()

    try:
      async for event in self._agent.stream_message(content):
        if isinstance(event, AgentTextDelta):
          response_parts.append(event.text)
          self._maybe_edit_stream(
            channel_id,
            thinking_id,
            "".join(response_parts),
            last_edit_time,
            loop,
          )
          last_edit_time = loop.time()

        elif isinstance(event, AgentToolStart):
          await self._render_tool_start(event, channel_id)

        elif isinstance(event, AgentToolResult):
          await self._render_tool_result(event, channel_id)

        # AgentDone is handled after the loop

    except Exception as exc:
      logger.error("Agent stream error: %s", exc)
      error_msg = f"Error: {exc}"
      if thinking_id is not None and self._adapter:
        await self._adapter.edit_message(channel_id, thinking_id, error_msg)
      return response_parts, exc

    return response_parts, None

  async def _render_tool_start(
    self,
    event: AgentToolStart,
    channel_id: int,
  ) -> None:
    """Render a tool-start event by appending a line to the thinking message.

    Tool activity lines accumulate in the thinking message itself so
    everything stays in one message — no separate tool-log to scroll past.

    Args:
        event: The AgentToolStart event.
        channel_id: Discord channel ID.
    """
    if not self.config.tool_embed_enabled:
      return
    inputs_max_length = 80
    inputs = str(event.input)
    if len(inputs) > inputs_max_length:
      inputs = inputs[:inputs_max_length] + "..."

    line = f"▶ `{event.name}` - `{inputs}`"
    self._tool_lines.setdefault(channel_id, []).append(line)
    self._flush_tool_lines(channel_id)

  async def _render_tool_result(
    self,
    event: AgentToolResult,
    channel_id: int,
  ) -> None:
    """Render a tool-result event by appending a line to the thinking message.

    Tool activity lines accumulate in the thinking message itself so
    everything stays in one message — no separate tool-log to scroll past.

    Args:
        event: The AgentToolResult event.
        channel_id: Discord channel ID.
    """
    if not self.config.tool_embed_enabled:
      return

    icon = "✖" if event.is_error else "✓"
    tool_name = event.name or "tool"
    content_max_length = 80
    content = str(event.content).replace("\n", "|")
    if len(content) > content_max_length:
      content = content[:content_max_length] + "..."

    line = f"{icon} `{tool_name}` - `{content}`"
    self._tool_lines.setdefault(channel_id, []).append(line)
    self._flush_tool_lines(channel_id)

  def _flush_tool_lines(self, channel_id: int) -> None:
    """Re-render the thinking message after a tool event.

    Delegates to the unified progress renderer so tool lines and the
    accumulated streamed text are shown TOGETHER in one message — they
    no longer clobber each other.

    Args:
        channel_id: Discord channel ID.
    """
    session = self._session_map.get_or_create(channel_id)
    self._render_progress(channel_id, session.pending_message_id)

  def _render_progress(
    self,
    channel_id: int,
    thinking_id: int | None,
  ) -> None:
    """Render combined progress (tool lines + streamed text) to the thinking message.

    Single source of truth for the in-progress message. Both the tool
    path and the text-delta path funnel through here, so the message
    always shows everything accumulated so far instead of flipping
    between tool-only and text-only content.

    The edit is scheduled on the tracked ``_pending_stream_edit`` future
    so finalization (or a newer progress render) can cancel a stale edit
    before it lands — the same race guard the text path already used.

    Args:
        channel_id: Discord channel ID.
        thinking_id: ID of the thinking placeholder message.
    """
    if thinking_id is None or self._adapter is None:
      return

    lines = self._tool_lines.get(channel_id, [])
    text = self._stream_text.get(channel_id, "")

    parts: list[str] = []
    if lines:
      parts.append("\n".join(lines))
    # Show the streamed text so far with an in-progress marker; fall back
    # to the placeholder when no text has arrived yet.
    parts.append(text + "..." if text else "⏳ Thinking...")

    content = "\n\n".join(parts)
    content = tail_for_stream(content, self.config.max_message_length)

    # Supersede any pending edit — this render reflects the latest state.
    self._cancel_pending_stream_edit()
    self._pending_stream_edit = asyncio.ensure_future(
      self._adapter.edit_message(channel_id, thinking_id, content)
    )

  async def _upload_images_in_text(self, text: str, channel_id: int) -> str:
    """Detect local image paths in response text and upload to Discord.

    Handles two patterns:
    1. Markdown images: ![alt](/path/to/image.png)
    2. Plain text paths: "sitting at /tmp/spectre_images/spectre.png"

    Uploads each image via send_image() and replaces the reference
    with "[📎 image attached]" so the text remains clean.

    Args:
        text: The final response text.
        channel_id: Discord channel ID.

    Returns:
        Text with image references replaced by attachment markers.
    """
    if self._adapter is None:
      return text

    # Pattern 1: Markdown images ![alt](/path)
    md_pattern = re.compile(r"!\[([^\]]*)\]\((/[^\)]+)\)")
    for m in list(md_pattern.finditer(text)):
      img_path = m.group(2)
      if await self._try_upload(img_path, channel_id):
        text = text.replace(m.group(0), "📎 [image attached]")

    # Pattern 2: Plain text paths (common image extensions)
    plain_pattern = re.compile(
      r"(/(?:\w+/)*\.(?:png|jpg|jpeg|gif|webp|bmp))",
      re.IGNORECASE,
    )
    for m in list(plain_pattern.finditer(text)):
      img_path = m.group(1)
      if await self._try_upload(img_path, channel_id):
        text = text.replace(m.group(0), "📎 [image attached]")

    return text

  async def _try_upload(self, image_path: str, channel_id: int) -> bool:
    """Try to upload a single image to Discord.

    Args:
        image_path: Absolute path to the image file.
        channel_id: Discord channel ID.

    Returns:
        True if the upload succeeded, False otherwise.
    """
    if self._adapter is None:
      return False
    try:
      from pathlib import Path

      path = Path(image_path)
      if not path.exists():
        logger.warning("Image path does not exist: %s", image_path)
        return False
      await self._adapter.send_image(channel_id, path)
      logger.info("Uploaded image to Discord: %s", image_path)
      return True
    except Exception:
      logger.exception("Failed to upload image: %s", image_path)
      return False

  def _cancel_pending_stream_edit(self) -> None:
    """Cancel any pending streaming edit future to prevent it from
    overwriting a finalized message.
    """
    if self._pending_stream_edit is not None and not self._pending_stream_edit.done():
      self._pending_stream_edit.cancel()
      self._pending_stream_edit = None

  def _maybe_edit_stream(
    self,
    channel_id: int,
    message_id: int | None,
    current_text: str,
    last_edit_time: float | None,
    loop: asyncio.AbstractEventLoop,
  ) -> None:
    """Edit the streaming message if enough time has passed.

    Uses asyncio.ensure_future to avoid blocking the event loop.
    The pending future is tracked so it can be cancelled before
    finalization or when a newer edit supersedes it.

    Args:
        channel_id: Discord channel ID.
        message_id: ID of the message being edited.
        current_text: Current accumulated response text.
        last_edit_time: Time of the last edit (None for first edit).
        loop: Current asyncio event loop.
    """
    if message_id is None or self._adapter is None:
      return

    # Always record the latest text so a tool flush in between throttled
    # edits still renders the up-to-date streamed text.
    self._stream_text[channel_id] = current_text

    if self.config.edit_stream_interval <= 0:
      return  # Streaming edits disabled

    now = loop.time()
    if (
      last_edit_time is None
      or (now - last_edit_time) >= self.config.edit_stream_interval
    ):
      # Render combined progress (tool lines + this text) through the
      # single tracked future.
      self._render_progress(channel_id, message_id)

  def _restore_history(self, session: "ChannelSession") -> None:
    """Restore channel history into the agent's history list.

    Replaces the agent's current history with the channel's
    conversation history, so the agent has context for this
    specific channel's conversation.

    Args:
        session: The ChannelSession containing serialized history.
    """
    from sr2.models import Message, TextBlock

    # Agent.history is a getter-only property (delegates to Session) — never
    # assign to it. Setting session_id rebuilds a fresh Session (empty
    # history) under the shared Runtime, which both resets history and
    # points the agent at this channel's frame. The setter carries the run
    # context over to the replacement Session.
    self._agent.session_id = session.session_id

    for entry in session.history:
      role = entry.get("role", "user")
      blocks = []
      for block in entry.get("content", []):
        if block.get("type") == "text":
          blocks.append(TextBlock(text=block.get("text", "")))
      if blocks:
        self._agent.history.append(Message(role=role, content=blocks))
