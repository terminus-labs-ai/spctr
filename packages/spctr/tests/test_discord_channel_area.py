"""Channel -> area resolution on the Discord interface.

Spec: ``specs/channel-area-injection.md`` (bead spc-48).

  AC 1  — a message in a parent channel named ``fractured-roots`` results in
          ``RunContext.area == "fractured-roots"`` at the time the agent runs.
  AC 2  — a follow-up message inside the auto-created thread resolves to the
          same area as the message that created the thread.
  AC 3  — a ``channel_areas`` entry overrides the derived name for that
          channel ID.
  AC 4  — a ``channel_areas`` entry with an empty-string value yields no area.
  AC 5  — a DM, or a channel whose name is unreadable, yields no area.
  AC 6  — adding a ``channel_areas`` entry takes effect on the next message
          with no process restart.
  AC 23 — one INFO line per Discord message records the resolved area and how
          it was derived.

Discord always resolves areas, so "no area" reaches ``RunContext`` as the
empty string, never as ``None``: ``None`` means "this interface does not do
areas" and would let ``PlanResolver`` fall through to ``SR2_PROJECT``/cwd —
the silent-substitution bug this feature exists to remove (FR 10, NFR
"Failure mode", and the *Unmapped channel* decision).
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.events import AgentDone, AgentTextDelta
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.config_source import DiscordConfigSource
from sr2_spectre.interfaces.discord.interface import DiscordInterface

PARENT_ID = 555
THREAD_ID = 999
FRACTURED = "fractured-roots"


# ---------------------------------------------------------------------------
# Pure name derivation — FR 3
# ---------------------------------------------------------------------------

class TestDeriveAreaName:
    @pytest.mark.parametrize(
        "channel_name,expected",
        [
            ("fractured-roots", "fractured-roots"),
            ("Fractured-Roots", "fractured-roots"),
            ("FRACTURED-ROOTS", "fractured-roots"),
            ("🔥fractured-roots🔥", "fractured-roots"),
            ("  general  ", "general"),
            ("--general--", "general"),
            ("sr2_spectre", "sr2_spectre"),
            ("area-2", "area-2"),
            # Nothing alphanumeric survives the strip: no name, so no area.
            # "" must not leak downstream — resolve_area is spec'd never to
            # return it, and "" on RunContext means "explicitly no area".
            ("---", None),
            ("\U0001f525", None),
            ("", None),
        ],
    )
    def test_derivation(self, channel_name: str, expected: str | None) -> None:
        from sr2_spectre.interfaces.discord.handler import derive_area_name

        assert derive_area_name(channel_name) == expected

    def test_none_stays_none(self) -> None:
        from sr2_spectre.interfaces.discord.handler import derive_area_name

        assert derive_area_name(None) is None


# ---------------------------------------------------------------------------
# Pure override lookup — FR 4, FR 5, FR 6
# ---------------------------------------------------------------------------

class TestResolveArea:
    @pytest.mark.parametrize(
        "channel_id,channel_name,channel_areas,expected",
        [
            (PARENT_ID, "Fractured-Roots", {}, (FRACTURED, "derived")),
            (PARENT_ID, FRACTURED, {"555": "grindsourced"}, ("grindsourced", "override")),
            (PARENT_ID, FRACTURED, {"111": "grindsourced"}, (FRACTURED, "derived")),
            (PARENT_ID, FRACTURED, {"555": ""}, (None, "override")),
            (None, None, {}, (None, None)),
            (None, None, {"555": "grindsourced"}, (None, None)),
            (PARENT_ID, "---", {}, (None, None)),
        ],
        ids=[
            "derived-name",
            "override-wins",
            "entry-for-another-channel",
            "empty-override-is-no-area",
            "unreadable-channel",
            "unreadable-channel-with-a-map",
            "name-strips-to-empty",
        ],
    )
    def test_resolution(
        self,
        channel_id: int | None,
        channel_name: str | None,
        channel_areas: dict[str, str],
        expected: tuple[str | None, str | None],
    ) -> None:
        """No area is None — never "" (the interface owns that distinction).

        Provenance names a winning override or usable derived name. It is
        None when neither source produces an area.
        """
        from sr2_spectre.interfaces.discord.handler import resolve_area

        assert resolve_area(channel_id, channel_name, channel_areas) == expected


# ---------------------------------------------------------------------------
# DiscordConfig.channel_areas
# ---------------------------------------------------------------------------

class TestChannelAreasConfig:
    def test_defaults_to_empty_map(self) -> None:
        assert DiscordConfig().channel_areas == {}

    def test_accepts_id_to_name_entries(self) -> None:
        config = DiscordConfig(channel_areas={"555": "grindsourced", "666": ""})
        assert config.channel_areas == {"555": "grindsourced", "666": ""}


# ---------------------------------------------------------------------------
# Adapter — the area-bearing channel, including the thread parent walk (FR 2, 6)
# ---------------------------------------------------------------------------

class TestAreaChannel:
    def _adapter(self):
        from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter

        return DiscordBotAdapter(DiscordConfig(token="fake"))

    def test_text_channel_is_its_own_area_channel(self) -> None:
        discord = pytest.importorskip("discord")
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = PARENT_ID
        channel.name = FRACTURED

        assert self._adapter().area_channel(channel) == (PARENT_ID, FRACTURED)

    def test_thread_resolves_to_its_parent(self) -> None:
        discord = pytest.importorskip("discord")
        parent = MagicMock(spec=discord.TextChannel)
        parent.id = PARENT_ID
        parent.name = FRACTURED
        thread = MagicMock(spec=discord.Thread)
        thread.id = THREAD_ID
        thread.name = "help me with factorio"
        thread.parent = parent

        assert self._adapter().area_channel(thread) == (PARENT_ID, FRACTURED)

    def test_orphaned_thread_has_no_area_channel(self) -> None:
        discord = pytest.importorskip("discord")
        thread = MagicMock(spec=discord.Thread)
        thread.id = THREAD_ID
        thread.name = "orphan"
        thread.parent = None

        assert self._adapter().area_channel(thread) == (None, None)

    def test_text_channel_with_no_readable_name_has_no_area_channel(self) -> None:
        discord = pytest.importorskip("discord")
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = PARENT_ID
        channel.name = None

        assert self._adapter().area_channel(channel) == (None, None)

    def test_dm_has_no_area_channel(self) -> None:
        discord = pytest.importorskip("discord")
        dm = MagicMock(spec=discord.DMChannel)
        dm.id = 4242

        assert self._adapter().area_channel(dm) == (None, None)


# ---------------------------------------------------------------------------
# Interface helpers
# ---------------------------------------------------------------------------

def _make_recording_agent() -> MagicMock:
    """A mock Agent that records the run context in force when it is driven."""
    agent = MagicMock()
    agent.history = []
    agent.session_id = "discord-test"
    agent.run_context = None
    state: dict[str, Any] = {"ctx": None}
    seen: list[Any] = []

    def _set_run_context(ctx: Any) -> None:
        state["ctx"] = ctx
        agent.run_context = ctx

    agent.set_run_context = MagicMock(side_effect=_set_run_context)

    async def _stream(_text: str) -> Any:
        seen.append(state["ctx"])
        for ev in (AgentTextDelta(text="ok"), AgentDone(tool_calls_executed=0)):
            yield ev

    agent.stream_message = _stream
    agent.contexts_at_run = seen
    return agent


def _make_mock_adapter(
    areas: dict[int, tuple[int | None, str | None]] | None = None,
    thread_ids: tuple[int, ...] = (),
    new_thread_id: int | None = None,
) -> MagicMock:
    """Mock adapter whose ``area_channel`` answers per channel object."""
    mapping = areas or {}
    adapter = MagicMock()
    adapter.bot_id = 11111
    adapter.bot_mentions = ["<@11111>"]
    adapter.start = AsyncMock()
    adapter.stop = AsyncMock()
    adapter.send_message = AsyncMock(return_value=MagicMock(id=888))
    adapter.edit_message = AsyncMock()
    adapter.send_embed = AsyncMock()
    adapter.set_message_handler = MagicMock()
    adapter.create_thread = AsyncMock(return_value=new_thread_id)
    adapter.is_thread_channel = MagicMock(
        side_effect=lambda ch: getattr(ch, "id", None) in thread_ids
    )
    adapter.area_channel = MagicMock(
        side_effect=lambda ch: mapping.get(getattr(ch, "id", None), (None, None))
    )

    typing_ctx = AsyncMock()
    typing_ctx.__aenter__ = AsyncMock()
    typing_ctx.__aexit__ = AsyncMock(return_value=None)
    adapter.channel_typing = MagicMock(return_value=typing_ctx)
    return adapter


def _message(channel_id: int = PARENT_ID, message_id: int = 1) -> MagicMock:
    message = MagicMock()
    message.content = "hello"
    message.id = message_id
    channel = MagicMock()
    channel.id = channel_id
    message.channel = channel
    message.author = MagicMock(id=99999)
    return message


async def _started(
    adapter: MagicMock,
    config: DiscordConfig | None = None,
    config_source: DiscordConfigSource | None = None,
) -> tuple[DiscordInterface, MagicMock]:
    interface = DiscordInterface(
        config=config or DiscordConfig(), config_source=config_source
    )
    agent = _make_recording_agent()
    with patch(
        "sr2_spectre.interfaces.discord.interface.DiscordBotAdapter",
        return_value=adapter,
    ):
        await interface.start(agent)
    return interface, agent


# ---------------------------------------------------------------------------
# AC 1-6 — what the agent sees
# ---------------------------------------------------------------------------

class TestStampedArea:
    async def test_parent_channel_name_becomes_the_area(self) -> None:
        adapter = _make_mock_adapter({PARENT_ID: (PARENT_ID, FRACTURED)})
        interface, agent = await _started(adapter)

        await interface._process_message(_message())

        ctx = agent.contexts_at_run[0]
        assert ctx.area == FRACTURED
        assert ctx.interface == "discord"
        assert ctx.mode == RunMode.INTERACTIVE
        assert ctx.source is None

    async def test_thread_follow_up_keeps_the_parent_area(self) -> None:
        adapter = _make_mock_adapter(
            areas={
                PARENT_ID: (PARENT_ID, FRACTURED),
                THREAD_ID: (PARENT_ID, FRACTURED),
            },
            thread_ids=(THREAD_ID,),
            new_thread_id=THREAD_ID,
        )
        interface, agent = await _started(adapter, DiscordConfig(auto_thread=True))

        await interface._process_message(_message(channel_id=PARENT_ID, message_id=1))
        await interface._process_message(_message(channel_id=THREAD_ID, message_id=2))

        assert [c.area for c in agent.contexts_at_run] == [FRACTURED, FRACTURED]

    async def test_area_comes_from_the_message_channel_not_the_new_thread(self) -> None:
        """auto_thread creates a fresh thread whose ID maps to nothing; the
        area must still come from the parent channel the message arrived in.
        """
        adapter = _make_mock_adapter(
            areas={PARENT_ID: (PARENT_ID, FRACTURED)},
            new_thread_id=777,
        )
        interface, agent = await _started(adapter, DiscordConfig(auto_thread=True))

        await interface._process_message(_message())

        assert agent.contexts_at_run[0].area == FRACTURED

    async def test_channel_areas_entry_overrides_the_derived_name(self) -> None:
        adapter = _make_mock_adapter({PARENT_ID: (PARENT_ID, FRACTURED)})
        interface, agent = await _started(
            adapter, DiscordConfig(channel_areas={str(PARENT_ID): "grindsourced"})
        )

        await interface._process_message(_message())

        assert agent.contexts_at_run[0].area == "grindsourced"

    async def test_empty_channel_areas_entry_stamps_explicit_no_area(self) -> None:
        adapter = _make_mock_adapter({PARENT_ID: (PARENT_ID, FRACTURED)})
        interface, agent = await _started(
            adapter, DiscordConfig(channel_areas={str(PARENT_ID): ""})
        )

        await interface._process_message(_message())

        assert agent.contexts_at_run[0].area == ""

    async def test_dm_or_unreadable_channel_stamps_explicit_no_area(self) -> None:
        adapter = _make_mock_adapter({})  # area_channel yields (None, None)
        interface, agent = await _started(adapter)

        await interface._process_message(_message(channel_id=4242))

        assert agent.contexts_at_run[0].area == ""

    async def test_new_entry_applies_to_the_next_message(self) -> None:
        """AC 6 — no restart: the second message sees the reloaded map."""
        holder = {"cfg": DiscordConfig()}
        source = DiscordConfigSource(lambda: holder["cfg"], holder["cfg"])
        adapter = _make_mock_adapter({PARENT_ID: (PARENT_ID, FRACTURED)})
        interface, agent = await _started(adapter, config_source=source)

        await interface._process_message(_message(message_id=1))

        holder["cfg"] = DiscordConfig(channel_areas={str(PARENT_ID): "grindsourced"})
        source.reload()  # what the adapter does on every inbound message

        await interface._process_message(_message(message_id=2))

        assert [c.area for c in agent.contexts_at_run] == [FRACTURED, "grindsourced"]


# ---------------------------------------------------------------------------
# Stamping order
# ---------------------------------------------------------------------------

class TestStampOrdering:
    async def test_response_guard_runs_before_area_stamp(self) -> None:
        """A non-mention must not add an area context in mention-only mode."""
        adapter = _make_mock_adapter({PARENT_ID: (PARENT_ID, FRACTURED)})
        interface, agent = await _started(adapter, DiscordConfig(mention_only=True))

        base_calls = list(agent.set_run_context.call_args_list)
        await interface._process_message(_message())

        assert agent.set_run_context.call_args_list == base_calls
        assert base_calls[-1].args[0].area is None

    async def test_area_stamp_uses_config_reloaded_by_apply_agent_config(self) -> None:
        """The current message's config reload precedes its area stamp."""
        holder = {"cfg": DiscordConfig()}
        source = DiscordConfigSource(lambda: holder["cfg"], holder["cfg"])
        adapter = _make_mock_adapter({PARENT_ID: (PARENT_ID, FRACTURED)})
        interface, agent = await _started(adapter, config_source=source)
        holder["cfg"] = DiscordConfig(channel_areas={str(PARENT_ID): "fresh"})

        with patch.object(interface, "_apply_agent_config", side_effect=source.reload) as apply:
            await interface._process_message(_message())

        assert apply.call_count == 1
        assert agent.contexts_at_run[0].area == "fresh"


# ---------------------------------------------------------------------------
# AC 23 — observability
# ---------------------------------------------------------------------------

def _area_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "area" in r.getMessage().lower()
    ]


class TestAreaLogging:
    async def test_one_line_per_message_naming_the_derived_area(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter = _make_mock_adapter({PARENT_ID: (PARENT_ID, FRACTURED)})
        interface, _agent = await _started(adapter)

        with caplog.at_level(logging.INFO):
            await interface._process_message(_message())

        lines = _area_lines(caplog)
        assert len(lines) == 1
        assert FRACTURED in lines[0]
        assert "channel_areas" not in lines[0]

    async def test_line_reports_a_channel_areas_override(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter = _make_mock_adapter({PARENT_ID: (PARENT_ID, FRACTURED)})
        interface, _agent = await _started(
            adapter, DiscordConfig(channel_areas={str(PARENT_ID): "grindsourced"})
        )

        with caplog.at_level(logging.INFO):
            await interface._process_message(_message())

        lines = _area_lines(caplog)
        assert len(lines) == 1
        assert "grindsourced" in lines[0]
        assert "channel_areas" in lines[0]

    async def test_line_reports_the_absence_of_an_area(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter = _make_mock_adapter({})
        interface, _agent = await _started(adapter)

        with caplog.at_level(logging.INFO):
            await interface._process_message(_message(channel_id=4242))

        lines = _area_lines(caplog)
        assert len(lines) == 1
        assert lines[0] == "area=none (channel=None)"

    async def test_line_reports_an_empty_channel_areas_override(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter = _make_mock_adapter({PARENT_ID: (PARENT_ID, FRACTURED)})
        interface, _agent = await _started(
            adapter, DiscordConfig(channel_areas={str(PARENT_ID): ""})
        )

        with caplog.at_level(logging.INFO):
            await interface._process_message(_message())

        lines = _area_lines(caplog)
        assert len(lines) == 1
        assert "area=none" in lines[0]
        assert "channel_areas override" in lines[0]
        assert str(PARENT_ID) in lines[0]


# ---------------------------------------------------------------------------
# Run context after a session rebuild
# ---------------------------------------------------------------------------

def _real_agent_config():
    from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig

    return SpectreConfig(
        agent=AgentConfig(name="test"),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static", "config": {"text": "hi"}}],
                },
                {
                    "name": "conversation",
                    "target": "messages",
                    "resolvers": [{"type": "session"}, {"type": "input"}],
                },
            ]
        },
    )


def _channel_session(session_id: str = "discord-parent-555") -> MagicMock:
    session = MagicMock()
    session.session_id = session_id
    session.history = []
    return session


class TestRunContextSurvivesSessionRebuild:
    def test_named_area_survives_session_rebuild(self) -> None:
        from sr2_spectre.agent import Agent

        with patch("sr2_spectre.session.SR2") as mock_sr2:
            mock_sr2.return_value = MagicMock()
            agent = Agent(config=_real_agent_config())
            agent.set_run_context(
                RunContext(
                    interface="discord",
                    mode=RunMode.INTERACTIVE,
                    source=None,
                    area=FRACTURED,
                )
            )
            interface = DiscordInterface(config=DiscordConfig())
            interface._agent = agent

            interface._restore_history(_channel_session())

            assert agent.run_context is not None
            assert agent.run_context.area == FRACTURED
            provider = mock_sr2.call_args.kwargs["run_context_provider"]
            assert provider()["area"] == FRACTURED

    def test_explicit_no_area_survives_session_rebuild(self) -> None:
        from sr2_spectre.agent import Agent

        with patch("sr2_spectre.session.SR2") as mock_sr2:
            mock_sr2.return_value = MagicMock()
            agent = Agent(config=_real_agent_config())
            agent.set_run_context(
                RunContext(
                    interface="discord",
                    mode=RunMode.INTERACTIVE,
                    source=None,
                    area="",
                )
            )
            interface = DiscordInterface(config=DiscordConfig())
            interface._agent = agent

            interface._restore_history(_channel_session())

            assert agent.run_context is not None
            assert agent.run_context.area == ""
            provider = mock_sr2.call_args.kwargs["run_context_provider"]
            assert provider()["area"] == ""

    def test_absent_context_remains_absent_after_session_rebuild(self) -> None:
        from sr2_spectre.agent import Agent

        with patch("sr2_spectre.session.SR2") as mock_sr2:
            mock_sr2.return_value = MagicMock()
            agent = Agent(config=_real_agent_config())
            interface = DiscordInterface(config=DiscordConfig())
            interface._agent = agent

            interface._restore_history(_channel_session())

            assert agent.run_context is None
            provider = mock_sr2.call_args.kwargs["run_context_provider"]
            assert provider() is None
