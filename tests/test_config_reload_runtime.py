"""Whole-config reload — the agent, not just the interface, tracks the file.

The Discord bot already re-read its own settings on every message, but the
agent's models, endpoints, pipeline and tools stayed frozen at process start.
A wrong ``base_url`` therefore survived every restart-free fix an operator
could make, and survived the restart too if the file itself was wrong.

These tests cover the second half: one reload per message now refreshes the
whole SpectreConfig, and ``Runtime.apply_config`` decides what a live process
can actually re-seat.

Covers:
  A. SpectreConfigSource — reload, last-good fallback, pinned fields
  B. LiveLLM — retarget semantics and delegation
  C. Runtime.apply_config — what gets applied, and what is left alone
  D. Session — config adoption and deferred SR2 rebuild
  E. End to end — editing base_url on disk retargets a running agent
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sr2_spectre.config import (
    AgentConfig,
    McpServerConfig,
    ModelConfig,
    SpectreConfig,
    ToolConfig,
)
from sr2_spectre.config_source import SpectreConfigSource
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.live_llm import LiveLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline_dict(system_text: str = "You are helpful.") -> dict:
    return {
        "layers": [
            {
                "name": "system",
                "target": "system",
                "resolvers": [{"type": "static", "config": {"text": system_text}}],
            },
            {
                "name": "tools",
                "target": "tools",
                "resolvers": [],
                "tool_providers": [{"type": "spectre_tools"}],
            },
            {
                "name": "conversation",
                "target": "messages",
                "resolvers": [{"type": "session"}, {"type": "input"}],
            },
        ]
    }


def _make_config(
    *,
    name: str = "test",
    model: str = "test-model",
    base_url: str = "http://good:11438/v1",
    tools: list[ToolConfig] | None = None,
    mcp_servers: list[McpServerConfig] | None = None,
    discord: DiscordConfig | None = None,
    system_text: str = "You are helpful.",
    params: dict | None = None,
) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(
            name=name,
            tools=tools or [],
            mcp_servers=mcp_servers or [],
        ),
        models={
            "default": ModelConfig(model=model, base_url=base_url, params=params or {})
        },
        pipeline=_pipeline_dict(system_text),
        discord=discord,
    )


# ---------------------------------------------------------------------------
# A. SpectreConfigSource
# ---------------------------------------------------------------------------

class TestSpectreConfigSource:
    def test_reload_picks_up_a_changed_endpoint(self):
        loaded = _make_config(base_url="http://new:11438/v1")
        source = SpectreConfigSource(
            loader=lambda: loaded, initial=_make_config(base_url="http://old:11438/v1")
        )

        assert source.reload().models["default"].base_url == "http://new:11438/v1"
        assert source.current.models["default"].base_url == "http://new:11438/v1"

    def test_bad_config_keeps_the_last_good_one_in_force(self):
        def _explode() -> SpectreConfig:
            raise ValueError("bad yaml")

        initial = _make_config(base_url="http://good:11438/v1")
        source = SpectreConfigSource(loader=_explode, initial=initial)

        assert source.reload() is initial
        assert source.current.models["default"].base_url == "http://good:11438/v1"

    def test_recovers_after_the_file_parses_again(self):
        state = {"broken": True}

        def _load() -> SpectreConfig:
            if state["broken"]:
                raise ValueError("bad yaml")
            return _make_config(base_url="http://fixed:11438/v1")

        source = SpectreConfigSource(loader=_load, initial=_make_config())
        source.reload()
        state["broken"] = False

        assert source.reload().models["default"].base_url == "http://fixed:11438/v1"

    def test_load_failure_is_logged_once_not_per_message(self, caplog):
        def _explode() -> SpectreConfig:
            raise ValueError("bad yaml")

        source = SpectreConfigSource(loader=_explode, initial=_make_config())
        with caplog.at_level(logging.WARNING, logger="sr2_spectre.config_source"):
            for _ in range(5):
                source.reload()

        assert caplog.text.count("reload failed") == 1

    def test_mcp_servers_are_pinned(self):
        initial = _make_config(
            mcp_servers=[McpServerConfig(name="s1", type="stdio", command=["original"], args=[])]
        )
        loaded = _make_config(
            mcp_servers=[McpServerConfig(name="s1", type="stdio", command=["replacement"], args=[])]
        )
        source = SpectreConfigSource(loader=lambda: loaded, initial=initial)

        # A live stdio server owns a subprocess a reload cannot respawn.
        assert source.reload().agent.mcp_servers[0].command == ["original"]

    def test_agent_name_is_pinned(self):
        source = SpectreConfigSource(
            loader=lambda: _make_config(name="renamed"),
            initial=_make_config(name="miranda"),
        )
        assert source.reload().agent.name == "miranda"

    def test_discord_token_is_pinned_but_siblings_reload(self):
        source = SpectreConfigSource(
            loader=lambda: _make_config(
                discord=DiscordConfig(token="new-token", channels=[222])
            ),
            initial=_make_config(
                discord=DiscordConfig(token="gateway-token", channels=[111])
            ),
        )
        current = source.reload()

        # The nested pin must not freeze the whole discord block.
        assert current.discord.token == "gateway-token"
        assert current.discord.channels == [222]

    def test_pinned_change_warns_once(self, caplog):
        loaded = _make_config(name="renamed")
        source = SpectreConfigSource(loader=lambda: loaded, initial=_make_config(name="miranda"))

        with caplog.at_level(logging.WARNING, logger="sr2_spectre.config_source"):
            for _ in range(3):
                source.reload()

        assert caplog.text.count("cannot be applied") == 1
        assert "agent.name" in caplog.text

    def test_reload_logs_field_names_not_values(self, caplog):
        source = SpectreConfigSource(
            loader=lambda: _make_config(
                discord=DiscordConfig(token="super-secret", channels=[9])
            ),
            initial=_make_config(discord=DiscordConfig(token="super-secret")),
        )
        with caplog.at_level(logging.INFO, logger="sr2_spectre.config_source"):
            source.reload()

        assert "super-secret" not in caplog.text
        assert "discord" in caplog.text

    def test_static_source_never_changes(self):
        config = _make_config()
        source = SpectreConfigSource.static(config)
        assert source.reload() is config


# ---------------------------------------------------------------------------
# B. LiveLLM
# ---------------------------------------------------------------------------

class TestLiveLLM:
    def test_unchanged_config_is_not_retargeted(self):
        cfg = ModelConfig(model="m", base_url="http://a/v1")
        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            live = LiveLLM(cfg)
            MockLLM.reset_mock()
            assert live.retarget(ModelConfig(model="m", base_url="http://a/v1")) is False
            MockLLM.assert_not_called()

    def test_changed_endpoint_rebuilds_the_inner_callable(self):
        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            live = LiveLLM(ModelConfig(model="m", base_url="http://bad/v1"))
            MockLLM.reset_mock()

            assert live.retarget(ModelConfig(model="m", base_url="http://good/v1")) is True

        assert MockLLM.call_args.kwargs["base_url"] == "http://good/v1"

    def test_retarget_forwards_api_key_and_params(self):
        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            live = LiveLLM(ModelConfig(model="m", base_url="http://a/v1"))
            MockLLM.reset_mock()
            live.retarget(
                ModelConfig(
                    model="m2",
                    base_url="http://b/v1",
                    api_key="k",
                    params={"temperature": 0.5},
                )
            )

        kwargs = MockLLM.call_args.kwargs
        assert kwargs["model"] == "m2"
        assert kwargs["api_key"] == "k"
        assert kwargs["temperature"] == 0.5

    def test_model_property_reads_through_to_the_current_target(self):
        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock(model="openai/first")
            live = LiveLLM(ModelConfig(model="first", base_url="http://a/v1"))
            assert live.model == "openai/first"

            MockLLM.return_value = MagicMock(model="openai/second")
            live.retarget(ModelConfig(model="second", base_url="http://b/v1"))
            assert live.model == "openai/second"

    @pytest.mark.asyncio
    async def test_stream_delegates_to_the_current_target(self):
        from sr2.protocols.llm import CompletionRequest, StreamEvent

        async def _first(_request):
            yield StreamEvent(type="text", text="from-old")

        async def _second(_request):
            yield StreamEvent(type="text", text="from-new")

        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            MockLLM.return_value = MagicMock(stream=_first)
            live = LiveLLM(ModelConfig(model="m", base_url="http://a/v1"))

            request = CompletionRequest(messages=[])
            assert [e.text async for e in live.stream(request)] == ["from-old"]

            MockLLM.return_value = MagicMock(stream=_second)
            live.retarget(ModelConfig(model="m", base_url="http://b/v1"))

            assert [e.text async for e in live.stream(request)] == ["from-new"]


# ---------------------------------------------------------------------------
# C. Runtime.apply_config
# ---------------------------------------------------------------------------

@pytest.fixture
def runtime_factory():
    """Build Runtimes with the LLM construction stubbed out."""
    def _build(config: SpectreConfig):
        from sr2_spectre.runtime import Runtime
        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            return Runtime(config=config)
    return _build


class TestRuntimeApplyConfig:
    def test_identical_config_applies_nothing(self, runtime_factory):
        config = _make_config()
        runtime = runtime_factory(config)
        assert runtime.apply_config(_make_config()) == []

    def test_changed_endpoint_retargets_the_llm(self, runtime_factory):
        runtime = runtime_factory(_make_config(base_url="http://bad:11438/v1"))

        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            applied = runtime.apply_config(_make_config(base_url="http://good:11438/v1"))

        assert "models" in applied
        assert MockLLM.call_args.kwargs["base_url"] == "http://good:11438/v1"
        assert runtime.llm.model_config.base_url == "http://good:11438/v1"

    def test_config_in_force_is_updated(self, runtime_factory):
        runtime = runtime_factory(_make_config(base_url="http://bad/v1"))
        runtime.apply_config(_make_config(base_url="http://good/v1"))
        assert runtime.config.models["default"].base_url == "http://good/v1"

    def test_added_tool_is_registered(self, runtime_factory):
        runtime = runtime_factory(_make_config())
        assert "file_read" not in runtime.registry

        applied = runtime.apply_config(
            _make_config(
                tools=[
                    ToolConfig(
                        name="file_read",
                        class_path="sr2_spectre.tools.builtins.file_read.FileReadTool",
                    )
                ]
            )
        )

        assert "agent.tools" in applied
        assert "file_read" in runtime.registry

    def test_removed_tool_is_retired(self, runtime_factory):
        runtime = runtime_factory(
            _make_config(
                tools=[
                    ToolConfig(
                        name="file_read",
                        class_path="sr2_spectre.tools.builtins.file_read.FileReadTool",
                    )
                ]
            )
        )
        assert "file_read" in runtime.registry

        runtime.apply_config(_make_config(tools=[]))

        # Overwriting by name is not enough — the agent must stop being offered
        # a tool its config no longer grants it.
        assert "file_read" not in runtime.registry

    def test_mcp_registered_tools_survive_a_tool_reload(self, runtime_factory):
        runtime = runtime_factory(_make_config())
        runtime.registry.register(
            name="mcp_bridge_tool",
            description="from an MCP server",
            input_schema={},
            fn=lambda: None,
        )

        runtime.apply_config(
            _make_config(
                tools=[
                    ToolConfig(
                        name="file_read",
                        class_path="sr2_spectre.tools.builtins.file_read.FileReadTool",
                    )
                ]
            )
        )

        assert "mcp_bridge_tool" in runtime.registry

    def test_a_broken_tool_does_not_take_down_a_running_process(
        self, runtime_factory, caplog
    ):
        runtime = runtime_factory(_make_config())

        with caplog.at_level(logging.WARNING, logger="sr2_spectre.runtime"):
            runtime.apply_config(
                _make_config(
                    tools=[
                        ToolConfig(name="nope", class_path="does.not.Exist"),
                    ]
                )
            )

        assert "failed to register" in caplog.text

    def test_a_broken_tool_at_startup_still_fails_fast(self, runtime_factory):
        # Startup keeps its fail-fast contract: a config that cannot build is a
        # broken config, and the process should not come up pretending it works.
        with pytest.raises(Exception):
            runtime_factory(
                _make_config(tools=[ToolConfig(name="nope", class_path="does.not.Exist")])
            )

    def test_pipeline_change_is_reported(self, runtime_factory):
        runtime = runtime_factory(_make_config(system_text="old"))
        applied = runtime.apply_config(_make_config(system_text="new"))
        assert "pipeline" in applied


# ---------------------------------------------------------------------------
# D. Session config adoption
# ---------------------------------------------------------------------------

class TestSessionConfigAdoption:
    def test_session_adopts_the_new_config(self, runtime_factory):
        runtime = runtime_factory(_make_config(base_url="http://bad/v1"))
        session = runtime.new_session(frame_id="f1")

        runtime.apply_config(_make_config(base_url="http://good/v1"))

        assert session.config.models["default"].base_url == "http://good/v1"

    def test_pipeline_change_marks_the_session_stale(self, runtime_factory):
        runtime = runtime_factory(_make_config(system_text="old"))
        session = runtime.new_session(frame_id="f1")
        assert session._sr2_stale is False

        runtime.apply_config(_make_config(system_text="new"))

        assert session._sr2_stale is True

    def test_model_only_change_does_not_rebuild_sr2(self, runtime_factory):
        runtime = runtime_factory(_make_config(base_url="http://bad/v1"))
        session = runtime.new_session(frame_id="f1")
        original_sr2 = session.sr2

        runtime.apply_config(_make_config(base_url="http://good/v1"))

        # The LiveLLM swap already reached this session; rebuilding the SR2
        # would be needless churn.
        assert session._sr2_stale is False
        assert session.sr2 is original_sr2

    @pytest.mark.asyncio
    async def test_stale_sr2_is_rebuilt_on_the_next_turn(self, runtime_factory):
        runtime = runtime_factory(_make_config(system_text="old"))
        session = runtime.new_session(frame_id="f1")
        original_sr2 = session.sr2

        runtime.apply_config(_make_config(system_text="new"))

        rebuilt = MagicMock()
        rebuilt.seed_session = MagicMock()

        async def _turn(user_input):
            from sr2.protocols.llm import StreamEvent
            yield StreamEvent(type="text", text="hi")

        rebuilt.turn = _turn

        with patch.object(session, "_build_sr2", return_value=rebuilt) as build:
            [ev async for ev in session.stream_message("hello")]

        build.assert_called_once()
        assert session.sr2 is rebuilt
        assert session.sr2 is not original_sr2
        assert session._sr2_stale is False

    @pytest.mark.asyncio
    async def test_history_survives_an_sr2_rebuild(self, runtime_factory):
        runtime = runtime_factory(_make_config(system_text="old"))
        session = runtime.new_session(frame_id="f1")

        def _fake_sr2():
            sr2 = MagicMock()
            sr2.seed_session = MagicMock()

            async def _turn(user_input):
                from sr2.protocols.llm import StreamEvent
                yield StreamEvent(type="text", text="ok")

            sr2.turn = _turn
            return sr2

        session.sr2 = _fake_sr2()
        [ev async for ev in session.stream_message("first")]
        assert len(session.history) == 2

        runtime.apply_config(_make_config(system_text="new"))
        with patch.object(session, "_build_sr2", side_effect=lambda: _fake_sr2()):
            [ev async for ev in session.stream_message("second")]

        # Spectre owns history and re-seeds SR2 each turn, so a rebuilt SR2
        # must not cost the conversation its transcript.
        assert len(session.history) == 4
        assert session.history[0].content[0].text == "first"


# ---------------------------------------------------------------------------
# E. End to end — a file edit retargets a running agent
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = """\
agent:
  name: miranda
models:
  default:
    model: qwen3.8:27B
    base_url: {base_url}
pipeline:
  layers:
    - name: system
      target: system
      resolvers:
        - type: static
          config:
            text: You are helpful.
discord:
  token: gateway-token
  channels: [111]
"""


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "sr2home"
    home.mkdir()
    monkeypatch.setenv("SR2_HOME", str(home))

    path = tmp_path / "miranda.yaml"
    path.write_text(
        textwrap.dedent(_CONFIG_TEMPLATE.format(base_url="http://192.168.50.177:11438/v1"))
    )
    return path


class TestEndToEndEndpointFix:
    def test_correcting_base_url_on_disk_retargets_a_running_agent(self, config_file):
        """The failure this whole change exists for.

        A typo'd host in the agent file makes every reply fail with a
        connection error. Correcting the file must be enough — the next message
        should reach the fixed endpoint without restarting the bot.
        """
        from sr2_spectre.agent import Agent
        from sr2_spectre.cli import build_spectre_config_source, resolve_config

        initial = resolve_config(config_file, cwd=config_file.parent, env={})
        source = build_spectre_config_source(config_file, config_file.parent, initial)

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            agent = Agent(config=initial)
        assert agent._runtime.llm.model_config.base_url == "http://192.168.50.177:11438/v1"

        config_file.write_text(
            textwrap.dedent(_CONFIG_TEMPLATE.format(base_url="http://192.168.50.117:11438/v1"))
        )

        with patch("sr2_spectre.live_llm.LiteLLMCallable") as MockLLM:
            applied = agent.apply_config(source.reload())

        assert "models" in applied
        assert MockLLM.call_args.kwargs["base_url"] == "http://192.168.50.117:11438/v1"

    def test_open_conversations_follow_the_new_endpoint(self, config_file):
        from sr2_spectre.agent import Agent
        from sr2_spectre.cli import build_spectre_config_source, resolve_config

        initial = resolve_config(config_file, cwd=config_file.parent, env={})
        source = build_spectre_config_source(config_file, config_file.parent, initial)

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            agent = Agent(config=initial)

        # The session opened before the fix must not be stranded on the old
        # endpoint — that is what the LiveLLM indirection buys.
        session_llm = agent._runtime._sessions and next(iter(agent._runtime._sessions))._llm

        config_file.write_text(
            textwrap.dedent(_CONFIG_TEMPLATE.format(base_url="http://192.168.50.117:11438/v1"))
        )
        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            agent.apply_config(source.reload())

        assert session_llm is agent._runtime.llm
        assert session_llm.model_config.base_url == "http://192.168.50.117:11438/v1"

    def test_a_broken_edit_leaves_the_bot_on_the_last_good_config(self, config_file):
        from sr2_spectre.cli import build_spectre_config_source, resolve_config

        initial = resolve_config(config_file, cwd=config_file.parent, env={})
        source = build_spectre_config_source(config_file, config_file.parent, initial)

        config_file.write_text("models:\n  default:\n    model: [unclosed\n")

        assert source.reload().models["default"].model == "qwen3.8:27B"


# ---------------------------------------------------------------------------
# F. Discord wiring — one reload per message feeds both halves
# ---------------------------------------------------------------------------

class TestDiscordViewWiring:
    def test_view_reload_refreshes_the_whole_config(self):
        configs = [
            _make_config(
                base_url="http://old/v1", discord=DiscordConfig(token="t", channels=[111])
            ),
            _make_config(
                base_url="http://new/v1", discord=DiscordConfig(token="t", channels=[222])
            ),
        ]
        from sr2_spectre.interfaces.discord.config_source import DiscordConfigView

        source = SpectreConfigSource(loader=lambda: configs[1], initial=configs[0])
        view = DiscordConfigView(source)

        assert view.reload().channels == [222]
        # The same single read also refreshed the agent's half of the config.
        assert view.source.current.models["default"].base_url == "http://new/v1"

    def test_adapter_accepts_the_view(self):
        from sr2_spectre.interfaces.discord.adapter import DiscordBotAdapter
        from sr2_spectre.interfaces.discord.config_source import DiscordConfigView

        config = _make_config(discord=DiscordConfig(token="t", channels=[111]))
        view = DiscordConfigView(SpectreConfigSource.static(config))
        adapter = DiscordBotAdapter(view)

        assert adapter.config.channels == [111]

    def test_interface_pushes_the_reloaded_config_into_the_agent(self):
        from sr2_spectre.interfaces.discord.config_source import DiscordConfigView
        from sr2_spectre.interfaces.discord.interface import DiscordInterface

        configs = [
            _make_config(base_url="http://old/v1", discord=DiscordConfig(token="t")),
            _make_config(base_url="http://new/v1", discord=DiscordConfig(token="t")),
        ]
        source = SpectreConfigSource(loader=lambda: configs[1], initial=configs[0])
        view = DiscordConfigView(source)

        interface = DiscordInterface(config_source=view)
        interface._agent = MagicMock()

        view.reload()
        interface._apply_agent_config()

        applied = interface._agent.apply_config.call_args.args[0]
        assert applied.models["default"].base_url == "http://new/v1"

    def test_plain_discord_source_leaves_the_agent_alone(self):
        from sr2_spectre.interfaces.discord.config_source import DiscordConfigSource
        from sr2_spectre.interfaces.discord.interface import DiscordInterface

        interface = DiscordInterface(
            config_source=DiscordConfigSource.static(DiscordConfig(token="t"))
        )
        interface._agent = MagicMock()

        interface._apply_agent_config()

        # Nothing but the Discord slice was ever loaded, so there is no agent
        # config to apply.
        interface._agent.apply_config.assert_not_called()
