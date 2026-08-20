"""Tests for the --agent <name> CLI shorthand (spc-25).

The --agent flag resolves <name> to <agents_dir>/<name>.yaml, where
agents_dir defaults to ~/.sr2/agents/.

Acceptance criteria:
1. --agent edi resolves to ~/.sr2/agents/edi.yaml
2. --agents-dir override changes the base directory
3. Positonal config still works without --agent
4. --agent with a non-existent file raises FileNotFoundError
5. --agent takes precedence over positional config (or errors — we error for clarity)
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sr2_spectre.cli import _parse_args, resolve_agent_config_path


# ---------------------------------------------------------------------------
# 1: resolve_agent_config_path — core resolution logic
# ---------------------------------------------------------------------------

def _fake_interface_cls(instance):
    """Class whose instantiation returns ``instance`` (for _load_interface patches)."""
    return type("_FakeInterfaceCls", (), {"__new__": staticmethod(lambda cls: instance)})


class TestResolveAgentConfigPath:
    """Tests for the pure-function agent path resolution."""

    def test_default_agents_dir(self, tmp_path: Path) -> None:
        """--agent name resolves to ~/.sr2/agents/<name>.yaml by default."""
        path = resolve_agent_config_path("edi")
        assert path.name == "edi.yaml"
        assert str(path).endswith(".sr2/agents/edi.yaml")

    def test_custom_agents_dir(self, tmp_path: Path) -> None:
        """--agents-dir override changes the base directory."""
        custom_dir = tmp_path / "my_agents"
        custom_dir.mkdir()
        path = resolve_agent_config_path("edi", agents_dir=custom_dir)
        assert path == custom_dir / "edi.yaml"

    def test_expands_tilde(self, tmp_path: Path) -> None:
        """Tilde in agents_dir is expanded."""
        path = resolve_agent_config_path("edi", agents_dir=Path("~/.my-agents"))
        assert not str(path).startswith("~")

    def test_name_with_hyphens(self, tmp_path: Path) -> None:
        """Agent names with hyphens are supported."""
        path = resolve_agent_config_path("edi-zorah", agents_dir=Path("/tmp"))
        assert path.name == "edi-zorah.yaml"

    def test_name_with_underscores(self, tmp_path: Path) -> None:
        """Agent names with underscores are supported."""
        path = resolve_agent_config_path("edi_zorah", agents_dir=Path("/tmp"))
        assert path.name == "edi_zorah.yaml"

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        """The returned path is absolute."""
        path = resolve_agent_config_path("edi", agents_dir=tmp_path)
        assert path.is_absolute()


# ---------------------------------------------------------------------------
# 2: CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseAgentFlag:
    """Tests for argparse integration."""

    def test_agent_flag_parsed(self) -> None:
        """--agent edi is parsed correctly."""
        args = _parse_args(["--agent", "edi", "config.yaml"])
        assert args.agent == "edi"

    def test_agent_flag_defaults_to_none(self) -> None:
        """Without --agent, args.agent is None."""
        args = _parse_args(["config.yaml"])
        assert args.agent is None

    def test_agents_dir_flag_parsed(self, tmp_path: Path) -> None:
        """--agents-dir is parsed correctly."""
        args = _parse_args([
            "--agent", "edi",
            "--agents-dir", str(tmp_path / "custom"),
            "config.yaml",
        ])
        assert args.agents_dir == str(tmp_path / "custom")

    def test_agents_dir_defaults_to_none(self) -> None:
        """Without --agents-dir, args.agents_dir is None."""
        args = _parse_args(["--agent", "edi", "config.yaml"])
        assert args.agents_dir is None


# ---------------------------------------------------------------------------
# 3: run_async integration — --agent resolves the config path
# ---------------------------------------------------------------------------

class TestRunAsyncWithAgent:
    """Integration tests for --agent in the full run path."""

    @pytest.mark.asyncio
    async def test_agent_flag_resolves_config_path(self, tmp_path: Path) -> None:
        """When --agent is set, resolve_config receives the resolved YAML path."""
        from unittest.mock import AsyncMock, MagicMock
        from sr2_spectre.core import TurnResult

        # Create the agent config file
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        agent_yaml = agents_dir / "edi.yaml"
        agent_yaml.write_text(
            textwrap.dedent("""\
                agent:
                  name: edi
                models:
                  default:
                    model: test
                    base_url: http://localhost:11434/v1
                pipeline:
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
            """)
        )

        mock_config = MagicMock()
        mock_config.agent.name = "edi"
        mock_config.models = {"default": MagicMock(model="test", base_url=None)}

        captured_config_path: list = []

        def capture_resolve_config(path, **kwargs):
            captured_config_path.append(path)
            return mock_config

        mock_interface = MagicMock()
        mock_interface.start = AsyncMock()
        mock_interface.stop = AsyncMock()
        mock_interface.run = AsyncMock()

        mock_agent = AsyncMock()

        with (
            patch("sr2_spectre.cli.resolve_config", side_effect=capture_resolve_config),
            patch("sr2_spectre.cli._configure_logging"),
            patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(mock_interface)),
            patch("sr2_spectre.cli.Agent", return_value=mock_agent),
        ):
            await run_async([
                "--agent", "edi",
                "--agents-dir", str(agents_dir),
                "placeholder.yaml",
            ])

        assert len(captured_config_path) == 1
        assert captured_config_path[0] == agent_yaml

    @pytest.mark.asyncio
    async def test_agent_flag_with_default_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--agent without --agents-dir uses ~/.sr2/agents/."""
        from unittest.mock import AsyncMock, MagicMock
        from sr2_spectre.core import TurnResult

        # Fake home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        agents_dir = fake_home / ".sr2" / "agents"
        agents_dir.mkdir(parents=True)
        agent_yaml = agents_dir / "liara.yaml"
        agent_yaml.write_text(
            textwrap.dedent("""\
                agent:
                  name: liara
                models:
                  default:
                    model: test
                    base_url: http://localhost:11434/v1
                pipeline:
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
            """)
        )

        mock_config = MagicMock()
        mock_config.agent.name = "liara"
        mock_config.models = {"default": MagicMock(model="test", base_url=None)}

        captured_config_path: list = []

        def capture_resolve_config(path, **kwargs):
            captured_config_path.append(path)
            return mock_config

        mock_interface = MagicMock()
        mock_interface.start = AsyncMock()
        mock_interface.stop = AsyncMock()
        mock_interface.run = AsyncMock()

        mock_agent = AsyncMock()

        with (
            patch("sr2_spectre.cli.resolve_config", side_effect=capture_resolve_config),
            patch("sr2_spectre.cli._configure_logging"),
            patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(mock_interface)),
            patch("sr2_spectre.cli.Agent", return_value=mock_agent),
            patch("pathlib.Path.home", return_value=fake_home),
        ):
            await run_async([
                "--agent", "liara",
                "placeholder.yaml",
            ])

        assert len(captured_config_path) == 1
        assert captured_config_path[0] == agent_yaml

    @pytest.mark.asyncio
    async def test_no_agent_flag_uses_positional_config(self, tmp_path: Path) -> None:
        """Without --agent, the positional config argument is used directly."""
        from unittest.mock import AsyncMock, MagicMock

        config_yaml = tmp_path / "my_config.yaml"
        config_yaml.write_text(
            textwrap.dedent("""\
                agent:
                  name: positional
                models:
                  default:
                    model: test
                    base_url: http://localhost:11434/v1
                pipeline:
                  layers:
                    - name: system
                      target: system
                      resolvers:
                        - type: static
                          config:
                            text: You are helpful.
            """)
        )

        mock_config = MagicMock()
        mock_config.agent.name = "positional"
        mock_config.models = {"default": MagicMock(model="test", base_url=None)}

        captured_config_path: list = []

        def capture_resolve_config(path, **kwargs):
            captured_config_path.append(path)
            return mock_config

        mock_interface = MagicMock()
        mock_interface.start = AsyncMock()
        mock_interface.stop = AsyncMock()
        mock_interface.run = AsyncMock()

        mock_agent = AsyncMock()

        with (
            patch("sr2_spectre.cli.resolve_config", side_effect=capture_resolve_config),
            patch("sr2_spectre.cli._configure_logging"),
            patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(mock_interface)),
            patch("sr2_spectre.cli.Agent", return_value=mock_agent),
        ):
            await run_async([str(config_yaml)])

        assert len(captured_config_path) == 1
        # Should use the positional path directly, not resolve through agents dir
        assert Path(captured_config_path[0]) == config_yaml


# ---------------------------------------------------------------------------
# 4: run_async exported for tests
# ---------------------------------------------------------------------------

from sr2_spectre.cli import run_async
