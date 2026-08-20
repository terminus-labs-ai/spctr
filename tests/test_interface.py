"""Tests for the Interface protocol and CLI interface flag.

Covers:
- Interface protocol has name, start, stop, run
- Both builtin implementations satisfy Interface
- --interface flag works
- Dead code removed: PluginConfig, HeartbeatConfig, OutputPlugin, PluginRegistry
"""
from __future__ import annotations

import pytest

from sr2_spectre.cli import _parse_args


# ---------------------------------------------------------------------------
# Interface protocol
# ---------------------------------------------------------------------------

def test_interface_protocol_has_required_attributes() -> None:
    """Interface Protocol must define name, start, stop, run."""
    from sr2_spectre.interfaces import Interface

    protocol_attrs = Interface.__protocol_attrs__
    assert "name" in protocol_attrs
    assert "start" in protocol_attrs
    assert "stop" in protocol_attrs
    assert "run" in protocol_attrs


def test_single_shot_satisfies_interface() -> None:
    """SingleShotInterface must satisfy the Interface protocol at runtime."""
    from sr2_spectre.interfaces import Interface
    from sr2_spectre.interfaces.single_shot import SingleShotInterface

    instance = SingleShotInterface()
    assert isinstance(instance, Interface)


def test_tui_satisfies_interface() -> None:
    """TUIInterface must satisfy the Interface protocol at runtime."""
    from sr2_spectre.interfaces import Interface
    from sr2_spectre.interfaces.tui import TUIInterface

    instance = TUIInterface()
    assert isinstance(instance, Interface)


def test_interface_exports_only_interface() -> None:
    """sr2_spectre.interfaces.__all__ must only export Interface."""
    import sr2_spectre.interfaces
    assert sr2_spectre.interfaces.__all__ == ["Interface"]


# ---------------------------------------------------------------------------
# CLI --interface flag
# ---------------------------------------------------------------------------

def test_parse_args_interface_defaults_to_repl() -> None:
    """REPL is the default interactive interface (obsidian-hp8c)."""
    args = _parse_args(["config.yaml", "hello"])
    assert args.interface == "repl"


def test_parse_args_interface_flag() -> None:
    args = _parse_args(["config.yaml", "--interface", "tui"])
    assert args.interface == "tui"


# ---------------------------------------------------------------------------
# Dead code removal
# ---------------------------------------------------------------------------

def test_plugin_config_removed() -> None:
    """PluginConfig must no longer be importable from config."""
    with pytest.raises(ImportError):
        from sr2_spectre.config import PluginConfig  # noqa: F401


def test_heartbeat_config_removed() -> None:
    """HeartbeatConfig must no longer be importable from config."""
    with pytest.raises(ImportError):
        from sr2_spectre.config import HeartbeatConfig  # noqa: F401


def test_spectre_config_no_plugins_field() -> None:
    """SpectreConfig must not have a plugins field."""
    from sr2_spectre.config import SpectreConfig
    assert "plugins" not in SpectreConfig.model_fields


def test_spectre_config_no_heartbeat_field() -> None:
    """SpectreConfig must not have a heartbeat field."""
    from sr2_spectre.config import SpectreConfig
    assert "heartbeat" not in SpectreConfig.model_fields


def test_no_plugins_directory() -> None:
    """src/sr2_spectre/plugins/ directory must not exist."""
    import sr2_spectre
    import pathlib
    pkg_root = pathlib.Path(sr2_spectre.__file__).parent
    assert not (pkg_root / "plugins").exists()


def test_no_output_plugin_or_registry() -> None:
    """OutputPlugin and PluginRegistry must no longer exist."""
    with pytest.raises(ImportError):
        from sr2_spectre.interfaces import OutputPlugin  # noqa: F401
    with pytest.raises(ImportError):
        from sr2_spectre.interfaces import PluginRegistry  # noqa: F401


# ---------------------------------------------------------------------------
# _load_interface loads correct classes
# ---------------------------------------------------------------------------

def test_load_interface_tui() -> None:
    """_load_interface returns the interface CLASS; instantiate explicitly."""
    from sr2_spectre.cli import _load_interface
    from sr2_spectre.interfaces.tui import TUIInterface

    cls = _load_interface("tui")
    assert cls is TUIInterface
