"""Tests for the Interface protocol and CLI interface flag.

Covers:
- Interface protocol has name, start, stop, run
- Both builtin implementations satisfy Interface
- --interface flag works
- Dead code removed: PluginConfig, HeartbeatConfig, OutputPlugin, PluginRegistry
"""
from __future__ import annotations

import pytest

from spctr.cli import _parse_args


# ---------------------------------------------------------------------------
# Interface protocol
# ---------------------------------------------------------------------------

def test_interface_protocol_has_required_attributes() -> None:
    """Interface Protocol must define name, start, stop, run."""
    from spctr.interfaces import Interface

    protocol_attrs = Interface.__protocol_attrs__
    assert "name" in protocol_attrs
    assert "start" in protocol_attrs
    assert "stop" in protocol_attrs
    assert "run" in protocol_attrs


def test_single_shot_satisfies_interface() -> None:
    """SingleShotInterface must satisfy the Interface protocol at runtime."""
    from spctr.interfaces import Interface
    from spctr.interfaces.single_shot import SingleShotInterface

    instance = SingleShotInterface()
    assert isinstance(instance, Interface)


def test_tui_satisfies_interface() -> None:
    """TUIInterface must satisfy the Interface protocol at runtime."""
    from spctr.interfaces import Interface
    from spctr.interfaces.tui import TUIInterface

    instance = TUIInterface()
    assert isinstance(instance, Interface)


def test_interface_exports_only_interface() -> None:
    """spctr.interfaces.__all__ must only export Interface."""
    import spctr.interfaces
    assert spctr.interfaces.__all__ == ["Interface"]


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
        from spctr.config import PluginConfig  # noqa: F401


def test_heartbeat_config_removed() -> None:
    """HeartbeatConfig must no longer be importable from config."""
    with pytest.raises(ImportError):
        from spctr.config import HeartbeatConfig  # noqa: F401


def test_spctr_config_no_plugins_field() -> None:
    """SpctrConfig must not have a plugins field."""
    from spctr.config import SpctrConfig
    assert "plugins" not in SpctrConfig.model_fields


def test_spctr_config_no_heartbeat_field() -> None:
    """SpctrConfig must not have a heartbeat field."""
    from spctr.config import SpctrConfig
    assert "heartbeat" not in SpctrConfig.model_fields


def test_no_plugins_directory() -> None:
    """src/spctr/plugins/ directory must not exist."""
    import spctr
    import pathlib
    pkg_root = pathlib.Path(spctr.__file__).parent
    assert not (pkg_root / "plugins").exists()


def test_no_output_plugin_or_registry() -> None:
    """OutputPlugin and PluginRegistry must no longer exist."""
    with pytest.raises(ImportError):
        from spctr.interfaces import OutputPlugin  # noqa: F401
    with pytest.raises(ImportError):
        from spctr.interfaces import PluginRegistry  # noqa: F401


# ---------------------------------------------------------------------------
# _load_interface loads correct classes
# ---------------------------------------------------------------------------

def test_load_interface_tui() -> None:
    """_load_interface returns the interface CLASS; instantiate explicitly."""
    from spctr.cli import _load_interface
    from spctr.interfaces.tui import TUIInterface

    cls = _load_interface("tui")
    assert cls is TUIInterface
