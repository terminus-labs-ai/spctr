"""Unit cage — spc-125 (Connect Miranda's CLI to safe no-area Zulip chat).

A proves ONLY the pure CLI/config entrypoint behaviour:

- the actual argument parser accepts ``--interface zulip`` (frozen J1B form
  ``python -m sr2_spectre.cli <isolated-config> --interface zulip --log-file …``),
- the interface name ``zulip`` resolves through the CLI's own loading seam to
  the ``ZulipInterface`` class,
- startup config resolution retains a validated ``zulip`` block on
  ``SpectreConfig`` and fails explicitly when that block is missing or
  invalid — never silently dropping Zulip routing.

Deliberately NOT here (E owns in tests/test_zulip_chat_e2e.py against the
real CLI/Runtime/Session/SR2): reply delivery, topic-history isolation, the
populated-topic refusal, startup replay, and the trigger/bot/channel filters.
Configuration/credential validation itself is accepted spc-123 proof and HTTP
transport is accepted spc-124 proof — neither is re-caged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sr2_spectre.cli import _load_interface, _parse_args, resolve_config
from sr2_spectre.interfaces.zulip import ZulipConfig

DIEGO_ID = 8                                 # Diego (spec Dependencies)
ZULIPRC = "/tmp/spc-125-unused.zuliprc"      # never read: no credential I/O here
NO_AREA_CHANNEL = 42

# Minimal accepted agent/models/pipeline shape; tests only pass explicit cwd/env.
VALID_TAIL: dict[str, Any] = {
    "agent": {"name": "miranda"},
    "models": {"default": {"model": "synthetic-model"}},
    "pipeline": {"layers": []},
}


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Fresh SR2_HOME/cwd and no inherited memory DSN or ambient area."""
    sr2_home, cwd = tmp_path / "sr2home", tmp_path / "project"
    sr2_home.mkdir(), cwd.mkdir()
    monkeypatch.setenv("SR2_HOME", str(sr2_home))
    monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
    monkeypatch.delenv("SR2_AREA", raising=False)
    pos = tmp_path / "config.yaml"
    return cwd, {"SR2_HOME": str(sr2_home)}, pos


class TestCliAcceptsZulip:
    def test_zulip_interface_parses(self):
        """AC1 entry point: --interface zulip is an accepted CLI choice."""
        args = _parse_args(["cfg.yaml", "--interface", "zulip"])
        assert args.interface == "zulip"

    def test_zulip_resolves_to_zulip_interface_class(self):
        """AC1: name resolution reaches an interface whose name is 'zulip'."""
        assert _load_interface("zulip").name == "zulip"


class TestConfigRetainsZulipBlock:
    def test_valid_block_survives_resolution(self, isolated):
        cwd, env, pos = isolated
        _write_yaml(pos, {**VALID_TAIL, "zulip": {
            "zuliprc": ZULIPRC,
            "allowed_user_ids": [DIEGO_ID],
            "channels": {NO_AREA_CHANNEL: ""},
        }})
        zulip = resolve_config(pos, cwd=cwd, env=env).zulip
        assert isinstance(zulip, ZulipConfig)
        assert zulip.channels == {NO_AREA_CHANNEL: ""}
        assert zulip.allowed_user_ids == [DIEGO_ID]

    def test_absent_block_is_none(self, isolated):
        cwd, env, pos = isolated
        _write_yaml(pos, dict(VALID_TAIL))
        assert resolve_config(pos, cwd=cwd, env=env).zulip is None

    @pytest.mark.parametrize("broken", [
        {"zuliprc": ZULIPRC, "allowed_user_ids": ["8"], "channels": {}},
        {"zuliprc": ZULIPRC, "allowed_user_ids": [DIEGO_ID], "channels": {"42": ""}},
        {"zuliprc": ZULIPRC, "allowed_user_ids": [DIEGO_ID], "channels": {NO_AREA_CHANNEL: None}},
        {"allowed_user_ids": [DIEGO_ID], "channels": {NO_AREA_CHANNEL: ""}},
    ])
    def test_invalid_block_fails_resolution(self, isolated, broken):
        """AC16: an invalid Zulip block fails startup resolution explicitly;
        it never resolves to a config that silently drops or weakens the
        block (e.g. area None instead of the explicit "" / missing ID)."""
        cwd, env, pos = isolated
        _write_yaml(pos, {**VALID_TAIL, "zulip": broken})
        with pytest.raises(Exception):
            resolve_config(pos, cwd=cwd, env=env)
