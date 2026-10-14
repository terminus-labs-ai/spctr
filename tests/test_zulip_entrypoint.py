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


class TestStartupAcceptsMappedAreas:
    """spc-120: the temporary nonempty-area startup rejection (spc-119/125
    intermediate contract, enforced by the interface's own startup gate per
    the accepted J1B proof) is lifted — a mapped area starts instead of
    failing startup, so routing never has to lose its area. The file-level
    resolution below stays green at both cuts (the gate lived in the
    interface, not the resolver); the direct startup-validator row goes red
    until the routing release ships, and its routing-aware companion
    (``test_mapped_area_channel_resolves_and_admits_through_entrypoint_
    config``) keeps the pair from being satisfiable by a gate lift alone. E
    owns replacing the corresponding real-CLI refusal in
    tests/test_zulip_chat_e2e.py; an interface-level start() assertion is NOT
    made here on purpose — spc-119/125 deliberately froze the Zulip transport
    without a network-mock helper, and a unit-level real start() would dial
    the network and swallow arbitrary transport failures. The real startup
    proof is E's journey."""

    def test_mapped_area_channel_survives_resolution(self, isolated):
        cwd, env, pos = isolated
        _write_yaml(pos, {**VALID_TAIL, "zulip": {
            "zuliprc": ZULIPRC,
            "allowed_user_ids": [DIEGO_ID],
            "channels": {NO_AREA_CHANNEL: "", 44: "obsidian"},
        }})
        zulip = resolve_config(pos, cwd=cwd, env=env).zulip
        assert zulip is not None
        assert zulip.channels == {NO_AREA_CHANNEL: "", 44: "obsidian"}

    def test_mapped_area_interface_startup_not_rejected(self):
        """The startup validator the Zulip front door calls first (spc-119
        convention, frozen by the accepted J1B proof) must accept a mapped
        area after this cut. Asserted directly — no ``start()`` and no
        network: a unit-level real ``start()`` would dial a synthetic server,
        swallow arbitrary transport errors, and leak the client session,
        and the contract forbids replicating the interface-journey harness
        here. Today this is red because the temporary gate raises and names
        the mapped channel; it turns green when the lift lands."""
        from sr2_spectre.interfaces.zulip import validate_no_area_only_routing

        cfg = ZulipConfig(
            zuliprc=ZULIPRC,
            allowed_user_ids=[DIEGO_ID],
            channels={NO_AREA_CHANNEL: "", 44: "obsidian"},
        )
        assert validate_no_area_only_routing(cfg) is None

    def test_mapped_area_channel_resolves_and_admits(self, isolated):
        """Routing-aware companion to the two gate-lift rows (B3 item 2):
        the mapped-area config that resolves through the entry point must
        ALSO route (area carried, not lost) and admit an allowed human —
        deleting the gate while the facade still admits no-area channels
        only goes red here. Today red at the routing import/step."""
        from sr2_spectre.cli import resolve_config
        from sr2_spectre.interfaces.zulip import (
            is_configured_no_area_channel,
            resolve_channel_area,
        )

        cwd, env, pos = isolated
        _write_yaml(pos, {**VALID_TAIL, "zulip": {
            "zuliprc": ZULIPRC,
            "allowed_user_ids": [DIEGO_ID],
            "channels": {NO_AREA_CHANNEL: "", 44: "obsidian"},
        }})
        zulip = resolve_config(pos, cwd=cwd, env={}).zulip
        assert zulip is not None
        assert resolve_channel_area(channel_id=44, config=zulip) == "obsidian"
        assert is_configured_no_area_channel(
            channel_id=44,
            sender_id=DIEGO_ID,
            sender_is_bot=False,
            config=zulip,
        )

