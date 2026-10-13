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
    interface, not the resolver); the interface-level startup assertion
    goes red until the routing release ships. E owns replacing the
    corresponding real-CLI refusal in tests/test_zulip_chat_e2e.py."""

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

    async def test_mapped_area_interface_startup_not_rejected(self, tmp_path):
        """Interface-level startup with a mapped area must not raise the
        temporary "area routing not supported" gate. After the lift, either
        start() succeeds against the synthetic zuliprc or it fails later
        (unreachable synthetic server / no network) — a failure mode that is
        not the temporary routing gate. No model work is possible either
        way: no event is ever polled here, and a model turn requires an
        admitted trigger."""
        from sr2_spectre.interfaces.zulip import ZulipInterface

        zuliprc = tmp_path / "synthetic.zuliprc"
        zuliprc.write_text(
            "[zulip]\nemail = bot@example.invalid\n"
            "key = synthetic-key\nsite = https://zulip.example.invalid\n"
        )
        cfg = ZulipConfig(
            zuliprc=str(zuliprc),
            allowed_user_ids=[DIEGO_ID],
            channels={NO_AREA_CHANNEL: "", 44: "obsidian"},
        )
        try:
            await ZulipInterface(config=cfg).start(agent=None)
        except ValueError as exc:
            # The temporary restriction is exactly what must no longer fire;
            # a ValueError whose message names the area-routing restriction
            # is that gate, still in force → red.
            assert not (
                "area" in str(exc).lower() and "routing" in str(exc).lower()
            ), (
                "startup still rejects a mapped area with the temporary "
                f"no-area-only gate: {exc}"
            )
            raise  # an unrelated ValueError is still a startup failure
        except Exception:
            pass  # credential/network/transport failure — not the gate

