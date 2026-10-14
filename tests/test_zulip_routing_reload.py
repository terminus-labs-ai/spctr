"""Unit cage — spc-120 (Add Zulip area routing and live config reload).

Public seams this cage pins (names declared so C's first commit matches; no
signature or class discovery happens in this file):

- ``resolve_channel_area(*, channel_id, config)`` — the pure channel→area
  routing decision in ``sr2_spectre.interfaces.zulip.config`` (mirrors the
  accepted ``is_configured_no_area_channel`` keyword style): a configured
  channel gets its mapped area, a configured empty string stays the explicit
  no-area, and an unconfigured channel answers ``None`` — a positive
  non-answer, even when the process is planted with a working-directory name
  and an ``SR2_AREA`` trap (AC2, spec Interfaces: "No implicit channel-name
  or launch-directory fallback").
- ``ZulipConfigView`` — the view class in
  ``sr2_spectre.interfaces.zulip.config_source`` (mirrors the accepted
  ``DiscordConfigView``): one whole-config ``reload()`` feeds the routing
  slice AND ``Agent.apply_config``'s same single read; last-good retained on
  a malformed reload (AC6, spec FR6).
- startup resolution of the whole-config source for the Zulip entry point.

Deliberately NOT here: the live interface/transport/Session orchestration,
resolver-context and compiled-request proofs, and the per-inbound-message
invocation of the reload (whether C's adapter actually calls it on each
message). E owns those in ``tests/test_zulip_routing_reload_e2e.py`` through
the real CLI (``test_area_routing_and_isolation``,
``test_live_reload_and_last_good``). Accepted spc-123 credential/schema proof
and spc-124 transport proof are not re-caged. No HTTP, no model, no
subprocess, no sockets in this file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from sr2_spectre.config import SpectreConfig
from sr2_spectre.config_source import SpectreConfigSource
from sr2_spectre.interfaces.zulip import ZulipConfig, is_configured_no_area_channel

DIEGO_ID = 8                     # Diego (spec Dependencies)
NO_AREA_CHANNEL = 42             # explicit no-area channel (area "")
ALPHA_CHANNEL = 44               # mapped channel (area AREA_OBSIDIAN)
BETA_CHANNEL = 45                # mapped channel, area BETA_AREA
UNMAPPED_CHANNEL = 43            # no entry at all — must never be routed

ZULIPRC = "/tmp/spc-120-unused.zuliprc"     # never read: no credential I/O

AREA_OBSIDIAN = "obsidian"


# ---------------------------------------------------------------------------
# Fixtures / helpers — isolated resolution, no inherited area or DSN.
# ---------------------------------------------------------------------------

VALID_TAIL: dict[str, Any] = {
    "agent": {"name": "miranda"},
    "models": {"default": {"model": "synthetic-model"}},
    "pipeline": {"layers": []},
}


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data))


def _zulip_block(**over: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "zuliprc": ZULIPRC,
        "allowed_user_ids": [DIEGO_ID],
        "channels": {
            NO_AREA_CHANNEL: "",
            ALPHA_CHANNEL: AREA_OBSIDIAN,
            BETA_CHANNEL: AREA_OBSIDIAN,
        },
    }
    block.update(over)
    return block


def _spectre(zulip: ZulipConfig | None, model: str = "synthetic-model") -> SpectreConfig:
    return SpectreConfig(
        agent={"name": "miranda"},
        models={"default": {"model": model}},
        pipeline={"layers": []},
        zulip=zulip,
    )


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Fresh SR2_HOME/cwd; no inherited memory DSN or ambient area."""
    sr2_home, cwd = tmp_path / "sr2home", tmp_path / "project"
    sr2_home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("SR2_HOME", str(sr2_home))
    monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
    monkeypatch.delenv("SR2_AREA", raising=False)
    return cwd, sr2_home


# Seam imports (``resolve_channel_area`` in ``interfaces/zulip/config.py``,
# ``ZulipConfigView`` in ``interfaces/zulip.config_source.py``) are LAZY inside
# each test: an unshipped seam fails only its own row, never the whole file
# via one collection error. Both keep the accepted keyword style — called
# with ``channel_id=``/``config=`` only; no discovery or shape probing.


# ---------------------------------------------------------------------------
# AC2 — pure channel→area routing, no implicit fallback
# ---------------------------------------------------------------------------


class TestAreaRoutingDecision:
    def test_mapped_channel_reaches_resolvers_with_its_area(self):
        from sr2_spectre.interfaces.zulip.config import resolve_channel_area

        route = resolve_channel_area
        cfg = ZulipConfig(
            zuliprc=ZULIPRC,
            allowed_user_ids=[DIEGO_ID],
            channels={ALPHA_CHANNEL: AREA_OBSIDIAN},
        )
        assert route(channel_id=ALPHA_CHANNEL, config=cfg) == AREA_OBSIDIAN

    def test_explicit_no_area_channel_stays_empty_string(self):
        """RunContext.area="" is the explicit no-area contract — the routing
        answer must be "", never None (None permits consumer fallback)."""
        from sr2_spectre.interfaces.zulip.config import resolve_channel_area

        route = resolve_channel_area
        cfg = ZulipConfig(
            zuliprc=ZULIPRC,
            allowed_user_ids=[DIEGO_ID],
            channels={NO_AREA_CHANNEL: ""},
        )
        area = route(channel_id=NO_AREA_CHANNEL, config=cfg)
        assert area == ""
        assert area is not None

    def test_unmapped_channel_is_not_routed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """AC2 head pin: an explicitly unmapped channel answers None — no
        area at all — even with the exact accidental-inheritance inputs
        planted as traps: the process working directory is a real directory
        named like a vault area and SR2_AREA is exported. A routing decision
        that consults either (or raises instead of answering) fails here;
        downstream, RunContext.area=None is precisely the value that permits
        a fallback, so the non-answer must be produced here, explicitly."""
        trap_area = "cwd-trap-area-spc120"
        trap_dir = tmp_path / trap_area
        trap_dir.mkdir()
        monkeypatch.chdir(trap_dir)
        monkeypatch.setenv("SR2_AREA", trap_area)

        from sr2_spectre.interfaces.zulip.config import resolve_channel_area

        route = resolve_channel_area
        cfg = ZulipConfig(
            zuliprc=ZULIPRC,
            allowed_user_ids=[DIEGO_ID],
            channels={NO_AREA_CHANNEL: ""},
        )
        area = route(channel_id=UNMAPPED_CHANNEL, config=cfg)
        assert area is None, (
            "an unmapped channel must get a positive non-answer (None), "
            f"not a fallback value; got {area!r}"
        )

    def test_mapped_area_channel_never_admits_a_bot(self):
        """AC3 boundary at the routing decision: mapped-area channels are
        routed (area != None) yet a bot event in one stays unadmitted
        (reply-loop rule). This row does NOT pin positive area-channel
        admission — the no-area predicate it calls cannot observe it; that
        is pinned by ``test_zulip_config_credentials.py::TestChannelAdmission::
        test_mapped_area_channel_admits_allowed_human`` and by E's
        ``test_area_routing_and_isolation``."""
        from sr2_spectre.interfaces.zulip.config import resolve_channel_area

        route = resolve_channel_area
        cfg = ZulipConfig(
            zuliprc=ZULIPRC,
            allowed_user_ids=[DIEGO_ID],
            channels={ALPHA_CHANNEL: AREA_OBSIDIAN},
        )
        assert route(channel_id=ALPHA_CHANNEL, config=cfg) is not None
        assert not is_configured_no_area_channel(
            channel_id=ALPHA_CHANNEL,
            sender_id=DIEGO_ID,
            sender_is_bot=True,
            config=cfg,
        )

    def test_routing_reads_the_mapping_not_the_configured_channel_count(self):
        """Each configured channel routes to ITS OWN area; two channels with
        different areas never merge into one answer."""
        from sr2_spectre.interfaces.zulip.config import resolve_channel_area

        route = resolve_channel_area
        cfg = ZulipConfig(
            zuliprc=ZULIPRC,
            allowed_user_ids=[DIEGO_ID],
            channels={ALPHA_CHANNEL: "alpha-area", BETA_CHANNEL: "beta-area"},
        )
        assert route(channel_id=ALPHA_CHANNEL, config=cfg) == "alpha-area"
        assert route(channel_id=BETA_CHANNEL, config=cfg) == "beta-area"


# ---------------------------------------------------------------------------
# AC6 — whole-config reload through the existing source, Zulip slice
# ---------------------------------------------------------------------------


class TestWholeConfigReloadSlice:
    def test_current_starts_at_the_resolved_zulip_block(self):
        initial = _spectre(
            ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                        channels={NO_AREA_CHANNEL: ""})
        )
        from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView

        view = ZulipConfigView(SpectreConfigSource.static(initial))
        assert view.current.channels == {NO_AREA_CHANNEL: ""}

    def test_view_reload_picks_up_routing_and_allowlist_changes(self):
        """One reload per inbound message refreshes the Zulip routing slice —
        a channel added/moved on disk routes differently on the next read."""
        configs = [
            _spectre(ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                                 channels={NO_AREA_CHANNEL: ""})),
            _spectre(ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                                 channels={ALPHA_CHANNEL: AREA_OBSIDIAN})),
        ]
        source = SpectreConfigSource(loader=lambda: configs[1], initial=configs[0])
        from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView

        view = ZulipConfigView(source)

        assert view.current.channels == {NO_AREA_CHANNEL: ""}
        reloaded = view.reload()
        assert reloaded.channels == {ALPHA_CHANNEL: AREA_OBSIDIAN}
        assert view.current.channels == {ALPHA_CHANNEL: AREA_OBSIDIAN}

    def test_view_slice_decides_admission_and_routing_after_reload(self):
        """The reloaded slice is the routing/admission input of record:
        is_configured_no_area_channel and the routing decision see the edit
        on the slice reload() returns and on current. Covers the AC3 boundary
        (user allowlist and mapping changes) without re-caging the accepted
        spc-123 admission matrix.

        B3-correction note: the pre-edit arm asserted an allowed human in
        the mapped channel stayed UNADMITTED — the expired AC16-INTERMEDIATE
        contract, contradicting item 1's shipped contract. The arm is
        inverted in place; the bot arm stays unadmitted."""
        configs = [
            _spectre(ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                                 channels={NO_AREA_CHANNEL: ""})),
            _spectre(ZulipConfig(zuliprc=ZULIPRC,
                                 allowed_user_ids=[DIEGO_ID, 11],
                                 channels={ALPHA_CHANNEL: AREA_OBSIDIAN,
                                           NO_AREA_CHANNEL: ""})),
        ]
        source = SpectreConfigSource(loader=lambda: configs[1], initial=configs[0])
        from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView

        view = ZulipConfigView(source)

        before = view.current
        assert 11 not in before.allowed_user_ids
        assert is_configured_no_area_channel(
            channel_id=NO_AREA_CHANNEL, sender_id=11,
            sender_is_bot=False, config=before,
        ) is False

        after = view.reload()
        assert 11 in after.allowed_user_ids
        assert is_configured_no_area_channel(
            channel_id=NO_AREA_CHANNEL, sender_id=11,
            sender_is_bot=False, config=after,
        ) is True
        # Shipped contract (upgraded in place from the expired no-area-only
        # arm): the reloaded mapping admits an allowlisted human in the
        # mapped channel.
        assert is_configured_no_area_channel(
            channel_id=ALPHA_CHANNEL, sender_id=11,
            sender_is_bot=False, config=after,
        ) is True
        # AC3: a bot event stays unadmitted on the reloaded slice even in a
        # mapped channel (reply-loop rule outranks everything).
        assert is_configured_no_area_channel(
            channel_id=ALPHA_CHANNEL, sender_id=11,
            sender_is_bot=True, config=after,
        ) is False
        assert view.current is after
        # The reloaded slice is the routing input of record: the edit decides
        # the area answer through the same object the interface reads.
        from sr2_spectre.interfaces.zulip.config import resolve_channel_area

        route = resolve_channel_area
        assert route(channel_id=ALPHA_CHANNEL, config=after) == AREA_OBSIDIAN

    def test_missing_zulip_block_reads_as_explicitly_empty(self):
        """A whole-config load without a zulip block yields an empty slice
        (nobody admitted), never an error at read time and never an implicit
        routing population."""
        from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView

        view = ZulipConfigView(
            SpectreConfigSource.static(_spectre(None))
        )
        assert view.current.allowed_user_ids == []
        assert view.current.channels == {}

    def test_same_single_read_refreshes_the_agents_half(self):
        """AC6 (spec Interfaces / accepted DiscordConfigView pattern): the
        view reload is the process's ONE whole-config read — the agent half
        (models/endpoint) of the same load is refreshed too, so
        ``Agent.apply_config`` can consume it without a second parse."""
        configs = [
            _spectre(ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                                 channels={NO_AREA_CHANNEL: ""})),
            # Built through normal SpectreConfig construction (not
            # model_copy): the entry is a validated ModelConfig, exactly as
            # the disk loader would hand it over.
            _spectre(ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                                 channels={ALPHA_CHANNEL: AREA_OBSIDIAN}),
                     model="edited-model"),
        ]
        source = SpectreConfigSource(loader=lambda: configs[1], initial=configs[0])
        from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView

        view = ZulipConfigView(source)

        view.reload()

        assert source.current.models["default"].model == "edited-model"
        assert source.current.zulip is not None
        assert source.current.zulip.channels == {ALPHA_CHANNEL: AREA_OBSIDIAN}

    def test_malformed_reload_keeps_the_last_good_zulip_slice(self):
        """FR6: a broken file never takes the process down — routing stays
        on the last known-good mapping."""
        good = _spectre(ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                                    channels={NO_AREA_CHANNEL: "",
                                              ALPHA_CHANNEL: AREA_OBSIDIAN}))

        def _loader() -> SpectreConfig:
            raise ValueError("bad yaml")

        source = SpectreConfigSource(loader=_loader, initial=good)
        from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView

        view = ZulipConfigView(source)

        assert view.reload().channels == {NO_AREA_CHANNEL: "",
                                          ALPHA_CHANNEL: AREA_OBSIDIAN}
        assert source.current is good  # returned, not swapped, not crashed

    def test_recovery_after_a_malformed_reload(self):
        """After the file parses again, the next reload applies the new
        routing; the last-good state is not sticky past recovery."""
        good_before = _spectre(ZulipConfig(zuliprc=ZULIPRC,
                                           allowed_user_ids=[DIEGO_ID],
                                           channels={NO_AREA_CHANNEL: ""}))
        good_after = _spectre(ZulipConfig(zuliprc=ZULIPRC,
                                          allowed_user_ids=[DIEGO_ID],
                                          channels={BETA_CHANNEL: AREA_OBSIDIAN}))
        state: list[Any] = [ValueError("mid-save"), good_after]

        def _loader() -> SpectreConfig:
            item = state.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        source = SpectreConfigSource(loader=_loader, initial=good_before)
        from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView

        view = ZulipConfigView(source)

        assert view.reload().channels == {NO_AREA_CHANNEL: ""}   # last-good
        assert view.reload().channels == {BETA_CHANNEL: AREA_OBSIDIAN}

    def test_reload_failure_is_logged_without_secret_values(
        self, caplog: pytest.LogCaptureFixture
    ):
        """NFR: reload logs field names, never config values."""
        marker = "SECRET-MARKER-spc120-9f3a"
        good = _spectre(ZulipConfig(zuliprc=marker, allowed_user_ids=[DIEGO_ID],
                                    channels={NO_AREA_CHANNEL: ""}))
        source = SpectreConfigSource(
            loader=lambda: (_ for _ in ()).throw(ValueError("bad yaml")),
            initial=good,
        )
        from sr2_spectre.interfaces.zulip.config_source import ZulipConfigView

        view = ZulipConfigView(source)
        with caplog.at_level(logging.WARNING):
            view.reload()
        assert marker not in caplog.text


# ---------------------------------------------------------------------------
# AC6 — startup wiring: the Zulip entry point gets a reloadable whole source
# ---------------------------------------------------------------------------


class TestZulipWholeConfigSourceWiring:
    def test_source_loader_reloads_edited_block_from_disk(self, isolated_dirs):
        """build_spectre_config_source (the CLI's whole-config source
        builder) re-runs full resolution, so an edit to the zulip block —
        including a mapped area that the frozen startup block could not
        carry — is picked up on the next reload()."""
        from sr2_spectre.cli import build_spectre_config_source, resolve_config

        cwd, _home = isolated_dirs
        pos = cwd / "config.yaml"
        _write_yaml(pos, {**VALID_TAIL, "zulip": _zulip_block()})
        initial = resolve_config(pos, cwd=cwd, env={})
        source = build_spectre_config_source(pos, cwd, initial)

        _write_yaml(pos, {**VALID_TAIL,
                          "zulip": _zulip_block(channels={ALPHA_CHANNEL: AREA_OBSIDIAN})})

        assert source.current.zulip.channels == {
            NO_AREA_CHANNEL: "", ALPHA_CHANNEL: AREA_OBSIDIAN,
            BETA_CHANNEL: AREA_OBSIDIAN,
        }
        reloaded = source.reload()
        assert reloaded.zulip is not None
        assert reloaded.zulip.channels == {ALPHA_CHANNEL: AREA_OBSIDIAN}

    def test_broken_edit_on_disk_keeps_last_good_block(self, isolated_dirs):
        from sr2_spectre.cli import build_spectre_config_source, resolve_config

        cwd, _home = isolated_dirs
        pos = cwd / "config.yaml"
        _write_yaml(pos, {**VALID_TAIL, "zulip": _zulip_block()})
        initial = resolve_config(pos, cwd=cwd, env={})
        source = build_spectre_config_source(pos, cwd, initial)

        pos.write_text("zulip:\n  channels: {42: \"\",,,\n")

        assert source.reload().zulip.channels == {
            NO_AREA_CHANNEL: "", ALPHA_CHANNEL: AREA_OBSIDIAN,
            BETA_CHANNEL: AREA_OBSIDIAN,
        }


# ---------------------------------------------------------------------------
# AC2/AC3 — startup accepts mapped areas and future named areas
# ---------------------------------------------------------------------------


class TestStartupAcceptsMappedAreas:
    """The temporary nonempty-area restriction is lifted by this cut:
    mapped (and not-yet-existing) named areas resolve at startup; only
    structurally invalid mappings fail."""

    def test_resolution_accepts_mapped_and_no_area_channels(self, isolated_dirs):
        """A valid routing map — named areas plus the explicit "" channel —
        resolves without error. Red today because the startup gate rejects
        any nonempty area (spc-119 intermediate contract)."""
        from sr2_spectre.cli import resolve_config

        cwd, _home = isolated_dirs
        pos = cwd / "config.yaml"
        _write_yaml(pos, {**VALID_TAIL, "zulip": _zulip_block()})
        config = resolve_config(pos, cwd=cwd, env={})
        assert config.zulip is not None
        assert config.zulip.channels[ALPHA_CHANNEL] == AREA_OBSIDIAN
        assert config.zulip.channels[NO_AREA_CHANNEL] == ""

    def test_future_named_area_resolves_without_live_vault_access(
        self, isolated_dirs
    ):
        """Spec FR / implementation note: "Source config may contain future
        named areas; validate mapping shape, not live vault access." A never-
        created area identity resolves fine — resolution opens no vault.
        (Pin: green before and after the lift — the gate never lived in
        resolve_config; it stays as the mapping-shape guard.)"""
        from sr2_spectre.cli import resolve_config

        cwd, _home = isolated_dirs
        pos = cwd / "config.yaml"
        _write_yaml(pos, {**VALID_TAIL,
                          "zulip": _zulip_block(channels={ALPHA_CHANNEL: "does-not-exist-yet"})})
        config = resolve_config(pos, cwd=cwd, env={})
        assert config.zulip.channels == {ALPHA_CHANNEL: "does-not-exist-yet"}
