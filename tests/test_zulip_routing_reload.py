"""Unit cage — spc-120 (Add Zulip area routing and live config reload).

A proves ONLY the pure routing/reload/config-boundary behavior of the Zulip
front door:

- a pure channel→area routing decision: a configured channel gets its mapped
  area, a configured empty string stays the explicit no-area, and an
  unconfigured channel is never admitted — so it can never fall through to a
  working-directory area (AC2, spec Interfaces: "No implicit channel-name or
  launch-directory fallback");
- one resolved config per inbound message through the existing whole-config
  source: reload → slice for routing AND the same single read for
  ``Agent.apply_config``, last-good retained on a malformed reload (AC6,
  spec FR6, mirroring the accepted ``DiscordConfigView`` pattern);
- startup resolution of the whole-config source for the Zulip entry point.

Deliberately NOT here: the Zulip view class itself (C owns
``interfaces/zulip/config_source.py``), the live interface/transport/Session
orchestration, resolver-context and compiled-request proofs — E owns those in
``tests/test_zulip_routing_reload_e2e.py`` through the real CLI. Accepted
spc-123 credential/schema proof and spc-124 transport proof are not re-caged.
No HTTP, no model, no subprocess, no sockets in this file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from sr2_spectre.config import SpectreConfig
from sr2_spectre.config_source import SpectreConfigSource
from sr2_spectre.interfaces.zulip import ZulipConfig, is_configured_no_area_channel

DIEGO_ID = 8                     # Diego (spec Dependencies)
NO_AREA_CHANNEL = 42             # explicit no-area channel (area "")
ALPHA_CHANNEL = 44               # mapped channel (area AREA_OBSIDIAN)
BETA_CHANNEL = 45                # mapped channel, area BETA_AREA
UNMAPPED_CHANNEL = 43            # no entry at all — must never be admitted

ZULIPRC = "/tmp/spc-120-unused.zuliprc"     # never read: no credential I/O

# Routing decision — the observable A pins. Tests resolve the function by
# signature (the ONLY callable exported from the config module taking exactly
# (channel_id, config)), so no private seam and no name is assumed before C
# ships the capability.
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


def _spectre(zulip: ZulipConfig | None) -> SpectreConfig:
    return SpectreConfig(
        agent={"name": "miranda"},
        models={"default": {"model": "synthetic-model"}},
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


def resolve_routing_fn() -> Callable[..., str | None]:
    """Return the public pure routing decision from the Zulip config module.

    Signature contract (the spec's routing rule): called with the keyword
    arguments ``channel_id`` and ``config``, returns the area identity for a
    configured channel, ``""`` for an explicitly no-area channel, and None
    (or raises) for an unconfigured channel. Discovered by signature so the
    test fails as missing-capability (not attribute guesswork) until the
    routing release provides it.
    """
    import importlib
    import inspect
    import pkgutil

    import sr2_spectre.interfaces.zulip as pkg

    modules = [pkg] + [
        importlib.import_module(m)
        for m in (
            f"{pkg.__name__}.{mi.name}"
            for mi in pkgutil.iter_modules(pkg.__path__)
            if not mi.name.startswith("_")
        )
    ]
    candidates = []
    for mod in modules:
        for name, obj in vars(mod).items():
            if not callable(obj) or name.startswith("_"):
                continue
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            params = sig.parameters
            if "channel_id" in params and "config" in params:
                candidates.append(obj)
    if not candidates:
        pytest.fail(
            "missing capability: no public callable(channel_id=..., "
            "config=...) routing decision in sr2_spectre.interfaces.zulip "
            "(spc-120 area routing)"
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# AC2 — pure channel→area routing, no implicit fallback
# ---------------------------------------------------------------------------


class TestAreaRoutingDecision:
    def test_mapped_channel_reaches_resolvers_with_its_area(self):
        route = resolve_routing_fn()
        cfg = ZulipConfig(
            zuliprc=ZULIPRC,
            allowed_user_ids=[DIEGO_ID],
            channels={ALPHA_CHANNEL: AREA_OBSIDIAN},
        )
        assert route(channel_id=ALPHA_CHANNEL, config=cfg) == AREA_OBSIDIAN

    def test_explicit_no_area_channel_stays_empty_string(self):
        """RunContext.area="" is the explicit no-area contract — the routing
        answer must be "", never None (None permits consumer fallback)."""
        route = resolve_routing_fn()
        cfg = ZulipConfig(
            zuliprc=ZULIPRC,
            allowed_user_ids=[DIEGO_ID],
            channels={NO_AREA_CHANNEL: ""},
        )
        area = route(channel_id=NO_AREA_CHANNEL, config=cfg)
        assert area == ""
        assert area is not None

    def test_unmapped_channel_is_not_routed(self):
        """AC2: an explicitly unmapped channel cannot accidentally inherit
        the service's working-directory area — it gets no area answer at
        all (None or an explicit rejection), never a name or cwd fallback."""
        route = resolve_routing_fn()
        cfg = ZulipConfig(
            zuliprc=ZULIPRC,
            allowed_user_ids=[DIEGO_ID],
            channels={NO_AREA_CHANNEL: ""},
        )
        try:
            area = route(channel_id=UNMAPPED_CHANNEL, config=cfg)
        except Exception:
            return  # an explicit rejection is also a non-answer
        assert area is None

    def test_mapped_area_channel_never_admits_a_bot(self):
        """AC3 boundary at the routing decision: mapped-area channels are
        routed (area != None) yet are NOT admitted by the accepted no-area
        trigger predicate — bot events stay trigger-free either way, and
        area-channel admission is not regressed into ""-only."""
        route = resolve_routing_fn()
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
        route = resolve_routing_fn()
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


def build_zulip_view(
    source: SpectreConfigSource,
) -> tuple[Any, Callable[[], SpectreConfig]]:
    """Build C's Zulip whole-config view over the accepted source machinery.

    Tries the package module, then the interfaces.zulip.config_source module
    named in the bead's scope; the view is any public class exposing
    ``reload()`` → ZulipConfig-shaped slice plus ``source`` → the same
    SpectreConfigSource. Fails as missing-capability until the routing
    release ships it. Returns (view, whole_config_getter).
    """
    import importlib
    import inspect

    classes: list[type] = []
    for mod_name in (
        "sr2_spectre.interfaces.zulip",
        "sr2_spectre.interfaces.zulip.config_source",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        classes.extend(
            obj
            for name, obj in vars(mod).items()
            if inspect.isclass(obj)
            and callable(getattr(obj, "reload", None))
            and hasattr(obj, "source")
        )
    if not classes:
        pytest.fail(
            "missing capability: no Zulip config view (reload() + source) "
            "exported from sr2_spectre.interfaces.zulip (spc-120 reload)"
        )
    view = classes[0](source)
    return view, lambda: view.source.current


class TestWholeConfigReloadSlice:
    def test_current_starts_at_the_resolved_zulip_block(self):
        initial = _spectre(
            ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                        channels={NO_AREA_CHANNEL: ""})
        )
        view, _whole = build_zulip_view(SpectreConfigSource.static(initial))
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
        view, _whole = build_zulip_view(source)

        assert view.current.channels == {NO_AREA_CHANNEL: ""}
        reloaded = view.reload()
        assert reloaded.channels == {ALPHA_CHANNEL: AREA_OBSIDIAN}
        assert view.current.channels == {ALPHA_CHANNEL: AREA_OBSIDIAN}

    def test_view_slice_decides_admission_and_routing_after_reload(self):
        """The reloaded slice is the routing/admission input of record:
        is_configured_no_area_channel and the new routing decision see the
        edit on the slice reload() returns and on current. Covers the AC3
        boundary (user allowlist and mapping changes) without re-caging the
        accepted spc-123 admission matrix."""
        from sr2_spectre.interfaces.zulip import is_configured_no_area_channel

        configs = [
            _spectre(ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                                 channels={NO_AREA_CHANNEL: ""})),
            _spectre(ZulipConfig(zuliprc=ZULIPRC,
                                 allowed_user_ids=[DIEGO_ID, 11],
                                 channels={ALPHA_CHANNEL: AREA_OBSIDIAN,
                                           NO_AREA_CHANNEL: ""})),
        ]
        source = SpectreConfigSource(loader=lambda: configs[1], initial=configs[0])
        view, _whole = build_zulip_view(source)

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
        assert is_configured_no_area_channel(
            channel_id=ALPHA_CHANNEL, sender_id=11,
            sender_is_bot=False, config=after,
        ) is False
        # AC3: a bot event stays unadmitted on the reloaded slice even in a
        # mapped channel (reply-loop rule outranks everything).
        assert is_configured_no_area_channel(
            channel_id=ALPHA_CHANNEL, sender_id=11,
            sender_is_bot=True, config=after,
        ) is False
        assert view.current is after

    def test_missing_zulip_block_reads_as_explicitly_empty(self):
        """A whole-config load without a zulip block yields an empty slice
        (nobody admitted), never an error at read time and never an implicit
        routing population."""
        view, _whole = build_zulip_view(
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
            _spectre(ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                                 channels={ALPHA_CHANNEL: AREA_OBSIDIAN})),
        ]
        configs[1] = configs[1].model_copy(
            update={"models": {"default": {"model": "edited-model"}}}
        )
        source = SpectreConfigSource(loader=lambda: configs[1], initial=configs[0])
        view, whole = build_zulip_view(source)

        view.reload()

        assert whole().models["default"].model == "edited-model"
        assert whole().zulip is not None
        assert whole().zulip.channels == {ALPHA_CHANNEL: AREA_OBSIDIAN}

    def test_malformed_reload_keeps_the_last_good_zulip_slice(self):
        """FR6: a broken file never takes the process down — routing stays
        on the last known-good mapping."""
        good = _spectre(ZulipConfig(zuliprc=ZULIPRC, allowed_user_ids=[DIEGO_ID],
                                    channels={NO_AREA_CHANNEL: "",
                                              ALPHA_CHANNEL: AREA_OBSIDIAN}))

        def _loader() -> SpectreConfig:
            raise ValueError("bad yaml")

        source = SpectreConfigSource(loader=_loader, initial=good)
        view, whole = build_zulip_view(source)

        assert view.reload().channels == {NO_AREA_CHANNEL: "",
                                          ALPHA_CHANNEL: AREA_OBSIDIAN}
        assert whole() is good  # returned, not swapped, not crashed

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
        view, _whole = build_zulip_view(source)

        assert view.reload().channels == {NO_AREA_CHANNEL: ""}   # last-good
        assert view.reload().channels == {BETA_CHANNEL: AREA_OBSIDIAN}

    def test_reload_failure_is_logged_without_secret_values(
        self, caplog: pytest.LogCaptureFixture
    ):
        """NFR: reload logs name changed fields, never config values."""
        marker = "SECRET-MARKER-spc120-9f3a"
        good = _spectre(ZulipConfig(zuliprc=marker, allowed_user_ids=[DIEGO_ID],
                                    channels={NO_AREA_CHANNEL: ""}))
        source = SpectreConfigSource(
            loader=lambda: (_ for _ in ()).throw(ValueError("bad yaml")),
            initial=good,
        )
        view, _whole = build_zulip_view(source)
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
        created area identity resolves fine — resolution opens no vault."""
        from sr2_spectre.cli import resolve_config

        cwd, _home = isolated_dirs
        pos = cwd / "config.yaml"
        _write_yaml(pos, {**VALID_TAIL,
                          "zulip": _zulip_block(channels={ALPHA_CHANNEL: "does-not-exist-yet"})})
        config = resolve_config(pos, cwd=cwd, env={})
        assert config.zulip.channels == {ALPHA_CHANNEL: "does-not-exist-yet"}
