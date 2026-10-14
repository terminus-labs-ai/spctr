"""Unit cage — spc-123 (Validate Zulip configuration and credentials, AC17).

A proves the PURE Zulip configuration, credential-loading and admission
decisions through narrow public seams of the ``interfaces.zulip`` package
and its modules:

- ``ZulipConfig`` (``interfaces.zulip.config``) — the spec's config block:
  ``zuliprc`` credential path (no inline key), optional ``api_url``,
  explicit integer ``allowed_user_ids``, and ``channels`` mapping channel
  ID → area identity (``""`` = explicitly no area).
- ``ZulipCredentials`` — the parsed credential record (email/api_key/site)
  returned by the loader; its fields are the spec's public contract.
- ``load_zulip_credentials`` — synthetic zuliprc reading with explicit,
  non-leaking failures (spec FR2: "Read Miranda's bot credentials from the
  saved zuliprc path, with no inline key"; Interfaces: "Missing credentials
  ... fail startup explicitly"; NFR: "Do not expose credentials in
  messages, logs ...").
- ``ZulipCredentialError`` — the declared credential failure-boundary
  exception type, exported from the package seam.
- ``is_configured_no_area_channel`` — the pure trigger-admission decision
  (spec FR5: "Restrict the pilot to Diego's user ID and explicitly
  configured channel IDs; ignore bot messages as triggers").
- ``TestNonemptyAreaGateIntermediate`` — spc-123-era rows were upgraded by
  spc-120 to the shipped routing behavior (mapped areas are valid): the
  named startup validator (``validate_no_area_only_routing`` in this cut)
  must accept a mapped-area map instead of rejecting it. The rows no longer
  probe for the helper's existence — they call it directly and fail if it
  rejects a mapped area.

Everything runs against synthetic data under ``tmp_path``: no network, no
model, no CLI dispatch, no Session orchestration — AC17's "no HTTP or
model work" is honored by the test itself (no aiohttp/socket/subprocess in
this file).

ROUTING CONTRACT (spc-120, supersedes the AC16-INTERMEDIATE note of the
spc-119/123 cuts): mapped-area channels are valid startup configuration;
the temporary nonempty-area rejection exists nowhere in this file, and the
routed area itself is proven by the spc-120 routing/journey cages. The
original approval record of the removed assertions stays in git history.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sr2_spectre.interfaces.zulip import (
    ZulipCredentialError,
    ZulipCredentials,
    is_configured_no_area_channel,
    load_zulip_credentials,
)
from sr2_spectre.interfaces.zulip.config import ZulipConfig

# ---------------------------------------------------------------------------
# Fixtures — identities mirror the spec's Dependencies (Miranda bot user ID
# 9, Diego user ID 8); everything synthetic, under tmp_path; no real realm,
# no production zuliprc, no secret.
# ---------------------------------------------------------------------------

BOT_ID = 9          # Miranda bot (spec Dependencies)
DIEGO_ID = 8        # Diego (spec Dependencies)
INTRUDER_ID = 7     # a human who is not on the allowlist

BOT_EMAIL = "miranda-bot@zulip.example"
SITE = "https://normandy.example.com"
API_URL = "http://127.0.0.1:8899"   # loopback override; never dialled here
SECRET = "synthetic-key-DO-NOT-LEAK-1234"
API_KEY = "synthetic-key-for-tests-0000"

NO_AREA_CHANNEL = 42
OTHER_NO_AREA_CHANNEL = 43
UNCONFIGURED_CHANNEL = 99
AREA_CHANNEL = 44


def make_config(**kw: Any) -> ZulipConfig:
    """Spec-schema ZulipConfig: zuliprc path, optional api_url, ID lists."""
    base: dict[str, Any] = {
        "zuliprc": "/tmp/spc-123-unused.zuliprc",
        "api_url": API_URL,
        "allowed_user_ids": [DIEGO_ID],
        "channels": {NO_AREA_CHANNEL: ""},
    }
    base.update(kw)
    return ZulipConfig(**base)


@pytest.fixture
def zuliprc(tmp_path: Path) -> Path:
    """Synthetic zuliprc (standard ``key =`` field) under /tmp."""
    path = tmp_path / "miranda.zuliprc"
    path.write_text(
        f"[zulip]\nemail = {BOT_EMAIL}\nkey = {API_KEY}\nsite = {SITE}\n"
    )
    return path


@pytest.fixture
def incomplete_zuliprc(tmp_path: Path) -> Path:
    """zuliprc carrying a real-looking secret but a required field missing."""
    path = tmp_path / "partial.zuliprc"
    path.write_text(f"[zulip]\nemail = {BOT_EMAIL}\nkey = {SECRET}\n")
    return path


@pytest.fixture
def unreadable_zuliprc(tmp_path: Path) -> Path:
    """Present but unreadable (no read permission) credential file whose
    body carries a marker that must never surface in an error message."""
    path = tmp_path / "locked.zuliprc"
    path.write_text(f"[zulip]\nemail = {BOT_EMAIL}\nkey = {SECRET}\nsite = {SITE}\n")
    path.chmod(0)
    yield path
    path.chmod(0o600)  # cleanup so tmp teardown can remove it


def load_expect_credential_error(path: Path) -> ZulipCredentialError:
    """Load a path that must fail and return the raised credential error.

    Fails with a distinct message when nothing is raised, so classification
    tests report "not rejected" rather than an attribute error.
    """
    with pytest.raises(ZulipCredentialError) as exc_info:
        load_zulip_credentials(str(path))
    return exc_info.value


# ---------------------------------------------------------------------------
# ZulipConfig schema — explicit integer-ID allowlists only (spec Interfaces)
# ---------------------------------------------------------------------------


class TestZulipConfigSchema:
    def test_defaults_are_explicitly_empty(self):
        """Missing allowlists mean 'admits nobody' and 'no channels',
        never an implicit fallback population."""
        cfg = ZulipConfig(zuliprc="/tmp/spc-123-unparsed.zuliprc")
        assert cfg.allowed_user_ids == []
        assert cfg.channels == {}
        assert cfg.api_url is None

    def test_zuliprc_is_a_required_credential_path(self):
        with pytest.raises(ValidationError):
            ZulipConfig()  # type: ignore[call-arg]

    def test_channels_map_id_to_area_including_empty(self):
        cfg = make_config(
            channels={NO_AREA_CHANNEL: "", OTHER_NO_AREA_CHANNEL: "obsidian"}
        )
        assert cfg.channels[NO_AREA_CHANNEL] == ""
        assert cfg.channels[OTHER_NO_AREA_CHANNEL] == "obsidian"
        assert cfg.allowed_user_ids == [DIEGO_ID]

    def test_non_integer_channel_key_rejected(self):
        """Spec: "no implicit channel-name or launch-directory fallback" —
        channel keys are explicit integer Zulip IDs."""
        with pytest.raises(ValidationError):
            make_config(channels={"not-an-id": ""})

    def test_digit_string_channel_key_rejected(self):
        """AC17: "explicit integer user/channel admission" — a digit string
        is not an explicit integer ID, so ``{"42": ...}`` (as it could
        arrive from a YAML-quoted ID) must not be admitted; the key is the
        integer literal 42 only."""
        with pytest.raises(ValidationError):
            make_config(channels={"42": ""})

    def test_non_integer_user_id_rejected(self):
        """Spec: explicit integer user IDs only."""
        with pytest.raises(ValidationError):
            make_config(allowed_user_ids=["not-an-id"])

    def test_digit_string_user_id_rejected(self):
        """AC17: "explicit integer user ... admission" — the allowlist must
        carry integer literals; a digit string like ``"8"`` (YAML-quoted)
        is not an explicit integer and is not coerced into one."""
        with pytest.raises(ValidationError):
            make_config(allowed_user_ids=["8"])

    def test_non_string_area_rejected(self):
        """An area identity is a string; ``""`` is the explicit no-area
        value, so a list is not a legal mapping value."""
        with pytest.raises(ValidationError):
            make_config(channels={NO_AREA_CHANNEL: ["not", "a", "string"]})

    def test_zuliprc_must_be_a_string_path(self):
        """AC17 pins the credential *path* as a string; an integer or list
        there is not a path and must be rejected by the schema."""
        with pytest.raises(ValidationError):
            make_config(zuliprc=12345)
        with pytest.raises(ValidationError):
            make_config(zuliprc=["x"])


# ---------------------------------------------------------------------------
# Credentials — synthetic zuliprc, explicit failures, no secret disclosure
# (spec FR2; Interfaces: "Missing credentials ... fail startup explicitly";
# NFR: "Do not expose credentials in messages, logs, configuration examples
# or Beads")
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_parses_synthetic_zuliprc(self, zuliprc: Path):
        creds = load_zulip_credentials(str(zuliprc))
        assert creds.email == BOT_EMAIL
        assert creds.api_key == API_KEY
        assert creds.site == SITE

    def test_missing_file_raises_credential_error(self, tmp_path: Path):
        with pytest.raises(ZulipCredentialError):
            load_zulip_credentials(str(tmp_path / "absent.zuliprc"))

    def test_malformed_file_raises_credential_error(self, tmp_path: Path):
        """Genuinely malformed INI: the parse itself fails with
        configparser.DuplicateOptionError — a class distinct from
        MissingSectionHeaderError, the error the wrong-section fixture below
        produces — so the two tests stay separate observables. (Verified
        against configparser directly here rather than assumed; both parse
        routes a loader might use, read_file and read_string, normalize
        DuplicateOptionError to MissingSectionHeaderError, so a loader that
        swallows the parse error and re-dispatches on has_section() cannot
        satisfy this and the section test with one indistinguishable arm.)"""
        body = "[zulip]\nemail = a@example.invalid\nemail = b@example.invalid\n"
        bad = tmp_path / "bad.zuliprc"
        bad.write_text(body)
        with pytest.raises(configparser.DuplicateOptionError):
            configparser.ConfigParser().read_string(body)
        with pytest.raises(ZulipCredentialError):
            load_zulip_credentials(str(bad))

    def test_unreadable_file_raises_credential_error(self, tmp_path: Path):
        """Unreadable even though it exists: the loader must not mistake an
        open failure for "file not found". ``configparser.read()`` swallows
        PermissionError into an empty result, so a loader built on it can
        only satisfy this with an explicit readability probe; fixture mode
        0o644 proves the rejection comes from the permission, not absence.
        (Runs as non-root: root could read a 0o000 file.)"""
        if not hasattr(os, "geteuid") or os.geteuid() == 0:
            pytest.skip("file-permission boundary requires a non-root user")
        path = tmp_path / "locked-raises.zuliprc"
        path.write_text(f"[zulip]\nemail = {BOT_EMAIL}\nkey = {SECRET}\nsite = {SITE}\n")
        path.chmod(0)
        try:
            with pytest.raises(ZulipCredentialError):
                load_zulip_credentials(str(path))
        finally:
            path.chmod(0o644)

    def test_missing_zulip_section_raises_credential_error(self, tmp_path: Path):
        wrong = tmp_path / "wrong-section.zuliprc"
        wrong.write_text(f"[other]\nemail = {BOT_EMAIL}\nkey = {SECRET}\n")
        probe = configparser.ConfigParser()
        probe.read_string(wrong.read_text())  # parses cleanly: INI-valid, wrong section
        assert not probe.has_section("zulip")
        with pytest.raises(ZulipCredentialError):
            load_zulip_credentials(str(wrong))

    def test_incomplete_file_raises_credential_error(self, incomplete_zuliprc: Path):
        with pytest.raises(ZulipCredentialError):
            load_zulip_credentials(str(incomplete_zuliprc))

    def test_incomplete_file_error_never_leaks_the_api_key(
        self, incomplete_zuliprc: Path
    ):
        """AC17: missing/invalid credentials are reported without disclosing
        their contents — the file here carries a live-looking secret that
        must not surface in the exception text."""
        error = load_expect_credential_error(incomplete_zuliprc)
        assert SECRET not in str(error)
        assert SECRET not in repr(error)

    def test_unreadable_file_error_never_leaks_file_contents(
        self, unreadable_zuliprc: Path
    ):
        """AC17 "without disclosing their contents" on the one failure path
        whose message necessarily names the file: even a file the reader
        could not open must not have its body echoed into str() or repr()."""
        if not hasattr(os, "geteuid") or os.geteuid() == 0:
            # Root bypasses the permission bit; the unreadable-directory
            # variant (test_directory_as_path_error_never_leaks_marker)
            # keeps the no-disclosure guarantee tested on this run.
            pytest.skip("file-permission boundary requires a non-root user")
        error = load_expect_credential_error(unreadable_zuliprc)
        assert SECRET not in str(error)
        assert SECRET not in repr(error)
        assert BOT_EMAIL not in str(error)

    def test_directory_as_path_error_never_leaks_marker(self, tmp_path: Path):
        """AC17 no-disclosure on an unreadable path that is not a permission
        failure: a directory with a synthetic marker file inside cannot be
        opened as a file, and its marker must not surface in the error."""
        directory = tmp_path / "locked-dir"
        directory.mkdir()
        (directory / "marker.zuliprc").write_text(
            f"[zulip]\nemail = {BOT_EMAIL}\nkey = {SECRET}\nsite = {SITE}\n"
        )
        error = load_expect_credential_error(directory)
        assert SECRET not in str(error)
        assert SECRET not in repr(error)
        assert BOT_EMAIL not in str(error)

    def test_interpolating_key_value_fails_without_leaking_raw_value(
        self, tmp_path: Path
    ):
        """Regression: a key value containing an interpolation marker
        (``%(missing)s``) must not escape as a raw configparser
        interpolation error — that error embeds the raw value, disclosing
        the credential. The loader reports a sanitized credential failure
        (or preserves a valid value) and never echoes the raw value."""
        marker = "LEAK-MARKER-INTERP-98765"
        path = tmp_path / "interp.zuliprc"
        path.write_text(
            f"[zulip]\nemail = {BOT_EMAIL}\n"
            f"key = {marker} %(missing)s\nsite = {SITE}\n"
        )
        try:
            creds = load_zulip_credentials(str(path))
            assert creds.api_key == f"{marker} %(missing)s"
        except Exception as exc:
            assert isinstance(exc, ZulipCredentialError)
            assert marker not in str(exc)
            assert marker not in repr(exc)

    def test_invalid_utf8_file_fails_without_leaking_bytes(
        self, tmp_path: Path
    ):
        """Regression: a credential file with invalid UTF-8 must not escape
        as a raw UnicodeDecodeError — its repr embeds the decoded bytes,
        disclosing the file contents. The loader reports a sanitized
        credential failure without echoing the bytes."""
        marker = "LEAK-MARKER-UTF8-13579"
        path = tmp_path / "binary.zuliprc"
        # Raw invalid UTF-8 bytes, spliced into otherwise valid UTF-8.
        # Built at the byte level: a Python str cannot hold 0xff, so
        # concatenating the bytes is the only way the file actually
        # contains the undecodable sequence.
        raw = b"\xff\xfe\x81"
        path.write_bytes(
            f"[zulip]\nemail = {BOT_EMAIL}\nkey = {marker} ".encode("utf-8")
            + raw
            + f"\nsite = {SITE}\n".encode("utf-8")
        )
        # Verify the fixture is genuinely undecodable as UTF-8 BEFORE
        # exercising the public loader, so the red proof below is
        # attributable to the bytes, not to fixture construction.
        try:
            path.read_bytes().decode("utf-8")
            raise AssertionError(
                "fixture is decodable as UTF-8; the test would be "
                "exercising the wrong boundary"
            )
        except UnicodeDecodeError:
            pass  # fixture is genuinely invalid UTF-8, as intended
        error = load_expect_credential_error(path)
        assert marker not in str(error)
        assert marker not in repr(error)
        assert b"\xff\xfe\x81" not in repr(error).encode("utf-8", "replace")

    def test_credential_error_type_is_exported_and_catchable(self, tmp_path: Path):
        """The declared credential-boundary type is catchable at the package
        seam: every rejected synthetic credential above surfaced as a real
        exception instance of the exported class."""
        # Catchability: the exported class is a real, raisable/catchable type.
        with pytest.raises(ZulipCredentialError):
            raise ZulipCredentialError("synthetic instance for catch")
        # The exported class is the type every rejection above surfaced:
        # re-run a representative set of failing paths through the loader.
        bad = tmp_path / "bad.zuliprc"
        bad.write_text("[zulip]\nemail = a@example.invalid\nemail = b@example.invalid\n")
        wrong = tmp_path / "wrong-section.zuliprc"
        wrong.write_text(f"[other]\nemail = {BOT_EMAIL}\nkey = {SECRET}\n")
        partial = tmp_path / "partial.zuliprc"
        partial.write_text(f"[zulip]\nemail = {BOT_EMAIL}\nkey = {SECRET}\n")
        for path in (tmp_path / "absent.zuliprc", bad, wrong, partial):
            error = load_expect_credential_error(path)
            assert isinstance(error, ZulipCredentialError)


class TestNonemptyAreaGateIntermediate:
    """spc-120 upgrade (staged behavior transition, replacing the
    AC16-INTERMEDIATE gate of spc-119/123): mapped-area channels are now
    supported routing, so a nonempty area mapping is a VALID startup
    configuration — it must never be rejected, and it must never silently
    lose its area either. The former rejection assertions
    (``validate_no_area_only_routing``) are removed with the temporary gate
    they caged; the original approval record for those assertions stays in
    git history and the artifact record. All sender/channel admission and
    credential assertions in this file are unchanged.
    """

    def test_config_accepts_nonempty_area_values(self):
        """The schema parses nonempty areas — now a permanent routing
        contract, not a schema-only tolerance ahead of a rejection gate."""
        cfg = make_config(
            channels={NO_AREA_CHANNEL: "", OTHER_NO_AREA_CHANNEL: "obsidian"}
        )
        assert cfg.channels[OTHER_NO_AREA_CHANNEL] == "obsidian"

    def test_area_mapping_requires_no_live_vault_access(self):
        """Implementation note: "validate mapping shape, not live vault
        access" — a never-created area identity is a valid mapping and its
        mere presence in a config object touches no vault."""
        cfg = make_config(channels={AREA_CHANNEL: "not-created-yet"})
        assert cfg.channels[AREA_CHANNEL] == "not-created-yet"

    def test_mapped_area_startup_restriction_removed(self):
        """Gate-lift signal, paired with the routing-aware companion
        ``test_mapped_area_channel_admits_allowed_human`` (B3 item 2): on its
        own this row goes green on a one-line gate lift with no routing.
        Pin the current startup validator directly. The temporary gate that lives at interface startup today (spc-119/123
        intermediate contract) accepts a no-area-only map and raises on a
        mapped area; after spc-120 removes it, whatever startup validation
        remains must accept the same mixed map without raising. Calling it
        positively is red today (the gate raises) and green after the lift
        either way; silent area loss stays caged by the routing decision and
        journey tests, not by this row."""
        from sr2_spectre.interfaces.zulip import validate_no_area_only_routing

        cfg = make_config(channels={NO_AREA_CHANNEL: "",
                                    AREA_CHANNEL: "obsidian"})
        assert validate_no_area_only_routing(cfg) is None


class TestChannelAdmission:
    def test_configured_no_area_channel_admitted(self):
        cfg = make_config()
        assert is_configured_no_area_channel(
            channel_id=NO_AREA_CHANNEL,
            sender_id=DIEGO_ID,
            sender_is_bot=False,
            config=cfg,
        )

    def test_unconfigured_channel_ignored(self):
        cfg = make_config()
        assert not is_configured_no_area_channel(
            channel_id=UNCONFIGURED_CHANNEL,
            sender_id=DIEGO_ID,
            sender_is_bot=False,
            config=cfg,
        )

    def test_channel_membership_decides_on_configured_ids_not_channel_name(self):
        """Spec: "no implicit channel-name or launch-directory fallback" —
        an event carrying a stringified ID is not a match for integer 42."""
        cfg = make_config()
        assert not is_configured_no_area_channel(
            channel_id="42", sender_id=DIEGO_ID, sender_is_bot=False, config=cfg
        )

    def test_mapped_area_channel_admits_allowed_human(self):
        """spc-120 shipped contract (replaces the expired AC16-INTERMEDIATE
        "nonempty area never admits" pin): a mapped-area channel with an
        allowed human sender IS admitted, so lifting the startup gate
        without implementing area-channel admission goes red here. AC3 stays
        caged alongside: bot and off-allowlist events in the same mapped
        channel, and an allowed human in an unconfigured channel, all stay
        unadmitted."""
        cfg = make_config(channels={NO_AREA_CHANNEL: "", AREA_CHANNEL: "obsidian"})
        assert is_configured_no_area_channel(
            channel_id=AREA_CHANNEL,
            sender_id=DIEGO_ID,
            sender_is_bot=False,
            config=cfg,
        )
        assert not is_configured_no_area_channel(
            channel_id=AREA_CHANNEL,
            sender_id=DIEGO_ID,
            sender_is_bot=True,
            config=cfg,
        )
        assert not is_configured_no_area_channel(
            channel_id=AREA_CHANNEL,
            sender_id=INTRUDER_ID,
            sender_is_bot=False,
            config=cfg,
        )
        assert not is_configured_no_area_channel(
            channel_id=UNCONFIGURED_CHANNEL,
            sender_id=DIEGO_ID,
            sender_is_bot=False,
            config=cfg,
        )


class TestSenderAdmission:
    def test_disallowed_user_ignored(self):
        cfg = make_config()
        assert not is_configured_no_area_channel(
            channel_id=NO_AREA_CHANNEL,
            sender_id=INTRUDER_ID,
            sender_is_bot=False,
            config=cfg,
        )

    def test_bot_never_admitted_even_when_allowlisted(self):
        """Reply-loop prevention (spec FR5: "ignore bot messages as
        triggers"): the bot-authored rule outranks an allowlisted bot ID,
        so a bot reply can never trigger itself."""
        cfg = make_config(allowed_user_ids=[DIEGO_ID, BOT_ID])
        assert not is_configured_no_area_channel(
            channel_id=NO_AREA_CHANNEL,
            sender_id=BOT_ID,
            sender_is_bot=True,
            config=cfg,
        )

    def test_empty_allowlist_admits_no_human(self):
        """An empty allowlist is explicit nobody — no implicit admission."""
        cfg = make_config(allowed_user_ids=[])
        assert not is_configured_no_area_channel(
            channel_id=NO_AREA_CHANNEL,
            sender_id=DIEGO_ID,
            sender_is_bot=False,
            config=cfg,
        )
