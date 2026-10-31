"""Regression tests — spc-177 (accept Zulip's standard [api] zuliprc).

A zuliprc downloaded from Zulip's own settings page uses an ``[api]``
section with ``email``, ``key`` and ``site``; the historical workaround
format used a ``[zulip]`` section. ``load_zulip_credentials`` must accept
both (AC2), prefer ``[api]`` when both are present (AC3), and keep the
existing explicit, non-leaking failure behaviour (AC4).

Everything runs against synthetic data under ``tmp_path``: no network, no
real realm, no production zuliprc, no secret.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sr2_spectre.interfaces.zulip import (
    ZulipCredentialError,
    load_zulip_credentials,
)

BOT_EMAIL = "miranda-bot@zulip.example"
SITE = "https://normandy.example.com"
API_KEY = "synthetic-key-for-tests-0000"
SECRET = "synthetic-key-DO-NOT-LEAK-1234"


def load_expect_credential_error(path: Path) -> ZulipCredentialError:
    """Load a path that must fail and return the raised credential error."""
    with pytest.raises(ZulipCredentialError) as exc_info:
        load_zulip_credentials(str(path))
    return exc_info.value


class TestApiSection:
    def test_api_section_only_loads(self, tmp_path: Path):
        """AC1: the standard Zulip-downloaded format ([api]) loads with the
        same normalisation as [zulip] today."""
        path = tmp_path / "standard.zuliprc"
        path.write_text(
            f"[api]\nemail = {BOT_EMAIL}\nkey = {API_KEY}\nsite = {SITE}\n"
        )
        creds = load_zulip_credentials(str(path))
        assert creds.email == BOT_EMAIL
        assert creds.api_key == API_KEY
        assert creds.site == SITE

    def test_api_section_values_are_normalised(self, tmp_path: Path):
        """AC1: stripped values, site without a trailing / — identical to
        the [zulip] path."""
        path = tmp_path / "standard.zuliprc"
        path.write_text(
            f"[api]\n  email  =   {BOT_EMAIL}  \n"
            f"key = {API_KEY}\nsite = {SITE}/\n"
        )
        creds = load_zulip_credentials(str(path))
        assert creds.email == BOT_EMAIL
        assert creds.api_key == API_KEY
        assert creds.site == SITE

    def test_zulip_section_only_still_loads(self, tmp_path: Path):
        """AC2: backward compatibility with the existing [zulip] workaround
        file (covered by the spc-123 cage, pinned here as well)."""
        path = tmp_path / "legacy.zuliprc"
        path.write_text(
            f"[zulip]\nemail = {BOT_EMAIL}\nkey = {API_KEY}\nsite = {SITE}\n"
        )
        creds = load_zulip_credentials(str(path))
        assert creds.email == BOT_EMAIL
        assert creds.api_key == API_KEY
        assert creds.site == SITE

    def test_api_wins_when_both_sections_present(self, tmp_path: Path):
        """AC3: when both sections exist, [api] wins."""
        path = tmp_path / "both.zuliprc"
        path.write_text(
            f"[zulip]\nemail = legacy@example.invalid\n"
            f"key = legacy-key\nsite = https://legacy.example.invalid\n"
            f"[api]\nemail = {BOT_EMAIL}\nkey = {API_KEY}\nsite = {SITE}\n"
        )
        creds = load_zulip_credentials(str(path))
        assert creds.email == BOT_EMAIL
        assert creds.api_key == API_KEY
        assert creds.site == SITE

    def test_neither_section_fails_naming_accepted_sections(
        self, tmp_path: Path
    ):
        """AC4: missing both sections fails with a ZulipCredentialError whose
        message names the accepted sections and never the key value."""
        path = tmp_path / "other.zuliprc"
        path.write_text(f"[other]\nemail = {BOT_EMAIL}\nkey = {SECRET}\n")
        error = load_expect_credential_error(path)
        message = str(error)
        assert "[api]" in message
        assert "[zulip]" in message
        assert SECRET not in message
        assert SECRET not in repr(error)

    def test_missing_field_in_api_section_fails_naming_fields(
        self, tmp_path: Path
    ):
        """AC4: a required field missing from the chosen [api] section
        fails, naming the fields, never the key value."""
        path = tmp_path / "partial-api.zuliprc"
        path.write_text(f"[api]\nemail = {BOT_EMAIL}\nkey = {SECRET}\n")
        error = load_expect_credential_error(path)
        message = str(error)
        assert "site" in message
        assert SECRET not in message
        assert SECRET not in repr(error)

    def test_missing_field_in_zulip_section_still_fails(
        self, tmp_path: Path
    ):
        """AC4: the existing [zulip] incomplete-file behaviour is unchanged
        (the spc-123 fixture carries the secret; it must never surface)."""
        path = tmp_path / "partial-zulip.zuliprc"
        path.write_text(f"[zulip]\nemail = {BOT_EMAIL}\nkey = {SECRET}\n")
        error = load_expect_credential_error(path)
        message = str(error)
        assert "site" in message
        assert SECRET not in message
        assert SECRET not in repr(error)
