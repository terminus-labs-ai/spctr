"""Zulip credential record and the declared failure-boundary types.

This module holds only credential and error definitions. Transport behavior
(authenticated requests, events API, send-message) arrives with spc-124
(``Exercise Zulip transport through its public API``); nothing here performs
HTTP or model work.

The API key never appears in a URL, a log record, or an exception message:
credential failures name the path and the reason, never the contents. Values
are read through the raw (non-interpolating) view and the file is decoded
before parsing, so a ``%(...)`` marker in a value and invalid UTF-8 bytes
both surface as sanitized credential failures, never as a raw configparser
or codec error echoing the file.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass

_REQUIRED_FIELDS = ("email", "key", "site")


class ZulipCredentialError(Exception):
    """The zuliprc is missing, unreadable, unparseable, or incomplete.

    The message names the path and the reason only — never file contents,
    so a partial secret cannot leak through an exception or log line.
    """


class ZulipAuthError(Exception):
    """The server rejected the bot's credentials (generic reason only).

    Declared here for the transport slice (spc-124); raised there, never
    with secret material in the message.
    """


class ZulipTransportError(Exception):
    """A transport-level failure talking to the configured server.

    Declared here for the transport slice (spc-124); raised there.
    """


@dataclass(frozen=True)
class ZulipCredentials:
    """Bot identity loaded from a zuliprc (email, api key, canonical site)."""
    email: str
    api_key: str
    site: str


def load_zulip_credentials(path: str) -> ZulipCredentials:
    """Read a bot zuliprc (``[zulip]`` section: email, key, site).

    Failures are explicit and carry no file contents, so a partial secret
    can never leak through an exception message or log line. The file is
    opened directly rather than through ``ConfigParser.read`` because that
    helper swallows unreadable files (a missing file and a file you cannot
    read both come back as an empty result).
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except (OSError, UnicodeDecodeError):
        raise ZulipCredentialError(
            f"Zulip credential file could not be read: {path}"
        ) from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ZulipCredentialError(
            f"Zulip credential file is not valid UTF-8: {path}"
        ) from None
    try:
        # The raw, non-interpolating parser: a value such as
        # ``key = secret %(missing)s`` is a literal credential, not an
        # interpolation reference, so it is preserved verbatim.
        parser = configparser.RawConfigParser()
        parser.read_string(text)
    except configparser.Error:
        raise ZulipCredentialError(
            f"Malformed zuliprc file: {path}"
        ) from None
    if not parser.has_section("zulip"):
        raise ZulipCredentialError(
            f"zuliprc {path} has no [zulip] section"
        )
    missing = [
        field
        for field in _REQUIRED_FIELDS
        if not parser.get("zulip", field, fallback="").strip()
    ]
    if missing:
        raise ZulipCredentialError(
            f"zuliprc {path} is missing required field(s): "
            f"{', '.join(missing)}"
        )
    return ZulipCredentials(
        email=parser.get("zulip", "email").strip(),
        api_key=parser.get("zulip", "key").strip(),
        site=parser.get("zulip", "site").strip().rstrip("/"),
    )
