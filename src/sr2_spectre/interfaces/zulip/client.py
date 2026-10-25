"""Zulip credentials, failure-boundary types, and the public HTTP transport.

Credentials and errors ship here alongside the ``ZulipClient`` transport
(spc-124): authenticated calls to the configured server only — queue
registration, event polling, exact-topic history and send. No CLI, retry
policy or history restoration lives here; later slices build on these
methods.

The API key never appears in a URL, a log record, or an exception message:
credential failures name the path and the reason, never the contents. Values
are read through the raw (non-interpolating) view and the file is decoded
before parsing, so a ``%(...)`` marker in a value and invalid UTF-8 bytes
both surface as sanitized credential failures, never as a raw configparser
or codec error echoing the file.
"""
from __future__ import annotations

import base64
import configparser
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp

from sr2_spectre.interfaces.zulip.config import ZulipConfig

_REQUIRED_FIELDS = ("email", "key", "site")
_API_PREFIX = "/api/v1"
_DEFAULT_REQUEST_TIMEOUT = 30.0
# GET /api/v1/events is a LONG POLL: the server holds the connection open
# until an event or a heartbeat, roughly every 90 s
# (https://zulip.com/api/get-events). The poll is therefore bounded by its
# own timeout above the heartbeat interval, so an idle poll never trips the
# 30 s request timeout and logs a spurious transport failure (spc-178).
# 300 s (5 min) stays finite — a genuinely stuck poll still fails and the
# interface's log-and-back-off path re-arms — while covering five heartbeat
# intervals of server silence. Every other request keeps ``request_timeout``.
_DEFAULT_EVENT_POLL_TIMEOUT = 300.0


class ZulipError(Exception):
    """Base for the declared Zulip failure boundaries below."""


class ZulipCredentialError(Exception):
    """The zuliprc is missing, unreadable, unparseable, or incomplete.

    The message names the path and the reason only — never file contents,
    so a partial secret cannot leak through an exception or log line.
    """


class ZulipAuthError(ZulipError):
    """The server rejected the bot's credentials (generic reason only).

    Declared here for the transport slice (spc-124); raised there, never
    with secret material in the message.
    """


class ZulipTransportError(ZulipError):
    """A transport-level failure talking to the configured server.

    Declared here for the transport slice (spc-124); raised there.
    """


@dataclass(frozen=True)
class ZulipCredentials:
    """Bot identity loaded from a zuliprc (email, api key, canonical site)."""
    email: str
    api_key: str
    site: str


# Accepted zuliprc sections, in priority order (spc-177): ``[api]`` is the
# standard format Zulip itself downloads; ``[zulip]`` is the historical
# workaround. When both are present, ``[api]`` wins.
_ACCEPTED_SECTIONS = ("api", "zulip")


def load_zulip_credentials(path: str) -> ZulipCredentials:
    """Read a bot zuliprc (``[api]`` or ``[zulip]`` section: email, key, site).

    Zulip's own downloaded zuliprcs use an ``[api]`` section; the historical
    workaround used ``[zulip]``. Both are accepted, with ``[api]`` preferred
    when both are present. Failures are explicit and carry no file contents,
    so a partial secret can never leak through an exception message or log
    line. The file is opened directly rather than through
    ``ConfigParser.read`` because that helper swallows unreadable files (a
    missing file and a file you cannot read both come back as an empty
    result).
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
    section = next(
        (name for name in _ACCEPTED_SECTIONS if parser.has_section(name)),
        None,
    )
    if section is None:
        raise ZulipCredentialError(
            f"zuliprc {path} has no accepted section "
            f"({', '.join(f'[{name}]' for name in _ACCEPTED_SECTIONS)})"
        )
    missing = [
        field
        for field in _REQUIRED_FIELDS
        if not parser.get(section, field, fallback="").strip()
    ]
    if missing:
        raise ZulipCredentialError(
            f"zuliprc {path} is missing required field(s) in "
            f"[{section}]: {', '.join(missing)}"
        )
    return ZulipCredentials(
        email=parser.get(section, "email").strip(),
        api_key=parser.get(section, "key").strip(),
        site=parser.get(section, "site").strip().rstrip("/"),
    )


class ZulipClient:
    """Public Zulip transport over the official REST API (spc-124, AC18).

    Every call goes to ONE explicitly configured origin: ``config.api_url``
    when set (loopback/testing override), else the zuliprc ``site``. The bot
    authenticates with HTTP Basic (email, key) on each request; redirects
    are never followed (``max_redirects=0``), so a server-supplied Location
    cannot carry the credentials to another host.

    Failure boundary: HTTP 401/403 and protocol-level ``result: error``
    answers raise :class:`ZulipAuthError`; timeouts, unreachable servers,
    severed connections and malformed success bodies raise
    :class:`ZulipTransportError`. Neither message ever contains the API key
    or a server body — servers can echo request context into error text, so
    the body is read for status classification only and then dropped.

    Args:
        credentials: The bot identity from :func:`load_zulip_credentials`.
        config: The ``zulip`` config block; only ``api_url`` matters here.
        request_timeout: Seconds that bound each NON-event request.
        event_poll_timeout: Seconds that bound the events long poll
            (GET /api/v1/events). Defaults to
            :data:`_DEFAULT_EVENT_POLL_TIMEOUT`, above Zulip's ~90 s
            heartbeat interval (https://zulip.com/api/get-events) so an
            idle poll is never cut off; every other request keeps
            ``request_timeout``.
    """

    def __init__(
        self,
        *,
        credentials: ZulipCredentials,
        config: ZulipConfig,
        request_timeout: float | None = None,
        event_poll_timeout: float | None = None,
    ) -> None:
        self._origin = (config.api_url or credentials.site).rstrip("/")
        self._timeout = (
            _DEFAULT_REQUEST_TIMEOUT if request_timeout is None else request_timeout
        )
        # The events long poll gets its own bound. When the caller explicitly
        # sets request_timeout (e.g. a test shrinking every timeout), the poll
        # follows it so no call can hang longer than the test expects. When
        # neither timeout is set, the poll gets the longer default so an idle
        # poll never trips the 30 s request timeout (spc-178).
        if event_poll_timeout is not None:
            self._event_poll_timeout = event_poll_timeout
        elif request_timeout is not None:
            self._event_poll_timeout = request_timeout
        else:
            self._event_poll_timeout = _DEFAULT_EVENT_POLL_TIMEOUT
        self._auth_header = "Basic " + base64.b64encode(
            f"{credentials.email}:{credentials.api_key}".encode("utf-8")
        ).decode("ascii")
        self._session = aiohttp.ClientSession(
            headers={"Authorization": self._auth_header},
            trust_env=False,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._session.close()

    # -- public API ----------------------------------------------------------

    async def register(self) -> dict[str, Any]:
        """Register an event queue (POST /api/v1/register).

        Returns the protocol dict; ``queue_id`` is an OPAQUE STRING passed
        through untouched, ``last_event_id`` an int.
        """
        return await self._request(
            "POST", "/register", data={"events": json.dumps(["message"])}
        )

    async def get_events(
        self, queue_id: str, last_event_id: int = -1
    ) -> dict[str, Any]:
        """Poll the event queue (GET /api/v1/events).

        ``queue_id`` travels byte-for-byte as given; the answer's ``events``
        key holds the event array.

        The server holds this request open until an event or a heartbeat
        (about every 90 s, https://zulip.com/api/get-events), so it is
        bounded by the longer ``event_poll_timeout`` rather than the
        per-request ``request_timeout``; exceeding the poll timeout still
        raises :class:`ZulipTransportError` like any other transport
        failure.
        """
        return await self._request(
            "GET",
            "/events",
            query={"queue_id": queue_id, "last_event_id": last_event_id},
            timeout=self._event_poll_timeout,
        )

    async def get_topic_messages(
        self, channel_id: int, topic: str
    ) -> dict[str, Any]:
        """Fetch one topic's history (GET /api/v1/messages).

        The narrow is JSON-encoded per the construct-narrow contract: a
        ``channel`` operator with the integer channel ID and a ``topic``
        operator with the exact topic string. The answer's ``messages``
        key holds the protocol message list.
        """
        narrow = [
            {"operator": "channel", "operand": channel_id},
            {"operator": "topic", "operand": topic},
        ]
        return await self._request(
            "GET",
            "/messages",
            query={
                "narrow": json.dumps(narrow),
                "anchor": "newest",
                "num_before": 100,
                "num_after": 0,
            },
        )

    async def get_streams(self) -> dict[str, Any]:
        """List the realm's streams (GET /api/v1/streams).

        Returns the protocol dict; the answer's ``streams`` key holds the
        stream list. Reminder channel-name resolution (FR10) is the first
        consumer.
        """
        return await self._request("GET", "/streams")

    async def send_message(
        self,
        *,
        type: str = "stream",
        to: int,
        topic: str,
        content: str,
    ) -> Any:
        """Send a message (POST /api/v1/messages) to the given destination.

        Returns the new message ID (the protocol response's ``id``).
        """
        payload = await self._request(
            "POST",
            "/messages",
            data={"type": type, "to": to, "topic": topic, "content": content},
        )
        return payload["id"]

    async def send_direct(self, to: int, content: str) -> Any:
        """Send one private message (POST /api/v1/messages, type=private).

        The private-message destination is a JSON-encoded LIST of recipient
        user IDs (send-message contract), so the integer *to* is encoded
        here — the one place that encoding lives, and interface code never
        repeats it (spc-992). A private message has no topic; the empty
        topic field is sent for the realms that expect it. Returns the new
        message ID like :meth:`send_message`.
        """
        return await self.send_message(
            type="private", to=json.dumps([to]), topic="", content=content
        )

    # -- request core ----------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self._origin}{_API_PREFIX}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        try:
            async with self._session.request(
                method,
                url,
                data=data,
                # The events long poll passes its own bound; everything
                # else keeps the per-request timeout (spc-178).
                timeout=aiohttp.ClientTimeout(
                    total=self._timeout if timeout is None else timeout
                ),
                max_redirects=0,
                allow_redirects=False,
            ) as resp:
                status = resp.status
                raw = await resp.read()
            return self._classify(method, path, status, raw)
        except ZulipError:
            raise
        except (
            aiohttp.ClientError,
            TimeoutError,
            ConnectionError,
            OSError,
        ):
            # ClientError covers redirects refused via max_redirects=0.
            # Only the method/path is named — never the exception text,
            # which can carry request context.
            raise ZulipTransportError(
                f"Zulip request failed: {method} {_API_PREFIX}{path}"
            ) from None

    def _classify(
        self, method: str, path: str, status: int, raw_body: bytes
    ) -> dict[str, Any]:
        if status in (401, 403):
            raise ZulipAuthError(
                f"Zulip server rejected authentication (HTTP {status}): "
                f"{method} {_API_PREFIX}{path}"
            )
        if status >= 400:
            raise ZulipTransportError(
                f"Zulip server error (HTTP {status}): "
                f"{method} {_API_PREFIX}{path}"
            )
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ZulipTransportError(
                f"Zulip server returned a malformed response body: "
                f"{method} {_API_PREFIX}{path}"
            ) from None
        if not isinstance(payload, dict):
            raise ZulipTransportError(
                f"Zulip server returned a non-object response: "
                f"{method} {_API_PREFIX}{path}"
            )
        if payload.get("result") != "success":
            # Every field of a server error body is untrusted: msg may echo
            # request context (even the key) and code may be a string or a
            # nested object carrying it. No response field is interpolated
            # into the failure — only the method/path this client built.
            raise ZulipAuthError(
                f"Zulip server rejected the request: "
                f"{method} {_API_PREFIX}{path}"
            )
        return payload
