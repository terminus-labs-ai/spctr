"""Regression tests — spc-178 (event long-poll must not trip the 30 s request timeout).

GET /api/v1/events is a long poll: Zulip holds the connection open until an
event or a heartbeat (about every 90 s per https://zulip.com/api/get-events).
The transport previously bounded EVERY request with a 30 s total timeout, so
an idle poll timed out and the interface logged
``ERROR: Zulip event poll failed`` on every idle poll.

Fix under test: ``ZulipClient.get_events`` uses a per-request timeout above
the heartbeat interval (a named, documented constant), while every other
request keeps the ordinary ``request_timeout`` bound.

Proof shape (loopback only, no real seconds): a real aiohttp server on
127.0.0.1 holds /api/v1/events open for a configurable number of seconds and
then answers with the get-events contract shape; both timeouts are shrunk via
the public constructor so every assertion runs in well under a second. The
pre-fix shape is demonstrated by the constant itself: the default event-poll
timeout must exceed the default request timeout, so an idle poll that lasts
between the two bounds succeeds instead of tripping the request timeout.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from sr2_spectre.interfaces.zulip import (
    ZulipClient,
    ZulipConfig,
    ZulipCredentials,
    ZulipTransportError,
)
from sr2_spectre.interfaces.zulip.client import (
    _DEFAULT_EVENT_POLL_TIMEOUT,
    _DEFAULT_REQUEST_TIMEOUT,
)

# ---------------------------------------------------------------------------
# Synthetic identity (same shape as tests/test_zulip_client.py; no ~/.sr2).
# ---------------------------------------------------------------------------

BOT_EMAIL = "miranda-bot@zulip.example"
API_KEY = "synthetic-key-for-tests-0000"
CANONICAL_SITE = "https://normandy.invalid"  # TEST-NAP origin; never dialled
DIEGO_ID = 8
CHANNEL = 42

EVENTS_PATH = "/api/v1/events"
REGISTER_PATH = "/api/v1/register"


def zulip_test_credentials() -> ZulipCredentials:
    return ZulipCredentials(email=BOT_EMAIL, api_key=API_KEY, site=CANONICAL_SITE)


def config_for(origin_url: str, zuliprc: str) -> ZulipConfig:
    return ZulipConfig(
        zuliprc=zuliprc,
        api_url=origin_url,  # documented loopback/testing override
        allowed_user_ids=[DIEGO_ID],
        channels={CHANNEL: ""},
    )


def events_ok(queue_id: str, last_event_id: int) -> dict[str, Any]:
    """get-events contract: answers with the ``events`` ARRAY."""
    return {
        "queue_id": queue_id,
        "last_event_id": last_event_id,
        "events": [],
        "msg": "",
        "result": "success",
    }


def assert_no_secret(exc: BaseException) -> None:
    blob = f"{exc} {exc!r}"
    for marker in (API_KEY, BOT_EMAIL):
        assert marker not in blob, f"credential material {marker!r} in failure: {exc!r}"


# ---------------------------------------------------------------------------
# Loopback long-poll server: holds /api/v1/events open for ``hold`` seconds,
# then answers with the contract shape.
# ---------------------------------------------------------------------------


class _LongPollServer:
    def __init__(self, hold_seconds: float) -> None:
        self.hold = hold_seconds
        self.requests: list[dict[str, Any]] = []
        self.url = ""
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()

        async def handler(request: web.Request) -> web.StreamResponse:
            self.requests.append(
                {
                    "method": request.method,
                    "path": request.path,
                    "query": dict(request.query),
                    "headers": dict(request.headers),
                }
            )
            if request.path == EVENTS_PATH:
                await asyncio.sleep(self.hold)
            return web.json_response(events_ok("queue-spark", 3))

        app.router.add_route("*", "/{tail:.*}", handler)
        runner = web.AppRunner(app, handle_signals=False)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"  # type: ignore[union-attr]
        self._runner = runner

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


# ---------------------------------------------------------------------------
# AC1 — the per-poll timeout constant: named, above the heartbeat interval,
# strictly above the ordinary request timeout (the bug was the ordering).
# ---------------------------------------------------------------------------


class TestEventPollTimeoutConstant:
    def test_default_poll_timeout_exceeds_the_90s_heartbeat(self) -> None:
        # Zulip heartbeats about every 90 s (https://zulip.com/api/get-events):
        # a poll that answers at the heartbeat must never be cut off.
        assert _DEFAULT_EVENT_POLL_TIMEOUT > 90.0

    def test_poll_timeout_exceeds_the_default_request_timeout(self) -> None:
        # The bug: every request was bounded by the 30 s request timeout, so
        # an idle long poll (held open up to the ~90 s heartbeat) tripped it.
        # The poll bound must strictly exceed it so an idle poll between the
        # two values answers normally.
        assert _DEFAULT_EVENT_POLL_TIMEOUT > _DEFAULT_REQUEST_TIMEOUT
        assert _DEFAULT_REQUEST_TIMEOUT == 30.0

    def test_poll_timeout_stays_finite(self) -> None:
        # A genuinely stuck poll must still fail so the interface's
        # log-and-back-off path re-arms; the poll bound is a ceiling, not
        # infinity.
        assert 0 < _DEFAULT_EVENT_POLL_TIMEOUT <= 3600.0

    async def test_default_built_client_keeps_the_documented_bounds(
        self,
        tmp_path: Path,
    ) -> None:
        # No constructor overrides: the ordinary request bound stays 30 s and
        # the poll bound is the named constant, both visible through the
        # public defaults asserted above.
        zuliprc = tmp_path / "defaults.zuliprc"
        zuliprc.write_text(
            f"[zulip]\nemail = {BOT_EMAIL}\nkey = {API_KEY}\nsite = {CANONICAL_SITE}\n"
        )
        client = ZulipClient(
            credentials=zulip_test_credentials(),
            config=config_for("http://127.0.0.1:1", str(zuliprc)),
        )
        try:
            assert client._timeout == _DEFAULT_REQUEST_TIMEOUT
            assert client._event_poll_timeout == _DEFAULT_EVENT_POLL_TIMEOUT
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# AC2 — a poll answering after the request timeout but before the poll
# timeout returns normally (no ZulipError). Pre-fix this raised
# ZulipTransportError at the request-timeout bound.
# ---------------------------------------------------------------------------


class TestIdlePollSurvivesTheRequestTimeout:
    async def test_idle_poll_between_the_two_timeouts_returns_events(
        self,
        tmp_path: Path,
    ) -> None:
        # request timeout 0.1 s < hold 0.3 s < poll timeout 0.6 s.
        # Pre-fix (poll bounded by request_timeout) the client gave up at
        # 0.1 s and raised ZulipTransportError; with the fix it holds to the
        # 0.3 s answer and returns the events array.
        server = _LongPollServer(hold_seconds=0.3)
        await server.start()
        zuliprc = tmp_path / "miranda.zuliprc"
        zuliprc.write_text(
            f"[zulip]\nemail = {BOT_EMAIL}\nkey = {API_KEY}\nsite = {CANONICAL_SITE}\n"
        )
        client = ZulipClient(
            credentials=zulip_test_credentials(),
            config=config_for(server.url, str(zuliprc)),
            request_timeout=0.1,
            event_poll_timeout=0.6,
        )
        try:
            loop = asyncio.get_running_loop()
            started = loop.time()
            response = await asyncio.wait_for(
                client.get_events("queue-spark", 2), timeout=5
            )
            elapsed = loop.time() - started
            # The answer must actually have waited out the hold — the client
            # must not have cut the poll at the 0.1 s request timeout.
            assert elapsed >= 0.25, (
                f"poll answered after {elapsed:.2f}s — the poll was cut at "
                "the request timeout instead of the event-poll timeout"
            )
            assert response["result"] == "success"
            assert response["events"] == []
        finally:
            await client.aclose()
            await server.stop()
            assert server.requests, "no request reached the server"
            call = server.requests[0]
            assert call["path"] == EVENTS_PATH
            assert call["query"] == {"queue_id": "queue-spark", "last_event_id": "2"}


# ---------------------------------------------------------------------------
# AC3 — a poll that exceeds the event-poll timeout still raises the existing
# ZulipTransportError path.
# ---------------------------------------------------------------------------


class TestPollTimeoutStillRaises:
    async def test_poll_exceeding_the_poll_timeout_raises_transport_error(
        self,
        tmp_path: Path,
    ) -> None:
        # hold 1.5 s (the server never answers) > poll timeout 0.25 s.
        # The existing failure path is unchanged: ZulipTransportError, no
        # credential in the message.
        server = _LongPollServer(hold_seconds=1.5)
        await server.start()
        zuliprc = tmp_path / "miranda3.zuliprc"
        zuliprc.write_text(
            f"[zulip]\nemail = {BOT_EMAIL}\nkey = {API_KEY}\nsite = {CANONICAL_SITE}\n"
        )
        client = ZulipClient(
            credentials=zulip_test_credentials(),
            config=config_for(server.url, str(zuliprc)),
            request_timeout=0.1,
            event_poll_timeout=0.25,
        )
        try:
            loop = asyncio.get_running_loop()
            started = loop.time()
            with pytest.raises(ZulipTransportError) as excinfo:
                await asyncio.wait_for(
                    client.get_events("queue-spark", 2), timeout=5
                )
            elapsed = loop.time() - started
            # It must be the POLL timeout that fired (≈0.25 s), not the
            # request timeout (0.1 s) — the fix gives the poll its own bound.
            assert elapsed >= 0.22, (
                f"failed after {elapsed:.2f}s — the request timeout, not the "
                "event-poll timeout, fired"
            )
            assert_no_secret(excinfo.value)
        finally:
            await client.aclose()
            await server.stop()


# ---------------------------------------------------------------------------
# AC4 — the explicit request_timeout still bounds every OTHER request.
# ---------------------------------------------------------------------------


class TestRequestTimeoutStillBoundsOtherRequests:
    async def test_non_event_request_times_out_at_the_request_timeout(
        self,
        tmp_path: Path,
    ) -> None:
        # register (POST) against a server that never answers. request_timeout
        # 0.5 s must cut it off well before the (much larger) event-poll
        # timeout could: the non-event bound is unchanged by the fix.
        release = asyncio.Event()

        async def never_answers(request: web.Request) -> web.StreamResponse:
            await release.wait()
            return web.json_response({"result": "success", "msg": ""})

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", never_answers)
        runner = web.AppRunner(app, handle_signals=False)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"  # type: ignore[union-attr]
        zuliprc = tmp_path / "miranda4.zuliprc"
        zuliprc.write_text(
            f"[zulip]\nemail = {BOT_EMAIL}\nkey = {API_KEY}\nsite = {CANONICAL_SITE}\n"
        )
        client = ZulipClient(
            credentials=zulip_test_credentials(),
            config=config_for(url, str(zuliprc)),
            request_timeout=0.5,
            event_poll_timeout=5.0,  # deliberately huge: must NOT bound register
        )
        try:
            loop = asyncio.get_running_loop()
            started = loop.time()
            with pytest.raises((ZulipTransportError, TimeoutError)) as excinfo:
                await asyncio.wait_for(client.register(), timeout=5)
            elapsed = loop.time() - started
            # Bounded by the 0.5 s request timeout, not the 5 s poll timeout.
            assert elapsed < 2.0, (
                f"register took {elapsed:.1f}s — the request timeout is not "
                "bounding non-event requests"
            )
            assert_no_secret(excinfo.value)
        finally:
            release.set()
            await client.aclose()
            await runner.cleanup()
