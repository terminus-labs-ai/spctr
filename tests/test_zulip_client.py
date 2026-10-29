"""Unit cage — spc-124 (Exercise Zulip transport through its public API, AC18).

AC18: "Public Zulip transport uses only the configured authenticated server,
treats queue IDs as opaque strings, reads the events array, JSON-encodes
exact-topic history narrows, and sends to the specified channel/topic.
Authentication, timeout and transport failures are explicit without secret
disclosure."

Proof shape. Public client methods are driven against a REAL aiohttp server
on loopback (port 0): URLs, headers, query strings and bodies are observed
on the wire by the responding server — the HTTP boundary is the only
substituted seam. Each capability is exercised with protocol-shaped
responses from the official contracts the spec links: register answers with
an opaque STRING queue_id and numeric last_event_id; get-events answers with
the ``events`` ARRAY (never ``messages``); history narrows are JSON-encoded
operator/operand objects (channel operand = integer channel ID, topic
operand = the exact topic string); send answers with the numeric message
id. AC18's failure classes execute for real through a method: HTTP 401/403,
a never-answering server against a short request timeout, a connection
killed mid-response, a malformed success body. No credential material may
appear in any raised message or representation.

Server pinning. A second loopback "unrelated" server records requests; all
harness apps reject cookies. The transport follows NO redirects: an API
client that authenticates per request must not chase a server-supplied
Location off the configured origin (AC18's "no implicit redirect", spec
FR12 "Never send Zulip credentials to arbitrary linked hosts or
redirects"). Teardown audits: zero requests to the unrelated host, nothing
off /api/v1/, no credential in any request line, query or body.

The only seam pinned for construction: the public callable exported by
``sr2_spectre.interfaces.zulip`` (package level, alongside the accepted
spc-123 credential exports) as ``ZulipClient(credentials=…, config=…)`` —
the accepted ``ZulipCredentials`` record and the accepted ``ZulipConfig``
whose ``api_url`` is the documented loopback/testing origin override
(``docs/CONFIG-REFERENCE.md``). Module layout, extra optional keyword
arguments (a request timeout is probed for — "HTTP retry timing and module
layout … remain implementation choices"), and return shapes beyond the
assertions below stay free. The zuliprc site (``https://normandy.invalid``,
TEST-NAP) is dialled only by the tests that prove it is the default target.
"""

from __future__ import annotations

import asyncio
import base64
import json
import traceback
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl

import pytest
from aiohttp import web

from sr2_spectre.interfaces.zulip import (
    ZulipAuthError,
    ZulipClient,
    ZulipConfig,
    ZulipCredentials,
    ZulipTransportError,
)

# ---------------------------------------------------------------------------
# Synthetic identities (mirror spec Dependencies: Miranda bot user 9, Diego
# user 8). The credential markers below must never appear in any request
# line, query, body, error message or exception representation.
# ---------------------------------------------------------------------------

BOT_EMAIL = "miranda-bot@zulip.example"
API_KEY = "synthetic-key-for-tests-0000"
SECRET_MARKER = "secret-LEAK-CANARY-9999"
CANONICAL_SITE = "https://normandy.invalid"  # TEST-NAP origin; never dialled

CHANNEL = 42
OTHER_CHANNEL = 43
BOT_ID = 9
DIEGO_ID = 8
TOPIC = "castle plans"
TRICKY_TOPIC = 'one "quoted", [bracket] é & =?%/next'

EXPECTED_BASIC = "Basic " + base64.b64encode(
    f"{BOT_EMAIL}:{API_KEY}".encode()
).decode()

REGISTER_PATH = "/api/v1/register"
EVENTS_PATH = "/api/v1/events"
MESSAGES_PATH = "/api/v1/messages"

_TIMEOUT_KEYWORDS: tuple[str, ...] = ("timeout", "request_timeout")


def write_zuliprc(path: Path) -> str:
    path.write_text(
        f"[zulip]\nemail = {BOT_EMAIL}\nkey = {API_KEY}\nsite = {CANONICAL_SITE}\n"
    )
    return str(path)


def zulip_test_credentials() -> ZulipCredentials:
    """The record the accepted loader parses from ``write_zuliprc``."""
    return ZulipCredentials(email=BOT_EMAIL, api_key=API_KEY, site=CANONICAL_SITE)


# ---------------------------------------------------------------------------
# Protocol-shaped replies (official contract shapes; fields verbatim)
# ---------------------------------------------------------------------------


class Reply:
    """Wire status plus a JSON body, raw body, or ``kill`` (sever the
    connection while answering — a real transport failure)."""

    def __init__(
        self,
        *,
        status: int = 200,
        json_body: Any = None,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
        kill: bool = False,
    ) -> None:
        self.status = status
        self.json_body = json_body
        self.raw_body = raw_body
        self.headers = headers or {}
        self.kill = kill


def register_ok(
    queue_id: Any = "b60adfdc-4c9b-4d0c-9f7b-3e2a1d0f4e5b",
    last_event_id: int = 3,
) -> Reply:
    """register-queue contract: string queue_id, numeric last_event_id."""
    return Reply(
        json_body={
            "queue_id": queue_id,
            "last_event_id": last_event_id,
            "result": "success",
            "msg": "",
        }
    )


def message_event(message_id: int, content: str) -> dict[str, Any]:
    return {
        "type": "message",
        "message": {
            "id": message_id,
            "sender_id": DIEGO_ID,
            "content": content,
            "timestamp": 1760000000 + message_id,
            "type": "stream",
            "stream_id": CHANNEL,
            "topic": TOPIC,
        },
    }


def events_ok(events: list[dict[str, Any]], queue_id: str, last_event_id: int) -> Reply:
    """get-events contract: answers with the ``events`` ARRAY."""
    return Reply(
        json_body={
            "queue_id": queue_id,
            "last_event_id": last_event_id,
            "events": events,
            "msg": "",
            "result": "success",
        }
    )


HISTORY_OK = Reply(
    json_body={
        "messages": [
            {
                "id": 101,
                "sender_id": BOT_ID,
                "content": "later",
                "timestamp": 1760000001,
                "type": "stream",
                "stream_id": CHANNEL,
                "topic": TOPIC,
            },
            {
                "id": 100,
                "sender_id": DIEGO_ID,
                "content": "earlier",
                "timestamp": 1760000000,
                "type": "stream",
                "stream_id": CHANNEL,
                "topic": TOPIC,
            },
        ],
        "anchor": "newest",
        "found_anchor": 101,
        "found_newest": True,
        "found_oldest": True,
        "result": "success",
        "msg": "",
    }
)

SEND_OK = Reply(json_body={"id": 42, "result": "success", "msg": ""})


def auth_failure() -> Reply:
    return Reply(
        status=401,
        json_body={
            "result": "error",
            "code": "INVALID_API_KEY",
            "msg": "Invalid API key",
        },
    )


# ---------------------------------------------------------------------------
# Loopback harness — the only substituted boundary is the Zulip HTTP server.
# ---------------------------------------------------------------------------


class _Script:
    """Replies keyed by (METHOD, path). ``on(method, path, *replies)`` queues
    replies consumed one per matching request; the final queued reply
    repeats. Unmatched requests get the 500 fallback (which tests expect the
    client to surface as an explicit failure, never as success)."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], list[Reply]] = {}
        self.fallback = Reply(
            status=500,
            json_body={
                "result": "error",
                "code": "INTERNAL_SERVER_ERROR",
                "msg": "unhandled path in test",
            },
        )

    def on(self, method: str, path: str, *replies: Reply) -> None:
        self.entries[(method.upper(), path)] = list(replies)

    def next(self, method: str, path: str) -> Reply:
        queue = self.entries.get((method.upper(), path))
        if not queue:
            return self.fallback
        return queue.pop(0) if len(queue) > 1 else queue[0]


def _cookie_rejecting_app() -> web.Application:
    """App that never accepts or stores cookies: nothing can ride a redirect
    or a later request from the configured server to another host."""
    app = web.Application()

    @web.middleware
    async def no_cookies(request: web.Request, handler):  # type: ignore[no-untyped-def]
        if request.cookies:
            return web.Response(status=400, text="cookies are rejected here")
        response = await handler(request)
        if "Set-Cookie" in response.headers:
            response = web.Response(status=response.status, body=response.body)
        return response

    app.middlewares.append(no_cookies)
    return app


class TransportHarness:
    """Configured-server stand-in plus an unrelated host; both record every
    request that reaches them."""

    def __init__(self) -> None:
        self.zulip = _Script()
        self.unrelated = _Script()
        self.zulip_requests: list[dict[str, Any]] = []
        self.unrelated_requests: list[dict[str, Any]] = []
        self.zulip_url = ""
        self.unrelated_url = ""
        self._runners: list[web.AppRunner] = []

    async def start(self) -> None:
        for name, script, sink in (
            ("zulip", self.zulip, self.zulip_requests),
            ("unrelated", self.unrelated, self.unrelated_requests),
        ):
            app = _cookie_rejecting_app()
            app.router.add_route("*", "/{tail:.*}", self._handler(script, sink))
            runner = web.AppRunner(app, handle_signals=False)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
            setattr(self, f"{name}_url", f"http://127.0.0.1:{port}")
            self._runners.append(runner)

    def _handler(self, script: _Script, sink: list[dict[str, Any]]):
        async def handler(request: web.Request) -> web.StreamResponse:
            body = await request.read()
            sink.append(
                {
                    "method": request.method,
                    "path": request.path,
                    "raw_path": request.raw_path,
                    "query": dict(request.query),
                    "headers": dict(request.headers),
                    "body_text": body.decode("utf-8", "replace"),
                }
            )
            reply = script.next(request.method, request.path)
            if reply.kill:
                # Real transport failure; the harness reaps its sockets no
                # matter what the client did mid-response.
                raise RuntimeError("simulated connection drop")
            if reply.raw_body is not None:
                return web.Response(
                    status=reply.status, body=reply.raw_body, headers=reply.headers
                )
            return web.json_response(
                reply.json_body, status=reply.status, headers=reply.headers
            )

        return handler

    async def stop(self) -> None:
        for runner in self._runners:
            await runner.cleanup()

    # -- wire facts ----------------------------------------------------------

    def calls(self, method: str, path: str) -> list[dict[str, Any]]:
        return [
            r
            for r in self.zulip_requests
            if r["method"] == method.upper() and r["path"] == path
        ]

    def last(self, method: str, path: str) -> dict[str, Any]:
        found = self.calls(method, path)
        assert found, (
            f"no {method} {path} was recorded; saw "
            f"{[(r['method'], r['raw_path']) for r in self.zulip_requests]!r}"
        )
        return found[-1]

    def audit(self) -> None:
        if self.unrelated_requests:
            pytest.fail(
                "an authenticated request reached the unrelated host: "
                f"{[(r['method'], r['raw_path']) for r in self.unrelated_requests]!r}"
            )
        for req in self.zulip_requests:
            if not req["path"].startswith("/api/v1/"):
                pytest.fail(f"request escaped /api/v1/: {req['raw_path']!r}")
            blob = "\n".join(
                (req["raw_path"], json.dumps(req["query"]), req["body_text"])
            )
            for marker in (API_KEY, BOT_EMAIL, SECRET_MARKER):
                if marker in blob:
                    pytest.fail(
                        f"credential material {marker!r} sent in a URL/query/body: "
                        f"{req['method']} {req['raw_path']}"
                    )


@pytest.fixture
async def harness():
    h = TransportHarness()
    await h.start()
    try:
        yield h
    finally:
        await h.stop()
        h.audit()


# ---------------------------------------------------------------------------
# Client construction — the single pinned seam (see module docstring). The
# public client must also offer SOME way to close its transport; teardown
# uses it best-effort and stays silent if absent.
# ---------------------------------------------------------------------------


def _accepts_timeout_keyword() -> str | None:
    """Which optional request-timeout keyword the constructor accepts.

    A client bound to port 1 performs no I/O at construction, so the probe
    is safe; any construction failure means the candidate was rejected.
    """

    def probe(candidate: str | None) -> bool:
        extra: dict[str, Any] = {} if candidate is None else {candidate: 2.0}
        try:
            ZulipClient(
                credentials=zulip_test_credentials(),
                config=ZulipConfig(
                    zuliprc="/nonexistent.zuliprc",
                    api_url="http://127.0.0.1:1",
                    allowed_user_ids=[DIEGO_ID],
                    channels=[CHANNEL],
                ),
                **extra,
            )
            return True
        except Exception:
            return False

    if not probe(None):
        # Constructor rejects even the pinned two-kwarg construction; every
        # test below will fail on that directly. Do not add noise here.
        return None
    for candidate in _TIMEOUT_KEYWORDS:
        if probe(candidate):
            return candidate
    return None


async def _close_client(client: Any) -> None:
    for closer_name in ("aclose", "close"):
        closer = getattr(client, closer_name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass
        return


@pytest.fixture
async def client_factory():
    """The pinned public constructor; every built client is closed after the
    test. ``timeout_kw`` = accepted optional request-timeout keyword, or
    ``None`` when the constructor declares none (the timeout test then
    demands one)."""

    created: list[Any] = []

    def _make(**kwargs: Any) -> Any:
        client = ZulipClient(**kwargs)
        created.append(client)
        return client

    _make.timeout_kw = _accepts_timeout_keyword()  # type: ignore[attr-defined]
    yield _make
    for client in created:
        await _close_client(client)


@pytest.fixture
def make_client(client_factory):
    """Client aimed at an explicit ZulipConfig, credentials as loaded.
    ``timeout_kw`` (the accepted optional request-timeout keyword) rides
    along for tests that need to bound a call."""

    def _make(config: ZulipConfig, **extra: Any) -> Any:
        return client_factory(
            credentials=zulip_test_credentials(), config=config, **extra
        )

    _make.timeout_kw = client_factory.timeout_kw  # type: ignore[attr-defined]
    return _make


@pytest.fixture
def synthetic_zuliprc(tmp_path: Path) -> str:
    return write_zuliprc(tmp_path / "miranda.zuliprc")


def config_for(origin_url: str, zuliprc: str, **overrides: Any) -> ZulipConfig:
    base: dict[str, Any] = {
        "zuliprc": zuliprc,
        "api_url": origin_url,  # documented loopback/testing override
        "allowed_user_ids": [DIEGO_ID],
        "channels": [CHANNEL, OTHER_CHANNEL],
    }
    base.update(overrides)
    return ZulipConfig(**base)


# ---------------------------------------------------------------------------
# Helpers over recorded wire facts (return shapes stay implementation-free)
# ---------------------------------------------------------------------------


async def call_maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value


def queue_id_of(result: Any) -> Any:
    if isinstance(result, tuple):
        return result[0]
    if isinstance(result, dict) and "queue_id" in result:
        return result["queue_id"]
    return result


def last_event_id_of(result: Any) -> Any:
    if isinstance(result, tuple):
        return result[1]
    if isinstance(result, dict):
        return result.get("last_event_id", 0)
    return 0


def events_of(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        assert "events" in result, f"events result lost the events array: {result!r}"
        return result["events"]
    assert isinstance(result, list), (
        f"unexpected get_events result type: {type(result).__name__}"
    )
    return result


def messages_of(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        assert "messages" in result, f"history result lost messages: {result!r}"
        return result["messages"]
    assert isinstance(result, list), (
        f"unexpected history result type: {type(result).__name__}"
    )
    return result


def parsed_narrow(call: dict[str, Any]) -> Any:
    raw = call["query"].get("narrow")
    assert raw is not None, f"no narrow parameter recorded: {call['query']!r}"
    return json.loads(raw)


def form_body(call: dict[str, Any]) -> dict[str, str]:
    return dict(parse_qsl(call["body_text"], keep_blank_values=True))


def assert_authenticated_via_basic(call: dict[str, Any]) -> None:
    """Authentication rides in the Authorization header — HTTP Basic per the
    official docs — and never in the URL/query/body (audited separately)."""
    assert call["headers"].get("Authorization") == EXPECTED_BASIC, (
        "expected HTTP Basic with the zuliprc identity; got "
        f"{call['headers'].get('Authorization')!r}"
    )


def assert_no_secret(err: BaseException) -> None:
    blob = str(err) + repr(err)
    for marker in (API_KEY, SECRET_MARKER, EXPECTED_BASIC):
        assert marker not in blob, f"failure disclosed {marker!r}: {blob!r}"


def assert_formatted_traceback_has_no_secret(err: BaseException) -> None:
    """No credential marker in the traceback an operator actually reads.

    A clean top-level exception is not enough: Python renders
    ``__cause__``/``__context__`` into the formatted traceback, so a
    secretless exception chained to a secret-carrying one still discloses
    the credential in logs and on the console. Rendering is what the
    operator sees, so rendering is what must stay clean — the internal
    chain shape, the wording and the file paths are free.
    """
    rendered = "".join(
        traceback.format_exception(type(err), err, err.__traceback__)
    )
    assert rendered, "formatted traceback is empty — nothing was rendered"
    for marker in (API_KEY, SECRET_MARKER, EXPECTED_BASIC):
        assert marker not in rendered, (
            f"the operator-visible traceback disclosed {marker!r}:\n{rendered}"
        )


# ---------------------------------------------------------------------------
# AC18 — the configured authenticated server, and nothing else
# ---------------------------------------------------------------------------


class TestConfiguredServerOnly:
    async def test_requests_reach_the_configured_origin_as_authenticated_api_v1_calls(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        """Origin membership is wire-level: this server exists only at
        ``config.api_url``; the unrelated-host audit (harness fixture)
        proves no request went anywhere else."""
        harness.zulip.on("POST", REGISTER_PATH, register_ok())
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        await call_maybe_await(client.register())

        call = harness.last("POST", REGISTER_PATH)
        assert_authenticated_via_basic(call)

    async def test_without_api_url_override_the_zuliprc_site_is_the_default_target(
        self, make_client: Callable[..., Any], tmp_path: Path
    ) -> None:
        """``api_url=None``: the zuliprc site becomes the server.
        normandy.invalid (TEST-NAP) cannot complete a connection, so the
        call must fail explicitly and promptly — never silently succeed or
        hang — and the failure carries no credential."""
        path = write_zuliprc(tmp_path / "site.zuliprc")
        config = ZulipConfig(
            zuliprc=path, allowed_user_ids=[DIEGO_ID], channels=[CHANNEL]
        )
        client = make_client(config)
        with pytest.raises(
            (ZulipAuthError, ZulipTransportError, TimeoutError)
        ) as excinfo:
            await asyncio.wait_for(client.register(), timeout=20)
        assert_no_secret(excinfo.value)

    async def test_redirect_to_the_unrelated_host_is_not_followed(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        """A 302 (toward the unrelated host) must not move the authenticated
        conversation: the transport follows no redirects, so the unrelated
        host receives nothing and the redirect surfaces as an explicit
        failure."""
        harness.zulip.on(
            "POST",
            REGISTER_PATH,
            Reply(
                status=302,
                json_body={"result": "error", "code": "REDIRECT", "msg": "moved"},
                headers={"Location": f"{harness.unrelated_url}/api/v1/register"},
            ),
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        with pytest.raises((ZulipAuthError, ZulipTransportError)) as excinfo:
            await asyncio.wait_for(client.register(), timeout=15)
        assert_no_secret(excinfo.value)

        assert harness.unrelated_requests == [], (
            "the transport followed a redirect off the configured server"
        )
        for req in harness.zulip_requests:
            assert "Cookie" not in req["headers"]


# ---------------------------------------------------------------------------
# AC18 — queue IDs are opaque strings
# ---------------------------------------------------------------------------


class TestOpaqueQueueId:
    async def test_register_returns_the_exact_opaque_string_queue_id(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on("POST", REGISTER_PATH, register_ok(queue_id="abc-123-def"))
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        result = await call_maybe_await(client.register())

        assert queue_id_of(result) == "abc-123-def"

    async def test_guid_like_queue_id_survives_unchanged(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on(
            "POST",
            REGISTER_PATH,
            register_ok(queue_id="b60adfdc-4c9b-4d0c-9f7b-3e2a1d0f4e5b"),
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        result = await call_maybe_await(client.register())

        assert queue_id_of(result) == "b60adfdc-4c9b-4d0c-9f7b-3e2a1d0f4e5b"

    async def test_queue_id_is_round_tripped_to_get_events_verbatim(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        """What register answered must come back byte-identical: any
        int()/str-rewrite of the ID would change the query parameter."""
        opaque = "550e8400-e29b-41d4-a716-446655440000"
        harness.zulip.on("POST", REGISTER_PATH, register_ok(queue_id=opaque))
        harness.zulip.on(
            "GET", EVENTS_PATH, events_ok([], queue_id=opaque, last_event_id=3)
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        result = await call_maybe_await(client.register())

        await call_maybe_await(
            client.get_events(queue_id_of(result), last_event_id_of(result))
        )

        assert harness.last("GET", EVENTS_PATH)["query"].get("queue_id") == opaque


# ---------------------------------------------------------------------------
# AC18 — reads the events array
# ---------------------------------------------------------------------------


class TestEventsArray:
    async def test_get_events_returns_the_events_array_content(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        opaque = "queue-opaque-1"
        harness.zulip.on(
            "GET",
            EVENTS_PATH,
            events_ok(
                [
                    message_event(101, "hello from Diego"),
                    message_event(102, "and again"),
                ],
                queue_id=opaque,
                last_event_id=5,
            ),
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        events = events_of(await call_maybe_await(client.get_events(opaque, 4)))

        assert [e["message"]["id"] for e in events] == [101, 102]
        call = harness.last("GET", EVENTS_PATH)
        assert call["query"].get("queue_id") == opaque
        assert call["query"].get("last_event_id") == "4"

    async def test_empty_events_array_yields_no_events(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on(
            "GET", EVENTS_PATH, events_ok([], queue_id="q1", last_event_id=7)
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        events = events_of(await call_maybe_await(client.get_events("q1", 6)))

        assert events == []

    async def test_events_are_read_from_the_events_key_not_messages(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        """Protocol fidelity: the events live under ``events``. A client
        reading the get-messages-style ``messages`` key would silently drop
        Diego's event — and no fallback may paper over the protocol."""
        opaque = "queue-opaque-2"
        harness.zulip.on(
            "GET",
            EVENTS_PATH,
            Reply(
                json_body={
                    "queue_id": opaque,
                    "last_event_id": 9,
                    "events": [message_event(101, "only in events")],
                    "messages": [],
                    "msg": "",
                    "result": "success",
                }
            ),
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        events = events_of(await call_maybe_await(client.get_events(opaque, 8)))

        assert [e["message"]["id"] for e in events] == [101]


# ---------------------------------------------------------------------------
# AC18 — JSON-encoded exact-topic history narrows
# ---------------------------------------------------------------------------


class TestExactTopicHistoryNarrow:
    async def test_narrow_is_json_encoded_channel_and_topic_operators(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on("GET", MESSAGES_PATH, HISTORY_OK)
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        await call_maybe_await(client.get_topic_messages(CHANNEL, TOPIC))

        narrow = parsed_narrow(harness.last("GET", MESSAGES_PATH))
        assert isinstance(narrow, list) and len(narrow) == 2, narrow
        terms = {(t["operator"], t["operand"]) for t in narrow}
        assert terms == {("channel", CHANNEL), ("topic", TOPIC)}, (
            "exact-topic retrieval requires a narrow of exactly the channel "
            "operator with the integer channel ID and the topic operator "
            "with the exact topic string (official construct-narrow "
            "contract; the spec narrows 'by the configured channel and that "
            "topic') — the 'stream' alias is NOT required"
        )

    async def test_topic_with_json_and_url_metacharacters_survives_the_round_trip(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        """Quotes, commas, brackets, slashes and non-ASCII in the topic must
        arrive exactly — hand-rolled quoting (e.g. Python repr) corrupts
        these."""
        harness.zulip.on("GET", MESSAGES_PATH, HISTORY_OK)
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        await call_maybe_await(client.get_topic_messages(OTHER_CHANNEL, TRICKY_TOPIC))

        narrow = parsed_narrow(harness.last("GET", MESSAGES_PATH))
        by_op = {t["operator"]: t["operand"] for t in narrow}
        assert by_op["channel"] == OTHER_CHANNEL
        assert by_op["topic"] == TRICKY_TOPIC

    async def test_history_answer_surfaces_the_server_messages(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on("GET", MESSAGES_PATH, HISTORY_OK)
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        messages = messages_of(
            await call_maybe_await(client.get_topic_messages(CHANNEL, TOPIC))
        )

        assert len(messages) == 2
        assert {m["id"] for m in messages} == {100, 101}
        assert all(m["topic"] == TOPIC for m in messages)
        # chronological order/anchoring/pagination is spc-125's contract


# ---------------------------------------------------------------------------
# AC18 — sends to the specified channel/topic
# ---------------------------------------------------------------------------


class TestSendToSpecifiedDestination:
    async def test_send_targets_the_given_channel_and_topic(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on("POST", MESSAGES_PATH, SEND_OK)
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        sent_id = await call_maybe_await(
            client.send_message(
                type="stream", to=CHANNEL, topic=TOPIC, content="Miranda replies here"
            )
        )

        call = harness.last("POST", MESSAGES_PATH)
        assert_authenticated_via_basic(call)
        body = form_body(call)
        assert body.get("type") == "stream"
        assert body.get("to") == str(CHANNEL)
        assert body.get("topic") == TOPIC
        assert body.get("content") == "Miranda replies here"
        assert str(sent_id) == "42"

    async def test_send_escapes_destination_and_content_metacharacters(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on("POST", MESSAGES_PATH, SEND_OK)
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        await call_maybe_await(
            client.send_message(
                type="stream",
                to=OTHER_CHANNEL,
                topic=TRICKY_TOPIC,
                content="multi\nline = body & more",
            )
        )

        body = form_body(harness.last("POST", MESSAGES_PATH))
        assert body.get("to") == str(OTHER_CHANNEL)
        assert body.get("topic") == TRICKY_TOPIC
        assert body.get("content") == "multi\nline = body & more"


# ---------------------------------------------------------------------------
# AC18 — explicit authentication, timeout and transport failures
# ---------------------------------------------------------------------------


class TestExplicitFailures:
    async def test_http_401_raises_auth_failure_without_the_api_key(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on("POST", REGISTER_PATH, auth_failure())
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        with pytest.raises(ZulipAuthError) as excinfo:
            await call_maybe_await(client.register())
        assert_no_secret(excinfo.value)

    async def test_http_403_raises_auth_failure_without_the_api_key(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on(
            "POST",
            MESSAGES_PATH,
            Reply(
                status=403,
                json_body={
                    "result": "error",
                    "code": "NOT_AUTHORIZED",
                    "msg": "You are not allowed in this channel",
                },
            ),
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        with pytest.raises(ZulipAuthError) as excinfo:
            await call_maybe_await(
                client.send_message(
                    type="stream", to=CHANNEL, topic=TOPIC, content="denied"
                )
            )
        assert_no_secret(excinfo.value)

    async def test_timeout_is_explicit_through_a_public_method(
        self,
        client_factory: Callable[..., Any],
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    ) -> None:
        """A server that accepts the connection and never answers. With the
        client's request timeout the method must fail explicitly and
        promptly — not hang and not report success. The timeout arrives
        through the public constructor; this cut owns exposing one."""
        timeout_kw = client_factory.timeout_kw  # type: ignore[attr-defined]
        assert timeout_kw is not None, (
            "the public client constructor must expose a request-timeout "
            "keyword (tried: " + ", ".join(_TIMEOUT_KEYWORDS) + ")"
        )

        release = asyncio.Event()

        async def never_answers(request: web.Request) -> web.StreamResponse:
            await release.wait()
            return web.json_response({"result": "success", "msg": ""})

        app = _cookie_rejecting_app()
        app.router.add_route("*", "/{tail:.*}", never_answers)
        runner = web.AppRunner(app, handle_signals=False)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        stall_url = (
            f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"  # type: ignore[union-attr]
        )
        try:
            client = make_client(
                config_for(stall_url, synthetic_zuliprc), **{timeout_kw: 0.5}
            )
            loop = asyncio.get_running_loop()
            started = loop.time()
            with pytest.raises(
                (ZulipTransportError, TimeoutError)
            ) as excinfo:
                # Outer bound generous: the assertion is that the client's
                # own timeout fires, not that it fires instantly.
                await asyncio.wait_for(
                    client.get_events("queue-under-timeout", 0), timeout=10
                )
            elapsed = loop.time() - started
            assert elapsed < 10.0, f"client hung {elapsed:.1f}s on a stalled server"
            assert_no_secret(excinfo.value)
        finally:
            release.set()
            await runner.cleanup()

    async def test_malformed_success_body_raises_transport_failure(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        """A 200 whose body is not usable JSON is an explicit transport
        failure, never a hang, a crash in JSON internals, or a success."""
        harness.zulip.on(
            "GET", EVENTS_PATH, Reply(raw_body=b'{"result":"succ', status=200)
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        with pytest.raises(ZulipTransportError) as excinfo:
            await asyncio.wait_for(client.get_events("queue-bad-body", 0), timeout=15)
        assert_no_secret(excinfo.value)

    async def test_unreachable_default_site_fails_fast_under_its_timeout(
        self, make_client: Callable[..., Any], tmp_path: Path
    ) -> None:
        """The default target (zuliprc site, TEST-NAP) cannot complete a
        connection. With a short request timeout the client must not sit
        inside a long-poll window: the explicit failure arrives well before
        an events long-poll would, and carries no credential."""
        timeout_kw = getattr(make_client, "timeout_kw", None)
        assert timeout_kw is not None, (
            "the public client constructor must expose a request-timeout "
            "keyword (tried: " + ", ".join(_TIMEOUT_KEYWORDS) + ")"
        )
        path = write_zuliprc(tmp_path / "site.zuliprc")
        config = ZulipConfig(
            zuliprc=path, allowed_user_ids=[DIEGO_ID], channels=[CHANNEL]
        )
        client = make_client(config, **{timeout_kw: 2.0})
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(
            (ZulipAuthError, ZulipTransportError, TimeoutError)
        ) as excinfo:
            await asyncio.wait_for(client.register(), timeout=10)
        elapsed = loop.time() - started
        assert elapsed < 8.0, (
            f"unreachable server took {elapsed:.1f}s with a 2s request "
            "timeout — the timeout is not bounding the call"
        )
        assert_no_secret(excinfo.value)

    async def test_connection_killed_mid_response_raises_transport_failure(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        harness.zulip.on("POST", MESSAGES_PATH, Reply(kill=True))
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        with pytest.raises(ZulipTransportError) as excinfo:
            await call_maybe_await(
                client.send_message(
                    type="stream", to=CHANNEL, topic=TOPIC, content="dropped"
                )
            )
        assert_no_secret(excinfo.value)

    async def test_error_body_embedding_the_key_never_leaks_through_the_failure(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,
    
    ) -> None:
        """Servers can echo request context into error bodies. A
        protocol-level failure (HTTP 200, result=error) whose msg embeds the
        key must still raise explicitly without disclosing it."""
        harness.zulip.on(
            "POST",
            MESSAGES_PATH,
            Reply(
                json_body={
                    "result": "error",
                    "code": "INVALID_INPUT",
                    "msg": f"server saw credential {API_KEY} and secret {SECRET_MARKER}",
                }
            ),
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        with pytest.raises((ZulipAuthError, ZulipTransportError)) as excinfo:
            await call_maybe_await(
                client.send_message(
                    type="stream", to=CHANNEL, topic=TOPIC, content="echo probe"
                )
            )
        assert_no_secret(excinfo.value)

    async def test_http_error_body_with_key_as_direct_string_never_leaks(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,

    ) -> None:
        """Repair regression (AC18, no secret disclosure): a protocol-level
        refusal (HTTP 200, ``result=error`` — the branch that actually
        parses the body) whose PROTOCOL ``code`` FIELD IS the synthetic API
        key string, while ``msg`` stays a safe diagnostic. A compliant
        client names the code field in its own failure text, so an
        implementation that redacts only ``msg`` still discloses the key
        here. The call fails explicitly and the key is absent from str(),
        repr() and the formatted traceback an operator reads. ``register()``
        carries this protocol-code-string case; the nested-object case below
        carries the structured variant."""
        harness.zulip.on(
            "POST",
            REGISTER_PATH,
            Reply(
                status=200,
                json_body={
                    "result": "error",
                    "code": API_KEY,
                    "msg": "safe diagnostic",
                },
            ),
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        with pytest.raises((ZulipAuthError, ZulipTransportError)) as excinfo:
            await call_maybe_await(client.register())
        assert_no_secret(excinfo.value)
        assert API_KEY not in f"{excinfo.value}"
        assert API_KEY not in f"{excinfo.value!r}"
        assert_formatted_traceback_has_no_secret(excinfo.value)

    async def test_http_error_body_with_key_nested_in_an_object_never_leaks(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,

    ) -> None:
        """Repair regression (AC18, no secret disclosure): the same protocol
        ``code``-field disclosure carried as a NESTED OBJECT instead of a
        string — HTTP 200 ``result=error``, so the body is parsed and the
        structured ``code`` is what a body-inspecting failure would
        interpolate. A server may serialize structured error data; no field
        of it may reach the failure's str(), repr() or the operator-visible
        formatted traceback, and the call still fails explicitly."""
        harness.zulip.on(
            "POST",
            REGISTER_PATH,
            Reply(
                status=200,
                json_body={
                    "result": "error",
                    "code": {"symbol": "INVALID_INPUT", "echoed": API_KEY},
                    "msg": "safe diagnostic",
                },
            ),
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        with pytest.raises((ZulipAuthError, ZulipTransportError)) as excinfo:
            await call_maybe_await(client.register())
        assert_no_secret(excinfo.value)
        assert API_KEY not in f"{excinfo.value}"
        assert API_KEY not in f"{excinfo.value!r}"
        assert_formatted_traceback_has_no_secret(excinfo.value)

    async def test_protocol_error_with_key_nested_in_the_code_never_leaks(
        self, harness: TransportHarness,
        make_client: Callable[..., Any],
        synthetic_zuliprc: str,

    ) -> None:
        """Repair regression (AC18, no secret disclosure): HTTP 200 with
        result=error whose ``code`` field is an object embedding the key —
        the shape an implementation that formats the code field into its own
        message would disclose through f-string expansion. No error-field
        material (nested or not) may appear in str(), repr() or the
        operator-visible formatted traceback, while the exported
        ZulipAuthError/ZulipTransportError raised through the public
        ``register()`` call is what makes the failure explicit."""
        harness.zulip.on(
            "POST",
            REGISTER_PATH,
            Reply(
                json_body={
                    "result": "error",
                    "code": {"name": "UNAUTHORIZED", "credential": API_KEY},
                    "msg": "authentication failed",
                }
            ),
        )
        client = make_client(config_for(harness.zulip_url, synthetic_zuliprc))
        with pytest.raises((ZulipAuthError, ZulipTransportError)) as excinfo:
            await call_maybe_await(client.register())
        assert_no_secret(excinfo.value)
        assert API_KEY not in f"{excinfo.value}"
        assert API_KEY not in f"{excinfo.value!r}"
        assert_formatted_traceback_has_no_secret(excinfo.value)
