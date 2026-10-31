"""spc-1014 — Zulip Thinking placeholder, final edit and error fallback (AC7, AC8).

Case-table rows of specs/zulip-channel-access.md, one test (or one
parametrized case) each, named ``test_<row>_...``. The real
``ZulipInterface`` poll loop runs over a fake transport and a fake agent
whose Sessions stream scripted events; both write to one shared ordered
log, so a test can see whether the placeholder left before the Session
stream started. The fake transport answers the first message POST with ID
500, the next with 501, and so on. No model, no network.

Defaults unless a row says otherwise: ``thinking_message`` "⏳ Thinking...",
``empty_response_message`` "(No response generated)", ``error_message``
"Error: {error}", ``max_message_length`` 10000.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web

from sr2_spectre.events import AgentDone, AgentTextDelta, AgentToolStart
from sr2_spectre.interfaces.common import chunk_message
from sr2_spectre.interfaces.zulip import (
    ZulipAuthError,
    ZulipClient,
    ZulipConfig,
    ZulipInterface,
    ZulipTransportError,
)
from sr2_spectre.interfaces.zulip.client import ZulipCredentials
from sr2_spectre.resume import ResumeResult

DIEGO_ID = 8
BOT_ID = 9
CHANNEL = 2
TOPIC = "t"
THINKING = "⏳ Thinking..."

# A 120-character answer of short words, so chunking splits at spaces.
LONG_ANSWER = " ".join(f"w{i:03d}" for i in range(24)) + "!"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _FakeSession:
    """Streams the scripted outcome for each message content.

    ``script[content]`` is a list of answer fragments (one AgentTextDelta
    each) or an exception the stream raises after it starts.
    """

    async def restore_history(self) -> ResumeResult:
        return ResumeResult([], interrupted=False, restore_failed=False)

    def __init__(self, log: list, script: dict[str, Any]) -> None:
        self._log = log
        self._script = script
        self.context: Any = None

    def set_run_context(self, context: Any) -> None:
        self.context = context

    async def stream_message(self, text: str):
        self._log.append(("STREAM", text))
        outcome = self._script[text]
        if isinstance(outcome, BaseException):
            raise outcome
        yield AgentToolStart(tool_id="t1", name="terminal", input={})
        for fragment in outcome:
            yield AgentTextDelta(text=fragment)
        yield AgentDone(tool_calls_executed=1)

    def close(self) -> None:
        pass


class _FakeAgent:
    def __init__(self, log: list, script: dict[str, Any]) -> None:
        self.config = SimpleNamespace(agent=SimpleNamespace(name="miranda"))
        self._log = log
        self._script = script

    def create_session(self, frame_id: str) -> _FakeSession:
        return _FakeSession(self._log, self._script)

    def apply_config(self, config: Any) -> None:
        pass


class _FakeClient:
    """Scripted event batches; every send/edit attempt joins the shared log.

    ``fail_post(form)`` says whether a POST raises; ``fail_patch`` holds the
    message IDs whose edit raises. A failed attempt is logged with
    ``ok=False``.
    """

    def __init__(self, interface: Any, batches: list[list[dict]], log: list,
                 fail_post=lambda form: False, fail_patch: set[int] = frozenset()):
        self._interface = interface
        self._batches = list(batches)
        self._log = log
        self._fail_post = fail_post
        self._fail_patch = set(fail_patch)
        self._next_id = 500

    async def register(self) -> dict:
        return {"queue_id": "q", "last_event_id": 0, "user_id": BOT_ID,
                "full_name": "Miranda",
                "realm_users": [{"user_id": DIEGO_ID, "is_bot": False},
                                {"user_id": BOT_ID, "is_bot": True}]}

    async def get_events(self, queue_id: str, last_event_id: int) -> dict:
        if not self._batches:
            self._interface._running = False
            return {"events": []}
        messages = self._batches.pop(0)
        return {"events": [{"id": m["id"], "type": "message", "message": m}
                           for m in messages]}

    async def get_topic_messages(self, channel_id: int, topic: str) -> dict:
        return {"messages": []}

    async def send_message(self, **kwargs: Any) -> int:
        if self._fail_post(kwargs):
            self._log.append(("POST", kwargs, False))
            raise ZulipTransportError("Zulip request failed: POST /api/v1/messages")
        self._log.append(("POST", kwargs, True))
        message_id = self._next_id
        self._next_id += 1
        return message_id

    async def update_message(self, message_id: int, content: str) -> None:
        if message_id in self._fail_patch:
            self._log.append(("PATCH", message_id, content, False))
            raise ZulipAuthError("Zulip server rejected the request: PATCH")
        self._log.append(("PATCH", message_id, content, True))

    async def aclose(self) -> None:
        pass


_next_msg = [100]


def _msg(content: str, topic: str = TOPIC) -> dict:
    _next_msg[0] += 1
    return {"id": _next_msg[0], "type": "stream", "stream_id": CHANNEL,
            "display_recipient": "sandbox", "subject": topic,
            "sender_id": DIEGO_ID, "content": content}


def _post(content: str, topic: str = TOPIC, ok: bool = True) -> tuple:
    return ("POST", {"type": "stream", "to": CHANNEL, "topic": topic,
                     "content": content}, ok)


async def _drive(monkeypatch, batches: list[list[dict]], script: dict[str, Any],
                 **over: Any) -> list:
    """Run the real poll loop; return the shared ordered log."""
    import sr2_spectre.interfaces.zulip.interface as iface_mod

    fail_post = over.pop("fail_post", lambda form: False)
    fail_patch = over.pop("fail_patch", frozenset())
    log: list = []
    interface = ZulipInterface(ZulipConfig(zuliprc="/unused.zuliprc",
                                           users=[DIEGO_ID], **over))
    client = _FakeClient(interface, batches, log, fail_post, fail_patch)
    monkeypatch.setattr(iface_mod, "load_zulip_credentials",
                        lambda path: ZulipCredentials(
                            email="bot@test.local", api_key="k",
                            site="http://127.0.0.1:1"))
    monkeypatch.setattr(iface_mod, "ZulipClient", lambda **kw: client)
    agent = _FakeAgent(log, script)
    await interface.start(agent)
    try:
        await interface.run(agent)
    finally:
        await interface.stop()
    return log


# ---------------------------------------------------------------------------
# AC7 — placeholder before the model call; the answer replaces it by edit
# ---------------------------------------------------------------------------


async def test_AC7_a_placeholder_posted_in_topic_before_stream_starts(monkeypatch):
    log = await _drive(monkeypatch, [[_msg("hi")]], {"hi": ["done"]})
    assert log[0] == _post(THINKING)
    assert log.index(("STREAM", "hi")) == 1


async def test_AC7_b_answer_replaces_placeholder_by_edit(monkeypatch):
    log = await _drive(monkeypatch, [[_msg("hi")]], {"hi": ["do", "ne"]})
    assert log == [_post(THINKING), ("STREAM", "hi"),
                   ("PATCH", 500, "done", True)]


async def test_AC7_c_empty_answer_shows_empty_response_message(monkeypatch):
    log = await _drive(monkeypatch, [[_msg("hi")]], {"hi": []})
    assert log == [_post(THINKING), ("STREAM", "hi"),
                   ("PATCH", 500, "(No response generated)", True)]


async def test_AC7_d_long_answer_edits_first_chunk_then_posts_rest(monkeypatch):
    assert len(LONG_ANSWER) == 120
    chunks = chunk_message(LONG_ANSWER, 50)
    assert len(chunks) >= 3 and all(len(c) <= 50 for c in chunks)
    log = await _drive(monkeypatch, [[_msg("hi")]], {"hi": [LONG_ANSWER]},
                       max_message_length=50)
    assert log == [_post(THINKING), ("STREAM", "hi"),
                   ("PATCH", 500, chunks[0], True),
                   *[_post(chunk) for chunk in chunks[1:]]]


async def test_AC7_e_configured_thinking_message_is_the_placeholder(monkeypatch):
    log = await _drive(monkeypatch, [[_msg("hi")]], {"hi": ["done"]},
                       thinking_message="working…")
    assert log == [_post("working…"), ("STREAM", "hi"),
                   ("PATCH", 500, "done", True)]


# ---------------------------------------------------------------------------
# AC8 — fallbacks: failed post, failed edit, model error, loop survives
# ---------------------------------------------------------------------------


async def test_AC8_a_failed_placeholder_post_still_runs_and_posts_answer(monkeypatch):
    log = await _drive(monkeypatch, [[_msg("hi")]], {"hi": ["done"]},
                       fail_post=lambda form: form["content"] == THINKING)
    assert log == [_post(THINKING, ok=False), ("STREAM", "hi"), _post("done")]


async def test_AC8_b_failed_final_edit_posts_answer_as_new_messages(monkeypatch):
    chunks = chunk_message(LONG_ANSWER, 50)
    log = await _drive(monkeypatch, [[_msg("hi")]], {"hi": [LONG_ANSWER]},
                       max_message_length=50, fail_patch={500})
    assert log == [_post(THINKING), ("STREAM", "hi"),
                   ("PATCH", 500, chunks[0], False),
                   *[_post(chunk) for chunk in chunks]]


@pytest.mark.parametrize("template,expected", [
    pytest.param(None, "Error: boom", id="default-template"),
    pytest.param("Miranda failed ({error})", "Miranda failed (boom)",
                 id="configured-template"),
])
async def test_AC8_c_model_error_edits_placeholder_and_sends_nothing_else(
    monkeypatch, template, expected
):
    over = {} if template is None else {"error_message": template}
    log = await _drive(monkeypatch, [[_msg("hi")]],
                       {"hi": RuntimeError("boom")}, **over)
    assert log == [_post(THINKING), ("STREAM", "hi"),
                   ("PATCH", 500, expected, True)]


@pytest.mark.parametrize("first_outcome,first_failures", [
    pytest.param(RuntimeError("boom"), {}, id="model-error"),
    pytest.param(["first answer"],
                 {"fail_post": lambda form: form["topic"] == "t1"},
                 id="every-post-in-topic-1-fails"),
    pytest.param(["first answer"],
                 {"fail_patch": {500},
                  "fail_post": lambda form: form["content"] == "first answer"},
                 id="edit-and-fallback-post-fail"),
])
async def test_AC8_d_failure_in_message_1_never_stops_message_2(
    monkeypatch, first_outcome, first_failures
):
    script = {"one": first_outcome, "two": ["second answer"]}
    # Message 1 and 2 arrive in one batch; message 2 in its own topic, and a
    # third message in a later batch proves the poll loop kept running.
    log = await _drive(
        monkeypatch,
        [[_msg("one", topic="t1"), _msg("two", topic="t2")],
         [_msg("three", topic="t3")]],
        {**script, "three": ["third answer"]},
        **first_failures,
    )
    # Topics run concurrently, so entries of different topics interleave;
    # each topic's own entries keep their order.
    two = log.index(("STREAM", "two"))
    assert any(e[0] == "POST" and e[1]["topic"] == "t2" for e in log[:two])
    assert any(e[0] == "PATCH" and e[2:] == ("second answer", True)
               for e in log[two:])
    three = log.index(("STREAM", "three"))
    assert any(e[0] == "PATCH" and e[2:] == ("third answer", True)
               for e in log[three:])


async def test_AC8_d_model_error_batch_delivers_message_2_exactly(monkeypatch):
    log = await _drive(
        monkeypatch, [[_msg("one", topic="t1"), _msg("two", topic="t2")]],
        {"one": RuntimeError("boom"), "two": ["second answer"]},
    )
    assert log == [
        _post(THINKING, topic="t1"), ("STREAM", "one"),
        ("PATCH", 500, "Error: boom", True),
        _post(THINKING, topic="t2"), ("STREAM", "two"),
        ("PATCH", 501, "second answer", True),
    ]


# ---------------------------------------------------------------------------
# Transport: ZulipClient.update_message against a loopback endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
async def edit_endpoint():
    """Loopback endpoint for PATCH /api/v1/messages/{id}; records requests.

    Message 500 succeeds, 501 answers HTTP 500, 502 answers a protocol
    error (the realm's edit time limit, for example).
    """
    requests: list[tuple[str, str, dict]] = []

    async def edit(request: web.Request) -> web.Response:
        requests.append((request.method, request.path,
                         dict(await request.post())))
        message_id = request.match_info["message_id"]
        if message_id == "501":
            return web.json_response({"result": "error"}, status=500)
        if message_id == "502":
            return web.json_response({"result": "error",
                                      "msg": "edit time limit passed"})
        return web.json_response({"result": "success"})

    app = web.Application()
    app.router.add_patch("/api/v1/messages/{message_id}", edit)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:
        await runner.cleanup()


async def test_update_message_patches_message_content(edit_endpoint):
    url, requests = edit_endpoint
    client = ZulipClient(
        credentials=ZulipCredentials(email="bot@test.local", api_key="k",
                                     site="https://normandy.invalid"),
        config=ZulipConfig(zuliprc="/unused.zuliprc", api_url=url),
    )
    try:
        assert await client.update_message(500, "done") is None
        with pytest.raises(ZulipTransportError):
            await client.update_message(501, "x")
        with pytest.raises(ZulipAuthError):
            await client.update_message(502, "x")
    finally:
        await client.aclose()
    assert requests[0] == ("PATCH", "/api/v1/messages/500", {"content": "done"})
    assert len(requests) == 3


def test_working_indicator_config_defaults():
    config = ZulipConfig(zuliprc="/unused.zuliprc")
    assert config.max_message_length == 10000
    assert config.thinking_message == THINKING
    assert config.empty_response_message == "(No response generated)"
    assert config.error_message == "Error: {error}"


def test_chunking_is_one_implementation_shared_with_discord():
    from sr2_spectre.interfaces import common
    from sr2_spectre.interfaces.discord import handler

    assert handler.chunk_message is common.chunk_message
    assert handler.tail_for_stream is common.tail_for_stream
