"""J1B (E2E) — spc-125: Connect Miranda's CLI to safe no-area Zulip chat.

Runs the REAL user entry point (``python -m spctr.cli <config>
--interface zulip --log-file …``) against substituted loopback Zulip and
model HTTP endpoints only.  Real: CLI, config resolution, Zulip interface,
Agent/Runtime/Session, SR2 pipeline, topic_area/session/input resolvers,
HTTP transport.  Substituted: the two external HTTP boundaries, synthetic
credentials and data — nothing else.

Journeys / AC:
- test_chat_routing_and_filters           → AC1 + AC16 (admission/isolation)
- test_same_topic_two_configured_channels → AC16 (per-channel identity/isolation)
- test_existing_topic_refusal             → AC14
- test_existing_topic_probe_failure_blocks_turn → AC14 (probe failure)
- test_supported_area_starts_and_routes  → AC2/16 (supported-area proof,
  replacing the temporary nonempty-area refusal of the pre-routing cut)

Model-request inspection goes through ``ZulipCliRun.model_messages()``, the
``messages`` arrays of the actual OpenAI-compatible request bodies.

Bot authorship (AC1): the fake server answers a successful send with the id
and ``sender_id`` of the created message (send-message contract), and its
only credential identity is the Miranda bot from the synthetic zuliprc.
The test additionally records the Basic username of every send request:
authorship is the authenticated request identity, never form metadata.

Discrimination: the assertions are written against the spec, so they would
fail on an unconditional refusal (no reply at all), on a global facade
(histories leaking between topics), on bot loops (model calls for bot
triggers, proven with a continuous 5-second idle window), on an allowlist
that masks the bot filter (the bot id is allowlisted while bot-marked), on
inferred/cwd area (marker leaks into no-area turns), on a failed history
probe falling through to a model turn, on queued cold-topic siblings being
treated as prior content, and on a startup that silently accepts a nonempty
area.
"""
from __future__ import annotations

import asyncio
import base64
import re
import sys
import time
import pytest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zulip_e2e_support import (  # noqa: E402
    AREA_MARKER,
    BOT_ID,
    CONFIGURED_CHANNEL,
    CWD_MARKER,
    DECOY_MARKER,
    DIEGO_ID,
    OTHER_ID,
    OWN_ID,
    REPLY_MARKER,
    SECOND_CONFIGURED_CHANNEL,
    SHARED_TOPIC,
    THINKING_MESSAGE,
    UNCONFIGURED_CHANNEL,
    Harness,
    ZulipCliRun,
)

BOT_SEND_USER = "bot@e2e.local"  # synthetic zuliprc identity

# AC16 (FR 9a, spc-1056).  Pinned here — as the frozen
# ``test_existing_topic_refusal`` pins the refusal by its topic vocabulary —
# because a bare ``len(replies) == N`` cannot tell "admitted and answered"
# from "blocked by an explicit message": both send exactly one reply.  The
# admission rows assert the block texts are absent and the model answered;
# the block rows assert the block text itself landed.
REFUSAL_VOCABULARY = ("history", "earlier", "saved conversation", "new topic")
PROBE_FAILURE_VOCABULARY = ("could not check", "history", "try again")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reply_for(run: ZulipCliRun, channel: int, topic: str) -> list[dict]:
    """Bot-authored replies this process sent to (channel, topic).

    The send form encodes ``to`` as a string (form-encoded transport), so
    compare the channel with ``str(channel``; strict equality against the
    int would silently match nothing.
    """
    return [s for s in run.zulip.sends
            if s.get("to") == str(channel) and s.get("topic") == topic]


def _request_texts(messages_list: list[list[dict]]) -> list[str]:
    """Flatten all message content of each recorded request's messages."""
    return [
        "\n".join(m.get("content") or "" for m in msgs if isinstance(m, dict))
        for msgs in messages_list
    ]


def _system_of(messages_list: list[list[dict]]) -> str:
    parts = []
    for msgs in messages_list:
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system":
                parts.append(m.get("content") or "")
    return "\n".join(parts)


def _user_of(messages_list: list[list[dict]]) -> list[str]:
    parts = []
    for msgs in messages_list:
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                parts.append(m.get("content") or "")
    return parts


def _all_message_content(messages_list: list[list[dict]]) -> str:
    parts = []
    for msgs in messages_list:
        for m in msgs:
            if isinstance(m, dict) and m.get("role") != "system":
                parts.append(m.get("content") or "")
    return "\n".join(parts)


def _flatten(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages
                     if isinstance(m, dict))


def _occurrences(haystack: str, needle: str) -> int:
    return haystack.count(needle)


def _assert_history_order(request: list[dict], expected: list[tuple[str, str]],
                          current: tuple[str, str]) -> None:
    """One request contains exactly the completed history and current input.

    A no-area pipeline may have no system message. Compare the complete
    conversation, including every assistant response, so duplicates, missing
    history, extra future turns and incorrect role order all fail.
    """
    turns = [(m.get("role"), str(m.get("content") or ""))
             for m in request if isinstance(m, dict)
             and m.get("role") != "system"]
    assert turns == expected + [current], (
        f"expected ordered history {expected + [current]!r}; got {turns!r}")


class _HeldTurn:
    """Hold only the actual model response; leave event delivery natural."""

    def __init__(self, run: ZulipCliRun, channel: int, topic: str,
                 content: str) -> None:
        self.run = run
        self.channel = channel
        self.topic = topic
        self.content = content
        self.model_gen = 0
        self.sends_before = len(run.zulip.sends)

    def arm_and_trigger(self) -> None:
        self.model_gen = self.run.model.hold_responses()
        self.run.zulip.push_message(self.channel, self.topic, DIEGO_ID,
                                    self.content)
        assert self.run.model.wait_request_arrived(
            timeout=10.0, generation=self.model_gen), _fail(
                self.run, "A's actual model request was not captured")
        request = self.run.model_messages()[self.model_gen - 1]
        assert self.content in _flatten(request), request
        # Only A's working-indicator placeholder (posted before the model
        # call, spc-1014) has left the process; its answer has not.
        sends = self.run.zulip.sends
        assert (len(sends) == self.sends_before + 1
                and sends[-1]["content"] == THINKING_MESSAGE), _fail(
            self.run, "A replied before its model response was released")

    def trigger_queued(self, content: str) -> None:
        # No event gate: a responsive receiver may drain B immediately; a
        # serial receiver may naturally leave B queued until A completes.
        # Both must avoid overlapping same-topic model requests and replies.
        self.run.zulip.push_message(self.channel, self.topic, DIEGO_ID, content)
        self.run.assert_quiescent(self.model_gen, self.sends_before + 1,
                                  window=2.0)

    def release(self) -> None:
        self.run.model.release_responses()


def _fail(run: ZulipCliRun, message: str) -> str:
    return (f"{message}\nmodel_requests={len(run.model.requests)} "
            f"sends={run.zulip.sends} "
            f"events_calls={run.zulip.events_calls} "
            f"history_calls={run.zulip.history_calls} "
            f"log_tail={run.log_text[-2000:]!r} "
            f"child_tail={run.child_output()[-1500:]!r}")


def _startup_failure(run: ZulipCliRun) -> str:
    rc = run.proc.poll() if run.proc is not None else None
    return (f"CLI never started; returncode={rc} "
            f"child_output={run.child_output()[-2000:]!r} "
            f"log={run.log_text[-1500:]!r}")


def _wait_exited(run: ZulipCliRun, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if run.proc is not None and run.proc.poll() is not None:
            return
        time.sleep(0.2)


def _wait_events_drained(run: ZulipCliRun, timeout: float = 30.0) -> None:
    """Block until the endpoint queue is empty: every event so far was
    delivered to the CLI (evidence for 'the ignored event was delivered')."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not run.zulip._events:
            return
        time.sleep(0.05)
    raise AssertionError(_fail(run, "event queue never drained (undelivered?)"))


def _assert_fresh_no_work(run: ZulipCliRun, channel: int, topic: str,
                          model_count: int, send_count: int) -> None:
    """Ignored trigger on a fresh never-opened topic: after its delivery is
    acknowledged (queue drained), the model and send counters must hold
    exactly for one full continuous idle window (F1 D3/D4).

    The history-probe arm is channel-qualified: an implementation that probed
    the topic for an ignored trigger (in THAT channel) is refused here, while
    the same topic title legitimately probed in an admitted channel stays
    allowed.
    """
    assert (channel, topic) not in run.zulip.history_calls, (
        _fail(run, f"ignored trigger probed history for {channel}/{topic!r}"))
    _wait_events_drained(run)
    run.assert_quiescent(model_count, send_count)


# ---------------------------------------------------------------------------
# AC1 + AC16 — routing, filters, isolation, explicit no-area
# ---------------------------------------------------------------------------


@pytest.mark.baseline_debt("spc-965")
def test_chat_routing_and_filters(tmp_path):
    """AC1/16: allowed message answered in exact topic; filters; isolation; no-area."""
    harness = Harness(workdir=tmp_path / "j1b-route")
    run, cwd, env = harness.build()
    # D5 audit: every /api/v1/messages POST must carry the bot's Basic
    # identity. Recording happens in the endpoint dispatch BEFORE the reply
    # ledger append, so any send attempt — even a mis-admitted bot-triggered
    # turn's — is caught here.
    send_auth: list[str | None] = []

    def audit(form: dict) -> None:
        auth = run.zulip.send_headers[-1].get("authorization", "")
        user: str | None = None
        if auth.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                user = decoded.split(":", 1)[0]
            except (ValueError, UnicodeDecodeError):
                user = "<undecodable>"
        send_auth.append(user)

    run.zulip.on_send = audit
    try:
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)

        # Startup performs no history replay and no model work.
        time.sleep(1.0)
        assert len(run.model.requests) == 0, _fail(
            run, "startup made model work or history replay")
        assert run.zulip.events_calls >= 1, _fail(
            run, "CLI never began polling after registration")
        assert run.zulip.history_calls == [], _fail(
            run, "startup fetched topic history (replay)")

        # AC1: allowed user in configured channel gets a bot reply in the
        # exact topic, authored by the bot identity.
        run.zulip.push_message(CONFIGURED_CHANNEL, "alpha", DIEGO_ID, "hello alpha")
        assert run.wait_idle(expected_replies=1), _fail(
            run, "no reply for the allowed trigger in alpha")
        replies = _reply_for(run, CONFIGURED_CHANNEL, "alpha")
        assert len(replies) == 1, _fail(run, "expected 1 reply in alpha")
        assert replies[0]["type"] == "stream"
        assert replies[0]["to"] == str(CONFIGURED_CHANNEL)
        assert replies[0]["topic"] == "alpha"
        assert REPLY_MARKER in replies[0]["content"]
        # AC1 bot authorship: the ONLY identity that can create a message on
        # this fake server is the zuliprc bot — its send response echoes the
        # server-side author id — and the recorded per-send evidence agrees.
        assert run.zulip.last_send_response.get("sender_id") == BOT_ID, (
            f"fake server echoed a non-bot author: {run.zulip.last_send_response}")
        assert send_auth == [BOT_SEND_USER], (
            f"every reply send must be authenticated as the bot; saw {send_auth}")

        # One model request, and its current message is exactly the trigger.
        msgs = run.model_messages()
        assert len(msgs) == 1, _fail(run, "expected exactly 1 model request")
        assert _user_of(msgs[:1]) == ["hello alpha"], (
            f"model request user content must be exactly the trigger: {msgs[0]}")

        # AC16: a second topic gets its own Session with isolated history.
        run.zulip.push_message(CONFIGURED_CHANNEL, "beta", DIEGO_ID, "hello beta")
        assert run.wait_idle(expected_replies=2), _fail(
            run, "no reply for the allowed trigger in beta")
        replies_beta = _reply_for(run, CONFIGURED_CHANNEL, "beta")
        assert len(replies_beta) == 1 and REPLY_MARKER in replies_beta[0]["content"]
        msgs = run.model_messages()
        assert len(msgs) == 2, _fail(run, "expected exactly 2 model requests")
        assert "hello alpha" not in _request_texts(msgs[1:])[0], (
            "alpha's trigger leaked into beta's request")
        assert "hello beta" not in _request_texts(msgs[:1])[0], (
            "beta's trigger leaked into alpha's request")

        # AC16 fresh-topic probe evidence: each new topic is probed exactly
        # once for identity; a warm topic is never re-probed.
        assert run.zulip.history_calls.count((CONFIGURED_CHANNEL, "alpha")) == 1
        assert run.zulip.history_calls.count((CONFIGURED_CHANNEL, "beta")) == 1

        # AC16: bot trigger — no model work (no reply loop).  The bot id is
        # ALSO in the configured allowlist, so ignoring it proves the bot
        # metadata filter, not the user allowlist.  It targets a FRESH,
        # never-opened topic, so a mis-admission cannot be absorbed as a
        # warm-session message, and the idle window is continuous.
        run.zulip.push_message(CONFIGURED_CHANNEL, "botfresh", BOT_ID,
                               "bot chatter")
        _assert_fresh_no_work(run, CONFIGURED_CHANNEL, "botfresh", 2, 2)

        # AC16: disallowed human — fresh topic, no model work.
        run.zulip.push_message(CONFIGURED_CHANNEL, "unallowed", OTHER_ID,
                               "unallowed msg")
        _assert_fresh_no_work(run, CONFIGURED_CHANNEL, "unallowed", 2, 2)

        # AC16: unconfigured channel — fresh topic, no model work.
        run.zulip.push_message(UNCONFIGURED_CHANNEL, "offch", DIEGO_ID,
                               "off-channel")
        _assert_fresh_no_work(run, UNCONFIGURED_CHANNEL, "offch", 2, 2)

        # AC16: explicit no-area — no area marker of any kind in the
        # compiled request (synthetic topic markers via real resolvers).
        all_system = _system_of(run.model_messages())
        for marker, kind in ((AREA_MARKER, "real area"),
                             (CWD_MARKER, "cwd-inherited area"),
                             (DECOY_MARKER, "other topic's area")):
            assert marker not in all_system, (
                f"{kind} marker leaked into a no-area turn")

        # Session identity from the public session-log seam: every actual
        # ``session.start`` envelope's session_id carries the agent, the exact
        # resolved server origin and the channel; each topic owns exactly one
        # distinct identity carrying its exact topic, and an ignored trigger
        # never gained an identity.  All of it is envelope content — the file
        # name only discovers the records.
        envelopes = run.session_start_envelopes()
        assert envelopes, (
            "no session.start envelope found — no runtime Session identity to "
            f"assert (dir={run.session_log_dir})")
        for _name, envelope in envelopes:
            sid = str(envelope.get("session_id"))
            assert "miranda" in sid, (
                f"runtime Session identity lacks the agent: id={sid!r}")
            assert run.zulip.base_url in sid, (
                f"runtime Session identity lacks the exact resolved server "
                f"origin {run.zulip.base_url!r}: id={sid!r}")
            assert re.search(rf"[^A-Za-z0-9]{CONFIGURED_CHANNEL}[^A-Za-z0-9]",
                             sid), (
                f"runtime Session identity lacks the channel id: id={sid!r}")
        all_ids = run.runtime_session_ids("miranda")
        alpha_id, alpha_file = run.assert_topic_identity(
            "miranda", run.zulip._handler.port, CONFIGURED_CHANNEL, "alpha")
        beta_id, beta_file = run.assert_topic_identity(
            "miranda", run.zulip._handler.port, CONFIGURED_CHANNEL, "beta")
        assert alpha_id != beta_id, (
            f"alpha and beta share one Session identity: {alpha_id} {beta_id}")
        assert alpha_file != beta_file, (
            f"alpha and beta share one Session log: {alpha_file}")
        for ignored_topic in ("botfresh", "unallowed", "offch"):
            assert run.count_identities("miranda", ignored_topic) == 0, (
                f"an ignored trigger created a Session for {ignored_topic!r}: "
                f"{sorted(set(all_ids.values()))}")

        # Ordered history within one topic: alpha's second turn must see its
        # first turn's user message in session history.
        run.zulip.push_message(CONFIGURED_CHANNEL, "alpha", DIEGO_ID,
                               "follow-up alpha")
        assert run.wait_idle(expected_replies=3), _fail(
            run, "no reply for alpha's follow-up")
        third = run.model_messages()[2]
        third_users = [m.get("content") for m in third
                       if isinstance(m, dict) and m.get("role") == "user"]
        assert "hello alpha" in third_users and "follow-up alpha" in third_users, (
            f"same-topic follow-up lost its ordered history: {third_users}")
        # ...but beta must never see alpha's follow-up either.
        assert "follow-up alpha" not in _request_texts(run.model_messages()[1:2])[0]
        assert run.zulip.history_calls.count((CONFIGURED_CHANNEL, "alpha")) == 1, (
            "warm topic was re-probed")

        # Concurrent same-topic triggers under a HELD model response: one
        # Session, ordered history, no overlapping turn.  The earlier
        # push-both-and-flatten version could not tell a queued turn from a
        # concurrent one.  Here the model endpoint CAPTURES A's request and
        # holds its HTTP response open, so A is demonstrably in flight inside
        # the model boundary while B's trigger is enqueued.
        held = _HeldTurn(run, CONFIGURED_CHANNEL, "alpha", "concurrent A")
        held.arm_and_trigger()
        # A's model REQUEST has arrived (captured) and its response is
        # parked; nothing of A's turn but its placeholder may leave the
        # process while held:
        assert len(run.zulip.sends) == 4 and not run.zulip.settled(), _fail(
            run, "a reply was delivered while its model response was held")

        # Enqueue B while A's captured model response remains held.
        # Event delivery has no fixture gate; observe no overlap for a bounded
        # window, then require eventual exactly-once ordered completion.
        held.trigger_queued("concurrent B")
        assert len(run.model.requests) == 4, _fail(
            run, "a second model request began while A's response was held "
                 "(overlapping same-topic turn)")
        assert len(run.zulip.sends) == 4 and not run.zulip.settled(), _fail(
            run, "B was answered before A's held turn completed")
        assert run.assert_topic_identity(
            "miranda", run.zulip._handler.port, CONFIGURED_CHANNEL, "alpha",
        ) == (alpha_id, alpha_file), (
            "B spawned a second Session for alpha's in-flight turn")

        # Release A; both turns complete, in trigger order.
        held.release()
        assert run.wait_idle(expected_replies=5), _fail(
            run, "held same-topic turns never both completed")
        alpha_replies = _reply_for(run, CONFIGURED_CHANNEL, "alpha")
        assert len(alpha_replies) == 4, (
            f"concurrent same-topic messages must both be answered in order, "
            f"got {len(alpha_replies)} replies")
        _wait_events_drained(run)
        run.assert_quiescent(5, 5, window=1.0)

        # EACH captured request independently, in trigger order:
        # A's request carries alpha's completed history plus currentA
        # exactly once and no futureB; B's request additionally carries
        # A's assistant response, then currentB exactly once, with correct
        # user/assistant role order and no duplicates.
        req_a = run.model_messages()[3]
        _assert_history_order(req_a, [
            ("user", "hello alpha"),
            ("assistant", REPLY_MARKER),
            ("user", "follow-up alpha"),
            ("assistant", REPLY_MARKER),
        ], current=("user", "concurrent A"))
        assert "concurrent B" not in _flatten(req_a), (
            f"future B leaked into A's request: {req_a}")

        req_b = run.model_messages()[4]
        _assert_history_order(req_b, [
            ("user", "hello alpha"),
            ("assistant", REPLY_MARKER),
            ("user", "follow-up alpha"),
            ("assistant", REPLY_MARKER),
            # A's completed response precedes B's current input.
            ("user", "concurrent A"),
            ("assistant", REPLY_MARKER),
        ], current=("user", "concurrent B"))
        assert _occurrences(_flatten(req_b), "concurrent B") == 1, (
            f"B's current input must appear exactly once: {req_b}")

        # Both concurrent turns share alpha's ONE actual identity and log.
        run.assert_reused_identity("miranda", alpha_id, alpha_file)
        run.assert_topic_identity(
            "miranda", run.zulip._handler.port, CONFIGURED_CHANNEL, "alpha")

        # No send during the run was ever authenticated as anyone but the bot.
        assert set(send_auth) == {BOT_SEND_USER}, (
            f"a reply was sent with a non-bot identity: {send_auth}")

    finally:
        run.zulip.on_send = None
        harness.close(run)


# ---------------------------------------------------------------------------
# AC16 — the SAME exact topic in two configured channels: distinct identities
# ---------------------------------------------------------------------------


@pytest.mark.baseline_debt("spc-965")
def test_same_topic_two_configured_channels(tmp_path):
    """AC16: one exact topic title in two admitted channels stays isolated.

    Two configured no-area channels are admitted (the unconfigured-channel
    filter is retained alongside them).  The identical exact topic title —
    case and meaningful punctuation included — is opened in both.  Each must
    get its own turn, its own reply to its own channel, isolated history with
    no cross-channel leakage of prior turns *or* of the current input, and a
    DISTINCT runtime Session identity carrying the exact resolved server
    origin plus the channel and URL-escaped exact topic.  Repeated turns in
    one of those topics must reuse that one identity.
    """
    harness = Harness(
        workdir=tmp_path / "j1b-twoc",
        channels={CONFIGURED_CHANNEL: "", SECOND_CONFIGURED_CHANNEL: ""},
    )
    run, cwd, env = harness.build()
    try:
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)
        assert run.zulip.history_calls == [], _fail(
            run, "startup fetched topic history (replay)")

        # Same exact title (punctuation/case significant) in both channels.
        run.zulip.push_message(CONFIGURED_CHANNEL, SHARED_TOPIC,
                               DIEGO_ID, "relay in 42")
        run.zulip.push_message(SECOND_CONFIGURED_CHANNEL, SHARED_TOPIC,
                               DIEGO_ID, "relay in 44")
        assert run.wait_idle(expected_replies=2), _fail(
            run, "the same exact topic in two channels was not answered twice")
        assert len(run.model.requests) == 2, _fail(
            run, "each channel's copy of the topic needs its own turn")

        # Exact reply destinations: each reply went to its own channel.
        in42 = _reply_for(run, CONFIGURED_CHANNEL, SHARED_TOPIC)
        in44 = _reply_for(run, SECOND_CONFIGURED_CHANNEL, SHARED_TOPIC)
        assert len(in42) == 1 and in42[0]["to"] == str(CONFIGURED_CHANNEL), (
            f"channel {CONFIGURED_CHANNEL} reply misrouted: {run.zulip.sends}")
        assert len(in44) == 1 and in44[0]["to"] == str(SECOND_CONFIGURED_CHANNEL), (
            f"channel {SECOND_CONFIGURED_CHANNEL} reply misrouted: {run.zulip.sends}")
        assert in42[0]["topic"] == SHARED_TOPIC
        assert in44[0]["topic"] == SHARED_TOPIC
        assert REPLY_MARKER in in42[0]["content"] and REPLY_MARKER in in44[0]["content"]

        # Isolated model contexts: each request carries only its own trigger.
        texts = _request_texts(run.model_messages())
        assert len(texts) == 2
        first = next(t for t in texts if "relay in 42" in t)
        second = next(t for t in texts if "relay in 44" in t)
        assert "relay in 44" not in first, (
            f"channel 44's current input leaked into channel 42's request: {first}")
        assert "relay in 42" not in second, (
            f"channel 42's current input leaked into channel 44's request: {second}")

        # One history probe per (channel, exact topic): the identity really is
        # channel-qualified, so the same title is a different topic elsewhere.
        assert run.zulip.history_calls.count(
            (CONFIGURED_CHANNEL, SHARED_TOPIC)) == 1, run.zulip.history_calls
        assert run.zulip.history_calls.count(
            (SECOND_CONFIGURED_CHANNEL, SHARED_TOPIC)) == 1, run.zulip.history_calls

        # Runtime identities from the public session-log seam: each channel's
        # copy has its own ACTUAL identity carrying the agent, the exact
        # resolved request origin, its channel and the exact PASSED topic —
        # read from the envelope ``session_id``, with no file-name-encoding
        # coupling (the topic argument below is the same SHARED_TOPIC constant
        # this journey opened, never a hardcoded helper-internal spelling).
        topic_in_id = SHARED_TOPIC
        id42, file42 = run.assert_topic_identity(
            "miranda", run.zulip._handler.port, CONFIGURED_CHANNEL, topic_in_id)
        id44, file44 = run.assert_topic_identity(
            "miranda", run.zulip._handler.port, SECOND_CONFIGURED_CHANNEL,
            topic_in_id)
        assert id42 != id44, (
            "both channels share one Session identity for the same topic name")
        # Negative controls reject a different server/channel/exact topic
        # using the same real Session envelopes, without altering production.
        for port, channel, topic in [
            (run.zulip._handler.port + 1, CONFIGURED_CHANNEL, SHARED_TOPIC),
            (run.zulip._handler.port, UNCONFIGURED_CHANNEL, SHARED_TOPIC),
            (run.zulip._handler.port, CONFIGURED_CHANNEL, SHARED_TOPIC + "2"),
        ]:
            with pytest.raises(AssertionError):
                run.assert_topic_identity("miranda", port, channel, topic)

        # One Session log file per actual identity (no merged global facade).
        assert run.identity_file_names("miranda", id42) == [file42]
        assert run.identity_file_names("miranda", id44) == [file44]

        # Follow-ups: each channel's turn sees ITS OWN prior turn only, and the
        # repeated turns reuse the ONE actual identity asserted above.
        run.zulip.push_message(CONFIGURED_CHANNEL, SHARED_TOPIC,
                               DIEGO_ID, "follow-up in 42")
        run.zulip.push_message(SECOND_CONFIGURED_CHANNEL, SHARED_TOPIC,
                               DIEGO_ID, "follow-up in 44")
        assert run.wait_idle(expected_replies=4), _fail(
            run, "follow-ups in the shared topic never got replies")
        assert len(run.model.requests) == 4, _fail(
            run, "each follow-up needs exactly one turn")
        f42 = next(t for t in _request_texts(run.model_messages()[2:])
                   if "follow-up in 42" in t)
        f44 = next(t for t in _request_texts(run.model_messages()[2:])
                   if "follow-up in 44" in t)
        assert "relay in 42" in f42 and "follow-up in 42" in f42, (
            f"channel 42 lost its own ordered history: {f42}")
        assert "relay in 44" not in f42 and "follow-up in 44" not in f42, (
            f"channel 44 history leaked into channel 42: {f42}")
        assert "relay in 44" in f44 and "follow-up in 44" in f44, (
            f"channel 44 lost its own ordered history: {f44}")
        assert "relay in 42" not in f44 and "follow-up in 42" not in f44, (
            f"channel 42 history leaked into channel 44: {f44}")
        # One identity per (channel, topic) across all four turns: repeated
        # turns reuse the SAME actual identity, still exactly one Session log.
        run.assert_reused_identity("miranda", id42, file42)
        run.assert_reused_identity("miranda", id44, file44)
        assert len(run.zulip.history_calls) == 2, (
            f"warm topics were re-probed: {run.zulip.history_calls}")

        # The unconfigured-channel filter is RETAINED next to two admitted
        # channels: no model work, no reply, no probe, no Session.
        model_before, send_before = len(run.model.requests), len(run.zulip.sends)
        run.zulip.push_message(UNCONFIGURED_CHANNEL, SHARED_TOPIC,
                               DIEGO_ID, "off-channel same title")
        _assert_fresh_no_work(run, UNCONFIGURED_CHANNEL, SHARED_TOPIC,
                              model_before, send_before)
        # The shared title IS probed in the two admitted channels, so the
        # filter proof must be channel-qualified: never probed there, never
        # answered there, and no Session identity for that channel.  The
        # identity check reads ACTUAL envelopes: exactly the two admitted
        # (channel, topic) identities exist; an identity for channel 43 with
        # the same title would be a third distinct identity.
        assert (UNCONFIGURED_CHANNEL, SHARED_TOPIC) not in run.zulip.history_calls
        assert not _reply_for(run, UNCONFIGURED_CHANNEL, SHARED_TOPIC)
        assert run.count_identities("miranda", topic_in_id) == 2, (
            "the unconfigured channel created a Session identity for the "
            f"shared title: {sorted(set(run.runtime_session_ids('miranda').values()))}")
        assert run.assert_topic_identity(
            "miranda", run.zulip._handler.port, CONFIGURED_CHANNEL,
            topic_in_id)[0] == id42
        assert run.assert_topic_identity(
            "miranda", run.zulip._handler.port, SECOND_CONFIGURED_CHANNEL,
            topic_in_id)[0] == id44
    finally:
        harness.close(run)


# ---------------------------------------------------------------------------
# AC14 — existing topic refusal without model work
# ---------------------------------------------------------------------------


@pytest.mark.baseline_debt("spc-965")
def test_existing_topic_refusal(tmp_path):
    """AC14: populated cold topics get explicit refusal, no model work."""
    harness = Harness(workdir=tmp_path / "j1b-refusal")
    run, cwd, env = harness.build()
    try:
        # A topic populated BEFORE the process saw it — bot alert plus a
        # user message. Both are prior content: the topic must be refused.
        run.zulip.seed_message(CONFIGURED_CHANNEL, "populated",
                               BOT_ID, "ALERT: disk full")
        run.zulip.seed_message(CONFIGURED_CHANNEL, "populated",
                               DIEGO_ID, "please investigate")

        # The topic is populated before either CLI process starts. Restart
        # the real CLI against that same server history, then prove passive
        # startup: no history probe, replayed model request or bot send.
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)
        run.assert_quiescent(0, 0, window=1.0)
        assert run.zulip.history_calls == [], _fail(run, "startup replayed history")
        run.collect()
        harness.launch(run, cwd, env)
        deadline = time.monotonic() + 45.0
        while run.zulip.register_calls < 2 and time.monotonic() < deadline:
            assert run.proc.poll() is None, _startup_failure(run)
            time.sleep(0.05)
        assert run.zulip.register_calls == 2, _startup_failure(run)
        run.assert_quiescent(0, 0, window=1.0)
        assert run.zulip.history_calls == [], _fail(run, "restart replayed history")

        # Diego's new message in that topic gets an explicit refusal reply,
        # not an answer, and no model/tool work.
        run.zulip.push_message(CONFIGURED_CHANNEL, "populated",
                               DIEGO_ID, "what about this?")
        assert run.wait_idle(expected_replies=1), _fail(
            run, "populated topic produced no explicit refusal reply")
        replies = _reply_for(run, CONFIGURED_CHANNEL, "populated")
        assert len(replies) == 1, _fail(run, "expected exactly 1 refusal reply")
        assert REPLY_MARKER not in replies[0]["content"], (
            "the model answered a populated topic — refusal failed")
        text = replies[0]["content"].lower()
        assert any(phrase in text for phrase in
                   ("history", "earlier", "saved conversation", "new topic")), (
            f"refusal is not explicit about unsupported history: {replies[0]!r}")
        assert len(run.model.requests) == 0, (
            _fail(run, "model work was performed for a populated topic"))

        # A bot-only cold topic (the trigger itself is the only user message)
        # is also populated and must be refused without model work.
        run.zulip.seed_message(CONFIGURED_CHANNEL, "bot-only",
                               BOT_ID, "ALERT: only bot messages here")
        run.zulip.push_message(CONFIGURED_CHANNEL, "bot-only",
                               DIEGO_ID, "hello?")
        assert run.wait_idle(expected_replies=2), _fail(
            run, "bot-only populated topic produced no refusal reply")
        assert len(run.model.requests) == 0, (
            _fail(run, "bot-only populated topic triggered model work"))
        bot_only_replies = _reply_for(run, CONFIGURED_CHANNEL, "bot-only")
        assert len(bot_only_replies) == 1
        assert REPLY_MARKER not in bot_only_replies[0]["content"]

        # FROZEN REQUIREMENT — multiple fresh queued messages for the SAME
        # never-seen topic arriving in ONE poll batch are concurrent triggers
        # of that topic's single Session, not prior content for each other.
        # Deterministic delivery: hold the CLI's long-poll, enqueue all three,
        # release — they are returned inside a single events response.
        run.zulip.hold_next_poll()
        # The hold must arm before we enqueue: wait for an events request to
        # be parked, so the batch cannot be split across two poll responses.
        assert run.zulip.wait_poll_arrived(timeout=30.0), _fail(
            run, "CLI never issued a poll to hold (long-poll not in use?)")
        ids = [run.zulip.push_message(CONFIGURED_CHANNEL, "batch",
                                      DIEGO_ID, f"batch msg {i}")
               for i in (1, 2, 3)]
        run.zulip.release_poll()
        assert run.wait_cold_batch(topics=[(CONFIGURED_CHANNEL, "batch")],
                                   probes=1, replies=5), _fail(
            run, f"batch of 3 fresh same-topic cold messages ({ids}) was never "
                 "fully handled")
        assert len(run.model.requests) == 3, _fail(
            run, "the batch's 3 fresh messages must each get exactly one turn")
        batch_replies = _reply_for(run, CONFIGURED_CHANNEL, "batch")
        assert len(batch_replies) == 3, (
            f"expected 3 replies to the fresh batch topic, got {batch_replies}")
        for r in batch_replies:
            assert REPLY_MARKER in r["content"], (
                f"fresh same-topic batch message was refused instead of "
                f"handled: {r!r}")
        assert run.zulip.history_calls.count((CONFIGURED_CHANNEL, "batch")) == 1, (
            "the batch topic must get ONE Session, probed once")
        run.assert_topic_identity(
            "miranda", run.zulip._handler.port, CONFIGURED_CHANNEL, "batch")
        # Ordered history: the last turn sees all three triggers exactly once
        # each, in order.
        users = _user_of(run.model_messages()[-1:])
        assert [u for u in users if u.startswith("batch msg")] == [
            "batch msg 1", "batch msg 2", "batch msg 3"], (
            f"batch history not ordered/deduplicated: {users}")

        # Distinct fresh cold topics in the SAME batch stay isolated, each
        # handled normally (one turn, one Session, no cross-seeding).
        run.zulip.hold_next_poll()
        assert run.zulip.wait_poll_arrived(timeout=30.0), _fail(
            run, "CLI never issued a poll to hold (long-poll not in use?)")
        run.zulip.push_message(CONFIGURED_CHANNEL, "t1", DIEGO_ID, "solo one")
        run.zulip.push_message(CONFIGURED_CHANNEL, "t2", DIEGO_ID, "solo two")
        run.zulip.release_poll()
        assert run.wait_cold_batch(topics=[(CONFIGURED_CHANNEL, "t1"),
                                           (CONFIGURED_CHANNEL, "t2")],
                                   probes=1, replies=7), _fail(
            run, "distinct fresh cold topics in one batch were not both handled")
        assert len(run.model.requests) == 5, _fail(
            run, "distinct fresh cold topics in one batch each need their own turn")
        t1_users = _request_texts(
            [m for m in run.model_messages()
             if any(isinstance(x, dict) and x.get("content") == "solo one"
                    for x in m)])
        t2_users = _request_texts(
            [m for m in run.model_messages()
             if any(isinstance(x, dict) and x.get("content") == "solo two"
                    for x in m)])
        assert len(t1_users) == 1 and "solo two" not in t1_users[0], (
            f"t1/t2 histories cross-seeded: {t1_users}")
        assert len(t2_users) == 1 and "solo one" not in t2_users[0], (
            f"t1/t2 histories cross-seeded: {t2_users}")

        # A fresh topic after the refusals still works normally.
        run.zulip.push_message(CONFIGURED_CHANNEL, "fresh", DIEGO_ID, "fresh topic")
        assert run.wait_idle(expected_replies=8), _fail(
            run, "fresh topic after refusals produced no reply")
        fresh_replies = _reply_for(run, CONFIGURED_CHANNEL, "fresh")
        assert len(fresh_replies) == 1
        assert REPLY_MARKER in fresh_replies[0]["content"]
        # The populated topics' content never entered the fresh turn.
        fresh_text = _all_message_content(run.model_messages()[-1:])
        assert "ALERT: disk full" not in fresh_text
        assert "please investigate" not in fresh_text
        assert "ALERT: only bot messages here" not in fresh_text

    finally:
        harness.close(run)


@pytest.mark.baseline_debt("spc-965")
def test_existing_topic_probe_failure_blocks_turn(tmp_path):
    """AC14: a failed history probe blocks the model turn with an explicit reply."""
    harness = Harness(workdir=tmp_path / "j1b-probe")
    run, cwd, env = harness.build()
    try:
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)

        # Audit ANY /api/v1/messages POST — before the reply ledger append —
        # so even a fall-through model turn's reply (if the implementation
        # wrongly answered) is caught, not just ledger-visible sends.
        attempts: list[dict] = []
        run.zulip.on_send = lambda form: attempts.append(dict(form))

        # The cold topic holds ONLY the trigger — but the history probe
        # itself fails. The turn must be blocked, not answered from partial
        # (empty) context.
        run.zulip.fail_history = True
        run.zulip.push_message(CONFIGURED_CHANNEL, "probe-fail",
                               DIEGO_ID, "trigger on broken probe")
        assert run.wait_idle(expected_replies=1), _fail(
            run, "failed history probe produced no explicit blocking reply")
        replies = _reply_for(run, CONFIGURED_CHANNEL, "probe-fail")
        assert len(replies) == 1
        assert REPLY_MARKER not in replies[0]["content"], (
            "model answered despite a failed history probe")
        assert len(run.model.requests) == 0, (
            _fail(run, "model work was performed after a failed history probe"))
        assert len(attempts) == 1 and attempts[0].get("topic") == "probe-fail", (
            f"expected exactly one blocking send attempt: {attempts}")

        # The blocking response itself failed to deliver (server rejects the
        # send): still zero model work — no fall-through despite the error.
        run.zulip.fail_send = True
        run.zulip.push_message(CONFIGURED_CHANNEL, "probe-sendfail",
                               DIEGO_ID, "trigger on broken probe and send")
        sends_before = len(run.zulip.sends)
        _wait_events_drained(run)
        run.assert_quiescent(0, sends_before)
        assert len(attempts) == 2, (
            f"the failed-send case must still attempt its explicit reply: "
            f"{attempts}")

    finally:
        run.zulip.on_send = None
        harness.close(run)


# ---------------------------------------------------------------------------
# AC2/16 — a mapped nonempty area is supported: it starts and routes
# (upgraded in place from the temporary pre-routing refusal proof — the gate
# lifted by spc-120; the admission and history-refusal proofs above stand
# unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.baseline_debt("spc-965")
def test_supported_area_starts_and_routes(tmp_path):
    """AC2/16: a mapped nonempty area is valid configuration.  The CLI that
    the pre-routing cut refused to start now starts, answers in the topic,
    and compiles its mapped area's documents — while the cwd-area and other
    areas' documents stay out (no fallback, no duplication) and the
    channel's own history stays isolated."""
    harness = Harness(workdir=tmp_path / "j1b-area", nonempty_area=True)
    run, cwd, env = harness.build()
    try:
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)

        # The mapped channel admits Diego and answers in the exact topic.
        run.zulip.push_message(CONFIGURED_CHANNEL, "area-topic", DIEGO_ID,
                               "hello mapped area")
        assert run.wait_idle(expected_replies=1), _fail(
            run, "the supported mapped-area channel never got a reply")
        replies = _reply_for(run, CONFIGURED_CHANNEL, "area-topic")
        assert len(replies) == 1 and REPLY_MARKER in replies[0]["content"]

        # The mapped area reached the resolvers: its AGENTS.md and source
        # NOW.md are in the compiled request, each exactly once.
        text = "\n".join(_request_texts(run.model_messages()))
        assert f"{AREA_MARKER} AGENTS" in text and f"{AREA_MARKER} NOW" in text, (
            _fail(run, f"mapped area spctr's documents missing from the "
                       f"compiled request"))
        assert text.count(f"{AREA_MARKER} AGENTS") == 1, (
            _fail(run, "mapped-area AGENTS injected more than once"))
        assert text.count(f"{AREA_MARKER} NOW") == 1, (
            _fail(run, "mapped-area NOW injected more than once"))

        # No accidental inheritance and no multi-area bleed.
        assert CWD_MARKER not in text, (
            _fail(run, "cwd area leaked into a mapped-area turn"))
        assert DECOY_MARKER not in text, (
            _fail(run, "an unmapped area's documents leaked into the turn"))

        # The filters and isolation the pilot relies on survived the gate
        # lift: a bot event on the mapped channel does no model work, and an
        # unconfigured channel still triggers nothing.
        run.zulip.push_message(CONFIGURED_CHANNEL, "bot-topic", BOT_ID,
                               "bot chatter")
        _assert_fresh_no_work(run, CONFIGURED_CHANNEL, "bot-topic", 1, 1)
        run.zulip.push_message(UNCONFIGURED_CHANNEL, "offch", DIEGO_ID,
                               "off-channel")
        _assert_fresh_no_work(run, UNCONFIGURED_CHANNEL, "offch", 1, 1)
    finally:
        harness.close(run)


# ---------------------------------------------------------------------------
# AC16 (FR 9a, amendment 2026-11-07) — messages from the bot's OWN
# registration account are context, not prior content.
#
# These journeys run with archive restore disabled
# (``Harness(resume_enabled=False)``): the restore short-circuit inside
# ``_ensure_session`` admits (and short-circuits) whenever a restore reports
# failure, which every cold-topic turn of this harness hit once
# ``provenance_store_path`` went empty — the probe, i.e. the FR 9a path,
# would never run.  Disabling restore reproduces the cold-topic state the
# probe was built for; every existing journey keeps its own config bytes and
# its own tracked outcome (spc-965).  The fixture's own registration id is
# OWN_ID, distinct from BOT_ID, the author the fake records for the bot's
# sends — see the constant's comment in ``zulip_e2e_support``.
# ---------------------------------------------------------------------------


def _non_system_turns(request: list[dict]) -> list[tuple[str, str]]:
    """(role, content) of every non-system message, request order."""
    return [(m.get("role"), str(m.get("content") or ""))
            for m in request if isinstance(m, dict)
            and m.get("role") != "system"]


def _blocked_reply(reply: dict, vocabulary: tuple[str, ...]) -> None:
    """One blocking reply: the model's own answer marker absent, and explicit
    about why it blocked (vocabulary, as the frozen refusal test phrases it)."""
    assert REPLY_MARKER not in reply["content"], (
        f"the model answered anyway: {reply!r}")
    text = reply["content"].lower()
    assert any(phrase in text for phrase in vocabulary), (
        f"reply is not explicit about blocking: {reply!r}")


def test_own_account_cold_topic_admitted_with_context_ac16a(tmp_path):
    """AC16-a: own-account seeds are placed before the trigger as assistant
    context, the turn is admitted, the reply lands, nothing refuses.

    The request must carry the seeds verbatim, in seed order, before the
    trigger, as assistant role (never user), and exactly once. The request
    is read under a model-response gate, so the assertions run while the
    turn is the topic's only activity.
    """
    harness = Harness(workdir=tmp_path / "ac16a", resume_enabled=False)
    run, cwd, env = harness.build()
    try:
        first = run.zulip.seed_message(CONFIGURED_CHANNEL, "ask",
                                       OWN_ID, "AC16A_OWN_ONE")
        second = run.zulip.seed_message(CONFIGURED_CHANNEL, "ask",
                                        OWN_ID, "AC16A_OWN_TWO")
        assert first < second  # the fixture's chronological order: ONE, then TWO
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)

        held = _HeldTurn(run, CONFIGURED_CHANNEL, "ask", "AC16A_TRIGGER")
        held.arm_and_trigger()
        request = run.model_messages()[held.model_gen - 1]

        turns = _non_system_turns(request)
        asst = [text for role, text in turns if role == "assistant"]
        users = [text for role, text in turns if role == "user"]
        assert "AC16A_OWN_ONE" in asst and "AC16A_OWN_TWO" in asst, (
            f"own-account seeds missing from the request as assistant context: "
            f"{turns!r}")
        assert users.count("AC16A_TRIGGER") == 1, (
            f"trigger missing or duplicated as a user turn: {users!r}")
        assert "AC16A_TRIGGER" not in asst and "AC16A_OWN_ONE" not in users \
            and "AC16A_OWN_TWO" not in users, (
            f"a seed entered as a command or the trigger as context: {turns!r}")
        trigger_at = next(i for i, turn in enumerate(turns)
                          if turn[1] == "AC16A_TRIGGER")
        assert trigger_at == len(turns) - 1, (
            f"the trigger is not the current input: {turns!r}")
        # Chronological order pinned on (role, text) pairs: exactly the two
        # seeds, assistant-role, in seed order, before the trigger.
        seed_turns = [turn for turn in turns if turn[1] in
                      ("AC16A_OWN_ONE", "AC16A_OWN_TWO")]
        assert seed_turns == [("assistant", "AC16A_OWN_ONE"),
                              ("assistant", "AC16A_OWN_TWO")], (
            f"seeds not assistant-role, ordered and unique: {turns!r}")
        assert turns.index(seed_turns[0]) < trigger_at, (
            f"seeds not placed before the trigger: {turns!r}")
        flat = _flatten(request)
        for seed in ("AC16A_OWN_ONE", "AC16A_OWN_TWO"):
            assert _occurrences(flat, seed) == 1, (
                f"{seed} appears more than once in the request: {turns!r}")

        # Release: the parked HTTP request IS answered afterwards (the fake
        # blocks on the release gate, then returns its answer), so the held
        # turn delivers its own reply: its placeholder was posted pre-model
        # and edited into the answer (spc-1014).  The fixture's ledger records
        # each bot message as the topic finally shows it, hence exactly one
        # reply here.
        held.release()
        assert run.wait_idle(expected_replies=1), _fail(
            run, "admitted own-account topic never got its answer")
        replies = _reply_for(run, CONFIGURED_CHANNEL, "ask")
        assert len(replies) == 1, _fail(
            run, f"expected exactly the answer in 'ask', got {replies!r}")
        assert REPLY_MARKER in replies[0]["content"], (
            f"the reply is not the model's answer: {replies[0]!r}")
        text = replies[0]["content"].lower()
        assert not any(phrase in text for phrase in REFUSAL_VOCABULARY), (
            f"refusal text sent for an admitted topic: {replies[0]!r}")
        assert run.zulip.history_calls.count(
            (CONFIGURED_CHANNEL, "ask")) == 1, run.zulip.history_calls
        # A turn on a warm second topic after an admitted own-account topic:
        # the admission and the seeded context never leak across topics.
        run.zulip.push_message(CONFIGURED_CHANNEL, "other", DIEGO_ID,
                               "AC16A_ISOLATION")
        assert run.wait_idle(expected_replies=2), _fail(
            run, "the isolation topic never got its reply")
        other_request = run.model_messages()[-1]
        other_text = _flatten(other_request)
        assert "AC16A_ISOLATION" in other_text, _fail(
            run, f"isolation topic's trigger missing: {other_request!r}")
        for seed in ("AC16A_OWN_ONE", "AC16A_OWN_TWO"):
            assert seed not in other_text, (
                f"own-account context leaked across topics: {other_request!r}")
    finally:
        harness.close(run)


def test_own_account_plus_foreign_message_refused_ac16b(tmp_path):
    """AC16-b: one own-account seed plus a prior Diego message is still
    refused with the unsupported-history reply and no model work."""
    harness = Harness(workdir=tmp_path / "ac16b", resume_enabled=False)
    run, cwd, env = harness.build()
    try:
        run.zulip.seed_message(CONFIGURED_CHANNEL, "mixed",
                               OWN_ID, "AC16B_OWN_ASK")
        run.zulip.seed_message(CONFIGURED_CHANNEL, "mixed",
                               DIEGO_ID, "AC16B_PRIOR_DIEGO")
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)

        run.zulip.push_message(CONFIGURED_CHANNEL, "mixed",
                               DIEGO_ID, "AC16B_TRIGGER")
        assert run.wait_idle(expected_replies=1), _fail(
            run, "mixed own-account + foreign topic produced no refusal reply")
        replies = _reply_for(run, CONFIGURED_CHANNEL, "mixed")
        assert len(replies) == 1, _fail(run, f"expected 1 refusal: {replies!r}")
        _blocked_reply(replies[0], REFUSAL_VOCABULARY)
        assert len(run.model.requests) == 0, (
            _fail(run, "model work was performed for a foreign prior message"))
        # A refusal blocks; it never seeds. Nothing of the refused topic
        # reaches the next fresh topic's turn.
        run.zulip.push_message(CONFIGURED_CHANNEL, "fresh", DIEGO_ID,
                               "AC16B_FRESH")
        assert run.wait_idle(expected_replies=2), _fail(
            run, "fresh topic after the AC16-b refusal never got its reply")
        fresh = _reply_for(run, CONFIGURED_CHANNEL, "fresh")
        assert len(fresh) == 1 and REPLY_MARKER in fresh[0]["content"], (
            f"fresh topic after AC16-b refusal was not answered: {fresh!r}")
        fresh_text = _flatten(run.model_messages()[-1])
        assert "AC16B_OWN_ASK" not in fresh_text \
            and "AC16B_PRIOR_DIEGO" not in fresh_text, (
            "the refused topic's content entered the fresh turn")
    finally:
        harness.close(run)


def test_other_bot_message_refused_ac16c(tmp_path):
    """AC16-c: another BOT's message is not own-account context — refused,
    no model work (FR 9 unchanged for foreign senders)."""
    harness = Harness(workdir=tmp_path / "ac16c", resume_enabled=False)
    run, cwd, env = harness.build()
    try:
        run.zulip.seed_message(CONFIGURED_CHANNEL, "otherbot",
                               BOT_ID, "AC16C_OTHER_BOT_ALERT")
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)

        run.zulip.push_message(CONFIGURED_CHANNEL, "otherbot",
                               DIEGO_ID, "AC16C_TRIGGER")
        assert run.wait_idle(expected_replies=1), _fail(
            run, "other-bot-populated topic produced no refusal reply")
        replies = _reply_for(run, CONFIGURED_CHANNEL, "otherbot")
        assert len(replies) == 1, _fail(run, f"expected 1 refusal: {replies!r}")
        _blocked_reply(replies[0], REFUSAL_VOCABULARY)
        assert len(run.model.requests) == 0, (
            _fail(run, "model work was performed for another bot's message"))
    finally:
        harness.close(run)


def test_probe_failure_blocks_ac16d(tmp_path):
    """AC16-d: a malformed probe answer blocks the turn with the probe-failure
    reply and no model work (unchanged for the FR 9a code path)."""
    harness = Harness(workdir=tmp_path / "ac16d", resume_enabled=False)
    run, cwd, env = harness.build()
    try:
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)

        attempts: list[dict] = []
        run.zulip.on_send = lambda form: attempts.append(dict(form))
        run.zulip.fail_history = True
        run.zulip.push_message(CONFIGURED_CHANNEL, "probebroken",
                               DIEGO_ID, "AC16D_TRIGGER")
        assert run.wait_idle(expected_replies=1), _fail(
            run, "broken probe produced no explicit blocking reply")
        replies = _reply_for(run, CONFIGURED_CHANNEL, "probebroken")
        assert len(replies) == 1, _fail(run, f"expected 1 block reply: {replies!r}")
        _blocked_reply(replies[0], PROBE_FAILURE_VOCABULARY)
        assert len(run.model.requests) == 0, (
            _fail(run, "model work was performed after a broken probe"))
        assert len(attempts) == 1, (
            f"expected exactly one blocking send attempt: {attempts}")
    finally:
        run.zulip.on_send = None
        harness.close(run)


async def test_own_account_classifier_at_probe_seam_ac16e(tmp_path, monkeypatch):
    """AC16-e: the real cold-topic classifier over mixed senders — trigger id,
    an admitted id, the bot's own registration id and a foreign sender.

    Drives the shipped ``ZulipInterface`` once per entry, with that entry as
    the sole probe result, and reads the outcome only through observable
    behavior: admitted (the turn's model call runs, no reply) or refused
    (``_REFUSAL_TEXT`` sent, no model call).  The three exclusions are
    independent, so an entry carrying every other id still refuses.
    """
    import spctr.interfaces.zulip.client as client_mod
    import spctr.interfaces.zulip.interface as iface_mod
    from spctr.interfaces.zulip import ZulipConfig, ZulipInterface
    from spctr.interfaces.zulip.client import ZulipCredentials

    trigger_id, admitted_id = 1001, 1002
    own_content = "AC16E-OWN-CONTEXT"
    entries = {
        # Each case probes a fetch holding ONLY this entry, so the outcome
        # shows what that entry alone is: an excluded id (the trigger, or a
        # sibling this batch admitted), own-account context, or prior
        # content.  "sibling_*" entries carry an id that is a trigger only
        # when a second message rides in the same poll batch.
        "trigger": {"id": trigger_id, "sender_id": DIEGO_ID,
                    "content": "AC16E-TRIGGER-TEXT"},
        "sibling_in_batch": {"id": admitted_id, "sender_id": DIEGO_ID,
                             "content": "AC16E-SIBLING-IN-BATCH"},
        "sibling_not_in_batch": {"id": admitted_id, "sender_id": DIEGO_ID,
                                 "content": "AC16E-SIBLING-ELSEWHERE"},
        "own_account": {"id": 1003, "sender_id": OWN_ID,
                        "content": own_content},
        "foreign": {"id": 1004, "sender_id": BOT_ID,
                    "content": "AC16E-FOREGN-PRIOR"},
        "foreign_human": {"id": 1006, "sender_id": DIEGO_ID,
                          "content": "AC16E-FOREGN-HUMAN"},
        "own_foreign_id": {"id": 1005, "sender_id": OWN_ID,
                           "trigger_id": trigger_id,
                           "admitted_ids": [admitted_id, trigger_id],
                           "content": own_content},
        "foreign_own_id": {"id": 1007, "sender_id": BOT_ID,
                           "content": "AC16E-FOREGN-AT-ADMITTED-ID"},
        "own_at_trigger_id": {"id": trigger_id, "sender_id": OWN_ID,
                              "content": own_content},
    }

    async def _probe_case(name: str):
        entry = entries[name]
        # The batch-exclusion half of the classifier needs a real second
        # trigger riding the same poll batch — admitted IDs come from the
        # batch itself, never from a hand-written set.  Both sibling cases
        # probe the SAME entry (a foreign prior message at the sibling's id):
        # delivered in the batch it is excluded, absent from it it is prior.
        sibling = {"id": admitted_id, "type": "stream",
                   "stream_id": CONFIGURED_CHANNEL,
                   "display_recipient": "sandbox", "subject": "probe",
                   "sender_id": DIEGO_ID, "content": "AC16E-SECOND-TRIGGER"}
        batches = [[{"id": trigger_id, "type": "stream",
                     "stream_id": CONFIGURED_CHANNEL,
                     "display_recipient": "sandbox", "subject": "probe",
                     "sender_id": DIEGO_ID, "content": "AC16E-TRIGGER"}]]
        if name == "sibling_in_batch":
            batches[0].append(sibling)
        sent: list[dict] = []
        turns: list[tuple[str, list]] = []
        probe_answer: list = []

        class _Session:
            def __init__(self) -> None:
                self.history = []

            async def restore_history(self):
                from spctr.resume import ResumeResult
                return ResumeResult([], interrupted=False, restore_failed=False)

            def set_run_context(self, ctx) -> None:
                pass

            async def stream_message(self, text: str):
                turns.append((text, list(self.history)))
                from spctr.events import AgentDone, AgentTextDelta
                yield AgentTextDelta(text="model answered")
                yield AgentDone(tool_calls_executed=0)

            def close(self) -> None:
                pass

        class _Agent:
            config = SimpleNamespace(agent=SimpleNamespace(name="miranda"))

            def create_session(self, frame_id: str):
                return _Session()

            def apply_config(self, config) -> None:
                pass

        class _Client:
            async def register(self) -> dict:
                # The registration contract's own-account identity: user_id.
                return {"queue_id": "q", "last_event_id": 0,
                        "user_id": OWN_ID,
                        "realm_users": [{"user_id": DIEGO_ID,
                                         "is_bot": False},
                                        {"user_id": OWN_ID, "is_bot": True}]}

            async def get_events(self, queue_id: str, last_event_id: int):
                if not batches:
                    interface._running = False
                    await asyncio.sleep(0)
                    return {"events": []}
                messages = batches.pop(0)
                return {"events": [{"id": m["id"], "type": "message",
                                    "message": m} for m in messages]}

            async def get_topic_messages(self, channel_id: int, topic: str):
                probe_answer.append(entry)
                return {"messages": [entry]}

            async def send_message(self, **kwargs) -> int:
                sent.append(kwargs)
                return len(sent)

            async def update_message(self, message_id: int,
                                     content: str) -> None:
                sent[message_id - 1] = {**sent[message_id - 1],
                                        "content": content}

            async def aclose(self) -> None:
                pass

        monkeypatch.setattr(iface_mod, "ZulipClient", client_mod.ZulipClient)
        monkeypatch.setattr(iface_mod, "load_zulip_credentials",
                            lambda path: ZulipCredentials(
                                email="bot@test.local", api_key="k",
                                site="http://127.0.0.1:1"))
        interface = ZulipInterface(ZulipConfig(
            zuliprc="/unused.zuliprc", users=[DIEGO_ID],
            channels=[CONFIGURED_CHANNEL]))
        client = _Client()
        monkeypatch.setattr(iface_mod, "ZulipClient", lambda **kw: client)
        agent = _Agent()
        await interface.start(agent)
        try:
            await interface.run(agent)
        finally:
            await interface.stop()
        return turns, sent, probe_answer

    # Admitted: the trigger's own id is excluded, a sibling's id is excluded
    # exactly when that sibling rode this poll batch, and no own-sender entry
    # is prior content (fresh id, the admitted sibling id, even the trigger's
    # own id) — an own-sender entry enters history as assistant context.
    for name in ("trigger", "own_account", "own_foreign_id",
                 "own_at_trigger_id"):
        turns, sent, probes = await _probe_case(name)
        assert len(turns) == 1, (
            f"{name}: expected exactly one admitted turn; turns={turns!r}")
        refused = [s for s in sent
                   if any(phrase in str(s.get("content", "")).lower()
                          for phrase in REFUSAL_VOCABULARY)]
        assert refused == [], (
            f"{name}: admitted turn also sent the unsupported-history text")
        text, history = turns[0]
        assert text == "AC16E-TRIGGER"
        if name == "sibling_in_batch":
            # Both triggers of the batch admitted off one probe, in order —
            # that one probe is the single fetched entry (the ledger the case
            # returns), with the sibling excluded by its batch id.
            assert [t for t, _ in turns] == [
                "AC16E-TRIGGER", "AC16E-SECOND-TRIGGER"], (
                f"batch pair did not both run off the one probe: {turns!r}")
            assert probes == [entries["sibling_in_batch"]], (
                f"batch pair probed more than once: {probes!r}")
        elif name in ("own_account", "own_foreign_id", "own_at_trigger_id"):
            assert [(m.role, "".join(b.text for b in m.content))
                    for m in history] == [("assistant", own_content)], (
                f"own-account context not assistant-authored before the turn: "
                f"{history!r}")
        else:
            assert history == [], (
                f"{name}: excluded entry wrongly entered history: {history!r}")

    # Refused: foreign senders stay prior content, and neither the trigger id
    # nor an id this batch never admitted turns a foreign sender into context.
    for name in ("foreign", "foreign_human", "foreign_own_id",
                 "sibling_not_in_batch"):
        turns, sent, probes = await _probe_case(name)
        assert turns == [], f"{name}: foreign sender was admitted"
        assert len(sent) == 1, f"{name}: expected exactly one refusal reply"
        assert any(phrase in str(sent[0].get("content", "")).lower()
                   for phrase in REFUSAL_VOCABULARY), (
            f"{name}: refusal reply not the unsupported-history text: {sent!r}")
