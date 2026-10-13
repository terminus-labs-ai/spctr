"""J1B (E2E) — spc-125: Connect Miranda's CLI to safe no-area Zulip chat.

Runs the REAL user entry point (``python -m sr2_spectre.cli <config>
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
- test_nonempty_area_rejected             → AC16 (startup failure)

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

import base64
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zulip_e2e_support import (  # noqa: E402
    AREA_MARKER,
    BOT_ID,
    CONFIGURED_CHANNEL,
    CWD_MARKER,
    DECOY_MARKER,
    DIEGO_ID,
    OTHER_ID,
    REPLY_MARKER,
    SECOND_CONFIGURED_CHANNEL,
    SHARED_TOPIC,
    UNCONFIGURED_CHANNEL,
    Harness,
    ZulipCliRun,
)

BOT_SEND_USER = "bot@e2e.local"  # synthetic zuliprc identity


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
    """Assert one captured model request's conversation shape.

    - every compiled prompt begins with its system context;
    - ``expected`` (the prior turns, in chronological role order) appears as
      an in-order subsequence of the non-system messages;
    - ``current`` (this turn's input) appears LAST, after all history;
    - the current content appears EXACTLY once (the trigger is never
      duplicated into the history);
    - no assistant turn precedes the user turn it answers (strict overall
      user/assistant alternation from the first user message on).
    """
    msgs = [m for m in request if isinstance(m, dict)]
    assert msgs and msgs[0].get("role") == "system", (
        f"compiled request must start with system context: {request}")
    turns = [(m.get("role"), str(m.get("content") or "")) for m in msgs[1:]]
    body = turns
    # In-order history subsequence.
    pos = 0
    for role, content in expected:
        while pos < len(body) and not (body[pos][0] == role
                                       and content in body[pos][1]):
            pos += 1
        assert pos < len(body), (
            f"history turn ({role}, {content!r}) missing from its position in "
            f"the request; turns were {turns}")
        pos += 1
    cur_role, cur_content = current
    matching = [i for i, (role, text) in enumerate(body)
                if role == cur_role and cur_content in text]
    assert matching, (
        f"current input ({cur_role}, {cur_content!r}) absent from request: {turns}")
    assert _occurrences(_flatten(msgs), cur_content) == 1, (
        f"current input {cur_content!r} must appear exactly once: {turns}")
    assert matching[-1] == len(body) - 1, (
        f"current input must be the last turn, got {turns}")
    # Strict user/assistant alternation over the conversation body.
    conv = [role for role, _ in body]
    for i in range(len(conv) - 1):
        assert conv[i] != conv[i + 1], (
            f"conversation roles must alternate, got {conv}")
    assert conv[-1] == cur_role


class _HeldTurn:
    """Drive one model turn held open at the actual HTTP response boundary.

    The signal for "the CLI is waiting on this in-flight turn" is a decisive
    NEGATIVE measured at the transport: the model endpoint captures A's
    request and parks the response (A in flight), and while held the CLI
    issues NO NEW events request (arrival quiescence).  A CLI that had
    dispatched A and moved on would poll again within milliseconds — as it
    demonstrably does between turns.  The trigger itself is delivered by
    holding the event lane opaque across arming: the held trigger's poll
    arrival is observed, then its model request is observed; the lane stays
    opaque afterwards so nothing else can be delivered during the hold.
    """

    def __init__(self, run: ZulipCliRun, channel: int, topic: str,
                 content: str) -> None:
        self.run = run
        self.channel = channel
        self.topic = topic
        self.content = content
        self.model_gen = 0

    def _wait_quiescent(self, label: str, hold_remaining: float,
                        window: float = 2.0) -> None:
        """Assert the CLI issues NO new events request for *window* seconds.

        ``hold_remaining`` is the time (relative to now) before an armed hold
        crosses a client-side deadline; the observation MUST finish before
        that, so no parked request ever expires on its own.
        """
        run = self.run
        latest_start = time.monotonic() + max(0.5, hold_remaining - 2.0)
        while time.monotonic() < latest_start:
            mark = run.zulip.poll_mark()
            time.sleep(window)
            if not run.zulip.new_poll_arrivals(mark):
                return
        raise AssertionError(_fail(
            run, f"{label}: the CLI never went quiet on the event transport"))

    def arm_and_trigger(self) -> None:
        """Arm both gates, then trigger; prove the turn is in flight.

        1. Make event delivery opaque (existing parked polls are received
           and recorded, never answered).
        2. Enqueue the trigger; wait until the CLI's NEXT poll arrives and
           is parked, holding the trigger — delivery is now genuinely
           in-progress and the trigger is at the boundary, not in the CLI.
        3. Arm the model-response hold, release parked polls exactly once
           (the observed one returns the trigger; later polls park again),
           and wait until the model endpoint has CAPTURED this turn's
           request.
        4. Wait for the CLI to go transport-quiet: it parked a long-poll and
           now awaits the held model response, with no second request begun.
        """
        run = self.run
        self._t0 = time.monotonic()
        run.zulip.hold_next_events()
        # Arm the model-response hold BEFORE the trigger exists, anchored at
        # the current completed-request count: any request captured from
        # here on is a turn the test must account for (fail fast, never a
        # silent wedge).  The trigger is therefore admitted either way —
        # before this request (hold catches the request) or after (hold is
        # already armed) — so it can never slip through unanswered-but-held
        # later, and can never be answered-and-sent during the hold.
        self.model_gen = len(run.model.requests) + 1
        run.model.hold_responses()
        delivered_mark = run.zulip.poll_mark()
        run.zulip.push_message(self.channel, self.topic, DIEGO_ID,
                               self.content)
        assert run.zulip.wait_new_poll(delivered_mark, timeout=10.0), _fail(
            run, f"the CLI never polled for the held trigger {self.content!r}")
        # Answer ONLY the polls that had arrived up to now (the one that
        # carries the trigger); anything newer parks unanswered.
        deliver_through = run.zulip.poll_mark()
        run.zulip.release_and_answer_through(deliver_through)
        assert run.model.wait_request_arrived(
            timeout=10.0, generation=self.model_gen), _fail(
            run, f"held trigger {self.content!r} never reached the model "
                 "endpoint (no request captured)")
        assert len(run.zulip.sends) == 3, _fail(
            run, "a reply was delivered while the model response was held")
        self._wait_quiescent(
            "the CLI kept polling while its model response was held",
            hold_remaining=25.0 - (time.monotonic() - self._t0))
        assert len(run.model.requests) == self.model_gen, _fail(
            run, "a second model request began while A's response was held "
                 "(overlapping same-topic turn)")

    def trigger_queued(self, content: str) -> threading.Event:
        """Enqueue another trigger while the held turn is in flight.

        The event lane is opaque, so this message provably stays queued
        (never delivered to the CLI) until ``release`` runs; the returned
        flag flips when its reply send lands, which must happen only AFTER
        the held turn completes.
        """
        run = self.run
        assert not run.zulip._events_block.is_set(), (
            "the event lane must stay opaque during a held turn")
        run.zulip.push_message(self.channel, self.topic, DIEGO_ID, content)
        done = threading.Event()
        prev = run.zulip.on_send

        def hook(form: dict) -> None:
            if (form.get("to") == str(self.channel)
                    and form.get("topic") == self.topic
                    and content in (form.get("content") or "")):
                done.set()
            if prev is not None:
                prev(form)

        run.zulip.on_send = hook
        return done

    def release(self) -> None:
        self.run.model.release_responses()
        self.run.zulip.release_events()


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
        if not run.zulip._handler.last_request:
            return
        auth = run.zulip._handler.last_request.get("authorization", "")
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
        # parked; nothing of A's turn may leave the process while held:
        assert len(run.zulip.sends) == 3, _fail(
            run, "a reply was delivered while its model response was held")

        # Enqueue B's trigger while A is provably still in flight, and keep
        # B pending in the endpoint queue (a parked poll): B is admitted
        # only after the release, exactly once A's turn ends.
        pending = held.trigger_queued("concurrent B")
        assert len(run.model.requests) == 4, _fail(
            run, "a second model request began while A's response was held "
                 "(overlapping same-topic turn)")
        assert run.zulip.turn_parked(), _fail(
            run, "the CLI did not wait on A's in-flight model request")
        assert len(run.zulip.sends) == 3, _fail(
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
        assert not run.zulip.events_pending() and not pending.is_set(), (
            "the held-batch contract broke: B's event was never delivered")

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
        run.collect()
        harness.close(run)


# ---------------------------------------------------------------------------
# AC16 — the SAME exact topic in two configured channels: distinct identities
# ---------------------------------------------------------------------------


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
        topic_in_id = urllib.parse.quote(SHARED_TOPIC, safe="")
        id42, file42 = run.assert_topic_identity(
            "miranda", run.zulip._handler.port, CONFIGURED_CHANNEL, topic_in_id)
        id44, file44 = run.assert_topic_identity(
            "miranda", run.zulip._handler.port, SECOND_CONFIGURED_CHANNEL,
            topic_in_id)
        assert id42 != id44, (
            "both channels share one Session identity for the same topic name")
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
        run.collect()
        harness.close(run)


# ---------------------------------------------------------------------------
# AC14 — existing topic refusal without model work
# ---------------------------------------------------------------------------


def test_existing_topic_refusal(tmp_path):
    """AC14: populated cold topics get explicit refusal, no model work."""
    harness = Harness(workdir=tmp_path / "j1b-refusal")
    run, cwd, env = harness.build()
    try:
        harness.launch(run, cwd, env)
        assert run.wait_started(), _startup_failure(run)

        # A topic populated BEFORE the process saw it — bot alert plus a
        # user message. Both are prior content: the topic must be refused.
        run.zulip.seed_message(CONFIGURED_CHANNEL, "populated",
                               BOT_ID, "ALERT: disk full")
        run.zulip.seed_message(CONFIGURED_CHANNEL, "populated",
                               DIEGO_ID, "please investigate")

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
                                   probes=1, replies=3), _fail(
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
        assert len([n for n in (p.name for p in run.session_log_files("miranda"))
                    if "batch" in n]) == 1, (
            "concurrent same-topic batch spawned multiple Sessions")
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
                                   probes=1, replies=5), _fail(
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
        assert run.wait_idle(expected_replies=6), _fail(
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
        run.collect()
        harness.close(run)


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
        run.collect()
        harness.close(run)


# ---------------------------------------------------------------------------
# AC16 — nonempty area mapping fails startup
# ---------------------------------------------------------------------------


def test_nonempty_area_rejected(tmp_path):
    """AC16: nonempty area mapping fails startup explicitly, zero model calls."""
    harness = Harness(workdir=tmp_path / "j1b-area", nonempty_area=True)
    run, cwd, env = harness.build()
    try:
        harness.launch(run, cwd, env)

        # The process must fail (nonzero exit) within a bounded time.
        _wait_exited(run)
        assert run.proc is not None and run.proc.returncode is not None, (
            "CLI did not exit within the bound with a nonempty area mapping; "
            f"child_output={run.child_output()[-1500:]!r}")
        assert run.proc.returncode != 0, (
            f"CLI exited 0 with a nonempty area mapping; it must fail "
            f"startup explicitly. child_output="
            f"{run.child_output()[-1500:]!r}")
        # Zero model work and the event queue was never registered/serviced.
        assert len(run.model.requests) == 0, (
            "model was called despite the nonempty-area startup failure")
        assert run.zulip.register_calls == 0, (
            "event queue registered despite the nonempty-area startup failure")
        # The failure names the routing problem, not a generic crash.
        all_text = run.child_output() + run.log_text
        assert "area" in all_text.lower(), (
            f"startup failure did not name the area problem: {all_text[-1500:]!r}")
    finally:
        run.collect()
        harness.close(run)
