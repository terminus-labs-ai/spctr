"""E2E support for spc-125 (Connect Miranda's CLI to safe no-area Zulip chat).

Everything the frozen J1B journeys need to run the REAL user entry point
(``python -m sr2_spectre.cli <config> --interface zulip --log-file …``)
against substituted loopback boundaries only:

- a Zulip endpoint speaking the actual protocol (register with string
  ``queue_id`` and ``realm_users``/``realm_bots`` bot metadata, ``events``
  array polling, JSON-encoded exact-topic history narrow, stream send to
  channel/topic), with recorded requests;
- an OpenAI-compatible model endpoint streaming a deterministic answer;
- isolated SR2_HOME, cwd, model pointer, synthetic zuliprc and area-marker
  topic documents;
- bounded subprocess launch/kill/reap owned by the test.

No production code, mocks of Session/Runtime/Agent, or shared infrastructure
is touched: the only substitutions are the two external HTTP endpoints and
synthetic credentials/data, exactly as the frozen journey declares.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

DIEGO_ID = 8          # spec Dependencies: Diego user ID
BOT_ID = 9            # spec Dependencies: Miranda bot user ID
OTHER_ID = 7          # disallowed human user
CONFIGURED_CHANNEL = 42
UNCONFIGURED_CHANNEL = 43
SECOND_CONFIGURED_CHANNEL = 44   # second admitted no-area channel

# Exact topic title shared by two configured channels.  Its case, comma,
# space, question mark and sharp-significant characters are all meaningful:
# an implementation that keyed a lowercased, stripped or otherwise mutated
# topic form fails the exact-identity assertion, and a boundary-blind
# substring match cannot stretch this title into a neighbouring one.
SHARED_TOPIC = "Kickoff, A?B"

# Session identity assertions read the PUBLIC session-log seam documented in
# docs/INTERFACE-DEV-GUIDE.md ("Every Session writes a live JSONL log to
# $SR2_HOME/logs/sessions/<UTC>-<session-id>-<suffix>.jsonl", envelope fields
# including ``session_id``).  Everything expected of the identity is a public
# fact: the agent name, the resolved synthetic endpoint this run is
# configured against, the channel, and the exact passed topic — asserted on
# the ENVELOPE ``session_id``, so the expectation comes from public facts and
# actual runtime records, never from reading the adapter or from a mandated
# file-name encoding.

# Synthetic markers (spec readiness: detect unrelated area/history leakage).
AREA_MARKER = "SPC125_AREA_MARKER"      # real area's topic document
DECOY_MARKER = "SPC125_DECOY_MARKER"    # other topics' documents
CWD_MARKER = "SPC125_CWD_MARKER"        # document a cwd fallback would inject
REPLY_MARKER = "SPC125_REPLY_MARKER"    # the model's deterministic answer

# Bounded run (spec: bounded wait, kill and reap after it).
CLI_START_TIMEOUT = 45.0
CLI_SETTLE_TIMEOUT = 30.0
CLI_STOP_TIMEOUT = 10.0
# Ignored-trigger proof: an accepted-but-ignored event must produce NO model
# request for at least this full window (F1 D3: a single 0.6 s sample missed a
# real mutant whose turn fired ~1.2 s later).
IGNORED_IDLE_WINDOW = 5.0


# ---------------------------------------------------------------------------
# Loopback HTTP server plumbing (one shared, minimal synchronous handler)
# ---------------------------------------------------------------------------


def _bind_loopback() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _free_loopback() -> str:
    """A loopback origin that nothing serves (control for the credential site)."""
    return f"http://127.0.0.1:{_bind_loopback()}"


class _ServingHandler:
    """Minimal synchronous HTTP/1.1 handler; one worker thread per request.

    ``close_live()`` force-closes the connections that are open right now
    (shutdown(SHUT_RDWR)), so a request parked inside a response hold or an
    event long-poll aborts immediately instead of waiting out its bound.
    """

    def __init__(self, dispatch):
        self._dispatch = dispatch
        self.port = _bind_loopback()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._conns: set[socket.socket] = set()
        self._conns_lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self.last_error: str | None = None
        self.last_request: dict[str, str] = {}

    def _track(self, conn: socket.socket) -> None:
        with self._conns_lock:
            self._conns.add(conn)

    def _untrack(self, conn: socket.socket) -> None:
        with self._conns_lock:
            self._conns.discard(conn)

    def close_live(self) -> None:
        """Force-close every currently-open connection (held requests die)."""
        with self._conns_lock:
            conns = list(self._conns)
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def start(self) -> None:
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", self.port))
        server.listen(128)
        server.settimeout(0.2)  # keep the accept loop responsive to stop()

        def serve() -> None:
            while not self._stop.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self._track(conn)
                worker = threading.Thread(target=self._one, args=(conn,),
                                          daemon=True)
                self._workers.append(worker)
                worker.start()
            try:
                server.close()
            except OSError:
                pass

        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5.0
        probe = socket.socket()
        while time.monotonic() < deadline:
            try:
                probe.connect(("127.0.0.1", self.port))
                probe.close()
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError(f"loopback server did not come up on port {self.port}")

    def stop(self) -> None:
        self._stop.set()
        # Force-close live connections (e.g. a long-poll held open by the CLI)
        # so the listening socket and its per-connection threads actually die.
        self.close_live()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            assert not self._thread.is_alive(), "endpoint listener did not stop"
        deadline = time.monotonic() + 5.0
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        assert not any(w.is_alive() for w in self._workers), (
            "endpoint request worker survived bounded cleanup")
        assert not self._conns, "endpoint connections survived cleanup"

    def _one(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(30.0)
            raw = b""
            while b"\r\n\r\n" not in raw:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                raw += chunk
            head, _, body = raw.partition(b"\r\n\r\n")
            lines = head.decode("latin-1").splitlines()
            method, path, _protocol = lines[0].split(" ", 2)
            headers: dict[str, str] = {}
            for line in lines[1:]:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()
            self.last_request = headers
            length = int(headers.get("content-length", "0") or 0)
            while len(body) < length:
                more = conn.recv(65536)
                if not more:
                    break
                body += more
            try:
                status, out = self._dispatch(method, path, headers, body)
            except Exception as exc:
                raise RuntimeError(
                    f"dispatch failed: {method} {path} "
                    f"(content-length={headers.get('content-length')}, "
                    f"body={body[:300]!r})"
                ) from exc
            payload = out if isinstance(out, (bytes, bytearray)) else out.encode("utf-8")
            resp = (
                f"HTTP/1.1 {status} {'OK' if status == 200 else 'Error'}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode("latin-1") + payload
            conn.sendall(resp)
        except OSError:
            pass
        except Exception as exc:  # never let a fake endpoint hang the CLI
            self.last_error = repr(exc)
            try:
                conn.sendall(
                    b"HTTP/1.1 500 Error\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 2\r\nConnection: close\r\n\r\n{}"
                )
            except OSError:
                pass
        else:
            self.last_error = None
        finally:
            self._untrack(conn)
            conn.close()

class ZulipServer:
    """Loopback Zulip API with the actual protocol shapes, state and recording.

    ``seed_message`` pre-populates server-side history (a topic populated
    before the process starts).  ``push_message`` delivers a message event to
    the queue AND records it in history, so a subsequent history fetch already
    contains the trigger — exactly like a real server; tests must exclude it
    from "prior content" by ID, as the frozen journey requires.
    """

    def __init__(self, *, bot_id: int, user_ids: list[int]):
        self._handler = _ServingHandler(self._dispatch)
        self.queue_id = "q-00000000-0000-0000-0000-000000000000"
        self._events: list[dict] = []
        self._next_event_id = 0
        self._next_msg_id = 1000
        self._messages: dict[tuple[int, str], list[dict]] = {}
        self._bot_id = bot_id
        self._user_ids = user_ids
        self.register_calls = 0
        self.events_calls = 0
        self.history_calls: list[tuple[int, str]] = []
        self.sends: list[dict] = []
        self.send_headers: list[dict[str, str]] = []
        self.fail_history = False
        self.fail_send = False
        # Deterministic batch delivery (F1 D7): while held, an events request
        # blocks AFTER registering itself in the access log, so the test can
        # enqueue every message of one poll batch and release only then.  The
        # released request returns every queued event at once.
        self._poll_gate = threading.Event()
        self._poll_gate.set()
        self._poll_armed = threading.Event()
        self._delivered_ceiling: int | None = None
        self.last_send_response: dict = {}
        # Optional test hook: mutate/inspect the decoded form of every send
        # (e.g. prove a failed send never reaches the recorded ledger).
        self.on_send: "callable | None" = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._handler.port}"

    def hold_next_poll(self) -> None:
        """Block the next events request until :meth:`release_poll`."""
        self._poll_armed.clear()
        self._poll_gate.clear()

    def wait_poll_arrived(self, timeout: float) -> bool:
        """True once a held events request has reached the endpoint."""
        return self._poll_armed.wait(timeout)

    def release_poll(self) -> None:
        self._poll_gate.set()

    def events_pending(self) -> bool:
        """True while undelivered events remain in the endpoint queue."""
        return bool(self._events)

    # -- test-side state helpers ---------------------------------------------

    def seed_message(self, channel: int, topic: str, sender_id: int, content: str) -> int:
        msg_id = self._next_msg_id
        self._next_msg_id += 1
        self._messages.setdefault((channel, topic), []).append(
            {"id": msg_id, "type": "stream", "stream_id": channel,
             "subject": topic, "sender_id": sender_id, "content": content}
        )
        return msg_id

    def push_message(self, channel: int, topic: str, sender_id: int, content: str) -> int:
        """Deliver one message event to the queue AND record it in history.

        The next event id is floored at the highest id already handed to a
        poll (see ``_events_call``), so a queued message can never be lost by
        a concurrent long-poll that is about to be released empty.
        """
        msg_id = self._next_msg_id
        self._next_msg_id += 1
        message = {"id": msg_id, "type": "stream", "stream_id": channel,
                   "subject": topic, "sender_id": sender_id, "content": content}
        self._messages.setdefault((channel, topic), []).append(message)
        self._next_event_id = max(self._next_event_id,
                                  self._delivered_ceiling or -1) + 1
        self._events.append({"id": self._next_event_id, "type": "message",
                             "message": message})
        return msg_id

    def push_event(self, event: dict) -> None:
        """Push a raw event (e.g. a topic rename) that starts no work."""
        self._next_event_id = max(self._next_event_id,
                                  self._delivered_ceiling or -1) + 1
        event = dict(event)
        event.setdefault("id", self._next_event_id)
        self._events.append(event)

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._handler.start()

    def stop(self) -> None:
        # Release every lane first so a parked worker cannot outlive the
        # endpoint, then force-close any live connection (a parked poll dies
        # at once), then shut the listener down.  The journey's ``finally``
        # calls this after collecting/killing the child.
        self.release_poll()
        self._handler.close_live()
        self._handler.stop()

    # -- protocol dispatch ----------------------------------------------------

    def _dispatch(self, method, path, headers, body):
        parsed = urllib.parse.urlsplit(path)
        parts = parsed.path.split("/")
        if parts[1:3] != ["api", "v1"]:
            return 404, json.dumps({"result": "error", "msg": "not found"})
        resource = parts[3]
        if resource == "register" and method == "POST":
            return self._register()
        if resource == "events" and method == "GET":
            return self._events_call(parsed)
        if resource == "messages" and method == "GET":
            return self._history_call(parsed)
        if resource == "messages" and method == "POST":
            return self._send_call(body, headers)
        return 404, json.dumps({"result": "error", "msg": "not found"})

    def _register(self):
        self.register_calls += 1
        realm_users = [
            {"user_id": uid, "is_bot": False} for uid in self._user_ids
        ] + [{"user_id": self._bot_id, "is_bot": True}]
        return 200, json.dumps({
            "result": "success",
            "queue_id": self.queue_id,
            "last_event_id": 0,
            "realm_users": realm_users,
            "realm_bots": [{"bot_id": self._bot_id}],
        })

    def _events_call(self, parsed):
        held = not self._poll_gate.is_set()
        if held:
            # Armed hold: this long-poll request is live; tell the test it may
            # now enqueue its batch, then block until released.  The bound is
            # well under the client's per-request timeout so the held poll
            # never expires while armed.
            self._poll_armed.set()
            self._poll_gate.wait(timeout=10.0)
        self.events_calls += 1
        query = urllib.parse.parse_qs(parsed.query)
        last = int((query.get("last_event_id") or ["-1"])[0])
        events = [e for e in self._events if e["id"] > last]
        if held:
            # The released batch must be exactly what was enqueued while
            # armed; failing here is a harness-contract error, and the
            # queue must survive so the test can see what happened.
            assert events, "armed poll released with an empty batch"
            return 200, json.dumps({"result": "success", "events": events,
                                    "last_event_id": last})
        self._events = [e for e in self._events if e["id"] <= last]
        # Floor at what THIS response hands over: if the caller's cursor is
        # ahead (it saw events this queue no longer holds), keep the floor.
        newest = max(last, max((e["id"] for e in events), default=last))
        if self._delivered_ceiling is None:
            self._delivered_ceiling = newest
        else:
            self._delivered_ceiling = max(self._delivered_ceiling, newest)
        return 200, json.dumps({"result": "success", "events": events,
                                "last_event_id": newest})

    def _history_call(self, parsed):
        query = urllib.parse.parse_qs(parsed.query)
        narrow = json.loads((query.get("narrow") or ["null"])[0])
        channel = next(o["operand"] for o in narrow if o["operator"] == "channel")
        topic = next(o["operand"] for o in narrow if o["operator"] == "topic")
        channel, topic = int(channel), str(topic)
        self.history_calls.append((channel, topic))
        if self.fail_history:
            return 500, json.dumps({"result": "error", "msg": "injected failure"})
        messages = list(self._messages.get((channel, topic), []))
        return 200, json.dumps({"result": "success", "messages": messages})

    def _send_call(self, body, headers):
        self.send_headers.append(dict(headers))
        # aiohttp's dict `data` form-encodes the payload per the Zulip
        # send-message contract (type/to/topic/content as form fields).
        form = dict(urllib.parse.parse_qsl(body.decode("utf-8")))
        if self.on_send is not None:
            self.on_send(form)
        if self.fail_send:
            return 500, json.dumps({"result": "error", "msg": "injected send failure"})
        self.sends.append(form)
        # Real servers answer a successful send with the id AND author of the
        # created message (send-message contract).  The bot authorship of a
        # reply is therefore the authenticated request identity — this fake's
        # bot credentials — echoed as the sender; nothing in a form field can
        # invent another author.
        payload = {"result": "success",
                   "id": 5000 + len(self.sends),
                   "sender_id": self._bot_id}
        self.last_send_response = payload
        return 200, json.dumps(payload)


class ModelServer:
    """Loopback OpenAI-compatible streaming endpoint with recorded requests.

    ``hold_responses`` arms a real HTTP response gate: the NEXT request
    captured parks — connection open, unanswered — and the hold then
    DISARMS automatically so the released turn's own follow-up request (the
    runtime's natural next call) is answered normally instead of wedging the
    process.  While armed, every earlier request answers normally.  The
    parked request's body is captured, so a test can assert on it while it
    is still in flight.  The ENDPOINT OWNS the release: ``stop()`` — called
    by the journey's ``finally`` after the child is killed and reaped —
    releases the gate and force-closes parked connections, so a park can
    never outlive the test, and every armed wait inside the test is bounded
    well below the client's own timeouts.
    """

    def __init__(self, answer: str):
        self._handler = _ServingHandler(self._dispatch)
        self.answer = answer
        self.requests: list[dict] = []
        self._gate = threading.Event()   # set => release parked requests
        self._armed = False
        self._armed_index = 0
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._handler.port}"

    def start(self) -> None:
        self._handler.start()

    def stop(self) -> None:
        # Disarm, release any parked request, force-close live connections,
        # then shut the listener down (see class docstring).
        with self._lock:
            self._armed = False
        self._gate.set()
        self._handler.close_live()
        self._handler.stop()

    def hold_responses(self) -> int:
        """Park the NEXT model request; return its 1-based request number.

        The return value is the generation to pass to
        :meth:`wait_request_arrived`: when it is ``N``, the assertion is that
        request number ``N`` (and nothing older) is the held one.
        """
        with self._lock:
            self._armed = True
            self._armed_index = len(self.requests)
            self._gate.clear()
            return self._armed_index + 1

    def release_responses(self) -> None:
        with self._lock:
            self._armed = False
        self._gate.set()

    def wait_request_arrived(self, timeout: float = 10.0,
                             generation: int = 1) -> bool:
        """True once request #``generation`` has been CAPTURED (parked or not)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.requests) >= generation:
                return True
            time.sleep(0.02)
        return False

    def _dispatch(self, method, path, headers, body):
        if method == "POST" and path.rstrip("/").endswith("/chat/completions"):
            payload = json.loads(body.decode("utf-8"))
            with self._lock:
                park = self._armed and len(self.requests) >= self._armed_index
                if park:
                    # Park EXACTLY this one request; auto-disarm so later
                    # requests (the released turn's natural follow-ups)
                    # answer normally.
                    self._armed = False
                    self.requests.append(payload)
            if park:
                # The request bytes are captured; park THIS connection
                # without answering — "the model request is in flight".
                self._gate.wait(timeout=30.0)
            else:
                self.requests.append(payload)
            return 200, self._sse()
        return 404, json.dumps({"error": "not found"})

    def _sse(self) -> bytes:
        def chunk(obj):
            return f"data: {json.dumps(obj)}\n\n".encode("utf-8")

        return (
            chunk({"id": "c1", "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": self.answer},
                 "finish_reason": None}]})
            + chunk({"id": "c1", "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}]})
            + b"data: [DONE]\n\n"
        )


# ---------------------------------------------------------------------------
# Isolated environment + real CLI harness
# ---------------------------------------------------------------------------


@dataclass
class ZulipCliRun:
    """One bounded real-CLI subprocess with its endpoints and temp state."""

    # Session identity assertions read the PUBLIC session-log seam documented
    # in docs/INTERFACE-DEV-GUIDE.md ("Every Session writes a live JSONL log
    # to $SR2_HOME/logs/sessions/<UTC>-<session-id>-<suffix>.jsonl", one file
    # per live Session whose envelope carries ``session_id``).  Everything
    # expected of the identity is a public or test-owned fact: the agent
    # name, the exact resolved endpoint origin this run is configured
    # against, the integer channel, and the EXACT passed topic — all read
    # from the envelope's ``session_id``, never from a file-name pattern.
    # Files only DISCOVER the actual JSONL records; the seam's name
    # sanitization/64-character cap is incidental, so no naming-encoding
    # convention is asserted (the frozen spec declares none).

    workdir: Path
    sr2_home: Path
    config_path: Path
    zulip: ZulipServer
    model: ModelServer
    proc: subprocess.Popen | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    log_path: Path | None = None
    # Child output files: the launcher hands stdout/stderr to real files so
    # diagnostics survive even when the child dies mid-run (F1 D6).
    stdout_path: Path | None = None
    stderr_path: Path | None = None

    @property
    def session_log_dir(self) -> Path:
        return self.sr2_home / "logs" / "sessions"

    def child_output(self) -> str:
        """Everything the child wrote to stdout/stderr so far."""
        text = ""
        for path in (self.stdout_path, self.stderr_path):
            if path is not None and path.is_file():
                text += path.read_text(encoding="utf-8", errors="replace")
        return text

    def session_log_files(self, frame_fragment: str) -> list[Path]:
        """Session log files whose (sanitized) name carries *frame_fragment*."""
        if not self.session_log_dir.is_dir():
            return []
        return sorted(
            p for p in self.session_log_dir.iterdir()
            if p.suffix == ".jsonl" and frame_fragment in p.name
        )

    def session_log_events(self, frame_fragment: str) -> list[dict]:
        """All JSONL events across the matching session logs, in file order."""
        events: list[dict] = []
        for path in self.session_log_files(frame_fragment):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
        return events

    def session_start_envelopes(self) -> list[tuple[str, dict]]:
        """Every ``session.start`` envelope as (file name, envelope) pairs.

        Reads the PUBLIC session-log seam documented in
        ``docs/INTERFACE-DEV-GUIDE.md``: one live JSONL file per Session
        whose envelope carries ``session_id``.  This is the runtime record
        of the identity actually handed to ``Agent.create_session()``.  The
        file name only DISCOVERS the record; assertions read the envelope.
        """
        out: list[tuple[str, dict]] = []
        if not self.session_log_dir.is_dir():
            return out
        for path in sorted(self.session_log_dir.iterdir()):
            if path.suffix != ".jsonl":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                envelope = json.loads(line)
                if envelope.get("event") == "session.start":
                    out.append((path.name, envelope))
                    break
        return out

    def runtime_session_ids(self, agent: str) -> dict[str, str]:
        """Map every agent Session log's FILE name to its envelope ``session_id``.

        The envelope ``session_id`` (not the sanitized file name) is what an
        identity assertion reads; the file name only locates the record so a
        test can say "this Session log, this identity".
        """
        return {
            name: str(envelope.get("session_id"))
            for name, envelope in self.session_start_envelopes()
            if agent in str(envelope.get("session_id"))
        }

    def assert_topic_identity(self, agent: str, port: int, channel: int,
                              topic: str) -> tuple[str, str]:
        """Find one actual Session envelope with exact public frame identity.

        The public guide declares resolved origin, agent, integer channel
        and URL-escaped exact topic. The caller passes the original topic;
        encoding is performed here. File names only locate the JSONL record.
        """
        expected = (f"zulip:{agent}:http://127.0.0.1:{port}#{channel}#"
                    f"{urllib.parse.quote(topic, safe='')}")
        matches = [(str(e.get("session_id")), name)
                   for name, e in self.session_start_envelopes()
                   if e.get("session_id") == expected]
        assert len(matches) == 1, (
            f"expected one exact Session identity {expected!r}, found {matches}; "
            f"observed={self.runtime_session_ids(agent)}")
        return matches[0]

    def assert_reused_identity(self, agent: str, identity: str,
                               file_name: str) -> None:
        """The SAME actual identity, still exactly ONE Session log file.

        Repeated turns in one topic must reuse ONE runtime Session: one
        ``SessionLog`` is one file, so this file (and no other) still carries
        this exact envelope ``session_id`` and no second Session with the
        same identity was created.
        """
        ids = self.runtime_session_ids(agent)
        assert identity in ids.values(), (
            f"identity {identity!r} no longer has a session log: {ids}")
        same = [n for n, sid in ids.items() if sid == identity]
        assert len(same) == 1 and same[0] == file_name, (
            f"repeated turns spawned multiple Session logs for identity "
            f"{identity!r}: {same}")

    def count_identities(self, agent: str, topic: str) -> int:
        """Count actual Session identities for the exact original topic."""
        suffix = "#" + urllib.parse.quote(topic, safe="")
        return sum(sid.endswith(suffix)
                   for sid in set(self.runtime_session_ids(agent).values()))

    def identity_file_names(self, agent: str, identity: str) -> list[str]:
        """The file names whose envelope carries *identity* verbatim."""
        return [n for n, sid in self.runtime_session_ids(agent).items()
                if sid == identity]

    def model_messages(self) -> list[list[dict]]:
        """The message arrays of every model request, in order."""
        return [list(req.get("messages", [])) for req in self.model.requests]

    def wait_started(self, timeout: float = CLI_START_TIMEOUT) -> bool:
        """Wait until the CLI has registered its event queue (started)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # An events call proves registration happened even when the
            # registration response itself races past this check.
            if self.zulip.register_calls >= 1 or self.zulip.events_calls >= 1:
                return True
            if self.proc is not None and self.proc.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    def wait_idle(self, expected_replies: int, timeout: float = CLI_SETTLE_TIMEOUT) -> bool:
        """Wait until *expected_replies* bot sends have landed (bounded)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.zulip.sends) >= expected_replies:
                return True
            if self.proc is not None and self.proc.poll() is not None:
                return False
            time.sleep(0.1)
        return len(self.zulip.sends) >= expected_replies

    def wait_ignorable_delivered(self, timeout: float = CLI_SETTLE_TIMEOUT) -> None:
        """Block until the CLI's poll cycle is idle, so any queued message
        (delivered or not) has been consumed by the client."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return
            if not self.zulip._events and self.zulip._poll_gate.is_set():
                return
            time.sleep(0.1)

    def assert_quiescent(self, model_count: int, send_count: int,
                         window: float = IGNORED_IDLE_WINDOW) -> None:
        """Raise unless model and send counters hold *exactly* at
        (model_count, send_count) for one full continuous idle window.

        This is the teeth behind "ignored triggers produce no model requests":
        any new model request or reply within the window fails, instead of a
        sleep-once sample that returns True unconditionally (F1 D3).
        """
        deadline = time.monotonic() + window
        while time.monotonic() < deadline:
            assert len(self.model.requests) == model_count, (
                f"{len(self.model.requests) - model_count} model request(s) "
                f"appeared within the {window:.0f}s idle window: "
                f"{self.model.requests[model_count:]}")
            assert len(self.zulip.sends) == send_count, (
                f"{len(self.zulip.sends) - send_count} unexpected reply/ send "
                f"within the {window:.0f}s idle window: "
                f"{self.zulip.sends[send_count:]}")
            if self.proc is not None and self.proc.poll() is not None:
                raise AssertionError(
                    "CLI exited while waiting for the idle window; "
                    f"child output: {self.child_output()[-1500:]!r}")
            time.sleep(0.1)

    def wait_cold_batch(self, topics: list[tuple[int, str]], probes: int,
                        replies: int, timeout: float = CLI_SETTLE_TIMEOUT) -> bool:
        """Wait until every cold topic in *topics* was probed at least once and
        *replies* sends landed (the deterministic same-batch cold-topic bound)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            probed = set(self.zulip.history_calls)
            if (len(self.zulip.sends) >= replies
                    and all(t in probed for t in topics)):
                return True
            if self.proc is not None and self.proc.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    @property
    def log_text(self) -> str:
        """The child's log file (CLI logging lands there), if any."""
        if self.log_path is None or not self.log_path.is_file():
            return ""
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def collect(self, timeout: float = CLI_STOP_TIMEOUT) -> None:
        """Stop the bounded run: SIGTERM, bounded wait, kill and reap."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=timeout)
                self.timed_out = True
        self.returncode = self.proc.returncode
        self.stdout = self.child_output()


def _child_env(sr2_home: Path, model_pointer: Path) -> dict[str, str]:
    """Isolated child environment: worktree sources first, no ambient state.

    Inherits no SR2_HOME, SR2_AREA, SPECTRE_MEMORY_DSN, active-model pointer,
    project var or session-log override; PYTHONPATH names the worktree ``src``
    so the editable install cannot select another checkout.
    """
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("SR2_HOME", "SR2_AREA", "SPECTRE_MEMORY_DSN",
                     "SR2_ACTIVE_MODEL_FILE", "SR2_PROJECT", "SR2_SESSION_LOG_DIR",
                     "LITELLM_LOG")
    }
    env["SR2_HOME"] = str(sr2_home)
    env["SR2_ACTIVE_MODEL_FILE"] = str(model_pointer)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _write_zuliprc(path: Path, site: str) -> None:
    path.write_text(
        f"[zulip]\nemail = bot@e2e.local\nkey = synthetic-key-spc125\n"
        f"site = {site}\n",
        encoding="utf-8",
    )


def _write_topic_docs(root: Path) -> None:
    """Synthetic topic documents with area markers (real topic_area resolvers).

    ``cwd-area`` is what an accidental working-directory fallback would inject
    (the CLI's cwd basename IS cwd-area); the others are legitimate per-area
    identities. None may appear in a no-area turn's compiled request.
    """
    def topic(folder: str, area: str, marker: str) -> None:
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(
            f"---\nkind: topic\nid: topic:{area}\n---\n", encoding="utf-8")
        (d / "NOW.md").write_text(
            f"---\nkind: continuity\nfor: topic:{area}\nmode: source\n---\n"
            f"{marker} NOW\n", encoding="utf-8")
        (d / "AGENTS.md").write_text(f"{marker} AGENTS\n", encoding="utf-8")

    topic("sr2/spectre", "sr2-spectre", AREA_MARKER)
    topic("family/decoy", "decoy", DECOY_MARKER)
    topic("proj/cwd-area", "cwd-area", CWD_MARKER)


_CONFIG_TEMPLATE = """
agent:
  name: miranda
  default_skills: false
  tools: []
  skills: []
  skills_dirs: []
  mcp_servers: []

models:
  default:
    model: e2e-synthetic-model
    base_url: {base_url}
    api_key: synthetic-api-key

active_model: default
memory_store_dsn: ""
provenance_store_path: ""

pipeline:
  token_budget: 8192
  max_tool_iterations: 1
  layers:
    - name: area
      target: system
      degradation_category: plan_knowledge
      resolvers:
        - type: topic_area
          config:
            topics_root: {topics_root}
            filename: AGENTS.md
            max_tokens: 4000
        - type: topic_area
          config:
            topics_root: {topics_root}
            filename: NOW.md
            max_tokens: 4000
    - name: conversation
      target: messages
      resolvers:
        - type: session
        - type: input

zulip:
  zuliprc: {zuliprc}
  api_url: "{api_url}"
  allowed_user_ids: [{diego_id}, 9]
  channels:
{channels_yaml}
"""


@dataclass
class Harness:
    """Builds and owns one isolated Zulip-CLI journey environment."""

    workdir: Path
    nonempty_area: bool = False
    channels: dict[int, str] | None = None
    # spc-120 J1C: a journey's own topic documents (default _write_topic_docs)
    # and the rendered config file, which edit_config rewrites in place.
    topic_docs: Callable[[Path], None] | None = None
    config_file: Path | None = None

    def edit_config(self, edits: dict[str, str]) -> None:
        """Replace literal segments of the running CLI's config file; each
        source segment must occur exactly once."""
        assert self.config_file is not None, "harness not built yet"
        text = self.config_file.read_text(encoding="utf-8")
        for old, new in edits.items():
            assert text.count(old) == 1, f"config edit source not unique: {old!r}"
            text = text.replace(old, new)
        self.config_file.write_text(text, encoding="utf-8")

    def build(self) -> tuple[ZulipCliRun, Path, dict[str, str]]:
        """Return (run, child cwd, child env). The caller must launch+close."""
        workdir = self.workdir
        sr2_home = workdir / "sr2-home"
        cwd = workdir / "cwd-area"
        (sr2_home / "logs" / "sessions").mkdir(parents=True, exist_ok=True)
        cwd.mkdir(parents=True, exist_ok=True)

        zulip = ZulipServer(bot_id=BOT_ID, user_ids=[DIEGO_ID, OTHER_ID])
        model = ModelServer(answer=REPLY_MARKER)
        zulip.start()
        model.start()

        # ``site`` remains the canonical origin for message-link validation;
        # the identity uses the RESOLVED request origin (api_url override), so
        # the credential site is set to a second, distinct, never-served
        # loopback origin.  An identity built from the credential site rather
        # than the resolved origin is then distinguishable, and one that drops
        # the origin entirely fails the same runtime assertion.
        zulip.credential_site = _free_loopback()
        zuliprc = workdir / "bot.zuliprc"
        _write_zuliprc(zuliprc, zulip.credential_site)
        pointer = workdir / "active-model"
        pointer.write_text("default\n", encoding="utf-8")

        topics_root = workdir / "topics"
        topics_root.mkdir(parents=True, exist_ok=True)
        (self.topic_docs or _write_topic_docs)(topics_root)

        channels = self.channels
        if channels is None:
            channels = {CONFIGURED_CHANNEL: "sr2-spectre"} if self.nonempty_area \
                else {CONFIGURED_CHANNEL: ""}
        channels_yaml = "\n".join(
            f"    {ch}: {area!r}" for ch, area in channels.items())
        config_text = _CONFIG_TEMPLATE.format(
            base_url=f"{model.base_url}/v1",
            topics_root=str(topics_root),
            zuliprc=str(zuliprc),
            api_url=zulip.base_url,
            diego_id=DIEGO_ID,
            channels_yaml=channels_yaml,
        )
        config_path = workdir / "config.yaml"
        config_path.write_text(config_text, encoding="utf-8")
        self.config_file = config_path

        env = _child_env(sr2_home, pointer)
        env["PATH"] = str(REPO_ROOT / ".venv" / "bin") + os.pathsep + env.get("PATH", "")
        run = ZulipCliRun(workdir=workdir, sr2_home=sr2_home,
                          config_path=config_path, zulip=zulip, model=model)
        log_path = workdir / "cli.log"
        log_path.write_text("", encoding="utf-8")
        run.log_path = log_path
        return run, cwd, env

    def launch(self, run: ZulipCliRun, cwd: Path, env: dict[str, str]) -> None:
        # Child output goes to FILES, not pipes (F1 D6): diagnostics survive
        # even when the child crashes before the test inspects them.
        run.stdout_path = run.workdir / "child-stdout.txt"
        run.stderr_path = run.workdir / "child-stderr.txt"
        out = open(run.stdout_path, "w", encoding="utf-8")
        err = open(run.stderr_path, "w", encoding="utf-8")
        run.proc = subprocess.Popen(
            [str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "sr2_spectre.cli",
             str(run.config_path), "--interface", "zulip",
             "--log-file", str(run.workdir / "cli.log")],
            stdout=out, stderr=err, text=True,
            env=env, cwd=str(cwd),
        )
        out.close()
        err.close()

    def close(self, run: ZulipCliRun) -> None:
        # Stop the CHILD first (SIGTERM, bounded wait, kill, reap — already
        # inside collect()), then release every endpoint latch and close the
        # listening sockets: a parked request is always released or its
        # connection force-closed before the endpoint disappears.
        try:
            run.collect()
        finally:
            try:
                run.zulip.stop()
            finally:
                run.model.stop()
        assert run.proc is None or run.proc.poll() is not None, (
            "CLI survived bounded cleanup")
