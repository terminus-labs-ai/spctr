"""spc-1112 (E2E): a written envelope must not be followed by a shutdown hang.

The field failure: a dispatch turn whose terminal tool timed out three times
left a non-daemon thread behind (release bcec676185e9, session
build-kodiak-123-20261121T023608Z). ``run_dispatch`` completed, wrote the
envelope, and returned; ``cli.main()`` then called ``sys.exit()``, interpreter
shutdown joined the leftover thread, and the process never exited — "The
executor did not finish joining its threads within 300 seconds", envelope on
disk, runner blocked. The bead runner now kills such children (vault-280);
this is the fix for the hang itself: the dispatch branch flushes stdio and
logging and leaves via ``os._exit``, skipping the thread joins.

Everything here runs the REAL user entry point — a subprocess running the
real ``spctr.cli`` through ``dispatch`` — on the invariant:

  once the envelope exists, the process exits with run_dispatch's exit code
  within a bounded time, no matter what non-daemon thread a tool left alive.

Boundaries:
  Real — the CLI module entry point, config resolution, the dispatch turn,
    the tool registry and execution, envelope writing, and the interpreter
    shutdown path under test.
  Substituted — the LLM (a minimal OpenAI-compatible fake served by this file
    on an ephemeral localhost port, scripted to make exactly one tool call
    and then answer with the VERDICT line) and the leftover thread itself
    (a fake tool that starts a 600 s non-daemon sleeper and returns, the
    same stand-in J1 uses for the half-open transport's thread).

Expectations come from the bead, not the implementation: the process is gone
within the bound, its exit code is the envelope's (0 for a done run), and
the envelope file parses intact with the agent's final message.

Discrimination: against the pre-fix sources the child sits in interpreter
shutdown until killed, so the bounded run reports a timeout failure; on the
fixed build it exits in seconds. Verified red-on-base before handing off.

Isolation: a temp SR2_HOME plus a temp cwd mean no ambient ``~/.sr2`` or
project config is read; the agent YAML under a temp agents dir is the only
tier present. ``SR2_ACTIVE_MODEL_FILE`` and ``SPCTR_MEMORY_DSN`` are stripped
from the child environment. Nothing outside the temp dir is written.

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_dispatch_shutdown_e2e.py -v
"""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Upper bound for the whole child run: one local scripted model exchange, one
# instant tool call, one write. Generous because the failure under test is a
# process that NEVER exits — this must be a bound, not a performance gate.
BOUNDS_TIMEOUT_SECONDS = 90.0

LLM_MODEL = "fake-dispatch-model"

# The fake tool: __call__ starts a non-daemon sleeper and returns immediately,
# standing in for the timed-out tool call that stranded a live thread in the
# field incident. The sleeper's name shows up in the "still alive" assertion's
# context; its duration must exceed the test bound by a wide margin.
_FAKE_TOOL_MODULE = '''
"""Fake tool for spc-1112: leaves a non-daemon thread alive, returns at once."""
import threading
import time


class StrayThreadTool:
    name = "stray_thread"
    description = "Starts a long-lived background thread and returns 'ok'."
    input_schema = {"type": "object", "properties": {}, "required": []}

    def __call__(self, **kwargs) -> str:
        t = threading.Thread(
            target=lambda: time.sleep(600), name="leftover-tool-thread"
        )
        t.start()
        return "ok"
'''

_CONFIG_TEMPLATE = """
agent:
  name: e2e-stray
  default_skills: false
  tools:
    - name: stray_thread
      class_path: fake_stray_tool.StrayThreadTool
  skills: []
  skills_dirs: []
  mcp_servers: []

models:
  default:
    model: {model}
    api_key: dummy
    base_url: {base_url}

active_model: default
provenance_store_path: ""

pipeline:
  token_budget: 8192
  max_tool_iterations: 4
  layers:
    - name: system
      target: system
      resolvers:
        - type: static
          config:
            text: "You are a test agent. Call the stray_thread tool once, then end with the line VERDICT: DONE."
    - name: tools
      target: tools
      resolvers: []
      tool_providers:
        - type: spctr_tools
    - name: conversation
      target: messages
      resolvers:
        - type: session
        - type: input
"""


def _sse(*chunks: dict) -> bytes:
    """OpenAI SSE body: one data line per chunk, then the terminal marker."""
    return b"".join(
        f"data: {json.dumps(c)}\n\n".encode() for c in chunks
    ) + b"data: [DONE]\n\n"


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "fake",
        "object": "chat.completion.chunk",
        "model": LLM_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _start_fake_llm():
    """Scripted OpenAI-compatible endpoint on an ephemeral port.

    First completion: the assistant calls ``stray_thread`` with no arguments.
    Later completions: the final answer ``VERDICT: DONE``. Counts requests so
    the test can prove the tool loop really ran.
    """
    state = {"calls": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self.send_error(404)
                return
            state["calls"] += 1
            if state["calls"] == 1:
                body = _sse(
                    _chunk({"role": "assistant", "content": ""}),
                    _chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stray_1",
                                    "type": "function",
                                    "function": {
                                        "name": "stray_thread",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        }
                    ),
                    _chunk({}, finish="tool_calls"),
                )
            else:
                body = _sse(
                    _chunk({"role": "assistant", "content": "all done"}),
                    _chunk({"content": "\\n\\nVERDICT: DONE"}),
                    _chunk({}, finish="stop"),
                )
            payload = body.encode() if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    return server, base_url, state


def _child_env(sr2_home: Path, fake_tool_dir: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("SR2_HOME", "SR2_ACTIVE_MODEL_FILE", "SPCTR_MEMORY_DSN")
    }
    env["SR2_HOME"] = str(sr2_home)
    # The worktree's src first (the venv's .pth points at the main checkout),
    # plus the dir holding the fake tool module for class_path import.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(fake_tool_dir)]
    )
    return env


def test_dispatch_exits_after_envelope_despite_stray_non_daemon_thread(tmp_path):
    server, base_url, llm_state = _start_fake_llm()
    try:
        sr2_home = tmp_path / "sr2-home"
        agents = sr2_home / "agents"
        agents.mkdir(parents=True)
        (agents / "e2e-stray.yaml").write_text(
            _CONFIG_TEMPLATE.format(model=LLM_MODEL, base_url=base_url),
            encoding="utf-8",
        )
        fake_dir = tmp_path / "faketool"
        fake_dir.mkdir()
        (fake_dir / "fake_stray_tool.py").write_text(
            _FAKE_TOOL_MODULE, encoding="utf-8"
        )
        out = tmp_path / "envelope.json"

        started = time.monotonic()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "spctr.cli",
                "dispatch",
                "--agent",
                "e2e-stray",
                "--agents-dir",
                str(agents),
                "--prompt",
                "Call the stray_thread tool once, then answer VERDICT: DONE.",
                "--timeout",
                "60",
                "--out",
                str(out),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(sr2_home, fake_dir),
            cwd=str(tmp_path),
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=BOUNDS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            timed_out = True
        elapsed = time.monotonic() - started

        ctx = (
            f"exit={proc.returncode} elapsed={elapsed:.1f}s "
            f"llm_calls={llm_state['calls']} "
            f"stdout_tail={(stdout or '')[-800:]!r} "
            f"stderr_tail={(stderr or '')[-1200:]!r}"
        )

        # The invariant: with the envelope written, the process goes away —
        # the leftover non-daemon thread must not hold interpreter shutdown.
        assert not timed_out, (
            "dispatch hung at shutdown after writing its envelope — the "
            "spc-1112 failure: sys.exit() joining the leftover tool thread "
            f"forever. {ctx}"
        )
        # Exit code comes from run_dispatch (done → 0), proving os._exit took
        # it rather than a crash or a kill.
        assert proc.returncode == 0, f"expected run_dispatch's exit code 0; {ctx}"
        # The envelope is intact and complete.
        assert out.exists(), f"no envelope written; {ctx}"
        envelope = json.loads(out.read_text(encoding="utf-8"))
        assert envelope["status"] == "done", envelope
        assert envelope["exit_code"] == 0, envelope
        assert "VERDICT: DONE" in envelope["message"], envelope
        # Discrimination guards: the scripted model made the turn exercise the
        # tool loop, so the stray thread really existed before shutdown.
        assert llm_state["calls"] >= 2, (
            f"fake LLM saw {llm_state['calls']} completion(s); the tool-call "
            f"round-trip never happened, so no stray thread was created. {ctx}"
        )
    finally:
        server.shutdown()
