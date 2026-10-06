"""Journey J1 (E2E, obsidian-c573): a bad MCP endpoint must not hang the process.

Boot race from the bead: on a reboot the Discord bots started before the glyph
container (localhost:8420/mcp) came up. The streamable-http transport's anyio
cancel scope raised ``asyncio.CancelledError`` out of ``MCPClient.connect()``;
``connect()`` only caught ``Exception``, so ``Runtime.initialize()`` never logged
'MCP server failed to connect' and the error escaped ``cli.main()``. Interpreter
shutdown then blocked on a leftover non-daemon thread: the process sat alive with
no Discord connection for ~8 hours while systemd reported a healthy unit, so
``Restart=always`` never fired.

Everything here runs the REAL user entry point — a subprocess running the real
``sr2_spectre.cli`` — through journey J1:

  a process configured with an MCP server whose endpoint refuses/aborts
  connections must not hang; startup either logs 'MCP server failed to connect'
  and proceeds, or fails fatally with a nonzero exit — and in BOTH cases the
  process must terminate within a bounded time.

Boundaries:
  Real — the CLI module entry point, config resolution, ``Runtime.initialize()``'s
    MCP path, ``MCPClient`` and the real streamable-http transport against a
    local endpoint on an ephemeral localhost port (no privileged port, no shared
    infra, no real bots).
  Substituted — the endpoint (a port that refuses, and one that accepts then
    aborts the connection mid-handshake); the LLM (pointed at the local relay's
    smallest model; J1 is about the startup path, so the model outcome is
    secondary); and, in the second test only, the stranded transport thread
    itself (a non-daemon sleeper standing in for the half-open transport's
    thread, so the hang AC3 guards against is actually reachable).

Expectations come from the bead, not the implementation: terminate within the
bound, exit code 0 or 1, and a visible startup record naming the MCP path.

Discrimination (this is a hang regression, and a runner that just waits can pass
on a broken build): ``test_j1_fatal_startup_with_stray_non_daemon_thread_*`` is
the red-on-base case — the base shape hangs at interpreter shutdown and is
reported as a timeout failure. Verified against both trees before handing off:
on the fixed build it exits 1 in ~1s; against the pre-fix sources it hangs until
killed. The plain-endpoint test proves the same bound along the unthreaded path
and that startup reaches the MCP path at all.

Isolation: a temp ``SR2_HOME`` plus a temp cwd mean tiers 1-3 never read
``~/.sr2`` or a project file; the positional config is the only tier present.
``SR2_ACTIVE_MODEL_FILE`` and ``SPECTRE_MEMORY_DSN`` are stripped from the child
environment. Nothing outside the temp dir is written.

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_startup_exit_e2e.py -v
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Upper bound for the full journey: MCP connect attempts plus, if startup gets
# that far, one small model call. Generous on purpose — the failure under test is
# a process that never exits, so this must be a bound, not a performance gate.
JOURNEY_TIMEOUT_SECONDS = 180.0

# Bound for the stray-thread case. The fix exits within ~2s (no retry, no model
# call); anything still alive after this is the 8-hour hang, expressed small.
STRAY_THREAD_TIMEOUT_SECONDS = 60.0

# The LLM is a substituted boundary for J1: startup is what is under test, so the
# smallest model on the local relay is used (discovered via
# GET http://localhost:11434/v1/models — qwen3.8-flash).
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen3.8-flash"

# Startup records that name the MCP path. Contract branch (a) is the Runtime
# warning; branch (b) is a fatal record whose traceback names the MCP client. A
# run that dies for an unrelated reason (bad config, import error) matches
# neither, which is what keeps this journey from passing on a build that never
# reached the MCP path.
_WARNING_MARKER = "MCP server failed to connect"
_FATAL_MARKERS = ("Fatal error", "Traceback (most recent call last)")
_MCP_PATH_MARKERS = (
    "sr2_spectre/mcp/client.py",
    "sr2_spectre.mcp.client",
    "MCPConnectionError",
)

# A non-daemon sleeper: the process cannot exit until it finishes, so a shutdown
# path that waits on non-daemon threads hangs. Stands in for the thread the
# half-opened transport stranded in the field incident.
_STRAY_THREAD = (
    "import threading, time\n"
    "threading.Thread(target=lambda: time.sleep(600), "
    "name='half-open-transport').start()\n"
)


# ---------------------------------------------------------------------------
# Fakes: local MCP endpoints on ephemeral ports
# ---------------------------------------------------------------------------


@dataclass
class FakeEndpoint:
    """A local endpoint the child fails against, plus a teardown hook."""

    url: str
    close: Callable[[], None]


def _endpoint_refuses() -> FakeEndpoint:
    """An ephemeral port with nothing listening: the connect is refused."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return FakeEndpoint(url=f"http://127.0.0.1:{port}/mcp", close=lambda: None)


def _endpoint_aborts_mid_handshake() -> FakeEndpoint:
    """An ephemeral port that accepts and then aborts the connection (RST).

    This is the boot-race shape from the bead: something is listening, then the
    connection dies under the transport while the handshake is in flight, so the
    transport's cancel scope fires mid-sequence.
    """
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    port = listener.getsockname()[1]
    stop = threading.Event()

    def _serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            time.sleep(0.02)
            # SO_LINGER with a zero timeout sends RST instead of a clean FIN.
            conn.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00" + b"\x00" * 8
            )
            conn.close()

    threading.Thread(target=_serve, name="fake-mcp-abort", daemon=True).start()

    def _close() -> None:
        stop.set()
        try:
            listener.close()
        except OSError:
            pass

    return FakeEndpoint(url=f"http://127.0.0.1:{port}/mcp", close=_close)


# ---------------------------------------------------------------------------
# Isolated config + time-bounded process run
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = """
agent:
  name: e2e-c573
  default_skills: false
  tools: []
  skills: []
  skills_dirs: []
  mcp_servers:
    - name: fake-endpoint
      type: streamable-http
      url: "{mcp_url}"

models:
  default:
    model: {model}
    base_url: {base_url}
    api_key: dummy

active_model: default
provenance_store_path: ""

pipeline:
  token_budget: 8192
  max_tool_iterations: 2
  layers:
    - name: system
      target: system
      resolvers:
        - type: static
          config:
            text: "Reply with exactly the word PONG and nothing else."
    - name: tools
      target: tools
      resolvers: []
      tool_providers: []
    - name: conversation
      target: messages
      resolvers:
        - type: session
        - type: input
"""


@dataclass
class BoundedRun:
    """Outcome of a time-bounded process run: it exited, or it hung."""

    returncode: int | None
    elapsed_seconds: float
    stdout: str
    stderr: str
    log_text: str
    timed_out: bool

    @property
    def all_text(self) -> str:
        return f"{self.stdout}\n{self.stderr}\n{self.log_text}"

    def tail(self, chars: int = 1500) -> str:
        return (
            f"exit={self.returncode} elapsed={self.elapsed_seconds:.1f}s "
            f"stdout_tail={self.stdout[-chars:]!r} "
            f"stderr_tail={self.stderr[-chars:]!r} log_tail={self.log_text[-chars:]!r}"
        )


def _child_env(sr2_home: Path) -> dict[str, str]:
    """Child environment: isolated SR2_HOME, worktree sources, no ambient config.

    ``PYTHONPATH`` must name the worktree's ``src``: the project venv resolves
    ``sr2_spectre`` through a ``.pth`` pointing at the main checkout, so without
    it the child would run that branch's code instead of this one's.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("SR2_HOME", "SR2_ACTIVE_MODEL_FILE", "SPECTRE_MEMORY_DSN")
    }
    env["SR2_HOME"] = str(sr2_home)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _run_bounded(
    cmd: list[str], *, env: dict[str, str] | None, cwd: Path, timeout: float
) -> BoundedRun:
    """Run ``cmd``, killing it if it outlives ``timeout``.

    A timeout is reported, not raised: the assertion — not an incidental
    TimeoutExpired traceback — must be what fails when the process hangs.
    """
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(cwd),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        returncode: int | None = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        returncode, timed_out = None, True
    return BoundedRun(
        returncode=returncode,
        elapsed_seconds=time.monotonic() - started,
        stdout=stdout or "",
        stderr=stderr or "",
        log_text="",
        timed_out=timed_out,
    )


def _write_isolated_config(workdir: Path, mcp_url: str) -> tuple[Path, Path, Path]:
    """Write the single-tier config and return (sr2_home, config, run_dir)."""
    sr2_home = workdir / "sr2-home"
    sr2_home.mkdir(parents=True, exist_ok=True)
    config_path = workdir / "config.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            mcp_url=mcp_url, model=LLM_MODEL, base_url=LLM_BASE_URL
        ),
        encoding="utf-8",
    )
    run_dir = workdir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return sr2_home, config_path, run_dir


def _cli_args(config_path: Path, log_file: Path) -> list[str]:
    return [
        str(config_path),
        "ping",
        "--interface",
        "single_shot",
        "--log-file",
        str(log_file),
    ]


def _finish(result: BoundedRun, log_file: Path) -> BoundedRun:
    result.log_text = (
        log_file.read_text(encoding="utf-8", errors="replace")
        if log_file.exists()
        else ""
    )
    return result


def _run_spectre(endpoint: FakeEndpoint, workdir: Path, log_file: Path) -> BoundedRun:
    """Launch the real CLI module against ``endpoint`` in an isolated home/cwd."""
    sr2_home, config_path, run_dir = _write_isolated_config(workdir, endpoint.url)
    result = _run_bounded(
        [sys.executable, "-m", "sr2_spectre.cli", *_cli_args(config_path, log_file)],
        env=_child_env(sr2_home),
        cwd=run_dir,
        timeout=JOURNEY_TIMEOUT_SECONDS,
    )
    return _finish(result, log_file)


def _assert_terminated(result: BoundedRun) -> None:
    """The bead's invariant: the process went away, promptly, on its own status."""
    assert not result.timed_out, (
        "spectre was still alive after "
        f"{JOURNEY_TIMEOUT_SECONDS:.0f}s and had to be killed — the obsidian-c573 "
        "failure: a process sitting up with no Discord connection while systemd "
        f"reported it healthy for ~8h. {result.tail()}"
    )
    assert result.returncode in (0, 1), (
        "a dead MCP endpoint must produce a normal exit status (0 after warning "
        f"and proceeding, or 1 after a fatal startup error); got "
        f"{result.returncode!r} (negative = killed by a signal). {result.tail()}"
    )


def _assert_mcp_startup_record(result: BoundedRun) -> None:
    """The failure must be visible and attributable to the MCP path."""
    text = result.all_text
    warned = _WARNING_MARKER in text
    fatal = any(marker in text for marker in _FATAL_MARKERS)
    assert warned or fatal, (
        "expected either the Runtime warning 'MCP server failed to connect' or a "
        f"fatal startup record in the log/stderr; got neither. {result.tail()}"
    )
    assert any(marker in text for marker in _MCP_PATH_MARKERS), (
        "the startup record must name the MCP connect path (the warning, or a "
        "traceback through sr2_spectre/mcp/client.py); otherwise this test would "
        f"pass on any unrelated startup crash. {result.tail(2500)}"
    )


@pytest.fixture
def workdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="spectre-c573-e2e-") as tmp:
        yield Path(tmp)


# ---------------------------------------------------------------------------
# J1 — startup against a dead MCP endpoint: warns or fails fatally, never hangs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint_factory",
    [_endpoint_refuses, _endpoint_aborts_mid_handshake],
    ids=["refused", "aborted-mid-handshake"],
)
def test_j1_startup_with_dead_mcp_endpoint_terminates(
    endpoint_factory: Callable[[], FakeEndpoint], workdir: Path
) -> None:
    """The real CLI, started with an unreachable MCP endpoint, terminates.

    Per the bead either outcome is correct: (a) ``Runtime.initialize()`` logs
    'MCP server failed to connect' and the run proceeds to a turn, or (b) startup
    fails fatally with a nonzero exit. What is never correct is staying alive.
    """
    endpoint = endpoint_factory()
    log_file = workdir / "spectre.log"
    try:
        result = _run_spectre(endpoint, workdir, log_file)
    finally:
        endpoint.close()

    _assert_terminated(result)
    _assert_mcp_startup_record(result)


# ---------------------------------------------------------------------------
# J1/AC3 — fatal startup with a live non-daemon thread: exits, does not hang
# ---------------------------------------------------------------------------


def test_j1_fatal_startup_with_stray_non_daemon_thread_exits_nonzero(
    workdir: Path,
) -> None:
    """The 8-hour shape, expressed small: fatal startup error + live non-daemon
    thread must still terminate with a nonzero status.

    This is the subprocess-level proof of AC3 through the real entry point and the
    discriminating case for the regression: with the pre-fix ``cli.main()`` the
    exception propagates to the interpreter's default shutdown path, which joins
    the non-daemon thread, and the process is still alive when this test kills it.
    The stranded transport thread is substituted with a sleeper so the hang is
    reachable on demand instead of depending on a timing race in the transport.

    Uses ``-c`` to start the thread before invoking the real ``cli.main()``; the
    config, config resolution, Runtime MCP path and transport are the real ones.
    """
    endpoint = _endpoint_refuses()
    log_file = workdir / "spectre.log"
    try:
        sr2_home, config_path, run_dir = _write_isolated_config(workdir, endpoint.url)
        bootstrap = (
            "import sys\n"
            "sys.argv = ['sr2-spectre'] + sys.argv[1:]\n"
            + _STRAY_THREAD
            + "from sr2_spectre.cli import main\n"
            "main()\n"
        )
        result = _run_bounded(
            [sys.executable, "-c", bootstrap, *_cli_args(config_path, log_file)],
            env=_child_env(sr2_home),
            cwd=run_dir,
            timeout=STRAY_THREAD_TIMEOUT_SECONDS,
        )
        result = _finish(result, log_file)
    finally:
        endpoint.close()

    assert not result.timed_out, (
        "fatal startup with a live non-daemon thread hung at interpreter exit "
        f"({STRAY_THREAD_TIMEOUT_SECONDS:.0f}s bound): the process must terminate "
        "so systemd Restart=always fires. " + result.tail()
    )
    assert result.returncode not in (0, None), (
        "a fatal startup error must exit with a nonzero status — even with a "
        f"non-daemon thread still alive — so Restart=always fires; got "
        f"{result.returncode!r}. " + result.tail()
    )
    _assert_mcp_startup_record(result)
