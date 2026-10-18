"""Tests for the 'dispatch' subcommand (headless run + JSON envelope).

Unit tests cover the response-contract parsing (final block, VERDICT, WROTE,
status -> exit code) with no LLM. The e2e test runs the real CLI entry point
in a subprocess — isolated temp SR2_HOME, no ambient config — pointed at the
local relay's smallest model, following the J1 journey pattern
(test_cli_startup_exit_e2e.py).

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_dispatch.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from sr2_spectre.dispatch import (
    DispatchEnvelope,
    STATUS_EXIT_CODES,
    classify,
    extract_wrote,
    final_message,
    parse_verdict,
)

# Local relay, smallest model (J1 pattern).
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen3.8-flash"
E2E_TIMEOUT_SECONDS = 120.0


# ---------------------------------------------------------------------------
# Unit: final block extraction
# ---------------------------------------------------------------------------


def test_final_message_takes_last_blank_separated_block():
    output = "chatter\n\nmore chatter\n\nVERDICT: DONE\nWROTE /tmp/a.md"
    assert final_message(output) == "VERDICT: DONE\nWROTE /tmp/a.md"


def test_final_message_single_block_and_blank_lines():
    assert final_message("only block") == "only block"
    assert final_message("block\n\n\n\n") == "block"
    assert final_message("") == ""
    assert final_message("\n\n  \n") == ""


def test_final_message_blank_lines_inside_blocks_are_preserved():
    # Single newlines are intra-block; only blank lines split.
    assert final_message("a\nb\n\nfinal\nline") == "final\nline"


# ---------------------------------------------------------------------------
# Unit: verdict parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("VERDICT: DONE", "done"),
        ("VERDICT: FAILED", "failed"),
        ("VERDICT: NEEDS-INPUT", "needs-input"),
        ("VERDICT: ESCALATION", "escalation"),
        ("VERDICT: done\nanswer text", "done"),  # case-insensitive
        ("Some preamble\n\nVERDICT: FAILED\nbecause X", "failed"),
        ("no verdict here", None),
        ("VERDICT: SOMETHING-NEW", None),  # unknown keyword is not a verdict
        # A verdict in an EARLIER block is not the final answer.
        ("VERDICT: DONE\n\nlater answer without verdict", None),
        # spc-947: a trailing reason on the same line classifies by keyword.
        ("VERDICT: ESCALATION: tests need a DB", "escalation"),
        ("VERDICT: FAILED — all 12 tests pass", "failed"),
        ("VERDICT: NEEDS-INPUT: which branch should I use?", "needs-input"),
        ("VERDICT: DONE — all 12 tests pass", "done"),
        ("Some preamble\n\nVERDICT: ESCALATION: tests need a DB", "escalation"),
        # Lowercase keyword prefix (skill contract: case-insensitive).
        ("verdict: needs-input", "needs-input"),
        ("verdict: escalation: build broke", "escalation"),
        # Unknown keyword with trailing text is still not a verdict.
        ("VERDICT: SOMETHING-NEW: whatever", None),
    ],
)
def test_parse_verdict(message, expected):
    assert parse_verdict(message) == expected


def test_classify_trailing_reason_and_case():
    """spc-947: classify() must see through trailing reason text and case."""
    assert classify("VERDICT: ESCALATION: tests need a DB") == "escalation"
    assert classify("VERDICT: FAILED — tests broke") == "failed"
    assert classify("verdict: needs-input") == "needs-input"
    # Unknown keyword (with or without a reason) keeps the lenient done.
    assert classify("VERDICT: SOMETHING-NEW: whatever") == "done"
    assert classify("no verdict line") == "done"


def test_extract_wrote():
    assert extract_wrote("WROTE /tmp/a.md\nWROTE /tmp/b.txt") == [
        "/tmp/a.md",
        "/tmp/b.txt",
    ]
    assert extract_wrote("no artifacts") == []
    assert extract_wrote("first\n\nVERDICT: DONE\nWROTE /tmp/x.md") == ["/tmp/x.md"]


def test_classify_precedence():
    assert classify("whatever", timed_out=True) == "timeout"
    assert classify("VERDICT: DONE", error=True) == "error"
    assert classify("VERDICT: DONE") == "done"
    assert classify("VERDICT: NEEDS-INPUT\nwhat do you want?") == "needs-input"
    # Lenient: a clean run without a VERDICT line is done.
    assert classify("plain answer, no contract") == "done"


def test_status_exit_codes_are_distinct():
    assert len(set(STATUS_EXIT_CODES.values())) == len(STATUS_EXIT_CODES)
    assert STATUS_EXIT_CODES["done"] == 0


def test_envelope_json_round_trips():
    env = DispatchEnvelope(
        agent="liara",
        session_id="dispatch-liara-123",
        status="done",
        exit_code=0,
        duration_s=1.5,
        message="VERDICT: DONE\nWROTE /tmp/a.md",
        wrote=["/tmp/a.md"],
        wrote_missing=[],
        raw_stdout="",
    )
    payload = json.loads(env.to_json())
    assert payload == asdict(env)


# ---------------------------------------------------------------------------
# E2E: real CLI subprocess against the local relay
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = """
agent:
  name: e2e-dispatch
  default_skills: false
  tools: []
  skills: []
  skills_dirs: []
  mcp_servers: []

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
            text: "You are a test agent. Follow the response contract exactly."
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

_E2E_PROMPT = "Reply with exactly this one line and nothing else: VERDICT: DONE"


# Variant of the e2e config with the terminal tool enabled — needed to
# reproduce a timeout that arrives while a tool subprocess is running.
_TIMEOUT_CONFIG_TEMPLATE = (
    _CONFIG_TEMPLATE.replace(
        "  tools: []\n",
        "  tools:\n"
        "    - name: terminal\n"
        "      class_path: sr2_spectre.tools.builtins.terminal.TerminalTool\n"
        "      config:\n"
        "        timeout: 300\n",
    )
    # The tools layer must advertise the registry or the model sees no tools.
    .replace("      tool_providers: []\n", "      tool_providers:\n        - type: spectre_tools\n")
)


def _write_isolated_home(
    workdir: Path, template: str = _CONFIG_TEMPLATE
) -> tuple[Path, Path]:
    """Temp SR2_HOME with an agents dir + agent yaml; returns (home, agent yaml)."""
    home = workdir / "sr2-home"
    agents = home / "agents"
    agents.mkdir(parents=True)
    (agents / "e2e-dispatch.yaml").write_text(
        template.format(model=LLM_MODEL, base_url=LLM_BASE_URL),
        encoding="utf-8",
    )
    return home, agents


def _child_env(sr2_home: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("SR2_HOME", "SR2_ACTIVE_MODEL_FILE", "SPECTRE_MEMORY_DSN")
    }
    env["SR2_HOME"] = str(sr2_home)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _run_dispatch_cli(
    argv: list[str], *, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess:
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "sr2_spectre.cli", "dispatch", *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        timeout=E2E_TIMEOUT_SECONDS,
    )
    proc.elapsed_seconds = time.monotonic() - started  # type: ignore[attr-defined]
    return proc


def test_dispatch_e2e_done_envelope_to_stdout(tmp_path):
    home, agents = _write_isolated_home(tmp_path)
    env = _child_env(home)

    proc = _run_dispatch_cli(
        [
            "--agent", "e2e-dispatch",
            "--agents-dir", str(agents),
            "--prompt", _E2E_PROMPT,
            "--timeout", "90",
        ],
        env=env,
        cwd=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr[-2000:]
    envelope = json.loads(proc.stdout)
    assert envelope["agent"] == "e2e-dispatch"
    assert envelope["status"] == "done"
    assert envelope["exit_code"] == 0
    assert envelope["session_id"].startswith("dispatch-e2e-dispatch-")
    assert "VERDICT: DONE" in envelope["message"]
    assert envelope["wrote"] == []
    assert envelope["duration_s"] >= 0


def test_dispatch_e2e_out_file(tmp_path):
    home, agents = _write_isolated_home(tmp_path)
    env = _child_env(home)
    out = tmp_path / "nested" / "result.json"

    proc = _run_dispatch_cli(
        [
            "--agent", "e2e-dispatch",
            "--agents-dir", str(agents),
            "--prompt", _E2E_PROMPT,
            "--out", str(out),
            "--timeout", "90",
        ],
        env=env,
        cwd=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr[-2000:]
    envelope = json.loads(out.read_text(encoding="utf-8"))
    assert envelope["status"] == "done"
    # With --out, stdout stays clean (no JSON mixed in for callers to parse).
    assert proc.stdout.strip() == ""


def test_dispatch_unknown_agent_exits_error(tmp_path):
    home, _ = _write_isolated_home(tmp_path)
    env = _child_env(home)

    proc = _run_dispatch_cli(
        ["--agent", "no-such-agent", "--prompt", "hi"],
        env=env,
        cwd=tmp_path,
    )
    assert proc.returncode == STATUS_EXIT_CODES["error"]
    assert "config error" in proc.stderr


def _surviving_pids(marker: str) -> list[int]:
    """PIDs of running processes whose /proc cmdline contains *marker*."""
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if marker.encode() in cmdline:
            pids.append(int(entry.name))
    return pids


def test_dispatch_e2e_timeout_leaves_no_surviving_children(tmp_path):
    """spc-947 defect 3: a turn timeout while a terminal command is running
    must leave no surviving child process after dispatch exits.

    The agent is told to write a marker file (its own cwd) before starting
    a long sleep; the marker proves the command really was running when the
    timeout fired, and the /proc scan proves nothing from it survived.
    """
    home, agents = _write_isolated_home(tmp_path, template=_TIMEOUT_CONFIG_TEMPLATE)
    env = _child_env(home)
    marker = tmp_path / "started.txt"
    prompt = (
        "Use the terminal tool to run EXACTLY this one command "
        "(a single terminal tool call):\n"
        f"echo started > {marker} && sleep 39.7\n"
        "Wait for it to finish; do not stop early."
    )

    proc = _run_dispatch_cli(
        [
            "--agent", "e2e-dispatch",
            "--agents-dir", str(agents),
            "--prompt", prompt,
            "--timeout", "20",
        ],
        env=env,
        cwd=tmp_path,
    )

    assert proc.returncode == 4, proc.stderr[-2000:]
    envelope = json.loads(proc.stdout)
    assert envelope["status"] == "timeout"
    assert marker.exists(), "terminal command never started; test is a no-op"
    assert _surviving_pids("39.7") == []


def test_dispatch_prompt_and_prompt_file_are_mutually_exclusive(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, "-m", "sr2_spectre.cli", "dispatch",
            "--agent", "x",
            "--prompt", "a",
            "--prompt-file", "/tmp/does-not-matter",
        ],
        capture_output=True,
        text=True,
        env=_child_env(tmp_path),
        cwd=str(tmp_path),
    )
    assert proc.returncode != 0
    assert "mutually exclusive" in proc.stderr
