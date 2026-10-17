"""Dispatch subcommand — headless single-shot run with a machine-readable envelope.

Usage:
    sr2-spectre dispatch --agent liara --prompt "..."
    sr2-spectre dispatch --agent liara --prompt-file /tmp/p.txt --out /tmp/result.json

Runs one single_shot-style turn in-process (same config resolution, same
RunContext as ``--interface single_shot``) and emits a JSON envelope:
the agent's final message, a verdict parsed from the response contract,
extracted ``WROTE`` artifact paths, and a status mapped to a process
exit code.  The envelope goes to ``--out`` (a file) or to stdout.

The intended response contract (embedded by the dispatcher in the prompt)
is: end with one block, first line ``VERDICT: <DONE|FAILED|NEEDS-INPUT|
ESCALATION>``, optional ``WROTE <absolute-path>`` lines for artifacts.
A clean run without a VERDICT line is classified ``done`` (lenient).

Exit codes: 0 done, 1 failed, 2 needs-input, 3 escalation, 4 timeout,
5 error (crash / bad usage / config failure).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

from sr2_spectre.agent import Agent
from sr2_spectre.cli import _configure_logging, resolve_agent_config_path, resolve_config
from sr2_spectre.interfaces.single_shot import SingleShotInterface

logger = logging.getLogger(__name__)

# Raw-stdout capture is a debugging aid (what the turn printed besides the
# final message).  Cap it so the envelope stays small on chatty runs.
_RAW_STDOUT_CAP = 100_000

_VERDICT_RE = re.compile(r"^\s*VERDICT:\s*([A-Za-z][A-Za-z-]*)\s*$", re.MULTILINE)
_WROTE_RE = re.compile(r"^\s*WROTE\s+(\S+)\s*$", re.MULTILINE)

# Contract verdict keyword -> envelope status.
_VERDICT_STATUS = {
    "DONE": "done",
    "FAILED": "failed",
    "NEEDS-INPUT": "needs-input",
    "ESCALATION": "escalation",
}

STATUS_EXIT_CODES = {
    "done": 0,
    "failed": 1,
    "needs-input": 2,
    "escalation": 3,
    "timeout": 4,
    "error": 5,
}


def final_message(output: str) -> str:
    """The last non-empty blank-line-separated block of a run's output.

    Same convention as harbinger's dispatch: everything before the final
    block is chatter; the final block is the answer.
    """
    blocks = [block.strip() for block in re.split(r"\n\s*\n", output)]
    blocks = [block for block in blocks if block]
    return blocks[-1] if blocks else ""


def parse_verdict(message: str) -> str | None:
    """Status from the final block's VERDICT line, or None if absent."""
    match = _VERDICT_RE.search(final_message(message))
    if match is None:
        return None
    return _VERDICT_STATUS.get(match.group(1).upper())


def extract_wrote(message: str) -> list[str]:
    """Artifact paths named by WROTE lines in the final block, in order."""
    return _WROTE_RE.findall(final_message(message))


def classify(message: str, *, error: bool = False, timed_out: bool = False) -> str:
    """Envelope status from the run outcome and the agent's verdict."""
    if timed_out:
        return "timeout"
    if error:
        return "error"
    return parse_verdict(message) or "done"


@dataclass
class DispatchEnvelope:
    """Machine-readable result of one dispatch run."""

    agent: str
    session_id: str
    status: str
    exit_code: int
    duration_s: float
    message: str
    wrote: list[str] = field(default_factory=list)
    wrote_missing: list[str] = field(default_factory=list)
    raw_stdout: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def _parse_dispatch_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sr2-spectre dispatch",
        description=(
            "Run one headless agent turn and emit a JSON envelope with the "
            "final message, parsed verdict, WROTE paths, and status."
        ),
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name — resolves to <agents_dir>/<name>.yaml "
             "(agents dir defaults to $SR2_HOME/agents/, override with --agents-dir).",
    )
    parser.add_argument(
        "--agents-dir",
        type=str,
        default=None,
        help="Directory containing agent YAML files "
             "(default: $SR2_HOME/agents/, or ~/.sr2/agents/)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text (mutually exclusive with --prompt-file; "
             "neither reads stdin).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Read the prompt from this file instead of --prompt.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session ID (default: dispatch-<agent>-<unix-seconds>).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Turn timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write the envelope JSON to this file instead of stdout.",
    )
    return parser.parse_args(argv)


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None and args.prompt_file is not None:
        raise SystemExit(
            "dispatch: --prompt and --prompt-file are mutually exclusive "
            "(or pass neither to read the prompt from stdin)"
        )
    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None:
        try:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"dispatch: cannot read prompt file: {exc}")
    else:
        prompt = sys.stdin.read()
    if not prompt.strip():
        raise SystemExit("dispatch: empty prompt")
    return prompt


async def _dispatch_async(
    config,
    agent_name: str,
    prompt: str,
    session_id: str,
    timeout: float,
) -> DispatchEnvelope:
    """Run one turn and build the envelope. Never raises (except on cancel)."""
    agent = Agent(config=config, session_id=session_id)
    # The single_shot interface owns the RunContext (HEADLESS mode); we drive
    # the turn ourselves so the final text is captured, not printed.
    interface = SingleShotInterface(prompt=prompt)

    started = time.monotonic()
    message = ""
    raw_stdout = ""
    error = False
    timed_out = False
    try:
        await agent.initialize()
        await interface.start(agent)
        try:
            async with asyncio.timeout(timeout):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    result = await agent.handle_user_message(prompt)
                message = result.text or ""
                raw_stdout = buffer.getvalue()
        except TimeoutError:
            logger.warning("dispatch: turn timed out after %.0fs", timeout)
            timed_out = True
    except Exception:
        logger.exception("dispatch: agent run failed")
        error = True
    finally:
        await interface.stop()
        await agent.aclose()

    status = classify(message, error=error, timed_out=timed_out)
    wrote = extract_wrote(message)
    return DispatchEnvelope(
        agent=agent_name,
        session_id=session_id,
        status=status,
        exit_code=STATUS_EXIT_CODES[status],
        duration_s=round(time.monotonic() - started, 1),
        message=message,
        wrote=wrote,
        wrote_missing=[p for p in wrote if not Path(p).exists()],
        raw_stdout=raw_stdout[:_RAW_STDOUT_CAP],
    )


def run_dispatch(argv: list[str]) -> int:
    """Synchronous entry point for 'sr2-spectre dispatch ...'. Returns the exit code."""
    args = _parse_dispatch_args(argv)
    prompt = _resolve_prompt(args)

    _configure_logging("INFO", str(
        Path.home() / ".sr2-spectre" / "spectre-dispatch.log"
    ))

    try:
        config_path = resolve_agent_config_path(
            args.agent,
            agents_dir=Path(args.agents_dir) if args.agents_dir else None,
        )
        config = resolve_config(config_path, cwd=Path.cwd(), env=dict(os.environ))
    except Exception as exc:
        print(f"dispatch: config error: {exc}", file=sys.stderr)
        return STATUS_EXIT_CODES["error"]

    agent_name = config.agent.name
    session_id = args.session_id or f"dispatch-{agent_name}-{int(time.time())}"

    try:
        envelope = asyncio.run(_dispatch_async(
            config, agent_name, prompt, session_id, args.timeout
        ))
    except BaseException as exc:
        # Cancel or crash outside the turn (e.g. asyncio.run teardown): report,
        # never leak a traceback onto the JSON stdout.
        logger.exception("dispatch: run aborted: %s", exc)
        envelope = DispatchEnvelope(
            agent=agent_name,
            session_id=session_id,
            status="error",
            exit_code=STATUS_EXIT_CODES["error"],
            duration_s=0.0,
            message="",
        )

    payload = envelope.to_json()
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return envelope.exit_code
