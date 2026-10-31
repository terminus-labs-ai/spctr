"""Complete Step tool — verify-gated task completion for planning workflows.

Runs the task's ``verify:`` command, flips status pending→done ONLY on green,
and emits a ``plan_step_completed`` event signal (returned as structured data
so the Agent's tool executor can broadcast it on the SR2 event bus).

Requires a ``findings`` argument (free text; ``"none"`` is valid) as the cheap
externalize-or-not gate before context is burned.

Registered as a builtin tool; takes ``plans_root`` via constructor config so
it can locate the plan/task files on disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sr2_spectre.config import resolve_sr2_home
from sr2_spectre.planning import TaskFrontmatter, TaskStatus, parse_file, split_frontmatter
from sr2_spectre.tools.output import PostExecuteEvent, ToolOutput

logger = logging.getLogger(__name__)


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the subprocess's whole process group (spawns run in their own
    session, so its PGID is the child's PID). Killing only the direct child
    would orphan its children, which can outlive a timed-out or cancelled
    verify (spc-947 defect class). Falls back to killing just the child if
    the group is already gone."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Best-effort reap so the transport is cleaned up. Bounded: a killed
    shell can leave an orphaned child holding the pipes open, which makes an
    unbounded wait() hang; the exit status is not needed here.
    CancelledError must never be swallowed: it re-raises so an outside
    cancellation keeps propagating."""
    try:
        await asyncio.wait_for(proc.wait(), timeout=0.5)
    except asyncio.CancelledError:
        raise
    except Exception:
        pass


def frame_id(plan_slug: str, task_slug: str) -> str:
    """Canonical frame id for a plan task (see the ``frame`` primitive).

    Single source of truth for the ``plan:<plan>/<task>`` frame identifier so
    the transformer that burns a completed step's context matches exactly what
    this tool stamped.
    """
    return f"plan:{plan_slug}/{task_slug}"


@dataclass
class CompleteStepResult:
    """Structured return from a complete_step call."""

    success: bool
    plan: str
    task: str
    order: int
    frame: str
    message: str


class CompleteStepTool:
    """Verify-gated task completion tool.

    Schema:
        plan_slug (str): Plan directory name (e.g. "step-compaction").
        task_slug (str): Task file slug without extension (e.g. "02-cli-flag").
        findings (str): Free-text cross-step discoveries or "none".

    The tool reads the task file from ``<plans_root>/<plan_slug>/<task_slug>.md``,
    runs its ``verify:`` command, flips status to ``done`` on success, and
    appends findings to ``<plans_root>/<plan_slug>/_findings.md``.

    Returns a JSON-serializable dict with ``success``, ``plan``, ``task``,
    ``order``, and ``frame`` fields when verification passes — enabling the
    agent to emit a ``plan_step_completed`` event.
    """

    name = "complete_step"
    description = (
        "Complete a planning task: runs the task's verify command, flips "
        "status pending→done on success, and externalizes findings. "
        "Requires a findings argument (free text; 'none' is valid). "
        "Only use AFTER all work for the task is finished and verified."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "plan_slug": {
                "type": "string",
                "description": "Plan directory name (e.g. 'step-compaction').",
            },
            "task_slug": {
                "type": "string",
                "description": "Task file slug (e.g. '02-cli-flag').",
            },
            "findings": {
                "type": "string",
                "description": (
                    "Cross-step discoveries or 'none'. Forces a conscious "
                    "externalize-or-not decision before the step's context burns."
                ),
            },
        },
        "required": ["plan_slug", "task_slug", "findings"],
    }

    # Verify command wall-clock budget (seconds).
    _VERIFY_TIMEOUT = 120

    def __init__(self, plans_root: str | None = None) -> None:
        raw = plans_root or str(resolve_sr2_home() / "plans")
        self._plans_root = Path(raw).expanduser().resolve()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def __call__(
        self,
        plan_slug: str,
        task_slug: str,
        findings: str,
    ) -> str | ToolOutput:
        """Execute the complete-step flow.

        On success, returns a ``ToolOutput`` with a ``plan_step_completed``
        post-execute event attached. On failure, returns a plain JSON string
        (no events to emit).
        """
        result = await self._complete(plan_slug, task_slug, findings)
        serialized = self._serialize(result)
        if result.success:
            return ToolOutput(
                result=serialized,
                events=[
                    PostExecuteEvent(
                        event_name="plan_step_completed",
                        phase="completed",
                        source_layer="plan",
                        data={
                            "frame": result.frame,
                            "plan": result.plan,
                            "task": result.task,
                            "order": result.order,
                        },
                    ),
                ],
            )
        return serialized

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _complete(
        self,
        plan_slug: str,
        task_slug: str,
        findings: str,
    ) -> CompleteStepResult:
        frame = frame_id(plan_slug, task_slug)
        task_file = self._plans_root / plan_slug / f"{task_slug}.md"

        def fail(message: str, order: int = 0) -> CompleteStepResult:
            return CompleteStepResult(
                success=False,
                plan=plan_slug,
                task=task_slug,
                order=order,
                frame=frame,
                message=message,
            )

        # 1. Read + parse the task file (reusing the planning parser).
        task = self._read_task(task_file)
        if task is None:
            return fail(f"Task file not found or unparseable: {task_file}")

        # 2. Status must be pending.
        if task.status is not TaskStatus.PENDING:
            return fail(
                f"Task {task_slug} is not pending (status: {task.status.value})",
                order=task.order,
            )

        # 3. Verify command must exist.
        verify_cmd = task.verify.strip()
        if not verify_cmd:
            return fail(
                f"Task {task_slug} has no 'verify:' command in frontmatter",
                order=task.order,
            )

        # 4. Run verify — only a green run may flip status.
        verify_ok, verify_output = await self._run_verify(verify_cmd)
        if not verify_ok:
            return fail(
                f"Verification FAILED for {task_slug}:\n{verify_output}",
                order=task.order,
            )

        # 5. Flip status pending→done.
        self._flip_status(task_file)

        # 6. Externalize findings (unless explicitly "none").
        if findings.strip().lower() != "none":
            findings_file = self._plans_root / plan_slug / "_findings.md"
            self._append_findings(findings_file, task_slug, findings.strip())

        return CompleteStepResult(
            success=True,
            plan=plan_slug,
            task=task_slug,
            order=task.order,
            frame=frame,
            message=f"Task {task_slug} verified and marked complete.",
        )

    @staticmethod
    def _read_task(task_file: Path) -> TaskFrontmatter | None:
        """Parse the task file via the shared planning parser.

        Returns the ``TaskFrontmatter`` for a valid ``kind: task`` file, or
        None if the file is missing, unparseable, or not a task.
        """
        fm = parse_file(task_file)
        if not isinstance(fm, TaskFrontmatter):
            logger.warning("No valid task frontmatter in %s", task_file)
            return None
        return fm

    async def _run_verify(self, command: str) -> tuple[bool, str]:
        """Run the verify command in a subprocess. Returns (success, output)."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Own session/process group: on timeout or cancellation the
                # whole group is killed, so verify children (e.g. a pytest
                # grandchild) cannot outlive the tool call.
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface any spawn failure as a failed verify
            return False, f"Verify command error: {exc}"

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._VERIFY_TIMEOUT
            )
        except asyncio.CancelledError:
            # A cancel from outside arrives here; the broad `except Exception`
            # below does not catch CancelledError, and the pre-fix code killed
            # nothing on this path. Kill the group, then let the cancellation
            # propagate unchanged.
            _kill_process_group(proc)
            await _reap(proc)
            raise
        except (asyncio.TimeoutError, TimeoutError):
            _kill_process_group(proc)
            await _reap(proc)
            return False, f"Verify command timed out: {command}"
        except Exception as exc:  # noqa: BLE001 — surface any spawn failure as a failed verify
            return False, f"Verify command error: {exc}"
        output = (
            stdout.decode(errors="replace") + stderr.decode(errors="replace")
        ).strip()
        return proc.returncode == 0, output

    @staticmethod
    def _flip_status(task_file: Path) -> None:
        """Flip status to done in the frontmatter using YAML-aware replacement.

        Parses the frontmatter YAML, mutates the ``status`` field, and
        re-serializes the YAML block. This is tolerant of spacing, case,
        and quoting variants that the frontmatter parser already accepts.

        Bounded to the frontmatter block so body content is never corrupted.

        Raises ``ValueError`` if the status line cannot be located (e.g. the
        file has no ``status`` key at all — a programming error in the call
        site since the caller should have verified the field exists).
        """
        text = task_file.read_text(encoding="utf-8")
        result = split_frontmatter(text)
        if result is None:
            logger.warning("No valid frontmatter in %s — status not flipped", task_file.name)
            return

        fm_block, body = result

        # Extract raw YAML between delimiters.
        inner = fm_block[3:]  # strip opening "---\n"
        # Remove trailing "\n---" (the closing delimiter) and any trailing newline.
        # fm_block looks like "---\n<yaml>\n---\n"
        closing_idx = inner.rfind("\n---")
        if closing_idx == -1:
            logger.warning("Malformed frontmatter in %s — status not flipped", task_file.name)
            return
        raw_yaml = inner[:closing_idx].strip()
        trailing = inner[closing_idx:]  # "\n---\n" (preserved exactly)

        # Parse the YAML block into a mapping.
        try:
            fm_data: dict[str, Any] = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            logger.warning("Cannot parse frontmatter YAML in %s — status not flipped", task_file.name)
            return

        if not isinstance(fm_data, dict):
            logger.warning("Frontmatter in %s is not a mapping — status not flipped", task_file.name)
            return

        # Find the actual key name used (YAML keys are case-sensitive).
        # We need to find the line in the raw YAML that sets the status,
        # so we can replace it while preserving the original key casing.
        status_key = None
        for key in fm_data:
            if key.strip().lower() == "status":
                status_key = key
                break

        if status_key is None:
            raise ValueError(
                f"No 'status' key found in frontmatter of {task_file.name} — "
                "this should not happen (caller verified status is pending)."
            )

        # Replace the status line in the raw YAML text.
        # Use a regex that matches: <key>: <value> (tolerant of spacing/quoting).
        pattern = re.compile(
            r"^(" + re.escape(status_key) + r")\s*:\s*"  # key + colon
            r'(?:"([^"]*)"|\'([^\']*)\'|(\S+))',          # quoted or unquoted value
            re.MULTILINE,
        )
        new_raw = pattern.sub(
            rf"\1: {TaskStatus.DONE.value}", raw_yaml
        )
        if new_raw == raw_yaml:
            raise ValueError(
                f"Failed to replace status in {task_file.name}: "
                f"pattern did not match the status line for key {status_key!r}."
            )

        # Reconstruct the frontmatter block.
        new_fm_block = f"---\n{new_raw}\n{trailing}"
        task_file.write_text(new_fm_block + body, encoding="utf-8")
        logger.info("Flipped %s status to done", task_file.name)

    @staticmethod
    def _append_findings(findings_file: Path, task_slug: str, findings: str) -> None:
        """Append findings to _findings.md under a task heading."""
        existing = ""
        if findings_file.exists():
            try:
                existing = findings_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                existing = ""

        entry = f"\n### {task_slug}\n{findings}\n"
        content = existing.rstrip() + entry
        findings_file.write_text(content, encoding="utf-8")
        logger.info("Appended findings for %s to %s", task_slug, findings_file)

    @staticmethod
    def _serialize(result: CompleteStepResult) -> str:
        """Serialize result to a JSON string for tool output."""
        payload: dict[str, Any] = {
            "success": result.success,
            "plan": result.plan,
            "task": result.task,
            "order": result.order,
            "frame": result.frame,
            "message": result.message,
        }
        return json.dumps(payload)
