"""Builtin tool for running an isolated agent frame in the calling process."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from spctr.agent import Agent
from spctr.cli import resolve_agent_config_path, resolve_config
from spctr.dispatch import STATUS_EXIT_CODES, classify, extract_wrote
from spctr.events import (
    AgentDone,
    AgentEvent,
    AgentHandoffEnd,
    AgentHandoffStart,
    AgentTextDelta,
)
from spctr.interfaces.single_shot import SingleShotInterface
from spctr.session import HandoffHandle


_caller_session: ContextVar[Any | None] = ContextVar("handoff_caller", default=None)


@contextmanager
def bind_caller(session: Any) -> Iterator[None]:
    """Bind the Session executing this tool for the duration of one call."""
    token = _caller_session.set(session)
    try:
        yield
    finally:
        _caller_session.reset(token)


class HandoffTool:
    name = "handoff"

    def __init__(
        self,
        agents_root: str,
        default_timeout: int = 5400,
        description: str = (
            "Hand the session to another agent in a fresh frame and return "
            "its envelope when the frame ends."
        ),
        agent_description: str = "Agent config name under <agents_root>/agents/",
        prompt_description: str = "Instructions for the target agent.",
        timeout_description: str = "Seconds; default from tool config.",
        frame_id_description: str = (
            "Optional frame id; default handoff-<agent>-<unix seconds>."
        ),
        nested_error: str = "nested handoff refused",
        active_error: str = "handoff already active",
        unavailable_error: str = "handoff requires a caller session",
        config_error: str = "cannot load {path}: {error}",
        run_error: str = "{error}",
    ) -> None:
        root = Path(agents_root).expanduser()
        if not root.is_absolute():
            raise ValueError("agents_root must be an absolute path")
        self.agents_root = root
        self.default_timeout = default_timeout
        self.description = description
        self._nested_error = nested_error
        self._active_error = active_error
        self._unavailable_error = unavailable_error
        self._config_error = config_error
        self._run_error = run_error
        self.input_schema = {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": agent_description},
                "prompt": {"type": "string", "description": prompt_description},
                "timeout": {"type": "number", "description": timeout_description},
                "frame_id": {"type": "string", "description": frame_id_description},
            },
            "required": ["agent", "prompt"],
        }

    def _envelope(
        self,
        agent: str,
        frame_id: str,
        status: str,
        duration_s: float,
        message: str,
    ) -> str:
        wrote = extract_wrote(message)
        return json.dumps(
            {
                "agent": agent,
                "frame_id": frame_id,
                "status": status,
                "exit_code": STATUS_EXIT_CODES[status],
                "duration_s": duration_s,
                "message": message,
                "wrote": wrote,
                "wrote_missing": [path for path in wrote if not Path(path).exists()],
            }
        )

    async def __call__(
        self,
        agent: str,
        prompt: str,
        timeout: float | None = None,
        frame_id: str | None = None,
    ) -> str:
        caller = _caller_session.get()
        frame_id = frame_id or f"handoff-{agent}-{int(time.time())}"
        if caller is None:
            return self._envelope(agent, frame_id, "error", 0.0, self._unavailable_error)
        if caller.is_handoff_target:  # (dec: dec-359)
            return self._envelope(agent, frame_id, "error", 0.0, self._nested_error)
        if caller.active_handoff is not None:  # (dec: dec-359)
            return self._envelope(agent, frame_id, "error", 0.0, self._active_error)

        agents_dir = self.agents_root / "agents"
        target_path = agents_dir / f"{agent}.yaml"
        try:
            config_path = resolve_agent_config_path(agent, agents_dir=agents_dir)
            config = resolve_config(
                config_path,
                cwd=Path.cwd(),
                env={**os.environ, "SR2_HOME": str(self.agents_root)},
            )
            target = Agent(config=config, session_id=frame_id)
        except Exception as exc:
            message = self._config_error.format(path=target_path, error=exc)
            return self._envelope(agent, frame_id, "error", 0.0, message)

        target._session.is_handoff_target = True
        interface = SingleShotInterface(prompt=prompt)
        started = time.monotonic()
        message_parts: list[str] = []

        async def _run_target() -> None:
            await target.initialize()
            await interface.start(target)
            limit = self.default_timeout if timeout is None else timeout
            async with asyncio.timeout(limit):
                async for event in target.stream_message(prompt):
                    if isinstance(event, AgentTextDelta):
                        message_parts.append(event.text)
                    if not isinstance(event, AgentDone):
                        caller.emit_handoff_event(replace(event, frame_id=frame_id))

        target_task = asyncio.create_task(_run_target())
        handle = HandoffHandle(frame_id, agent, started, target_task.cancel)
        caller.active_handoff = handle
        caller._log_event("handoff.start", {"frame_id": frame_id, "agent": agent})
        caller.emit_handoff_event(AgentHandoffStart(frame_id=frame_id, agent=agent))
        status = "done"
        try:
            try:
                await target_task
            except asyncio.CancelledError:
                if not handle.cancel_requested:
                    raise
                status = "cancelled"
            except TimeoutError:
                status = "timeout"
            else:
                status = classify("".join(message_parts))
        except Exception as exc:
            status = "error"
            message_parts.append(self._run_error.format(error=exc))
        finally:
            try:
                await interface.stop()
                await target.aclose()
            except Exception as exc:
                status = "error"
                message_parts.append(self._run_error.format(error=exc))
            finally:
                caller.active_handoff = None

        duration_s = max(0.001, round(time.monotonic() - started, 3))
        message = "".join(message_parts)
        caller.emit_handoff_event(
            AgentHandoffEnd(
                frame_id=frame_id,
                agent=agent,
                status=status,
                duration_s=duration_s,
            )
        )
        caller._log_event(
            "handoff.end",
            {
                "frame_id": frame_id,
                "status": status,
                "duration_ms": round(duration_s * 1000, 3),
            },
        )
        return self._envelope(agent, frame_id, status, duration_s, message)
