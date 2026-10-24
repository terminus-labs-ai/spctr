"""Session-identity plumbing for SPECTRE_SESSION_ID (spc-137).

Covers the seam contract:
  1. The ContextVar module is a leaf (imports nothing from sr2_spectre).
  2. Session.stream_message sets current_session_id for the whole turn —
     the log stem when session logging is on, the frame_id otherwise —
     and restores the previous value on exit.
  3. Concurrent turns in separate tasks each observe their own value.
  4. The terminal tool, called through the real Session tool-executor
     seam during a turn, exports the same value to its subprocess.
"""

from __future__ import annotations

import asyncio
import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sr2.models import ToolResultBlock, ToolUseBlock
from sr2.protocols.llm import StreamEvent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.run_log import SessionLogManager
from sr2_spectre.session import Session
from sr2_spectre.session_identity import current_session_id
from sr2_spectre.tools.builtins.terminal import TerminalTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test"),
        models={
            "default": ModelConfig(model="test-model", base_url="http://test:8000")
        },
        pipeline={
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [
                        {"type": "static", "config": {"text": "You are helpful."}}
                    ],
                },
                {
                    "name": "conversation",
                    "target": "messages",
                    "resolvers": [{"type": "session"}, {"type": "input"}],
                },
            ]
        },
    )


def _make_sr2_mock(on_call) -> MagicMock:
    """Mock SR2 whose turn() invokes on_call() at call time, then ends.

    on_call may be sync or async; both are awaited/handled at turn() call
    time, inside the same task the turn flows through.
    """

    sr2 = MagicMock()
    sr2.seed_session = MagicMock()

    async def _turn(user_input):
        out = on_call()
        if asyncio.iscoroutine(out):
            await out
        yield StreamEvent(type="text", text="ok")
        yield StreamEvent(type="end")

    sr2.turn = _turn
    return sr2


def _make_session(
    frame_id: str,
    sr2: MagicMock,
    log_manager: SessionLogManager | None = None,
) -> Session:
    with patch("sr2_spectre.session.SR2", return_value=sr2):
        session = Session(
            frame_id=frame_id,
            config=_make_config(),
            llm=MagicMock(),
            registry=MagicMock(),
            log_manager=log_manager,
        )
    return session


def _open_log_ctx(session: Session) -> None:
    """Open the session log through the shared run-context seam."""
    session.set_run_context(
        RunContext(interface="test", mode=RunMode.INTERACTIVE, source="/tmp")
    )


# ---------------------------------------------------------------------------
# 1. Leaf-module contract
# ---------------------------------------------------------------------------

def test_session_identity_module_is_a_leaf() -> None:
    """session_identity must import nothing from sr2_spectre (no cycles)."""
    import sr2_spectre.session_identity as si

    tree = ast.parse(Path(inspect.getsourcefile(si)).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [
        name for name in imported if name == "sr2_spectre" or name.startswith(
            "sr2_spectre."
        )
    ]
    assert offenders == []


def test_current_session_id_defaults_to_none() -> None:
    assert current_session_id.get() is None


# ---------------------------------------------------------------------------
# 2. Session.stream_message sets the value per turn (log stem / frame_id)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_turn_with_log_sets_contextvar_to_log_stem(tmp_path, monkeypatch):
    """Logging on: during the turn, current_session_id == the log file stem."""
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(tmp_path / "logs"))
    manager = SessionLogManager()
    session = _make_session("frame-log", _make_sr2_mock(lambda: None), manager)
    _open_log_ctx(session)
    assert session._log is not None
    stem = session._log.path.stem
    assert stem.endswith("frame-log") or "frame-log" in stem

    observed: list[str | None] = []

    async def _record() -> None:
        observed.append(current_session_id.get())

    session.sr2 = _make_sr2_mock(_record)
    async for _ in session.stream_message("hi"):
        pass

    assert observed == [stem]
    # Restored after the turn.
    assert current_session_id.get() is None


@pytest.mark.asyncio
async def test_turn_without_log_sets_contextvar_to_frame_id(tmp_path, monkeypatch):
    """Logging off (no log opened): the value is the session's frame_id."""
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", "off")
    manager = SessionLogManager()
    session = _make_session("frame-off", _make_sr2_mock(lambda: None), manager)
    _open_log_ctx(session)
    assert session._log is None  # opening failed because logging is off

    observed: list[str | None] = []

    async def _record() -> None:
        observed.append(current_session_id.get())

    session.sr2 = _make_sr2_mock(_record)
    async for _ in session.stream_message("hi"):
        pass

    assert observed == ["frame-off"]
    assert current_session_id.get() is None


@pytest.mark.asyncio
async def test_turn_restores_previous_value_after_streaming():
    """A caller's pre-existing value is back once the turn generator ends."""
    session = _make_session("frame-restore", _make_sr2_mock(lambda: None))
    token = current_session_id.set("outer-value")
    try:
        async for _ in session.stream_message("hi"):
            pass
        assert current_session_id.get() == "outer-value"
    finally:
        current_session_id.reset(token)
    assert current_session_id.get() is None


# ---------------------------------------------------------------------------
# 3. Concurrent sessions in separate tasks keep their own values
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_turns_observe_their_own_frame_id():
    """Two Sessions, two tasks: each turn sees only its own frame_id."""

    async def _run_one(frame_id: str) -> list[str | None]:
        session = _make_session(frame_id, _make_sr2_mock(lambda: None))
        # The tool call inside the turn waits, so both turns overlap in
        # their tool phase; each must still see its own context value.
        observed: list[str | None] = []

        async def _record_and_wait() -> None:
            observed.append(current_session_id.get())
            await asyncio.sleep(0.05)

        session.sr2 = _make_sr2_mock(_record_and_wait)
        async for _ in session.stream_message("hi"):
            pass
        return observed

    results = await asyncio.gather(
        _run_one("frame-A"),
        _run_one("frame-B"),
    )
    assert results == [["frame-A"], ["frame-B"]]


# ---------------------------------------------------------------------------
# 4. Through the real Session tool-executor seam, the terminal tool sees it
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_during_logged_turn_sees_log_stem(tmp_path, monkeypatch):
    """A terminal command run during a logged turn exports the log stem."""
    from sr2_spectre.tools.registry import ToolRegistry

    monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(tmp_path / "logs"))
    manager = SessionLogManager()
    session = _make_session("frame-term", _make_sr2_mock(lambda: None), manager)
    # Swap in a real registry holding the real terminal tool so the call
    # flows through Session._run_tool exactly as in production.
    registry = ToolRegistry()
    registry.register_from_class_path(
        "sr2_spectre.tools.builtins.terminal.TerminalTool"
    )
    session._registry = registry
    _open_log_ctx(session)
    assert session._log is not None
    stem = session._log.path.stem

    use = ToolUseBlock(
        id="tu1", name="terminal", input={"command": 'echo "$SPECTRE_SESSION_ID"'}
    )
    results: list[ToolResultBlock] = []

    async def _run_tool(user_input):
        # The SR2 pipeline awaits the tool executor inline in the turn flow
        # (sr2 orchestrator); emulate that here through the real callback.
        result = await session._execute_tool(use)
        results.append(result)
        yield StreamEvent(type="tool_use_emitted", tool_uses=[use])
        yield StreamEvent(type="tool_result_received", tool_results=[result])
        yield StreamEvent(type="end")

    session.sr2 = MagicMock()
    session.sr2.seed_session = MagicMock()
    session.sr2.turn = _run_tool

    async for _ in session.stream_message("hi"):
        pass

    assert len(results) == 1
    assert str(results[0].content).strip() == stem
    assert current_session_id.get() is None
