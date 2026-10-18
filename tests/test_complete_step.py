"""Tests for CompleteStepTool and post-execute event system (spc-14).

Covers:
- Successful verify + status flip + findings
- Missing task file
- Task not pending
- Missing verify command
- Failed verify
- ToolOutput with PostExecuteEvent on success
- Plain result (no events) on failure
- Generic event dispatch in Session._execute_tool (no name-magic)
- ToolOutput unwrapping by executor
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sr2.models import ToolResultBlock, ToolUseBlock

from tests.conftest import _surviving_pids

from sr2_spectre.tools.builtins.complete_step import (
    CompleteStepResult,
    CompleteStepTool,
)
from sr2_spectre.tools.output import PostExecuteEvent, ToolOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plans_root(tmp_path: Path) -> Path:
    """Create a temporary plans directory."""
    root = tmp_path / "plans"
    root.mkdir()
    plan_dir = root / "test-plan"
    plan_dir.mkdir()
    return root


@pytest.fixture
def task_file(plans_root: Path) -> Path:
    """Create a standard pending task file."""
    task = plans_root / "test-plan" / "01-setup.md"
    task.write_text(
        textwrap.dedent(
            """\
            ---
            kind: task
            plan: test-plan
            order: 1
            status: pending
            verify: echo "tests pass"
            title: Setup infrastructure
            ---

            Set up the project infrastructure.
            """
        )
    )
    return task


@pytest.fixture
def tool(plans_root: Path) -> CompleteStepTool:
    return CompleteStepTool(plans_root=str(plans_root))


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


def _extract_json(result: str | ToolOutput) -> dict:
    """Helper: unwrap ToolOutput if needed, then parse JSON."""
    if isinstance(result, ToolOutput):
        return json.loads(result.result)
    return json.loads(result)


class TestCompleteStepSuccess:
    async def test_successful_completion(self, tool: CompleteStepTool, task_file: Path) -> None:
        result = await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="none",
        )
        data = _extract_json(result)

        assert data["success"] is True
        assert data["plan"] == "test-plan"
        assert data["task"] == "01-setup"
        assert data["order"] == 1
        assert data["frame"] == "plan:test-plan/01-setup"

        # Verify status flip
        content = task_file.read_text()
        assert "status: done" in content
        assert "status: pending" not in content

    async def test_success_returns_tool_output_with_events(
        self, tool: CompleteStepTool, task_file: Path
    ) -> None:
        """Successful complete_step returns ToolOutput with PostExecuteEvent."""
        result = await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="none",
        )

        assert isinstance(result, ToolOutput)
        assert len(result.events) == 1

        event = result.events[0]
        assert event.event_name == "plan_step_completed"
        assert event.phase == "completed"
        assert event.source_layer == "plan"
        assert event.data["frame"] == "plan:test-plan/01-setup"
        assert event.data["plan"] == "test-plan"
        assert event.data["task"] == "01-setup"
        assert event.data["order"] == 1

    async def test_findings_appended(
        self, tool: CompleteStepTool, task_file: Path, plans_root: Path
    ) -> None:
        await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="The --plugin flag is also in tests/conftest.py",
        )

        findings_file = plans_root / "test-plan" / "_findings.md"
        assert findings_file.exists()
        content = findings_file.read_text()
        assert "01-setup" in content
        assert "The --plugin flag is also in tests/conftest.py" in content

    async def test_findings_none_is_noop(
        self, tool: CompleteStepTool, task_file: Path, plans_root: Path
    ) -> None:
        await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="none",
        )

        findings_file = plans_root / "test-plan" / "_findings.md"
        assert not findings_file.exists()

    async def test_findings_appended_to_existing(
        self, tool: CompleteStepTool, task_file: Path, plans_root: Path
    ) -> None:
        findings_file = plans_root / "test-plan" / "_findings.md"
        findings_file.write_text("### 00-previous\nPrevious finding\n")

        await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="New discovery",
        )

        content = findings_file.read_text()
        assert "00-previous" in content  # preserved
        assert "01-setup" in content  # appended

    async def test_flip_does_not_corrupt_body_mention(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """A 'status: pending' mention in the body must NOT be rewritten —
        only the frontmatter status flips."""
        task = plans_root / "test-plan" / "04-bodymention.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 4
                status: pending
                verify: echo ok
                title: Body mentions status
                ---

                Document the old behavior where `status: pending` stayed forever.
                """
            )
        )
        await tool(plan_slug="test-plan", task_slug="04-bodymention", findings="none")

        content = task.read_text()
        # Frontmatter flipped...
        assert "status: done" in content
        # ...but the body's literal mention is untouched.
        assert "`status: pending` stayed forever" in content


class TestCompleteStepFailures:
    async def test_missing_task_file(self, tool: CompleteStepTool) -> None:
        result = await tool(
            plan_slug="test-plan",
            task_slug="99-nonexistent",
            findings="none",
        )
        data = _extract_json(result)
        assert data["success"] is False
        assert "not found" in data["message"].lower()

    async def test_failure_returns_plain_string(self, tool: CompleteStepTool) -> None:
        """Failed complete_step returns a plain string (no ToolOutput wrapper)."""
        result = await tool(
            plan_slug="test-plan",
            task_slug="99-nonexistent",
            findings="none",
        )
        assert isinstance(result, str)
        assert not isinstance(result, ToolOutput)

    async def test_task_not_pending(self, tool: CompleteStepTool, task_file: Path) -> None:
        task_file.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 1
                status: done
                verify: echo ok
                title: Already done
                ---

                Done task.
                """
            )
        )
        result = await tool(
            plan_slug="test-plan",
            task_slug="01-setup",
            findings="none",
        )
        data = _extract_json(result)
        assert data["success"] is False
        assert "not pending" in data["message"].lower()

    async def test_missing_verify_command(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        task = plans_root / "test-plan" / "02-noverify.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 2
                status: pending
                title: No verify
                ---

                Task with no verify command.
                """
            )
        )
        result = await tool(
            plan_slug="test-plan",
            task_slug="02-noverify",
            findings="none",
        )
        data = _extract_json(result)
        assert data["success"] is False
        assert (
            "no 'verify:" in data["message"]
            or ("no" in data["message"].lower() and "verify" in data["message"].lower())
        )

    async def test_failed_verify(self, tool: CompleteStepTool, plans_root: Path) -> None:
        task = plans_root / "test-plan" / "03-broken.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 3
                status: pending
                verify: exit 1
                title: Broken task
                ---

                This task's verify fails.
                """
            )
        )
        result = await tool(
            plan_slug="test-plan",
            task_slug="03-broken",
            findings="none",
        )
        data = _extract_json(result)
        assert data["success"] is False
        assert "failed" in data["message"].lower()

        # Status should NOT be flipped
        content = task.read_text()
        assert "status: pending" in content


class TestCompleteStepToolSchema:
    def test_tool_has_required_schema_fields(self) -> None:
        tool = CompleteStepTool()
        assert tool.name == "complete_step"
        assert tool.description
        assert "findings" in tool.input_schema["required"]
        assert "plan_slug" in tool.input_schema["required"]
        assert "task_slug" in tool.input_schema["required"]
        assert tool.input_schema["type"] == "object"


# ---------------------------------------------------------------------------
# Status flip spacing variants (spc-13)
# ---------------------------------------------------------------------------


class TestStatusFlipSpacingVariants:
    """spc-13: _flip_status must handle spacing/case variants that the
    parser already tolerates (status:pending, Status: Pending, etc.)."""

    async def test_uppercase_status_value(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """status: PENDING (uppercase value) must flip to done."""
        task = plans_root / "test-plan" / "05-uppercase-val.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 5
                status: PENDING
                verify: echo ok
                title: Uppercase value
                ---

                Task with uppercase status value.
                """
            )
        )
        result = await tool(
            plan_slug="test-plan",
            task_slug="05-uppercase-val",
            findings="none",
        )
        data = _extract_json(result)
        assert data["success"] is True

        from sr2_spectre.planning import parse_file

        parsed = parse_file(task)
        assert parsed is not None
        assert parsed.status.value == "done"

    async def test_uppercase_status_key(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """Status: Pending (capital S) must flip correctly."""
        task = plans_root / "test-plan" / "06-uppercase.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 6
                Status: Pending
                verify: echo ok
                title: Uppercase
                ---

                Task with uppercase status key.
                """
            )
        )
        result = await tool(
            plan_slug="test-plan",
            task_slug="06-uppercase",
            findings="none",
        )
        data = _extract_json(result)
        assert data["success"] is True

        # Verify the status line was actually changed in the file.
        content = task.read_text()
        assert "Status: done" in content
        assert "Pending" not in content

    async def test_mixed_spacing_and_case(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """STATUS : Pending (uppercase key, extra space) must flip."""
        task = plans_root / "test-plan" / "07-mixed.md"
        task.write_text(
            "---\n"
            "kind: task\n"
            "plan: test-plan\n"
            "order: 7\n"
            "STATUS : Pending\n"
            "verify: echo ok\n"
            "title: Mixed\n"
            "---\n"
            "\n"
            "Task with mixed spacing and case.\n"
        )
        result = await tool(
            plan_slug="test-plan",
            task_slug="07-mixed",
            findings="none",
        )
        data = _extract_json(result)
        assert data["success"] is True

        # Verify the status line was actually changed.
        content = task.read_text()
        assert "STATUS: done" in content or "STATUS : done" in content
        assert "Pending" not in content and "pending" not in content

    async def test_flip_failure_returns_error(
        self, tool: CompleteStepTool, plans_root: Path
    ) -> None:
        """If the status line can't be matched, complete_step must FAIL
        instead of silently succeeding."""
        task = plans_root / "test-plan" / "08-quoted.md"
        task.write_text(
            textwrap.dedent(
                """\
                ---
                kind: task
                plan: test-plan
                order: 8
                status: "pending"
                verify: echo ok
                title: Quoted
                ---

                Task with quoted status value.
                """
            )
        )
        result = await tool(
            plan_slug="test-plan",
            task_slug="08-quoted",
            findings="none",
        )
        data = _extract_json(result)
        assert data["success"] is True

        from sr2_spectre.planning import parse_file

        parsed = parse_file(task)
        assert parsed is not None
        assert parsed.status.value == "done"


# ---------------------------------------------------------------------------
# Post-execute event system tests (spc-14)
# ---------------------------------------------------------------------------


class TestToolOutput:
    """Tests for the ToolOutput wrapper and PostExecuteEvent."""

    def test_tool_output_default_events(self) -> None:
        """ToolOutput with no events is valid."""
        output = ToolOutput(result="hello")
        assert output.result == "hello"
        assert output.events == []

    def test_tool_output_with_events(self) -> None:
        output = ToolOutput(
            result="done",
            events=[
                PostExecuteEvent(
                    event_name="test_event",
                    data={"key": "value"},
                ),
            ],
        )
        assert len(output.events) == 1
        assert output.events[0].event_name == "test_event"

    def test_tool_output_truthiness(self) -> None:
        """ToolOutput should be truthy even with empty result."""
        assert bool(ToolOutput(result="")) is True
        assert bool(ToolOutput(result=None)) is True

    def test_post_execute_event_defaults(self) -> None:
        event = PostExecuteEvent(event_name="my_event")
        assert event.phase == "completed"
        assert event.source_layer == "plan"
        assert event.data == {}


class TestSessionToolOutputDispatch:
    """Session._execute_tool dispatches PostExecuteEvents generically."""

    async def test_executor_dispatches_tool_output_events(self) -> None:
        """When a tool returns ToolOutput with events, they are dispatched on the bus."""
        from sr2_spectre.session import Session

        # Build a minimal session with mocks
        mock_config = MagicMock()
        mock_config.agent.tool_result_max_bytes = 1024

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(
            return_value=ToolOutput(
                result="tool_result",
                events=[
                    PostExecuteEvent(
                        event_name="custom_event",
                        data={"foo": "bar"},
                        source_layer="test",
                    ),
                ],
            )
        )

        # Create a mock SR2 with a mock bus
        mock_sr2 = MagicMock()
        mock_sr2.bus.queue = MagicMock()

        session = Session(
            frame_id="test-frame",
            config=mock_config,
            llm=AsyncMock(),
            registry=mock_registry,
        )
        # Replace SR2 instance with our mock
        session.sr2 = mock_sr2

        block = ToolUseBlock(
            id="tool-1",
            name="my_tool",
            input={},
        )
        result = await session._execute_tool(block)

        # The ToolOutput should be unwrapped — content is the inner result
        assert result.content == "tool_result"

        # Event should be queued on the bus
        assert mock_sr2.bus.queue.call_count == 1
        queued_event = mock_sr2.bus.queue.call_args[0][0]
        assert queued_event.name == "custom_event"
        assert queued_event.data == {"foo": "bar"}
        assert queued_event.source_layer == "test"

    async def test_executor_plain_result_no_dispatch(self) -> None:
        """Plain string results (no ToolOutput) produce no bus events."""
        from sr2_spectre.session import Session

        mock_config = MagicMock()
        mock_config.agent.tool_result_max_bytes = 1024

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value="plain_result")

        mock_sr2 = MagicMock()
        mock_sr2.bus.queue = MagicMock()

        session = Session(
            frame_id="test-frame",
            config=mock_config,
            llm=AsyncMock(),
            registry=mock_registry,
        )
        session.sr2 = mock_sr2

        block = ToolUseBlock(
            id="tool-1",
            name="some_tool",
            input={},
        )
        result = await session._execute_tool(block)

        assert result.content == "plain_result"
        assert mock_sr2.bus.queue.call_count == 0

    async def test_executor_no_name_magic(self) -> None:
        """The executor does not special-case any tool name.

        This verifies the OCP fix: events come from the tool, not from
        the executor sniffing block.name.
        """
        from sr2_spectre.session import Session
        import inspect

        source = inspect.getsource(Session._execute_tool)
        assert 'block.name == "complete_step"' not in source
        assert "complete_step" not in source

    async def test_executor_multiple_events(self) -> None:
        """A tool can declare multiple post-execute events."""
        from sr2_spectre.session import Session

        mock_config = MagicMock()
        mock_config.agent.tool_result_max_bytes = 1024

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(
            return_value=ToolOutput(
                result="multi",
                events=[
                    PostExecuteEvent(event_name="event_a", data={"n": 1}),
                    PostExecuteEvent(event_name="event_b", data={"n": 2}),
                ],
            )
        )

        mock_sr2 = MagicMock()
        mock_sr2.bus.queue = MagicMock()

        session = Session(
            frame_id="test-frame",
            config=mock_config,
            llm=AsyncMock(),
            registry=mock_registry,
        )
        session.sr2 = mock_sr2

        block = ToolUseBlock(
            id="tool-1",
            name="multi_tool",
            input={},
        )
        await session._execute_tool(block)

        assert mock_sr2.bus.queue.call_count == 2
        calls = [c[0][0] for c in mock_sr2.bus.queue.call_args_list]
        assert calls[0].name == "event_a"
        assert calls[1].name == "event_b"


# ---------------------------------------------------------------------------
# Backward compat: _is_complete_step_success removed from agent.py
# ---------------------------------------------------------------------------


class TestAgentModuleClean:
    """Verify agent.py no longer has complete_step name-magic."""

    def test_agent_no_complete_step_sniffing(self) -> None:
        """agent.py should not contain complete_step name-magic."""
        import inspect
        from sr2_spectre.agent import Agent

        source = inspect.getsource(Agent)
        assert "complete_step" not in source
        assert "_is_complete_step_success" not in source

    def test_agent_no_json_import(self) -> None:
        """agent.py no longer imports json for result parsing."""
        from sr2_spectre.agent import __dict__ as agent_dict
        # The module shouldn't have _is_complete_step_success
        assert "_is_complete_step_success" not in dir(__import__("sr2_spectre.agent", fromlist=[""]))


# ---------------------------------------------------------------------------
# Process-group cleanup (spc-947 defect class, spc-947 G finding 2):
# verify timeout and cancel must not orphan the command's children
# ---------------------------------------------------------------------------


class TestVerifyProcessGroupCleanup:
    async def test_timeout_kills_child_processes(
        self, plans_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On the verify timeout, the whole process group must die.

        The pre-fix code returned the failure tuple without killing, so
        `sh -c 'sleep <marker>'` (and any grandchild) outlived the tool.
        """
        tool = CompleteStepTool(plans_root=str(plans_root))
        monkeypatch.setattr(CompleteStepTool, "_VERIFY_TIMEOUT", 0.5)

        ok, output = await tool._run_verify("sh -c 'exec sleep 9943'")
        assert ok is False
        assert "timed out" in output

        await asyncio.sleep(0.3)  # give the kernel a moment to reap
        assert _surviving_pids("9943") == []

    async def test_cancel_kills_child_processes(
        self, plans_root: Path
    ) -> None:
        """An outside cancel must kill the group, not just abandon it.

        The pre-fix broad `except Exception` did not catch CancelledError,
        so a cancelled turn left the verify command running.
        """
        tool = CompleteStepTool(plans_root=str(plans_root))
        task = asyncio.create_task(tool._run_verify("sh -c 'exec sleep 9944'"))
        await asyncio.sleep(0.5)  # let the shell and child start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.3)
        assert _surviving_pids("9944") == []
