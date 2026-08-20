"""Tests for the --trace CLI flag (sr2-8 / FR12).

Acceptance criteria:
1. --trace defaults to False when omitted
2. --trace is True when flag is present
3. When --trace set, a CollectingTracer is instantiated and passed into Agent's SR2
4. Trace output appears in stdout AFTER the reply when --trace is set
5. No trace output when --trace is NOT set
6. Reply still appears when --trace is set
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from sr2_spectre.cli import _parse_args, run_async
from sr2_spectre.core import TurnResult


# ---------------------------------------------------------------------------
# 1 & 2: Argument parser — trace flag presence and default
# ---------------------------------------------------------------------------

def _fake_interface_cls(instance):
    """Class whose instantiation returns ``instance`` (for _load_interface patches)."""
    return type("_FakeInterfaceCls", (), {"__new__": staticmethod(lambda cls: instance)})


def test_parse_args_trace_defaults_to_false() -> None:
    args = _parse_args(["config.yaml", "hello"])
    assert args.trace is False


def test_parse_args_trace_is_true_when_flag_given() -> None:
    args = _parse_args(["config.yaml", "hello", "--trace"])
    assert args.trace is True


# ---------------------------------------------------------------------------
# Helpers for run_async integration tests
# ---------------------------------------------------------------------------

def _make_mock_config() -> MagicMock:
    """Minimal SpectreConfig stand-in."""
    config = MagicMock()
    config.agent.name = "test-agent"
    config.agent.tools = []
    config.models = {"default": MagicMock(model="test-model", base_url=None)}
    return config


def _make_mock_interface(reply: str = "Paris") -> MagicMock:
    """Interface whose run() calls agent.handle_user_message and prints result."""
    iface = MagicMock()
    iface.start = AsyncMock()
    iface.stop = AsyncMock()

    async def _run(agent: MagicMock) -> None:
        result = await agent.handle_user_message("What is the capital of France?")
        print(result.text)

    iface.run = _run
    return iface


# ---------------------------------------------------------------------------
# 3: CollectingTracer is instantiated and wired into SR2 when --trace is set
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_flag_wires_collecting_tracer_into_agent(
    capsys: pytest.CaptureFixture,
) -> None:
    """When --trace is set, Agent.__init__ must receive a CollectingTracer as tracer."""
    mock_config = _make_mock_config()
    captured_tracer: list = []

    with (
        patch("sr2_spectre.cli.resolve_config", return_value=mock_config),
        patch("sr2_spectre.cli._configure_logging"),
        patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(_make_mock_interface()),),
        patch("sr2_spectre.cli.Agent") as MockAgent,
    ):
        # Agent() returns an async-capable mock
        mock_agent_instance = AsyncMock()
        mock_agent_instance.handle_user_message.return_value = TurnResult(
            text="Paris", tool_calls_executed=0, total_tokens=10
        )
        MockAgent.return_value = mock_agent_instance

        # Capture the tracer passed to Agent.__init__.
        # This test verifies only that the CLI passes a CollectingTracer to Agent.
        # SR2-level wiring is Agent's responsibility and is tested at the Agent level.
        def capture_init(*args, **kwargs):
            captured_tracer.append(kwargs.get("tracer"))
            return mock_agent_instance

        MockAgent.side_effect = capture_init

        await run_async(["config.yaml", "What is the capital of France?", "--trace"])

    # Agent must have been constructed with a tracer keyword argument
    assert len(captured_tracer) == 1, "Agent was not instantiated"
    tracer = captured_tracer[0]

    from sr2.pipeline.tracing import CollectingTracer
    assert isinstance(tracer, CollectingTracer), (
        f"Expected CollectingTracer, got {type(tracer)}"
    )


# ---------------------------------------------------------------------------
# 4 & 6: With --trace, reply appears first, then trace output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_output_printed_after_reply(
    capsys: pytest.CaptureFixture,
) -> None:
    """Trace rendered output appears in stdout after the reply, not before.

    Strategy: patch render_trace so it emits a known sentinel string, then
    verify the sentinel appears in stdout after the reply text.  This avoids
    depending on what records the CollectingTracer actually collected (which
    would be empty in a unit test with a mocked Agent anyway) while still
    testing the observable ordering requirement.
    """
    mock_config = _make_mock_config()

    TRACE_SENTINEL = "<<TRACE_OUTPUT_SENTINEL>>"

    with (
        patch("sr2_spectre.cli.resolve_config", return_value=mock_config),
        patch("sr2_spectre.cli._configure_logging"),
        patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(_make_mock_interface("Paris")),),
        patch("sr2_spectre.cli.Agent") as MockAgent,
        # render_trace may be imported as a name into cli (from ... import render_trace)
        # or called qualified (sr2.pipeline.tracing.render_trace).  Patch both so the
        # test works regardless of which import form the implementer chooses.
        # create=True allows patching names that don't exist in the module yet.
        patch(
            "sr2_spectre.cli.render_trace",
            return_value=TRACE_SENTINEL,
            create=True,
        ),
        patch(
            "sr2.pipeline.tracing.render_trace",
            return_value=TRACE_SENTINEL,
        ),
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.handle_user_message.return_value = TurnResult(
            text="Paris", tool_calls_executed=0, total_tokens=10
        )
        MockAgent.return_value = mock_agent_instance

        await run_async(["config.yaml", "What is the capital of France?", "--trace"])

    captured = capsys.readouterr()
    stdout = captured.out

    # Reply must be present (AC6)
    assert "Paris" in stdout, f"Expected reply 'Paris' in stdout, got: {stdout!r}"

    # Trace output must be present (AC4)
    assert TRACE_SENTINEL in stdout, (
        f"Expected trace sentinel in stdout, got: {stdout!r}"
    )

    # Reply must appear BEFORE trace in stdout (AC4)
    reply_pos = stdout.index("Paris")
    trace_pos = stdout.index(TRACE_SENTINEL)
    assert reply_pos < trace_pos, (
        f"Expected reply before trace: reply at {reply_pos}, trace at {trace_pos}"
    )


# ---------------------------------------------------------------------------
# 5: Without --trace, no trace output in stdout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_trace_output_when_flag_absent(
    capsys: pytest.CaptureFixture,
) -> None:
    """When --trace is omitted, render_trace is never called — no timeline rendered."""
    mock_config = _make_mock_config()

    with (
        patch("sr2_spectre.cli.resolve_config", return_value=mock_config),
        patch("sr2_spectre.cli._configure_logging"),
        patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(_make_mock_interface("Paris")),),
        patch("sr2_spectre.cli.Agent") as MockAgent,
        patch("sr2_spectre.cli.render_trace", return_value="TRACE", create=True) as mock_rt,
    ):
        mock_agent_instance = AsyncMock()
        mock_agent_instance.handle_user_message.return_value = TurnResult(
            text="Paris", tool_calls_executed=0, total_tokens=10
        )
        MockAgent.return_value = mock_agent_instance

        await run_async(["config.yaml", "What is the capital of France?"])

    captured = capsys.readouterr()
    stdout = captured.out

    # Reply is present
    assert "Paris" in stdout

    # render_trace must not have been called at all
    mock_rt.assert_not_called()
