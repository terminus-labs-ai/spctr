"""Tests for obsidian-40c: Compiled-turn trace — CLI integration (FR5).

Covers:
  FR5: --trace output: after render_trace(...), also prints render_compiled_request output
       when tracer.compiled_request is not None.
       When --trace is not set, render_compiled_request is never called.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sr2_spectre.cli import run_async
from sr2_spectre.core import TurnResult


# ---------------------------------------------------------------------------
# Helpers — mirrors test_trace_flag.py conventions
# ---------------------------------------------------------------------------


def _fake_interface_cls(instance):
    """Class whose instantiation returns ``instance`` (for _load_interface patches)."""
    return type("_FakeInterfaceCls", (), {"__new__": staticmethod(lambda cls: instance)})


def _make_mock_config() -> MagicMock:
    """Minimal SpectreConfig stand-in."""
    config = MagicMock()
    config.agent.name = "test-agent"
    config.agent.tools = []
    config.models = {"default": MagicMock(model="test-model", base_url=None)}
    return config


def _make_mock_interface(reply: str = "Paris") -> MagicMock:
    """Mock interface whose run() calls agent.handle_user_message and prints result."""
    plugin = MagicMock()
    plugin.start = AsyncMock()
    plugin.stop = AsyncMock()

    async def _run(agent: MagicMock) -> None:
        result = await agent.handle_user_message("What is the capital of France?")
        print(result.text)

    plugin.run = _run
    return plugin


def _make_mock_agent_instance(reply: str = "Paris") -> AsyncMock:
    mock = AsyncMock()
    mock.handle_user_message.return_value = TurnResult(
        text=reply, tool_calls_executed=0, total_tokens=10
    )
    return mock


COMPILED_REQUEST_SENTINEL = "<<COMPILED_REQUEST_SENTINEL>>"
TRACE_SENTINEL = "<<TRACE_SENTINEL>>"


def _agent_side_effect_with_tracer(reply: str = "Paris"):
    """Factory: returns a MockAgent side_effect that captures the tracer kwarg and
    populates compiled_request on it — simulating what the real engine does."""
    from sr2.pipeline.tracing import CollectingTracer

    mock_instance = _make_mock_agent_instance(reply)

    def _side_effect(*args, **kwargs):
        tracer = kwargs.get("tracer")
        if isinstance(tracer, CollectingTracer):
            # Simulate the engine calling tracer.on_compile() during run().
            # In production, engine.run() → _compile_request() → tracer.on_compile().
            # The mock agent bypasses the engine, so we populate it here.
            from sr2.protocols.llm import CompletionRequest
            from sr2.models import Message, TextBlock
            tracer.on_compile(CompletionRequest(
                messages=[Message(role="user", content=[TextBlock(text="test")])],
            ))
        return mock_instance

    return _side_effect, mock_instance


# ---------------------------------------------------------------------------
# FR5a — render_compiled_request is printed when --trace is set and
#         compiled_request is not None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compiled_request_output_printed_when_trace_set(
    capsys: pytest.CaptureFixture,
) -> None:
    """When --trace is set and compiled_request is not None, render_compiled_request output appears in stdout."""
    mock_config = _make_mock_config()

    side_effect, mock_instance = _agent_side_effect_with_tracer("Paris")

    with (
        patch("sr2_spectre.cli.resolve_config", return_value=mock_config),
        patch("sr2_spectre.cli._configure_logging"),
        patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(_make_mock_interface("Paris"))),
        patch("sr2_spectre.cli.Agent") as MockAgent,
        patch(
            "sr2_spectre.cli.render_trace",
            return_value=TRACE_SENTINEL,
            create=True,
        ),
        patch(
            "sr2_spectre.cli.render_compiled_request",
            return_value=COMPILED_REQUEST_SENTINEL,
            create=True,
        ) as mock_rcr,
    ):
        MockAgent.side_effect = side_effect

        await run_async(["config.yaml", "What is the capital of France?", "--trace"])

    captured = capsys.readouterr()
    stdout = captured.out

    # render_compiled_request must have been called
    mock_rcr.assert_called_once()

    # Its return value must appear in stdout
    assert COMPILED_REQUEST_SENTINEL in stdout, (
        f"Expected compiled request sentinel in stdout, got: {stdout!r}"
    )


@pytest.mark.asyncio
async def test_compiled_request_output_appears_after_firing_trace(
    capsys: pytest.CaptureFixture,
) -> None:
    """render_compiled_request output appears after render_trace output in stdout."""
    mock_config = _make_mock_config()

    side_effect, _ = _agent_side_effect_with_tracer("Paris")

    with (
        patch("sr2_spectre.cli.resolve_config", return_value=mock_config),
        patch("sr2_spectre.cli._configure_logging"),
        patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(_make_mock_interface("Paris"))),
        patch("sr2_spectre.cli.Agent") as MockAgent,
        patch(
            "sr2_spectre.cli.render_trace",
            return_value=TRACE_SENTINEL,
            create=True,
        ),
        patch(
            "sr2_spectre.cli.render_compiled_request",
            return_value=COMPILED_REQUEST_SENTINEL,
            create=True,
        ),
    ):
        MockAgent.side_effect = side_effect

        await run_async(["config.yaml", "What is the capital of France?", "--trace"])

    captured = capsys.readouterr()
    stdout = captured.out

    assert TRACE_SENTINEL in stdout, f"Firing trace sentinel not found in: {stdout!r}"
    assert COMPILED_REQUEST_SENTINEL in stdout, f"Compiled request sentinel not found in: {stdout!r}"

    trace_pos = stdout.index(TRACE_SENTINEL)
    compiled_pos = stdout.index(COMPILED_REQUEST_SENTINEL)
    assert trace_pos < compiled_pos, (
        f"Expected firing trace before compiled request: "
        f"trace at {trace_pos}, compiled at {compiled_pos}"
    )


@pytest.mark.asyncio
async def test_compiled_request_render_called_with_compiled_request_attribute(
    capsys: pytest.CaptureFixture,
) -> None:
    """render_compiled_request is called with tracer.compiled_request as its argument."""
    mock_config = _make_mock_config()
    captured_args: list = []

    def capturing_rcr(request):
        captured_args.append(request)
        return COMPILED_REQUEST_SENTINEL

    side_effect, _ = _agent_side_effect_with_tracer("Paris")

    with (
        patch("sr2_spectre.cli.resolve_config", return_value=mock_config),
        patch("sr2_spectre.cli._configure_logging"),
        patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(_make_mock_interface("Paris"))),
        patch("sr2_spectre.cli.Agent") as MockAgent,
        patch(
            "sr2_spectre.cli.render_trace",
            return_value=TRACE_SENTINEL,
            create=True,
        ),
        patch(
            "sr2_spectre.cli.render_compiled_request",
            side_effect=capturing_rcr,
            create=True,
        ),
    ):
        MockAgent.side_effect = side_effect

        await run_async(["config.yaml", "What is the capital of France?", "--trace"])

    # render_compiled_request must have been called with a non-None argument
    assert len(captured_args) == 1, "render_compiled_request was not called exactly once"
    assert captured_args[0] is not None, (
        "render_compiled_request was called with None — "
        "should only be called when compiled_request is not None"
    )


# ---------------------------------------------------------------------------
# FR5b — render_compiled_request is NOT called when --trace is absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_compiled_request_not_called_when_trace_absent(
    capsys: pytest.CaptureFixture,
) -> None:
    """When --trace is not set, render_compiled_request is never called."""
    mock_config = _make_mock_config()

    with (
        patch("sr2_spectre.cli.resolve_config", return_value=mock_config),
        patch("sr2_spectre.cli._configure_logging"),
        patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(_make_mock_interface("Paris"))),
        patch("sr2_spectre.cli.Agent") as MockAgent,
        patch(
            "sr2_spectre.cli.render_compiled_request",
            return_value=COMPILED_REQUEST_SENTINEL,
            create=True,
        ) as mock_rcr,
    ):
        MockAgent.return_value = _make_mock_agent_instance("Paris")

        await run_async(["config.yaml", "What is the capital of France?"])

    mock_rcr.assert_not_called()

    captured = capsys.readouterr()
    assert COMPILED_REQUEST_SENTINEL not in captured.out


@pytest.mark.asyncio
async def test_compiled_request_sentinel_absent_from_stdout_without_trace_flag(
    capsys: pytest.CaptureFixture,
) -> None:
    """Without --trace, compiled request output does not appear in stdout even if patched."""
    mock_config = _make_mock_config()

    with (
        patch("sr2_spectre.cli.resolve_config", return_value=mock_config),
        patch("sr2_spectre.cli._configure_logging"),
        patch("sr2_spectre.cli._load_interface", return_value=_fake_interface_cls(_make_mock_interface("Paris"))),
        patch("sr2_spectre.cli.Agent") as MockAgent,
        patch(
            "sr2_spectre.cli.render_trace",
            return_value=TRACE_SENTINEL,
            create=True,
        ),
        patch(
            "sr2_spectre.cli.render_compiled_request",
            return_value=COMPILED_REQUEST_SENTINEL,
            create=True,
        ),
    ):
        MockAgent.return_value = _make_mock_agent_instance("Paris")

        # No --trace flag
        await run_async(["config.yaml", "What is the capital of France?"])

    captured = capsys.readouterr()
    assert COMPILED_REQUEST_SENTINEL not in captured.out
    assert TRACE_SENTINEL not in captured.out
