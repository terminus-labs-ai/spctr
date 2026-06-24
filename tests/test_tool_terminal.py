"""Tests for TerminalTool.

These exercise the tool against REAL subprocesses (echo/printf/sleep/sh).
The subprocess is the tool's system boundary, so we drive real commands
rather than mocking asyncio internals like communicate() — that keeps the
tests grounded in observable behavior and robust to implementation changes
(e.g. switching from communicate() to incremental stream pumping).
"""
import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_terminal_class_attributes() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    assert isinstance(TerminalTool.name, str) and TerminalTool.name
    assert isinstance(TerminalTool.description, str) and TerminalTool.description
    assert isinstance(TerminalTool.input_schema, dict)


def test_terminal_input_schema_requires_command() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    schema = TerminalTool.input_schema
    assert "command" in schema.get("properties", {})
    assert "command" in schema.get("required", [])


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_terminal_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path("sr2_spectre.tools.builtins.terminal.TerminalTool")
    assert "terminal" in reg


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_returns_stdout() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    tool = TerminalTool()
    result = await tool(command="echo hello world")

    assert "hello world" in result


@pytest.mark.asyncio
async def test_terminal_combines_stdout_and_stderr() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    tool = TerminalTool()
    result = await tool(command="printf 'out\\n'; printf 'err\\n' >&2")

    assert "out" in result
    assert "err" in result


@pytest.mark.asyncio
async def test_terminal_empty_output_returns_empty_string() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    tool = TerminalTool()
    result = await tool(command="true")

    assert result == ""


# ---------------------------------------------------------------------------
# Non-zero exit code — output returned, no exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_nonzero_exit_returns_output_not_raises() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    tool = TerminalTool()
    result = await tool(command="printf 'command not found\\n' >&2; exit 127")

    assert "command not found" in result


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_timeout_raises_timeout_error() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    tool = TerminalTool(timeout=1)
    with pytest.raises(TimeoutError, match="sleep 999"):
        await tool(command="sleep 999")


@pytest.mark.asyncio
async def test_terminal_timeout_message_contains_command() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    cmd = "sleep 999 # some_long_running_command --flag"
    tool = TerminalTool(timeout=1)
    with pytest.raises(TimeoutError) as exc_info:
        await tool(command=cmd)

    assert cmd in str(exc_info.value)


@pytest.mark.asyncio
async def test_terminal_timeout_includes_partial_output() -> None:
    """On timeout, output produced before the kill is preserved in the error.

    Prints a line, then sleeps past the timeout. The raised TimeoutError must
    carry BOTH the timeout signal and the partial output already emitted, so
    the model is not starved of what actually ran. Regression for the
    silent-stop bug: timed-out pytest discarded all output, leaving the model
    with nothing and ending the turn.

    The marker is COMPUTED by the shell (21+21=42), so "42" appears in the
    subprocess OUTPUT but never in the command literal — proving the assertion
    passes because output was captured, not because the command string is
    echoed back in the error message.
    """
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    tool = TerminalTool(timeout=1)
    with pytest.raises(TimeoutError) as exc_info:
        await tool(command="echo $((21+21)); sleep 10")

    msg = str(exc_info.value)
    assert "42" in msg  # partial output captured before the kill
    assert "timed out" in msg.lower()  # timeout still signalled


# ---------------------------------------------------------------------------
# Custom timeout constructor arg
# ---------------------------------------------------------------------------

def test_terminal_custom_timeout_stored() -> None:
    from sr2_spectre.tools.builtins.terminal import TerminalTool

    tool = TerminalTool(timeout=60)
    assert tool.timeout == 60
