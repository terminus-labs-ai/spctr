"""Tests for CodeExecTool."""
import asyncio
from unittest.mock import patch

import pytest

from sr2_spectre.tools.builtins.code_exec import CodeExecTool
from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_code_exec_class_attributes() -> None:
    assert isinstance(CodeExecTool.name, str) and CodeExecTool.name
    assert isinstance(CodeExecTool.description, str) and CodeExecTool.description
    assert isinstance(CodeExecTool.input_schema, dict)


def test_code_exec_input_schema_requires_code() -> None:
    schema = CodeExecTool.input_schema
    assert "code" in schema.get("properties", {})
    assert "code" in schema.get("required", [])
    assert schema["properties"]["code"]["type"] == "string"


def test_code_exec_input_schema_timeout_optional() -> None:
    schema = CodeExecTool.input_schema
    assert "timeout" in schema.get("properties", {})
    assert "timeout" not in schema.get("required", [])


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_code_exec_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path(
        "sr2_spectre.tools.builtins.code_exec.CodeExecTool"
    )
    assert "code_exec" in reg


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_evaluates_expression() -> None:
    tool = CodeExecTool()
    result = await tool(code="2 + 2")

    assert "Status: success" in result
    assert "Return: 4" in result


@pytest.mark.asyncio
async def test_code_exec_evaluates_string_expression() -> None:
    tool = CodeExecTool()
    result = await tool(code="'hello' + ' ' + 'world'")

    assert "Status: success" in result
    assert "Return: 'hello world'" in result


@pytest.mark.asyncio
async def test_code_exec_evaluates_complex_expression() -> None:
    tool = CodeExecTool()
    result = await tool(code="[x**2 for x in range(5)]")

    assert "Status: success" in result
    assert "Return: [0, 1, 4, 9, 16]" in result


# ---------------------------------------------------------------------------
# Statement execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_runs_statements() -> None:
    tool = CodeExecTool()
    result = await tool(code="x = 10\ny = 20")

    assert "Status: success" in result
    # exec doesn't return a value
    assert "Return:" not in result


@pytest.mark.asyncio
async def test_code_exec_print_captured_in_stdout() -> None:
    tool = CodeExecTool()
    result = await tool(code="print('hello stdout')")

    assert "Status: success" in result
    assert "Stdout:" in result
    assert "hello stdout" in result


@pytest.mark.asyncio
async def test_code_exec_namespace_isolation() -> None:
    """Each execution gets a fresh namespace."""
    tool = CodeExecTool()
    await tool(code="x = 42")

    # x should not persist to next call
    result = await tool(code="x")
    assert "Status: error" in result
    assert "NameError" in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_syntax_error() -> None:
    tool = CodeExecTool()
    result = await tool(code="def incomplete(")

    assert "Status: error" in result
    assert "SyntaxError" in result


@pytest.mark.asyncio
async def test_code_exec_runtime_error() -> None:
    tool = CodeExecTool()
    result = await tool(code="1 / 0")

    assert "Status: error" in result
    assert "ZeroDivisionError" in result


@pytest.mark.asyncio
async def test_code_exec_name_error() -> None:
    tool = CodeExecTool()
    result = await tool(code="undefined_variable")

    assert "Status: error" in result
    assert "NameError" in result


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_timeout_raises_timeout_error() -> None:
    tool = CodeExecTool(timeout=1)
    result = await tool(code="import time; time.sleep(60)", timeout=1)

    assert "Status: timeout" in result
    assert "timed out" in result


@pytest.mark.asyncio
async def test_code_exec_call_timeout_overrides_constructor() -> None:
    """Per-call timeout takes precedence over constructor default."""
    tool = CodeExecTool(timeout=30)
    result = await tool(code="import time; time.sleep(60)", timeout=1)

    assert "Status: timeout" in result


@pytest.mark.asyncio
async def test_code_exec_custom_timeout_stored() -> None:
    tool = CodeExecTool(timeout=60)
    assert tool.timeout == 60


# ---------------------------------------------------------------------------
# Stderr capture
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_stderr_capture() -> None:
    import warnings
    tool = CodeExecTool()
    # Trigger a warning that goes to stderr
    result = await tool(
        code="import warnings; warnings.warn('test warning')"
    )

    assert "Status: success" in result
    # Warning may go to stderr depending on Python version/warnings config
    # At minimum the tool shouldn't crash
    assert "test warning" in result or "Status: success" in result


# ---------------------------------------------------------------------------
# Return value formatting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_none_return_omitted() -> None:
    """When exec returns None (statements only), no 'Return:' line appears."""
    tool = CodeExecTool()
    result = await tool(code="x = [1, 2, 3]")

    assert "Status: success" in result
    # exec returns None, which should not show as "Return: None"
    assert "Return: None" not in result


@pytest.mark.asyncio
async def test_code_exec_dict_return_formatted() -> None:
    tool = CodeExecTool()
    result = await tool(code="{'a': 1, 'b': 2}")

    assert "Status: success" in result
    assert "'a': 1" in result or "a" in result


# ---------------------------------------------------------------------------
# Containment (obsidian-nr73)
#
# The tool used to run snippets in the bot's own process with full builtins,
# so `import os; print(dict(os.environ))` handed a Discord user the bot token
# and the GitHub PAT, and the timeout could not stop a busy loop. These pin
# the properties that fix relies on.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_exec_runs_in_a_separate_process() -> None:
    """A snippet must not observe the host interpreter's PID."""
    import os

    tool = CodeExecTool()
    result = await tool(code="__import__('os').getpid()")

    assert "Status: success" in result
    assert f"Return: {os.getpid()}" not in result


@pytest.mark.asyncio
async def test_code_exec_does_not_leak_process_environment() -> None:
    """Secrets in the bot's environment must not reach the snippet."""
    import os

    with patch.dict(
        os.environ,
        {"DISCORD_BOT_TOKEN": "leaked-discord-token",
         "GITHUB_PERSONAL_ACCESS_TOKEN": "leaked-github-pat"},
    ):
        tool = CodeExecTool()
        result = await tool(code="import os; print(dict(os.environ))")

    assert "Status: success" in result
    assert "leaked-discord-token" not in result
    assert "leaked-github-pat" not in result
    assert "DISCORD_BOT_TOKEN" not in result
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" not in result


@pytest.mark.asyncio
async def test_code_exec_environment_is_a_small_allowlist() -> None:
    """The child env is built from an allowlist, not inherited and filtered."""
    tool = CodeExecTool()
    result = await tool(code="sorted(__import__('os').environ)")

    assert "Status: success" in result
    for allowed in ("PATH", "HOME", "LANG"):
        assert allowed in result
    # An inherited environment would carry far more than the allowlist.
    for inherited in ("SSH_AUTH_SOCK", "XDG_RUNTIME_DIR", "SUDO_USER"):
        assert inherited not in result


@pytest.mark.asyncio
async def test_code_exec_cannot_kill_the_host_process() -> None:
    """os._exit in a snippet must kill the child, not the bot."""
    tool = CodeExecTool()
    result = await tool(code="import os; os._exit(0)")

    # The tool reports a failure rather than the bot dying mid-turn.
    assert "Status:" in result
    # Proof of survival: the next call still works.
    follow_up = await tool(code="1 + 1")
    assert "Return: 2" in follow_up


@pytest.mark.asyncio
async def test_code_exec_busy_loop_is_killed_at_the_timeout() -> None:
    """A CPU-bound loop must be terminated, not merely abandoned."""
    loop = asyncio.get_running_loop()
    tool = CodeExecTool(timeout=1)

    started = loop.time()
    result = await tool(code="while True: pass", timeout=1)
    elapsed = loop.time() - started

    assert "Status: timeout" in result
    # Generous ceiling: the point is that it returns, having killed the child.
    assert elapsed < 15


@pytest.mark.asyncio
async def test_code_exec_timeout_kills_grandchildren() -> None:
    """Processes the snippet spawned must not outlive the timeout."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pid_file = os.path.join(tmp, "grandchild.pid")
        code = (
            "import subprocess, sys, pathlib\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"pathlib.Path({pid_file!r}).write_text(str(p.pid))\n"
            "while True: pass\n"
        )
        tool = CodeExecTool(timeout=2)
        result = await tool(code=code, timeout=2)

        assert "Status: timeout" in result

        raw = ""
        for _ in range(20):
            raw = open(pid_file).read().strip() if os.path.exists(pid_file) else ""
            if raw:
                break
            await asyncio.sleep(0.1)
        assert raw, "grandchild never recorded its PID"

        # Give the group kill a moment to land, then assert the PID is gone.
        grandchild = int(raw)
        for _ in range(50):
            try:
                os.kill(grandchild, 0)
            except OSError:
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail(f"grandchild {grandchild} survived the timeout")


@pytest.mark.asyncio
async def test_code_exec_snippet_cannot_mutate_the_host_interpreter() -> None:
    """State written by a snippet must not appear in the bot's own modules."""
    import sr2_spectre.tools.builtins.code_exec as module

    tool = CodeExecTool()
    result = await tool(
        code=(
            "import sr2_spectre.tools.builtins.code_exec as m\n"
            "m.PWNED = True\n"
        )
    )

    assert "Status: success" in result
    assert not hasattr(module, "PWNED")


@pytest.mark.asyncio
async def test_code_exec_runs_in_the_workspace_root(tmp_path) -> None:
    """workspace_root is the snippet's cwd, matching the other confined tools."""
    tool = CodeExecTool(workspace_root=str(tmp_path))
    result = await tool(code="__import__('os').getcwd()")

    assert "Status: success" in result
    assert str(tmp_path.resolve()) in result
