"""Tests for sr2_spectre.scoping — per-session systemd scope entry.

Covers obsidian-y10f acceptance criteria:
- Re-exec under systemd-run when not already scoped.
- Skip when SR2_SCOPE_UNIT is set (gmap/workflow already scoped).
- Skip when INVOCATION_ID is set (systemd service).
- Skip when systemd-run is unavailable (clear message, no failure).
- Unit naming from --session-id or PID.
- Configurable memory/swap limits via environment.
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import patch

import pytest

from sr2_spectre.scoping import (
    _DEFAULT_MEMORY_MAX,
    _DEFAULT_MEMORY_SWAP_MAX,
    _memory_max,
    _memory_swap_max,
    _unit_name,
    maybe_enter_scope,
)


class TestUnitName:
    def test_uses_session_id_flag(self):
        argv = ["config.yaml", "hello", "--session-id", "my-bead-42"]
        assert _unit_name(argv) == "spectre-my-bead-42"

    def test_uses_session_id_equals_form(self):
        argv = ["--session-id=role-C-001", "--agent", "edi"]
        assert _unit_name(argv) == "spectre-role-C-001"

    def test_falls_back_to_pid(self, monkeypatch):
        monkeypatch.setattr(os, "getpid", lambda: 99999)
        argv = ["config.yaml", "--interface", "repl"]
        assert _unit_name(argv) == "spectre-99999"

    def test_sanitizes_unsafe_chars(self):
        argv = ["--session-id", "foo/bar baz!"]
        result = _unit_name(argv)
        assert "/" not in result
        assert " " not in result
        assert "!" not in result
        assert result.startswith("spectre-")

    def test_preserves_safe_chars(self):
        argv = ["--session-id", "abc.def:ghi_jkl-mno"]
        assert _unit_name(argv) == "spectre-abc.def:ghi_jkl-mno"


class TestConfigDefaults:
    def test_memory_max_default(self, monkeypatch):
        monkeypatch.delenv("SR2_SCOPE_MEMORY_MAX", raising=False)
        assert _memory_max() == _DEFAULT_MEMORY_MAX

    def test_memory_max_override(self, monkeypatch):
        monkeypatch.setenv("SR2_SCOPE_MEMORY_MAX", "10G")
        assert _memory_max() == "10G"

    def test_swap_max_default(self, monkeypatch):
        monkeypatch.delenv("SR2_SCOPE_MEMORY_SWAP_MAX", raising=False)
        assert _memory_swap_max() == _DEFAULT_MEMORY_SWAP_MAX

    def test_swap_max_override(self, monkeypatch):
        monkeypatch.setenv("SR2_SCOPE_MEMORY_SWAP_MAX", "2G")
        assert _memory_swap_max() == "2G"


class TestSkipConditions:
    def test_skips_when_sr2_scope_unit_set(self, monkeypatch):
        monkeypatch.setenv("SR2_SCOPE_UNIT", "gmap-existing.scope")
        with patch("sr2_spectre.scoping.os.execvp") as mock_exec:
            maybe_enter_scope(["config.yaml"])
            mock_exec.assert_not_called()

    def test_skips_when_invocation_id_set(self, monkeypatch):
        monkeypatch.delenv("SR2_SCOPE_UNIT", raising=False)
        monkeypatch.setenv("INVOCATION_ID", "abc123def456")
        with patch("sr2_spectre.scoping.os.execvp") as mock_exec:
            maybe_enter_scope(["config.yaml"])
            mock_exec.assert_not_called()

    def test_skips_when_sr2_no_scope_set(self, monkeypatch):
        monkeypatch.delenv("SR2_SCOPE_UNIT", raising=False)
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.setenv("SR2_NO_SCOPE", "1")
        with patch("sr2_spectre.scoping.os.execvp") as mock_exec:
            maybe_enter_scope(["config.yaml"])
            mock_exec.assert_not_called()

    def test_skips_when_systemd_run_unavailable(self, monkeypatch, capsys):
        monkeypatch.delenv("SR2_SCOPE_UNIT", raising=False)
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.delenv("SR2_NO_SCOPE", raising=False)
        with (
            patch("sr2_spectre.scoping._systemd_run_available", return_value=False),
            patch("sr2_spectre.scoping.os.execvp") as mock_exec,
        ):
            maybe_enter_scope(["config.yaml"])
            mock_exec.assert_not_called()
        err = capsys.readouterr().err
        assert "systemd-run not found" in err

    def test_skips_when_systemd_user_not_running(self, monkeypatch, capsys):
        monkeypatch.delenv("SR2_SCOPE_UNIT", raising=False)
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.delenv("SR2_NO_SCOPE", raising=False)
        with (
            patch("sr2_spectre.scoping._systemd_run_available", return_value=True),
            patch("sr2_spectre.scoping._systemd_user_running", return_value=False),
            patch("sr2_spectre.scoping.os.execvp") as mock_exec,
        ):
            maybe_enter_scope(["config.yaml"])
            mock_exec.assert_not_called()
        err = capsys.readouterr().err
        assert "not reachable" in err


class TestReExec:
    def test_execs_systemd_run_with_correct_args(self, monkeypatch):
        monkeypatch.delenv("SR2_SCOPE_UNIT", raising=False)
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.delenv("SR2_NO_SCOPE", raising=False)
        monkeypatch.setenv("SR2_SCOPE_MEMORY_MAX", "8G")
        monkeypatch.setenv("SR2_SCOPE_MEMORY_SWAP_MAX", "1G")
        monkeypatch.setattr(os, "getpid", lambda: 12345)

        fake_argv = ["/usr/bin/sr2-spectre", "config.yaml", "hello"]
        monkeypatch.setattr(sys, "argv", fake_argv)

        with (
            patch("sr2_spectre.scoping._systemd_run_available", return_value=True),
            patch("sr2_spectre.scoping._systemd_user_running", return_value=True),
            patch("sr2_spectre.scoping.os.execvp") as mock_exec,
        ):
            maybe_enter_scope(["config.yaml", "hello"])

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[0] == "systemd-run"
        cmd = args[1]
        assert "--user" in cmd
        assert "--scope" in cmd
        assert "--quiet" in cmd
        assert "--unit=spectre-12345" in cmd
        assert "-p" in cmd
        assert "MemoryMax=8G" in cmd
        assert "MemorySwapMax=1G" in cmd
        assert "--" in cmd
        assert cmd[cmd.index("--") + 1:] == fake_argv

    def test_sets_sr2_scope_unit_before_exec(self, monkeypatch):
        monkeypatch.delenv("SR2_SCOPE_UNIT", raising=False)
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.delenv("SR2_NO_SCOPE", raising=False)
        monkeypatch.setattr(os, "getpid", lambda: 42)
        monkeypatch.setattr(sys, "argv", ["/usr/bin/sr2-spectre"])

        env_snapshot: dict[str, str] = {}

        def capture_exec(file, args, **kwargs):
            env_snapshot.update(os.environ)

        with (
            patch("sr2_spectre.scoping._systemd_run_available", return_value=True),
            patch("sr2_spectre.scoping._systemd_user_running", return_value=True),
            patch("sr2_spectre.scoping.os.execvp", side_effect=capture_exec),
        ):
            maybe_enter_scope([])

        assert env_snapshot.get("SR2_SCOPE_UNIT") == "spectre-42"

    def test_uses_session_id_in_unit_name(self, monkeypatch):
        monkeypatch.delenv("SR2_SCOPE_UNIT", raising=False)
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.delenv("SR2_NO_SCOPE", raising=False)
        monkeypatch.setattr(sys, "argv", ["/usr/bin/sr2-spectre", "--session-id", "bead-7"])

        with (
            patch("sr2_spectre.scoping._systemd_run_available", return_value=True),
            patch("sr2_spectre.scoping._systemd_user_running", return_value=True),
            patch("sr2_spectre.scoping.os.execvp") as mock_exec,
        ):
            maybe_enter_scope(["--session-id", "bead-7"])

        cmd = mock_exec.call_args[0][1]
        assert "--unit=spectre-bead-7" in cmd


class TestOOMToolErrorSurfacing:
    """AC2 integration: a tool call whose child OOM-kills surfaces as a tool
    error and sibling calls in the same session continue normally."""

    @pytest.mark.asyncio
    async def test_oom_killed_child_surfaces_as_tool_output(self, tmp_path):
        """A subprocess killed by SIGKILL (simulating cgroup OOM) returns its
        partial output with 'Killed' rather than crashing the session."""
        from sr2_spectre.tools.builtins.terminal import TerminalTool

        tool = TerminalTool(timeout=10)
        # Allocate past RLIMIT_AS (64MB); kernel sends SIGKILL.
        result = await tool(
            "bash -c 'ulimit -v 65536; python3 -c \""
            "import sys; sys.stdout.write('before_kill\\\\n'); sys.stdout.flush(); "
            "data = bytearray(100*1024*1024)"
            "\"'"
        )
        # The session survived: we got a result string back.
        assert isinstance(result, str)
        # The kernel or shell reports the kill.
        assert "Killed" in result or "before_kill" in result

    @pytest.mark.asyncio
    async def test_sibling_call_survives_oom_sibling(self):
        """Concurrent tool calls: one OOM'd, the other completes normally."""
        from sr2_spectre.tools.builtins.terminal import TerminalTool

        tool = TerminalTool(timeout=10)
        oom_cmd = (
            "bash -c 'ulimit -v 65536; python3 -c \""
            "data = bytearray(100*1024*1024)"
            "\"'"
        )
        ok_cmd = "echo alive"

        results = await asyncio.gather(
            tool(oom_cmd),
            tool(ok_cmd),
            return_exceptions=True,
        )
        oom_result, ok_result = results
        # The healthy call completed normally.
        assert isinstance(ok_result, str)
        assert "alive" in ok_result
        # The OOM call either returned partial output or raised — either way
        # it did not take down the event loop or the sibling.
        assert isinstance(oom_result, str) or isinstance(oom_result, Exception)


class TestSystemdUserRunning:
    def test_returns_true_for_running(self, monkeypatch):
        import subprocess
        monkeypatch.delenv("SR2_SCOPE_UNIT", raising=False)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = b"running\n"
            from sr2_spectre.scoping import _systemd_user_running
            assert _systemd_user_running() is True

    def test_returns_true_for_degraded(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = b"degraded\n"
            from sr2_spectre.scoping import _systemd_user_running
            assert _systemd_user_running() is True

    def test_returns_false_on_oserror(self):
        with patch("subprocess.run", side_effect=OSError("not found")):
            from sr2_spectre.scoping import _systemd_user_running
            assert _systemd_user_running() is False

    def test_returns_false_on_timeout(self):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("x", 5)):
            from sr2_spectre.scoping import _systemd_user_running
            assert _systemd_user_running() is False
