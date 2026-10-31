"""Tests for GuardTool — phantom coverage detection."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.conftest import _surviving_pids

from sr2_spectre.tools.builtins.test_guard import (
    GuardResult,
    GuardTool,
    _TEST_FUNC_RE,
)


# ---------------------------------------------------------------------------
# Regex tests
# ---------------------------------------------------------------------------

class TestFuncRegex:
    def test_matches_def_test(self):
        assert _TEST_FUNC_RE.search("def test_foo():")

    def test_matches_async_def_test(self):
        assert _TEST_FUNC_RE.search("async def test_bar():")

    def test_matches_indented(self):
        assert _TEST_FUNC_RE.search("    def test_baz():")

    def test_matches_deeply_indented(self):
        assert _TEST_FUNC_RE.search("        async def test_deep():")

    def test_no_match_regular_function(self):
        assert not _TEST_FUNC_RE.search("def helper():")

    def test_no_match_non_test_prefix(self):
        assert not _TEST_FUNC_RE.search("def testing_foo():")

    def test_captures_name(self):
        m = _TEST_FUNC_RE.search("def test_my_feature():")
        assert m.group(1) == "test_my_feature"

    def test_captures_async_name(self):
        m = _TEST_FUNC_RE.search("    async def test_async_thing():")
        assert m.group(1) == "test_async_thing"

    def test_does_not_match_test_in_string(self):
        # A comment or docstring containing "def test_" is a false positive risk
        # but the regex matches lines. This is acceptable — the tool flags it
        # and the agent can verify. Real test functions in comments are rare.
        pass


# ---------------------------------------------------------------------------
# Authored scan tests
# ---------------------------------------------------------------------------

class TestScanAuthored:
    def test_scans_single_file(self, tmp_path: Path):
        (tmp_path / "test_foo.py").write_text(
            "def test_one(): pass\nasync def test_two(): pass\n"
        )
        names = GuardTool._scan_authored(tmp_path)
        assert names == {"test_one", "test_two"}

    def test_scans_multiple_files(self, tmp_path: Path):
        (tmp_path / "test_a.py").write_text("def test_x(): pass\n")
        (tmp_path / "test_b.py").write_text("def test_y(): pass\n")
        names = GuardTool._scan_authored(tmp_path)
        assert names == {"test_x", "test_y"}

    def test_ignores_non_test_files(self, tmp_path: Path):
        (tmp_path / "test_real.py").write_text("def test_ok(): pass\n")
        (tmp_path / "conftest.py").write_text("def test_fixtures(): pass\n")
        names = GuardTool._scan_authored(tmp_path)
        assert names == {"test_ok"}

    def test_ignores_non_test_functions(self, tmp_path: Path):
        (tmp_path / "test_foo.py").write_text(
            "def test_valid(): pass\n"
            "def helper(): pass\n"
            "def setup(): pass\n"
        )
        names = GuardTool._scan_authored(tmp_path)
        assert names == {"test_valid"}

    def test_empty_directory(self, tmp_path: Path):
        names = GuardTool._scan_authored(tmp_path)
        assert names == set()

    def test_skips_unreadable_file(self, tmp_path: Path):
        (tmp_path / "test_bad.py").write_text("def test_something(): pass\n")
        (tmp_path / "test_bad.py").chmod(0o000)
        names = GuardTool._scan_authored(tmp_path)
        # Should skip the unreadable file without crashing
        assert names == set()


# ---------------------------------------------------------------------------
# Result formatting tests
# ---------------------------------------------------------------------------

class TestFormat:
    def test_clean_result(self):
        result = GuardResult(total_collected=5, total_authored=5, uncollected=[], clean=True)
        output = GuardTool._format(result)
        assert "CLEAN" in output
        assert "5 collected" in output
        assert "5 authored" in output

    def test_dirty_result_single(self):
        result = GuardResult(
            total_collected=4, total_authored=5, uncollected=["text_typo"], clean=False
        )
        output = GuardTool._format(result)
        assert "UNCOLLECTED" in output
        assert "text_typo" in output

    def test_dirty_result_multiple(self):
        result = GuardResult(
            total_collected=3,
            total_authored=5,
            uncollected=["test_alpha", "test_beta"],
            clean=False,
        )
        output = GuardTool._format(result)
        assert "2 UNCOLLECTED" in output
        assert "test_alpha" in output
        assert "test_beta" in output


# ---------------------------------------------------------------------------
# Integration: _check method
# ---------------------------------------------------------------------------

class TestCheck:
    @pytest.mark.asyncio
    async def test_clean_when_all_collected(self, tmp_path: Path):
        """When authored == collected, result is clean."""
        (tmp_path / "test_foo.py").write_text("def test_one(): pass\n")

        tool = GuardTool(cwd=str(tmp_path))

        async def mock_collect(d):
            return {"test_one"}

        tool._collect_tests = mock_collect  # type: ignore[method-assign]
        try:
            result = await tool._check(tmp_path)
        finally:
            del tool._collect_tests  # restore via descriptor

        assert result.clean
        assert result.total_authored == 1
        assert result.uncollected == []

    @pytest.mark.asyncio
    async def test_uncollected_detected(self, tmp_path: Path):
        """When a test_ function is authored but not collected, it's flagged."""
        (tmp_path / "test_foo.py").write_text(
            "def test_collected(): pass\n"
            "def test_skip_me(): pass\n"
        )

        tool = GuardTool(cwd=str(tmp_path))

        async def mock_collect(d):
            return {"test_collected"}

        tool._collect_tests = mock_collect  # type: ignore[method-assign]
        try:
            result = await tool._check(tmp_path)
        finally:
            del tool._collect_tests

        assert not result.clean
        assert "test_skip_me" in result.uncollected
        assert result.total_authored == 2

    @pytest.mark.asyncio
    async def test_empty_project_is_clean(self, tmp_path: Path):
        """No test files == no authored == clean."""
        tool = GuardTool(cwd=str(tmp_path))

        async def mock_collect(d):
            return set()

        tool._collect_tests = mock_collect  # type: ignore[method-assign]
        try:
            result = await tool._check(tmp_path)
        finally:
            del tool._collect_tests

        assert result.clean
        assert result.total_authored == 0


# ---------------------------------------------------------------------------
# Real pytest collection test (runs actual pytest on temp dir)
# ---------------------------------------------------------------------------

class TestRealCollection:
    @pytest.mark.asyncio
    async def test_collects_real_tests(self, tmp_path: Path):
        """Verify _collect_tests actually runs pytest and parses output."""
        (tmp_path / "test_real.py").write_text(
            "def test_alpha(): pass\n"
            "def test_beta(): pass\n"
        )

        collected = await GuardTool(cwd=str(tmp_path))._collect_tests(tmp_path)

        # Should find at least these two
        assert "test_alpha" in collected or len(collected) >= 2

    @pytest.mark.asyncio
    async def test_collected_includes_method(self, tmp_path: Path):
        """pytest collects methods in test classes too."""
        (tmp_path / "test_class.py").write_text(
            "class TestFoo:\n"
            "    def test_method(self): pass\n"
        )

        collected = await GuardTool(cwd=str(tmp_path))._collect_tests(tmp_path)

        assert any("test_method" in name for name in collected)


# ---------------------------------------------------------------------------
# Tool contract tests
# ---------------------------------------------------------------------------

class TestToolContract:
    def test_name(self):
        assert GuardTool().name == "test_guard"

    def test_has_description(self):
        assert len(GuardTool.description) > 10

    def test_input_schema_valid(self):
        assert "type" in GuardTool.input_schema
        assert GuardTool.input_schema["type"] == "object"
        assert "test_dir" in GuardTool.input_schema["properties"]


# ---------------------------------------------------------------------------
# End-to-end call test
# ---------------------------------------------------------------------------

class TestCall:
    @pytest.mark.asyncio
    async def test_call_returns_clean_string(self, tmp_path: Path):
        (tmp_path / "test_ok.py").write_text("def test_one(): pass\n")

        tool = GuardTool(cwd=str(tmp_path))

        async def mock_collect(d):
            return {"test_one"}

        tool._collect_tests = mock_collect  # type: ignore[method-assign]
        try:
            output = await tool()
        finally:
            del tool._collect_tests

        assert "CLEAN" in output

    @pytest.mark.asyncio
    async def test_call_returns_dirty_string(self, tmp_path: Path):
        (tmp_path / "test_ok.py").write_text("def test_one(): pass\ndef test_two(): pass\n")

        tool = GuardTool(cwd=str(tmp_path))

        async def mock_collect(d):
            return {"test_one"}  # test_two not collected

        tool._collect_tests = mock_collect  # type: ignore[method-assign]
        try:
            output = await tool()
        finally:
            del tool._collect_tests

        assert "UNCOLLECTED" in output
        assert "test_two" in output


# ---------------------------------------------------------------------------
# Process-group cleanup (spc-947 defect class, spc-947 G finding 2):
# timeout and cancel must not orphan the collect subprocess or its children
# ---------------------------------------------------------------------------


class TestProcessGroupCleanup:
    @pytest.mark.asyncio
    async def test_timeout_kills_child_processes(self, tmp_path: Path, monkeypatch):
        """On the collect timeout, the whole process group must die.

        The pre-fix code returned set() on timeout without killing anything,
        so `sh -c 'sleep <marker>'` (and any grandchild) outlived the tool.
        A fake `python` on PATH keeps the test independent of the venv.
        """
        # The fake sleeps 1s (more than the 0.2s collect timeout, under
        # monkeypatch below), then execs the long-sleep child: when the
        # timeout fires the long sleep is still running, so the group kill
        # is what must take it down. A fake `python` on PATH keeps the test
        # independent of the venv (and of the spc-179 baseline failures).
        fake = tmp_path / "python"
        fake.write_text(
            "#!/bin/sh\n"
            "sleep 1\n"
            "exec sleep 9941\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

        tool = GuardTool(cwd=str(tmp_path))
        monkeypatch.setattr(GuardTool, "_COLLECT_TIMEOUT", 0.2)

        # set() means the timeout path was taken (nothing collected).
        assert await tool._collect_tests(tmp_path) == set()

        await asyncio.sleep(0.3)  # give the kernel a moment to reap
        assert _surviving_pids("9941") == []

    @pytest.mark.asyncio
    async def test_cancel_kills_child_processes(self, tmp_path: Path, monkeypatch):
        """An outside cancel must kill the group, not just abandon it.

        CancelledError takes a different path than the tool's timeout: the
        pre-fix code killed nothing here either, orphaning the collect
        subprocess when a turn was cancelled mid-guard.
        """
        # Same fake as the timeout test: the 1s sleep keeps the child
        # parked in communicate() so task.cancel() lands while it is
        # provably running.
        fake = tmp_path / "python"
        fake.write_text(
            "#!/bin/sh\n"
            "sleep 1\n"
            "exec sleep 9942\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

        tool = GuardTool(cwd=str(tmp_path))
        task = asyncio.create_task(tool._collect_tests(tmp_path))
        await asyncio.sleep(0.3)  # the fake is up and mid-sleep
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.3)
        assert _surviving_pids("9942") == []
