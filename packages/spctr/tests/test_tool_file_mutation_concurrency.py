"""Concurrency tests for the file-mutating builtin tools (edit, file_write).

SR2 runs every tool call of a turn with asyncio.gather, so calls that target
the same file must not lose updates. A small delay is injected at the
filesystem boundary (builtins.open) for files under the test workspace to
widen the read/write window deterministically.
"""
import asyncio
import builtins
import os
import time

import pytest

from sr2_spectre.tools.builtins.edit import EditTool
from sr2_spectre.tools.builtins.file_write import FileWriteTool

_DELAY = 0.02
_N = 8


@pytest.fixture
def slow_fs(tmp_path, monkeypatch):
    """Delay every open() of a path under tmp_path."""
    real_open = builtins.open
    root = str(tmp_path.resolve())

    def slow_open(file, *args, **kwargs):
        try:
            target = os.path.realpath(os.fspath(file))
        except TypeError:
            target = ""
        if target.startswith(root):
            time.sleep(_DELAY)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", slow_open)


def _marker_file(path, n=_N):
    path.write_text("\n".join(f"marker{i}" for i in range(n)) + "\n", encoding="utf-8")


async def _edit_all(tool, path, indices):
    return await asyncio.gather(
        *(tool(path, f"marker{i}", f"DONE{i}") for i in indices)
    )


@pytest.mark.asyncio
async def test_concurrent_edits_different_regions_all_land(tmp_path, slow_fs) -> None:
    target = tmp_path / "f.txt"
    _marker_file(target)
    tool = EditTool(workspace_root=str(tmp_path))

    results = await _edit_all(tool, str(target), range(_N))

    assert all(r == f"Made 1 replacement(s) in {target}" for r in results)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines == [f"DONE{i}" for i in range(_N)]


@pytest.mark.asyncio
async def test_concurrent_edits_two_regions_both_land(tmp_path, slow_fs) -> None:
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = EditTool(workspace_root=str(tmp_path))

    r1, r2 = await asyncio.gather(
        tool(str(target), "alpha", "ALPHA"),
        tool(str(target), "beta", "BETA"),
    )

    assert r1 == f"Made 1 replacement(s) in {target}"
    assert r2 == f"Made 1 replacement(s) in {target}"
    assert target.read_text(encoding="utf-8") == "ALPHA\nBETA\n"


@pytest.mark.asyncio
async def test_concurrent_edits_via_separate_tool_instances_all_land(
    tmp_path, slow_fs
) -> None:
    target = tmp_path / "f.txt"
    _marker_file(target)
    tools = [EditTool(workspace_root=str(tmp_path)) for _ in range(_N)]

    await asyncio.gather(
        *(t(str(target), f"marker{i}", f"DONE{i}") for i, t in enumerate(tools))
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines == [f"DONE{i}" for i in range(_N)]


@pytest.mark.asyncio
async def test_concurrent_edits_relative_and_absolute_path_both_land(
    tmp_path, slow_fs
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = EditTool(workspace_root=str(tmp_path))

    await asyncio.gather(
        tool("f.txt", "alpha", "ALPHA"),
        tool(str(target), "beta", "BETA"),
    )

    assert target.read_text(encoding="utf-8") == "ALPHA\nBETA\n"


@pytest.mark.asyncio
async def test_concurrent_edits_via_symlink_and_real_path_both_land(
    tmp_path, slow_fs
) -> None:
    target = tmp_path / "f.txt"
    link = tmp_path / "link.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    link.symlink_to(target)
    tool = EditTool(workspace_root=str(tmp_path))

    await asyncio.gather(
        tool(str(link), "alpha", "ALPHA"),
        tool(str(target), "beta", "BETA"),
    )

    assert target.read_text(encoding="utf-8") == "ALPHA\nBETA\n"


@pytest.mark.asyncio
async def test_concurrent_edit_and_file_write_equal_a_serial_order(
    tmp_path, slow_fs
) -> None:
    target = tmp_path / "f.txt"
    edit = EditTool(workspace_root=str(tmp_path))
    writer = FileWriteTool(workspace_root=str(tmp_path))

    for round_no in range(20):
        target.write_text("alpha original\n", encoding="utf-8")

        await asyncio.gather(
            edit(str(target), "alpha", "ALPHA"),
            writer(str(target), "alpha written\n"),
        )

        # edit-then-write -> "alpha written"; write-then-edit -> "ALPHA written"
        assert target.read_text(encoding="utf-8") in {
            "alpha written\n",
            "ALPHA written\n",
        }, f"round {round_no}"


@pytest.mark.asyncio
async def test_failed_edit_does_not_block_later_edit(tmp_path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("alpha\nalpha\nbeta\n", encoding="utf-8")
    tool = EditTool(workspace_root=str(tmp_path))

    with pytest.raises(ValueError):
        await tool(str(target), "missing", "x")
    with pytest.raises(ValueError):
        await tool(str(target), "alpha", "x")
    assert target.read_text(encoding="utf-8") == "alpha\nalpha\nbeta\n"

    result = await asyncio.wait_for(tool(str(target), "beta", "BETA"), timeout=5)

    assert result == f"Made 1 replacement(s) in {target}"
    assert target.read_text(encoding="utf-8") == "alpha\nalpha\nBETA\n"


@pytest.mark.asyncio
async def test_concurrent_failing_and_succeeding_edits_both_resolve(
    tmp_path, slow_fs
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = EditTool(workspace_root=str(tmp_path))

    results = await asyncio.wait_for(
        asyncio.gather(
            tool(str(target), "missing", "x"),
            tool(str(target), "alpha", "ALPHA"),
            tool(str(target), "beta", "BETA"),
            return_exceptions=True,
        ),
        timeout=10,
    )

    assert isinstance(results[0], ValueError)
    assert results[1] == f"Made 1 replacement(s) in {target}"
    assert results[2] == f"Made 1 replacement(s) in {target}"
    assert target.read_text(encoding="utf-8") == "ALPHA\nBETA\n"


@pytest.mark.asyncio
async def test_file_write_creates_parent_dirs_and_edit_follows(tmp_path) -> None:
    writer = FileWriteTool(workspace_root=str(tmp_path))
    edit = EditTool(workspace_root=str(tmp_path))

    result = await writer("a/b/c.txt", "hello\n")
    assert result == f"Written 6 bytes to {tmp_path / 'a/b/c.txt'}"

    await edit("a/b/c.txt", "hello", "bye")
    assert (tmp_path / "a/b/c.txt").read_text(encoding="utf-8") == "bye\n"
