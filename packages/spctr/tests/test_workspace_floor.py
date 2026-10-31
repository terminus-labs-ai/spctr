"""Tests for the workspace floor on the reading tools.

SR2_WORKSPACE used to confine writes only: file_read, grep, glob and
read_symbol took no workspace_root, so runtime.py skipped them and an agent
could read anything the process could. These pin the floor for reads.
"""
import os

import pytest

from sr2_spectre.tools.builtins.file_read import FileReadTool
from sr2_spectre.tools.builtins.glob import GlobTool
from sr2_spectre.tools.builtins.grep import GrepTool
from sr2_spectre.tools.builtins.read_symbol import ReadSymbolTool
from sr2_spectre.tools.workspace_floor import WorkspaceFloor


@pytest.fixture
def workspace(tmp_path):
    """A workspace with a secret parked just outside it."""
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "player.gd").write_text("extends Node\n\nfunc jump():\n\tpass\n")
    (ws / "notes.md").write_text("hello workspace\n")
    (tmp_path / "secret.env").write_text("TOKEN=super-secret\n")
    return ws, tmp_path


# --- the tools now accept the kwarg at all ---------------------------------

@pytest.mark.parametrize("cls", [FileReadTool, GrepTool, GlobTool, ReadSymbolTool])
def test_tool_accepts_workspace_root(cls, tmp_path):
    """runtime.py only injects the floor into tools whose __init__ takes it."""
    assert cls(workspace_root=str(tmp_path)) is not None


@pytest.mark.parametrize("cls", [FileReadTool, GrepTool, GlobTool, ReadSymbolTool])
def test_runtime_detects_the_kwarg(cls):
    from sr2_spectre.runtime import _tool_accepts_workspace_root
    path = f"{cls.__module__}.{cls.__name__}"
    assert _tool_accepts_workspace_root(path)


# --- reads are confined -----------------------------------------------------

@pytest.mark.asyncio
async def test_file_read_rejects_an_absolute_path_outside(workspace):
    ws, root = workspace
    tool = FileReadTool(workspace_root=str(ws))
    with pytest.raises(ValueError, match="outside workspace"):
        await tool(path=str(root / "secret.env"))


@pytest.mark.asyncio
async def test_file_read_rejects_traversal(workspace):
    ws, _ = workspace
    tool = FileReadTool(workspace_root=str(ws))
    with pytest.raises(ValueError, match="outside workspace"):
        await tool(path="../secret.env")


@pytest.mark.asyncio
async def test_file_read_resolves_relative_against_the_root_not_cwd(workspace):
    ws, _ = workspace
    tool = FileReadTool(workspace_root=str(ws))
    assert "hello workspace" in await tool(path="notes.md")


@pytest.mark.asyncio
async def test_file_read_is_unconfined_without_a_root(workspace):
    """Standalone runs that never set SR2_WORKSPACE keep working."""
    ws, root = workspace
    assert "super-secret" in await FileReadTool()(path=str(root / "secret.env"))


@pytest.mark.asyncio
async def test_read_symbol_rejects_a_path_outside(workspace):
    ws, root = workspace
    tool = ReadSymbolTool(workspace_root=str(ws))
    with pytest.raises(ValueError, match="outside workspace"):
        await tool(file_path=str(root / "secret.env"), symbol_name="TOKEN")


@pytest.mark.asyncio
async def test_grep_defaults_to_the_workspace_root(workspace):
    """'.' must mean the workspace, not wherever the process happens to be."""
    ws, root = workspace
    os.chdir(root)
    out = await GrepTool(workspace_root=str(ws))(pattern="hello", regex=False)
    assert "notes.md" in out
    assert "super-secret" not in out


@pytest.mark.asyncio
async def test_grep_does_not_reach_outside_via_a_path_argument(workspace):
    ws, root = workspace
    tool = GrepTool(workspace_root=str(ws))
    with pytest.raises(ValueError, match="outside workspace"):
        await tool(pattern="TOKEN", path=str(root), regex=False)


@pytest.mark.asyncio
async def test_glob_defaults_to_the_workspace_root(workspace):
    ws, root = workspace
    os.chdir(root)
    out = await GlobTool(workspace_root=str(ws))(pattern="**/*.gd")
    assert "player.gd" in out


@pytest.mark.asyncio
async def test_glob_pattern_cannot_escape_the_root(workspace):
    """root_dir alone does not stop '../': the results must be filtered too."""
    ws, _ = workspace
    out = await GlobTool(workspace_root=str(ws))(pattern="../*.env")
    assert "secret.env" not in out


# --- the helper -------------------------------------------------------------

def test_floor_off_allows_everything(tmp_path):
    floor = WorkspaceFloor(None)
    assert floor.contains("/etc/passwd")
    floor.check("/etc/passwd")


def test_floor_follows_symlinks_out(tmp_path):
    """A link planted inside must not be a way out."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "outside.txt").write_text("x")
    (ws / "link.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(ValueError, match="outside workspace"):
        WorkspaceFloor(str(ws)).check("link.txt")
