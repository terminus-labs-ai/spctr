"""Tests for workspace confinement (spc-51, FR1-FR4).

Validates that file_write, edit, and terminal tools enforce a workspace root
boundary. Paths outside the root are rejected with ValueError.
"""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.tools.builtins.file_write import FileWriteTool
from sr2_spectre.tools.builtins.edit import EditTool
from sr2_spectre.tools.builtins.terminal import TerminalTool
from sr2_spectre.planning.resolver import PlanResolver
from sr2_spectre.runtime import Runtime
from sr2_spectre.config import SpectreConfig
from sr2.config.models import ResolverConfig


# ---------------------------------------------------------------------------
# Workspace root resolution (FR1)
# ---------------------------------------------------------------------------


class TestWorkspaceRootResolution:
    """FR1: Workspace root from SR2_WORKSPACE env var, fallback cwd."""

    def test_root_from_env_var(self, tmp_path, monkeypatch):
        """SR2_WORKSPACE env var sets the workspace root."""
        workspace = tmp_path / "worktree"
        workspace.mkdir()
        monkeypatch.setenv("SR2_WORKSPACE", str(workspace))

        from sr2_spectre.workspace import resolve_workspace_root
        root = resolve_workspace_root()
        assert str(root) == str(Path(workspace).resolve())

    def test_root_fallback_to_cwd(self, tmp_path, monkeypatch):
        """When SR2_WORKSPACE is unset, fall back to os.getcwd()."""
        monkeypatch.delenv("SR2_WORKSPACE", raising=False)
        monkeypatch.chdir(str(tmp_path))

        from sr2_spectre.workspace import resolve_workspace_root
        root = resolve_workspace_root()
        assert str(root) == str(tmp_path.resolve())

    def test_root_is_canonicalized(self, tmp_path, monkeypatch):
        """The resolved root is absolute and canonicalized (realpath)."""
        workspace = tmp_path / "real"
        workspace.mkdir()
        link = tmp_path / "link"
        link.symlink_to(workspace)
        monkeypatch.setenv("SR2_WORKSPACE", str(link))

        from sr2_spectre.workspace import resolve_workspace_root
        root = resolve_workspace_root()
        assert str(root) == str(workspace.resolve())


# ---------------------------------------------------------------------------
# FileWriteTool workspace floor (FR3)
# ---------------------------------------------------------------------------


class TestFileWriteWorkspaceFloor:
    """FR3: file_write rejects paths outside workspace root."""

    def _make_tool(self, workspace: Path) -> FileWriteTool:
        return FileWriteTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_accept_relative_path_inside_root(self, tmp_path):
        """Relative path writes land under workspace root."""
        tool = self._make_tool(tmp_path)
        await tool(path="subdir/file.txt", content="hello")
        assert (tmp_path / "subdir" / "file.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_accept_absolute_path_inside_root(self, tmp_path):
        """Absolute path inside the root is accepted."""
        target = tmp_path / "allowed.txt"
        tool = self._make_tool(tmp_path)
        await tool(path=str(target), content="inside")
        assert target.read_text() == "inside"

    @pytest.mark.asyncio
    async def test_reject_absolute_path_outside_root(self, tmp_path):
        """Absolute path outside the root raises ValueError."""
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "leak.txt"

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        tool = self._make_tool(workspace)

        with pytest.raises(ValueError, match="outside workspace"):
            await tool(path=str(target), content="leaked")

        # File must NOT have been created
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_reject_dotdot_traversal(self, tmp_path):
        """Path with .. escaping the root is rejected."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        subdir = workspace / "sub"
        subdir.mkdir()

        tool = self._make_tool(workspace)

        # Try to escape via ..
        escape_path = str(subdir / ".." / ".." / "secret.txt")
        with pytest.raises(ValueError, match="outside workspace"):
            await tool(path=escape_path, content="escaped")

        # File must NOT exist outside workspace
        assert not (tmp_path / "secret.txt").exists()

    @pytest.mark.asyncio
    async def test_no_workspace_root_is_permissive(self, tmp_path):
        """When workspace_root is None (unset), tool is permissive (back-compat)."""
        tool = FileWriteTool()  # no workspace_root
        target = tmp_path / "free.txt"
        await tool(path=str(target), content="free write")
        assert target.read_text() == "free write"

    @pytest.mark.asyncio
    async def test_error_message_names_path_and_root(self, tmp_path):
        """ValueError message includes the offending path and the workspace root."""
        outside = tmp_path / "forbidden"
        outside.mkdir()
        workspace = tmp_path / "ws"
        workspace.mkdir()

        tool = self._make_tool(workspace)
        with pytest.raises(ValueError) as exc_info:
            await tool(path=str(outside / "x.txt"), content="nope")

        msg = str(exc_info.value)
        assert str(outside) in msg or "forbidden" in msg
        assert str(workspace) in msg or "ws" in msg


# ---------------------------------------------------------------------------
# EditTool workspace floor (FR3)
# ---------------------------------------------------------------------------


class TestEditWorkspaceFloor:
    """FR3: edit rejects paths outside workspace root."""

    def _make_tool(self, workspace: Path) -> EditTool:
        return EditTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_accept_path_inside_root(self, tmp_path):
        """Edit on a file inside the root succeeds."""
        target = tmp_path / "file.txt"
        target.write_text("hello world")
        tool = self._make_tool(tmp_path)
        await tool(path=str(target), old_string="world", new_string="spectre")
        assert target.read_text() == "hello spectre"

    @pytest.mark.asyncio
    async def test_reject_absolute_path_outside_root(self, tmp_path):
        """Edit on a file outside the root raises ValueError."""
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "file.txt"
        target.write_text("should not edit")

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        tool = self._make_tool(workspace)

        with pytest.raises(ValueError, match="outside workspace"):
            await tool(path=str(target), old_string="not", new_string="NOT")

        # File must NOT have been modified
        assert target.read_text() == "should not edit"

    @pytest.mark.asyncio
    async def test_reject_dotdot_traversal(self, tmp_path):
        """Edit with .. escaping root is rejected."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")

        tool = self._make_tool(workspace)
        escape_path = str(workspace / ".." / "outside.txt")
        with pytest.raises(ValueError, match="outside workspace"):
            await tool(path=escape_path, old_string="secret", new_string="leaked")

        assert outside.read_text() == "secret"

    @pytest.mark.asyncio
    async def test_no_workspace_root_is_permissive(self, tmp_path):
        """When workspace_root is None, edit is permissive (back-compat)."""
        target = tmp_path / "free.txt"
        target.write_text("before")
        tool = EditTool()  # no workspace_root
        await tool(path=str(target), old_string="before", new_string="after")
        assert target.read_text() == "after"


# ---------------------------------------------------------------------------
# TerminalTool workspace floor (FR4)
# ---------------------------------------------------------------------------


class TestTerminalWorkspaceFloor:
    """FR4: terminal runs with cwd = workspace root."""

    def _make_tool(self, workspace: Path) -> TerminalTool:
        return TerminalTool(workspace_root=str(workspace))

    @pytest.mark.asyncio
    async def test_cwd_is_workspace_root(self, tmp_path):
        """Terminal commands run with cwd set to the workspace root."""
        workspace = tmp_path / "my_workspace"
        workspace.mkdir()

        tool = self._make_tool(workspace)
        result = await tool(command="pwd")
        assert str(workspace.resolve()) in result

    @pytest.mark.asyncio
    async def test_no_workspace_root_uses_inherited_cwd(self, tmp_path):
        """When workspace_root is None, cwd is not forced (back-compat)."""
        tool = TerminalTool()  # no workspace_root
        result = await tool(command="pwd")
        # Should be the process cwd, not constrained
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_cwd_passed_to_subprocess(self, tmp_path):
        """Verify cwd is explicitly passed to create_subprocess_shell."""
        workspace = tmp_path / "ws"
        workspace.mkdir()

        with patch("asyncio.create_subprocess_shell", new=AsyncMock()) as mock_shell:
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b"/tmp/ws\n", b""))
            proc.returncode = 0
            mock_shell.return_value = proc

            tool = self._make_tool(workspace)
            await tool(command="pwd")

            # Verify cwd was passed
            call_kwargs = mock_shell.call_args[1]
            assert call_kwargs.get("cwd") == str(workspace)


# ---------------------------------------------------------------------------
# Resolver regression (FR2)
# ---------------------------------------------------------------------------


class TestResolverWorktreeRegression:
    """FR2: SR2_PROJECT env var overrides cwd walk in PlanResolver."""

    @pytest.mark.asyncio
    async def test_sr2_project_env_overrides_worktree_name(self, tmp_path, monkeypatch):
        """When cwd is a worktree named 'spc-47', SR2_PROJECT=sr2-spectre
        makes _resolve_project() return 'sr2-spectre'."""
        # Create a fake worktree directory named "spc-47"
        worktree = tmp_path / "spc-47"
        worktree.mkdir()
        # Worktrees have .git as a file, not a directory
        (worktree / ".git").write_text("gitdir: /tmp/harbinger-worktrees/spc-47/.git")

        plans_dir = worktree / "plans"
        plans_dir.mkdir()

        # Create knowledge files flat under the knowledge root.
        # The resolver globs knowledge_root/*.md and filters by project field.
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        (knowledge_dir / "arch.md").write_text(
            "---\nkind: project-knowledge\nproject: sr2-spectre\n---\n\nSpectre knowledge.\n"
        )
        (knowledge_dir / "noise.md").write_text(
            "---\nkind: project-knowledge\nproject: spc-47\n---\n\nNoise.\n"
        )

        monkeypatch.setenv("SR2_PROJECT", "sr2-spectre")
        monkeypatch.chdir(str(worktree))

        resolver = PlanResolver(
            ResolverConfig(
                type="plan",
                config={
                    "project": "__auto__",
                    "plans_root": str(plans_dir),
                    "knowledge_root": str(knowledge_dir),
                },
            )
        )

        # The resolver should derive "sr2-spectre" from SR2_PROJECT,
        # NOT "spc-47" from the cwd walk
        project = resolver._resolve_project()
        assert project == "sr2-spectre"

        # Resolve to verify L1 picks up the right knowledge
        from sr2.pipeline.events import Event, EventPhase
        event = Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")
        result = await resolver.resolve([event])
        text = result.content[0].text
        assert "Spectre knowledge." in text
        assert "Noise." not in text

    @pytest.mark.asyncio
    async def test_worktree_without_sr2_project_returns_worktree_name(self, tmp_path, monkeypatch):
        """Without SR2_PROJECT, cwd walk on worktree returns worktree dir name (baseline)."""
        worktree = tmp_path / "spc-99"
        worktree.mkdir()
        (worktree / ".git").write_text("gitdir: /tmp/harbinger-worktrees/spc-99/.git")

        monkeypatch.delenv("SR2_PROJECT", raising=False)
        monkeypatch.chdir(str(worktree))

        resolver = PlanResolver(
            ResolverConfig(type="plan", config={"project": "__auto__"})
        )
        project = resolver._resolve_project()
        # Without SR2_PROJECT, it falls back to cwd walk → worktree dir name
        assert project == "spc-99"


# ---------------------------------------------------------------------------
# Runtime wiring (FR1 integration)
# ---------------------------------------------------------------------------


class TestRuntimeWorkspaceWiring:
    """Runtime injects workspace_root into tool configs when SR2_WORKSPACE is set."""

    def _make_minimal_config(self, tools: list[dict] | None = None) -> dict:
        tools = tools or []
        return {
            "agent": {
                "name": "test",
                "tools": tools,
                "skills": [],
                "skills_dirs": [],
                "mcp_servers": [],
            },
            "models": {
                "default": {
                    "model": "openai/fake",
                    "base_url": "http://localhost:11434/v1",
                }
            },
            "pipeline": {
                "token_budget": 200000,
                "max_tool_iterations": 5,
                "layers": [],
            },
        }

    def _build_runtime(self, config_dict: dict) -> "Runtime":
        """Build a Runtime from a raw config dict."""
        spectre_config = SpectreConfig(**config_dict)
        return Runtime(spectre_config)

    def test_runtime_sets_workspace_root_from_env(self, tmp_path, monkeypatch):
        """Runtime resolves SR2_WORKSPACE and stores it."""
        workspace = tmp_path / "worktree"
        workspace.mkdir()
        monkeypatch.setenv("SR2_WORKSPACE", str(workspace))

        config_dict = self._make_minimal_config()
        runtime = self._build_runtime(config_dict)

        assert runtime.workspace_root == str(workspace.resolve())

    def test_runtime_workspace_root_none_when_env_unset(self, tmp_path, monkeypatch):
        """Runtime workspace_root is None when SR2_WORKSPACE is not set."""
        monkeypatch.delenv("SR2_WORKSPACE", raising=False)

        config_dict = self._make_minimal_config()
        runtime = self._build_runtime(config_dict)

        assert runtime.workspace_root is None

    def test_runtime_injects_workspace_root_into_tool_config(
        self, tmp_path, monkeypatch
    ):
        """Runtime passes workspace_root to FileWriteTool via config."""
        workspace = tmp_path / "worktree"
        workspace.mkdir()
        monkeypatch.setenv("SR2_WORKSPACE", str(workspace))

        config_dict = self._make_minimal_config(
            tools=[
                {
                    "name": "file_write",
                    "class_path": "sr2_spectre.tools.builtins.file_write.FileWriteTool",
                }
            ]
        )
        runtime = self._build_runtime(config_dict)

        # The tool should have been constructed with the workspace root
        spec = runtime.registry._tools["file_write"]
        # We can verify by checking the registered function's __self__
        tool_instance = spec.fn.__self__
        assert tool_instance.workspace_root == workspace.resolve()

    def test_runtime_does_not_override_explicit_workspace_root(
        self, tmp_path, monkeypatch
    ):
        """If tool config already has workspace_root, Runtime doesn't override."""
        env_workspace = tmp_path / "env_ws"
        env_workspace.mkdir()
        explicit_workspace = tmp_path / "explicit_ws"
        explicit_workspace.mkdir()
        monkeypatch.setenv("SR2_WORKSPACE", str(env_workspace))

        config_dict = self._make_minimal_config(
            tools=[
                {
                    "name": "file_write",
                    "class_path": "sr2_spectre.tools.builtins.file_write.FileWriteTool",
                    "config": {"workspace_root": str(explicit_workspace)},
                }
            ]
        )
        runtime = self._build_runtime(config_dict)

        spec = runtime.registry._tools["file_write"]
        tool_instance = spec.fn.__self__
        assert tool_instance.workspace_root == explicit_workspace.resolve()
