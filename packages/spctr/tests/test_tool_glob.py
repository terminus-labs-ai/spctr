"""Tests for GlobTool."""
import os

import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_glob_class_attributes() -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    assert isinstance(GlobTool.name, str) and GlobTool.name
    assert isinstance(GlobTool.description, str) and GlobTool.description
    assert isinstance(GlobTool.input_schema, dict)


def test_glob_name_is_glob() -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    assert GlobTool.name == "glob"


def test_glob_input_schema_properties() -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    props = GlobTool.input_schema.get("properties", {})
    assert "pattern" in props
    assert "path" in props


def test_glob_input_schema_required() -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    required = GlobTool.input_schema.get("required", [])
    # pattern is the only required input; path is optional.
    assert "pattern" in required
    assert "path" not in required
    assert required == ["pattern"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_glob_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path("sr2_spectre.tools.builtins.glob.GlobTool")
    assert "glob" in reg


# ---------------------------------------------------------------------------
# Single-level "*" match (relative, sorted, filtered by extension)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_glob_single_level_star_match(tmp_path) -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "c.txt").write_text("", encoding="utf-8")

    tool = GlobTool()
    result = await tool(pattern="*.py", path=str(tmp_path))

    assert isinstance(result, str)
    lines = result.splitlines()
    # Exact relative names, sorted ascending; c.txt absent.
    assert lines == ["a.py", "b.py"]


# ---------------------------------------------------------------------------
# "*" does NOT recurse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_glob_single_star_does_not_recurse(tmp_path) -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    (tmp_path / "top.py").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.py").write_text("", encoding="utf-8")

    tool = GlobTool()
    result = await tool(pattern="*.py", path=str(tmp_path))

    assert isinstance(result, str)
    lines = result.splitlines()
    assert "top.py" in lines
    # Single star is non-recursive: the nested file must not appear.
    assert "nested.py" not in lines
    assert os.path.join("sub", "nested.py") not in lines


# ---------------------------------------------------------------------------
# "**" recurses across nested directories
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_glob_double_star_recurses(tmp_path) -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    (tmp_path / "top.py").write_text("", encoding="utf-8")
    nested_dir = tmp_path / "sub" / "deep"
    nested_dir.mkdir(parents=True)
    (nested_dir / "nested.py").write_text("", encoding="utf-8")

    tool = GlobTool()
    result = await tool(pattern="**/*.py", path=str(tmp_path))

    assert isinstance(result, str)
    lines = result.splitlines()
    # Recursive "**/*.py" matches both the top-level and the nested .py file.
    assert "top.py" in lines
    assert os.path.join("sub", "deep", "nested.py") in lines


# ---------------------------------------------------------------------------
# Subdirectory literal pattern
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_glob_subdirectory_literal_pattern(tmp_path) -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.md").write_text("", encoding="utf-8")

    tool = GlobTool()
    result = await tool(pattern="sub/*.md", path=str(tmp_path))

    assert isinstance(result, str)
    lines = result.splitlines()
    assert os.path.join("sub", "x.md") in lines


# ---------------------------------------------------------------------------
# Results are sorted ascending
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_glob_results_are_sorted(tmp_path) -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    names = ["zebra.txt", "alpha.txt", "mango.txt"]
    for name in names:
        (tmp_path / name).write_text("", encoding="utf-8")

    tool = GlobTool()
    result = await tool(pattern="*.txt", path=str(tmp_path))

    assert isinstance(result, str)
    lines = result.splitlines()
    assert lines == sorted(names)


# ---------------------------------------------------------------------------
# Results are relative, not absolute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_glob_results_are_relative_not_absolute(tmp_path) -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    (tmp_path / "a.py").write_text("", encoding="utf-8")

    tool = GlobTool()
    result = await tool(pattern="*.py", path=str(tmp_path))

    assert isinstance(result, str)
    lines = result.splitlines()
    assert lines == ["a.py"]
    for line in lines:
        # Each returned path is relative to the root, not absolute.
        assert not line.startswith(str(tmp_path))
        assert not os.path.isabs(line)


# ---------------------------------------------------------------------------
# Empty result returns a non-error string, does not raise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_glob_empty_result_returns_string_not_raise(tmp_path) -> None:
    from sr2_spectre.tools.builtins.glob import GlobTool

    tool = GlobTool()
    # Should not raise on zero matches.
    result = await tool(pattern="*.nonexistent", path=str(tmp_path))

    assert isinstance(result, str)
    # Non-error string signalling that nothing was found.
    assert "no file" in result.lower()
