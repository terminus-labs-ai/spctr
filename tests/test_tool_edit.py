"""Tests for EditTool."""
import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_edit_class_attributes() -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    assert isinstance(EditTool.name, str) and EditTool.name
    assert isinstance(EditTool.description, str) and EditTool.description
    assert isinstance(EditTool.input_schema, dict)


def test_edit_name_is_edit() -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    assert EditTool.name == "edit"


def test_edit_input_schema_properties() -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    props = EditTool.input_schema.get("properties", {})
    assert "path" in props
    assert "old_string" in props
    assert "new_string" in props
    assert "replace_all" in props


def test_edit_input_schema_required() -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    required = EditTool.input_schema.get("required", [])
    assert "path" in required
    assert "old_string" in required
    assert "new_string" in required
    # replace_all is optional and must NOT be required
    assert "replace_all" not in required


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_edit_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path("sr2_spectre.tools.builtins.edit.EditTool")
    assert "edit" in reg


# ---------------------------------------------------------------------------
# Successful unique-match replacement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_replaces_unique_match(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("hello world", encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="world", new_string="spectre")

    assert target.read_text(encoding="utf-8") == "hello spectre"


@pytest.mark.asyncio
async def test_edit_returns_confirmation_string(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("alpha beta gamma", encoding="utf-8")

    tool = EditTool()
    result = await tool(path=str(target), old_string="beta", new_string="BETA")

    assert isinstance(result, str) and result


@pytest.mark.asyncio
async def test_edit_preserves_full_file_content(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    original = "line one\nNEEDLE here\nline three\ntrailing\n"
    target = tmp_path / "file.txt"
    target.write_text(original, encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="NEEDLE", new_string="REPLACED")

    expected = "line one\nREPLACED here\nline three\ntrailing\n"
    assert target.read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# Zero-match rule
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_zero_match_raises_value_error(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("hello world", encoding="utf-8")

    tool = EditTool()
    with pytest.raises(ValueError):
        await tool(path=str(target), old_string="missing", new_string="x")


@pytest.mark.asyncio
async def test_edit_zero_match_does_not_modify_file(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    original = "untouched content\n"
    target = tmp_path / "file.txt"
    target.write_text(original, encoding="utf-8")

    tool = EditTool()
    with pytest.raises(ValueError):
        await tool(path=str(target), old_string="nope", new_string="x")

    assert target.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Ambiguous multi-match rule (replace_all=False)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_multi_match_raises_value_error(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("dup dup dup", encoding="utf-8")

    tool = EditTool()
    with pytest.raises(ValueError):
        await tool(path=str(target), old_string="dup", new_string="x")


@pytest.mark.asyncio
async def test_edit_multi_match_does_not_modify_file(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    original = "dup dup dup"
    target = tmp_path / "file.txt"
    target.write_text(original, encoding="utf-8")

    tool = EditTool()
    with pytest.raises(ValueError):
        await tool(path=str(target), old_string="dup", new_string="x")

    assert target.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# replace_all=True
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_replace_all_replaces_every_occurrence(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("dup dup dup", encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="dup", new_string="x", replace_all=True)

    assert target.read_text(encoding="utf-8") == "x x x"


@pytest.mark.asyncio
async def test_edit_replace_all_returns_successfully_with_multiple_matches(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("a-a-a-a", encoding="utf-8")

    tool = EditTool()
    result = await tool(path=str(target), old_string="a", new_string="b", replace_all=True)

    assert isinstance(result, str) and result
    assert target.read_text(encoding="utf-8") == "b-b-b-b"


@pytest.mark.asyncio
async def test_edit_replace_all_single_match(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("only one here", encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="one", new_string="1", replace_all=True)

    assert target.read_text(encoding="utf-8") == "only 1 here"


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_missing_file_raises_file_not_found(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    tool = EditTool()
    with pytest.raises(FileNotFoundError):
        await tool(
            path=str(tmp_path / "nonexistent.txt"),
            old_string="a",
            new_string="b",
        )


# ---------------------------------------------------------------------------
# Byte-for-byte preservation edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_old_string_at_start_preserves_rest(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    original = "START middle end\n"
    target = tmp_path / "file.txt"
    target.write_text(original, encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="START", new_string="BEGIN")

    assert target.read_text(encoding="utf-8") == "BEGIN middle end\n"


@pytest.mark.asyncio
async def test_edit_old_string_at_end_preserves_rest(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    original = "begin middle FINISH"
    target = tmp_path / "file.txt"
    target.write_text(original, encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="FINISH", new_string="DONE")

    assert target.read_text(encoding="utf-8") == "begin middle DONE"


@pytest.mark.asyncio
async def test_edit_preserves_surrounding_whitespace_and_newlines(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    original = "\n\n  indented TARGET line  \n\n\ttabbed\n"
    target = tmp_path / "file.txt"
    target.write_text(original, encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="TARGET", new_string="X")

    expected = "\n\n  indented X line  \n\n\ttabbed\n"
    assert target.read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# Edge cases: new_string containing old_string, empty new_string
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_new_string_contains_old_string_no_infinite_loop(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("foo", encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="foo", new_string="foofoo")

    # Single replacement only; no recursion into the freshly-written text.
    assert target.read_text(encoding="utf-8") == "foofoo"


@pytest.mark.asyncio
async def test_edit_replace_all_new_string_contains_old_string(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("ab ab", encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="ab", new_string="abab", replace_all=True)

    # Each original occurrence replaced exactly once; no infinite loop.
    assert target.read_text(encoding="utf-8") == "abab abab"


@pytest.mark.asyncio
async def test_edit_empty_new_string_deletes_match(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    target = tmp_path / "file.txt"
    target.write_text("keep DELETEME keep", encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="DELETEME ", new_string="")

    assert target.read_text(encoding="utf-8") == "keep keep"


@pytest.mark.asyncio
async def test_edit_same_old_and_new_string_leaves_rest_untouched(tmp_path) -> None:
    from sr2_spectre.tools.builtins.edit import EditTool

    original = "prefix SAME suffix\nsecond line\n"
    target = tmp_path / "file.txt"
    target.write_text(original, encoding="utf-8")

    tool = EditTool()
    await tool(path=str(target), old_string="SAME", new_string="SAME")

    # Replacing a unique match with itself must not corrupt the rest of the file.
    assert target.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Relative path resolution (workspace root vs cwd)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_resolves_relative_path_against_workspace_root(tmp_path) -> None:
    """When workspace_root is set, relative paths resolve against it, not cwd."""
    from sr2_spectre.tools.builtins.edit import EditTool

    # Workspace root is tmp_path; file lives inside a subdirectory.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subdir = workspace / "src"
    subdir.mkdir()
    target = subdir / "file.txt"
    target.write_text("hello world", encoding="utf-8")

    tool = EditTool(workspace_root=str(workspace))

    # Call with relative path from workspace root, not cwd.
    result = await tool(path="src/file.txt", old_string="world", new_string="spectre")

    assert target.read_text(encoding="utf-8") == "hello spectre"
    assert isinstance(result, str) and result


@pytest.mark.asyncio
async def test_edit_relative_path_fails_when_cwd_not_workspace(tmp_path) -> None:
    """Relative paths resolve against workspace_root even when cwd is elsewhere."""
    from sr2_spectre.tools.builtins.edit import EditTool
    import os

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subdir = workspace / "src"
    subdir.mkdir()
    target = subdir / "file.txt"
    target.write_text("hello world", encoding="utf-8")

    # Create a separate directory and set it as cwd (simulates cwd != workspace)
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    original_cwd = os.getcwd()
    try:
        os.chdir(str(other_dir))

        tool = EditTool(workspace_root=str(workspace))
        # This SHOULD work — relative path resolves against workspace_root, not cwd
        result = await tool(path="src/file.txt", old_string="world", new_string="spectre")

        assert target.read_text(encoding="utf-8") == "hello spectre"
    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_edit_resolve_path_absolute_passthrough(tmp_path) -> None:
    """_resolve_path leaves absolute paths unchanged."""
    from sr2_spectre.tools.builtins.edit import EditTool

    tool = EditTool(workspace_root=str(tmp_path / "workspace"))
    assert tool._resolve_path("/etc/passwd") == "/etc/passwd"


@pytest.mark.asyncio
async def test_edit_resolve_path_no_workspace(tmp_path) -> None:
    """_resolve_path returns raw path when workspace_root is None."""
    from sr2_spectre.tools.builtins.edit import EditTool

    tool = EditTool()
    assert tool._resolve_path("relative/path") == "relative/path"
    assert tool._resolve_path("/absolute/path") == "/absolute/path"
