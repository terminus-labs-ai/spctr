"""Tests for FileWriteTool."""
import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_file_write_class_attributes() -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    assert isinstance(FileWriteTool.name, str) and FileWriteTool.name
    assert isinstance(FileWriteTool.description, str) and FileWriteTool.description
    assert isinstance(FileWriteTool.input_schema, dict)


def test_file_write_input_schema_requires_path_and_content() -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    schema = FileWriteTool.input_schema
    props = schema.get("properties", {})
    required = schema.get("required", [])
    assert "path" in props
    assert "content" in props
    assert "path" in required
    assert "content" in required


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_file_write_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path("sr2_spectre.tools.builtins.file_write.FileWriteTool")
    assert "file_write" in reg


# ---------------------------------------------------------------------------
# Successful writes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_write_creates_file_with_content(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    target = tmp_path / "output.txt"
    tool = FileWriteTool()
    await tool(path=str(target), content="hello spectre")

    assert target.read_text(encoding="utf-8") == "hello spectre"


@pytest.mark.asyncio
async def test_file_write_returns_confirmation_string(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    target = tmp_path / "output.txt"
    tool = FileWriteTool()
    result = await tool(path=str(target), content="data")

    assert isinstance(result, str) and result


@pytest.mark.asyncio
async def test_file_write_confirmation_contains_byte_count(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    content = "hello"  # 5 bytes in UTF-8
    target = tmp_path / "output.txt"
    tool = FileWriteTool()
    result = await tool(path=str(target), content=content)

    assert "5" in result


@pytest.mark.asyncio
async def test_file_write_confirmation_contains_path(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    target = tmp_path / "output.txt"
    tool = FileWriteTool()
    result = await tool(path=str(target), content="x")

    assert str(target) in result


# ---------------------------------------------------------------------------
# Parent directory creation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_write_creates_missing_parent_directories(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    target = tmp_path / "a" / "b" / "c" / "file.txt"
    tool = FileWriteTool()
    await tool(path=str(target), content="nested")

    assert target.exists()
    assert target.read_text(encoding="utf-8") == "nested"


# ---------------------------------------------------------------------------
# Overwrite existing file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_write_overwrites_existing_file(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    target = tmp_path / "existing.txt"
    target.write_text("old content", encoding="utf-8")

    tool = FileWriteTool()
    await tool(path=str(target), content="new content")

    assert target.read_text(encoding="utf-8") == "new content"


@pytest.mark.asyncio
async def test_file_write_overwrite_does_not_raise(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    tool = FileWriteTool()
    # Should not raise any exception
    result = await tool(path=str(target), content="replaced")
    assert result  # returns a truthy confirmation string


# ---------------------------------------------------------------------------
# Empty content
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_write_empty_content(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_write import FileWriteTool

    target = tmp_path / "empty.txt"
    tool = FileWriteTool()
    result = await tool(path=str(target), content="")

    assert target.exists()
    assert target.read_text(encoding="utf-8") == ""
    assert isinstance(result, str)
    assert "0" in result or str(target) in result
