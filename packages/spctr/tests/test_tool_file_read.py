"""Tests for FileReadTool."""
import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_file_read_class_attributes() -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    assert isinstance(FileReadTool.name, str) and FileReadTool.name
    assert isinstance(FileReadTool.description, str) and FileReadTool.description
    assert isinstance(FileReadTool.input_schema, dict)


def test_file_read_input_schema_requires_path() -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    schema = FileReadTool.input_schema
    assert "path" in schema.get("properties", {})
    assert "path" in schema.get("required", [])


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_file_read_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path("sr2_spectre.tools.builtins.file_read.FileReadTool")
    assert "file_read" in reg


# ---------------------------------------------------------------------------
# Successful reads
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_read_returns_file_contents(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    target = tmp_path / "hello.txt"
    target.write_text("hello spectre", encoding="utf-8")

    tool = FileReadTool()
    result = await tool(path=str(target))

    assert result == "hello spectre"


@pytest.mark.asyncio
async def test_file_read_multiline_file(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    content = "line one\nline two\nline three\n"
    target = tmp_path / "multi.txt"
    target.write_text(content, encoding="utf-8")

    tool = FileReadTool()
    result = await tool(path=str(target))

    assert result == content


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_read_missing_file_raises_file_not_found(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    tool = FileReadTool()
    with pytest.raises(FileNotFoundError):
        await tool(path=str(tmp_path / "nonexistent.txt"))


# ---------------------------------------------------------------------------
# Size limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_read_exceeds_max_bytes_raises_value_error(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    target = tmp_path / "big.txt"
    target.write_bytes(b"x" * 100)

    tool = FileReadTool(max_bytes=50)
    with pytest.raises(ValueError):
        await tool(path=str(target))


@pytest.mark.asyncio
async def test_file_read_size_error_message_contains_path(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    target = tmp_path / "oversized.txt"
    target.write_bytes(b"y" * 200)

    tool = FileReadTool(max_bytes=10)
    with pytest.raises(ValueError) as exc_info:
        await tool(path=str(target))

    assert str(target) in str(exc_info.value)


@pytest.mark.asyncio
async def test_file_read_size_error_message_contains_size(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    target = tmp_path / "oversized.txt"
    target.write_bytes(b"z" * 200)

    tool = FileReadTool(max_bytes=10)
    with pytest.raises(ValueError) as exc_info:
        await tool(path=str(target))

    # The error should mention the actual size (200)
    assert "200" in str(exc_info.value)


@pytest.mark.asyncio
async def test_file_read_exactly_at_limit_succeeds(tmp_path) -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    target = tmp_path / "exact.txt"
    target.write_text("a" * 50, encoding="utf-8")

    tool = FileReadTool(max_bytes=50)
    result = await tool(path=str(target))

    assert result == "a" * 50


# ---------------------------------------------------------------------------
# Custom max_bytes constructor arg
# ---------------------------------------------------------------------------

def test_file_read_custom_max_bytes_stored() -> None:
    from sr2_spectre.tools.builtins.file_read import FileReadTool

    tool = FileReadTool(max_bytes=512)
    assert tool.max_bytes == 512
