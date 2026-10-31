"""Tests for ReadSymbolTool — grounding tool for reading full type definitions."""
import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_read_symbol_class_attributes() -> None:
    from sr2_spectre.tools.builtins.read_symbol import ReadSymbolTool

    assert isinstance(ReadSymbolTool.name, str) and ReadSymbolTool.name
    assert isinstance(ReadSymbolTool.description, str) and ReadSymbolTool.description
    assert isinstance(ReadSymbolTool.input_schema, dict)


def test_read_symbol_name_is_read_symbol() -> None:
    from sr2_spectre.tools.builtins.read_symbol import ReadSymbolTool

    assert ReadSymbolTool.name == "read_symbol"


def test_read_symbol_input_schema_properties() -> None:
    from sr2_spectre.tools.builtins.read_symbol import ReadSymbolTool

    props = ReadSymbolTool.input_schema.get("properties", {})
    assert "file_path" in props
    assert "symbol_name" in props
    assert "context_lines" in props


def test_read_symbol_input_schema_required() -> None:
    from sr2_spectre.tools.builtins.read_symbol import ReadSymbolTool

    required = ReadSymbolTool.input_schema.get("required", [])
    assert "file_path" in required
    assert "symbol_name" in required
    assert "context_lines" not in required


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_read_symbol_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path(
        "sr2_spectre.tools.builtins.read_symbol.ReadSymbolTool"
    )
    assert "read_symbol" in reg


# ---------------------------------------------------------------------------
# find_symbol() — core logic (engine-independent, pure Python)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pyfile(tmp_path):
    """Create a Python file with a class and a function for testing."""
    pyfile = tmp_path / "models.py"
    pyfile.write_text(
        '''"""Sample module for testing symbol lookup."""
from pydantic import BaseModel, Field


class SimpleConfig(BaseModel):
    """A simple config model."""
    name: str
    type: str
    count: int = 0


class NestedConfig(BaseModel):
    """Config with nested structure."""
    outer_name: str
    inner: SimpleConfig
    items: list[str] = Field(default_factory=list)


def helper_func(x: int) -> str:
    """A helper function."""
    return str(x)


class Container:
    """A class with methods."""

    def method_one(self) -> None:
        """First method."""
        pass

    def method_two(self, value: str) -> int:
        """Second method with a parameter."""
        return len(value)


def top_level_function(a: int, b: str = "default") -> dict:
    """A top-level function with multiple params.

    Args:
        a: An integer value.
        b: A string with a default.

    Returns:
        A dict combining the arguments.
    """
    return {"a": a, "b": b}
''',
        encoding="utf-8",
    )
    return pyfile


def test_find_class_returns_full_body(sample_pyfile) -> None:
    """Finding a class returns the complete class body with all fields."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(sample_pyfile), "SimpleConfig")

    assert info.name == "SimpleConfig"
    assert info.kind == "class"
    assert "name: str" in info.body
    assert "type: str" in info.body
    assert "count: int" in info.body


def test_find_class_returns_all_fields_not_just_one(sample_pyfile) -> None:
    """This is the grounding fix: the FULL field list, not a grep hit.

    The original problem (bead obsidian-ye0) was that grep returned only the
    line matching a field name, missing other required fields. This test
    validates that read_symbol returns ALL fields.
    """
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(sample_pyfile), "NestedConfig")

    # All three fields must be present in the result
    assert "outer_name: str" in info.body
    assert "inner: SimpleConfig" in info.body
    assert "items: list[str]" in info.body


def test_find_function_returns_signature_and_body(sample_pyfile) -> None:
    """Finding a function returns the signature, docstring, and body."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(sample_pyfile), "helper_func")

    assert info.name == "helper_func"
    assert info.kind == "function"
    assert "def helper_func" in info.body
    assert "A helper function" in info.body


def test_find_method_returns_method_body(sample_pyfile) -> None:
    """Finding a method returns the method signature and body."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(sample_pyfile), "method_two")

    assert info.name == "method_two"
    assert info.kind == "method"
    assert "def method_two" in info.body
    assert "value: str" in info.body


def test_find_nonexistent_symbol_raises(sample_pyfile) -> None:
    """Searching for a symbol that doesn't exist raises ValueError."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    with pytest.raises(ValueError, match="not found"):
        find_symbol(str(sample_pyfile), "DoesNotExist")


def test_find_nonexistent_file_raises() -> None:
    """Searching a nonexistent file raises FileNotFoundError."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    with pytest.raises(FileNotFoundError):
        find_symbol("/nonexistent/path/file.py", "SomeClass")


def test_find_top_level_function(sample_pyfile) -> None:
    """Finding a function with a multi-line docstring returns the full body."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(sample_pyfile), "top_level_function")

    assert info.kind == "function"
    assert "a: int" in info.body
    assert "b: str" in info.body
    assert "Returns:" in info.body  # docstring content


def test_find_class_line_numbers(sample_pyfile) -> None:
    """Line numbers are 1-based and cover the full definition."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(sample_pyfile), "SimpleConfig")

    assert info.start_line > 0
    assert info.end_line >= info.start_line


def test_find_class_file_path_preserved(sample_pyfile) -> None:
    """The file path from the SymbolInfo matches the input path."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(sample_pyfile), "SimpleConfig")

    assert str(sample_pyfile) in info.file_path


# ---------------------------------------------------------------------------
# Tool async interface
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_symbol_tool_returns_structured_output(sample_pyfile) -> None:
    """The async tool wrapper returns a formatted string with metadata."""
    from sr2_spectre.tools.builtins.read_symbol import ReadSymbolTool

    tool = ReadSymbolTool()
    result = await tool(file_path=str(sample_pyfile), symbol_name="SimpleConfig")

    # Header metadata
    assert "SimpleConfig" in result
    assert "class" in result
    # Definition content
    assert "name: str" in result
    assert "type: str" in result
    assert "count: int" in result
    # Delimiters
    assert "---" in result


@pytest.mark.asyncio
async def test_read_symbol_tool_nonexistent_raises(sample_pyfile) -> None:
    """The tool raises FileNotFoundError for bad file paths."""
    from sr2_spectre.tools.builtins.read_symbol import ReadSymbolTool

    tool = ReadSymbolTool()
    with pytest.raises(FileNotFoundError):
        await tool(file_path="/no/such/file.py", symbol_name="X")


@pytest.mark.asyncio
async def test_read_symbol_tool_nonexistent_symbol_raises(sample_pyfile) -> None:
    """The tool raises ValueError for symbols not in the file."""
    from sr2_spectre.tools.builtins.read_symbol import ReadSymbolTool

    tool = ReadSymbolTool()
    with pytest.raises(ValueError):
        await tool(file_path=str(sample_pyfile), symbol_name="Nope")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.fixture
def edge_case_file(tmp_path):
    """Create a file with tricky formatting."""
    pyfile = tmp_path / "edge.py"
    pyfile.write_text(
        '''class MyClass:
    """Class with single-line body."""
    x: int = 1


class OtherClass:
    pass


# This is a comment about MyClass
class SimilarName(MyClass):
    """A subclass."""
    y: str = "hello"
''',
        encoding="utf-8",
    )
    return pyfile


def test_find_class_does_not_confuse_similar_names(edge_case_file) -> None:
    """Finding 'MyClass' returns MyClass, not SimilarName which also mentions it."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(edge_case_file), "MyClass")

    assert info.name == "MyClass"
    assert "x: int" in info.body
    # Should NOT contain the body of SimilarName
    assert "y: str" not in info.body


def test_find_class_single_line_body(edge_case_file) -> None:
    """A class with a single-field body is captured completely."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(edge_case_file), "OtherClass")

    assert info.name == "OtherClass"
    assert "pass" in info.body


def test_find_subclass(edge_case_file) -> None:
    """Finding a subclass returns its own definition, not the parent's."""
    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(edge_case_file), "SimilarName")

    assert info.name == "SimilarName"
    assert "y: str" in info.body
    # Should not contain the parent's field
    assert "x: int" not in info.body


# ---------------------------------------------------------------------------
# Grounding scenario: the original problem (bead obsidian-ye0)
# ---------------------------------------------------------------------------

def test_grounding_scenario_pydantic_model_fields(tmp_path) -> None:
    """Reproduce the original grounding problem: constructing McpServerConfig.

    The agent grepped for 'mcp_servers' and only saw the field line, not the
    full McpServerConfig definition. With read_symbol, it gets ALL required
    fields in one call.
    """
    # Simulate a config file like the real one
    config_file = tmp_path / "config.py"
    config_file.write_text(
        '''from pydantic import BaseModel, Field


class McpServerConfig(BaseModel):
    """Configuration for a single MCP server connection."""
    name: str
    type: str                     # "stdio" or "http"
    command: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
''',
        encoding="utf-8",
    )

    from sr2_spectre.tools.builtins.read_symbol import find_symbol

    info = find_symbol(str(config_file), "McpServerConfig")

    # The critical test: ALL required fields are present
    assert "name: str" in info.body
    assert "type: str" in info.body
    assert "command:" in info.body
    assert "args:" in info.body
    assert "env:" in info.body
    assert "url:" in info.body

    # Verify it's the complete class body
    assert info.kind == "class"
    assert info.start_line == 4  # 'class McpServerConfig' line


def test_grounding_scenario_returns_more_than_grep_would(tmp_path) -> None:
    """read_symbol returns strictly more context than a single grep hit.

    Grep for 'mcp_servers' in config.py returns one line. read_symbol for
    'McpServerConfig' returns the entire class definition with all 6 fields.
    """
    config_file = tmp_path / "config.py"
    config_file.write_text(
        '''from pydantic import BaseModel, Field


class McpServerConfig(BaseModel):
    name: str
    type: str
    command: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""


class AgentConfig(BaseModel):
    name: str = "spectre"
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
''',
        encoding="utf-8",
    )

    from sr2_spectre.tools.builtins.read_symbol import find_symbol
    from sr2_spectre.tools.builtins.grep import _grep

    # What grep would return for 'name' in McpServerConfig:
    grep_result = _grep(
        "name",
        str(config_file),
        None,  # no glob
        False,  # literal
        500,
        100,
        set(),
    )

    # What read_symbol returns for the full type:
    symbol_info = find_symbol(str(config_file), "McpServerConfig")

    # The symbol result should contain MORE field lines than the grep result
    # (grep finds 'name' in both classes; read_symbol gives the full
    # McpServerConfig definition)
    grep_line_count = len(grep_result.strip().split("\n"))
    symbol_line_count = len(symbol_info.body.split("\n"))

    assert symbol_line_count > grep_line_count, (
        f"read_symbol should return more context ({symbol_line_count} lines) "
        f"than grep ({grep_line_count} lines)"
    )

    # And the symbol result should contain fields that grep for 'name' wouldn't
    assert "type: str" in symbol_info.body
    assert "command:" in symbol_info.body
