"""Tests for ToolRegistry."""
import pytest
from sr2_spectre.tools.registry import ToolRegistry


def _sync_add(a: int, b: int) -> str:
    return str(a + b)


async def _async_greet(name: str) -> str:
    return f"Hello, {name}!"


def test_register_and_list() -> None:
    reg = ToolRegistry()
    reg.register(
        name="add",
        description="Add two numbers",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        fn=_sync_add,
    )
    assert "add" in reg
    assert len(reg) == 1
    assert "add" in reg.list_names()


@pytest.mark.asyncio
async def test_execute_sync_tool() -> None:
    reg = ToolRegistry()
    reg.register(
        name="add",
        description="Add",
        input_schema={},
        fn=_sync_add,
    )
    result = await reg.execute("add", {"a": 2, "b": 3})
    assert result == "5"


@pytest.mark.asyncio
async def test_execute_async_tool() -> None:
    reg = ToolRegistry()
    reg.register(
        name="greet",
        description="Greet someone",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        fn=_async_greet,
    )
    result = await reg.execute("greet", {"name": "Diego"})
    assert result == "Hello, Diego!"


@pytest.mark.asyncio
async def test_execute_missing_tool() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="nonexistent"):
        await reg.execute("nonexistent", {})


# ---- to_sr2_definitions ----

def test_to_sr2_definitions_empty() -> None:
    from sr2.models import ToolDefinition
    reg = ToolRegistry()
    result = reg.to_sr2_definitions()
    assert result == []


def test_to_sr2_definitions_single() -> None:
    from sr2.models import ToolDefinition
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    reg = ToolRegistry()
    reg.register(name="search", description="Search the web", input_schema=schema, fn=lambda q: q)
    result = reg.to_sr2_definitions()
    assert len(result) == 1
    assert isinstance(result[0], ToolDefinition)
    assert result[0].name == "search"
    assert result[0].description == "Search the web"
    assert result[0].input_schema == schema


def test_to_sr2_definitions_multiple_order() -> None:
    reg = ToolRegistry()
    reg.register(name="alpha", description="A", input_schema={}, fn=lambda: None)
    reg.register(name="beta",  description="B", input_schema={}, fn=lambda: None)
    result = reg.to_sr2_definitions()
    assert len(result) == 2
    assert [d.name for d in result] == ["alpha", "beta"]


def test_to_sr2_definitions_schema_is_lossless() -> None:
    """input_schema dict is passed through unchanged, not re-serialized."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    reg = ToolRegistry()
    reg.register(name="t", description="", input_schema=schema, fn=lambda x: x)
    result = reg.to_sr2_definitions()
    assert result[0].input_schema == schema
