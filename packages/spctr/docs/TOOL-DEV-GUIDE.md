# SR2 Spectre — Tool Development Guide

How to build custom tools for SR2 Spectre. A tool is a Python class with a name, description, input schema, and a callable that executes it.

## Tool Anatomy

Every tool is a class with four class attributes and one `__call__` method:

```python
class MyTool:
    name = "my_tool"
    description = "What this tool does, written for the LLM to understand."
    input_schema = {
        "type": "object",
        "properties": {
            "arg1": {
                "type": "string",
                "description": "Description of arg1.",
            },
        },
        "required": ["arg1"],
    }

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    async def __call__(self, arg1: str) -> str:
        # Do the work
        return result
```

### Required class attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `name` | `str` | Tool identifier. Must be lowercase, no spaces. Used by the LLM and config. |
| `description` | `str` | What the tool does. **Written for the LLM** — this is how the model decides when to call your tool. |
| `input_schema` | `dict` | JSON Schema describing the tool's parameters. The LLM uses this to construct arguments. |

### The `__call__` method

- Must accept keyword arguments matching your `input_schema` properties.
- Can be sync or async. The registry detects `asyncio.iscoroutinefunction()` automatically.
- **Prefer async** for I/O (network, filesystem, subprocess). Use `run_in_executor` for CPU-bound work.
- Return a string or a value that can be converted to string. The return value is what the LLM sees.

### Constructor

Accept `**kwargs` or typed parameters from the config section:

```python
# Config:
#   tools:
#     - name: my_tool
#       class_path: my_package.tools:MyTool
#       config:
#         timeout: 60
#         api_key: sk-...

def __init__(self, timeout: int = 30, api_key: str = "") -> None:
    self.timeout = timeout
    self.api_key = api_key
```

## Registration

Tools are registered in the config under `agent.tools`:

```yaml
agent:
  tools:
    - name: my_tool
      class_path: my_package.tools:MyTool
      config:
        timeout: 60
```

The `class_path` uses `module.submodule:ClassName` format. The registry:
1. Splits on the last `.` to get module path and class name.
2. Imports the module and gets the class.
3. Instantiates with `config` kwargs.
4. Registers `name`, `description`, `input_schema`, and `__call__`.

## Minimal Example

```python
"""dice_tool.py — roll dice for the agent."""
from __future__ import annotations

import random


class DiceTool:
    """Roll one or more dice and return the results."""

    name = "dice"
    description = "Roll dice. Supports 'NdM' notation (e.g., '2d6' = two six-sided dice). Returns individual rolls and total."
    input_schema = {
        "type": "object",
        "properties": {
            "notation": {
                "type": "string",
                "description": "Dice notation like '2d6', '1d20', '4d10'. Default: '1d6'.",
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        pass

    def __call__(self, notation: str = "1d6") -> str:
        parts = notation.lower().split("d")
        if len(parts) != 2:
            return f"Invalid dice notation: {notation}. Use format NdM (e.g., 2d6)."

        try:
            count, sides = int(parts[0]), int(parts[1])
        except ValueError:
            return f"Invalid dice notation: {notation}. Numbers only."

        if count < 1 or count > 100:
            return f"Count must be 1-100, got {count}."
        if sides < 1 or sides > 1000:
            return f"Sides must be 1-1000, got {sides}."

        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls)
        return f"Rolled {count}d{sides}: [{', '.join(map(str, rolls))}] = {total}"
```

Register in config:

```yaml
agent:
  tools:
    - name: dice
      class_path: my_tools.dice_tool:MyTool
```

## Error Handling

**Never let exceptions propagate.** Wrap errors in the return string:

```python
async def __call__(self, url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    except httpx.TimeoutException:
        return f"Request timed out after {self.timeout}s: {url}"
    except httpx.HTTPStatusError as e:
        return f"HTTP {e.response.status_code}: {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"
```

The LLM reads the return value. If you raise an exception, the agent framework catches it and reports a generic error. If you return a descriptive string, the LLM can reason about the failure.

## Output Limits

Tool results are truncated at `agent.tool_result_max_bytes` (default: 64KB). Design your tools with this in mind:

- **Paginate** large results.
- **Summarize** rather than returning raw data when possible.
- **Truncate early** in your tool rather than relying on the framework truncation.

Example:

```python
async def __call__(self, path: str) -> str:
    try:
        content = await loop.run_in_executor(None, read_file, path)
    except Exception as e:
        return f"Error reading {path}: {e}"

    if len(content) > 5000:
        return content[:5000] + f"\n\n[... file truncated, {len(content)} total chars]"
    return content
```

## Testing Your Tool

Write unit tests that exercise the tool's logic:

```python
import pytest
from my_tools.dice_tool import DiceTool


def test_dice_default_roll() -> None:
    tool = DiceTool()
    result = tool()
    assert "Rolled 1d6" in result
    assert "=" in result


def test_dice_custom_notation() -> None:
    tool = DiceTool()
    result = tool(notation="2d10")
    assert "Rolled 2d10" in result


def test_dice_invalid_notation() -> None:
    tool = DiceTool()
    result = tool(notation="abc")
    assert "Invalid" in result


@pytest.mark.asyncio
async def test_tool_registered() -> None:
    """Tool can be registered via class_path."""
    from sr2_spectre.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_from_class_path(
        "my_tools.dice_tool:DiceTool",
        config={},
    )
    assert "dice" in registry
```

See `tests/test_tool_terminal.py`, `tests/test_tool_grep.py`, etc., for complete examples.

## JSON Schema Reference

The `input_schema` uses standard JSON Schema. Common patterns:

### Required string parameter

```json
{
  "type": "object",
  "properties": {
    "command": {
      "type": "string",
      "description": "The shell command to execute."
    }
  },
  "required": ["command"]
}
```

### Optional parameters with defaults

```json
{
  "type": "object",
  "properties": {
    "pattern": {
      "type": "string",
      "description": "Regex pattern to search for."
    },
    "path": {
      "type": "string",
      "description": "File or directory to search.",
      "default": "."
    },
    "max_results": {
      "type": "integer",
      "description": "Maximum number of results.",
      "default": 10
    },
    "recursive": {
      "type": "boolean",
      "description": "Search subdirectories.",
      "default": true
    }
  },
  "required": ["pattern"]
}
```

### No required parameters (all optional)

```json
{
  "type": "object",
  "properties": {
    "notation": {
      "type": "string",
      "description": "Dice notation. Default: 1d6."
    }
  },
  "required": []
}
```

## Best Practices

1. **Write descriptions for the LLM, not humans.** The LLM reads `description` and `input_schema` to decide when and how to call your tool. Be explicit about what inputs it needs and what it returns.

2. **Keep tools focused.** One tool, one job. A "read_file" tool is better than a "file_operations" tool that does read/write/move/delete.

3. **Validate inputs early.** Check arguments at the start of `__call__` and return descriptive errors for invalid inputs.

4. **Use async for I/O.** The agent runs tools in a single event loop. Blocking calls freeze the entire agent.

5. **Bound output.** Never return unbounded data. Set explicit limits and truncate with context.

6. **Don't rely on global state.** Tools should be self-contained. If you need shared state (like a session or config), pass it through the constructor.

7. **Test the tool independently.** Test the `__call__` method directly before testing registration. The tool's logic should be testable without the agent framework.

## Reference: Built-in Tools

All built-in tools live in `src/sr2_spectre/tools/builtins/`:

| Tool | File | Async? | Highlights |
|------|------|--------|------------|
| `terminal` | `terminal.py` | Yes | `asyncio.create_subprocess_shell`, timeout |
| `file_read` | `file_read.py` | Yes | `run_in_executor`, max_bytes limit |
| `file_write` | `file_write.py` | Yes | Creates parent dirs |
| `edit` | `edit.py` | Yes | String replacement in files |
| `grep` | `grep.py` | Yes (executor) | Regex/literal, glob filter, binary skip |
| `glob` | `glob.py` | Yes (executor) | File pattern matching |
| `web_search` | `web_search.py` | Yes | SearXNG JSON API |
| `code_exec` | `code_exec.py` | Yes | Isolated Python execution |
| `read_symbol` | `read_symbol.py` | Yes | Extract Python class/function definitions |
| `complete_step` | `complete_step.py` | Yes | Plan step verification |
| `load_skill` | `load_skill.py` | Yes | Load skill files from disk |
| `test_guard` | `test_guard.py` | Yes | Phantom test detection |
