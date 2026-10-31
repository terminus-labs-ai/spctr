"""MCPToolBridge — wraps an mcp.types.Tool + call_tool callable as a ToolRegistry-compatible async callable."""
from __future__ import annotations

import asyncio.coroutines
from typing import Any, Callable


class MCPToolBridge:
    """Adapts an MCP tool definition and its call_tool coroutine into a single callable.

    Attributes:
        name: Tool name from the MCP tool definition.
        description: Tool description; empty string when mcp_tool.description is None.
        input_schema: JSON schema dict from mcp_tool.inputSchema.
    """

    # Make asyncio.iscoroutinefunction(instance) return True so ToolRegistry
    # sets is_async=True when registering a bridge as the fn callable.
    _is_coroutine = asyncio.coroutines._is_coroutine

    def __init__(self, mcp_tool: Any, call_tool: Callable[..., Any]) -> None:
        self.name: str = mcp_tool.name
        self.description: str = mcp_tool.description if mcp_tool.description is not None else ""
        self.input_schema: dict = mcp_tool.inputSchema
        self._call_tool = call_tool

    async def __call__(self, **kwargs: Any) -> str:
        """Invoke the MCP tool and return all text-block content joined with newlines."""
        result = await self._call_tool(self.name, kwargs)
        return "\n".join(
            block.text
            for block in result.content
            if block.type == "text"
        )
