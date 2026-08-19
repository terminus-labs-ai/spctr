"""Tool registry — discover, define, and execute tools."""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    """A registered tool with metadata."""
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    is_async: bool = field(default=False, repr=False)


class ToolRegistry:
    """Register tools and produce definitions for relay."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        fn: Callable[..., Any],
    ) -> None:
        """Register a tool callable."""
        import asyncio
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            fn=fn,
            is_async=asyncio.iscoroutinefunction(fn),
        )

    def register_from_class_path(
        self,
        class_path: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Dynamically load and register a tool class.

        Expects the class to have:
        - name, description, input_schema class attributes
        - __call__ (sync or async)

        Or a factory that returns an instance with those attributes.

        Returns:
            The name the tool registered under, so callers that need to track
            what a given config contributed (and later retire it) can.
        """
        config = config or {}
        module_path, class_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        tool_instance = cls(**config)

        self.register(
            name=tool_instance.name,
            description=tool_instance.description,
            input_schema=tool_instance.input_schema,
            fn=tool_instance.__call__ if hasattr(tool_instance, "__call__") else tool_instance,
        )
        return tool_instance.name

    def unregister(self, name: str) -> bool:
        """Drop a tool by name. Returns True if it was registered.

        Used when a config reload removes a tool that a previous config
        declared: overwriting by name is not enough, the stale entry has to
        leave the registry or the agent keeps being offered a tool its config
        no longer grants it.
        """
        return self._tools.pop(name, None) is not None

    def to_sr2_definitions(self) -> list["ToolDefinition"]:
        """Return tool definitions as SR2 ToolDefinition objects.

        This is the canonical domain shape consumed by SpectreToolProvider.
        """
        from sr2.models import ToolDefinition
        return [
            ToolDefinition(
                name=spec.name,
                description=spec.description,
                input_schema=spec.input_schema,
            )
            for spec in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name with the given arguments."""
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Tool not registered: {name}")

        if spec.is_async:
            return await spec.fn(**arguments)
        else:
            return spec.fn(**arguments)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
