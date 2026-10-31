"""Spectre pipeline providers — registered via entry points into SR2.

SpectreToolProvider:
  - Entry point group: sr2.tool_providers
  - Entry point name:  spectre_tools
  - Reads ToolRegistry from deps.tool_source (typed ToolSource dependency)
  - Returns registry's ToolDefinitions to the SR2 tools compilation layer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sr2.pipeline.events import Event, EventSubscription

if TYPE_CHECKING:
    from sr2.config.models import ToolProviderConfig
    from sr2.models import ToolDefinition
    from sr2.pipeline.dependencies import Dependencies
    from sr2.pipeline.protocols import ToolSource


class SpectreToolProvider:
    """SR2 ToolProvider that surfaces spectre's ToolRegistry into the pipeline.

    Registered as ``spectre_tools`` under the ``sr2.tool_providers`` entry-point
    group. SR2 discovers it lazily on the first call to ``_TOOL_PROVIDERS.get()``.

    Lifecycle:
    - ``build(config, deps)`` — called once at SR2 init; reads ToolRegistry from
      ``deps.tool_source``.
    - ``provide(events)`` — called each turn by the Layer when subscribed events
      arrive; returns the registry's current ToolDefinitions.
    """

    name: str = "spectre_tools"

    def __init__(self) -> None:
        # Populated by build()
        self._registry: ToolSource
        self.subscriptions: list[EventSubscription] = []
        self.max_executions: int = 1
        self.execution_count: int = 0

    @classmethod
    def build(
        cls,
        config: "ToolProviderConfig",
        deps: "Dependencies",
    ) -> "SpectreToolProvider":
        """Construct provider from pipeline config and injected dependencies.

        Raises RuntimeError if deps.tool_source is absent — the caller
        (Agent) is responsible for passing it via SR2(tool_source=...).
        """
        self = cls()
        # Fail fast at build time if the harness didn't inject a tool source.
        if deps.tool_source is None:
            raise RuntimeError(
                "SpectreToolProvider requires deps.tool_source; "
                "pass it via SR2(tool_source=registry)."
            )
        self._registry = deps.tool_source
        self.subscriptions = [EventSubscription(event_name="turn_start")]
        self.max_executions = config.max_executions
        self.execution_count = 0
        return self

    async def provide(self, events: list[Event]) -> list["ToolDefinition"]:
        """Return SR2 ToolDefinitions from the bound ToolRegistry."""
        self.execution_count += 1
        return self._registry.to_sr2_definitions()
