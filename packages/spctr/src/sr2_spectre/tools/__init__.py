"""Tool system — definitions, execution, and registry.

Tools are plain callables wrapped in a consistent interface. Spectre
registers them and passes definitions to relay on each turn.
"""

from sr2_spectre.tools.output import PostExecuteEvent, ToolOutput

__all__ = ["PostExecuteEvent", "ToolOutput"]
