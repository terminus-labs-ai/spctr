"""Tool system — definitions, execution, and registry.

Tools are plain callables wrapped in a consistent interface. spctr
registers them and passes definitions to relay on each turn.
"""

from spctr.tools.output import PostExecuteEvent, ToolOutput

__all__ = ["PostExecuteEvent", "ToolOutput"]
