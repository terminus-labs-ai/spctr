"""sr2-spectre — SR2 agent runtime.

Spectre owns agent identity, conversation history, the tool registry, interfaces, and sessions. SR2 owns the tool execution loop.
All LLM calls flow through sr2-relay — spectre never imports sr2 core directly.
"""

__version__ = "0.1.0"
