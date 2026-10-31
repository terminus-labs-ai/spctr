"""Interface system — the core protocol for spectre I/O channels.

An Interface handles input, drives the agent loop, and renders output.
Built-in implementations: SingleShotInterface, TUIInterface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent

__all__ = ["Interface"]


@runtime_checkable
class Interface(Protocol):
    """Core interface protocol for spectre I/O channels.

    An Interface manages its own lifecycle (start/stop) and drives a
    user-interactive loop via ``run()`` that receives input, routes it
    to the agent, and renders the response.
    """
    name: str

    async def start(self, agent: "Agent") -> None: ...
    async def stop(self) -> None: ...
    async def run(self, agent: "Agent") -> None: ...
