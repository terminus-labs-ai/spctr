"""Single-shot interface — run one turn, print response, exit.

Usage: sr2-spectre config.yaml "What is 2+2?" --interface single_shot
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent

from sr2_spectre.core import RunContext, RunMode


class SingleShotInterface:
    name = "single_shot"

    def __init__(self, prompt: str | None = None) -> None:
        self._prompt = prompt

    async def start(self, agent: "Agent") -> None:
        """Set headless run context."""
        agent.set_run_context(RunContext(
            interface="single_shot",
            mode=RunMode.HEADLESS,
            source=None,
        ))

    async def stop(self) -> None:
        """No shutdown needed."""
        pass

    async def run(self, agent: "Agent") -> None:
        """Run a single turn and exit."""
        prompt = self._prompt
        if prompt is None:
            # Read from command line args or stdin
            if len(sys.argv) > 1:
                prompt = " ".join(sys.argv[1:])
            else:
                print("Enter prompt (Ctrl+D to submit):", file=sys.stderr)
                prompt = sys.stdin.read().strip()

        if not prompt:
            print("Error: no prompt provided", file=sys.stderr)
            sys.exit(1)

        result = await agent.handle_user_message(prompt)
        print(result.text)
