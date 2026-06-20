# SR2 Spectre — Interface Development Guide

How to build custom interfaces for SR2 Spectre. An interface is an I/O channel that receives user input, drives the agent loop, and renders the response.

## The Interface Protocol

Every interface implements the `Interface` Protocol from `sr2_spectre.interfaces`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Interface(Protocol):
    name: str                          # Short identifier: "tui", "discord", etc.

    async def start(self, agent: "Agent") -> None: ...
    async def stop(self) -> None: ...
    async def run(self, agent: "Agent") -> None: ...
```

### Method semantics

| Method | When called | Purpose |
|--------|------------|---------|
| `start(agent)` | Before `run()` | Initialize resources, set `RunContext` on the agent |
| `run(agent)` | After `start()` | The main loop. Blocks until the interface is done |
| `stop()` | On shutdown | Clean up resources (connections, file handles, etc.) |

### `start()` — set the run context

Call `agent.set_run_context()` in `start()` to tell the agent how it's being driven:

```python
from sr2_spectre.core import RunContext, RunMode

async def start(self, agent: "Agent") -> None:
    agent.set_run_context(RunContext(
        interface="my_interface",
        mode=RunMode.INTERACTIVE,  # or RunMode.HEADLESS
        source=None,               # context-specific: cwd, channel ID, etc.
    ))
```

- **`interface`**: Your interface name — shown in logs and diagnostics.
- **`mode`**: `RunMode.INTERACTIVE` if the user is present (TUI, Discord). `RunMode.HEADLESS` for scripting/CI.
- **`source`**: Optional context (working directory, channel name, request ID).

## Driving the Agent

You have two ways to interact with the agent:

### Option A: `handle_user_message()` — simple request/response

Returns a `TurnResult` with the complete response text after all tool calls finish.

```python
result = await agent.handle_user_message(prompt)
print(result.text)
# result.tool_calls_executed — number of tool calls in this turn
# result.total_tokens — total tokens used
```

Use this for single-shot, batch, or non-streaming interfaces.

### Option B: `stream_message()` — event stream

Yields `AgentEvent` objects as they happen. Use this for interfaces that show progress in real time (TUI, Discord streaming).

```python
async for event in agent.stream_message(prompt):
    if isinstance(event, AgentTextDelta):
        sys.stdout.write(event.text)
        sys.stdout.flush()
    elif isinstance(event, AgentToolStart):
        print(f"\n⚙ {event.name}(...)")
    elif isinstance(event, AgentToolResult):
        status = "✓" if not event.is_error else "✗"
        print(f"{status} {event.name} done")
    elif isinstance(event, AgentThinkingDelta):
        sys.stdout.write(event.text)
    elif isinstance(event, AgentDone):
        print(f"\n[{event.tool_calls_executed} tools]")
```

### Available event types

| Event | Attributes | Meaning |
|-------|-----------|---------|
| `AgentTextDelta` | `text: str` | Chunk of assistant text |
| `AgentThinkingDelta` | `text: str` | Chunk of reasoning/thinking text |
| `AgentToolStart` | `name: str`, `input: dict`, `tool_id: str` | Tool call starting |
| `AgentToolResult` | `name: str`, `content: str`, `is_error: bool`, `tool_id: str` | Tool call finished |
| `AgentDone` | `tool_calls_executed: int` | Turn complete |

## Minimal Example

Here's a bare-bones interface that reads from stdin and writes to stdout:

```python
"""MyConsoleInterface — reads from stdin, writes to stdout."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sr2_spectre.core import RunContext, RunMode

if TYPE_CHECKING:
    from sr2_spectre.agent import Agent


class MyConsoleInterface:
    """A simple console interface."""
    name = "myconsole"

    async def start(self, agent: "Agent") -> None:
        agent.set_run_context(RunContext(
            interface="myconsole",
            mode=RunMode.INTERACTIVE,
            source=None,
        ))

    async def stop(self) -> None:
        pass  # Nothing to clean up

    async def run(self, agent: "Agent") -> None:
        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                break

            if not prompt.strip():
                continue

            if prompt.strip() in ("/quit", "/exit"):
                break

            result = await agent.handle_user_message(prompt)
            print(result.text)
```

## Registering Your Interface

Spectre loads interfaces by class path. There are two approaches:

### Approach 1: CLI `--interface` flag (for built-in or pre-registered interfaces)

The CLI resolves interface names through the `_load_interface` function. Add your interface to the known interfaces in `cli.py`:

```python
# In cli.py, extend the interface resolution
KNOWN_INTERFACES = {
    "single_shot": "sr2_spectre.interfaces.single_shot:SingleShotInterface",
    "tui": "sr2_spectre.interfaces.tui:TUIInterface",
    "discord": "sr2_spectre.interfaces.discord:DiscordInterface",
    "myconsole": "my_package.interfaces:MyConsoleInterface",  # Add yours
}
```

### Approach 2: Direct instantiation (for custom runners)

```python
from my_package.interfaces import MyConsoleInterface

interface = MyConsoleInterface()
await interface.start(agent)
await interface.run(agent)
await interface.stop()
```

## Best Practices

1. **Set `RunContext` in `start()`** — The agent uses this for logging and mode-specific behavior.

2. **Use `stream_message()` for interactive interfaces** — Users want to see responses as they arrive, not wait for the full response.

3. **Handle `KeyboardInterrupt` gracefully** — Don't let Ctrl+C crash the agent. Catch it and clean up.

4. **Separate rendering from logic** — Keep your interface's rendering code (printing, embedding, etc.) separate from message routing logic.

5. **Clean up in `stop()`** — Close connections, flush buffers, disconnect signals. Don't leak resources.

6. **Don't block the event loop** — Use `async` I/O throughout. A blocking `input()` in a `run()` loop should be `prompt_async()` or similar.

7. **Respect the session model** — Each interface can maintain its own session state (like Discord's per-channel history). The agent's `history` and `session_id` are per-session.

## Testing Your Interface

Mock the agent and verify your interface routes messages correctly:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from sr2_spectre.events import AgentDone, AgentTextDelta
from my_package.interfaces import MyConsoleInterface


@pytest.mark.asyncio
async def test_interface_runs_and_sends_prompt() -> None:
    """Interface sends user input to the agent and prints response."""
    agent = AsyncMock()
    from sr2_spectre.core import TurnResult
    agent.handle_user_message = AsyncMock(return_value=TurnResult(
        text="Hello!",
        tool_calls_executed=0,
        total_tokens=10,
    ))

    interface = MyConsoleInterface()
    await interface.start(agent)

    # Verify run context was set
    # (check agent.set_run_context was called)

    await interface.stop()
```

See `tests/test_discord_interface.py` and `tests/test_tui_scaffold.py` for complete examples.

## Reference: Built-in Interfaces

| Interface | File | Mode | Highlights |
|-----------|------|------|------------|
| `SingleShotInterface` | `interfaces/single_shot.py` | Headless | One prompt, one response, exit |
| `TUIInterface` | `interfaces/tui.py` | Interactive | prompt-toolkit, streaming, slash commands, session save/load |
| `DiscordInterface` | `interfaces/discord/interface.py` | Interactive | Per-channel sessions, message streaming via edits, embeds |
