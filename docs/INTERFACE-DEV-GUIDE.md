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
- **`area`**: Optional. Leave it unset unless your interface resolves areas — see below.

### `RunContext.area`: three states

`area` defaults to `None`, and resolvers see it through the run-context
provider, which omits the key entirely when it is `None`. That gives three
states, and consumers are required to tell all three apart:

| `RunContext.area` | What resolvers see | Meaning |
|---|---|---|
| `None` (default) | key **absent** | this interface does not resolve areas — fall through to existing behavior |
| `""` | key present, empty | explicitly **no** area — inject nothing, do **not** fall through |
| `"fractured-roots"` | key present, non-empty | use it |

Most interfaces want the default. `tui`, `single_shot` and `repl` all leave
`area` at `None`, so the `plan` resolver keeps deriving its project from
`SR2_PROJECT` and the `cwd` `.git` walk exactly as before.

Only set it if your interface genuinely knows which area a message belongs to.
If you do, **never stamp `None` to mean "no area"** — that reads as "this
interface does not do areas" and lets consumers fall through to `cwd`, which
silently answers from the wrong area's context. Stamp `""` instead.

`RunContext` is frozen, so stamping per message means constructing a new one
and calling `set_run_context()` again. That is safe: `Session` reads the run
context at resolve time within the same turn, and turns on one session are
serialized under the session lock. The Discord interface does exactly this in
`_process_message` — see [Area resolution](CONFIG-REFERENCE.md#area-resolution).

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

## Live config reload (long-running interfaces)

An interface that runs for days — a chat bot, a daemon — outlives the config it
started with. Without opting in, the agent it drives is frozen at the config
resolved at process start, so fixing a wrong `models.default.base_url` means a
restart. Short-lived interfaces (`single_shot`, `tui`) do not need any of this.

Opting in is two pieces: take a config **source** instead of a config object,
and hand the reloaded config to the agent on each inbound message.

```python
from sr2_spectre.config_source import SpectreConfigSource


class MyBotInterface:
    def __init__(self, config_source: SpectreConfigSource) -> None:
        self._source = config_source
        self._agent = None

    async def start(self, agent) -> None:
        self._agent = agent

    async def _on_inbound_message(self, text: str) -> None:
        # One re-read per message, at the single entry point for inbound
        # traffic. reload() returns the config now in force.
        config = self._source.reload()

        # Hand it to the agent: models, endpoints, pipeline, tools and skills
        # are re-seated as needed. Returns the areas actually applied, or []
        # when nothing changed.
        self._agent.apply_config(config)

        # ... your own settings come off `config` too ...
        async for event in self._agent.stream_message(text):
            ...
```

Build the source with `cli.build_spectre_config_source(config_path, cwd,
initial)`, which re-runs the same 4-tier resolution as startup.

### Rules

1. **Reload once, at the single entry point for inbound messages.** Not per
   handler, not per field read. Everything downstream reads `source.current`
   so one message is handled against one config.

2. **Never let a reload failure escape.** `reload()` already absorbs a
   malformed or missing file and keeps the last known-good config in force —
   do not add a `try` that turns that into a crash, and do not call the loader
   yourself.

3. **Call `agent.apply_config()` before you handle the message**, so the reply
   is produced against the config that was on disk when it arrived.

4. **Do not cache config values on your interface.** Read through to
   `source.current` (or the object `reload()` returned) at the point of use, or
   an edit will apply to some of your behaviour and not the rest.

5. **Pinned fields are not yours to apply.** `agent.name`, `agent.mcp_servers`,
   the store paths and interface credentials are held at their startup values
   by the source itself and warn once. If your interface has a field of its own
   that cannot change under a live process (a session token, a bound port), add
   it to a `PINNED_FIELDS` tuple rather than applying it and hoping.

See [CONFIG-REFERENCE.md § Live reload](CONFIG-REFERENCE.md#live-reload) for
the full hot-versus-restart breakdown, and
`interfaces/discord/config_source.py` for a worked example — `DiscordConfigView`
shows how one read per message can serve both the interface's own settings and
the agent's.

## Best Practices

1. **Set `RunContext` in `start()`** — The agent uses this for logging and mode-specific behavior. Leave `area` unset unless your interface resolves areas; if it does, re-stamp per message and use `""` — never `None` — for "no area".

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
