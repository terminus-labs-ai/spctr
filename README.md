# SR2 Spectre

A full agent runtime powered by [SR2](https://github.com/terminus-labs-ai/sr2) — streaming responses, tool execution, and a polished TUI out of the box.

## What it does

- **Streaming TUI** — responses stream token by token, tool calls are visible as they execute
- **Built-in tools** — `terminal`, `file_read`, `file_write`, `web_search` (SearXNG)
- **Interface system** — `single_shot` for scripting, `tui` for interactive use; Discord coming soon
- **SR2 pipeline** — layered context compilation with token budgets, memory, and degradation

## Install

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/terminus-labs-ai/sr2-spectre
cd sr2-spectre
uv sync
```

Spectre depends on [SR2](https://github.com/terminus-labs-ai/sr2). Clone it alongside and `uv` will pick it up via the path dependency in `pyproject.toml`.

```bash
git clone https://github.com/terminus-labs-ai/sr2 ../sr2
```

## Configure

Copy the example config and fill in your LLM endpoint:

```bash
cp config.example.yaml my-config.yaml
```

Edit `my-config.yaml` — at minimum set `models.default.model` and `models.default.base_url`.

Spectre works with any OpenAI-compatible API (Ollama, LM Studio, vLLM) and hosted providers (OpenAI, Anthropic via LiteLLM).

Long-running interfaces (the Discord bot) re-read the whole config on every
inbound message, so fixing a model or endpoint applies to the next message
without a restart. A few fields still need one — see
[Live reload](docs/CONFIG-REFERENCE.md#live-reload).

## Usage

**Interactive TUI:**
```bash
sr2-spectre my-config.yaml --interface tui
```

**Single-shot (pipe-friendly):**
```bash
sr2-spectre my-config.yaml "summarise this file"
echo "what is 2+2" | sr2-spectre my-config.yaml
```

**Debug pipeline with trace:**
```bash
sr2-spectre my-config.yaml --trace "list files in /tmp"
```

### TUI commands

| Command | Action |
|---------|--------|
| `/help` | Show available commands |
| `/tools` | List registered tools |
| `/reset` | Start a new session |
| `/quit` | Exit |

Ctrl+C during a response interrupts the current stream and re-prompts.

## Tools

Tools are registered in your config under `agent.tools`. Each tool is a Python class with `name`, `description`, `input_schema`, and an async `__call__`.

| Tool | Description |
|------|-------------|
| `terminal` | Run shell commands, returns stdout+stderr |
| `file_read` | Read a file from disk |
| `file_write` | Write a file to disk, creates parent dirs |
| `web_search` | Search via SearXNG JSON API |

Custom tools follow the same interface — point `class_path` at your class and it's registered automatically.

## Architecture

```
sr2-spectre/
  src/sr2_spectre/
    agent.py            # Agent — owns history, tool loop, stream_message()
    cli.py              # CLI entry point
    config.py           # SpectreConfig (Pydantic)
    events.py           # AgentEvent types (TextDelta, ToolStart, ToolResult, Done)
    providers.py        # SpectreToolProvider — bridges tool registry into SR2 pipeline
    interfaces/
      single_shot.py    # Non-interactive single-turn interface
      tui.py            # Interactive streaming TUI
    tools/
      registry.py       # ToolRegistry — register, define, execute tools
      builtins/         # Built-in tool implementations
```

Spectre owns conversation history, the tool registry, and event translation. SR2 owns the tool execution loop, context compilation, token budgets, and LLM calls. The boundary is `agent.stream_message()` → `sr2.turn()` — SR2 loops internally on tool use, and Spectre renders the resulting stream events.

## Development

```bash
uv run pytest          # run all tests
uv run ruff check .    # lint
```

## License

MIT
