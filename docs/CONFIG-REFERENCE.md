# SR2 Spectre — Configuration Reference

Complete reference for all configuration fields, resolution order, and environment variables.

## File structure

A Spectre config is a YAML file with three top-level sections:

```yaml
agent:      # Spectre-owned concerns (tools, MCP, skills)
models:     # LLM endpoints
pipeline:   # SR2 pipeline (layers, budgets, resolvers)
```

Copy `config.example.yaml` as a starting point.

---

## Agent section

```yaml
agent:
  name: my-agent
  tools: []
  mcp_servers: []
  skills: []
  tool_result_max_bytes: 65536
```

### `agent.name` (str, default: `"spectre"`)

Human-readable name for the agent. Used in logs and session identifiers.

### `agent.tools` (list[ToolConfig])

Built-in tools to register. Each entry:

```yaml
  tools:
    - name: terminal
      class_path: sr2_spectre.tools.builtins.terminal.TerminalTool
      config:
        timeout: 30
```

- **`name`** (str, required): Tool identifier. The LLM sees this name when deciding which tool to call.
- **`class_path`** (str, required): Python import path to the tool class. Format: `module.submodule.ClassName`.
- **`config`** (dict, optional): Tool-specific configuration passed to the constructor as `**kwargs`.

#### Available built-in tools

| `name` | `class_path` | Config options |
|--------|-------------|----------------|
| `terminal` | `sr2_spectre.tools.builtins.terminal.TerminalTool` | `timeout` (int, default: 30) |
| `file_read` | `sr2_spectre.tools.builtins.file_read.FileReadTool` | `max_bytes` (int, default: 1000000) |
| `file_write` | `sr2_spectre.tools.builtins.file_write.FileWriteTool` | none |
| `edit` | `sr2_spectre.tools.builtins.edit.EditTool` | none |
| `grep` | `sr2_spectre.tools.builtins.grep.GrepTool` | none |
| `glob` | `sr2_spectre.tools.builtins.glob.GlobTool` | none |
| `web_search` | `sr2_spectre.tools.builtins.web_search.WebSearchTool` | `base_url` (str, required — SearXNG URL), `max_results` (int, default: 5) |
| `code_exec` | `sr2_spectre.tools.builtins.code_exec.CodeExecTool` | `timeout` (int, default: 10) |
| `read_symbol` | `sr2_spectre.tools.builtins.read_symbol.ReadSymbolTool` | none |
| `complete_step` | `sr2_spectre.tools.builtins.complete_step.CompleteStepTool` | `plans_root` (str, default: `~/.sr2/plans`) |
| `load_skill` | `sr2_spectre.tools.builtins.load_skill.LoadSkillTool` | none |
| `test_guard` | `sr2_spectre.tools.builtins.test_guard.TestGuardTool` | none |

### `agent.mcp_servers` (list[McpServerConfig])

External MCP (Model Context Protocol) servers to connect at startup. Tools from these servers are registered alongside built-in tools.

```yaml
  mcp_servers:
    - name: searxng
      type: http
      url: http://localhost:8080
    - name: glyph
      type: http
      url: http://localhost:8420/mcp
    - name: beads
      type: stdio
      command: ["beads-mcp"]
      args: ["serve"]
```

- **`name`** (str, required): Display name for this server.
- **`type`** (str, required): `"stdio"` (command-based) or `"http"` (SSE transport).
- **`command`** (list[str], optional): Command to run for stdio servers.
- **`args`** (list[str], optional): Additional arguments for the command.
- **`env`** (dict[str, str], optional): Environment variables for the subprocess.
- **`url`** (str, required for http): Server URL.

### `agent.skills` (list[SkillConfig])

Loadable skill files — knowledge packages loaded at runtime.

```yaml
  skills:
    - name: my-skill
      path: /path/to/skill.md
      description: "A useful skill"
      version: "0.1.0"
      tags: ["engineering", "python"]
```

- **`name`** (str, required): Skill identifier.
- **`path`** (str, required): File path. Supports `~` and `${VAR}` expansion.
- **`description`** (str, optional): Override the auto-derived description.
- **`version`** (str, default: `"0.1.0"`): Skill version.
- **`tags`** (list[str], optional): Tags for skill filtering.

### `agent.tool_result_max_bytes` (int, default: `65536`)

Maximum size of a tool result before truncation. Results exceeding this limit are truncated to prevent context explosion.

---

## Models section

```yaml
models:
  default:
    model: openai/qwen3:27b
    base_url: http://localhost:11434/v1
    params:
      temperature: 0.7
      top_p: 0.9
```

A dictionary mapping named endpoints to `ModelConfig`. The `"default"` key is required.

### `model` (str, required)

Model identifier. Format: `{provider}/{model_name}` (e.g., `openai/gpt-4o`, `openai/qwen3:27b`). The provider prefix determines which LiteLLM provider is used.

### `base_url` (str, optional)

Base URL for the LLM endpoint. Omit for hosted APIs (OpenAI, Anthropic via LiteLLM) — they use their default endpoints.

Required for local servers (Ollama, LM Studio, vLLM).

### `params` (dict, optional)

Sampling parameters forwarded to the LLM provider. Common options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | provider default | Randomness (0.0–1.0) |
| `top_p` | float | provider default | Nucleus sampling threshold |
| `max_tokens` | int | provider default | Max response tokens |
| `stream` | bool | `true` | Enable streaming |

---

## Pipeline section

The pipeline section is SR2's native `PipelineConfig`. It defines context compilation layers, token budgets, and tool iteration limits.

```yaml
pipeline:
  token_budget: 200000
  max_tool_iterations: 40
  layers:
    - name: system
      target: system
      resolvers:
        - type: static
          config:
            text: |
              You are a helpful AI assistant.
    - name: tools
      target: tools
      resolvers: []
      tool_providers:
        - type: spectre_tools
    - name: conversation
      target: messages
      resolvers:
        - type: session
        - type: input
```

### `pipeline.token_budget` (int, default: `200000`)

Maximum total tokens across all compiled context layers.

### `pipeline.max_tool_iterations` (int, default: `40`)

Maximum number of LLM → tool → LLM cycles per turn. Prevents infinite tool-use loops.

### `pipeline.layers` (list[LayerConfig])

Ordered list of context compilation layers. Each layer:

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Layer identifier |
| `target` | str | Target: `"system"`, `"tools"`, or `"messages"` |
| `resolvers` | list | Resolver configs that produce content for this layer |
| `tool_providers` | list | Tool provider configs (for `target: "tools"` layers) |

#### Resolver types

| `type` | Description | Config keys |
|--------|-------------|-------------|
| `static` | Static text (system prompt) | `text` (str) |
| `session` | Session history from current turn | none |
| `input` | Current user message | none |
| `plan` | Plan file resolution | `plans_root` (str), `project` (str) |
| `memory` | Memory store lookup | `scope` (str), `limit` (int), `prefix` (str) |
| `knowledge` | Knowledge file resolution | `knowledge_root` (str) |

---

## Config Resolution (4-tier merge)

Spectre uses a 4-tier configuration resolution system. Later tiers override earlier ones:

| Tier | File | Scope |
|------|------|-------|
| 1 | `$SR2_HOME/config.yaml` | User global defaults |
| 2 | `$SR2_HOME/spectre.yaml` | Spectre-specific defaults |
| 3 | `<cwd>/.spectre.yaml` | Project overrides |
| 4 | Positional file (`sr2-spectre my-config.yaml`) | Active run config |

Missing tier files are silently skipped. The positional file (tier 4) must exist.

### `extends:` key

Any config file can use `extends:` to inherit from another file. The extended file is resolved relative to the declaring file's directory. Supports `${VAR}` interpolation.

```yaml
extends: ../agents/base.yaml

agent:
  name: override-agent
```

Circular `extends:` chains raise `CircularExtendsError`.

### Path resolution

Paths in config files support:
- `~` expansion (e.g., `~/.sr2/plans`)
- `${VAR}` environment variable interpolation (e.g., `${SR2_HOME}/config.yaml`)
- Relative paths resolved against the declaring file's directory

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SR2_HOME` | `~/.sr2` | Base directory for SR2/Spectre configs and data |

---

## Discord Interface

The Discord interface has its own top-level config section, `discord`:

```yaml
discord:
  token: "your-bot-token"
  channels: []
  mention_only: false
  max_message_length: 2000
  edit_stream_interval: 1.0
  tool_embed_enabled: true
  auto_thread: false
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `token` | str | `""` | Discord bot token (restart required to change) |
| `channels` | list[int] | `[]` | Channel IDs to monitor (empty = all) |
| `mention_only` | bool | `false` | Only respond when mentioned |
| `max_message_length` | int | `2000` | Max chars per message (Discord limit) |
| `edit_stream_interval` | float | `1.0` | Seconds between stream edits (0 = disabled) |
| `tool_embed_enabled` | bool | `true` | Show tool execution as embeds |
| `auto_thread` | bool | `false` | Start a thread for each new conversation |

### Live reload

The Discord bot re-reads its config on **every inbound message**, so an edit
to any tier applies to the next message — no process restart. Details:

- `token` is the exception: the bot is already logged in with it, so a changed
  token is ignored (and logged) until the process restarts.
- A config that fails to parse or validate leaves the last good config in
  force; the bot keeps running and logs the error once. Fix the file and the
  next message picks it up.
- A reload swaps the whole config at once, so no message is handled against a
  half-applied edit. Reads are not frozen for the duration of a reply, though:
  if a second message lands while a reply is still streaming, its reload is
  visible to the reply in progress.
- The reload is a synchronous re-read of the config tiers (~7 ms against a
  typical agent file) on the bot's event loop, once per message.

---

## CLI flags

| Flag | Description |
|------|-------------|
| `<config>` | Positional: path to config file (tier 4) |
| `--interface <name>` | Interface to use: `tui`, `single_shot`, `discord` |
| `--plugin <name>` | Deprecated alias for `--interface` |
| `--prompt <text>` | Prompt text for single-shot mode |
| `--trace` | Print compiled context to stderr before LLM call |
| `--agent <name>` | Resolve agent config by name |
| `--dry-run` | Show merged config without running |
| `--show-provenance` | Show config source for each key (with `--dry-run`) |
