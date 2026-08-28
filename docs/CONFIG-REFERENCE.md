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

**Restart required to change.** Frame ids are derived from it, so changing it
under a running process would orphan every open conversation.

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

**Restart required to change.** Each server owns a connected transport, and a
`stdio` server owns a subprocess; neither can be re-seated by a config reload.

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

Every field here is [live-reloadable](#live-reload): correcting a model name or
a wrong `base_url` applies to the next message, including in conversations that
are already open. No restart needed.

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

`project: __auto__` resolves per turn, highest priority first: the run
context's `area` (see [Area resolution](#area-resolution)), then
`SR2_PROJECT`, then the nearest `.git` walking up from `cwd`, then the `cwd`
basename. An explicitly empty area stops resolution there — no knowledge is
injected and no fallback runs. When `knowledge_root` is left implicit it
follows from the resolved name, `~/.sr2/knowledge/<project>/`.

The `project:` field keeps its name; it is the one point where an incoming
area is translated to this resolver's notion of a project.

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
  guilds: []
  users: []
  mention_only: false
  max_message_length: 2000
  edit_stream_interval: 1.0
  tool_embed_enabled: true
  auto_thread: false
  channel_areas: {}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `token` | str | `""` | Discord bot token (**restart required** — the gateway session is already open with it) |
| `channels` | list[int] | `[]` | Channel IDs to monitor (empty = all). A **thread** counts as in-list when its parent channel is — so `auto_thread: true` follow-ups in auto-created threads keep working |
| `guilds` | list[int] | `[]` | Guild (server) IDs to respond in (empty = every guild the bot is in). Messages from other servers are dropped; DMs are unaffected |
| `users` | list[int] | `[]` | User IDs to respond to (empty = every user). When set, this is the strictest filter — messages from other users are dropped in any guild or channel |
| `mention_only` | bool | `false` | Only respond when mentioned |
| `max_message_length` | int | `2000` | Max chars per message (Discord limit) |
| `edit_stream_interval` | float | `1.0` | Seconds between stream edits (0 = disabled) |
| `tool_embed_enabled` | bool | `true` | Show tool execution as embeds |
| `auto_thread` | bool | `false` | Start a thread for each new conversation |
| `channel_areas` | dict[str, str] | `{}` | Channel ID (as a **string**) to area name. Overrides the name derived from the channel. `""` means "this channel has no area". Keyed on the **parent** channel for threads |

### Area resolution

On every inbound message the interface derives an **area** from the message's
channel and stamps it on the agent's `RunContext`. Resolution runs per
message, so a `channel_areas` edit applies to the next message with no
restart.

Order, highest priority first:

1. `channel_areas[str(channel_id)]` when the ID is present in the map — an
   empty-string value means "this channel has no area".
2. Otherwise the channel name, lowercased, with leading and trailing
   non-alphanumeric characters stripped. No other transformation.

For a Thread, the **parent** channel supplies both the ID and the name, so a
follow-up inside an auto-created thread resolves to the same area as the
message that opened it. A DM, an orphaned thread, an unreadable channel name,
or a name with nothing alphanumeric in it (`---`, a bare emoji) all yield no
area.

Discord always stamps *something*: "no area" reaches `RunContext` as `""`,
never `None`. That distinction is load-bearing — see the [interface
development guide](INTERFACE-DEV-GUIDE.md#runcontextarea-three-states).

The interface does not check whether the derived area exists anywhere;
existence is each consumer's concern. Today the only consumer is the `plan`
resolver (see [Resolver types](#resolver-types)). A channel with no area
injects no project knowledge rather than falling back to whatever `cwd`
happens to be.

Each message logs one INFO line naming the area and how it was derived:

```
area=fractured-roots (derived from channel name, channel=123456789)
area=fractured-roots (channel_areas override, channel=123456789)
area=none (channel=123456789)
```

Renaming a Discord channel silently changes the area it resolves to. That is
an accepted trade for not hand-maintaining a map of numeric IDs; this log line
and the `channel_areas` override are the mitigations.

### Config reload

The Discord bot re-reads the **whole** config on every inbound message — not
just this section. `token` is the only field here that needs a restart. See
[Live reload](#live-reload) for what applies immediately and what does not.

---

## Live reload

A long-running interface re-reads the **whole resolved config on every inbound
message**, so most edits apply to the next message with no process restart.
This is on for the Discord interface. Short-lived interfaces (`single_shot`,
`tui`) resolve their config once at startup and never reload.

The re-read runs the full 4-tier merge, so an edit to *any* tier is picked up —
not just the agent file.

### What applies immediately

| Config | Effect on the next message |
|--------|----------------------------|
| `models.default.model` | Subsequent calls use the new model |
| `models.default.base_url` | Subsequent calls hit the new endpoint |
| `models.default.api_key` | Subsequent calls use the new key |
| `models.default.params` | New sampling params take effect |
| `pipeline.*` | Each conversation rebuilds its SR2 on its next turn |
| `agent.tools` | Tools added, removed, or reconfigured |
| `agent.skills`, `agent.skills_dirs` | Skill registry rebuilt |
| `agent.tool_result_max_bytes` | New truncation limit |
| `discord.*` except `token` | Channel filters, threading, streaming, embeds |

### What needs a restart

These are wired into a live connection, subprocess or identity at startup, and
a reload has no way to rebuild them. The file value is **ignored** and the
value the process started with stays in force.

| Config | Why |
|--------|-----|
| `agent.name` | Frame ids derive from it; changing it orphans open conversations |
| `agent.mcp_servers` | Each server owns a connected transport; `stdio` owns a subprocess |
| `memory_store_dsn` | Backs a connection opened once at startup |
| `provenance_store_path` | Backs a connection opened once at startup |
| `discord.token` | The gateway session was opened with it |

A change to one of these logs a warning **once** — not per message — then keeps
the startup value:

```
Spectre config: 'agent.mcp_servers' changed on disk but cannot be applied
without a restart — keeping the value the process started with.
```

If an edit seems to be ignored and you see no warning repeating in the log,
check for that single line: it is the tell that you edited a pinned field.
Restart the process to apply it:

```bash
systemctl --user restart sr2-discord@<agent>
```

### Guarantees

- **A bad config cannot take the process down.** A file that is missing,
  malformed, fails validation, or is caught mid-save leaves the last known-good
  config in force. The error is logged once (repeats suppressed, so a broken
  file does not flood the log with a line per message). Fix the file and the
  next message picks it up, logging `reload recovered`.
- **No message is handled against a half-applied edit.** A reload swaps in a
  whole new config object rather than mutating fields.
- **Reads are not frozen for the duration of a reply.** If a second message
  lands while a reply is still streaming, its reload is visible to the reply in
  progress from that point on.
- **An endpoint fix reaches conversations that are already open.** Sessions
  hold a stable LLM handle whose target is swapped underneath them, so you do
  not have to start a new conversation for a corrected `base_url` to take
  effect.
- **A pipeline edit costs no history.** The conversation transcript is owned
  outside SR2 and re-seeded on every turn, so rebuilding a session's SR2 loses
  no messages. The rebuild is deferred to that session's next turn, so a reply
  that is mid-stream is never disturbed.
- **A tool that fails to construct costs you that tool, not the bot.** On
  reload the failure is logged and the previous registration (if any) stays.
  At *startup* the same failure is fatal — a config that cannot build should
  not come up pretending it works.

### Cost

Steady state is one config re-read plus an equality check, synchronously on the
interface's event loop, once per message. Measured at **~6 ms/message** against
a real agent file with an `extends:` chain (`~/.sr2/agents/miranda.yaml`,
2 MCP servers, 8 tools).

Rebuild work — tools, skills, the LLM handle, SR2 instances — is gated on an
observed change, so a reload that finds nothing new costs the re-read and the
compare, nothing more. That cost is paid per inbound message, so it scales with
traffic, not with config size.

### Implementation

| Piece | Where |
|-------|-------|
| Reload machinery, pinned fields | `sr2_spectre/config_source.py` |
| Swappable LLM handle | `sr2_spectre/live_llm.py` |
| Applying a reload to a live process | `Runtime.apply_config` |
| Per-conversation adoption, SR2 rebuild | `Session.apply_config` |
| Discord wiring (one read serves both halves) | `interfaces/discord/config_source.py` |

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
