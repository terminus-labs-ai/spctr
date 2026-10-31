# SR2 Spectre — Quickstart

Get Spectre running in 5 minutes with a local LLM.

## Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** (Python package manager)
- **A local LLM** running on an OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, etc.)

## Step 1: Clone

```bash
git clone https://github.com/terminus-labs-ai/sr2
git clone https://github.com/terminus-labs-ai/sr2-spectre
cd sr2-spectre
```

Spectre depends on the SR2 engine. Clone both repos as sibling directories — `uv` resolves the path dependency automatically.

## Step 2: Install

```bash
uv sync
```

## Step 3: Start your LLM

If using Ollama (example with a 27B model):

```bash
ollama serve
# In another terminal:
ollama run qwen3:27b
```

Note the endpoint — by default Ollama exposes `http://localhost:11434/v1`.

## Step 4: Configure

```bash
cp config.example.yaml my-config.yaml
```

Edit `my-config.yaml` — at minimum change the model section:

```yaml
models:
  default:
    model: openai/qwen3:27b       # adjust for your model
    base_url: http://localhost:11434/v1
```

Everything else in the example config works out of the box.

## Step 5: Run

**Interactive TUI:**
```bash
uv run sr2-spectre my-config.yaml --interface tui
```

**Single-shot (one question, one answer):**
```bash
uv run sr2-spectre my-config.yaml "Explain the architecture of this project"
```

**Pipe input:**
```bash
cat README.md | uv run sr2-spectre my-config.yaml "Summarize this"
```

## What happens next?

- **TUI mode:** Type your prompt, watch tokens stream in real time. Tool calls are visible as they execute.
- **Single-shot mode:** Spectre reads the prompt (or stdin), calls your LLM, prints the response, and exits.
- **With tools enabled:** The LLM can use `terminal`, `file_read`, `file_write`, and other built-in tools. Responses are richer because the model can actually read files, run commands, and explore the system.

## Next steps

- **[README.md](README.md)** — full documentation: architecture, tools, development workflow
- **Enable MCP servers** — add external tool sources (Glyph, SearXNG, custom tools) via `agent.mcp_servers` in your config
- **Enable planning** — uncomment the `complete_step` tool and add the PlanResolver to your pipeline for multi-step autonomous task execution
- **Custom tools** — implement the `Tool` interface and register via `class_path` in your config

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'sr2'` | Clone the `sr2` repo as a sibling directory (`../sr2`) |
| LLM returns empty responses | Check `model` name matches what your server exposes; verify `base_url` is correct |
| Tool calls timeout | Increase `agent.tools[].config.timeout` (default: 30s) |
| `uv sync` fails | Ensure `../sr2` exists and has a `pyproject.toml` |
