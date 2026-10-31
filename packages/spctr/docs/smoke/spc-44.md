# Smoke Test Runbook — spc-44: streamable-http MCP transport

**Proves:** MCPClient can connect to an MCP server over streamable-http (the transport glyph serves), in addition to stdio and SSE. **Does NOT cover:** the searxng stdio SIGKILL-on-shutdown cleanup noise (benign, deferred — see close reason).

## Setup

Requires the glyph MCP server running locally on its streamable-http endpoint (default `http://localhost:8420/mcp`).

```bash
cd ~/git/sr2-spectre
```

## Scenarios

### 1. Unit: streamable-http branch returns one bridge per tool

```bash
uv run pytest tests/test_mcp_client.py -k streamable -q
```

**Expect:** `2 passed` — connect returns bridges, transport failure raises MCPConnectionError.

### 2. Unit: runtime maps type=streamable-http to the right client

```bash
uv run pytest tests/test_mcp_wiring.py -k streamable -q
```

**Expect:** `1 passed`.

### 3. Live: glyph connects over streamable-http and lists tools

```bash
uv run python -c "import asyncio; from sr2_spectre.mcp.client import MCPClient; c=MCPClient('streamable-http', url='http://localhost:8420/mcp'); print(sorted(b.name for b in asyncio.run(c.connect())))"
```

**Expect:** a sorted list including `search`, `lookup`, `get_context`, `list_sources` (glyph's tools). No `MCPConnectionError`.

### 4. Full suite unaffected

```bash
uv run pytest -q
```

**Expect:** `1274 passed` (or more), exit 0.

## Teardown

None — no state written.

## Pass Criteria

| # | Scenario | Pass |
|---|----------|------|
| 1 | streamable unit tests | [ ] |
| 2 | runtime wiring test | [ ] |
| 3 | live glyph connect | [ ] |
| 4 | full suite green | [ ] |

## Next Action

All green → glyph MCP is live in spectre runs. Remaining spc-44 sub-item (searxng stdio cleanup) deferred to obsidian-rry-adjacent follow-up if the shutdown noise ever causes a real failure.
