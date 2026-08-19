# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## v2 Product-Owner Build Sessions

For a spec-defined task bead, `/data/obsidian/workflow/README.md` and
`session-pacing.md` override the generic completion text above. Diego is
not a midpoint reviewer: run A → B → C → D and stop only for the five
canonical escalation triggers. `ACCEPTED` is Agent D automated acceptance,
not a claim of human review; it closes the bead and permits delivery under
the active project profile.

Before a v2 run, this file must state the exact full-suite command and the
project mandates Agent D checks. If either is absent, state the fallback
and use per-step human review. Do not write smoke-test runbooks for v2
work, and do not `git pull --rebase` after acceptance: integration that
changes the accepted range requires a new acceptance run.


## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Conventions & Patterns

### Live config reload (obsidian-fvfs, obsidian-bzfb)

Long-running interfaces re-read the **whole** resolved `SpectreConfig` on every
inbound message. Do not tell the operator to restart an agent for a config edit
without checking whether the field is pinned.

- **Hot** (applies to the next message): `models.default.*`, `pipeline.*`,
  `agent.tools`, `agent.skills`, `agent.skills_dirs`,
  `agent.tool_result_max_bytes`, all `discord.*` except `token`.
- **Pinned** (needs a process restart, warns once, keeps the startup value):
  `agent.name`, `agent.mcp_servers`, `memory_store_dsn`,
  `provenance_store_path`, `discord.token`.

Flow: `cli.build_spectre_config_source` → `SpectreConfigSource` (last-good
fallback; a malformed file never takes the process down) → `DiscordConfigView`
gives the adapter the Discord slice of that same single read →
`Runtime.apply_config` re-seats the rest, every branch gated on an observed
change so a no-op reload costs one equality check.

Two invariants worth preserving when touching this:

- Sessions hold a `LiveLLM` (`src/sr2_spectre/live_llm.py`), not a bare
  `LiteLLMCallable`. SR2 captures its LLM for the life of a session, so without
  that indirection an endpoint fix would only reach *new* conversations.
- A pipeline change rebuilds a session's SR2, but the rebuild is **deferred to
  that session's next turn, under its own lock** (`Session._sr2_stale`). Doing
  it eagerly would swap the SR2 out from under an in-flight reply whose tool
  executor is still publishing to `sr2.bus`.

**Test seam:** patch `sr2_spectre.live_llm.LiteLLMCallable`, *not*
`sr2_spectre.runtime.LiteLLMCallable` — LLM construction moved to
`live_llm.build_llm`. Docs: `docs/CONFIG-REFERENCE.md` § Live reload;
`docs/INTERFACE-DEV-GUIDE.md` § Live config reload.

### Memory store backend (obsidian-cor)

The Runtime selects the memory store backend at construction via
`Runtime._resolve_memory_dsn(config)`:

- `config.memory_store_dsn` is authoritative when set. A non-empty DSN selects
  `PostgresMemoryStore` (persistent, shared across processes). `""` explicitly
  disables persistence → in-memory, even if the env var is set.
- When `config.memory_store_dsn` is `None`, the `SPECTRE_MEMORY_DSN` env var is
  used if non-empty; otherwise the dict-backed `InMemoryMemoryStore`.

The selected store is shared across all sessions and closed in `Runtime.aclose()`
(Postgres `close()` is synchronous; in-memory has none). Persistent store lives
in `sr2.memory.pg_store`. Smoke runbook: `docs/smoke/obsidian-cor.md`.
