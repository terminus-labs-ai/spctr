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

Full test suite:

```bash
.venv/bin/python -m pytest
```

Targeted run for a single file or test node, as usual with pytest:

```bash
.venv/bin/python -m pytest tests/test_some_module.py -v
```

## Architecture Overview

### Resolver/transformer pipeline

Spectre plugs its planning behaviour into SR2 as pipeline extensions rather
than forking SR2 itself. `src/sr2_spectre/planning/resolver.py` defines
`PlanResolver`, which registers on the `sr2.resolvers` entry point (name
`plan`) and subscribes to the `turn_start` event. On every turn it re-reads
the plan/knowledge files under the configured `plans_root`/`knowledge_root`
(nothing is cached across turns, so status edits made mid-run show up on the
next turn) and injects three layered, delimited sections into the prompt:
L1 project knowledge, L2 the active plan, and L3 the current task, with L3
treated as most protected and L1 dropped first if a token budget is
exceeded. `src/sr2_spectre/planning/transformer.py` defines
`StepCompactionTransformer`, registered on `sr2.transformers` (name
`step_compaction`), which subscribes to `plan_step_completed` events fired
by the `complete_step` tool after its verify step passes. It is a
deterministic content-block filter with no LLM call: it burns every content
block tagged with the just-closed task frame and replaces it with a single
breadcrumb text block, keeping conversation history compact without losing
narrative continuity. Together, the resolver injects planning context in and
the transformer burns completed-task context back out once it is verified
done.

### Four-tier config resolution

`src/sr2_spectre/config.py` (`load_resolved_config_with_provenance`, around
lines 557-605) resolves a single `SpectreConfig` by merging four tiers,
lowest to highest priority:

1. `$SR2_HOME/config.yaml`
2. `$SR2_HOME/spectre.yaml`
3. `<cwd>/.spectre.yaml`
4. the extends-resolved positional config file passed on the command line

Tiers 1-3 are optional and silently skipped if missing; the positional file
(tier 4) is required and wins over all lower tiers. Each tier is merged onto
the accumulated result with `merge_configs`, and a parallel provenance map
tracks which file contributed each resolved field, which is what lets
Spectre tell an operator exactly which config file set a given value.

### Interface/runtime split

Spectre ships multiple front ends that all sit on top of one shared
`Runtime` (`src/sr2_spectre/runtime.py`): `single_shot`
(`src/sr2_spectre/interfaces/single_shot.py`, for scripting — one prompt in,
one response out, process exits), `repl` (default;
`src/sr2_spectre/interfaces/repl.py`, interactive terminal session via
prompt_toolkit + Rich — native selection and shortcuts), `tui`
(`src/sr2_spectre/interfaces/tui.py`, Textual full-screen app, legacy),
and Discord (`src/sr2_spectre/interfaces/discord/`, for a persistent bot
process). Each interface is a thin adapter that turns its own I/O model
(stdin/stdout, a terminal UI loop, or Discord gateway events) into calls
against the same `Runtime`, which owns session state, the SR2 pipeline, the
LLM, and the memory store. This split keeps interface-specific concerns
(rendering, event loops, platform APIs) out of the core agent logic, and
lets long-running interfaces (tui, discord) apply live config reloads to
that shared `Runtime` without needing interface-specific reload code.

### `agent.tools` merges additively — no revocation

`src/sr2_spectre/config_merge.py` implements the named-list merge rule used
throughout config merging (`_named_merge`, around lines 74-107): when a list
of dicts all share a `name` field — which is how `agent.tools` entries are
shaped — the child tier's list is merged with the parent tier's list by
`name`, not replaced. Parent entries keep their position (merged in-place if
the child also declares that name, otherwise left as-is), and any tool name
the child declares that the parent didn't gets appended. There is no
mechanism in this merge rule to delete or "subtract" an entry: a child tier
cannot express "the parent granted this tool, but I do not want it in this
context."

Security consequence: because `agent.tools` is a named list, a
higher-priority, more specific config (e.g. a positional project config,
tier 4) can only add to or override individual fields of a tool the parent
already granted — it can never revoke a tool that a broader, lower-priority
parent config (e.g. `$SR2_HOME/config.yaml`, tier 1) already listed. Anyone
relying on a narrower config to lock down or reduce an agent's tool
permissions relative to a broader default needs to be aware that `agent.tools`
cannot do this; the tool has to be removed at the tier that granted it.

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
