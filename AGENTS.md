# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

> **Architecture in one line:** Issues live in a local Dolt database
> (`.beads/dolt/`); cross-machine sync uses `bd dolt push/pull` (a
> git-compatible protocol), stored under `refs/dolt/data` on your git
> remote — separate from `refs/heads/*` where your code lives.
> `.beads/issues.jsonl` is a passive export, not the wire protocol.
>
> See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md)
> for the one-screen overview and anti-patterns (don't treat JSONL as the
> source of truth; don't `bd import` during normal operation; don't
> reach for third-party Dolt hosting before trying the default).

**Vault-synced checkout caveat:** the guidance above assumes a Dolt-synced
repo. A vault-synced checkout (e.g. `/data/obsidian`) syncs beads via a
git-tracked `.beads/issues.jsonl` instead, and must NOT run `bd dolt push`,
`bd dolt pull`, or `bd import` there — do not run those three commands in
such a checkout. That checkout's own CLAUDE.md is authoritative there and
overrides this file's Dolt-sync instructions.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

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

## Build & Test

Full test suite (from the main checkout):

```bash
.venv/bin/python -m pytest
```

Targeted run for a single file or test node, as usual with pytest:

```bash
.venv/bin/python -m pytest tests/test_some_module.py -v
```

The suite is self-isolating: `tests/conftest.py` points `SR2_HOME` at a
per-test temp dir and drops an inherited `SR2_SESSION_LOG_DIR`, so it passes
with or without the dispatch launch's session-log environment. No `env -u`
prefix is needed.

### Running tests in a build worktree

Build sessions work in Git worktrees under `/data/worktrees/sr2-spectre/`
(one per bead), which carry no `.venv`. Run tests there with the main
checkout's venv interpreter and `PYTHONPATH=src`, so the worktree source
shadows the editable install of `sr2_spectre` while sr2 core still resolves
to the editable `../sr2` install:

```bash
cd /data/worktrees/sr2-spectre/<bead> && PYTHONPATH=src /home/shepard/git/sr2-spectre/.venv/bin/python -m pytest
```

When the change also edits sr2 core, prepend its source so core imports come
from there instead of the editable install:
`PYTHONPATH=src:/home/shepard/git/sr2/src`.

Never run `uv sync` or `uv pip install -e .` inside a worktree. The
`sr2 = { path = "../sr2", editable = true }` dependency resolves against the
worktree parent (`/data/worktrees/sr2-spectre/sr2`), which does not exist, so
sync fails partway and leaves a half-provisioned `.venv` behind — delete that
directory if one appears. The recorded full-suite baseline assumes the
main-checkout venv: a fresh standalone worktree venv diverges from it (for
example the 7 zulip e2e FileNotFound failures). The dispatcher runs roles
against the stable main-checkout install and injects no `PYTHONPATH`; the
worktree code reaches tests only through the `PYTHONPATH` in the command
above.

## Mandates

Project identity statements. Agent G and the review lenses grade every
change against them; a violation is escalation trigger 3.

- **Every behaviour and every message is a YAML parameter.** Any switch,
  threshold, timing, retry or guard that changes what Spectre does, and any
  text shown to a user or sent to a model, is a field in the YAML config
  (`SpectreConfig` in `src/sr2_spectre/config.py`, documented in
  `config.example.yaml` and `docs/CONFIG-REFERENCE.md`). Behaviour that
  belongs to the SR2 pipeline is a field in SR2 `PipelineConfig`, not a
  Spectre-side side-config or split hook. No hardcoded message strings, no
  constants that a deployment might want to vary. (Diego, 2026-10-20, spc-959.)
- **Discover the roster, never hardcode it.** Agents are the YAML files
  under `$SR2_HOME/agents` (fallback `~/.sr2/agents`). Tooling, tests and
  other projects read that directory; a literal list of agent names is a
  defect. (Diego, 2026-10-19; galaxy-map fixed its copy in gmap-48.)

## Vault continuity

When work is tracked from `/data/obsidian`, its source area `NOW.md` is a
compact present-state view. At meaningful close, rewrite it with one executable
next action near the top, at most five current-state bullets, and only current
bead/spec plus latest restart-relevant acceptance-pack pointers. Remove
completed or superseded detail; never append a session summary, log, open-bead
inventory, test baseline, or evidence. Never create or update `handoff.md`, and
never edit the generated vault-root `NOW.md`.

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
