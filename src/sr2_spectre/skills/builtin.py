"""Built-in spectre skills.

Packages loadable knowledge packages that ship with sr2-spectre so any
agent can load them on demand via the load_skill tool or through
directory discovery.

Currently ships two skills:
- **sr2-conventions** — how to work with the SR2 framework
- **solid-review** — architectural audit framework (SOLID, DRY, mandate alignment)
"""

from __future__ import annotations

from sr2_spectre.skills.core import Skill

# ---------------------------------------------------------------------------
# SR2 Conventions Skill Content
# ---------------------------------------------------------------------------

SR2_CONVENTIONS_CONTENT = """# SR2 Conventions — Loadable Skill

> **Purpose:** This skill teaches an agent how to work with the SR2 agent
> framework. Load it when you need to understand SR2's patterns, pipeline,
> and conventions for building or interacting with SR2-powered agents.

## Overview

SR2 is an agent orchestration framework. It owns context compilation, the
tool execution loop, token budgets, and LLM calls. Spectre (sr2-spectre)
is the agent runtime that wraps SR2 — it owns agent identity, session
management, tool execution, and interfaces.

**Key principle:** SR2 is a stateless per-round context compiler. The Agent
seeds it with prior history and passes the newest message each turn.

## Pipeline Architecture

SR2's pipeline is event-driven with resolvers, providers, and transformers:

- **Resolvers** — inject content into the prompt on every turn (e.g., plan
  files, knowledge bases). Subscribed to events like `turn_start`.
- **Providers** — supply tool definitions, system prompts, or other
  context elements each turn.
- **Transformers** — modify the event stream or context (e.g.,
  summarization, step compaction).

All pipeline components are registered via entry points:
- `sr2.resolvers` — knowledge/content resolvers
- `sr2.tool_providers` — tool definition providers
- `sr2.transformers` — context transformers

## Durable Plans

SR2 agents use durable markdown plans for multi-step work:

- Plans live in `~/.sr2/plans/<plan-slug>/` (one directory per plan).
- `_plan.md` — plan-shared file with goal, status, and constraints contract.
- `NN-slug.md` — one file per atomic task with frontmatter (kind, plan,
  order, status, verify, title).

**Plan workflow:**
1. Ground: read the affected code and understand the codebase.
2. Record contract: document understanding and constraints in `_plan.md`.
3. Write plan: create the directory and task files.
4. Execute current task (injected as "Current Task" by PlanResolver).
5. Verify: run the task's `verify:` command.
6. Advance: set `status: done` when verified green.
7. Final validation: full test suite + re-check constraints.

**PlanResolver** (dynamic, per-turn):
- Re-reads plan/knowledge directories each turn.
- Injects L1 (project knowledge), L2 (active plan body), L3 (current task).
- Configurable: `plans_root`, `knowledge_root`, `project`, `max_tokens`.

## Knowledge Layers

Three-layer injection keeps the working set tight:

- **L1 — Project Knowledge:** Durable, cross-plan files under
  `~/.sr2/knowledge/<project>/`. Filtered by `kind: project-knowledge`
  frontmatter with matching `project` field.
- **L2 — Plan Shared:** The `_plan.md` body of the currently open plan
  (goal + constraints).
- **L3 — Current Task:** The lowest-order `pending` task from the open plan.

## Frame Primitive (Advanced)

Every content block can be tagged with `meta["frame"]` — a purpose-scoped
span. Closing a frame releases its blocks from the prompt. This enables
step-compaction: when a plan step completes, the transformer burns the
now-redundant context for that step.

- Frame ID format: `plan:<plan-slug>/<task-slug>`
- Event: `plan_step_completed` carries the closed frame id.
- Transformer drops blocks whose frame matches; leaves a breadcrumb.

## Tool System

Spectre owns tool execution; SR2 owns tool definition injection.

- Tools are registered in `ToolRegistry` (spectre) and surfaced via
  `SpectreToolProvider` (SR2 pipeline entry point).
- Built-in tools: `terminal`, `file_read`, `file_write`, `edit`, `grep`,
  `glob`, `web_search`, `complete_step`.
- MCP tools are discovered dynamically via MCP client connections.

## Configuration

Spectre uses a 4-tier config merge:
1. `$SR2_HOME/config.yaml` — global defaults
2. `$SR2_HOME/spectre.yaml` — spectre-specific defaults
3. `<cwd>/.spectre.yaml` — project overrides
4. Positional file (e.g., `~/.sr2/agents/edi.yaml`) — wins over all

Config supports `extends:` for inheritance chains. Use `sr2-spectre config show --dry-run` for inspection.

## Key Conventions

- **One step at a time:** Never modify multiple plan tasks in one pass.
- **Verify before advancing:** A `done` task means verified green.
- **Don't skip grounding:** Understand the codebase before writing.
- **Tests are not optional:** Logic side gets unit tests; engine side gets run verification.
- **Decouple logic from engine:** Game rules are plain testable code; engine layer is a thin shell.
"""


def get_sr2_conventions_skill() -> Skill:
    """Return the built-in SR2 conventions skill."""
    return Skill(
        name="sr2-conventions",
        description="How to work with SR2: pipeline, plans, knowledge layers, tools, and conventions.",
        version="0.1.0",
        content=SR2_CONVENTIONS_CONTENT,
        tags=["sr2", "conventions", "pipeline", "planning"],
    )


# ---------------------------------------------------------------------------
# SOLID Review Skill Content
# ---------------------------------------------------------------------------

SOLID_REVIEW_CONTENT = r"""# SOLID Review Skill

> **Purpose:** Audit a scope of code against SOLID, DRY, project mandate
> alignment, test quality, and architectural debt. Produces a scannable
> review with file:line evidence and a prioritized recommendation list.
>
> **Usage:** Pass a diff or file scope to keep the review tractable for
> 27B models. Use `git diff <base>...<head>` for PR reviews, or list
> specific files/modules for targeted audits.

## When to Run This Skill

Run when:

- Asked to "review," "audit," "second pass," "check the architecture"
- A refactor or feature branch is ready for sign-off before merge
- After fixing previously-flagged issues — re-check requested
- Before merging a significant architectural change

Do NOT run when:

- The ask is a diff-level correctness review of a small PR — use
  `/code-review` instead
- The ask is about one specific bug or module — read it and answer
  directly without running the full review framework

## How to Run the Review

### Step 1 — Scope the review

**Always accept a scope argument.** For a 27B model, reviewing an entire
codebase in one pass produces low-signal output.

- **PR review:** `git diff <base>...<head>` — review only changed lines
- **Module review:** `find src/<module> -name "*.py"` — target a module
- **Full audit:** Only when explicitly requested; survey load-bearing files

If no scope is provided, default to the last 20 commits:
`git log --oneline -20` to identify what changed, then scope from there.

### Step 2 — Read the mandates

Before reading source code, read:

- Project's `CLAUDE.md` (and any nested ones)
- Architecture/design docs in `docs/` or `specs/`
- Repo `README.md`

Extract **identity statements** — sentences of the form "X is a Y,"
"X is not a Y," "X owns Z." These are the mandates you'll check code
against. Code violating them is the highest-priority finding.

### Step 3 — Survey the scope

- Read top-level entry points (public API surface within scope)
- Read protocols/interfaces/base classes
- Read the largest 3-5 changed files
- Sample 2-3 implementations of any pluggable extension point

Don't read every file. The signal is in load-bearing files.

### Step 4 — Run the four lenses

Work through them in order. Each finding needs **file:line evidence**.

#### S — Single Responsibility

Look for:

- Methods over ~60 lines doing multiple things
- Classes whose docstring says "two concerns coexist," "also handles"
- Functions taking a `kind`/`mode`/`type` string and switching on it

#### O — Open/Closed

Look for:

- `isinstance` chains on data types — adding a type modifies the method
- `if/elif/else` chains on enum values — adding enum modifies every site
- Hardcoded substring matching as magic-string inference
- Asymmetry: pattern exists (e.g., Strategy) but wasn't applied everywhere

#### L — Liskov

Look for:

- Concrete classes inheriting from `Protocol`
- Subclasses overriding methods to raise `NotImplementedError`
- Subclasses narrowing types or weakening postconditions

#### I — Interface Segregation

Look for:

- Protocols with many methods, most unused by any single client
- Special-case wrapper methods alongside general-purpose ones

#### D — Dependency Inversion

Look for:

- Module-level singletons of concrete classes (implicit globals)
- Hardcoded LLM prompts/model names/API endpoints
- Service locator patterns (untyped `extras: dict[str, Any]`)
- Multiple contradictory paths to inject the same dependency

### Step 5 — DRY violations

Look for:

- Constants defined in two files
- Same preamble logic repeated 3+ times
- Identical factory/build methods with same error messages
- Snapshot/record code duplicated for success and failure paths
- Module-level patterns instantiated identically with same workaround

### Step 6 — Project mandate alignment

For each identity statement from Step 2, find the code supposed to
enforce it. Ask:

- Does the code actually do what the mandate says?
- If "X owns Y" — is Y happening inside X, or is X delegating outward?
- If "X is not a Y" — has X drifted toward Y?

This lens surfaces the **largest architectural gap** because it's
invisible to SOLID/DRY checks.

### Step 7 — Test quality

Look for:

- Test files named after ticket IDs (red flag — pins implementation)
- `assert isinstance(...)` of internal classes
- `assert dataclasses.fields(X)` — testing dataclass shape
- Reach-ins to private attributes (`obj._private`)
- Mocks reaching into internal collaborators

Report test/source LOC ratio. Healthy is 2-3x; above ~3x is suspicious.

### Step 8 — Other gaps

- `asyncio.create_task(...)` with no reference held (orphan tasks)
- Exception handlers that log and swallow
- Dead config fields (declared, never read)
- Missing timeouts on network/IO calls
- `frozen=True` dataclasses with mutable container fields
- Hardcoded magic strings
- `_workaround_field` attributes existing solely for tests

## Output Format

```markdown
# Code review: [Branch / Module Name]

**BLUF**: 2-3 sentences. The biggest finding(s) in plain language.

---

## SOLID

### S — Single Responsibility
- **`File.method()`** (`path:line`) — what's wrong. Fix: one sentence.

### O — Open/Closed
...

### L — Liskov
...

### I — Interface Segregation
...

### D — Dependency Inversion
...

---

## DRY
- **What's duplicated** (`path:line` and `path:line`). Fix: ...

---

## [Project mandate] — biggest architectural gap
The project says: "[exact quote]."
What the code actually does: [observations]
What's missing / what to do: [specific changes]

---

## Tests — implementation-coupled
- **Test files named after ticket IDs**: [list]
- **Private-attribute reach-ins**: [count]
- **Test/source LOC ratio**: [N]x

---

## Other gaps
- **`File.thing`** (`path:line`) — description. Fix: ...

---

## What I'd prioritize
1. [Highest-impact change — often the mandate-violation]
2. [Largest structural cleanup]
3. [Tests / DRY cleanup that unblocks other refactors]
```

## Anti-Patterns to Avoid

- **Don't pad sections with weak findings.** If LSP has nothing real,
  leave it short. Signal over coverage.
- **Don't propose a refactor without file:line.** Concrete evidence only.
- **Don't propose abstractions that don't pay rent.** If the fix adds
  more complexity than it removes, say so and de-prioritize.
- **Don't lead with style nits.** Structural findings only.
- **Don't end without a priority list.** Always rank.
- **Don't review files you haven't read.** Pattern-matching from
  filenames is a failure mode.

## Calibration Notes

- The user responds well to BLUF + scannable sections + concrete
  file:line refs. Walls of text get skimmed.
- The **project-mandate-violation finding** is often the highest-leverage
  diagnostic.
- Findings that say "do X, here's the file:line" reduce cognitive load.
- For PR reviews scoped to a diff, focus SOLID/DRY on changed code only.
  Mandate alignment and test quality may still reference broader context.
"""


def get_solid_review_skill() -> Skill:
    """Return the built-in solid-review skill."""
    return Skill(
        name="solid-review",
        description="Audit code against SOLID, DRY, project mandate alignment, test quality, and architectural debt. Accepts a diff or file scope for tractable reviews.",
        version="0.1.0",
        content=SOLID_REVIEW_CONTENT,
        tags=["review", "solid", "dry", "architecture", "audit"],
    )


# Default skill instances for registry bootstrap
DEFAULT_SKILLS = [
    get_sr2_conventions_skill(),
    get_solid_review_skill(),
]
