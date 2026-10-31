# Planning Guide — Multi-Step Task Protocol

> **Trigger:** For multi-step tasks, follow this guide before executing.
> Load this guide via `file_read` when you judge a task requires decomposition.

## Workflow

### 1. Ground

Read the affected code and understand the codebase. Identify:
- Which files, modules, and classes are involved
- Public APIs and entry points (`__init__` exports, CLI flags, etc.)
- Existing patterns (protocols, registries, conventions)
- Dependencies and coupling between components

### 2. Record Contract

Before writing anything, document the **Understanding & Constraints** — the
patterns, boundaries, and rules this change must preserve:
- SOLID principles, module boundaries
- Public API surface (what external code depends on)
- Architecture mandates (dependency direction, protocol usage)
- Style conventions (naming, docstrings, test patterns)

This contract goes into the `_plan.md` body and will be re-checked at final
validation.

### 3. Write Plan

Create a plan directory under `~/.sr2/plans/<plan-slug>/` with:

**`_plan.md`** (plan-shared file):
```yaml
---
kind: plan
slug: <plan-slug>
status: open
goal: "One-line description of what this plan achieves"
---

## Understanding & Constraints

[The contract from step 2 — patterns and boundaries to preserve]
```

**`NN-slug.md`** (one per atomic task, e.g., `01-dir-move.md`):
```yaml
---
kind: task
plan: <plan-slug>
order: 1
status: pending
verify: "uv run pytest tests/test_foo.py"
title: "Short label for this step"
---

[Detailed description of what this step does, which files to touch,
what verification means]
```

**Rules:**
- Each task must be **atomic** — one coherent unit of work verifiable in isolation.
- Each task must have a `verify:` command or description.
- Tasks are ordered by `order:` (1, 2, 3, …).
- A task is complete only after its `verify:` passes.

### 4. Execute Current Task

Work on the current task (the one injected as "Current Task" in your context).
Use your existing tools (`terminal`, `file_write`, `edit`, `file_read`, `grep`,
`glob`) — no special tools needed.

### 5. Verify

Run the task's `verify:` command. If it passes:
- Edit the task file's frontmatter: `status: pending` → `status: done`
- The next turn will automatically advance to the next pending task

If it fails, fix the issue and re-run verification. Do **not** advance until
green.

### 6. Repeat

Continue executing → verify → advance until all tasks are `done`.

### 7. Final Validation

When the last task is complete:
1. Run the full test suite (or the broadest verification command)
2. Re-check each constraint recorded in the `_plan.md` contract
3. Set `_plan.md` frontmatter: `status: open` → `status: done`

## Step-Sizing Discipline

> **Why this matters:** A single step must complete within the tool-loop budget
> (`max_tool_rounds: 40` by default). A step that bundles too many sub-tasks
> will blow the budget mid-execution, leaving the repo in an inconsistent state
> with no verification run. This has happened.

### Budget awareness

- **Target:** design each step to complete in **~20–25 tool calls** with room
  for debugging. If a step needs 30+ tool calls just for the happy path, split
  it.
- **Rule of thumb:** if you're tempted to say "while I'm here, also…" that's
  a signal the step is too big. File it as the next step instead.

### What counts as ONE step (atomic unit)

A single step should do **one coherent action**:
- ✅ Rename one protocol + update one consumer module
- ✅ Add one new class with tests
- ✅ Delete one dead-code class (no consumers)
- ✅ Update imports in one directory

A step that bundles **multiple concerns** is too big:
- ❌ Rename a protocol + fold a second protocol + delete two classes + directory
  rename + update all imports across the codebase (this was the failure mode in
  the wxd.5 probe — step-01 bundled 5 concerns, hit the tool limit at ~60%)
- ❌ Refactor core logic + rewrite tests + update CLI (three concerns)
- ❌ Add a new module + wire it into three existing consumers

### Intra-step checkpoints

For steps that genuinely involve multiple sub-actions within a single concern,
add **intermediate verification points** inside the task description:

```markdown
---
kind: task
plan: rename-plugin-to-interface
order: 1
status: pending
title: "Rename Plugin Protocol to Interface"
---

Rename `Plugin` Protocol → `Interface` in `interfaces/__init__.py`.

**Checkpoints:**
1. After renaming the class: verify `grep -r "class Plugin" src/` returns nothing
   unexpected (only `Interface` remains).
2. After updating imports in `cli.py`: verify `python -c "from sr2_spectre.cli import main"` succeeds.
3. After updating `__init__.py` exports: verify `python -c "from sr2_spectre.interfaces import Interface"` succeeds.
4. FINAL: `uv run pytest` — full suite green.
```

This way, if the tool limit hits mid-step, the checkpoint state shows exactly
how far you got and what's left — instead of a half-renamed codebase with no
signal.

## Important Rules

- **Never modify multiple tasks in one pass.** Work one step at a time.
- **Always verify before advancing.** A `done` task means verified.
- **Don't skip grounding.** Understanding the codebase first prevents rework.
- **Don't silently resume a stale plan.** If an open plan's `goal:` doesn't
  match the current task, surface the mismatch instead of proceeding.
- **Externalize findings before completing.** Before calling `complete_step`
  (or flipping `status: done`), append any cross-step discovery to
  `_findings.md` in the plan directory. These survive context compaction and
  are re-injected by the resolver on every turn.
