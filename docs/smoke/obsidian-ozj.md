# Smoke Test Runbook — obsidian-ozj: Base freshness checks (rebase-before-PR + stale-base reject)

**Proves:** Author workflow rebases before pushing PR; reviewer workflow checks base freshness before running tests. **Does NOT cover:** live cron run with actual gh PR (that requires authed gh CLI and a real repo state).

## Setup

```bash
cd ~/git/sr2-spectre
```

## Scenarios

### 1. Unit: Freshness check logic (FRESH verdict)

```bash
uv run pytest tests/test_pr_review_freshness.py -k "TestCheckFreshness::test_fresh" -q
```

**Expect:** `2 passed` — branch not behind main returns FRESH verdict.

### 2. Unit: Freshness check logic (STALE verdict)

```bash
uv run pytest tests/test_pr_review_freshness.py -k "TestCheckFreshness::test_stale" -q
```

**Expect:** `2 passed` — branch behind main returns STALE with rebase suggestion.

### 3. Unit: Author rebase check

```bash
uv run pytest tests/test_pr_review_freshness.py -k "TestAuthorRebaseCheck" -q
```

**Expect:** `5 passed` — returns None when fresh, instruction when stale, respects strategy.

### 4. Unit: Dataclass immutability

```bash
uv run pytest tests/test_pr_review_freshness.py -k "frozen" -q
```

**Expect:** `2 passed` — FreshnessResult is frozen.

### 5. Unit: Enum values

```bash
uv run pytest tests/test_pr_review_freshness.py -k "TestRebaseStrategy\|TestFreshnessVerdict" -q
```

**Expect:** `2 passed` — enum values match expected strings.

### 6. Integration: Squadron rules — author rebase step

```bash
uv run pytest tests/test_squadron_rules.py -k "rebase" -q
```

**Expect:** `2 passed` — author workflow includes rebase step + blocks on conflict.

### 7. Integration: Squadron rules — reviewer base freshness

```bash
uv run pytest tests/test_squadron_rules.py -k "freshness\|stale" -q
```

**Expect:** `2 passed` — reviewer workflow includes base freshness check + rejects stale branches.

### 8. Full PR review test suite (regression)

```bash
uv run pytest tests/test_pr_review*.py tests/test_squadron_rules.py -q
```

**Expect:** `77 passed` — all PR review + squadron rules tests pass.

### 9. Full squadron rules test suite

```bash
uv run pytest tests/test_squadron_rules.py -q
```

**Expect:** `13 passed` — all squadron rules tests pass.

## Teardown

None — no state changes beyond file edits.

## Pass Criteria

| # | Scenario | Pass |
|---|----------|------|
| 1 | Freshness FRESH verdict | [ ] |
| 2 | Freshness STALE verdict | [ ] |
| 3 | Author rebase check | [ ] |
| 4 | Dataclass immutability | [ ] |
| 5 | Enum values | [ ] |
| 6 | Squadron rules: author rebase | [ ] |
| 7 | Squadron rules: reviewer freshness | [ ] |
| 8 | Full PR review suite | [ ] |
| 9 | Full squadron rules suite | [ ] |

## Manual Verification (squadron-rules.md)

Open `~/.sr2/squadron-rules.md` and confirm:

1. **Author workflow step 5** reads "Rebase onto main" with `git fetch origin && git rebase origin/main` and bead-blocking on conflict.
2. **Reviewer workflow step 4** reads "Check base freshness" with `git rev-list --count HEAD..origin/main` and REJECT path for stale branches.
3. Step numbering is sequential in both workflows (no gaps or duplicates).
