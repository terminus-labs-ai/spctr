# Continuity Backend Selection — Bakeoff Status (sr2-100 / AC19)

**Verdict of record: NO-GO (2026-10-22 run, build/sr2-100 `f2de3af`). No
candidate qualified as the continuity default.** The lexicographic rule of
FR21 eliminated every candidate at stage 2 under both readings of that
clause. Amendment 2026-11-17 (sr2-178, owner rulings adopted in full) keeps
that verdict on record and rules that the *re-run* — not a re-adjudication —
answers the selection question against the amended corpus.

**Status (2026-11-18, sr2-184):** the four measurement defects the original
run found in its own harness are repaired and merged to `main` (sr2-177,
sr2-179, sr2-181; scored-set parity sr2-180 still open). The per-arm numbers
below are therefore annotated: where a repair moved them, the old number is
a reading of a broken instrument, not of the backend. The re-run slice
(sr2-182) regenerates the machine-readable report on the repaired harness
and the amended FR44 corpus and replaces this status section.

The original verdict document with every raw number lives on the delivery
branch — `packages/sr2/docs/evals/continuity-backend-selection.md` at
`build/sr2-100` (`f2de3af`) — together with its raw artifacts
(`results/backend-bakeoff.json`, per-run `observations.jsonl`), which stay
there as historical evidence. This document is the current state of AC19 on
`main`.

## What ran

Five arms over the fixed versioned corpus (`evals.continuity.scenarios.CORPUS`,
12 scenarios) with the budgets in that file untouched:

| Arm | Path | Delivered background |
|---|---|---|
| `baseline` | `run_baseline` | current full-history behavior (`seed_session` + `turn`) |
| `structured` | `run_candidate("structured")` | `StructuredIndex` |
| `vector_hybrid` | `run_candidate("vector_hybrid")` | `VectorHybridIndex` |
| `graph` | `run_candidate("graph")` | `GraphIndex` |
| `embedding` | `run_candidate("embedding")` | `EmbeddingContinuityIndex` over the committed FR40 vectors |

Every arm uses the deterministic extraction double (no model call), the pinned
price table, and the Metric Contract's latency constant. Each scenario runs
five times. All nine metrics plus `closed_residency` are published per turn.

## Which numbers the gate reads

Both arms publish aggregates, but the two published aggregates are **not**
comparable, and feeding them to the gate would score a scoring-set difference
(sr2-180, still open): the baseline sweep scores only gold-bearing turns
(`gold_turns`), while a candidate run publishes **every** FR40 turn —
including the six turns that declare no gold — and an unscored turn is 0 on
correctness, and on `recall` for the turns before A's decision re-enters the
request. The gate therefore reads the **gold-turn subset**: both arms
restricted to the corpus's gold-bearing retrieval turns, keyed by position
inside `gold_turns(scenario)` (`_canonical_rows`). Same inputs, same budgets,
same scored set — 65 scored rows in every arm.

The pre-repair gold-subset table (from `f2de3af`, with the repairs' effect
annotated per row):

| Arm | correctness | recall | stale | leak | irr. context | background tokens |
|---|---|---|---|---|---|---|
| baseline | 0.9846 | 1.0000 | 1 | 1 | 0.5726 | 7425 |
| structured | 0.9385 † | 0.9538 † | 1 ‡ | 1 | 0.5300 † | 7467 † |
| vector_hybrid | 0.9385 † | 0.9538 † | 1 ‡ | 1 | 0.5300 † | 7467 † |
| graph | 0.9846 † | 1.0000 † | 1 ‡ | 1 | 0.5300 † | 7476 † |
| embedding | 0.9385 † | 0.9538 † | 1→0 ‡ | 1 | 0.5300 † | 7467 † |

**The †-marked candidate numbers are invalidated by the merged repairs:**
they mix a scoring-set artifact (sr2-180, unrepaired — the same normalization
applies to the re-run) with broken-delivery measurements. **‡-marked counts
move concretely**: after the repairs, `embedding` resolves the composition
control slot through its own retrieval and no longer replays the pinned stale
value (`stale_current` 1 → 0); `structured`, `vector_hybrid` and `graph` keep
the planted control-plan counts (1 stale, 1 policy-leak) because under the
pressure-free corpus their index retrieves everything within budget, exactly
like the baseline. The baseline row is untouched: no repair touches
`run_baseline`'s delivery, and the frozen AC16 baseline cage passes
byte-identical on the merged tree.

## Raw per-scenario results (correctness / recall, pre-repair)

| Scenario | baseline | structured | vector_hybrid | graph | embedding |
|---|---|---|---|---|---|
| bounded-context-endurance-500-turn | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |
| cage-composition | 0.900 / 1.000 | 0.900 † | 0.900 † | 0.900 † | 0.900 ‡ |
| continuity-service-failure | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |
| correction-supersession | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |
| cross-project-dependency | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |
| episode_close_paraphrase_return | 1.000 / 1.000 | 0.400 † | 0.400 † | 1.000 † | 0.400 † |
| long-topic-switching-and-return | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |
| per-agent-policy-isolation | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |
| personal-preference-across-projects | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |
| project-resumption-intervening-live-changes | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |
| relevant-episodic-recovery | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |
| restart | 1.000 / 1.000 | 1.000 † | 1.000 † | 1.000 † | 1.000 † |

† After sr2-179 the candidate arms deliver these scenarios through their own
index; on the pressure-free corpus `structured`, `vector_hybrid` and `graph`
retrieve everything within budget, so their flags stay numerically equal to
the baseline — equality now means "retrieval returned everything" rather than
"nothing was retrieved", and it holds until the re-run's cap-overflow
scenario (FR44, sr2-182) gives the index work to do. The `embedding` arm's
`cage-composition` row (‡) is where the repaired routing first discriminates:
its offline hash-proxy retrieval misses the pinned stale block, flipping that
arm's stale read without touching any other row. The FR40 row's candidate
values move most: after sr2-177 every repetition delivers all six distinct
records (118 background tokens each, repetitions 1–5), so all four candidate
arms now read 0.143 / 0.286 correctness / recall per repetition — equal to
each other, and far below the pre-repair `0.400` (structured / vector_hybrid /
embedding, an artifact of index crowding leaving only early repetitions able
to retrieve A) and `1.000` (graph, an artifact of which duplicated versions
survived the crowding). Pre-repair, no repetition past the first was a clean
read for any candidate.

## The gate applied (FR21, in order, pre-repair)

| Stage | Result |
|---|---|
| 1 · disqualify policy leaks / stale-over-current | no candidate above baseline (both sit at the corpus's planted 1 and 1) |
| 2 · require improved correctness/recall over baseline | **all four fail** — three regress, one ties |
| 3 · minimize irrelevant context / hallucination | not reached |
| 4 · minimize token cost | not reached |
| 5 · minimize latency | not reached |

Amendment 2026-11-17 ruled stage 2's predicate: no regression in correctness
or recall versus the full-history baseline **plus** a strict improvement in
at least one (AC21 aligned to the same reading). Both readings were evaluated
in the original run with the same `no_go` outcome, so the ruling changes
nothing about the verdict recorded here — it fixes how the re-run will
adjudicate.

## Why no candidate could beat the original corpus

Two independent structural reasons, both still true of the twelve-scenario
corpus and both addressed by the re-run:

1. **The baseline sits at the ceiling on ten of twelve scenarios.** Full
   history delivers every declared block, so recall is 1.000 by construction.
   A bounded backend cannot improve a maximum; it can only tie it while
   spending budget to find the material again.
2. **Only one scenario consulted a candidate index at all (pre-repair).** On
   the ten FR31 scenarios and the cage fixture, the candidate run delivered
   background through the same full-history path as the baseline
   (`run_scenario_turn` never received the index) — provable from the
   artifacts: those 60 observations were byte-identical to the baseline's in
   all four candidate runs. sr2-179 makes the routing real; the corpus
   itself still contains no condition in which bounded retrieval could
   plausibly *beat* replaying everything (no cap overflow, no
   retrieval-difficulty gradient), which is exactly what FR44 adds in the
   re-run (sr2-182).

## Measurement repairs (merged to `main`, 2026-11-18, sr2-184)

| Defect found in `f2de3af` | Repair | Effect on the numbers above |
|---|---|---|
| Candidate index unused on 11 of 12 scenarios | sr2-179 (`94effe8`): FR31-scenario and cage-composition sweep background routed through the candidate's own index (eligible-content rule shared verbatim with the baseline; ranked delivery cut at budget; `request_text` published; offline deterministic embedding provider for the embedding arm) | † rows: candidate sweeps now genuinely use the backend; struct/vector/graph stay flag-equal to baseline only because the pressure-free corpus lets them retrieve everything; embedding flips the pinned stale read (1 → 0) |
| Shared index state across repetitions | sr2-177 (`5e31e01`): `_run_fr40_scenario` builds one fresh candidate index per repetition (embedding cache stays warm via the workdir-fixed vector store and a shared query cache, so row r7's exactly-once property holds) | FR40 row: all four candidate arms read 0.143 / 0.286 per repetition instead of run-index-dependent 0.400 / 1.000 |
| Latency carries no signal | sr2-181 (`e740324`): measured wall clock publishes as `latency.json` beside `metrics.json` in every run dir (per-scenario median/p95 + pooled aggregate, warm-up discarded), excluded from the R18 byte-identity proof; `load_latency()` feeds FR21's final stage | Stage 5 becomes rankable for the first time; the pinned placeholder remains in `metrics.json` |
| Scoring-set mismatch between arms | **sr2-180 — still open**; report-time normalization (`_canonical_rows` / gold-turn subset) stands until it lands | the †/‡ annotation split stays in force for the re-run until sr2-180 merges |

Each repair shipped RED-on-base / GREEN-on-fix cages
(`test_continuity_fr40_repetition_isolation.py`,
`test_continuity_sweep_index_delivery.py`,
`test_continuity_latency_publication.py`), all passing on the integrated
tree, with the frozen AC16 / AC27 cages unchanged.

## Reproduce

```bash
cd packages/sr2
PYTHONPATH=.:src python -m evals.continuity.runner            # baseline only
PYTHONPATH=.:src python -m pytest tests/evals/test_continuity_baseline.py \
    tests/evals/test_continuity_paraphrase.py \
    tests/evals/test_continuity_runner_retention.py \
    tests/evals/test_continuity_fr40_repetition_isolation.py \
    tests/evals/test_continuity_sweep_index_delivery.py \
    tests/evals/test_continuity_latency_publication.py
```

`run_bakeoff` (which writes `results/backend-bakeoff.json` and asserted
two-call byte identity) lives on `build/sr2-100` (`f2de3af`) and lands with
the re-run (sr2-182), which also ships the FR44 cap-overflow scenario and the
amended FR21 predicate. Two consecutive `run_bakeoff()` calls must produce
byte-identical JSON; measured latency publishes separately as `latency.json`
and is outside the byte-identity proof.

## What the re-run (sr2-182) answers

1. Whether any candidate passes the ruled stage-2 predicate **against the
   full-history baseline** once the FR44 cap-overflow scenario puts real
   pressure on the baseline and the repaired routing gives every candidate
   genuine selection work on every scenario.
2. With measured `latency.json` in place, whether stage 5's tie-breaker can
   finally rank — or records an unresolved tie.

No `configs/*.yaml` change ships with this status: with no selected backend,
there is no continuity default to write.
