"""sr2-100 / AC19 — backend bakeoff driver: gate logic, normalization, report.

Written by the sr2-100 analysis executor (no role-A cage exists for the
bakeoff; ``tests/evals/test_continuity_baseline.py`` and
``test_continuity_paraphrase.py`` pin the baseline and the FR40 candidate
path, which this module consumes unchanged). What is pinned here is the part
this bead owns: the lexicographic gate of FR21 applied in order, the
normalization that makes the two arms' scored sets identical, and the shape
plus reproducibility of the published ``backend-bakeoff.json``.

The metric *values* are not asserted (the corpus's numbers belong to the
baseline/candidate cages and to the committed report); what is asserted is
that a report exists, covers every arm and every corpus scenario with raw
per-scenario metrics, applies the gate, and names either a selected default
or ``no_go`` — which is exactly AC19's text.
"""

from __future__ import annotations

import json

import pytest
from evals.continuity import runner as crunner
from evals.continuity.candidates import CANDIDATES
from evals.continuity.scenarios import CORPUS, EPISODE_CLOSE_PARAPHRASE_RETURN

FR40 = CORPUS[EPISODE_CLOSE_PARAPHRASE_RETURN]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _totals(**overrides) -> dict:
    """A gold-subset totals row shaped like ``_subset_metrics`` output."""
    row = {
        "correctness": 0.5,
        "recall": 0.5,
        "stale_current": 0,
        "policy_leak": 0,
        "irrelevant_context": 0.4,
        "hallucination": 0,
        "prompt_background_tokens": {"prompt": 100, "background": 200},
        "cost": 0.1,
        "latency": {"median": 0.02, "p95": 0.03},
        "n_observations": 5,
    }
    row.update(overrides)
    return row


def _candidate_rows(turns_per_run: int) -> list[dict]:
    """Synthetic absolute-index observation rows: ``turns_per_run`` turns in
    each of the ``RUNS_PER_SCENARIO`` repetitions of the FR40 scenario."""
    out: list[dict] = []
    for run_index in range(1, crunner.RUNS_PER_SCENARIO + 1):
        for turn_index in range(1, turns_per_run + 1):
            out.append({
                "scenario_id": FR40.id,
                "run_index": run_index,
                "turn_index": turn_index,
                "prompt_tokens": 10,
                "background_tokens": 20,
                "response": "decision: Postgres",
                "flags": {"correct": 1.0, "recall": 1.0, "stale_current": 0,
                          "policy_leak": 0, "irrelevant_context": 0.0,
                          "hallucination": 0},
            })
    return out


# ---------------------------------------------------------------------------
# gate: lexicographic order, both qualification readings
# ---------------------------------------------------------------------------


def test_gate_selects_candidate_that_improves_recall_without_regression():
    verdict = crunner.evaluate_gate(
        _totals(correctness=0.8, recall=0.8),
        {"vector_hybrid": _totals(correctness=0.8, recall=0.9)},
    )
    assert verdict["status"] == "selected"
    assert verdict["selected_default"] == "vector_hybrid"
    assert verdict["deciding_stage"] == crunner.GATE_STAGES[1]
    outcome = verdict["candidates"]["vector_hybrid"]
    assert outcome["qualified_on_no_regression_reading"] is True
    assert outcome["qualified_on_strict_both_reading"] is False


def test_gate_requires_improvement_so_a_tie_is_no_go():
    base = _totals(correctness=0.98, recall=1.0)
    verdict = crunner.evaluate_gate(base, {"graph": dict(base)})
    assert verdict["status"] == "no_go"
    assert verdict["selected_default"] is None
    assert verdict["survivors"] == []
    assert verdict["deciding_stage"] == crunner.GATE_STAGES[1]


def test_gate_disqualifies_extra_policy_leak_before_scoring():
    verdict = crunner.evaluate_gate(
        _totals(correctness=0.5, recall=0.5, policy_leak=1),
        {"embedding": _totals(correctness=1.0, recall=1.0, policy_leak=2)},
    )
    outcome = verdict["candidates"]["embedding"]
    assert outcome["disqualified_by"] == ["policy_leak: 2 > baseline 1"]
    assert outcome["stage"] == crunner.GATE_STAGES[0]
    assert verdict["status"] == "no_go"


def test_gate_disqualifies_extra_stale_over_current_errors():
    verdict = crunner.evaluate_gate(
        _totals(correctness=0.5, recall=0.5, stale_current=1),
        {"graph": _totals(correctness=1.0, recall=1.0, stale_current=2)},
    )
    outcome = verdict["candidates"]["graph"]
    assert outcome["disqualified_by"] == ["stale_current: 2 > baseline 1"]
    assert verdict["status"] == "no_go"


def test_gate_eliminates_regression_in_one_quality_metric_even_when_other_improves():
    verdict = crunner.evaluate_gate(
        _totals(correctness=0.9, recall=0.9),
        {"structured": _totals(correctness=0.8, recall=1.0)},
    )
    outcome = verdict["candidates"]["structured"]
    assert outcome["delta_correctness"] < 0
    assert outcome["qualified_on_no_regression_reading"] is False
    assert verdict["status"] == "no_go"


def test_gate_ranks_survivors_irrelevant_context_then_cost_then_latency():
    base = _totals(correctness=0.5, recall=0.5)
    cheap_but_noisy = _totals(correctness=0.6, recall=0.5,
                              irrelevant_context=0.30, cost=0.9)
    tidy_but_pricier = _totals(correctness=0.6, recall=0.5,
                               irrelevant_context=0.20, cost=0.50)
    same_as_tidy_cheaper = _totals(correctness=0.6, recall=0.5,
                                   irrelevant_context=0.20, cost=0.40)
    verdict = crunner.evaluate_gate(base, {
        "a_cheap_noisy": cheap_but_noisy,
        "b_tidy_pricey": tidy_but_pricier,
        "c_tidy_cheap": same_as_tidy_cheaper,
    })
    assert verdict["status"] == "selected"
    # Irrelevant context outranks cost, so both tidy candidates beat the
    # cheap noisy one; among them cost decides.
    assert verdict["selected_default"] == "c_tidy_cheap"
    assert verdict["survivors"].index("a_cheap_noisy") == 2


def test_gate_breaks_final_tie_on_latency():
    base = _totals(correctness=0.5, recall=0.5)
    shared = {"correctness": 0.7, "recall": 0.5, "irrelevant_context": 0.1,
              "cost": 0.1}
    verdict = crunner.evaluate_gate(base, {
        "slow": _totals(**shared, latency={"median": 0.09, "p95": 0.12}),
        "fast": _totals(**shared, latency={"median": 0.01, "p95": 0.02}),
    })
    assert verdict["selected_default"] == "fast"


# ---------------------------------------------------------------------------
# normalization: the two arms' turn-index spaces
# ---------------------------------------------------------------------------


def test_canonical_rows_keeps_only_gold_turns_from_absolute_indices():
    rows = _candidate_rows(len(FR40.turns))
    canonical = crunner._canonical_rows(FR40, rows)
    gold_abs = [i for i, t in enumerate(FR40.turns, start=1)
                if tuple(t.gold_answers)]
    assert len(canonical) == crunner.RUNS_PER_SCENARIO * len(gold_abs)
    assert {row["gold_position"] for row in canonical} == {
        i for i in range(1, len(gold_abs) + 1)}
    assert all(row["turn_index"] in gold_abs for row in canonical)


def test_canonical_rows_agrees_between_positional_and_absolute_arms():
    absolute = _candidate_rows(len(FR40.turns))
    positional = [dict(row, turn_index=1) for row in
                  _candidate_rows(1)]
    canon_absolute = crunner._canonical_rows(FR40, absolute)
    canon_positional = crunner._canonical_rows(FR40, positional)
    keys = lambda rows: sorted((r["run_index"], r["gold_position"])
                               for r in rows)
    assert keys(canon_absolute) == keys(canon_positional)


def test_canonical_rows_rejects_unknown_shape():
    rows = _candidate_rows(3)
    with pytest.raises(AssertionError):
        crunner._canonical_rows(FR40, rows)


# ---------------------------------------------------------------------------
# publication: shape, coverage, determinism, guard rails
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def report(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("bakeoff")
    path = crunner.run_bakeoff(root)
    return json.loads(path.read_text())


def test_report_covers_every_arm_and_scenario_with_raw_metrics(report):
    assert set(report["inputs"]["arms"]) == {crunner.BASELINE_ARM,
                                             *sorted(CANDIDATES)}
    assert set(report["inputs"]["scenario_ids"]) == set(CORPUS)
    assert set(report["per_scenario_gold_subset"]) == set(CORPUS)
    for scenario_id, table in report["per_scenario_gold_subset"].items():
        assert set(table) == set(report["inputs"]["arms"])
        for arm, row in table.items():
            for key in ("correctness", "recall", "stale_current",
                        "policy_leak", "irrelevant_context", "hallucination",
                        "prompt_background_tokens", "cost", "latency",
                        "n_observations"):
                assert key in row, f"{scenario_id}/{arm} missing {key}"
        # Identical budgets, identical scored set across arms.
        counts = {row["n_observations"] for row in table.values()}
        assert len(counts) == 1, f"{scenario_id}: scored sets differ: {counts}"


def test_report_publishes_each_arm_verbatim_and_the_gate(report):
    for arm, published in report["published_metrics"].items():
        assert "correctness" in published and "cost" in published
        assert report["inputs"]["run_ids"][arm]
    gate = report["gate"]
    assert set(gate["candidates"]) == set(CANDIDATES)
    for outcome in gate["candidates"].values():
        assert outcome["stage"] in crunner.GATE_STAGES


def test_report_names_a_selected_default_or_concludes_no_go(report):
    verdict = report["verdict"]
    assert verdict["status"] in {"selected", "no_go"}
    if verdict["status"] == "no_go":
        assert verdict["selected_default"] is None
        assert report["gate"]["survivors"] == []
    else:
        assert verdict["selected_default"] in CANDIDATES
    assert report["limitations"]


def test_bakeoff_report_reproduces_byte_for_byte(tmp_path):
    first = crunner.run_bakeoff(tmp_path / "a")
    second = crunner.run_bakeoff(tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()


def test_partial_bakeoff_cannot_overwrite_the_published_report():
    with pytest.raises(ValueError):
        crunner.run_bakeoff(arms=["structured"])


def test_unknown_arm_or_scenario_is_refused(tmp_path):
    with pytest.raises(KeyError):
        crunner.run_bakeoff(tmp_path, arms=["nope"])
    with pytest.raises(KeyError):
        crunner.run_bakeoff(tmp_path, scenarios=["nope"])


def test_main_dispatches_the_bakeoff_subcommand(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["runner", "bakeoff",
                                     str(tmp_path / "via-main")])
    path = crunner.main(["bakeoff", str(tmp_path / "via-main")])
    assert path.name == crunner.BAKEOFF_FILENAME
    assert path.read_text() == (
        crunner.run_bakeoff(tmp_path / "second").read_text())
