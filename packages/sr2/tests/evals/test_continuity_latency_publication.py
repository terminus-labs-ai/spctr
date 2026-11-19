"""sr2-181 — measured-latency publication (rows L1-L8).

Amendment 2026-11-17 (sr2-178, owner ruling D3) took the measure branch
for FR21's final tie-breaker: wall-clock latency publishes as
``latency.json`` beside ``metrics.json`` in every published run dir,
excluded from the R18 byte-identity reproducibility proof, which covers
``metrics.json`` only — the pinned placeholder constants stay there.
FR21's final stage ranks arms by the document's median where it exists
and records an unresolved tie where it does not.

Rows (all on ``build/sr2-181``; RED on base origin/main 5b06e97):

L1  ``run_baseline`` publishes ``latency.json`` beside ``metrics.json``.
L2  Schema and coverage: top-level keys exactly ``aggregate`` and
    ``scenarios``; one entry per corpus scenario; entries carry numeric
    ``median``/``p95`` with ``0 < median <= p95``.
L3  Proof separation: two consecutive runs keep ``metrics.json`` byte-
    identical with the pinned latency placeholder, while each run also
    publishes a parseable ``latency.json`` (the measured bytes never
    enter the byte-identity check).
L4  Pure statistics: ``build_latency_document`` reports per-scenario
    ``latency_stats`` semantics (run 1 discarded as warm-up) and pools
    scored timings across scenarios, run-index-aligned, into
    ``aggregate`` — hand-computed expected values.
L5  Degenerate documents serialize: warm-up-only and empty inputs yield
    ``null`` entries (never NaN) under ``json.dumps(..., allow_nan=False)``.
L6  ``load_latency``: parsed document when present, ``None`` for pre-
    amendment run dirs (no artifact) and missing directories.
L7  ``run_scenario_candidate`` (non-FR40 sweep path) publishes a
    schema-valid ``latency.json`` covering the executed scenario.
L8  ``run_candidate`` (full corpus incl. FR40) publishes a schema-valid
    ``latency.json`` whose scenarios map covers the whole corpus and
    whose FR40 entry includes the candidate-index delivery timings.

Free: absolute wall-clock magnitudes (timers are machine-dependent);
the pinned placeholder values themselves (caged by AC16 row R17).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.continuity import models as cmodels
from evals.continuity import runner as crunner
from evals.continuity import scenarios as cscenarios

REPO_ROOT = Path(__file__).resolve().parents[2]

CURRENT = "backend: PostgreSQL"
SUPERSEDED = "backend: SQLite"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entries_ok(entry: dict, label: str) -> None:
    assert isinstance(entry, dict), f"L2: {label} must be an object"
    assert {"median", "p95"} <= set(entry), (
        f"L2: {label} must carry median and p95; got {sorted(entry)}")
    for key in ("median", "p95"):
        value = entry[key]
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"L2: {label}.{key} must be numeric, measured seconds; got {value!r}")
    assert 0 < entry["median"] <= entry["p95"], (
        f"L2: {label} must satisfy 0 < median <= p95; got {entry!r}")


def _latency_shape(run_dir: Path, expected_scenarios: set[str]) -> dict:
    path = run_dir / "latency.json"
    assert path.is_file(), (
        f"amended Metric Contract: run dir {run_dir.name} must publish "
        "latency.json beside metrics.json")
    document = json.loads(path.read_text())
    assert isinstance(document, dict), "L2: latency.json must be a JSON object"
    assert set(document) == {"aggregate", "scenarios"}, (
        f"L2: top-level keys must be exactly aggregate+scenarios; got "
        f"{sorted(document)}")
    assert set(document["scenarios"]) == expected_scenarios, (
        "L2: scenarios map must cover every corpus scenario executed by "
        "the run")
    for sid, entry in document["scenarios"].items():
        _entries_ok(entry, f"scenarios[{sid!r}]")
    _entries_ok(document["aggregate"], "aggregate")
    return document


def _mini_scenario() -> cmodels.Scenario:
    """A two-turn composition-shaped scenario for the candidate sweep
    path (row L7): small, deterministic, no FR40 machinery required."""
    blocks = (
        cmodels.Block(text=SUPERSEDED, layer="continuity_replay"),
        cmodels.Block(text=CURRENT, layer="continuity_replay"),
    )
    turns = tuple(
        cmodels.ScenarioTurn(
            gold_answers=(CURRENT,),
            current_values=(CURRENT,),
            superseded_values=(SUPERSEDED,),
            blocks=blocks,
            delivered=None if i == 1 else (CURRENT,),
        )
        for i in (1, 2)
    )
    return cmodels.Scenario(
        id="sr2-181-mini", agent_policy="unrestricted",
        runner_mode="full_history", token_budget=4000,
        prompt_budget=1000, turns=turns)


# ---------------------------------------------------------------------------
# Module-scoped baseline double run (L1/L2/L3 share it; run_baseline is
# expensive through the pipeline).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_pair(tmp_path_factory) -> tuple[Path, Path]:
    root_a = tmp_path_factory.mktemp("lat-a")
    root_b = tmp_path_factory.mktemp("lat-b")
    dir_a = crunner.run_baseline(output_root=root_a)
    dir_b = crunner.run_baseline(output_root=root_b)
    return dir_a, dir_b


# ===========================================================================
# L1 — the baseline publishes latency.json beside metrics.json.
# ===========================================================================


def test_l1_baseline_publishes_latency_json(baseline_pair):
    run_dir, _ = baseline_pair
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "latency.json").is_file(), (
        "L1: Amendment 2026-11-17 (sr2-178/sr2-181): the baseline run must "
        "publish measured latency.json beside metrics.json")


# ===========================================================================
# L2 — schema and corpus coverage.
# ===========================================================================


def test_l2_latency_document_schema_and_coverage(baseline_pair):
    run_dir, _ = baseline_pair
    _latency_shape(run_dir, set(cscenarios.CORPUS))


# ===========================================================================
# L3 — byte-identity proof untouched: metrics.json identical with the
#      pinned placeholder while each run publishes measured latency.
# ===========================================================================


def test_l3_metrics_stay_pinned_and_byte_identical(baseline_pair):
    dir_a, dir_b = baseline_pair
    bytes_a = (dir_a / "metrics.json").read_bytes()
    bytes_b = (dir_b / "metrics.json").read_bytes()
    assert bytes_a == bytes_b, (
        "L3: the R18 byte-identity proof (metrics.json only) must survive "
        "the latency publication")
    metrics = json.loads(bytes_a)
    latency = metrics["latency"]
    assert latency["median"] == crunner._LATENCY_PINNED_MEDIAN, (
        "L3: metrics.json keeps the pinned placeholder median; measured "
        "wall clock lives in latency.json")
    assert latency["p95"] == pytest.approx(
        crunner._LATENCY_PINNED_MEDIAN * crunner._LATENCY_P95_RATIO), (
        "L3: metrics.json keeps the pinned placeholder p95")


# ===========================================================================
# L4 — build_latency_document: per-scenario latency_stats semantics and
#      run-index-aligned pooling across scenarios.
# ===========================================================================


def test_l4_build_latency_document_matches_latency_stats():
    runs_alpha = [[9.0, 9.0], [1.0, 2.0], [3.0], [4.0], [5.0]]
    runs_beta = [[8.0], [0.5], [1.5]]
    document = crunner.build_latency_document(
        {"alpha": runs_alpha, "beta": runs_beta})
    # Per-scenario: exactly latency_stats over that scenario's runs
    # (warm-up discarded). Alpha scored: [1,2,3,4,5] -> median 3.0,
    # p95 5.0; beta scored: [0.5,1.5] -> median 1.5, p95 1.5. The huge
    # warm-up values must not leak into any statistic.
    assert document["scenarios"]["alpha"] == {"median": 3.0, "p95": 5.0}, (
        f"L4: alpha must discard the warm-up and pool its scored runs; "
        f"got {document['scenarios']['alpha']!r}")
    assert document["scenarios"]["beta"] == {"median": 1.5, "p95": 1.5}
    # Aggregate: pool run-index-aligned across scenarios, discard pooled
    # run 1 entirely: [1,2,0.5,3,1.5,4,5] -> sorted [0.5,1,1.5,2,3,4,5],
    # median 2.0, p95 5.0.
    pooled = [[9.0, 9.0, 8.0], [1.0, 2.0, 0.5], [3.0, 1.5], [4.0], [5.0]]
    assert document["aggregate"] == crunner.latency_stats(pooled), (
        f"L4: aggregate must equal latency_stats over run-index-aligned "
        f"pooling; got {document['aggregate']!r}")
    assert document["aggregate"] == {"median": 2.0, "p95": 5.0}


# ===========================================================================
# L5 — degenerate inputs publish null, never NaN (allow_nan=False safe).
# ===========================================================================


def test_l5_degenerate_documents_serialize_without_nan():
    document = crunner.build_latency_document(
        {"warmup-only": [[1.0], [], [], [], []]})
    assert document["scenarios"]["warmup-only"] == {"median": None,
                                                    "p95": None}, (
        "L5: a scenario without scored timings publishes null, never the "
        "warm-up figure")
    json.dumps(document, allow_nan=False)  # must not raise
    empty = crunner.build_latency_document({})
    assert empty["scenarios"] == {}
    assert empty["aggregate"] == {"median": None, "p95": None}
    json.dumps(empty, allow_nan=False)  # must not raise


# ===========================================================================
# L6 — load_latency reader for the FR21 final stage.
# ===========================================================================


def test_l6_load_latency(tmp_path):
    assert crunner.load_latency(tmp_path) is None, (
        "L6: a pre-amendment run dir without latency.json must read as "
        "None (FR21 final stage: unresolved tie)")
    document = crunner.build_latency_document({"x": [[1.0], [2.0]]})
    (tmp_path / "latency.json").write_text(json.dumps(document))
    loaded = crunner.load_latency(tmp_path)
    assert loaded == document, "L6: load_latency returns the parsed document"
    assert crunner.load_latency(tmp_path / "does-not-exist") is None


# ===========================================================================
# L7 — candidate single-scenario sweep publishes a valid document.
# ===========================================================================


def test_l7_run_scenario_candidate_publishes_latency(tmp_path):
    scenario = _mini_scenario()
    run_dir = crunner.run_scenario_candidate(scenario, "structured",
                                             tmp_path / "cand")
    assert (run_dir / "metrics.json").is_file()
    _latency_shape(run_dir, {scenario.id})


# ===========================================================================
# L8 — full candidate run incl. FR40 (index-delivery timings included).
# ===========================================================================


def test_l8_run_candidate_publishes_latency_with_fr40(tmp_path):
    run_dir = crunner.run_candidate("structured", tmp_path / "full")
    assert (run_dir / "metrics.json").is_file()
    document = _latency_shape(run_dir, set(cscenarios.CORPUS))
    assert "episode_close_paraphrase_return" in document["scenarios"], (
        "L8: the candidate document must cover the FR40 scenario too — "
        "its reference turn is where the candidate index does the work "
        "the tie-breaker measures")
