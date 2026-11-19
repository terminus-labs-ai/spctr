"""sr2-177 — the FR40 benchmark isolates candidate index state per repetition.

``_run_fr40_scenario`` used to build ONE candidate index per scenario run and
re-append the scenario's records on every repetition. Re-appending stacks
duplicate entries in the index, and because the reference turn asks for
``search(limit=len(records))``, those duplicates crowd distinct records out:
the delivered background degraded from six distinct records (repetition 1) to
two (repetitions 3-5), so a published FR40 score depended on ``run_index``
(structured/vector_hybrid/embedding 0.400 correctness and recall, graph
1.000 — instead of the 0.143 / 0.286 every arm produces from one clean read
of the backend).

Rows:

r1 — every candidate arm publishes identical reference-turn metrics for
     repetitions 2…N (repetition 1 may differ as the discarded warm-up).
r2 — the reference turn's delivered background holds distinct records: every
     stored turn of the scenario reaches it exactly once, in every
     repetition.

Both rows drive the pinned public entry point
``evals.continuity.runner.run_scenario_candidate`` over the committed FR40
scenario and read only the published ``observations.jsonl`` surface, so the
isolation is asserted through what the bakeoff consumes rather than through
runner internals. The reference turn is identified from the published data:
it is the turn whose request text carries *other* turns' texts, i.e. the turn
whose background arrived through the candidate index (every other turn is
delivered only its own declared blocks). The deterministic replay installs at
runner import, so these runs perform no network access; the AC27 cage rows r3
and r7 keep pinning closed residency and the exactly-once embedding property
that the per-repetition rebuild must preserve (the SQLite vector cache and
the shared query-vector cache stay warm across repetitions).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

SCENARIO_ID = "episode_close_paraphrase_return"

#: Where a turn's own text lives, mirroring the corpus convention the AC27
#: cage pins: the question declared in ``meta["question"]``, else the first
#: plain text attribute.
_TURN_TEXT_ATTRS = ("question", "input", "user_text", "prompt", "text")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _mod(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # explicit failure message over import error
        raise AssertionError(f"cannot import {name}: {exc}") from exc


def _runner():
    mod = _mod("evals.continuity.runner")
    for name in ("run_scenario_candidate", "RUNS_PER_SCENARIO"):
        _require(hasattr(mod, name),
                 f"evals.continuity.runner exposes no {name} — the pinned "
                 f"FR40 entry point is missing")
    return mod


def _scenario():
    scenarios = _mod("evals.continuity.scenarios")
    corpus = getattr(scenarios, "CORPUS", None)
    _require(isinstance(corpus, dict) and bool(corpus),
             "evals.continuity.scenarios exposes no non-empty CORPUS")
    scenario = corpus.get(SCENARIO_ID)
    _require(scenario is not None,
             f"FR40 scenario {SCENARIO_ID!r} missing from CORPUS "
             f"(ids: {sorted(corpus)})")
    _require(bool(getattr(scenario, "turns", ()) or ()),
             f"{SCENARIO_ID!r} declares no turns")
    return scenario


def _arm_ids() -> list[str]:
    """Every registered candidate arm, with the arms the bakeoff compares
    proven present."""
    candidates = _mod("evals.continuity.candidates")
    registry = getattr(candidates, "CANDIDATES", None)
    _require(isinstance(registry, dict) and bool(registry),
             "evals.continuity.candidates exposes no CANDIDATES registry")
    ids = sorted(registry)
    missing = {"structured", "vector_hybrid", "graph"} - set(ids)
    _require(not missing,
             f"the candidate registry lacks {sorted(missing)}: {ids}")
    _require(any("embedding" in i for i in ids),
             f"the candidate registry lacks an embedding arm: {ids}")
    return ids


def _turn_text(turn: Any) -> str:
    meta = getattr(turn, "meta", None)
    if isinstance(meta, dict):
        value = meta.get("question")
        if isinstance(value, str) and value:
            return value
    for name in _TURN_TEXT_ATTRS:
        value = getattr(turn, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _turn_texts(scenario: Any) -> list[str]:
    """One entry per declared turn, positionally indexed by ``turn_index``:
    the turn's own text, ``""`` when it declares none. The reference turn is
    a live query rather than a stored record, but it is not known by index
    here — the rows identify it from published data, and each row excludes
    only the turn it is looking at when counting foreign content."""
    texts = [_turn_text(turn) for turn in scenario.turns]
    texted = [text for text in texts if text]
    _require(len(texted) >= 3,
             f"{SCENARIO_ID!r} declares fewer than three texted turns "
             f"({len(texted)}) — distinguishing 'distinct records' from "
             "'repeated versions' needs several")
    _require(len(set(texted)) == len(texted),
             f"the turns of {SCENARIO_ID!r} are not pairwise distinct texts")
    # The rows count occurrences of one turn's text inside another turn's
    # delivered request, so no text may nest inside another: a nested pair
    # would let one delivered record satisfy two counts.
    nested = sorted(f"{a[:40]!r} inside {b[:40]!r}"
                    for a in texted for b in texted
                    if a != b and a in b)
    _require(not nested,
             f"the turn texts of {SCENARIO_ID!r} nest into each other, so "
             f"occurrence counts cannot attribute a delivery to one record: "
             f"{nested[:4]}")
    return texts


def _int(value: Any, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"published field {name!r} must be an int; got {value!r}")
    return value


def _observations(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "observations.jsonl"
    _require(path.is_file(), f"the run published no {path}")
    observations = [json.loads(line)
                    for line in path.read_text().splitlines() if line.strip()]
    _require(bool(observations), f"no observations published under {run_dir}")
    return observations


def _index_delivered(observation: dict, all_texts: list[str]) -> int:
    """How many turn texts OTHER than this turn's own arrived in its
    delivered request text (background through the candidate index)."""
    delivered = str(observation.get("request_text") or "")
    turn_index = _int(observation.get("turn_index"), "turn_index")
    own = all_texts[turn_index - 1] if 0 < turn_index <= len(all_texts) else ""
    return sum(1 for text in all_texts
               if text != own and text and text in delivered)


def _reference_turns(observations: list[dict],
                     all_texts: list[str]) -> dict[int, dict]:
    """``{run_index: observation}`` for each repetition's index-delivered
    turn: the one published turn carrying other turns' texts. A repetition
    whose delivered background arrived through the index in some other shape
    fails the row rather than silently reading a different turn."""
    by_rep: dict[int, dict] = {}
    for observation in observations:
        rep = _int(observation.get("run_index"), "run_index")
        foreign = _index_delivered(observation, all_texts)
        if foreign == 0:
            continue
        _require(rep not in by_rep,
                 f"repetition {rep} has two turns carrying other turns' "
                 "texts — the scenario no longer has a single "
                 "index-delivered reference turn")
        by_rep[rep] = observation
    _require(bool(by_rep),
             "no published turn carried another turn's text: nothing reached "
             "a turn through the candidate index")
    return by_rep


def _scored_surface(observation: dict) -> dict:
    """Everything a published observation measures, minus the repetition
    identity."""
    return {key: value for key, value in sorted(observation.items())
            if key != "run_index"}


def _run_arm(root: Path, arm_id: str) -> Path:
    runner = _runner()
    run_dir = Path(runner.run_scenario_candidate(_scenario(), arm_id,
                                                 root / arm_id))
    _require(run_dir.is_dir(),
             f"run_scenario_candidate({arm_id!r}) returned no run dir: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Row sr2-177-r1 — per-repetition FR40 metrics are equal across repetitions
# 2…N within an arm (repetition 1 may differ as the discarded warm-up).
# ---------------------------------------------------------------------------


def test_fr40_reference_turn_metrics_equal_across_repetitions(tmp_path):
    runner = _runner()
    runs = _int(runner.RUNS_PER_SCENARIO, "RUNS_PER_SCENARIO")
    _require(runs >= 3,
             f"RUNS_PER_SCENARIO={runs}: the row compares repetition 2 with "
             "at least one later scored repetition")
    scenario = _scenario()
    all_texts = _turn_texts(scenario)
    failures: list[str] = []
    for arm_id in _arm_ids():
        reps = _reference_turns(_observations(_run_arm(tmp_path, arm_id)),
                               all_texts)
        _require(sorted(reps) == list(range(1, runs + 1)),
                 f"{arm_id}: published index-delivered repetitions "
                 f"{sorted(reps)}, expected 1..{runs}")
        baseline = _scored_surface(reps[2])
        for rep in range(3, runs + 1):
            current = _scored_surface(reps[rep])
            if current != baseline:
                failures.append(
                    f"{arm_id}: repetition {rep} differs from repetition 2 "
                    f"(candidate index state leaked across repetitions)\n"
                    f"  rep 2:  {baseline}\n  rep {rep}: {current}")
    _require(not failures, "\n".join(failures))


# ---------------------------------------------------------------------------
# Row sr2-177-r2 — the reference turn's delivered record set holds distinct
# records: each stored turn's text reaches it exactly once in every
# repetition, rather than repeated versions standing in for crowded-out
# records.
# ---------------------------------------------------------------------------


def test_fr40_reference_turn_delivers_distinct_records_every_repetition(tmp_path):
    runner = _runner()
    runs = _int(runner.RUNS_PER_SCENARIO, "RUNS_PER_SCENARIO")
    scenario = _scenario()
    all_texts = _turn_texts(scenario)
    failures: list[str] = []
    for arm_id in _arm_ids():
        reps = _reference_turns(_observations(_run_arm(tmp_path, arm_id)),
                               all_texts)
        _require(sorted(reps) == list(range(1, runs + 1)),
                 f"{arm_id}: published index-delivered repetitions "
                 f"{sorted(reps)}, expected 1..{runs}")
        for rep in sorted(reps):
            observation = reps[rep]
            delivered = str(observation.get("request_text") or "")
            own_index = _int(observation.get("turn_index"), "turn_index")
            own = all_texts[own_index - 1]
            stored = [t for t in all_texts if t and t != own]
            absent = sorted(t[:60] for t in stored if delivered.count(t) == 0)
            repeated = {t[:60]: delivered.count(t) for t in stored
                        if delivered.count(t) > 1}
            if absent or repeated:
                failures.append(
                    f"{arm_id} repetition {rep}: the index-delivered turn "
                    f"must receive each stored record exactly once; missing "
                    f"{absent}, received more than once {repeated} "
                    f"(background tokens: "
                    f"{observation.get('background_tokens')})")
    _require(not failures, "\n".join(failures))
