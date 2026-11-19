"""sr2-179 acceptance cage — the candidate sweep delivers through the index.

Found by the AC19 bakeoff (sr2-100): ``run_candidate(candidate_id,
output_root)`` swept the ten FR31 scenarios and the ``cage-composition``
fixture with ``run_scenario_turn``, which never received the candidate
index — those 60 observations were byte-identical to the baseline's in all
four candidate runs, so "compared under identical inputs and budgets" was
real on ``episode_close_paraphrase_return`` alone.

Rows:

- **W1** Every non-FR40 sweep turn consults the candidate index: one
  search per published sweep observation (plus the five FR40 reference
  drives), on that turn's question, each returning hits; every published
  sweep observation carries ``request_text``.
- **W2** Substitution proof: an index that serves back only a declared
  subset delivers exactly that subset — the withheld eligible value is
  absent from the compiled request, present under a serve-all index. The
  background provably comes out of the index, not the full history.
- **W3** Budget: the scenario's ``token_budget`` cuts the ranked delivery
  (top-ranked hit always travels; later hits stop at the cap).
- **W4** All four arms: deterministic (byte-identical metrics.json across
  consecutive runs), fully offline (no external socket), and the published
  record shape is intact (sweep records carry ``request_text``; FR40
  judge rows unchanged).
- **W5** Control-plan symmetry and discrimination: on the composition
  control slot (run 1, turn 1) every arm still exercises the policy-leak
  path (eligible policy content reaches the restricted background through
  the index), the lexical/graph arms resolve the pinned stale exercise as
  the baseline does, and the embedding arm — whose retrieval genuinely
  ranks differently — resolves it otherwise. That flip is the bakeoff's
  discriminating power becoming real on an artifact surface.
"""

from __future__ import annotations

import json
import re
import socket
from collections import Counter
from pathlib import Path
from typing import Any, Self

from evals.continuity import candidates as cands
from evals.continuity import runner as crunner
from evals.continuity import scenarios as cscen
from evals.continuity.models import Block, Scenario, ScenarioTurn

FR40 = crunner._FR40
_SWEEP_CORE_KEYS = {"scenario_id", "run_index", "turn_index", "prompt_tokens",
                    "background_tokens", "response", "flags"}
_FLAG_KEYS = {"correct", "recall", "stale_current", "policy_leak",
              "irrelevant_context", "hallucination"}


def _read_obs(run_dir: Path) -> list[dict]:
    lines = (run_dir / "observations.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Offline guard (same contract as the AC27 cage: loopback open, everything
# else blocked and recorded).
# ---------------------------------------------------------------------------

_LOOPBACK = re.compile(r"^(127\.|localhost$|::1$|\[::1\]$)")


class _BlockedNetwork(RuntimeError):
    pass


class _NetworkBlocker:
    def __init__(self) -> None:
        self.external: list[Any] = []
        self._real: tuple = ()

    def __enter__(self) -> Self:
        real = (socket.socket, socket.create_connection, socket.getaddrinfo)
        self._real = real
        attempts = self.external

        def _guard_host(host):
            if not _LOOPBACK.search(str(host)):
                attempts.append(host)
                raise _BlockedNetwork(
                    f"live network access is disabled (attempted {host!r})")

        def _no_socket(*args, **kwargs):
            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, tuple) and arg:
                    _guard_host(arg[0])
                    return real[0](*args, **kwargs)
                if isinstance(arg, (bytes, bytearray)) and len(arg) in (4, 16):
                    attempts.append("<packed address>")
                    raise _BlockedNetwork("live socket connect is disabled")
            return real[0](*args, **kwargs)

        def _guarded_connection(address, *a, **k):
            _guard_host(address[0] if isinstance(address, tuple) else address)
            return real[1](address, *a, **k)

        def _guarded_getaddrinfo(host, *a, **k):
            _guard_host(host)
            return real[2](host, *a, **k)

        socket.socket = _no_socket  # type: ignore[misc]
        socket.create_connection = _guarded_connection  # type: ignore[assignment]
        socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        socket.socket, socket.create_connection, socket.getaddrinfo = \
            self._real


# ---------------------------------------------------------------------------
# W1 — every sweep turn consults the index (spy accounting)
# ---------------------------------------------------------------------------


class _SpyIndex:
    """Delegates to a real index, recording every drive."""

    def __init__(self, inner: Any, log: dict) -> None:
        self._inner = inner
        self._log = log

    async def append(self, records) -> None:
        self._log["appends"] += 1
        await self._inner.append(records)

    async def search(self, context, limit):
        hits = await self._inner.search(context, limit)
        self._log["searches"].append({
            "query": context.query,
            "budget": context.token_budget,
            "limit": limit,
            "hits": len(hits),
        })
        return hits

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_w1_every_sweep_turn_consults_the_candidate_index(tmp_path,
                                                          monkeypatch):
    arm = "structured"
    log = {"appends": 0, "searches": []}
    real_factory = cands.CANDIDATES[arm]

    def _spy_factory():
        return _SpyIndex(real_factory(), log)

    monkeypatch.setitem(cands.CANDIDATES, arm, _spy_factory)
    run_dir = crunner.run_candidate(arm, tmp_path / "w1")

    obs = _read_obs(run_dir)
    sweep = [o for o in obs if o["scenario_id"] != FR40]
    assert sweep, "no sweep observations published"

    # One search per published sweep turn plus the five FR40 reference
    # drives; every drive returned material.
    per_search = log["searches"]
    assert all(s["hits"] > 0 for s in per_search), (
        f"a consult returned nothing: {per_search}")
    expected = Counter()
    for scenario in cscen.CORPUS.values():
        if scenario.id == FR40:
            query = crunner._fr40_query(scenario)
            expected[query] += crunner.RUNS_PER_SCENARIO
            continue
        for turn in cscen.gold_turns(scenario):
            q = " ".join((turn.meta or {}).get("question", "").split())
            expected[q] += crunner.RUNS_PER_SCENARIO
    observed = Counter(s["query"] for s in per_search)
    assert observed == expected, (
        "searches are not one-per-sweep-turn (plus the five FR40 drives): "
        f"\n expected {dict(sorted(expected.items()))}"
        f"\n observed {dict(sorted(observed.items()))}")
    assert log["appends"] == sum(expected.values()), (
        "every consult must store the turn's eligible content first "
        f"(appends={log['appends']}, drives={sum(expected.values())})")

    # Artifact-level proof field: every sweep record publishes the request
    # its arm actually delivered.
    for o in sweep:
        rt = o.get("request_text")
        assert isinstance(rt, str) and rt, (
            f"sweep record lacks request_text: {o}")
        body = (o["response"].removeprefix(crunner._STALE_MARK + " ")
                if o["response"].startswith(crunner._STALE_MARK)
                else o["response"])
        first_value = body.split(";")[0].strip()
        assert first_value in rt, (
            f"the double's stated value is absent from the published "
            f"request: {o['scenario_id']} run {o['run_index']}")


# ---------------------------------------------------------------------------
# W2 / W3 — substitution and budget proofs on a controlled scenario
# ---------------------------------------------------------------------------

_FIRST = Block(text="first-key: alpha-value", layer="continuity_replay",
               observed_at="2026-08-01T00:00:00Z")
_SECOND = Block(text="second-key: beta-value", layer="continuity_replay",
                observed_at="2026-06-01T00:00:00Z")


def _fixture_scenario(token_budget: int = 4000) -> Scenario:
    """One retrieval turn, two eligible values (gold spans both). The
    perfect reader states alpha first (newer sourcing), so a serve-all
    index yields both keys and a subsetting index hides one verifiably."""
    turn = ScenarioTurn(
        gold_answers=("first-key: alpha-value",),
        current_values=("first-key: alpha-value", "second-key: beta-value"),
        superseded_values=(),
        blocks=(_FIRST, _SECOND),
        delivered=None,
        meta={"question": "which keys hold?"},
    )
    return Scenario(
        id="w2-substitution-fixture",
        turns=(turn,),
        token_budget=token_budget,
        prompt_budget=1000,
        agent_policy="unrestricted",
        runner_mode="full_history",
    )


class _StubIndex:
    """Serves back exactly the appended records matching ``keep`` (or all
    of them), preserving store order — the substitute backend that makes
    the routing observable."""

    def __init__(self, keep) -> None:
        self._keep = keep
        self._stored: list = []

    async def append(self, records) -> None:
        self._stored.extend(records)

    async def search(self, context, limit):
        from sr2.continuity.index import ContinuityCandidate
        from sr2.continuity.models import ContinuityView

        out = []
        for record in self._stored:
            if not self._keep(record):
                continue
            view = ContinuityView(
                subject_id=record.subject_id,
                current_record=record,
                status="active",
                history_ids=(record.id,),
            )
            out.append(ContinuityCandidate(
                record=view, score=1.0, score_components={},
                supporting_entry_ids=(),
                estimated_tokens=max(1, len(record.content) // 4)))
        return out[:limit]


def _run_with_stub(tmp_path, monkeypatch, scenario, keep) -> Path:
    monkeypatch.setitem(cands.CANDIDATES, "structured",
                        lambda: _StubIndex(keep))
    return crunner.run_scenario_candidate(scenario, "structured",
                                          tmp_path / "stub")


def test_w2_background_arrives_through_the_index_not_full_history(tmp_path,
                                                                  monkeypatch):
    scenario = _fixture_scenario()

    all_dir = _run_with_stub(tmp_path / "all", monkeypatch, scenario,
                             lambda record: True)
    all_rows = _read_obs(all_dir)
    assert all_rows, "serve-all stub published no observations"
    all_rt = all_rows[0]["request_text"]
    assert "first-key: alpha-value" in all_rt
    assert "second-key: beta-value" in all_rt, (
        "serve-all must deliver both eligible values through the index")
    assert all_rows[0]["flags"]["correct"] == 1.0

    # Substituting an index that hides one record hides it from the
    # compiled request entirely: no full-history fallback smuggles the
    # eligible-but-unretrieved value back in.
    subset_dir = _run_with_stub(
        tmp_path / "subset", monkeypatch, scenario,
        lambda record: record.content == _FIRST.text)
    subset_rows = _read_obs(subset_dir)
    subset_rt = subset_rows[0]["request_text"]
    assert "first-key: alpha-value" in subset_rt
    assert "second-key: beta-value" not in subset_rt, (
        "an unretrieved eligible value reached the request — the sweep is "
        "not delivering through the index")
    assert "beta-value" not in subset_rows[0]["response"]


def test_w3_token_budget_cuts_the_ranked_delivery(tmp_path, monkeypatch):
    # est tokens per record = max(1, len(content)//4) = 5; a budget of 6
    # fits exactly the top-ranked hit.
    scenario = _fixture_scenario(token_budget=6)
    run_dir = _run_with_stub(tmp_path / "cut", monkeypatch, scenario,
                             lambda record: True)
    rows = _read_obs(run_dir)
    rt = rows[0]["request_text"]
    assert "first-key: alpha-value" in rt, (
        "the top-ranked hit must always travel")
    assert "second-key: beta-value" not in rt, (
        f"a hit past the token budget travelled: {rt!r}")


# ---------------------------------------------------------------------------
# W4 — all four arms: deterministic, offline, shape intact
# ---------------------------------------------------------------------------


def test_w4_all_arms_deterministic_offline_and_shape_intact(tmp_path):
    for arm in sorted(cands.CANDIDATES):
        with _NetworkBlocker() as net:
            dir_a = crunner.run_candidate(arm, tmp_path / arm / "a")
            dir_b = crunner.run_candidate(arm, tmp_path / arm / "b")
        assert net.external == [], (
            f"{arm}: deterministic run reached the network: {net.external}")
        bytes_a = (dir_a / "metrics.json").read_bytes()
        bytes_b = (dir_b / "metrics.json").read_bytes()
        assert bytes_a == bytes_b, (
            f"{arm}: consecutive runs produced different metrics.json")

        rows = _read_obs(dir_a)
        sweep = [o for o in rows if o["scenario_id"] != FR40]
        assert len(sweep) == 60, f"{arm}: expected 60 sweep rows"
        for o in rows:
            missing = _SWEEP_CORE_KEYS - set(o)
            assert not missing, f"{arm}: record missing {missing}: {o}"
            missing_flags = _FLAG_KEYS - set(o["flags"])
            assert not missing_flags, (
                f"{arm}: flags missing {missing_flags}")
        for o in sweep:
            assert "request_text" in o, (
                f"{arm}: sweep record without request_text: "
                f"{o['scenario_id']}")
        fr40 = [o for o in rows if o["scenario_id"] == FR40]
        assert fr40 and all("judge_decision" in o for o in fr40), (
            f"{arm}: FR40 judge-decision publication moved")


# ---------------------------------------------------------------------------
# W5 — control-plan symmetry and genuine cross-arm discrimination
# ---------------------------------------------------------------------------


def _composition_slot(rows: list[dict], run_index: int,
                      turn_index: int) -> dict:
    hits = [o for o in rows if o["scenario_id"] == "cage-composition"
            and o["run_index"] == run_index
            and o["turn_index"] == turn_index]
    assert len(hits) == 1, f"composition ({run_index},{turn_index}): {hits}"
    return hits[0]


def test_w5_control_plan_exercised_by_every_arm_and_arms_discriminate(
        tmp_path):
    base_dir = crunner.run_baseline(tmp_path / "base")
    base_rows = _read_obs(base_dir)
    base_slot = _composition_slot(base_rows, 1, 1)
    assert base_slot["flags"]["policy_leak"] == 1, (
        "the pinned baseline control slot no longer exercises the leak")

    outcomes = {}
    for arm in sorted(cands.CANDIDATES):
        arm_dir = crunner.run_candidate(arm, tmp_path / f"w5-{arm}")
        rows = [o for o in _read_obs(arm_dir)]
        slot = _composition_slot(rows, 1, 1)
        # Identical inputs: the eligible policy content still reaches the
        # restricted background through the index — every arm exercises
        # the leak path exactly once here, as the baseline does.
        assert slot["flags"]["policy_leak"] == 1, (
            f"{arm}: control-plan policy content did not traverse the "
            "index into the restricted background")
        total_leaks = sum(o["flags"]["policy_leak"]
                          for o in rows if o["scenario_id"] != FR40)
        assert total_leaks == 1, (
            f"{arm}: sweep published {total_leaks} policy leaks; the "
            "control plan exists to produce exactly one")
        outcomes[arm] = slot

    # Discrimination is real: the embedding arm retrieves in a different
    # order and resolves the pinned stale exercise differently from the
    # baseline, while the lexical arms resolve it identically.
    for arm in ("structured", "vector_hybrid", "graph"):
        assert outcomes[arm]["response"] == base_slot["response"], (
            f"{arm}: composition control slot drifted from the baseline "
            "delivery unexpectedly")
    embed = outcomes["embedding"]
    assert embed["flags"]["stale_current"] != base_slot["flags"]["stale_current"], (
        "the embedding arm must resolve the control slot through its own "
        "retrieval — identical bytes to the baseline would mean the fix "
        "regressed to the original bug")
    assert embed["response"] != base_slot["response"]
    assert embed["request_text"], "candidate proof field missing"
