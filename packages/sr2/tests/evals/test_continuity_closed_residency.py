"""sr2-176 acceptance — single aggregate formula; closed_residency derived, never literal.

Found by Role G acceptance of sr2-163 (non-blocking sub-mandate, DRY):
``run_baseline`` carried an inline copy of the nine-metric aggregate
formula already written in ``runner._aggregate_metrics``, and the sweep
path's ``closed_residency`` came from ``_fr40_closed_residency`` returning
a literal 0 while the FR40 run computed the same quantity turn by turn
from the actual delivered background. The aggregate (metrics.json) and
the FR40 run dir could therefore disagree silently.

Rows:

- **D1** One aggregate formula: the baseline's published ``metrics.json``
  equals ``runner._aggregate_metrics`` applied to its own published
  observations — the shared helper, not a copy.
- **D2** FR40 derivation: every FR40 observation carries a per-turn
  ``flags.closed_residency`` counted from that turn's actual delivery,
  and the run's published aggregate is the sum of those flags (target 0
  — working-set exit; the index-delivered reference turn is compound
  record content, never a verbatim A turn).
- **D3** Sweep path: in a whole-corpus candidate run, ``closed_residency``
  in ``metrics.json`` equals the sum of the FR40 observations' flags in
  the same run's stream — the disagreement the bead names is impossible
  by construction. A non-FR40 scenario stream contributes nothing (no
  close turn), through the same derivation.
"""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path
from typing import Any, Self

from evals.continuity import candidates as cands
from evals.continuity import runner as crunner

FR40 = crunner._FR40

_LOOPBACK = re.compile(r"(127\.0\.0\.\.|::1|localhost)", re.IGNORECASE)


class _BlockedNetwork(RuntimeError):
    pass


class _NetworkBlocker:
    """Block every outbound socket except loopback, recording external
    attempts (same convention as the sibling continuity cages)."""

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
                    f"live network access is disabled (attempted {host!r})"
                )

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
        socket.socket, socket.create_connection, socket.getaddrinfo = self._real


def _read_obs(run_dir: Path) -> list[dict]:
    lines = (run_dir / "observations.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _embedding_key() -> str:
    keys = [k for k in cands.CANDIDATES if "embedding" in str(k).lower()]
    assert keys, f"no embedding candidate registered: {sorted(cands.CANDIDATES)}"
    return keys[0]


# ===========================================================================
# D0 — the derivation itself: FR40 flags summed, everything else ignored.
# ===========================================================================


def test_fr40_closed_residency_sums_fr40_flags_only():
    observations = [
        {"scenario_id": FR40, "flags": {"closed_residency": 3}},
        {"scenario_id": FR40, "flags": {}},  # legacy record without the flag
        {"scenario_id": "cage-composition",
         "flags": {"closed_residency": 99}},
        {"scenario_id": FR40, "flags": {"closed_residency": 4}},
    ]
    assert crunner._fr40_closed_residency(observations) == 7
    assert crunner._fr40_closed_residency([]) == 0


# ===========================================================================
# D1 — one aggregate formula: the baseline publishes the shared helper's
#      output for its own observations (no inline copy).
# ===========================================================================


def test_d1_baseline_metrics_equal_shared_aggregate(tmp_path):
    with _NetworkBlocker() as net:
        run_dir = crunner.run_baseline(output_root=tmp_path / "d1")
    observations = _read_obs(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    expected = crunner._aggregate_metrics(observations, crunner.load_pricing())
    assert metrics == expected, (
        "baseline metrics.json must be runner._aggregate_metrics over its own "
        f"published observations; diff: {sorted(set(metrics) ^ set(expected))}"
    )
    assert net.external == []


# ===========================================================================
# D2 — FR40: per-turn flags from actual delivery; the published aggregate
#      is their sum (target 0).
# ===========================================================================


def test_d2_fr40_aggregate_is_sum_of_per_turn_flags(tmp_path):
    scenario = crunner.CORPUS[FR40]
    with _NetworkBlocker() as net:
        run_dir = crunner._run_fr40_scenario(scenario, _embedding_key(),
                                             tmp_path / "d2", "fr40-run")
    observations = _read_obs(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    flags = [o["flags"].get("closed_residency") for o in observations]
    assert all(isinstance(f, int) and not isinstance(f, bool) for f in flags), (
        f"every FR40 observation must carry an int flags.closed_residency: {flags}")
    assert metrics["closed_residency"] == sum(flags), (
        "the FR40 run's published aggregate must be the sum of its own "
        f"per-turn flags: {metrics['closed_residency']} != {sum(flags)}")
    assert metrics["closed_residency"] == 0, (
        f"FR40 target: closed_residency == 0; got {metrics['closed_residency']!r}")
    references = [o for o in observations
                  if o["flags"].get("closed_on_turn") is not None]
    assert len(references) == crunner.RUNS_PER_SCENARIO
    assert net.external == []


# ===========================================================================
# D3 — sweep path: the whole-corpus aggregate derives from the FR40
#      observations in the same run's stream; a non-FR40 stream
#      contributes nothing, through the same derivation.
# ===========================================================================


def test_d3_sweep_aggregate_matches_fr40_observation_flags(tmp_path):
    with _NetworkBlocker() as net:
        run_dir = crunner.run_candidate(_embedding_key(), tmp_path / "d3")
    observations = _read_obs(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    fr40_flags = [int(o["flags"]["closed_residency"])
                  for o in observations if o.get("scenario_id") == FR40]
    assert fr40_flags, "the run must publish FR40 observations"
    assert metrics["closed_residency"] == sum(fr40_flags), (
        "sweep metrics.json and the FR40 run's observations must agree: "
        f"{metrics['closed_residency']} != {sum(fr40_flags)}")
    assert metrics["closed_residency"] == \
        crunner._fr40_closed_residency(observations)
    assert net.external == []


def test_d3_non_fr40_scenario_stream_contributes_nothing(tmp_path):
    scenario = next(s for s in crunner.CORPUS.values() if s.id != FR40)
    with _NetworkBlocker():
        run_dir = crunner.run_scenario_candidate(scenario, _embedding_key(),
                                                 tmp_path / "d3b")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["closed_residency"] == 0
    observations = _read_obs(run_dir)
    assert crunner._fr40_closed_residency(observations) == 0
