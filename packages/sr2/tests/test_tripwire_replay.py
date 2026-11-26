"""spc-1062 AC5-r1..r6: the tripwire replayed over real role-C sessions.

Spec agent-workflow-v3 Amendment 7 (D38, AC5): with the default thresholds
the tripwire stops before call 100 on the three stuck sessions and never
notes or stops on the three healthy ones.

Fixtures in ``tests/tripwire/`` are reduced from the eval transcripts
(``samara-failure-review-eval-2026-11-11/transcripts``): one JSON object per
executed tool call in session order — ``call`` (1-based index), ``name``,
``arguments`` (the command, for test-run-shaped commands only) and
``result`` (test-run output cut to its failure lines and summaries).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sr2.config.models import TripwireConfig
from sr2.tripwire import Tripwire

FIXTURES = Path(__file__).parent / "tripwire"

STUCK = ["sr2-128-c1", "sr2-128-c2", "obsidian-t4p8-c2"]
HEALTHY = ["sr2-142-c", "spc-162-c", "sr2-158-c"]

# Executed tool calls per session in the full transcripts (tool.start count).
CALLS = {
    "sr2-128-c1": 204,
    "sr2-128-c2": 200,
    "obsidian-t4p8-c2": 203,
    "sr2-142-c": 135,
    "spc-162-c": 58,
    "sr2-158-c": 74,
}


def _replay(session: str) -> tuple[list[tuple[int, str]], int]:
    """Feed every call to a default-enabled tripwire, stopping at a stop.

    Returns (actions as (call, action), calls observed).
    """
    tripwire = Tripwire(TripwireConfig(enabled=True))
    actions: list[tuple[int, str]] = []
    lines = (FIXTURES / f"{session}.jsonl").read_text().splitlines()
    for line in lines:
        call = json.loads(line)
        action = tripwire.observe(call["name"], call["arguments"], call["result"])
        assert tripwire.call_count == call["call"], "the tripwire numbers calls in session order"
        if action is not None:
            actions.append((call["call"], action))
        if action == "stop":
            break
    return actions, tripwire.call_count


@pytest.mark.parametrize("session", STUCK)
def test_stuck_session_stops_before_call_100(session: str) -> None:
    """AC5-r1..r3: a note first, then the stop, at a call index below 100."""
    actions, observed = _replay(session)
    stops =[call for call, action in actions if action == "stop"]
    assert len(stops) == 1, actions
    assert stops[0] < 100, actions
    assert observed == stops[0], "nothing is observed after the stop"
    assert actions[0][1] == "note" and actions[0][0] < stops[0], actions


@pytest.mark.parametrize("session", HEALTHY)
def test_healthy_session_never_trips(session: str) -> None:
    """AC5-r4..r6: no note and no stop, replayed to the session's last call."""
    actions, observed = _replay(session)
    assert actions == []
    assert observed == CALLS[session]


def test_fixtures_cover_every_call() -> None:
    """Each fixture holds every executed call of its session, in order."""
    for session, total in CALLS.items():
        lines = (FIXTURES / f"{session}.jsonl").read_text().splitlines()
        assert [json.loads(line)["call"] for line in lines] == list(range(1, total + 1))
