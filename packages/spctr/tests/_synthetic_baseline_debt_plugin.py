"""Synthetic ``baseline_debt`` markers for the guard test (AC6-r2, AC6-r3).

``test_baseline_debt_guard.py`` writes a tiny pytest project into a tmp dir:
one ``conftest.py`` that registers the marker, a test module carrying the
synthetic marker(s), and this plugin (imported from the tmp dir via
``PYTHONPATH``) which, when ``--report-baseline-debt`` is passed, emits every
collected ``baseline_debt`` marker as JSON so the guard can validate it.
The synthetic test names (``test_synthetic_no_bead_id`` /
``test_synthetic_bad_bead_id``) appear in no other test file, so their
markers are unambiguously the synthetic ones.
"""

from __future__ import annotations

import json
import sys

import pytest

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--report-baseline-debt",
        action="store_true",
        default=False,
        help="emit collected baseline_debt markers as JSON (guard test)",
    )


def pytest_configure(config: pytest.Config) -> None:
    # The real runs register baseline_debt in the member's own pyproject.toml;
    # the synthetic tmp project has no config file, so register it here.
    if config.inifile is None:
        config.addinivalue_line(
            "markers",
            "baseline_debt(bead_id): synthetic baseline-debt marker for the guard test",
        )


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    if not config.getoption("--report-baseline-debt"):
        return
    markers = [
        {"nodeid": item.nodeid, "args": list(mark.args), "kwargs": dict(mark.kwargs)}
        for item in items
        for mark in item.iter_markers(name="baseline_debt")
    ]
    sys.stdout.write("BASELINE_DEBT_MARKERS " + json.dumps(markers) + "\n")
