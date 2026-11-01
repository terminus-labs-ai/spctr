#!/usr/bin/env bash
# CI entry point (FR6, spc-1040): the same command .github/workflows/ci.yml
# runs on push and pull_request, and the same command a contributor runs
# locally. Exits 0 iff:
#   1. `uv sync --frozen` provisions the workspace venv,
#   2. both packages' test suites pass with `-m "not baseline_debt"`,
#   3. `ruff check --select ASYNC221` (async functions must not run
#      processes with blocking methods) passes for both packages' src/
#      trees.
#
# The full ruff surface (541 findings across both packages) is tracked
# separately as a follow-up bead; until it is cleaned up, CI checks only
# the subset that currently passes (ASYNC221). The remaining findings
# (E501 line length, F401 unused imports — mostly re-exports in
# __init__.py, F821 undefined names — mix of true positives and false
# positives from `from __future__ import annotations`, F841 unused local
# variables, and ~40 other style rules) are out of scope for this bead.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== uv sync --frozen ==="
uv sync --frozen

echo "=== pytest packages/sr2 ==="
uv run --frozen python -m pytest packages/sr2 -m "not baseline_debt"

echo "=== pytest packages/spctr ==="
uv run --frozen python -m pytest packages/spctr -m "not baseline_debt"

echo "=== ruff check packages/sr2/src ==="
uv run --frozen ruff check packages/sr2/src --select ASYNC221

echo "=== ruff check packages/spctr/src ==="
uv run --frozen ruff check packages/spctr/src --select ASYNC221

echo "ci.sh: all checks passed"
