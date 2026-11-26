#!/usr/bin/env bash
# Diff collected pytest node IDs between a base ref and the current head
# (spc-1066) — see scripts/collect_diff.py for the full contract and exit
# codes. Thin entry point mirroring scripts/run_tests.sh:
#
#   uv run --frozen python scripts/collect_diff.py <base-ref> [options]
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --frozen python scripts/collect_diff.py "$@"
