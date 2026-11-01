#!/usr/bin/env bash
# Run the full test suite of every workspace member from the repository
# root (spc-1038 monorepo layout).
#
# Each member runs in its own pytest invocation because both suites ship
# a root test package named `tests` (packages/*/tests/__init__.py), so a
# single pytest process over both directories collides on the conftest
# plugin registration (`tests.conftest` from two files). Per member,
# pytest picks up that member's own [tool.pytest.ini_options]
# (asyncio_mode, sr2's `pythonpath = ["tests"]`, spctr's
# `-m 'not live'`), exactly as each repo ran standalone.
#
# Usage: scripts/run_tests.sh [extra pytest args...]   # both members
#        MEMBER=packages/sr2 scripts/run_tests.sh     # one member
set -euo pipefail
cd "$(dirname "$0")/.."

status=0
for member in ${MEMBER:-packages/sr2 packages/spctr}; do
  uv run --frozen python -m pytest "$member" "$@" || status=1
done
exit $status
