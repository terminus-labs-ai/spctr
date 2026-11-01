"""CI workflow structure test (FR6, spc-1040 AC7-r2).

Parses `.github/workflows/ci.yml` and asserts:
  * `on` includes both `push` and `pull_request`,
  * a job runs `scripts/ci.sh` (the same command a contributor runs
    locally),
  * the `integration` job is present with a `pgvector` Postgres service
    (ported from `packages/sr2/.github/workflows/ci.yml`, which is
    removed by the monorepo assembly).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"


def _load() -> dict:
    assert CI_YML.exists(), "missing .github/workflows/ci.yml"
    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))


@pytest.mark.baseline_debt("spc-1040")
def test_workflow_runs_on_push_and_pull_request() -> None:
    """AC7-r2: `on` has push and pull_request."""
    cfg = _load()
    # YAML parses a bare `on:` as a Python bool (key `True`); normalise.
    on = cfg.get("on") if "on" in cfg else cfg.get(True)
    assert on is not None, "workflow has no `on:` trigger"
    assert "push" in on, "workflow does not run on push"
    assert "pull_request" in on, "workflow does not run on pull_request"


@pytest.mark.baseline_debt("spc-1040")
def test_a_job_runs_scripts_ci_sh() -> None:
    """AC7-r2: a job runs `scripts/ci.sh`."""
    cfg = _load()
    jobs = cfg.get("jobs") or {}
    assert jobs, "workflow has no jobs"
    found = False
    for job in jobs.values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            if "scripts/ci.sh" in run:
                found = True
    assert found, "no job step runs scripts/ci.sh"


@pytest.mark.baseline_debt("spc-1040")
def test_integration_job_with_pgvector_service_present() -> None:
    """AC7-r2: the integration job with a pgvector Postgres service is
    present (ported from the old sr2 workflow)."""
    cfg = _load()
    jobs = cfg.get("jobs") or {}
    integration = jobs.get("integration")
    assert integration is not None, "no integration job"
    services = integration.get("services") or {}
    assert "postgres" in services, "integration job has no postgres service"
    image = services["postgres"].get("image", "")
    assert "pgvector" in image, f"postgres service image is not pgvector: {image!r}"
