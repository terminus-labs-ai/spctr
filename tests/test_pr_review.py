"""Tests for the PR review workflow module (FR4-FR5).

Covers:
- FR4: Reviewer claim routing (self-review prevention, peer routing)
- FR5: Three-part merge gate (test suite → solid-review → judgment)
"""

from __future__ import annotations

import pytest

from sr2_spectre.pr_review import (
    ClaimRoutingError,
    GateResult,
    GateStep,
    GateVerdict,
    check_self_review,
)


# ---------------------------------------------------------------------------
# FR4: Reviewer claim routing
# ---------------------------------------------------------------------------

class TestCheckSelfReview:
    """An agent must never claim a review bead for a PR it authored."""

    def test_same_agent_raises_claim_routing_error(self):
        """An agent reviewing its own PR is a routing error."""
        with pytest.raises(ClaimRoutingError, match="self-review"):
            check_self_review(
                reviewer_agent="edi",
                pr_author_agent="edi",
            )

    def test_peer_agent_is_allowed(self):
        """A peer agent reviewing another agent's PR is fine."""
        # Should not raise
        check_self_review(
            reviewer_agent="tali",
            pr_author_agent="edi",
        )

    def test_different_agents_allowed_regardless_of_order(self):
        """Routing is symmetric — either agent can review the other's PR."""
        check_self_review(reviewer_agent="edi", pr_author_agent="tali")
        check_self_review(reviewer_agent="tali", pr_author_agent="edi")


class TestGateVerdict:
    """Verdict is a simple enum with MERGE and REJECT."""

    def test_merge_verdict(self):
        assert GateVerdict.MERGE.value == "MERGE"

    def test_reject_verdict(self):
        assert GateVerdict.REJECT.value == "REJECT"


class TestGateStep:
    """Three gate steps in order: test_suite, solid_review, judgment."""

    def test_step_names(self):
        assert GateStep.TEST_SUITE.value == "test_suite"
        assert GateStep.SOLID_REVIEW.value == "solid_review"
        assert GateStep.JUDGMENT.value == "judgment"

    def test_ordered(self):
        """Steps have a defined order: cheap → expensive."""
        steps = list(GateStep)
        assert steps == [GateStep.TEST_SUITE, GateStep.SOLID_REVIEW, GateStep.JUDGMENT]


class TestGateResult:
    """GateResult captures the outcome of the three-part gate."""

    def test_pass_result(self):
        result = GateResult(verdict=GateVerdict.MERGE, findings=[])
        assert result.verdict == GateVerdict.MERGE
        assert result.passed is True
        assert result.failures == []

    def test_fail_result_with_findings(self):
        findings = [
            ("test_suite", "pytest failed: 2 errors"),
            ("solid_review", "SRP violation in module.py:42"),
        ]
        result = GateResult(verdict=GateVerdict.REJECT, findings=findings)
        assert result.verdict == GateVerdict.REJECT
        assert result.passed is False
        assert len(result.failures) == 2

    def test_fail_result_single_step(self):
        """A gate can fail on a single step and stop."""
        result = GateResult(
            verdict=GateVerdict.REJECT,
            findings=[("test_suite", "pytest failed")],
        )
        assert len(result.failures) == 1
        assert result.failures[0].step == "test_suite"

    def test_result_is_frozen(self):
        """GateResult is immutable once created."""
        result = GateResult(verdict=GateVerdict.MERGE, findings=[])
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            result.verdict = GateVerdict.REJECT  # type: ignore
