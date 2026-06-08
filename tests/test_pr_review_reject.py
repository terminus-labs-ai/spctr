"""Tests for the PR review reject path — FR7.

Covers:
- Reject body formatting for gh pr review --request-changes
- Review round tracking (increment, cap at 3)
- Escalation to bd human after 3 rejected rounds
- RejectOutcome structure (bounce vs escalate)
"""

from __future__ import annotations

import pytest

from sr2_spectre.pr_review import (
    GateFinding,
    GateVerdict,
    RejectAction,
    RejectOutcome,
    build_reject_body,
    handle_reject,
    should_escalate,
)


# ---------------------------------------------------------------------------
# should_escalate
# ---------------------------------------------------------------------------

class TestShouldEscalate:
    """After 3 rejected rounds, escalate instead of bouncing."""

    def test_under_cap_returns_false(self):
        assert should_escalate(review_round=1, max_rounds=3) is False
        assert should_escalate(review_round=2, max_rounds=3) is False

    def test_at_cap_returns_true(self):
        assert should_escalate(review_round=3, max_rounds=3) is True

    def test_over_cap_returns_true(self):
        assert should_escalate(review_round=4, max_rounds=3) is True

    def test_custom_cap(self):
        assert should_escalate(review_round=5, max_rounds=5) is True
        assert should_escalate(review_round=4, max_rounds=5) is False


# ---------------------------------------------------------------------------
# build_reject_body
# ---------------------------------------------------------------------------

class TestBuildRejectBody:
    """Format findings into a review comment for gh pr review --request-changes."""

    def test_single_finding(self):
        body = build_reject_body(
            findings=[
                GateFinding(step="test_suite", message="pytest failed: 2 errors in test_agent.py"),
            ],
            review_round=1,
        )
        assert "test_suite" in body
        assert "pytest failed" in body
        assert "Round 1" in body

    def test_multiple_findings(self):
        body = build_reject_body(
            findings=[
                GateFinding(step="solid_review", message="SRP violation in module.py:42"),
                GateFinding(step="judgment", message="Module boundaries unclear"),
            ],
            review_round=2,
        )
        assert "solid_review" in body
        assert "SRP violation" in body
        assert "judgment" in body
        assert "Round 2" in body

    def test_empty_findings_produces_minimal_body(self):
        body = build_reject_body(findings=[], review_round=1)
        assert "Round 1" in body

    def test_includes_verdict(self):
        body = build_reject_body(
            findings=[GateFinding(step="test_suite", message="failed")],
            review_round=1,
        )
        assert "REJECT" in body

    def test_second_round_mentions_history(self):
        body = build_reject_body(
            findings=[GateFinding(step="solid_review", message="still broken")],
            review_round=2,
        )
        assert "Round 2" in body


# ---------------------------------------------------------------------------
# handle_reject
# ---------------------------------------------------------------------------

class TestHandleReject:
    """Core reject logic: bounce vs escalate based on round count."""

    def test_bounce_on_first_round(self):
        outcome = handle_reject(
            review_round=1,
            max_rounds=3,
            author_agent="tali",
            impl_bead_id="spc-50",
            review_bead_id="spc-51",
            pr_number=12,
            findings=[GateFinding(step="test_suite", message="pytest failed")],
        )
        assert outcome.action == RejectAction.BOUNCE
        assert outcome.review_round_after == 2
        assert outcome.author_agent == "tali"
        assert outcome.impl_bead_id == "spc-50"
        assert outcome.review_bead_id == "spc-51"
        assert outcome.pr_number == 12

    def test_bounce_on_second_round(self):
        outcome = handle_reject(
            review_round=2,
            max_rounds=3,
            author_agent="edi",
            impl_bead_id="spc-50",
            review_bead_id="spc-51",
            pr_number=12,
            findings=[GateFinding(step="solid_review", message="SRP violation")],
        )
        assert outcome.action == RejectAction.BOUNCE
        assert outcome.review_round_after == 3

    def test_escalate_on_third_round(self):
        outcome = handle_reject(
            review_round=3,
            max_rounds=3,
            author_agent="tali",
            impl_bead_id="spc-50",
            review_bead_id="spc-52",
            pr_number=13,
            findings=[GateFinding(step="judgment", message="fundamental design issue")],
        )
        assert outcome.action == RejectAction.ESCALATE
        assert outcome.review_round_after == 3
        assert outcome.escalation_message is not None
        assert "spc-50" in outcome.escalation_message
        assert "tali" in outcome.escalation_message

    def test_custom_max_rounds(self):
        outcome = handle_reject(
            review_round=5,
            max_rounds=5,
            author_agent="edi",
            impl_bead_id="spc-99",
            review_bead_id="spc-100",
            pr_number=20,
            findings=[GateFinding(step="test_suite", message="still failing")],
        )
        assert outcome.action == RejectAction.ESCALATE

    def test_bounce_includes_findings(self):
        outcome = handle_reject(
            review_round=1,
            max_rounds=3,
            author_agent="tali",
            impl_bead_id="spc-50",
            review_bead_id="spc-51",
            pr_number=12,
            findings=[
                GateFinding(step="test_suite", message="test_a failed"),
                GateFinding(step="solid_review", message="SRP violation"),
            ],
        )
        assert len(outcome.findings) == 2
        assert outcome.findings[0].message == "test_a failed"
        assert outcome.findings[1].message == "SRP violation"

    def test_pr_number_preserved(self):
        outcome = handle_reject(
            review_round=1,
            max_rounds=3,
            author_agent="edi",
            impl_bead_id="spc-77",
            review_bead_id="spc-78",
            pr_number=99,
            findings=[GateFinding(step="test_suite", message="fail")],
        )
        assert outcome.pr_number == 99

    def test_bounce_has_reject_body(self):
        outcome = handle_reject(
            review_round=1,
            max_rounds=3,
            author_agent="tali",
            impl_bead_id="spc-50",
            review_bead_id="spc-51",
            pr_number=12,
            findings=[GateFinding(step="test_suite", message="fail")],
        )
        assert outcome.reject_body is not None
        assert "REJECT" in outcome.reject_body

    def test_escalate_has_reject_body(self):
        outcome = handle_reject(
            review_round=3,
            max_rounds=3,
            author_agent="tali",
            impl_bead_id="spc-50",
            review_bead_id="spc-52",
            pr_number=13,
            findings=[GateFinding(step="judgment", message="fundamental issue")],
        )
        assert outcome.reject_body is not None
        assert "REJECT" in outcome.reject_body
