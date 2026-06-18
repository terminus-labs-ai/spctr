"""Tests for base freshness checks in the PR review workflow.

Covers:
- Author rebase-before-PR check (prevent stale branches from opening PRs)
- Reviewer base-freshness gate (reject PRs that are behind main)
- FreshnessResult structure and verdict logic
"""

from __future__ import annotations

import pytest

from sr2_spectre.pr_review import (
    FreshnessResult,
    FreshnessVerdict,
    RebaseStrategy,
    author_rebase_check,
    check_freshness,
)


# ---------------------------------------------------------------------------
# check_freshness
# ---------------------------------------------------------------------------

class TestCheckFreshness:
    """Determine whether a PR branch is fresh enough to review."""

    def test_fresh_when_not_behind(self):
        result = check_freshness(behind_by=0, ahead_by=3)
        assert result.verdict == FreshnessVerdict.FRESH
        assert result.behind_by == 0
        assert result.ahead_by == 3
        assert "up to date" in result.suggestion

    def test_fresh_when_ahead_only(self):
        """A branch ahead of main by its own commits is fresh."""
        result = check_freshness(behind_by=0, ahead_by=1)
        assert result.verdict == FreshnessVerdict.FRESH

    def test_stale_when_behind(self):
        result = check_freshness(behind_by=5, ahead_by=2)
        assert result.verdict == FreshnessVerdict.STALE
        assert result.behind_by == 5
        assert result.ahead_by == 2
        assert "Rebase" in result.suggestion or "rebase" in result.suggestion
        assert "5" in result.suggestion

    def test_stale_suggestion_mentions_rebase(self):
        result = check_freshness(behind_by=1, ahead_by=1)
        assert "rebase" in result.suggestion.lower()
        assert "origin/main" in result.suggestion

    def test_zero_ahead_zero_behind_is_fresh(self):
        """Edge case: branch identical to main (no feature commits yet)."""
        result = check_freshness(behind_by=0, ahead_by=0)
        assert result.verdict == FreshnessVerdict.FRESH

    def test_negative_behind_treated_as_fresh(self):
        """Negative behind_by (shouldn't happen, but defensive)."""
        result = check_freshness(behind_by=-1, ahead_by=2)
        assert result.verdict == FreshnessVerdict.FRESH

    def test_result_is_frozen(self):
        result = check_freshness(behind_by=0, ahead_by=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            result.verdict = FreshnessVerdict.STALE  # type: ignore


# ---------------------------------------------------------------------------
# author_rebase_check
# ---------------------------------------------------------------------------

class TestAuthorRebaseCheck:
    """Pre-PR check: should the author rebase before pushing?"""

    def test_fresh_branch_returns_none(self):
        """No rebasing needed when branch is up to date."""
        result = author_rebase_check(behind_by=0)
        assert result is None

    def test_stale_branch_returns_instruction(self):
        result = author_rebase_check(behind_by=3)
        assert result is not None
        assert "3" in result
        assert "Rebasing" in result or "rebase" in result.lower()

    def test_fail_strategy_refuses(self):
        result = author_rebase_check(behind_by=2, strategy=RebaseStrategy.FAIL)
        assert result is not None
        assert "Refusing" in result or "refusing" in result.lower()

    def test_rebase_strategy_suggests_rebase(self):
        result = author_rebase_check(behind_by=1, strategy=RebaseStrategy.REBASE)
        assert result is not None
        assert "Rebasing" in result

    def test_negative_behind_is_fresh(self):
        result = author_rebase_check(behind_by=-1)
        assert result is None


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------

class TestRebaseStrategy:
    def test_strategy_values(self):
        assert RebaseStrategy.REBASE.value == "rebase"
        assert RebaseStrategy.FAIL.value == "fail"


class TestFreshnessVerdict:
    def test_verdict_values(self):
        assert FreshnessVerdict.FRESH.value == "FRESH"
        assert FreshnessVerdict.STALE.value == "STALE"


# ---------------------------------------------------------------------------
# FreshnessResult structure
# ---------------------------------------------------------------------------

class TestFreshnessResult:
    def test_fresh_result_fields(self):
        result = FreshnessResult(
            verdict=FreshnessVerdict.FRESH,
            behind_by=0,
            ahead_by=5,
            suggestion="All good",
        )
        assert result.verdict == FreshnessVerdict.FRESH
        assert result.behind_by == 0
        assert result.ahead_by == 5
        assert result.suggestion == "All good"

    def test_stale_result_fields(self):
        result = FreshnessResult(
            verdict=FreshnessVerdict.STALE,
            behind_by=3,
            ahead_by=2,
            suggestion="Rebase needed",
        )
        assert result.verdict == FreshnessVerdict.STALE
        assert result.behind_by == 3

    def test_result_is_frozen(self):
        result = FreshnessResult(
            verdict=FreshnessVerdict.FRESH,
            behind_by=0,
            ahead_by=1,
            suggestion="ok",
        )
        with pytest.raises(Exception):
            result.behind_by = 5  # type: ignore
