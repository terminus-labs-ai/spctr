"""PR review workflow — merge gate and claim routing (FR4-FR5).

Implements the reviewer-side workflow for the agent-driven PR merge flow:
- FR4: Claim routing enforcement (an agent must never review its own PR).
- FR5: Three-part merge gate (test suite → solid-review → judgment).

This module is pure logic — no bash, no external process spawning.
The cron-dispatched agent invokes these checks as part of its Reviewer
workflow, guided by squadron-rules.md.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# FR4 — Claim routing
# ---------------------------------------------------------------------------

class ClaimRoutingError(ValueError):
    """Raised when an agent attempts to claim a review bead for its own PR.

    An agent must never review its own work. The existing claim logic
    routes review beads to the peer agent via assignee, but this error
    is the programmatic guard — if routing fails, this is thrown.
    """

    def __init__(self, reviewer: str, author: str) -> None:
        self.reviewer = reviewer
        self.author = author
        super().__init__(
            f"self-review prevented: {reviewer} cannot review a PR authored by {author}"
        )


def check_self_review(reviewer_agent: str, pr_author_agent: str) -> None:
    """Validate that the reviewer is not the PR author.

    Args:
        reviewer_agent: Name of the agent attempting the review (e.g., "edi").
        pr_author_agent: Name of the agent who authored the PR (e.g., "tali").

    Raises:
        ClaimRoutingError: If the reviewer and author are the same agent.
    """
    if reviewer_agent == pr_author_agent:
        raise ClaimRoutingError(reviewer=reviewer_agent, author=pr_author_agent)


# ---------------------------------------------------------------------------
# FR5 — Three-part merge gate
# ---------------------------------------------------------------------------

class GateStep(enum.Enum):
    """Ordered steps of the merge gate: cheap → expensive (fail fast).

    The reviewer must pass ALL three steps for a MERGE verdict.
    Failure at any step short-circuits — later steps are skipped.
    """

    TEST_SUITE = "test_suite"
    SOLID_REVIEW = "solid_review"
    JUDGMENT = "judgment"


class GateVerdict(enum.Enum):
    """Machine-recognizable verdict emitted by the merge gate."""

    MERGE = "MERGE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class GateFinding:
    """A single finding from a gate step.

    Attributes:
        step: Which gate step produced this finding.
        message: Human-readable description of the issue.
    """

    step: str
    message: str


@dataclass(frozen=True)
class GateResult:
    """Outcome of the three-part merge gate.

    Attributes:
        verdict: MERGE or REJECT.
        findings: List of (step, message) pairs for each failed check.
                  Empty if verdict is MERGE.
    """

    verdict: GateVerdict
    findings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if the verdict is MERGE."""
        return self.verdict == GateVerdict.MERGE

    @property
    def failures(self) -> list[GateFinding]:
        """Findings as GateFinding objects."""
        return [GateFinding(step=step, message=msg) for step, msg in self.findings]
