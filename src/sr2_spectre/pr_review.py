"""PR review workflow — merge gate, claim routing, reject path, and approve path (FR4-FR7).

Implements the reviewer-side workflow for the agent-driven PR merge flow:
- FR4: Claim routing enforcement (an agent must never review its own PR).
- FR5: Three-part merge gate (test suite → solid-review → judgment).
- FR6: Approve path — apply declared+verified LIVE-CONFIG, merge PR, close beads.
- FR7: Reject path — findings back to author, capped 3-round loop, bd human escalation.

This module is pure logic — no bash, no external process spawning.
The cron-dispatched agent invokes these checks as part of its Reviewer
workflow, guided by squadron-rules.md.
"""

from __future__ import annotations

import enum
import re
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


# ---------------------------------------------------------------------------
# FR6 — Approve path: apply config (declare + verify), merge, close
# ---------------------------------------------------------------------------

class ConfigChangeType(enum.Enum):
    """Types of config-affecting changes detectable in a PR diff."""

    TOOL = "tool"
    RESOLVER = "resolver"
    TRANSFORMER = "transformer"
    SKILL = "skill"


@dataclass(frozen=True)
class ConfigChange:
    """A config-affecting change detected in a PR diff.

    Attributes:
        change_type: Category of the config change.
        description: Human-readable description of the change.
        file_path: File path where the change was detected.
    """

    change_type: ConfigChangeType
    description: str
    file_path: str


def parse_live_config_section(pr_body: str) -> list[str] | None:
    """Parse the LIVE-CONFIG section from a PR body.

    The PR body template includes a `LIVE-CONFIG:` section that declares
    what config edits the reviewer must apply to ~/.sr2/ on merge.
    Returns a list of edit description lines, or None if the section is
    absent or contains only `none`.

    Args:
        pr_body: Full text of the PR body.

    Returns:
        List of edit description strings, or None if no config changes
        are declared (section absent or reads `none`).
    """
    # Match LIVE-CONFIG: header (case-insensitive)
    header_match = re.search(r"^LIVE-CONFIG\s*:", pr_body, re.MULTILINE | re.IGNORECASE)
    if not header_match:
        return None

    # Extract everything after the header until the end of body
    section_text = pr_body[header_match.end():]

    # Split into lines, strip, filter empty/comment lines
    lines: list[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)

    # Check for "none" sentinel
    if len(lines) == 1 and lines[0].lower() == "none":
        return None

    return lines if lines else None


# Patterns for detecting config-affecting additions in diff text.
# Only additions (lines starting with +) are relevant.
# Header patterns (matched on +++ lines):
_HEADER_PATTERNS: list[tuple[re.Pattern[str], ConfigChangeType, str]] = [
    # New tool: a new .py file in tools/builtins/
    (
        re.compile(r"^\+\+\+ b/src/.*/tools/builtins/.+\.py$"),
        ConfigChangeType.TOOL,
        "new builtin tool module",
    ),
    # New tool: standalone tool module
    (
        re.compile(r"^\+\+\+ b/src/.*/tools/.+\.py$"),
        ConfigChangeType.TOOL,
        "new tool module",
    ),
    # New skill: SKILL.md file being added
    (
        re.compile(r"^\+\+\+ b/.*/SKILL\.md$"),
        ConfigChangeType.SKILL,
        "new skill module",
    ),
    # New skill: skill Python module being added
    (
        re.compile(r"^\+\+\+ b/.*/skills/.+\.py$"),
        ConfigChangeType.SKILL,
        "new skill Python module",
    ),
]

# Content patterns for pyproject.toml entry point sections.
# Matched on lines within a +++ b/pyproject.toml hunk.
_ENTRY_POINT_PATTERNS: list[tuple[re.Pattern[str], ConfigChangeType, str]] = [
    (
        re.compile(r"^\+\[project\.entry-points\.\"?sr2\.resolvers\"?\]"),
        ConfigChangeType.RESOLVER,
        "new sr2.resolvers entry point",
    ),
    (
        re.compile(r"^\+\[project\.entry-points\.\"?sr2\.transformers\"?\]"),
        ConfigChangeType.TRANSFORMER,
        "new sr2.transformers entry point",
    ),
]


def scan_diff_for_config_changes(diff: str) -> list[ConfigChange]:
    """Scan a unified diff for config-affecting changes not declared in LIVE-CONFIG.

    Detects:
    - New tool modules (files added under tools/builtins/ or tools/)
    - New sr2.resolvers or sr2.transformers entry points in pyproject.toml
    - New skill modules (SKILL.md files or skill .py files)

    Only additions are flagged. Deletions are ignored.

    Args:
        diff: Unified diff text (e.g., output of `git diff base...head`).

    Returns:
        List of ConfigChange objects for each detected config-affecting
        addition. Empty list if nothing detected.
    """
    changes: list[ConfigChange] = []
    seen_signatures: set[str] = set()

    lines = diff.splitlines()
    current_file: str | None = None

    for line in lines:
        # Track which file we're in from +++ headers
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("+++ /dev/null"):
            current_file = None

        # Check header patterns against +++ lines
        if line.startswith("+++ "):
            for pattern, change_type, description in _HEADER_PATTERNS:
                if pattern.search(line):
                    sig = f"{change_type.value}:{current_file}"
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        changes.append(
                            ConfigChange(
                                change_type=change_type,
                                description=description,
                                file_path=current_file or "",
                            )
                        )
                    break

        # Check pyproject.toml content for entry point additions
        if current_file and "pyproject.toml" in current_file:
            for pattern, change_type, description in _ENTRY_POINT_PATTERNS:
                if pattern.search(line):
                    sig = f"{change_type.value}:{current_file}"
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        changes.append(
                            ConfigChange(
                                change_type=change_type,
                                description=description,
                                file_path=current_file,
                            )
                    )
                    break

    return changes


# ---------------------------------------------------------------------------
# FR7 — Reject path: findings back to author, capped 3-round loop
# ---------------------------------------------------------------------------

MAX_REVIEW_ROUNDS: int = 3
"""Default cap on reject rounds before escalating to Diego via bd human."""


class RejectAction(enum.Enum):
    """Outcome of a reject decision: bounce back to author or escalate."""

    BOUNCE = "BOUNCE"
    ESCALATE = "ESCALATE"


def should_escalate(review_round: int, max_rounds: int = MAX_REVIEW_ROUNDS) -> bool:
    """Determine whether to escalate to Diego instead of bouncing.

    After `max_rounds` rejected review rounds, the reviewer flags `bd human`
    for Diego rather than bouncing the impl bead back to the author again.

    Args:
        review_round: The current review round number (1-indexed).
        max_rounds: Maximum rounds before escalation (default 3).

    Returns:
        True if the round count has reached or exceeded the cap.
    """
    return review_round >= max_rounds


def build_reject_body(
    findings: list[GateFinding],
    review_round: int,
) -> str:
    """Format gate findings into a PR review body for `gh pr review --request-changes`.

    Produces a structured markdown body that the reviewer posts to the PR.
    Each finding is listed with its gate step and message. The review round
    is noted so the author knows how many attempts remain.

    Args:
        findings: List of GateFinding objects from the failed gate step(s).
        review_round: Current review round number (1-indexed).

    Returns:
        Formatted markdown string for the PR review comment.
    """
    lines: list[str] = []
    lines.append(f"**Verdict: REJECT** — Round {review_round}/{MAX_REVIEW_ROUNDS}")
    lines.append("")

    if findings:
        lines.append("## Findings")
        lines.append("")
        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. **{finding.step}**: {finding.message}")
        lines.append("")
    else:
        lines.append("Gate failed but no specific findings were recorded.")
        lines.append("")

    if review_round >= MAX_REVIEW_ROUNDS:
        lines.append(
            "> ⚠ This review has been rejected "
            f"{MAX_REVIEW_ROUNDS} times. Escalating to Diego via `bd human`."
        )
    else:
        lines.append(
            f"Please address the findings above and push a fix to the same branch. "
            f"Remaining attempts: {MAX_REVIEW_ROUNDS - review_round}."
        )

    return "\n".join(lines)


@dataclass(frozen=True)
class RejectOutcome:
    """Structured output of a reject decision.

    Attributes:
        action: BOUNCE (reassign to author) or ESCALATE (flag bd human).
        review_round_after: The incremented round counter after this rejection.
        author_agent: Name of the author agent to receive the bounce.
        impl_bead_id: The implementation bead ID to reassign or escalate.
        review_bead_id: The review bead ID to close.
        pr_number: GitHub PR number (stays open on reject).
        findings: All gate findings from this review round.
        reject_body: Formatted markdown for the PR review comment.
        escalation_message: Human-readable escalation note (only when ESCALATE).
    """

    action: RejectAction
    review_round_after: int
    author_agent: str
    impl_bead_id: str
    review_bead_id: str
    pr_number: int
    findings: list[GateFinding]
    reject_body: str
    escalation_message: str | None = None


def handle_reject(
    review_round: int,
    max_rounds: int = MAX_REVIEW_ROUNDS,
    *,
    author_agent: str,
    impl_bead_id: str,
    review_bead_id: str,
    pr_number: int,
    findings: list[GateFinding],
) -> RejectOutcome:
    """Process a gate rejection and decide whether to bounce or escalate.

    On a reject:
    - Rounds 1..N-1: Bounce the impl bead back to the author with findings
      in notes, increment the round counter, close the review bead.
    - Round N (cap): Post findings to PR, flag `bd human` for Diego instead
      of bouncing again. Close the review bead.

    The PR stays open in both cases.

    Args:
        review_round: Current review round number (1-indexed).
        max_rounds: Maximum rounds before escalation (default 3).
        author_agent: Name of the author agent (e.g., "tali").
        impl_bead_id: Implementation bead ID (e.g., "spc-50").
        review_bead_id: Review bead ID (e.g., "spc-51").
        pr_number: GitHub PR number.
        findings: Gate findings from the failed review.

    Returns:
        RejectOutcome with the action, incremented round, and formatted body.
    """
    reject_body = build_reject_body(findings=findings, review_round=review_round)
    escalate = should_escalate(review_round, max_rounds)

    if escalate:
        round_after = review_round  # Don't increment past the cap
        escalation_msg = (
            f"PR #{pr_number} rejected {review_round} times (cap: {max_rounds}). "
            f"Impl bead {impl_bead_id} (author: {author_agent}) needs human intervention. "
            f"Findings: {'; '.join(f.message for f in findings)}"
        )
    else:
        round_after = review_round + 1
        escalation_msg = None

    return RejectOutcome(
        action=RejectAction.ESCALATE if escalate else RejectAction.BOUNCE,
        review_round_after=round_after,
        author_agent=author_agent,
        impl_bead_id=impl_bead_id,
        review_bead_id=review_bead_id,
        pr_number=pr_number,
        findings=findings,
        reject_body=reject_body,
        escalation_message=escalation_msg,
    )
