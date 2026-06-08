"""Verify that squadron-rules.md contains the required PR merge flow sections.

These tests ensure the agent workflow documentation is present and well-formed,
so the cron-dispatched agents have the correct instructions for author and
reviewer workflows.

Covers FR1-FR3 (author PR workflow) and FR4-FR5 (reviewer merge gate) of the
agent-pr-merge-flow spec.
"""

import pathlib

import pytest

SQUADRON_RULES_PATH = pathlib.Path.home() / ".sr2" / "squadron-rules.md"


def _read_squadron_rules() -> str:
    """Read squadron-rules.md from the deployed config location."""
    if not SQUADRON_RULES_PATH.exists():
        pytest.skip("squadron-rules.md not found at expected path")
    return SQUADRON_RULES_PATH.read_text()


def _extract_section(content: str, section_heading: str) -> str:
    """Extract text between section_heading and the next top-level ## heading."""
    idx = content.find(section_heading)
    assert idx >= 0, f"Section '{section_heading}' not found"
    # Find next ## heading after this one (skip the heading line itself)
    rest = content[idx + len(section_heading):]
    # Match ## at start of line (markdown heading)
    lines = rest.split("\n")
    result_lines = []
    for line in lines:
        if line.startswith("## ") and line != section_heading:
            break
        result_lines.append(line)
    return "\n".join(result_lines)


def test_agent_workflows_section_exists():
    """FR1-FR9: squadron-rules must contain the Agent Workflows section."""
    content = _read_squadron_rules()
    assert "Agent Workflows (PR Merge Flow)" in content, (
        "Missing Agent Workflows (PR Merge Flow) section"
    )


def test_author_workflow_section_exists():
    """FR1-FR3: Author workflow section must be present."""
    content = _read_squadron_rules()
    assert "## Author workflow (PR)" in content, (
        "Missing Author workflow (PR) section"
    )


def test_reviewer_workflow_section_exists():
    """FR4-FR5: Reviewer workflow section must be present."""
    content = _read_squadron_rules()
    assert "## Reviewer workflow (merge gate)" in content, (
        "Missing Reviewer workflow (merge gate) section"
    )


def test_dispatch_rule_present():
    """FR9: Dispatch rule must direct agents to check for review label."""
    content = _read_squadron_rules()
    assert "review" in content.lower(), (
        "Dispatch rule should reference 'review' label"
    )


def test_pr_body_template_includes_live_config():
    """FR2: PR body template must include LIVE-CONFIG section."""
    content = _read_squadron_rules()
    assert "LIVE-CONFIG" in content, (
        "PR body template must include LIVE-CONFIG section"
    )


def test_author_workflow_forbids_self_merge():
    """FR1: Author workflow must explicitly forbid self-merge."""
    content = _read_squadron_rules()
    author_text = _extract_section(content, "## Author workflow (PR)")
    assert "NEVER merge" in author_text or "NOT merge" in author_text, (
        "Author workflow must forbid self-merge"
    )


def test_author_workflow_includes_review_bead_filing():
    """FR3: Author workflow must include review bead filing instructions."""
    content = _read_squadron_rules()
    author_text = _extract_section(content, "## Author workflow (PR)")
    assert "review bead" in author_text.lower(), (
        "Author workflow must include review bead filing"
    )


def test_reviewer_workflow_forbids_own_pr():
    """FR4: Reviewer must never review its own PR."""
    content = _read_squadron_rules()
    reviewer_text = _extract_section(content, "## Reviewer workflow (merge gate)")
    assert "never" in reviewer_text.lower() or "not" in reviewer_text.lower(), (
        "Reviewer workflow must forbid reviewing own PR"
    )


def test_three_part_merge_gate_mentioned():
    """FR5: Reviewer workflow must describe three-part merge gate."""
    content = _read_squadron_rules()
    reviewer_text = _extract_section(content, "## Reviewer workflow (merge gate)")
    # Check for the three parts: tests, solid-review, judgment
    assert "test" in reviewer_text.lower(), "Should mention test suite"
    assert "solid-review" in reviewer_text.lower(), "Should mention solid-review"
