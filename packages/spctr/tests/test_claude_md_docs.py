"""Tests that CLAUDE.md scaffold placeholders have been replaced with real docs.

Covers bead obsidian-n1r3. These are documentation-content assertions: they
read the text of CLAUDE.md and check for required / forbidden substrings.

Acceptance criteria (from the bead body):
  AC1. The `## Build & Test` section states the real command
       `.venv/bin/python -m pytest` and no longer contains the scaffold
       placeholder strings `# npm install` or `# npm test`.
  AC2. The `## Architecture Overview` section no longer contains the
       placeholder `_Add a brief overview of your project architecture_`,
       and its content mentions: the resolver/transformer pipeline, the
       4-tier config resolution, the interface/runtime split, and the
       additive-merge-with-no-subtraction behaviour of `agent.tools` plus
       its security implication.
"""

from pathlib import Path

CLAUDE_MD = Path(__file__).parent.parent / "CLAUDE.md"


def _read_claude_md() -> str:
    return CLAUDE_MD.read_text(encoding="utf-8")


def _section(text: str, header: str) -> str:
    """Return the body of a level-2 `## <header>` section.

    The body runs from the header line up to (but not including) the next
    level-2 `## ` header. Level-3 (`### `) subheaders stay inside the section.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"## {header}":
            start = i
            break
    assert start is not None, f"Section header ## {header} not found in CLAUDE.md"

    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].lstrip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            end = j
            break
    return "\n".join(lines[start:end])


# =========================================================================
# AC1 — Build & Test section
# =========================================================================


def test_build_test_section_has_real_pytest_command():
    """AC1: the real test command is documented."""
    section = _section(_read_claude_md(), "Build & Test")
    assert ".venv/bin/python -m pytest" in section


def test_build_test_section_has_no_npm_placeholders():
    """AC1: the scaffold npm placeholders are gone."""
    section = _section(_read_claude_md(), "Build & Test")
    assert "# npm install" not in section
    assert "# npm test" not in section


# =========================================================================
# AC2 — Architecture Overview section
# =========================================================================


def test_architecture_overview_has_no_scaffold_placeholder():
    """AC2: the unedited scaffold placeholder line is gone."""
    section = _section(_read_claude_md(), "Architecture Overview")
    assert "_Add a brief overview of your project architecture_" not in section


def test_architecture_overview_mentions_resolver_transformer_pipeline():
    """AC2: describes the resolver/transformer pipeline."""
    section = _section(_read_claude_md(), "Architecture Overview").lower()
    assert "resolver" in section
    assert "transformer" in section


def test_architecture_overview_mentions_four_tier_config_resolution():
    """AC2: describes the 4-tier config resolution."""
    section = _section(_read_claude_md(), "Architecture Overview")
    lower = section.lower()
    assert "tier" in lower
    # The four tiers, lowest to highest priority.
    assert "config.yaml" in section
    assert "spectre.yaml" in section
    assert ".spectre.yaml" in section
    assert "extends" in lower


def test_architecture_overview_mentions_interface_runtime_split():
    """AC2: describes the interface/runtime split (single_shot, tui, discord)."""
    section = _section(_read_claude_md(), "Architecture Overview").lower()
    assert "single_shot" in section or "single-shot" in section
    assert "tui" in section
    assert "discord" in section
    assert "runtime" in section


def test_architecture_overview_states_agent_tools_additive_no_subtract():
    """AC2: agent.tools merges additively with no way to subtract a tool."""
    section = _section(_read_claude_md(), "Architecture Overview")
    lower = section.lower()
    assert "agent.tools" in section
    # Additive named-list merge.
    assert "addit" in lower
    # No way to remove/subtract a tool once granted by a lower-priority tier.
    assert any(w in lower for w in ("subtract", "remove", "cannot remove", "no way to remove"))


def test_architecture_overview_states_tools_merge_security_consequence():
    """AC2: the additive-merge behaviour has a called-out security implication."""
    section = _section(_read_claude_md(), "Architecture Overview").lower()
    assert "security" in section
