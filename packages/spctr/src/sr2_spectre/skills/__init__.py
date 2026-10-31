"""Skill system — loadable knowledge packages for agents.

A skill is a named bundle of conventions, workflows, and procedural knowledge
that an agent can load on demand. Skills are the generalization of the
progressive-disclosure pattern (planning-guide.md loaded via file_read) into
a first-class, composable mechanism.

Usage:
    from sr2_spectre.skills import Skill, SkillRegistry

    registry = SkillRegistry()
    registry.register(Skill(name="sr2-conventions", ...))
    content = registry.get("sr2-conventions")
"""

from __future__ import annotations

from sr2_spectre.skills.core import Skill, SkillRegistry

__all__ = ["Skill", "SkillRegistry"]
