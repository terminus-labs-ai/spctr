"""Load Skill tool — runtime skill loading for agents.

Looks up a skill by name in the SkillRegistry and returns its content.
The agent receives the content as tool output and can include it in its
working context naturally.

This is the runtime wiring that makes the skills/ package (previously
dead code) actually usable by agents at runtime.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sr2_spectre.skills.core import SkillRegistry
from sr2_spectre.tools.output import ToolOutput

logger = logging.getLogger(__name__)


class LoadSkillTool:
    """Load a skill by name and return its content.

    Schema:
        skill_name (str): The skill identifier to load.
        list_only (bool): If True, return available skill names instead.

    The tool looks up the skill in the injected SkillRegistry and returns
    the content text. When list_only is True, it returns a JSON array of
    registered skill names with descriptions.
    """

    name = "load_skill"
    description = (
        "Load a skill by name and return its content. Skills are loadable "
        "knowledge packages (conventions, workflows, procedural knowledge). "
        "Use list_only=true to see available skills. The skill content is "
        "returned as text for you to reference in your work."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "The skill identifier to load (e.g. 'sr2-conventions').",
            },
            "list_only": {
                "type": "boolean",
                "description": (
                    "If true, return available skill names and descriptions "
                    "instead of loading a specific skill."
                ),
            },
        },
        "required": ["skill_name"],
    }

    def __init__(self, registry: SkillRegistry) -> None:
        """Initialize with a SkillRegistry instance.

        Args:
            registry: The SkillRegistry to look up skills from.
        """
        self._registry = registry

    async def __call__(
        self,
        skill_name: str,
        list_only: bool = False,
    ) -> str:
        """Execute the load-skill flow.

        Returns the skill content as a string, or a list of available
        skills when list_only is True.
        """
        if list_only:
            return self._list_skills()

        return self._load_skill(skill_name)

    def _load_skill(self, skill_name: str) -> str:
        """Load a skill by name and return its content."""
        content = self._registry.get_content(skill_name)
        if content is None:
            available = self._registry.list_names()
            return json.dumps({
                "error": f"Skill '{skill_name}' not found.",
                "available_skills": available,
            })

        # Wrap with metadata header so the agent knows what it loaded.
        skill = self._registry.get(skill_name)
        header = f"# Skill: {skill.name} (v{skill.version})\n"
        header += f"> {skill.description}\n\n"
        return header + content

    def _list_skills(self) -> str:
        """Return available skills as formatted text."""
        names = self._registry.list_names()
        if not names:
            return "No skills registered."

        lines: list[str] = ["## Available Skills\n"]
        for name in names:
            skill = self._registry.get(name)
            if skill:
                tags = ", ".join(skill.tags) if skill.tags else "none"
                lines.append(
                    f"- **{name}** (v{skill.version}): {skill.description} "
                    f"`[{tags}]`"
                )
        return "\n".join(lines)
