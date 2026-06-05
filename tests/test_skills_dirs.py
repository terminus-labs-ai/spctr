"""Tests for directory-based skill discovery (spc-29).

Covers:
- _parse_skill_frontmatter: valid skills, missing name, bad YAML, no frontmatter
- discover_skills_in_dir: multiple files, empty dir, non-existent dir
- discover_skills: multiple directories
- Runtime bootstrap: skills_dirs wiring into SkillRegistry
- Config: skills_dirs field in AgentConfig
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from sr2_spectre.config import (
    AgentConfig,
    ModelConfig,
    SpectreConfig,
)
from sr2_spectre.skills.core import (
    Skill,
    discover_skills,
    discover_skills_in_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKILL_WITH_FRONTMATTER = """\
---
name: my-awesome-skill
description: Does awesome things
version: 1.0.0
tags:
  - awesome
  - utilities
---
# My Awesome Skill

This is the skill body content.
"""

_SKILL_MINIMAL = """\
---
name: minimal-skill
---
Minimal skill content.
"""

_SKILL_NO_NAME = """\
---
description: Has no name
---
Body.
"""

_SKILL_BAD_YAML = """\
---
name: [unclosed
---
Body.
"""

_SKILL_NO_FRONTMATTER = """\
# Just a markdown file

No frontmatter here.
"""


def _base_config(skills_dirs: list[str] | None = None) -> SpectreConfig:
    """Build a minimal SpectreConfig for testing."""
    return SpectreConfig(
        agent=AgentConfig(
            name="test",
            tools=[],
            skills=[],
            skills_dirs=skills_dirs or [],
        ),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
                },
                {
                    "name": "tools",
                    "target": "tools",
                    "resolvers": [],
                    "tool_providers": [{"type": "spectre_tools"}],
                },
                {
                    "name": "conversation",
                    "target": "messages",
                    "resolvers": [{"type": "session"}, {"type": "input"}],
                },
            ]
        },
    )


# ---------------------------------------------------------------------------
# _parse_skill_frontmatter
# ---------------------------------------------------------------------------

class TestParseSkillFrontmatter:
    """_parse_skill_frontmatter extracts skills from markdown with frontmatter."""

    def test_valid_skill(self, tmp_path: Path):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = tmp_path / "skill.md"
        f.write_text(_SKILL_WITH_FRONTMATTER)

        skill = _parse_skill_frontmatter(_SKILL_WITH_FRONTMATTER, f)
        assert skill is not None
        assert skill.name == "my-awesome-skill"
        assert skill.description == "Does awesome things"
        assert skill.version == "1.0.0"
        assert skill.tags == ("awesome", "utilities")
        assert "This is the skill body content" in skill.content

    def test_minimal_skill(self, tmp_path: Path):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = tmp_path / "skill.md"
        skill = _parse_skill_frontmatter(_SKILL_MINIMAL, f)
        assert skill is not None
        assert skill.name == "minimal-skill"
        assert skill.description == "Skill: minimal-skill"
        assert skill.version == "0.1.0"
        assert skill.tags == ()
        assert "Minimal skill content" in skill.content

    def test_no_name_returns_none(self, tmp_path: Path, caplog):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = tmp_path / "skill.md"
        with caplog.at_level(logging.WARNING):
            skill = _parse_skill_frontmatter(_SKILL_NO_NAME, f)
        assert skill is None
        assert "No 'name' in frontmatter" in caplog.text

    def test_no_frontmatter_returns_none(self, tmp_path: Path, caplog):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = tmp_path / "skill.md"
        with caplog.at_level(logging.WARNING):
            skill = _parse_skill_frontmatter(_SKILL_NO_FRONTMATTER, f)
        assert skill is None
        assert "No frontmatter" in caplog.text

    def test_bad_yaml_returns_none(self, tmp_path: Path, caplog):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = tmp_path / "skill.md"
        with caplog.at_level(logging.WARNING):
            skill = _parse_skill_frontmatter(_SKILL_BAD_YAML, f)
        assert skill is None
        assert "YAML parse error" in caplog.text

    def test_tags_as_string(self, tmp_path: Path):
        """Tags can be a comma-separated string in frontmatter."""
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        content = """\
---
name: string-tags
tags: alpha, beta, gamma
---
Body.
"""
        f = tmp_path / "skill.md"
        skill = _parse_skill_frontmatter(content, f)
        assert skill is not None
        assert skill.tags == ("alpha", "beta", "gamma")


# ---------------------------------------------------------------------------
# discover_skills_in_dir
# ---------------------------------------------------------------------------

class TestDiscoverSkillsInDir:
    """discover_skills_in_dir scans a directory for skill files."""

    def test_discovers_multiple_skills(self, tmp_path: Path):
        (tmp_path / "skill-a.md").write_text(_SKILL_WITH_FRONTMATTER.replace(
            "my-awesome-skill", "skill-a"
        ))
        (tmp_path / "skill-b.md").write_text(_SKILL_MINIMAL.replace(
            "minimal-skill", "skill-b"
        ))

        skills = discover_skills_in_dir(tmp_path)
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"skill-a", "skill-b"}

    def test_skips_non_skill_files(self, tmp_path: Path, caplog):
        """Files without valid frontmatter are skipped with a warning."""
        (tmp_path / "good.md").write_text(_SKILL_MINIMAL)
        (tmp_path / "readme.md").write_text("# Just a readme\n\nNo frontmatter.")
        (tmp_path / "no-name.md").write_text(_SKILL_NO_NAME)

        skills = discover_skills_in_dir(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "minimal-skill"
        assert "No frontmatter" in caplog.text
        assert "No 'name' in frontmatter" in caplog.text

    def test_empty_directory(self, tmp_path: Path):
        skills = discover_skills_in_dir(tmp_path)
        assert skills == []

    def test_non_existent_directory(self, tmp_path: Path, caplog):
        with caplog.at_level(logging.WARNING):
            skills = discover_skills_in_dir(tmp_path / "does-not-exist")
        assert skills == []
        assert "does not exist" in caplog.text.lower()

    def test_only_globs_md_files(self, tmp_path: Path):
        """Only *.md files are discovered; other extensions are ignored."""
        (tmp_path / "skill.md").write_text(_SKILL_MINIMAL)
        (tmp_path / "skill.txt").write_text(_SKILL_MINIMAL)
        (tmp_path / "skill.py").write_text("# python file")

        skills = discover_skills_in_dir(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "minimal-skill"


# ---------------------------------------------------------------------------
# discover_skills (multi-directory)
# ---------------------------------------------------------------------------

class TestDiscoverSkills:
    """discover_skills scans multiple directories."""

    def test_aggregates_from_multiple_dirs(self, tmp_path: Path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "skill-1.md").write_text(_SKILL_MINIMAL.replace(
            "minimal-skill", "skill-1"
        ))
        (dir_b / "skill-2.md").write_text(_SKILL_MINIMAL.replace(
            "minimal-skill", "skill-2"
        ))

        skills = discover_skills([str(dir_a), str(dir_b)])
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"skill-1", "skill-2"}

    def test_empty_list(self):
        skills = discover_skills([])
        assert skills == []


# ---------------------------------------------------------------------------
# Runtime bootstrap with skills_dirs
# ---------------------------------------------------------------------------

class TestRuntimeSkillsDirsBootstrap:
    """Runtime discovers skills from skills_dirs during bootstrap."""

    def test_skills_dirs_loaded(self, tmp_path: Path):
        from sr2_spectre.runtime import Runtime

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "custom.md").write_text(_SKILL_WITH_FRONTMATTER)

        cfg = _base_config(skills_dirs=[str(skills_dir)])

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "my-awesome-skill" in runtime.skill_registry
        skill = runtime.skill_registry.get("my-awesome-skill")
        assert skill is not None
        assert skill.description == "Does awesome things"
        assert skill.tags == ("awesome", "utilities")

    def test_skills_dirs_and_default_coexist(self, tmp_path: Path):
        """Discovered skills and DEFAULT_SKILLS both present."""
        from sr2_spectre.runtime import Runtime

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "extra.md").write_text(_SKILL_MINIMAL)

        cfg = _base_config(skills_dirs=[str(skills_dir)])

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        # Builtin should still be there
        assert "sr2-conventions" in runtime.skill_registry
        # Discovered should also be there
        assert "minimal-skill" in runtime.skill_registry

    def test_skills_dirs_can_override_default(self, tmp_path: Path):
        """A discovered skill with the same name as a default overwrites it."""
        from sr2_spectre.runtime import Runtime

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "override.md").write_text(
            """\
---
name: sr2-conventions
description: My custom conventions override
version: 2.0.0
---
Custom conventions content.
"""
        )

        cfg = _base_config(skills_dirs=[str(skills_dir)])

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        skill = runtime.skill_registry.get("sr2-conventions")
        assert skill is not None
        assert skill.description == "My custom conventions override"
        assert skill.version == "2.0.0"

    def test_skills_dirs_empty(self, tmp_path: Path):
        """Empty skills_dirs doesn't crash."""
        from sr2_spectre.runtime import Runtime

        cfg = _base_config(skills_dirs=[])

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        # Only defaults should be present
        assert "sr2-conventions" in runtime.skill_registry
        assert "minimal-skill" not in runtime.skill_registry

    def test_load_skill_tool_sees_discovered_skills(self, tmp_path: Path):
        """The load_skill tool can discover and load discovered skills."""
        from sr2_spectre.runtime import Runtime

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "discoverable.md").write_text(_SKILL_WITH_FRONTMATTER)

        cfg = _base_config(skills_dirs=[str(skills_dir)])

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        # Verify via the load_skill tool
        spec = runtime.registry._tools["load_skill"]
        import asyncio
        result = asyncio.run(spec.fn("my-awesome-skill"))
        assert "Skill: my-awesome-skill" in result
        assert "Does awesome things" in result
        assert "This is the skill body content" in result

    def test_load_skill_list_includes_discovered(self, tmp_path: Path):
        """Listing skills includes discovered ones."""
        from sr2_spectre.runtime import Runtime

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "listed.md").write_text(_SKILL_MINIMAL)

        cfg = _base_config(skills_dirs=[str(skills_dir)])

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        spec = runtime.registry._tools["load_skill"]
        import asyncio
        result = asyncio.run(spec.fn("dummy", list_only=True))
        assert "minimal-skill" in result
        assert "sr2-conventions" in result


# ---------------------------------------------------------------------------
# AgentConfig.skills_dirs model
# ---------------------------------------------------------------------------

class TestAgentConfigSkillsDirs:
    """AgentConfig.skills_dirs is a list[str] with sensible defaults."""

    def test_default_empty_list(self):
        agent = AgentConfig(name="test")
        assert agent.skills_dirs == []

    def test_accepts_paths(self):
        agent = AgentConfig(
            name="test",
            skills_dirs=["~/.claude/skills", "/opt/skills"],
        )
        assert len(agent.skills_dirs) == 2
        assert "~/.claude/skills" in agent.skills_dirs
        assert "/opt/skills" in agent.skills_dirs

    def test_in_spectre_config(self):
        cfg = SpectreConfig(
            agent=AgentConfig(
                name="test",
                skills_dirs=["/custom/skills"],
            ),
            models={"default": ModelConfig(model="test", base_url="http://test")},
            pipeline={
                "layers": [
                    {
                        "name": "system",
                        "target": "system",
                        "resolvers": [{"type": "static", "config": {"text": "hi"}}],
                    },
                ]
            },
        )
        assert cfg.agent.skills_dirs == ["/custom/skills"]


# ---------------------------------------------------------------------------
# Per-file skills[] still work alongside skills_dirs
# ---------------------------------------------------------------------------

class TestPerFileAndDirsCoexist:
    """Per-file agent.skills[] and agent.skills_dirs[] both work."""

    def test_both_paths_loaded(self, tmp_path: Path):
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import SkillConfig

        # Create a skill file for per-file config
        per_file_skill = tmp_path / "per-file.md"
        per_file_skill.write_text("# Per-file skill content\n")

        # Create a directory with a discovered skill
        skills_dir = tmp_path / "discovered"
        skills_dir.mkdir()
        (skills_dir / "discovered.md").write_text(_SKILL_MINIMAL)

        cfg = _base_config(skills_dirs=[str(skills_dir)])
        cfg.agent.skills = [
            SkillConfig(
                name="per-file-skill",
                path=str(per_file_skill),
                description="Declared in skills[]",
            )
        ]

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "per-file-skill" in runtime.skill_registry
        assert "minimal-skill" in runtime.skill_registry
        assert "sr2-conventions" in runtime.skill_registry

    def test_per_file_overrides_discovered(self, tmp_path: Path):
        """Per-file skills[] loaded after skills_dirs, so they can override."""
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import SkillConfig

        # Create a discovered skill named "override-target"
        skills_dir = tmp_path / "discovered"
        skills_dir.mkdir()
        (skills_dir / "target.md").write_text(
            """\
---
name: override-target
description: From directory
version: 1.0.0
---
Discovered content.
"""
        )

        # Create a per-file skill with the same name
        per_file_skill = tmp_path / "override.md"
        per_file_skill.write_text("# Per-file override content\n")

        cfg = _base_config(skills_dirs=[str(skills_dir)])
        cfg.agent.skills = [
            SkillConfig(
                name="override-target",
                path=str(per_file_skill),
                description="From per-file config",
                version="2.0.0",
            )
        ]

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        skill = runtime.skill_registry.get("override-target")
        assert skill is not None
        # Per-file loads after dirs, so it wins
        assert skill.description == "From per-file config"
        assert skill.version == "2.0.0"
