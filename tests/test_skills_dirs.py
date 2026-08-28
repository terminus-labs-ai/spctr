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

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
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

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
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

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        skill = runtime.skill_registry.get("sr2-conventions")
        assert skill is not None
        assert skill.description == "My custom conventions override"
        assert skill.version == "2.0.0"

    def test_skills_dirs_empty(self, tmp_path: Path):
        """Empty skills_dirs doesn't crash."""
        from sr2_spectre.runtime import Runtime

        cfg = _base_config(skills_dirs=[])

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
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

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
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

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
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

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
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

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        skill = runtime.skill_registry.get("override-target")
        assert skill is not None
        # Per-file loads after dirs, so it wins
        assert skill.description == "From per-file config"
        assert skill.version == "2.0.0"


# ---------------------------------------------------------------------------
# Per-file skills[] path interpolation (obsidian-4eon)
# ---------------------------------------------------------------------------

class TestPerFileSkillsEnvVar:
    """agent.skills[].path interpolates ${VAR} like skills_dirs does."""

    def test_per_file_skill_env_var_path(self, tmp_path: Path):
        """`${SR2_WORKSPACE}/.agents/skills/x/SKILL.md` — the Grindforge case."""
        from sr2_spectre.config import SkillConfig
        from sr2_spectre.runtime import Runtime

        skill_dir = tmp_path / ".agents" / "skills" / "godot-shader"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Shader skill\n\nContent.\n")

        cfg = _base_config()
        cfg.agent.skills = [
            SkillConfig(
                name="godot-shader",
                path="${SR2_WORKSPACE}/.agents/skills/godot-shader/SKILL.md",
            )
        ]

        with patch.dict("os.environ", {"SR2_WORKSPACE": str(tmp_path)}), \
             patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "godot-shader" in runtime.skill_registry

    def test_per_file_skill_unset_env_var_warns_and_skips(self, tmp_path: Path, caplog, monkeypatch):
        """An unset ${VAR} skips the skill with a warning — like skills_dirs."""
        from sr2_spectre.config import SkillConfig
        from sr2_spectre.runtime import Runtime

        monkeypatch.delenv("SR2_DOES_NOT_EXIST_VAR", raising=False)
        cfg = _base_config()
        cfg.agent.skills = [
            SkillConfig(
                name="unresolvable",
                path="${SR2_DOES_NOT_EXIST_VAR}/skills/x.md",
            )
        ]

        with caplog.at_level(logging.WARNING), \
             patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "unresolvable" not in runtime.skill_registry
        assert "unresolvable" in caplog.text

    def test_per_file_skill_tilde_path(self, tmp_path: Path, monkeypatch):
        """Tilde paths keep working through the new resolution path."""
        from sr2_spectre.config import SkillConfig
        from sr2_spectre.runtime import Runtime

        (tmp_path / "tilde-skill.md").write_text("# Tilde\n\nContent.\n")
        monkeypatch.setenv("HOME", str(tmp_path))

        cfg = _base_config()
        cfg.agent.skills = [
            SkillConfig(name="tilde-skill", path="~/tilde-skill.md")
        ]

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "tilde-skill" in runtime.skill_registry


# ---------------------------------------------------------------------------
# Bundled <name>/SKILL.md layout (obsidian-8si8)
# ---------------------------------------------------------------------------

_BUNDLED_WITH_FRONTMATTER = """\
---
name: from-frontmatter
description: Frontmatter description
version: 2.0.0
tags: [alpha]
---
Bundled body.
"""

_BUNDLED_NO_FRONTMATTER = """\
# Godot Shader Skill

You are an expert at writing Godot 4.x shaders for 2D effects.

## Shader Types

More detail follows.
"""


def _bundle(root: Path, name: str, text: str) -> Path:
    """Write a bundled skill at ``root/<name>/SKILL.md``."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "SKILL.md"
    f.write_text(text)
    return f


class TestDeriveDescription:
    """_derive_description produces a usable, non-empty description."""

    def test_first_paragraph_after_heading(self):
        from sr2_spectre.skills.core import _derive_description

        body = "# Heading\n\nThe first real paragraph.\n\nA second one.\n"
        assert _derive_description(body, "x") == "The first real paragraph."

    def test_skips_fenced_code(self):
        from sr2_spectre.skills.core import _derive_description

        body = "# Heading\n\n```glsl\nshader_type canvas_item;\n```\n\nProse at last.\n"
        assert _derive_description(body, "x") == "Prose at last."

    def test_joins_wrapped_lines_of_one_paragraph(self):
        from sr2_spectre.skills.core import _derive_description

        body = "Line one\nline two\n\nSecond paragraph.\n"
        assert _derive_description(body, "x") == "Line one line two"

    def test_strips_blockquote_marker(self):
        from sr2_spectre.skills.core import _derive_description

        assert _derive_description("> Quoted intro.\n", "x") == "Quoted intro."

    def test_truncates_long_paragraph(self):
        from sr2_spectre.skills.core import _DESCRIPTION_MAX_CHARS, _derive_description

        body = "word " * 200
        out = _derive_description(body, "x")
        assert len(out) <= _DESCRIPTION_MAX_CHARS + 1  # +1 for the ellipsis
        assert out.endswith("…")

    def test_last_resort_when_no_prose(self):
        from sr2_spectre.skills.core import _derive_description

        assert _derive_description("# Only a heading\n", "my-skill") == "Skill: my-skill"
        assert _derive_description("", "my-skill") == "Skill: my-skill"

    def test_never_returns_empty(self):
        """Skill.__post_init__ raises on an empty description."""
        from sr2_spectre.skills.core import _derive_description

        for body in ("", "   \n\n  ", "#\n", "```\ncode\n```\n"):
            assert _derive_description(body, "n")


class TestBundledSkillParsing:
    """_parse_skill_frontmatter is lenient only when given a fallback_name."""

    def test_frontmatter_wins_over_path(self, tmp_path: Path):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = _bundle(tmp_path, "dir-name", _BUNDLED_WITH_FRONTMATTER)
        skill = _parse_skill_frontmatter(
            _BUNDLED_WITH_FRONTMATTER, f, fallback_name="dir-name"
        )
        assert skill is not None
        assert skill.name == "from-frontmatter"
        assert skill.description == "Frontmatter description"
        assert skill.version == "2.0.0"
        assert skill.tags == ("alpha",)

    def test_no_frontmatter_named_from_directory(self, tmp_path: Path):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = _bundle(tmp_path, "godot-shader", _BUNDLED_NO_FRONTMATTER)
        skill = _parse_skill_frontmatter(
            _BUNDLED_NO_FRONTMATTER, f, fallback_name="godot-shader"
        )
        assert skill is not None
        assert skill.name == "godot-shader"
        assert skill.description == (
            "You are an expert at writing Godot 4.x shaders for 2D effects."
        )
        assert skill.version == "0.1.0"
        assert skill.tags == ()
        # No frontmatter to strip: content is the whole file.
        assert skill.content == _BUNDLED_NO_FRONTMATTER

    def test_frontmatter_without_name_falls_back(self, tmp_path: Path):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = _bundle(tmp_path, "bundle-dir", _SKILL_NO_NAME)
        skill = _parse_skill_frontmatter(
            _SKILL_NO_NAME, f, fallback_name="bundle-dir"
        )
        assert skill is not None
        assert skill.name == "bundle-dir"
        # A description WAS supplied, so it is honoured.
        assert skill.description == "Has no name"

    def test_bad_yaml_falls_back(self, tmp_path: Path):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = _bundle(tmp_path, "bundle-dir", _SKILL_BAD_YAML)
        skill = _parse_skill_frontmatter(
            _SKILL_BAD_YAML, f, fallback_name="bundle-dir"
        )
        assert skill is not None
        assert skill.name == "bundle-dir"

    def test_non_mapping_frontmatter_falls_back(self, tmp_path: Path):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        text = "---\n- a\n- b\n---\nBody prose.\n"
        f = _bundle(tmp_path, "bundle-dir", text)
        skill = _parse_skill_frontmatter(text, f, fallback_name="bundle-dir")
        assert skill is not None
        assert skill.name == "bundle-dir"
        assert skill.description == "Body prose."

    def test_whitespace_only_name_falls_back(self, tmp_path: Path):
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        text = "---\nname: '   '\n---\nBody prose.\n"
        f = _bundle(tmp_path, "bundle-dir", text)
        skill = _parse_skill_frontmatter(text, f, fallback_name="bundle-dir")
        assert skill is not None
        assert skill.name == "bundle-dir"

    def test_flat_form_still_requires_frontmatter(self, tmp_path: Path, caplog):
        """No fallback_name means the old strict behavior, unchanged."""
        from sr2_spectre.skills.core import _parse_skill_frontmatter

        f = tmp_path / "notes.md"
        with caplog.at_level(logging.WARNING):
            skill = _parse_skill_frontmatter(_SKILL_NO_FRONTMATTER, f)
        assert skill is None
        assert "No frontmatter" in caplog.text
        assert "notes.md" in caplog.text


class TestBundledSkillDiscovery:
    """discover_skills_in_dir accepts both layouts and nothing else."""

    def test_discovers_bundled_skill(self, tmp_path: Path):
        _bundle(tmp_path, "godot-shader", _BUNDLED_NO_FRONTMATTER)

        skills = discover_skills_in_dir(str(tmp_path))
        assert [s.name for s in skills] == ["godot-shader"]

    def test_flat_without_frontmatter_still_skipped(self, tmp_path: Path):
        (tmp_path / "README.md").write_text("# Just a readme\n\nNot a skill.\n")

        assert discover_skills_in_dir(str(tmp_path)) == []

    def test_ignores_siblings_of_skill_md(self, tmp_path: Path):
        """README.md / SECURITY.md beside a real skill — caveman marketplace."""
        _bundle(tmp_path, "caveman", _BUNDLED_NO_FRONTMATTER)
        (tmp_path / "caveman" / "README.md").write_text(
            "---\nname: sneaky-readme\n---\nBody.\n"
        )
        (tmp_path / "caveman" / "SECURITY.md").write_text(
            "---\nname: sneaky-security\n---\nBody.\n"
        )

        names = [s.name for s in discover_skills_in_dir(str(tmp_path))]
        assert names == ["caveman"]

    def test_ignores_support_fragment_directories(self, tmp_path: Path):
        """`_shared/*.md` fragments — jobhunt-skills."""
        _bundle(tmp_path, "apply", _BUNDLED_NO_FRONTMATTER)
        shared = tmp_path / "_shared"
        shared.mkdir()
        (shared / "chrome-mcp.md").write_text(
            "---\nname: chrome-mcp\n---\nFragment.\n"
        )
        (shared / "writing-rules.md").write_text("# Rules\n\nProse.\n")

        names = [s.name for s in discover_skills_in_dir(str(tmp_path))]
        assert names == ["apply"]

    def test_ignores_depth_three_resources(self, tmp_path: Path):
        """`<name>/references/*.md` bundled resources — godot-interactive."""
        _bundle(tmp_path, "godot-interactive", _BUNDLED_WITH_FRONTMATTER)
        refs = tmp_path / "godot-interactive" / "references"
        refs.mkdir()
        (refs / "live-editor-tool-map.md").write_text(
            "---\nname: tool-map\n---\nReference.\n"
        )
        nested = tmp_path / "godot-interactive" / "nested" / "deeper"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("---\nname: too-deep\n---\nBody.\n")

        names = [s.name for s in discover_skills_in_dir(str(tmp_path))]
        assert names == ["from-frontmatter"]

    def test_both_layouts_coexist(self, tmp_path: Path):
        (tmp_path / "flat-skill.md").write_text(
            _SKILL_WITH_FRONTMATTER.replace("my-awesome-skill", "flat-skill")
        )
        _bundle(tmp_path, "bundled-skill", _BUNDLED_NO_FRONTMATTER)

        names = [s.name for s in discover_skills_in_dir(str(tmp_path))]
        assert sorted(names) == ["bundled-skill", "flat-skill"]

    def test_ordering_is_deterministic(self, tmp_path: Path):
        for n in ("c", "a", "b"):
            _bundle(tmp_path, n, _BUNDLED_NO_FRONTMATTER)
            (tmp_path / f"z{n}.md").write_text(f"---\nname: z{n}\n---\nBody.\n")

        runs = [[s.name for s in discover_skills_in_dir(str(tmp_path))] for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]
        # Flat form first, then bundled; each sorted.
        assert runs[0] == ["za", "zb", "zc", "a", "b", "c"]

    def test_last_registration_wins_within_a_dir(self, tmp_path: Path):
        """A flat file and a bundle claiming one name: bundled is registered last."""
        from sr2_spectre.skills.core import SkillRegistry

        (tmp_path / "dup.md").write_text(
            "---\nname: dup\ndescription: from flat\n---\nBody.\n"
        )
        _bundle(tmp_path, "dup", "---\ndescription: from bundle\n---\nBody.\n")

        registry = SkillRegistry()
        for skill in discover_skills_in_dir(str(tmp_path)):
            registry.register(skill)

        assert len(registry) == 1
        assert registry.get("dup").description == "from bundle"


class TestGrindsourcedShapedTree:
    """The motivating case: a real .agents/skills tree, 4 of 5 without frontmatter."""

    def test_discovers_every_bundle(self, tmp_path: Path):
        bare = (
            "godot-code-gen",
            "godot-live-edit",
            "godot-scene-design",
            "godot-shader",
        )
        for name in bare:
            _bundle(
                tmp_path,
                name,
                f"# {name} Skill\n\nYou are an expert at {name}.\n",
            )
        _bundle(
            tmp_path,
            "godot-interactive",
            "---\nname: godot-interactive\ndescription: Live editor workflows.\n---\nBody.\n",
        )
        # Bundled resources that must not register.
        refs = tmp_path / "godot-interactive" / "references"
        refs.mkdir()
        (refs / "fast-probe-presets.md").write_text("# Presets\n\nProse.\n")
        (tmp_path / "godot-interactive" / "agents").mkdir()
        (tmp_path / "godot-interactive" / "agents" / "openai.yaml").write_text("a: b\n")

        skills = discover_skills_in_dir(str(tmp_path))

        assert sorted(s.name for s in skills) == [
            "godot-code-gen",
            "godot-interactive",
            "godot-live-edit",
            "godot-scene-design",
            "godot-shader",
        ]
        # Every one carries a description the model can tell apart.
        by_name = {s.name: s for s in skills}
        assert by_name["godot-shader"].description == (
            "You are an expert at godot-shader."
        )
        assert by_name["godot-interactive"].description == "Live editor workflows."

    def test_reaches_the_registry_through_skills_dirs(self, tmp_path: Path):
        from sr2_spectre.runtime import Runtime

        skills_dir = tmp_path / ".agents" / "skills"
        _bundle(skills_dir, "godot-shader", _BUNDLED_NO_FRONTMATTER)

        cfg = _base_config(skills_dirs=[str(skills_dir)])
        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "godot-shader" in runtime.skill_registry

    def test_skills_dirs_interpolates_env_var(self, tmp_path: Path):
        """`${SR2_WORKSPACE}/.agents/skills` is the intended Grindforge config."""
        skills_dir = tmp_path / ".agents" / "skills"
        _bundle(skills_dir, "godot-shader", _BUNDLED_NO_FRONTMATTER)

        skills = discover_skills_in_dir(
            "${SR2_WORKSPACE}/.agents/skills",
            env={"SR2_WORKSPACE": str(tmp_path)},
        )
        assert [s.name for s in skills] == ["godot-shader"]


# ---------------------------------------------------------------------------
# agent.default_skills opt-out (obsidian-4z94)
# ---------------------------------------------------------------------------

class TestDefaultSkillsOptOut:
    """Builtin skills describe SR2 itself; a guest-facing agent can withhold them."""

    def _runtime(self, cfg):
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            return Runtime(config=cfg)

    def test_defaults_registered_by_default(self):
        """Unchanged behavior for every agent that does not opt out."""
        from sr2_spectre.skills.builtin import DEFAULT_SKILLS

        runtime = self._runtime(_base_config())

        for skill in DEFAULT_SKILLS:
            assert skill.name in runtime.skill_registry
        assert "load_skill" in runtime.registry

    def test_opt_out_withholds_builtins(self):
        from sr2_spectre.skills.builtin import DEFAULT_SKILLS

        cfg = _base_config()
        cfg.agent.default_skills = False
        runtime = self._runtime(cfg)

        for skill in DEFAULT_SKILLS:
            assert skill.name not in runtime.skill_registry
        assert len(runtime.skill_registry) == 0

    def test_opt_out_also_withholds_load_skill(self):
        """A tool whose only answer is 'No skills registered.' is dead weight."""
        cfg = _base_config()
        cfg.agent.default_skills = False
        runtime = self._runtime(cfg)

        assert "load_skill" not in runtime.registry

    def test_opt_out_keeps_load_skill_when_dirs_supply_skills(self, tmp_path: Path):
        """The Grindforge case: studio skills yes, SR2 internals no."""
        _bundle(tmp_path, "godot-shader", _BUNDLED_NO_FRONTMATTER)

        cfg = _base_config(skills_dirs=[str(tmp_path)])
        cfg.agent.default_skills = False
        runtime = self._runtime(cfg)

        assert "godot-shader" in runtime.skill_registry
        assert "sr2-conventions" not in runtime.skill_registry
        assert "solid-review" not in runtime.skill_registry
        assert "load_skill" in runtime.registry

    def test_reload_can_withdraw_builtins_and_the_tool(self):
        """Flipping the flag at runtime must reach an already-running agent."""
        cfg = _base_config()
        runtime = self._runtime(cfg)
        assert "sr2-conventions" in runtime.skill_registry
        assert "load_skill" in runtime.registry

        tightened = _base_config()
        tightened.agent.default_skills = False
        applied = runtime.apply_config(tightened)

        assert "agent.skills" in applied
        assert "sr2-conventions" not in runtime.skill_registry
        assert "load_skill" not in runtime.registry

    def test_reload_can_restore_builtins_and_the_tool(self):
        cfg = _base_config()
        cfg.agent.default_skills = False
        runtime = self._runtime(cfg)
        assert "load_skill" not in runtime.registry

        loosened = _base_config()
        applied = runtime.apply_config(loosened)

        assert "agent.skills" in applied
        assert "sr2-conventions" in runtime.skill_registry
        assert "load_skill" in runtime.registry

    def test_flag_defaults_true_on_agent_config(self):
        from sr2_spectre.config import AgentConfig

        assert AgentConfig().default_skills is True

    def test_flag_round_trips_through_yaml(self, tmp_path: Path):
        import yaml

        from sr2_spectre.config import AgentConfig

        raw = yaml.safe_load("name: brokkr\ndefault_skills: false\n")
        assert AgentConfig(**raw).default_skills is False
