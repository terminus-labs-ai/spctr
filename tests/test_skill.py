"""Tests for the skill system (sr2_spectre.skills).

Covers:
- Skill dataclass validation (frozen, required fields)
- SkillRegistry CRUD (register, get, list, find_by_tag, get_content)
- load_skill_from_path (file-based loading, error handling)
- Built-in sr2-conventions skill content
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sr2_spectre.skills.core import Skill, SkillRegistry, load_skill_from_path


# ---------------------------------------------------------------------------
# Skill construction
# ---------------------------------------------------------------------------

class TestSkillConstruction:
    """Skill dataclass validates required fields."""

    def test_minimal_skill(self):
        skill = Skill(name="test", description="A test skill")
        assert skill.name == "test"
        assert skill.description == "A test skill"
        assert skill.version == "0.1.0"
        assert skill.content == ""
        assert skill.tags == ()

    def test_full_skill(self):
        skill = Skill(
            name="full",
            description="Full skill",
            version="1.2.3",
            content="some content",
            tags=["a", "b"],
        )
        assert skill.version == "1.2.3"
        assert skill.content == "some content"
        assert skill.tags == ("a", "b")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name must not be empty"):
            Skill(name="", description="desc")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description must not be empty"):
            Skill(name="test", description="")

    def test_skill_is_frozen(self):
        skill = Skill(name="test", description="desc")
        with pytest.raises(Exception):  # FrozenInstanceError
            skill.name = "other"  # type: ignore[frozen-instantiation]

    def test_tags_are_immutable_tuple(self):
        """tags is stored as a tuple, even when constructed from a list."""
        # Pass a list — should be converted to tuple
        skill = Skill(name="test", description="desc", tags=["a", "b"])
        assert isinstance(skill.tags, tuple)
        assert skill.tags == ("a", "b")
        # Cannot mutate (frozen dataclass raises FrozenInstanceError)
        with pytest.raises(Exception):  # FrozenInstanceError
            skill.tags += ("c",)  # type: ignore[unreachable]

    def test_tags_empty_by_default(self):
        """Default tags is an empty tuple, not a list."""
        skill = Skill(name="test", description="desc")
        assert skill.tags == ()
        assert isinstance(skill.tags, tuple)


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class TestSkillRegistry:
    """SkillRegistry manages registration and lookup."""

    def _make_skill(
        self,
        name: str = "test",
        description: str = "Test skill",
        content: str = "content here",
        tags: tuple[str, ...] | None = None,
    ) -> Skill:
        return Skill(
            name=name,
            description=description,
            content=content,
            tags=tags or (),
        )

    def test_register_and_get(self):
        registry = SkillRegistry()
        skill = self._make_skill()
        registry.register(skill)

        result = registry.get("test")
        assert result is not None
        assert result.name == "test"
        assert result.content == "content here"

    def test_get_missing_returns_none(self):
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_get_content(self):
        registry = SkillRegistry()
        registry.register(self._make_skill(name="x", content="hello"))
        assert registry.get_content("x") == "hello"
        assert registry.get_content("missing") is None

    def test_list_names_sorted(self):
        registry = SkillRegistry()
        registry.register(self._make_skill(name="charlie"))
        registry.register(self._make_skill(name="alpha"))
        registry.register(self._make_skill(name="bravo"))
        assert registry.list_names() == ["alpha", "bravo", "charlie"]

    def test_find_by_tag(self):
        registry = SkillRegistry()
        registry.register(self._make_skill(name="a", tags=("sr2", "core")))
        registry.register(self._make_skill(name="b", tags=("planning",)))
        registry.register(self._make_skill(name="c", tags=("sr2", "planning")))

        assert len(registry.find_by_tag("sr2")) == 2
        assert len(registry.find_by_tag("planning")) == 2
        assert len(registry.find_by_tag("nonexistent")) == 0

    def test_contains(self):
        registry = SkillRegistry()
        registry.register(self._make_skill(name="test"))
        assert "test" in registry
        assert "missing" not in registry

    def test_len(self):
        registry = SkillRegistry()
        assert len(registry) == 0
        registry.register(self._make_skill(name="a"))
        registry.register(self._make_skill(name="b"))
        assert len(registry) == 2

    def test_overwrite_logs_warning(self, caplog):
        registry = SkillRegistry()
        registry.register(self._make_skill(name="test", content="v1"))
        registry.register(self._make_skill(name="test", content="v2"))
        assert "overwriting" in caplog.text.lower()
        assert registry.get_content("test") == "v2"


# ---------------------------------------------------------------------------
# load_skill_from_path
# ---------------------------------------------------------------------------

class TestLoadSkillFromPath:
    """load_skill_from_path loads skill content from a file."""

    def test_load_from_file(self, tmp_path: Path):
        content_file = tmp_path / "my-skill.md"
        content_file.write_text("# My Skill\n\nContent goes here.\n")

        skill = load_skill_from_path(
            name="my-skill",
            path=content_file,
            description="A skill from disk",
            tags=["disk"],  # load_skill_from_path accepts list, converts to tuple
        )

        assert skill.name == "my-skill"
        assert skill.description == "A skill from disk"
        assert skill.content == "# My Skill\n\nContent goes here.\n"
        assert "disk" in skill.tags

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Skill content file not found"):
            load_skill_from_path(
                name="missing",
                path=tmp_path / "does-not-exist.md",
            )

    def test_default_description_from_name(self, tmp_path: Path):
        content_file = tmp_path / "skill.md"
        content_file.write_text("content")

        skill = load_skill_from_path(
            name="auto-desc",
            path=content_file,
        )
        assert skill.description == "Skill: auto-desc"

    def test_path_with_subdirectory(self, tmp_path: Path):
        """Loading works with nested directory paths."""
        subdir = tmp_path / "skills" / "nested"
        subdir.mkdir(parents=True)
        content_file = subdir / "deep-skill.md"
        content_file.write_text("deep content")

        skill = load_skill_from_path(
            name="deep",
            path=content_file,
        )
        assert skill.content == "deep content"

    def test_path_object_accepted(self, tmp_path: Path):
        """Path objects (not just strings) are accepted."""
        content_file = tmp_path / "path-obj.md"
        content_file.write_text("path object content")

        skill = load_skill_from_path(
            name="path-obj",
            path=content_file,  # Pass as Path object
        )
        assert skill.content == "path object content"


# ---------------------------------------------------------------------------
# Built-in SR2 conventions skill
# ---------------------------------------------------------------------------

class TestBuiltinSkill:
    """Built-in sr2-conventions skill ships with correct content."""

    def test_builtin_skill_exists(self):
        from sr2_spectre.skills.builtin import get_sr2_conventions_skill

        skill = get_sr2_conventions_skill()
        assert skill.name == "sr2-conventions"
        assert "SR2" in skill.description or "sr2" in skill.description.lower()
        assert skill.version == "0.1.0"

    def test_builtin_content_has_key_sections(self):
        from sr2_spectre.skills.builtin import get_sr2_conventions_skill

        skill = get_sr2_conventions_skill()
        content = skill.content

        # Verify key topics are covered
        assert "Pipeline" in content or "pipeline" in content
        assert "Plan" in content or "plan" in content
        assert "Knowledge" in content or "knowledge" in content
        assert "Tool" in content or "tool" in content

    def test_builtin_tags(self):
        from sr2_spectre.skills.builtin import get_sr2_conventions_skill

        skill = get_sr2_conventions_skill()
        assert "sr2" in skill.tags
        assert "conventions" in skill.tags

    def test_default_skills_list(self):
        from sr2_spectre.skills.builtin import DEFAULT_SKILLS

        assert len(DEFAULT_SKILLS) >= 1
        assert DEFAULT_SKILLS[0].name == "sr2-conventions"

    def test_builtin_registers_cleanly(self):
        """The builtin skill can be registered in a registry without errors."""
        from sr2_spectre.skills.builtin import DEFAULT_SKILLS

        registry = SkillRegistry()
        for skill in DEFAULT_SKILLS:
            registry.register(skill)

        assert "sr2-conventions" in registry
        assert registry.get_content("sr2-conventions") is not None
        assert len(registry.get_content("sr2-conventions")) > 100  # substantial content


# ---------------------------------------------------------------------------
# Built-in SOLID Review skill
# ---------------------------------------------------------------------------

class TestSolidReviewSkill:
    """Built-in solid-review skill ships with correct content."""

    def test_solid_review_skill_exists(self):
        from sr2_spectre.skills.builtin import get_solid_review_skill

        skill = get_solid_review_skill()
        assert skill.name == "solid-review"
        assert "SOLID" in skill.description or "solid" in skill.description.lower()
        assert skill.version == "0.1.0"

    def test_solid_review_content_has_key_sections(self):
        from sr2_spectre.skills.builtin import get_solid_review_skill

        skill = get_solid_review_skill()
        content = skill.content

        # Verify review lenses are covered
        assert "Single Responsibility" in content
        assert "Open/Closed" in content
        assert "Liskov" in content
        assert "Interface Segregation" in content
        assert "Dependency Inversion" in content
        assert "DRY" in content

    def test_solid_review_content_has_scope_guidance(self):
        """The skill instructs agents to accept a scope argument."""
        from sr2_spectre.skills.builtin import get_solid_review_skill

        skill = get_solid_review_skill()
        content = skill.content

        assert "scope" in content.lower()
        assert "git diff" in content or "diff" in content.lower()

    def test_solid_review_content_has_output_format(self):
        """The skill includes an output format template."""
        from sr2_spectre.skills.builtin import get_solid_review_skill

        skill = get_solid_review_skill()
        content = skill.content

        assert "BLUF" in content
        assert "prioritize" in content.lower() or "priority" in content.lower()

    def test_solid_review_tags(self):
        from sr2_spectre.skills.builtin import get_solid_review_skill

        skill = get_solid_review_skill()
        assert "review" in skill.tags
        assert "solid" in skill.tags
        assert "dry" in skill.tags
        assert "architecture" in skill.tags
        assert "audit" in skill.tags

    def test_default_skills_includes_solid_review(self):
        from sr2_spectre.skills.builtin import DEFAULT_SKILLS

        names = [s.name for s in DEFAULT_SKILLS]
        assert "sr2-conventions" in names
        assert "solid-review" in names
        assert len(DEFAULT_SKILLS) >= 2

    def test_solid_review_registers_cleanly(self):
        """The solid-review skill registers in a registry without errors."""
        from sr2_spectre.skills.builtin import get_solid_review_skill

        registry = SkillRegistry()
        skill = get_solid_review_skill()
        registry.register(skill)

        assert "solid-review" in registry
        content = registry.get_content("solid-review")
        assert content is not None
        assert len(content) > 500  # substantial review framework
