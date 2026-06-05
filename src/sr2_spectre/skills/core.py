"""Core skill model and registry.

A **Skill** is a loadable knowledge package — a named bundle of conventions,
workflows, and procedural knowledge that an agent can load on demand.

A **SkillRegistry** manages discovery, registration, and content resolution
of skills. Skills can be registered programmatically or loaded from disk.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sr2_spectre.planning.frontmatter import extract_raw_frontmatter, split_frontmatter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skill:
    """A loadable knowledge package.

    Attributes:
        name: Unique identifier (e.g., "sr2-conventions").
        description: One-line description of what this skill teaches.
        version: Semantic version string (e.g., "0.1.0").
        content: The full text content injected when the skill is loaded.
        tags: Keywords for filtering/categorization (immutable tuple).
    """

    name: str
    description: str
    version: str = "0.1.0"
    content: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Enforce true immutability: convert mutable sequences to tuple.
        # Frozen dataclasses allow __set_attribute__ in __post_init__.
        object.__setattr__(self, "tags", tuple(self.tags))
        if not self.name:
            raise ValueError("Skill name must not be empty")
        if not self.description:
            raise ValueError("Skill description must not be empty")


def load_skill_from_path(
    name: str,
    path: str | Path,
    version: str = "0.1.0",
    description: str = "",
    tags: list[str] | tuple[str, ...] | None = None,
) -> Skill:
    """Load a skill's content from a file on disk.

    Args:
        name: Skill identifier.
        path: Path to the markdown/text file containing the skill content.
        version: Skill version.
        description: Override description (if empty, derived from filename).
        tags: Optional tags.

    Returns:
        A Skill with content populated from the file.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Skill content file not found: {p}")

    if not description:
        description = f"Skill: {name}"

    content = p.read_text(encoding="utf-8")
    return Skill(
        name=name,
        description=description,
        version=version,
        content=content,
        tags=tuple(tags) if tags else (),
    )


class SkillRegistry:
    """Registry for discoverable, loadable skills.

    Skills are registered by name and retrieved by name. The registry
    supports programmatic registration, file-based loading, and
    entry-point discovery.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill by its name.

        Args:
            skill: The Skill instance to register.

        Raises:
            ValueError: If a skill with this name is already registered.
        """
        if skill.name in self._skills:
            logger.warning(
                "Skill '%s' already registered — overwriting",
                skill.name,
            )
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Retrieve a skill by name.

        Args:
            name: The skill identifier.

        Returns:
            The Skill, or None if not found.
        """
        return self._skills.get(name)

    def get_content(self, name: str) -> str | None:
        """Get the content text for a skill.

        Convenience method: equivalent to ``registry.get(name).content``.

        Args:
            name: The skill identifier.

        Returns:
            The skill's content text, or None if not found.
        """
        skill = self._skills.get(name)
        if skill is None:
            return None
        return skill.content

    def list_names(self) -> list[str]:
        """Return all registered skill names, sorted alphabetically."""
        return sorted(self._skills.keys())

    def find_by_tag(self, tag: str) -> list[Skill]:
        """Find all skills that include the given tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of matching Skills, sorted by name.
        """
        return sorted(
            [s for s in self._skills.values() if tag in s.tags],
            key=lambda s: s.name,
        )

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)


# ---------------------------------------------------------------------------
# Directory-based skill discovery
# ---------------------------------------------------------------------------

def _parse_skill_frontmatter(text: str, file_path: Path) -> Skill | None:
    """Parse a skill file's frontmatter and return a Skill, or None to skip.

    Uses the shared frontmatter extractor from planning/frontmatter.py.
    Requires a ``name`` field in the YAML block.  The body after the
    frontmatter delimiters becomes the skill content.

    Args:
        text: Full file content.
        file_path: Path for logging context.

    Returns:
        A Skill instance, or None if the file should be skipped.
    """
    raw = extract_raw_frontmatter(text)
    if raw is None:
        logger.warning(
            "No frontmatter in %s — skipping as skill",
            file_path,
        )
        return None

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning(
            "YAML parse error in %s: %s — skipping as skill",
            file_path,
            exc,
        )
        return None

    if not isinstance(data, dict):
        logger.warning(
            "Frontmatter in %s is not a mapping — skipping as skill",
            file_path,
        )
        return None

    name = data.get("name")
    if not name:
        logger.warning(
            "No 'name' in frontmatter of %s — skipping as skill",
            file_path,
        )
        return None

    name = str(name).strip()
    description = str(data.get("description", "")).strip() or f"Skill: {name}"
    version = str(data.get("version", "0.1.0")).strip()

    raw_tags = data.get("tags", [])
    if isinstance(raw_tags, str):
        tags: tuple[str, ...] = tuple(t.strip() for t in raw_tags.split(",") if t.strip())
    elif isinstance(raw_tags, list):
        tags = tuple(str(t).strip() for t in raw_tags if str(t).strip())
    else:
        tags = ()

    # Extract body content (everything after the frontmatter block)
    result = split_frontmatter(text)
    if result is not None:
        _, body = result
        content = body
    else:
        content = text

    return Skill(
        name=name,
        description=description,
        version=version,
        content=content,
        tags=tags,
    )


def discover_skills_in_dir(
    dir_path: str | Path,
    env: dict[str, str] | None = None,
) -> list[Skill]:
    """Discover and load skills from a directory.

    Globs ``*.md`` files in *dir_path*, parses each file's YAML
    frontmatter for ``name`` / ``description`` / ``version`` / ``tags``,
    and returns a list of Skill instances.  Files without valid
    frontmatter or a ``name`` field are skipped with a warning.

    This is the bulk-loading counterpart to ``load_skill_from_path``
    (single-file).  It uses the shared frontmatter parser from
    ``planning/frontmatter.py`` — no second parser is written.

    Args:
        dir_path: Directory to scan (supports ~ and ${VAR} via
            ``resolve_path``).
        env: Environment variables for path interpolation.

    Returns:
        List of Skill instances discovered in the directory.
    """
    from sr2_spectre.path_resolution import resolve_path

    p = Path(dir_path).expanduser()

    # Handle ${VAR} interpolation for env vars in the path
    if env is None:
        env = dict(__import__("os").environ)

    try:
        resolved = resolve_path(str(p), Path.cwd(), env)
    except Exception:
        # resolve_path may raise ConfigPathError on bad ${VAR}
        logger.warning("Cannot resolve skill directory path: %s — skipping", p)
        return []

    if not resolved.is_dir():
        logger.warning("Skill directory does not exist: %s — skipping", resolved)
        return []

    skills: list[Skill] = []
    md_files = sorted(resolved.glob("*.md"))

    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Cannot read %s: %s — skipping", md_file, exc)
            continue

        skill = _parse_skill_frontmatter(text, md_file)
        if skill is not None:
            skills.append(skill)

    return skills


def discover_skills(
    skills_dirs: list[str],
    env: dict[str, str] | None = None,
) -> list[Skill]:
    """Discover skills from a list of directories.

    Iterates over *skills_dirs*, calling ``discover_skills_in_dir`` for
    each, and returns the concatenated list of discovered skills.

    Args:
        skills_dirs: List of directory paths to scan.
        env: Environment variables for path interpolation.

    Returns:
        Combined list of Skill instances from all directories.
    """
    all_skills: list[Skill] = []
    for dir_path in skills_dirs:
        discovered = discover_skills_in_dir(dir_path, env=env)
        all_skills.extend(discovered)
    return all_skills
