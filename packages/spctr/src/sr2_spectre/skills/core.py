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
    env: dict[str, str] | None = None,
) -> Skill:
    """Load a skill's content from a file on disk.

    Paths support ``~`` expansion and ``${VAR}`` interpolation. Plain relative
    paths remain relative to the process working directory.

    Args:
        name: Skill identifier.
        path: Path to the markdown/text file containing the skill content.
        version: Skill version.
        description: Override description (if empty, derived from filename).
        tags: Optional tags.
        env: Environment variables for ``${VAR}`` interpolation. Defaults to
            ``os.environ`` through the shared path resolver.

    Returns:
        A Skill with content populated from the file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ConfigPathError: If the path references an unset environment variable.
    """
    from sr2_spectre.path_resolution import resolve_path

    expanded = Path(path).expanduser()
    cwd_sentinel = Path.cwd() / "__skill_path__"
    p = resolve_path(str(expanded), cwd_sentinel, env)
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

    def clear(self) -> None:
        """Drop every registered skill.

        Used when a config reload changes the declared skill set. The registry
        is emptied and refilled in place rather than replaced, because the
        auto-injected ``load_skill`` tool holds a reference to this object.
        """
        self._skills.clear()

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

_DESCRIPTION_MAX_CHARS = 200


def _derive_description(body: str, name: str) -> str:
    """Derive a one-line description from a skill body.

    Used for bundled skills whose identity came from the path rather than
    from frontmatter.  The description is the only thing the model sees in
    ``load_skill(list_only=true)``, so ``Skill: {name}`` is a last resort
    rather than the default — a skill nobody can tell apart never gets
    loaded.  ``Skill.__post_init__`` rejects an empty description, so this
    always returns a non-empty string.

    Args:
        body: Skill content, frontmatter already stripped.
        name: Skill name, used for the last-resort string.

    Returns:
        The first non-heading, non-blank paragraph, truncated to
        ``_DESCRIPTION_MAX_CHARS``, or ``Skill: {name}`` when the body
        carries no prose.
    """
    paragraph: list[str] = []
    in_fence = False

    for raw_line in body.splitlines():
        line = raw_line.strip()

        # Fenced code is not prose. Toggling rather than skipping to the
        # closing fence keeps a code block that opens before any paragraph
        # from contributing its contents.
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if not line:
            # A blank line ends the paragraph, but only once one has
            # started: leading blanks and the gap after a heading are not
            # the end of anything.
            if paragraph:
                break
            continue

        if line.startswith("#"):
            continue

        # A leading blockquote marker is decoration, not description.
        paragraph.append(line.lstrip("> ").strip())

    derived = " ".join(part for part in paragraph if part).strip()
    if not derived:
        return f"Skill: {name}"

    if len(derived) > _DESCRIPTION_MAX_CHARS:
        head = derived[:_DESCRIPTION_MAX_CHARS]
        # Prefer a word boundary, but do not return an empty string when the
        # first "word" is longer than the cap.
        clipped = head.rsplit(" ", 1)[0].rstrip(" ,.;:") or head
        derived = f"{clipped}\u2026"

    return derived


def _load_frontmatter_mapping(
    text: str,
    file_path: Path,
    *,
    lenient: bool,
) -> dict[str, Any] | None:
    """Return a skill file's frontmatter as a mapping, or None if unusable.

    Logs the reason it is unusable.  *lenient* changes only the wording and
    the severity — whether an unusable block is fatal is the caller's
    decision, not this function's.

    Args:
        text: Full file content.
        file_path: Path for logging context.
        lenient: True when the caller can name the skill from its path.

    Returns:
        The frontmatter mapping, or None when absent, unparseable, or not
        a mapping.
    """
    outcome = "naming it from its directory" if lenient else "skipping as skill"

    raw = extract_raw_frontmatter(text)
    if raw is None:
        logger.log(
            logging.DEBUG if lenient else logging.WARNING,
            "No frontmatter in %s — %s",
            file_path,
            outcome,
        )
        return None

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning(
            "YAML parse error in %s: %s — %s",
            file_path,
            exc,
            outcome,
        )
        return None

    if not isinstance(data, dict):
        logger.warning(
            "Frontmatter in %s is not a mapping — %s",
            file_path,
            outcome,
        )
        return None

    return data


def _parse_skill_frontmatter(
    text: str,
    file_path: Path,
    *,
    fallback_name: str | None = None,
) -> Skill | None:
    """Parse a skill file's frontmatter and return a Skill, or None to skip.

    Uses the shared frontmatter extractor from planning/frontmatter.py.
    The body after the frontmatter delimiters becomes the skill content.

    Frontmatter is **required** for the flat ``<dir>/<name>.md`` form, where
    a file's mere presence says nothing about its intent: a stray README or
    notes file dropped in a skills directory must not become a skill.  It is
    **optional** for the bundled ``<dir>/<name>/SKILL.md`` form, where the
    path is itself the declaration — the directory names the skill and the
    filename says what it is.  Callers opt into that by passing
    *fallback_name*.  Frontmatter still wins wherever it is usable.

    Args:
        text: Full file content.
        file_path: Path for logging context.
        fallback_name: Name to fall back to when frontmatter is missing or
            carries no usable ``name``.  None means frontmatter is required
            and the file is skipped instead.

    Returns:
        A Skill instance, or None if the file should be skipped.
    """
    lenient = fallback_name is not None

    data = _load_frontmatter_mapping(text, file_path, lenient=lenient)
    if data is None:
        if not lenient:
            return None
        data = {}

    name = str(data.get("name") or "").strip()
    named_by_path = not name
    if named_by_path:
        if not lenient:
            logger.warning(
                "No 'name' in frontmatter of %s — skipping as skill",
                file_path,
            )
            return None
        name = str(fallback_name)

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

    description = str(data.get("description") or "").strip()
    if not description:
        # Path-derived identity gets a path-derived description. When
        # frontmatter named the skill, its author had the same chance to
        # describe it, so an omission there is respected rather than
        # second-guessed.
        description = (
            _derive_description(content, name)
            if named_by_path
            else f"Skill: {name}"
        )

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

    Two layouts are recognised, and only these two:

    ``<dir>/<name>.md``
        Flat form.  Frontmatter is required — a bare ``.md`` file says
        nothing about its own intent, so a README or a scratch note
        dropped in a skills directory must not silently become a skill.

    ``<dir>/<name>/SKILL.md``
        Bundled form, the Claude/agents convention.  Frontmatter is
        optional: the path already declares the skill, so a file with none
        is named after its directory.

    The bundled glob is deliberately ``*/SKILL.md`` — depth exactly two,
    filename exactly ``SKILL.md``.  That one constraint is what keeps the
    false positives out, and every one of them exists in the wild:
    ``<name>/README.md`` and ``<name>/SECURITY.md`` beside a real skill,
    ``_shared/*.md`` support fragments, and ``<name>/references/*.md``
    bundled resources.  A recursive ``**/*.md`` would register all of them.

    Ordering is flat-then-bundled, each sorted.  Registration is last-wins,
    so discovery order has to be stable across runs.

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

    # (file, fallback_name). A fallback name is what makes frontmatter
    # optional, so only the bundled form carries one.
    candidates: list[tuple[Path, str | None]] = [
        (f, None) for f in sorted(resolved.glob("*.md"))
    ]
    candidates += [
        (f, f.parent.name) for f in sorted(resolved.glob("*/SKILL.md"))
    ]

    for md_file, fallback_name in candidates:
        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Cannot read %s: %s — skipping", md_file, exc)
            continue

        skill = _parse_skill_frontmatter(
            text,
            md_file,
            fallback_name=fallback_name,
        )
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
