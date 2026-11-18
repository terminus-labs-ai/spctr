"""Resolve managed topic documents by stable identity on each turn."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

import yaml
from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.resolvers.markdown_file import MarkdownTokenBudgetError
from sr2.pipeline.token_counting import CHARS_PER_TOKEN
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

logger = logging.getLogger(__name__)
_DEFAULT_SUBSCRIPTION = EventSubscription(
  event_name="turn_start", phase=EventPhase.STARTING
)


class TopicAreaResolver:
  """Select managed documents from one eligible topic by area identity."""

  name: str = "topic_area"

  def __init__(
    self,
    config: ResolverConfig,
    run_context_provider: Callable[[], dict[str, str] | None] | None = None,
  ) -> None:
    root = config.config.get("topics_root")
    if (
      not isinstance(root, str)
      or not root
      or not Path(root).is_absolute()
      or any(character in root for character in "*?[]")
    ):
      raise ValueError("topic_area requires an absolute, non-glob topics_root.")
    filename = config.config.get("filename")
    self._validate_filename(filename)
    max_tokens = config.config.get("max_tokens")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
      raise ValueError("topic_area max_tokens must be a positive integer or None.")

    self._root = Path(root)
    self._filename = filename
    self._max_tokens = max_tokens
    self._run_context_provider = run_context_provider
    self._missing_warned = False
    self.max_executions = config.max_executions
    self.execution_count = 0
    self.subscriptions = build_subscriptions(
      config.subscriptions, PHASE_MAP, [_DEFAULT_SUBSCRIPTION]
    )

  @staticmethod
  def _validate_filename(filename: object) -> None:
    if not isinstance(filename, str) or not filename:
      raise ValueError("topic_area filename must be a nonempty relative path string.")
    segments = filename.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
      raise ValueError(
        "topic_area filename must be a relative path inside the topic "
        "(no empty segments, '.' or '..')."
      )

  @classmethod
  def build(cls, config: ResolverConfig, deps: Dependencies) -> TopicAreaResolver:
    return cls(config, run_context_provider=deps.run_context_provider)

  async def resolve(self, events: list[Event]) -> ResolvedContent:
    self.execution_count += 1
    text = self._resolve_text()
    tokens = len(text) // CHARS_PER_TOKEN
    if self._max_tokens is not None and tokens > self._max_tokens:
      raise MarkdownTokenBudgetError(
        f"TopicAreaResolver: {self._filename} exceeds token budget "
        f"({tokens} tokens; budget {self._max_tokens})."
      )
    return ResolvedContent(
      resolver_name=self.name,
      source_layer=self.name,
      content=[TextBlock(text=text)] if text else [],
      token_count=tokens,
    )

  @staticmethod
  def _read(path: Path, root: Path) -> str | None:
    """Read only documents whose resolved path stays inside the root."""
    try:
      resolved = path.resolve()
      if not resolved.is_relative_to(root):
        return None
      return resolved.read_text(encoding="utf-8")
    except (OSError, RuntimeError, UnicodeDecodeError):
      return None

  @classmethod
  def _metadata(cls, path: Path, root: Path) -> dict:
    text = cls._read(path, root)
    if text is None:
      return {}
    lines = text.splitlines()
    if not lines or lines[0] != "---":
      return {}
    try:
      end = lines.index("---", 1)
      metadata = yaml.safe_load("\n".join(lines[1:end]))
    except (ValueError, yaml.YAMLError):
      return {}
    return metadata if isinstance(metadata, dict) else {}

  def _missing(self, area: str) -> str:
    """Log the missing target once per session, then stay quiet."""
    if not self._missing_warned:
      self._missing_warned = True
      logger.warning(
        "TopicAreaResolver: topic %r has no %s; nothing to inject.",
        area,
        self._filename,
      )
    else:
      logger.debug(
        "TopicAreaResolver: topic %r has no %s; nothing to inject.",
        area,
        self._filename,
      )
    return ""

  def _glob_segments(self) -> tuple[list[str], str] | None:
    """Split the filename into a directory prefix and a glob pattern, if any."""
    segments = self._filename.split("/")
    if not any(character in segments[-1] for character in "*?[]"):
      return None
    return segments[:-1], segments[-1]

  def _select_paths(self, topic: Path, root: Path) -> tuple[list[Path], bool] | None:
    """Return files to load (plus a multi-file flag), or None when nothing matches."""
    glob_parts = self._glob_segments()
    if glob_parts is None:
      selected = topic / self._filename
      try:
        resolved = selected.resolve()
      except (OSError, RuntimeError):
        return None
      if not resolved.is_relative_to(root):
        return [], False
      if resolved.is_dir():
        files = [
          path
          for path in resolved.rglob("*")
          if path.is_file()
          and not path.is_symlink()
          and not any(part.startswith(".") for part in path.relative_to(resolved).parts)
        ]
        return sorted(files), True
      if resolved.is_file():
        return [selected], False
      return None
    prefix, pattern = glob_parts
    directory = topic.joinpath(*prefix)
    if not directory.is_dir():
      return None
    files = []
    for path in sorted(directory.glob(pattern)):
      if not path.is_file() or path.is_symlink():
        continue
      try:
        resolved = path.resolve()
      except (OSError, RuntimeError):
        continue
      if resolved.is_relative_to(root):
        files.append(path)
    return (files, True) if files else None

  def _resolve_text(self) -> str:
    context = self._run_context_provider() if self._run_context_provider else None
    area = context.get("area") if isinstance(context, dict) else None
    if not isinstance(area, str) or not area:
      logger.debug("TopicAreaResolver: no area supplied; skipping topic context.")
      return ""
    identity = f"topic:{area}"
    matches: list[Path] = []
    try:
      root = self._root.resolve()
      if not root.is_dir():
        logger.warning(
          "TopicAreaResolver: topics_root %s is not a directory; no topic context.",
          self._root,
        )
        return ""
      for directory, subdirs, files in os.walk(root):
        subdirs[:] = [
          name
          for name in subdirs
          if not name.startswith(".") and not (Path(directory) / name).is_symlink()
        ]
        if "README.md" not in files:
          continue
        topic = Path(directory)
        readme = self._metadata(topic / "README.md", root)
        if readme.get("kind") != "topic" or readme.get("id") != identity:
          continue
        now = self._metadata(topic / "NOW.md", root)
        if (
          now.get("kind") == "continuity"
          and now.get("for") == identity
          and now.get("mode") == "source"
        ):
          matches.append(topic)
    except (OSError, RuntimeError):
      logger.warning("TopicAreaResolver: cannot scan topics_root %s.", self._root)
      return ""
    if not matches:
      logger.debug(
        "TopicAreaResolver: no eligible topic for %r; nothing to inject.",
        area,
      )
      return ""
    if len(matches) > 1:
      logger.warning(
        "TopicAreaResolver: expected one eligible topic for %r; found %d.",
        area,
        len(matches),
      )
      return ""
    topic = matches[0]
    selection = self._select_paths(topic, root)
    if selection is None:
      return self._missing(area)
    files, multi = selection
    if not files:
      return self._missing(area)
    chunks: list[str] = []
    unread = 0
    for path in files:
      text = self._read(path, root)
      if text is None:
        unread += 1
        continue
      if multi:
        relative = path.relative_to(topic)
        text = f"### {relative}\n\n{text}"
      chunks.append(text)
    if unread:
      logger.warning(
        "TopicAreaResolver: skipped %d unreadable file(s) under %s.",
        unread,
        topic / self._filename,
      )
    if not chunks:
      return self._missing(area)
    return "\n\n".join(chunks)
