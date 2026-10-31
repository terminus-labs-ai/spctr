"""File situation source (sr2-102 / FR22).

Reads configured file paths and produces one ``Entry`` per file with
``LIVE_SOURCE`` authority, a sha256 revision of the content, and the
file path as the source reference.  Caches content by mtime so that
unchanged files are not re-read (FR23).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from sr2.continuity.index import ContinuityFailure
from sr2.models import TextBlock
from sr2.pipeline.provenance import ContextAuthority, ContextReason, Entry, EntryOrigin

from sr2_spectre.continuity.situation import SituationRequest, SituationSnapshot

_LAYER = "situation"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FileSituationProvider:
    """Reads one or more configured file paths as situation entries."""

    def __init__(self, *, paths: list[str], required: bool = True) -> None:
        self.paths = paths
        self.required = required
        self._cache: dict[str, tuple[float, str, Entry]] = {}

    async def refresh(self, request: SituationRequest) -> SituationSnapshot:
        entries: list[Entry] = []
        revisions: dict[str, str] = {}
        failures: list[ContinuityFailure] = []

        for path_str in self.paths:
            path = Path(path_str)
            try:
                stat = path.stat()
            except FileNotFoundError:
                if self.required:
                    failures.append(
                        ContinuityFailure(
                            capability="situation",
                            message=f"file source: {path_str} not found",
                            retryable=True,
                        )
                    )
                continue

            cached = self._cache.get(path_str)
            if cached is not None and cached[0] == stat.st_mtime:
                revision, entry = cached[1], cached[2]
            else:
                content = path.read_text()
                revision = _sha256(content)
                entry = Entry(
                    id=f"file-{path_str}-{revision[:12]}",
                    content=TextBlock(text=content),
                    sources=(),
                    origin=EntryOrigin(kind="resolver", name="file-source"),
                    layer=_LAYER,
                    session_id="situation",
                    created_at=datetime.now(timezone.utc),
                    reason=ContextReason.LIVE_SOURCE,
                    authority=ContextAuthority.LIVE_SOURCE,
                    meta={"source": str(path), "revision": revision},
                )
                self._cache[path_str] = (stat.st_mtime, revision, entry)

            entries.append(entry)
            revisions[f"file:{path_str}"] = revision

        return SituationSnapshot(
            entries=tuple(entries),
            refreshed_at=datetime.now(timezone.utc),
            revisions=revisions,
            failures=tuple(failures),
        )
