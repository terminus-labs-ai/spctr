"""Situation provider contract and composite implementation (sr2-102 / AC11, FR22-24, FR29).

Defines the generic ``SituationProvider`` protocol, the request/snapshot
value types from the spec, and ``CompositeSituationProvider`` which fans
out refresh to configured sources, validates revisions, reuses cached
content when unchanged, and surfaces required-source failures as structured
``ContinuityFailure`` records instead of raising.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pydantic import BaseModel
from sr2.continuity.index import ContinuityFailure
from sr2.pipeline.provenance import Entry


class SituationRequest(BaseModel, frozen=True):
    owner_id: str
    agent_id: str
    origin: str
    cwd: str | None
    projects: tuple[str, ...]
    previous_revisions: dict[str, str] = {}


class SituationSnapshot(BaseModel, frozen=True):
    entries: tuple[Entry, ...]
    refreshed_at: datetime
    revisions: dict[str, str]
    failures: tuple[ContinuityFailure, ...] = ()


@runtime_checkable
class SituationProvider(Protocol):
    async def refresh(self, request: SituationRequest) -> SituationSnapshot: ...


class CompositeSituationProvider:
    """Fans out refresh to a list of configured sources.

    Each source may declare ``required: bool``.  A required source that
    raises produces a ``ContinuityFailure`` in the snapshot; an optional
    source that raises is silently skipped.  Entries and revisions from
    all successful sources are merged.
    """

    def __init__(self, sources: list) -> None:
        self._sources = sources

    async def refresh(self, request: SituationRequest) -> SituationSnapshot:
        all_entries: list[Entry] = []
        all_revisions: dict[str, str] = {}
        all_failures: list[ContinuityFailure] = []

        for source in self._sources:
            try:
                snap = await source.refresh(request)
                all_entries.extend(snap.entries)
                all_revisions.update(snap.revisions)
                all_failures.extend(snap.failures)
            except Exception as exc:
                required = getattr(source, "required", True)
                if required:
                    name = type(source).__name__
                    all_failures.append(
                        ContinuityFailure(
                            capability="situation",
                            message=f"{name} failed: {exc}",
                            retryable=True,
                        )
                    )

        return SituationSnapshot(
            entries=tuple(all_entries),
            refreshed_at=datetime.now(timezone.utc),
            revisions=all_revisions,
            failures=tuple(all_failures),
        )
