"""Beads situation source (sr2-102 / FR22).

Runs ``bd list`` against a configured project directory and produces one
``Entry`` with the project's issue state.  Detects a beads project by the
presence of a ``.beads`` directory in the resolved working directory.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sr2.continuity.index import ContinuityFailure
from sr2.models import TextBlock
from sr2.pipeline.provenance import ContextAuthority, ContextReason, Entry, EntryOrigin

from sr2_spectre.continuity.situation import SituationRequest, SituationSnapshot

_LAYER = "situation"


class BeadsSituationProvider:
    """Reads beads project state via the ``bd`` CLI."""

    def __init__(
        self,
        *,
        project: str,
        required: bool = True,
        cwd: str | None = None,
    ) -> None:
        self.project = project
        self.required = required
        self._cwd = cwd

    async def refresh(self, request: SituationRequest) -> SituationSnapshot:
        cwd = self._cwd or request.cwd
        if not cwd:
            if self.required:
                return self._failure("beads source: no cwd available")
            return self._empty()

        beads_dir = Path(cwd) / ".beads"
        if not beads_dir.is_dir():
            if self.required:
                return self._failure(
                    f"beads source: no .beads directory in {cwd}"
                )
            return self._empty()

        try:
            result = subprocess.run(
                ["bd", "list"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            output = f"beads project {self.project} detected (bd list unavailable)"

        revision = datetime.now(timezone.utc).isoformat()
        entry = Entry(
            id=f"beads-{self.project}-{revision[:19]}",
            content=TextBlock(text=output),
            sources=(),
            origin=EntryOrigin(kind="resolver", name="beads-source"),
            layer=_LAYER,
            session_id="situation",
            created_at=datetime.now(timezone.utc),
            reason=ContextReason.LIVE_SOURCE,
            authority=ContextAuthority.LIVE_SOURCE,
            meta={"source": "beads", "revision": revision},
        )

        return SituationSnapshot(
            entries=(entry,),
            refreshed_at=datetime.now(timezone.utc),
            revisions={"beads": revision},
        )

    def _failure(self, message: str) -> SituationSnapshot:
        return SituationSnapshot(
            entries=(),
            refreshed_at=datetime.now(timezone.utc),
            revisions={},
            failures=(
                ContinuityFailure(
                    capability="situation",
                    message=message,
                    retryable=True,
                ),
            ),
        )

    def _empty(self) -> SituationSnapshot:
        return SituationSnapshot(
            entries=(),
            refreshed_at=datetime.now(timezone.utc),
            revisions={},
        )
