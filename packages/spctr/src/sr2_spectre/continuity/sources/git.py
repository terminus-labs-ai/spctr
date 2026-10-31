"""Git situation source (sr2-102 / FR22).

Reads git status and recent commits from a repository, producing one
``Entry`` with ``LIVE_SOURCE`` authority and HEAD sha as revision.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from sr2.continuity.index import ContinuityFailure
from sr2.models import TextBlock
from sr2.pipeline.provenance import ContextAuthority, ContextReason, Entry, EntryOrigin

from sr2_spectre.continuity.situation import SituationRequest, SituationSnapshot

_LAYER = "situation"
_MAX_COMMITS = 10


def _run_git(repo: str, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


class GitSituationProvider:
    """Reads git working state and recent commits."""

    def __init__(self, *, repo: str, required: bool = True) -> None:
        self.repo = repo
        self.required = required

    async def refresh(self, request: SituationRequest) -> SituationSnapshot:
        try:
            head = _run_git(self.repo, "rev-parse", "HEAD").strip()
            status = _run_git(self.repo, "status", "--porcelain")
            log = _run_git(
                self.repo, "log", f"-{self._max_commits()}",
                "--format=%h %s (%an, %ar)",
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            if self.required:
                return SituationSnapshot(
                    entries=(),
                    refreshed_at=datetime.now(timezone.utc),
                    revisions={},
                    failures=(
                        ContinuityFailure(
                            capability="situation",
                            message=f"git source: {exc}",
                            retryable=True,
                        ),
                    ),
                )
            return SituationSnapshot(
                entries=(),
                refreshed_at=datetime.now(timezone.utc),
                revisions={},
            )

        parts = [f"HEAD: {head}"]
        if status.strip():
            parts.append(f"Working state:\n{status}")
        else:
            parts.append("Working state: clean")
        parts.append(f"Recent commits:\n{log}")
        text = "\n".join(parts)

        entry = Entry(
            id=f"git-{head[:12]}",
            content=TextBlock(text=text),
            sources=(),
            origin=EntryOrigin(kind="resolver", name="git-source"),
            layer=_LAYER,
            session_id="situation",
            created_at=datetime.now(timezone.utc),
            reason=ContextReason.LIVE_SOURCE,
            authority=ContextAuthority.LIVE_SOURCE,
            meta={"source": "git", "revision": head},
        )

        return SituationSnapshot(
            entries=(entry,),
            refreshed_at=datetime.now(timezone.utc),
            revisions={"git": head},
        )

    def _max_commits(self) -> int:
        return _MAX_COMMITS
