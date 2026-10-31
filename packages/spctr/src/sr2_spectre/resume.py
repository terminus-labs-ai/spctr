"""Reconstruct text-only session history from the provenance archive."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sr2.models import Message, TextBlock
from sr2.pipeline.provenance import (
    TURN_ARCHIVE_LAYER,
    ContextAuthority,
    ContextLifecycle,
    ContextReason,
    ContextScope,
    Entry,
    EntryOrigin,
    ProvenanceStore,
)
from ulid import ULID

from sr2_spectre.config import ResumeConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeResult:
    history: list[Message]
    interrupted: bool
    restore_failed: bool


def _turn_bound(entry: Entry, name: str) -> int:
    value = entry.meta[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Archive metadata {name} must be an integer")
    return value


def _seq_key(entry: Entry) -> tuple[int, int]:
    seq = entry.meta.get("seq")
    if isinstance(seq, int) and not isinstance(seq, bool):
        return (0, seq)
    return (1, 0)


async def load_resumable_history(
    store: ProvenanceStore | None, session_id: str, config: ResumeConfig
) -> ResumeResult:
    """Read one session once, applying all rewinds before rebuilding history."""
    if not config.enabled:
        return ResumeResult([], interrupted=False, restore_failed=False)
    if store is None:
        logger.warning("History restore failed: no provenance store (session=%s)", session_id)
        return ResumeResult([], interrupted=False, restore_failed=True)

    try:
        entries = await store.get_session(session_id)
        turns: dict[int, list[Entry]] = {}
        rewinds: list[tuple[int, int]] = []
        for entry in entries:
            if entry.layer == TURN_ARCHIVE_LAYER:
                turn = _turn_bound(entry, "turn_seq")
                turns.setdefault(turn, []).append(entry)
            elif entry.layer == "session_rewind":
                rewinds.append((
                    _turn_bound(entry, "keep_through_turn_seq"),
                    _turn_bound(entry, "through_turn_seq"),
                ))

        last_turn = max(turns, default=-1)
        history: list[Message] = []
        interrupted = False
        for turn in sorted(turns):
            if any(keep < turn <= through for keep, through in rewinds):
                continue
            archived = sorted(turns[turn], key=_seq_key)
            user_text: list[str] = []
            assistant_text: list[str] = []
            for entry in archived:
                if isinstance(entry.content, Message):
                    target = user_text if entry.content.role == "user" else assistant_text
                    target.extend(
                        block.text for block in entry.content.content
                        if isinstance(block, TextBlock)
                    )
            last = archived[-1].content
            if turn == last_turn and not (
                isinstance(last, Message) and last.role == "assistant"
            ):
                assistant_text = [config.interrupted_marker]
                interrupted = True
            text = "".join(assistant_text)
            history.extend([
                Message(role="user", content=[TextBlock(text="".join(user_text))]),
                Message(role="assistant", content=[TextBlock(text=text)] if text else []),
            ])
        return ResumeResult(history, interrupted=interrupted, restore_failed=False)
    except Exception:
        logger.exception("History restore failed (session=%s)", session_id)
        return ResumeResult([], interrupted=False, restore_failed=True)


async def write_rewind(
    store: ProvenanceStore, session_id: str,
    keep_through_turn_seq: int, through_turn_seq: int,
) -> None:
    """Append a reset/retry range without deleting archive or lineage entries."""
    await store.write(Entry(
        id=str(ULID()),
        content=TextBlock(text="reset" if keep_through_turn_seq == -1 else "retry"),
        sources=(),
        origin=EntryOrigin(kind="resolver", name="session_rewind"),
        layer="session_rewind",
        session_id=session_id,
        created_at=datetime.now(tz=timezone.utc),
        reason=ContextReason.USER_INPUT,
        authority=ContextAuthority.USER,
        lifecycle=ContextLifecycle.ARCHIVE,
        scope=ContextScope(owner_id=session_id),
        meta={
            "keep_through_turn_seq": keep_through_turn_seq,
            "through_turn_seq": through_turn_seq,
        },
    ))
