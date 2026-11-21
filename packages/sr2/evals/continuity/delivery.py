"""Delivery partition policy for the continuity baseline (obsidian-wk29).

Extracted verbatim from ``_ProductDouble._carrier`` (sr2-91 acceptance
SOLID finding): the logic that classifies delivered envelope blocks as
the current turn's input versus restored background, stamps each entry's
declared sourcing time (``meta["observed_at"]``) and groups them into
message buckets. It is a pure component: no reference to the SR2 driving
loop, the product pipeline, or ``_ProductDouble`` state, so sr2-100
candidate runs can reuse the partition without the SR2 driving loop.

The entry-construction callback is injected (``make_entry``) precisely so
this module stays free of ``_ProductDouble``: the caller decides how an
entry is built for a delivered text and layer; this module only decides
the layer, the stamping and the grouping order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sr2.pipeline.provenance import Entry


class DeliveryClassifier:
    """Partition delivered envelope blocks into the background bucket and
    the current turn's input bucket, dropping blank entries (empty or
    whitespace-only) and stamping declared recency.

    Classification is by equality with ``input_text`` — the exact text
    the current turn's InputResolver delivers — never by guessing from
    content. Stamping follows the delivery window semantics: the input
    bucket carries the turn's declared prompt-window time (staleness
    reference point); a background value with no ``[observed …]`` stamp
    of its own takes its sourcing time from ``window`` when listed there,
    and values absent from the window travel unclassified (arrival order
    decides — R4 semantics). Declaring recency and locating the input are
    delivery knowledge, so this component — not the pipeline envelope's
    message shape — owns the partition."""

    def classify(
        self,
        texts: Sequence[str],
        *,
        input_text: str,
        window_time: str | None,
        window: dict[str, str],
        make_entry: Callable[[str, str], Entry],
    ) -> tuple[list[Entry], list[Entry]]:
        """Classify and stamp delivered ``texts`` (delivery order).

        ``texts`` — the non-blank delivered block texts, in delivery
        order (entry order = sourcing order, newest last). Blank entries
        (empty or whitespace-only) are dropped.

        ``input_text`` — the exact text of the current turn's input; a
        delivered text equal to it joins the input bucket ("input"
        layer), everything else the background bucket ("session" layer).

        ``window_time`` — declared time of the current turn's prompt
        window, applied to input-bucket entries when truthy.

        ``window`` — declared sourcing time per delivered value; applied
        to a background entry only when it carries no ``observed_at`` of
        its own and its text is listed in the window.

        ``make_entry`` — builds the provenance ``Entry`` for a delivered
        text and its layer.

        Returns ``(background, input_entries)``; within each bucket the
        input delivery order is preserved.
        """
        background: list[Entry] = []
        input_entries: list[Entry] = []
        entry_meta: dict[str, Any]
        for text in texts:
            if not text.strip():
                continue
            if text == input_text:
                entry = make_entry(text, "input")
                entry_meta = entry.meta
                if window_time:
                    entry_meta["observed_at"] = window_time
                input_entries.append(entry)
                continue
            entry = make_entry(text, "session")
            entry_meta = entry.meta
            if not entry_meta.get("observed_at") and text in window:
                entry_meta["observed_at"] = window[text]
            background.append(entry)
        return background, input_entries
