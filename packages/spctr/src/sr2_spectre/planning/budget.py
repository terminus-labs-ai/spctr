"""LayerBudget: token-budget allocator for layered content injection.

Extracted from PlanResolver to satisfy SRP — the resolver builds layers,
the budget allocator decides which survive.  Pure logic: no file I/O, no
engine imports, fully unit-testable.

Public API
----------
LayerBudget
    Takes a list of (header, content, priority) tuples and a token limit,
    returns the surviving (header, content) pairs after priority-aware
    eviction and possible tail truncation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Approximate chars-per-token constant (matches SR2 pipeline default).
CHARS_PER_TOKEN = 4

# Separator injected between layers when joining.
_LAYER_SEPARATOR = "\n---\n"


class LayerBudget:
    """Priority-aware token-budget allocator for layered content.

    Given layers with associated priorities (lower number = more protected),
    drops layers from lowest to highest priority until the combined output
    fits within *max_tokens*.  If the final remaining layer alone exceeds the
    budget, it is tail-truncated with a notice appended.

    Parameters
    ----------
    max_tokens : int
        Token budget for the combined output.
    """

    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        layers: list[tuple[str, str, int]],
    ) -> list[tuple[str, str]]:
        """Evict layers until the combined output fits *max_tokens*.

        Parameters
        ----------
        layers : list of (header, content, priority)
            Each layer has a header (e.g. ``"## Active Plan"``), content
            string, and integer priority (lower = more protected).

        Returns
        -------
        list of (header, content)
            Surviving layers in their original order, possibly reduced or
            truncated.  Returns an empty list when *layers* is empty.
        """
        max_chars = self.max_tokens * CHARS_PER_TOKEN

        if not layers:
            return []

        def _total_chars(hc: list[tuple[str, str]]) -> int:
            body = sum(len(h) + len(c) for h, c in hc)
            sep = max(0, len(hc) - 1) * len(_LAYER_SEPARATOR)
            return body + sep

        remaining: list[tuple[str, str]] = [
            (h, c) for h, c, _ in layers
        ]

        if _total_chars(remaining) <= max_chars:
            return remaining  # fits — no eviction needed

        # Sort by priority descending (highest number = least protected
        # = dropped first).  Break ties by original index (earliest first).
        indexed = list(enumerate(layers))
        indexed.sort(key=lambda t: (-t[1][2], t[0]))

        for _idx, (header, _content, _pri) in indexed:
            if len(remaining) <= 1:
                break  # last layer → truncate, not drop

            candidate = [
                (h, c) for h, c in remaining if h != header
            ]

            if _total_chars(candidate) <= max_chars:
                layer_name = header.replace("## ", "")
                logger.info(
                    "Token budget exceeded: dropped %s layer.",
                    layer_name,
                )
                remaining = candidate
                break

            # Didn't fit yet — keep this one removed and try next
            remaining = candidate

        # If remaining layers still exceed budget, truncate the last one.
        if remaining and _total_chars(remaining) > max_chars:
            remaining = self._truncate_last_layer(layers, remaining, max_chars)

        return remaining

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_last_layer(
        original: list[tuple[str, str, int]],
        remaining: list[tuple[str, str]],
        max_chars: int,
    ) -> list[tuple[str, str]]:
        """Tail-truncate the most protected remaining layer.

        Uses priority from *original* (which still carries the priority
        int) to find the most protected layer among *remaining*.
        """
        notice = "\n\n⚠️ Content truncated — token budget exceeded."

        if not remaining:
            return []

        # Build a priority lookup from the original layers.
        pri_map: dict[str, int] = {}
        for h, _, p in original:
            pri_map[h] = p

        # Find the highest-priority (lowest number) layer to truncate.
        # Among equal priority, pick the last in list order.
        truncate_idx = max(
            range(len(remaining)),
            key=lambda i: (-pri_map.get(remaining[i][0], 0), i),
        )

        header, content = remaining[truncate_idx]
        available = max(0, max_chars - len(header) - len(notice))

        if len(content) > available:
            remaining[truncate_idx] = (
                header, content[:available] + notice
            )

        return remaining
