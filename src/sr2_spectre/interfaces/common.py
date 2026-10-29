"""Pure helpers shared by the chat interfaces (Discord and Zulip).

No platform SDK is imported here. Each helper has exactly one
implementation, which every interface imports, so a channel's area is
derived the same way on Discord and on Zulip.
"""
from __future__ import annotations

from typing import Literal


def derive_area_name(channel_name: str | None) -> str | None:
    """Lowercase, strip leading/trailing non-alphanumerics. None stays None.

    Returns None when nothing alphanumeric survives the strip (e.g. "---",
    a bare emoji, or the empty string) — "" is never returned by this
    function.
    """
    if channel_name is None:
        return None

    lowered = channel_name.lower()
    start = 0
    end = len(lowered)
    while start < end and not lowered[start].isalnum():
        start += 1
    while end > start and not lowered[end - 1].isalnum():
        end -= 1

    stripped = lowered[start:end]
    return stripped or None


#: Where a resolved area came from: the ``channel_areas`` override map, the
#: channel name, or neither.
AreaProvenance = Literal["override", "derived"] | None


def resolve_area(
    channel_id: int | None,
    channel_name: str | None,
    channel_areas: dict[str, str],
) -> tuple[str | None, AreaProvenance]:
    """Resolve an area and its provenance.

    Overrides win when the channel ID is present in ``channel_areas``. An
    empty override maps to ``None`` but still reports ``"override"``. A
    usable channel name reports ``"derived"``. No matching override and no
    usable name reports ``(None, None)``.
    """
    if channel_id is not None:
        key = str(channel_id)
        if key in channel_areas:
            return channel_areas[key] or None, "override"

    derived = derive_area_name(channel_name)
    return (derived, "derived") if derived is not None else (None, None)
