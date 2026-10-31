"""Pure helpers shared by the chat interfaces (Discord and Zulip).

No platform SDK is imported here. Each helper has exactly one
implementation, which every interface imports, so a channel's area is
derived, and a long answer chunked, the same way on Discord and on Zulip.
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


def chunk_message(text: str, max_length: int = 2000) -> list[str]:
    """Split a long message into chunks that fit a chat length limit.

    Tries to split at paragraph boundaries first, then word boundaries.
    Never splits mid-word unless the word itself exceeds the limit.

    Args:
        text: The full response text.
        max_length: Maximum length per chunk (default: 2000, Discord's).

    Returns:
        List of string chunks, each <= max_length characters.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Split marker appended when we split mid-content
        split_marker = "..."
        # Content budget: max_length minus the split_marker
        content_budget = max_length - len(split_marker)

        # Try paragraph split first (double newline)
        para_break = remaining.rfind("\n\n", 0, content_budget)
        if para_break > content_budget // 2:
            chunks.append(remaining[:para_break] + "\n" + split_marker)
            remaining = remaining[para_break + 2:].lstrip("\n")
            continue

        # Try single newline
        line_break = remaining.rfind("\n", 0, content_budget)
        if line_break > content_budget // 2:
            chunks.append(remaining[:line_break] + "\n" + split_marker)
            remaining = remaining[line_break + 1:]
            continue

        # Try word boundary
        space = remaining.rfind(" ", 0, content_budget)
        if space > content_budget // 2:
            chunks.append(remaining[:space] + split_marker)
            remaining = remaining[space + 1:]
            continue

        # Hard split — word itself is too long
        chunks.append(remaining[:content_budget] + split_marker)
        remaining = remaining[content_budget:]

    return chunks


def tail_for_stream(text: str, max_length: int = 2000) -> str:
    """Trim in-progress streaming text to the LAST ``max_length`` chars.

    The live progress message is a single chat message edited in place, so
    it cannot grow past the length limit. Unlike the final response (which is
    split across several messages by :func:`chunk_message`), the in-progress
    view only needs the LATEST activity, so we keep the tail rather than the
    head. Keeping the head froze the message on stale tool lines once a long
    run first overflowed the limit.

    Args:
        text: The accumulated progress text.
        max_length: Maximum length of the rendered message (default: 2000).

    Returns:
        ``text`` unchanged when it fits, otherwise a leading ellipsis marker
        followed by the trailing content, total length <= ``max_length``.
    """
    if len(text) <= max_length:
        return text
    marker = "[…]\n"
    return marker + text[-(max_length - len(marker)):]
