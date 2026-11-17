"""Pure helpers shared by the chat interfaces (Discord and Zulip).

No platform SDK is imported here. Each helper has exactly one
implementation, which every interface imports, so a channel's area is
derived, a long answer chunked, and a turn's live tool/progress message
rendered, the same way on Discord and on Zulip.

The progress renderer below takes its templates, its line limit and its
thinking placeholder from the caller's interface config (spc-1022): neither
interface holds a copy of a tool-line format, a width constant or the
placeholder text.
"""
from __future__ import annotations

import re
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


#: The placeholders a tool-line template may name (spc-1015, spc-1022).
_TOOL_PLACEHOLDER = re.compile(r"\{(name|input|content)\}")


def render_tool_line(
    template: str, name: str, value: str, max_chars: int
) -> str:
    """Fill one tool-line template (spc-1015, spc-1022).

    *value* (a tool's ``str(input)`` or its result content) is cut to
    *max_chars* plus ``...`` and fills both ``{input}`` and ``{content}``;
    ``{name}`` is the tool name. One substitution pass, so a value that
    itself contains ``{name}`` is never re-filled.

    Args:
        template: The line template from config (``tool_start_line``,
            ``tool_result_ok_line``, ``tool_result_error_line``).
        name: The tool name.
        value: The tool input or result text, truncated as above.
        max_chars: The ``tool_line_max_chars`` limit for the value.

    Returns:
        The rendered line.
    """
    if len(value) > max_chars:
        value = value[:max_chars] + "..."
    values = {"name": name, "input": value, "content": value}
    return _TOOL_PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def render_progress_message(
    tool_lines: list[str],
    text: str,
    thinking_message: str,
    max_message_length: int,
) -> str:
    """Compose the live progress message of a turn (spc-1015, spc-1022).

    The tool lines so far, a blank line, then the streamed text plus
    ``...`` — or *thinking_message* while no text has arrived — tailed to
    *max_message_length* by :func:`tail_for_stream`. Both interfaces render
    their in-progress placeholder through this one function, so the tool
    path and the text path cannot drift apart or clobber each other.

    Args:
        tool_lines: The accumulated tool lines, oldest first.
        text: The streamed response text so far (``""`` before any).
        thinking_message: The placeholder text shown while *text* is empty
            (config ``thinking_message``).
        max_message_length: The message length limit to tail to.

    Returns:
        The message content, at most *max_message_length* characters.
    """
    parts: list[str] = []
    if tool_lines:
        parts.append("\n".join(tool_lines))
    parts.append(text + "..." if text else thinking_message)
    return tail_for_stream("\n\n".join(parts), max_message_length)
