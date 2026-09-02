"""Discord message handler — routing, commands, and message formatting.

This module is engine-independent: it contains pure logic for:
- Detecting whether a message should trigger a response
- Parsing slash commands
- Splitting long responses into Discord-compatible chunks
- Building embed payloads for tool execution updates

It does NOT import discord.py directly. The adapter layer bridges
discord.py objects to plain Python types for this handler.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

logger = logging.getLogger("sr2_spectre.discord.handler")


def should_respond(
    content: str,
    mention_only: bool,
    bot_id: int | None,
    bot_mentions: list[str] | None,
) -> bool:
    """Determine whether the bot should respond to a message.

    Args:
        content: The raw message content string.
        mention_only: If True, only respond to bot mentions.
        bot_id: The bot's numeric Discord ID (for <@ID> mentions).
        bot_mentions: Pre-rendered mention strings to check against
                      (e.g., ["<@123>", "<@!123>"]). These are the
                      mention formats discord.py provides via
                      Client.user.mention.

    Returns:
        True if the bot should process this message.
    """
    if not mention_only:
        return True

    if bot_mentions:
        for mention in bot_mentions:
            if mention in content:
                return True

    # Fallback: check numeric ID mention patterns
    if bot_id is not None:
        id_mention = f"<@{bot_id}>"
        id_mention_exclaim = f"<@!{bot_id}>"
        if id_mention in content or id_mention_exclaim in content:
            return True

    return False


# ---------------------------------------------------------------------------
# Area derivation — pure, no discord.py import (FR 3, FR 4, FR 5)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Slash command registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandContext:
    """Read-only context passed to slash command handlers.

    Provides session state so commands like /status can render
    per-channel information without importing discord.py or the
    interface layer.

    Attributes:
        channel_id: Discord channel ID.
        session_id: Spectre session ID for this channel.
        message_count: Number of messages in the channel's history.
    """
    channel_id: int
    session_id: str
    message_count: int
    active_model: str | None = None
    model_label: str | None = None
    area: str | None = None


# Handler signature: (rest: str, ctx: CommandContext) -> str | None
# Returns None for commands that don't produce a text response
# (e.g., /ask which triggers the agent loop).
SyncHandler = Callable[[str, CommandContext], str | None]


@dataclass(frozen=True)
class SlashCommand:
    """Declarative slash command definition.

    Attributes:
        name: Command name (without leading /).
        description: One-line description for /help text.
        handler: Sync function that processes the command.
    """
    name: str
    description: str
    handler: SyncHandler


# Registry of registered slash commands. Commands are looked up by name.
# The set of known commands is derived from registry keys.
_COMMAND_REGISTRY: dict[str, SlashCommand] = {}


def register_command(cmd: SlashCommand) -> SlashCommand:
    """Register a slash command in the global registry.

    Can be used as a decorator or called directly.
    """
    _COMMAND_REGISTRY[cmd.name] = cmd
    return cmd


def get_registered_commands() -> dict[str, SlashCommand]:
    """Return the full command registry (frozen snapshot)."""
    return dict(_COMMAND_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in command handlers
# ---------------------------------------------------------------------------

def _handle_ask(rest: str, ctx: CommandContext) -> str | None:
    """ /ask — triggers the agent loop with `rest` as input. Returns None."""
    return None


def _handle_reset(rest: str, ctx: CommandContext) -> str | None:
    """ /reset — resets the conversation for this channel."""
    return "Conversation reset for this channel."


def _handle_status(rest: str, ctx: CommandContext) -> str | None:
    """ /status — renders real per-channel session info."""
    lines = [
        f"**Session:** `{ctx.session_id}`",
        f"**Messages:** {ctx.message_count}",
    ]
    if ctx.active_model is not None:
        lines.append(f"**Model:** `{ctx.active_model}`")
    if ctx.model_label is not None:
        lines.append(f"**Endpoint:** {ctx.model_label}")
    if ctx.area is not None:
        lines.append(f"**Area:** `{ctx.area}`" if ctx.area else "**Area:** none")
    return "\n".join(lines)


HELP_TEMPLATE = """\
**Commands:**
{command_list}"""


def _build_help_text() -> str:
    """Build /help text from the command registry, excluding internal commands."""
    # /help itself is listed, but /hb is async-only and listed separately
    entries: list[str] = []
    for name, cmd in sorted(_COMMAND_REGISTRY.items()):
        entries.append(f"`/{name}` — {cmd.description}")
    # /hb is async (subprocess), not in the sync registry
    entries.append("`/hb` — Probe Harbinger: live slots, run outcomes, done & blocked beads")
    return HELP_TEMPLATE.format(command_list="\n".join(entries))


def _handle_help(rest: str, ctx: CommandContext) -> str | None:
    """ /help — shows available commands."""
    return _build_help_text()


def _handle_stop(rest: str, ctx: CommandContext) -> str | None:
    """ /stop — cancel the agent's current run in this channel.

    Registered only for its /help + native-tree description. The real logic
    runs in the Discord interface, which holds the per-channel run tasks.
    ``/cancel`` is a wired alias (see SLASH_COMMANDS).
    """
    return None


def _handle_model(rest: str, ctx: CommandContext) -> str | None:
    """ /model — list or switch the active model.

    Registered only so /help and the native command tree carry its
    description (mirroring /ask). The real logic runs in the Discord
    interface, which alone has the model list and the writable pointer file.
    Returns None; the interface intercepts /model before registry dispatch.
    """
    return None


def _handle_area(rest: str, ctx: CommandContext) -> str | None:
    """ /area — show the area this channel resolves to (read-only).

    Areas are set per-channel via the ``channel_areas`` config map (or derived
    from the channel name). This command only *reports* the resolved value;
    switching at runtime is intentionally not offered — ``config.yaml`` is
    mounted read-only and the area drives workspace confinement.
    """
    if ctx.area:
        return f"**Area:** `{ctx.area}`"
    return "**Area:** none — this channel resolves to no area."


def _handle_retry(rest: str, ctx: CommandContext) -> str | None:
    """ /retry — re-run the last user message in this channel.

    Registered only so /help and the native command tree carry its
    description (mirroring /ask). The real logic runs in the Discord
    interface, which holds the per-channel history and the agent run loop.
    Returns None; the interface intercepts /retry before registry dispatch.
    """
    return None


def render_model_command(
    rest: str,
    model_names: set[str],
    active_model: str,
    pointer_configured: bool,
) -> tuple[str, str | None]:
    """Pure decision logic for /model — no I/O.

    Args:
        rest: Text after "/model"; empty lists models, else names one.
        model_names: Keys of the configured ``models`` map.
        active_model: The model name currently in force.
        pointer_configured: Whether a writable pointer file exists to persist
            a switch (``SR2_ACTIVE_MODEL_FILE`` is set).

    Returns:
        ``(response_text, selection)`` — *selection* is the model name the
        caller should persist, or ``None`` when nothing should be written
        (a listing, an error, or re-selecting the active model).
    """
    arg = rest.strip()
    if not arg:
        lines = ["**Models:**"]
        for name in sorted(model_names):
            marker = "  ← active" if name == active_model else ""
            lines.append(f"`{name}`{marker}")
        return "\n".join(lines), None

    if arg not in model_names:
        available = ", ".join(f"`{n}`" for n in sorted(model_names)) or "(none)"
        return f"⚠ Unknown model `{arg}`. Available: {available}", None

    if not pointer_configured:
        return (
            "⚠ Can't switch model: no writable pointer file is configured "
            "(SR2_ACTIVE_MODEL_FILE is unset).",
            None,
        )

    if arg == active_model:
        return f"Already using `{arg}`.", None

    return f"Switched to `{arg}` — takes effect on the next message.", arg


# ---------------------------------------------------------------------------
# Register built-in commands
# ---------------------------------------------------------------------------

register_command(SlashCommand(
    name="ask",
    description="Send a message to the agent (default behavior without command)",
    handler=_handle_ask,
))

register_command(SlashCommand(
    name="reset",
    description="Start a new conversation in this channel",
    handler=_handle_reset,
))

register_command(SlashCommand(
    name="status",
    description="Show current session info (session ID, message count)",
    handler=_handle_status,
))

register_command(SlashCommand(
    name="help",
    description="Show this help message",
    handler=_handle_help,
))

register_command(SlashCommand(
    name="model",
    description="List models, or `/model <name>` to switch the active model",
    handler=_handle_model,
))

register_command(SlashCommand(
    name="stop",
    description="Stop the agent's current run in this channel (alias: /cancel)",
    handler=_handle_stop,
))

register_command(SlashCommand(
    name="area",
    description="Show the area this channel resolves to",
    handler=_handle_area,
))

register_command(SlashCommand(
    name="retry",
    description="Re-run the last message in this channel",
    handler=_handle_retry,
))


# Known slash commands (sync registry + async commands handled by the interface).
# parse_slash_command checks this set; handle_command only dispatches registry commands.
SLASH_COMMANDS: set[str] = set(_COMMAND_REGISTRY.keys()) | {"hb", "cancel"}


# ---------------------------------------------------------------------------
# /retry — recall the last user turn (pure; the interface runs the agent)
# ---------------------------------------------------------------------------


def _first_text(entry: dict) -> str | None:
    """Return the first text block's text from a history entry, or None."""
    for block in entry.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return None


def split_for_retry(history: list[dict]) -> tuple[str, list[dict]] | None:
    """Find the last user turn to re-run.

    Returns ``(text, history_before_that_turn)`` — the text of the most recent
    user message and the history truncated to just before it, so the caller can
    re-run that message and regenerate a fresh reply without duplicating the
    turn. Returns ``None`` when there is no user turn carrying text to retry.
    """
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            text = _first_text(history[i])
            if text:
                return text, history[:i]
    return None


# ---------------------------------------------------------------------------
# Slash command parsing & dispatch
# ---------------------------------------------------------------------------

def parse_slash_command(content: str) -> tuple[str | None, str]:
    """Parse a slash command from message content.

    Args:
        content: The message content string.

    Returns:
        (command_name, rest) where command_name is the command (without /)
        or None if no slash command, and rest is the remaining text after
        the command. If the content starts with "/" but isn't a known
        command, returns (None, content) — treated as regular text.
    """
    if not content or not content.strip().startswith("/"):
        return None, content

    parts = content.strip().split(maxsplit=1)
    cmd = parts[0][1:]  # Strip the leading "/"

    if cmd.lower() in SLASH_COMMANDS:
        rest = parts[1] if len(parts) > 1 else ""
        return cmd.lower(), rest

    # Unknown slash — treat as regular content
    return None, content


def handle_command(command: str, rest: str, ctx: CommandContext) -> str | None:
    """Process a slash command and return the response text, or None.

    Returns None for commands that don't produce a text response
    (e.g., /ask which triggers the agent loop).

    Args:
        command: The command name (already lowercase).
        rest: The remainder of the message after the command.
        ctx: Command context with session info.

    Returns:
        Response string, or None if the command doesn't produce text.
    """
    cmd = _COMMAND_REGISTRY.get(command)
    if cmd is None:
        return None
    return cmd.handler(rest, ctx)


# ---------------------------------------------------------------------------
# /hb — Harbinger status probe (runs the `harbinger status` CLI, no LLM)
# ---------------------------------------------------------------------------

DISCORD_MAX_LEN = 2000
_FENCE = "```"

# A runner takes (command, timeout_s) and returns (returncode, stdout, stderr).
SubprocessRunner = Callable[[list[str], float], Awaitable[tuple[int, str, str]]]


async def _default_runner(cmd: list[str], timeout_s: float) -> tuple[int, str, str]:
    """Run a command, capturing stdout/stderr. Raises TimeoutError on timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except (asyncio.TimeoutError, TimeoutError):
        proc.kill()
        await proc.wait()
        raise
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def _wrap_code_block(body: str) -> str:
    """Wrap text in a Discord code block, truncated to fit the length limit."""
    budget = DISCORD_MAX_LEN - (len(_FENCE) * 2) - 2  # fences + two newlines
    if len(body) > budget:
        body = body[: budget - 1] + "…"
    return f"{_FENCE}\n{body}\n{_FENCE}"


async def probe_harbinger_status(
    *,
    runner: SubprocessRunner | None = None,
    command: list[str] | None = None,
    timeout_s: float = 15.0,
) -> str:
    """Run `harbinger status` and return a Discord-ready message.

    Never raises: subprocess/timeout/spawn failures are turned into a short
    warning string the bot can post. ``runner`` is injectable for tests.
    """
    run = runner or _default_runner
    cmd = command or ["harbinger", "status"]
    try:
        rc, stdout, stderr = await run(cmd, timeout_s)
    except (asyncio.TimeoutError, TimeoutError):
        return f"⚠ `harbinger status` timed out after {timeout_s:.0f}s"
    except (FileNotFoundError, OSError) as e:
        logger.warning("harbinger probe failed to spawn: %s", e)
        return f"⚠ could not run `harbinger`: {e}"

    if rc != 0:
        detail = (stderr or stdout or "").strip() or "(no output)"
        return f"⚠ `harbinger status` failed (rc={rc}):\n" + _wrap_code_block(detail)
    return _wrap_code_block(stdout.strip() or "(no output)")


# ---------------------------------------------------------------------------
# Message chunking
# ---------------------------------------------------------------------------


def chunk_message(text: str, max_length: int = 2000) -> list[str]:
    """Split a long message into chunks that fit Discord's length limit.

    Tries to split at paragraph boundaries first, then word boundaries.
    Never splits mid-word unless the word itself exceeds the limit.

    Args:
        text: The full response text.
        max_length: Maximum length per chunk (default: 2000).

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


def tail_for_stream(text: str, max_length: int = DISCORD_MAX_LEN) -> str:
    """Trim in-progress streaming text to the LAST ``max_length`` chars.

    The live progress message is a single Discord message edited in place, so
    it cannot grow past the length limit. Unlike the final response (which is
    split across several messages by :func:`chunk_message`), the in-progress
    view only needs the LATEST activity, so we keep the tail rather than the
    head. Keeping the head froze the message on stale tool lines once a long
    run first overflowed the limit (the "40 minutes of no visibility" bug).

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


# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------


def build_tool_embed(
    tool_name: str,
    status: str,
    duration_ms: int | None = None,
    error: str | None = None,
) -> dict:
    """Build a Discord embed payload for tool execution updates.

    The returned dict matches Discord's embed structure and is passed
    to discord.py's Embed.from_dict() or the adapter's embed builder.

    Args:
        tool_name: Name of the tool being executed.
        status: Human-readable status ("running", "completed", "failed").
        duration_ms: Optional execution duration in milliseconds.
        error: Optional error message if the tool failed.

    Returns:
        Embed dict compatible with Discord's embed API.
    """
    color = _status_color(status)
    fields = []

    if duration_ms is not None:
        fields.append({
            "name": "Duration",
            "value": f"{duration_ms}ms",
            "inline": True,
        })

    if error is not None:
        fields.append({
            "name": "Error",
            "value": _truncate(error, 1024),
            "inline": False,
        })

    return {
        "title": f"🔧 {tool_name}",
        "description": status,
        "color": color,
        "fields": fields if fields else None,
    }


def _status_color(status: str) -> int:
    """Return a color hex (as int) based on tool status."""
    if status == "running":
        return 16753920   # Yellow: 0xFFE000
    elif status == "completed":
        return 65280      # Green: 0x00FF00
    elif status == "failed":
        return 16711680   # Red: 0xFF0000
    return 9497630        # Grey (default)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
