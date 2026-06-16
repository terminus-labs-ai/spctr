"""Tests for Discord handler — pure logic (no discord.py dependency).

Covers:
1.  should_respond() — mention filter logic
2.  parse_slash_command() — command parsing
3.  handle_command() — command response generation (registry-based)
4.  CommandContext — session context for command handlers
5.  SlashCommand registry — registration and discovery
6.  chunk_message() — Discord length limit splitting
7.  build_tool_embed() — embed payload construction
"""
from __future__ import annotations

import pytest

from sr2_spectre.interfaces.discord.handler import (
    CommandContext,
    SlashCommand,
    build_tool_embed,
    chunk_message,
    get_registered_commands,
    handle_command,
    parse_slash_command,
    probe_harbinger_status,
    register_command,
    should_respond,
)


# ---------------------------------------------------------------------------
# should_respond()
# ---------------------------------------------------------------------------

class TestShouldRespond:
    def test_always_responds_when_mention_only_false(self) -> None:
        """When mention_only=False, always respond."""
        assert should_respond("hello", False, None, None) is True
        assert should_respond("", False, None, None) is True

    def test_responds_to_bot_id_mention(self) -> None:
        """Responds when message contains <@BotID>."""
        assert should_respond("hi <@12345>", True, 12345, None) is True

    def test_responds_to_exclaim_mention(self) -> None:
        """Responds when message contains <@!BotID>."""
        assert should_respond("hi <@!12345>", True, 12345, None) is True

    def test_ignores_other_mentions(self) -> None:
        """Doesn't respond to mentions of other bots."""
        assert should_respond("hi <@99999>", True, 12345, None) is False

    def test_responds_to_pre_rendered_mention(self) -> None:
        """Responds when a pre-rendered mention string is found."""
        assert should_respond("hello <@12345>", True, 12345, ["<@12345>"]) is True

    def test_no_response_without_mention(self) -> None:
        """Doesn't respond when mention_only=True and no mention present."""
        assert should_respond("hello world", True, 12345, None) is False

    def test_empty_content_with_mention_only_false(self) -> None:
        """Empty content still responds when mention_only is False."""
        assert should_respond("", False, None, None) is True

    def test_empty_content_with_mention_only_true(self) -> None:
        """Empty content doesn't respond when mention_only is True."""
        assert should_respond("", True, 12345, None) is False


# ---------------------------------------------------------------------------
# parse_slash_command()
# ---------------------------------------------------------------------------

class TestParseSlashCommand:
    def test_parses_known_command(self) -> None:
        cmd, rest = parse_slash_command("/reset")
        assert cmd == "reset"
        assert rest == ""

    def test_parses_command_with_args(self) -> None:
        cmd, rest = parse_slash_command("/ask what is the meaning of life")
        assert cmd == "ask"
        assert rest == "what is the meaning of life"

    def test_unknown_slash_returns_none(self) -> None:
        cmd, rest = parse_slash_command("/unknown hello")
        assert cmd is None
        assert rest == "/unknown hello"

    def test_no_slash_returns_none(self) -> None:
        cmd, rest = parse_slash_command("hello world")
        assert cmd is None
        assert rest == "hello world"

    def test_empty_string_returns_none(self) -> None:
        cmd, rest = parse_slash_command("")
        assert cmd is None
        assert rest == ""

    def test_case_insensitive(self) -> None:
        cmd, rest = parse_slash_command("/RESET")
        assert cmd == "reset"
        assert rest == ""

    def test_slash_help_command(self) -> None:
        cmd, rest = parse_slash_command("/help")
        assert cmd == "help"
        assert rest == ""

    def test_slash_status_command(self) -> None:
        cmd, rest = parse_slash_command("/status")
        assert cmd == "status"
        assert rest == ""

    def test_slash_hb_command(self) -> None:
        cmd, rest = parse_slash_command("/hb")
        assert cmd == "hb"
        assert rest == ""


# ---------------------------------------------------------------------------
# CommandContext
# ---------------------------------------------------------------------------

class TestCommandContext:
    def test_command_context_fields(self) -> None:
        ctx = CommandContext(channel_id=123, session_id="discord-123", message_count=5)
        assert ctx.channel_id == 123
        assert ctx.session_id == "discord-123"
        assert ctx.message_count == 5

    def test_command_context_is_frozen(self) -> None:
        ctx = CommandContext(channel_id=123, session_id="discord-123", message_count=5)
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.channel_id = 456  # type: ignore


# ---------------------------------------------------------------------------
# SlashCommand registry
# ---------------------------------------------------------------------------

class TestSlashCommandRegistry:
    def test_builtin_commands_registered(self) -> None:
        """Built-in commands are registered at module import time."""
        cmds = get_registered_commands()
        assert "ask" in cmds
        assert "reset" in cmds
        assert "status" in cmds
        assert "help" in cmds
        # /hb is async-only — not in the sync registry
        assert "hb" not in cmds

    def test_slash_command_dataclass(self) -> None:
        cmd = SlashCommand(name="test", description="A test command", handler=lambda r, c: "ok")
        assert cmd.name == "test"
        assert cmd.description == "A test command"
        assert cmd.handler("", CommandContext(0, "s", 0)) == "ok"

    def test_register_command_returns_command(self) -> None:
        """register_command returns the command for chaining."""
        result = register_command(SlashCommand(
            name="echo_test",
            description="Echo test",
            handler=lambda r, c: r,
        ))
        assert result.name == "echo_test"
        # Clean up
        del get_registered_commands()["echo_test"]

    def test_get_registered_commands_returns_snapshot(self) -> None:
        """Modifying the snapshot doesn't affect the registry."""
        snapshot = get_registered_commands()
        snapshot["fake"] = SlashCommand(
            name="fake", description="fake", handler=lambda r, c: None
        )
        assert "fake" not in get_registered_commands()


# ---------------------------------------------------------------------------
# handle_command()
# ---------------------------------------------------------------------------

def _ctx(message_count: int = 0) -> CommandContext:
    """Helper to create a default CommandContext."""
    return CommandContext(
        channel_id=12345,
        session_id="discord-12345",
        message_count=message_count,
    )


class TestHandleCommand:
    def test_ask_returns_none(self) -> None:
        """ /ask returns None (triggers agent loop)."""
        assert handle_command("ask", "hello", _ctx()) is None

    def test_reset_returns_confirmation(self) -> None:
        response = handle_command("reset", "", _ctx())
        assert response is not None
        assert "reset" in response.lower()

    def test_status_returns_session_info(self) -> None:
        """ /status renders real session info from context."""
        ctx = CommandContext(channel_id=999, session_id="discord-999", message_count=7)
        response = handle_command("status", "", ctx)
        assert response is not None
        assert "discord-999" in response
        assert "7" in response
        assert "Session" in response
        assert "Messages" in response

    def test_status_with_zero_messages(self) -> None:
        """ /status works with zero messages."""
        ctx = CommandContext(channel_id=1, session_id="discord-1", message_count=0)
        response = handle_command("status", "", ctx)
        assert response is not None
        assert "0" in response

    def test_help_returns_help_text(self) -> None:
        response = handle_command("help", "", _ctx())
        assert response is not None
        assert "/ask" in response
        assert "/reset" in response
        assert "/status" in response
        assert "/help" in response
        assert "/hb" in response

    def test_hb_returns_none(self) -> None:
        """/hb produces no sync text — handled async in the interface."""
        assert handle_command("hb", "", _ctx()) is None

    def test_unknown_command_returns_none(self) -> None:
        assert handle_command("unknown", "stuff", _ctx()) is None


# ---------------------------------------------------------------------------
# chunk_message()
# ---------------------------------------------------------------------------

class TestChunkMessage:
    def test_short_message_returns_single_chunk(self) -> None:
        result = chunk_message("hello")
        assert result == ["hello"]

    def test_message_at_limit(self) -> None:
        text = "x" * 2000
        result = chunk_message(text, 2000)
        assert len(result) == 1
        assert result[0] == text

    def test_long_message_splits(self) -> None:
        text = "x" * 3000
        result = chunk_message(text, 2000)
        assert len(result) == 2
        for chunk in result:
            assert len(chunk) <= 2000  # Must not exceed max_length

    def test_paragraph_split_preferred(self) -> None:
        """Split at paragraph boundaries when possible."""
        text = "A" * 1000 + "\n\n" + "B" * 1500
        result = chunk_message(text, 2000)
        assert len(result) >= 2

    def test_word_boundary_split(self) -> None:
        """Split at word boundaries when no paragraph break available."""
        words = "word " * 300  # ~1800 chars
        text = words + "extra" * 200  # Well over 2000
        result = chunk_message(text, 2000)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 2003

    def test_hard_split_for_very_long_word(self) -> None:
        """Hard split when a single word exceeds the limit."""
        word = "x" * 2500
        result = chunk_message(word, 2000)
        assert len(result) >= 2

    def test_multiple_chunks(self) -> None:
        """Split into multiple chunks for very long text."""
        text = "x" * 6000
        result = chunk_message(text, 2000)
        assert len(result) >= 3


# ---------------------------------------------------------------------------
# build_tool_embed()
# ---------------------------------------------------------------------------

class TestBuildToolEmbed:
    def test_running_status(self) -> None:
        embed = build_tool_embed("search", "running")
        assert embed["title"] == "🔧 search"
        assert embed["description"] == "running"
        assert embed["color"] == 16753920

    def test_completed_status(self) -> None:
        embed = build_tool_embed("search", "completed")
        assert embed["color"] == 65280

    def test_failed_status(self) -> None:
        embed = build_tool_embed("search", "failed", error="not found")
        assert embed["color"] == 16711680
        assert embed["fields"] is not None
        assert any(f["name"] == "Error" for f in embed["fields"])

    def test_duration_field(self) -> None:
        embed = build_tool_embed("search", "completed", duration_ms=1500)
        assert embed["fields"] is not None
        duration_field = next(f for f in embed["fields"] if f["name"] == "Duration")
        assert duration_field["value"] == "1500ms"

    def test_no_fields_when_optional_absent(self) -> None:
        embed = build_tool_embed("search", "completed")
        assert embed["fields"] is None


# ---------------------------------------------------------------------------
# probe_harbinger_status()
# ---------------------------------------------------------------------------

class TestProbeHarbingerStatus:
    async def test_ok_wraps_stdout_in_code_block(self) -> None:
        async def fake_runner(cmd, timeout_s):
            return (0, "Harbinger status — live\nLive slots: busy=1", "")

        out = await probe_harbinger_status(runner=fake_runner)
        assert out.startswith("```")
        assert out.rstrip().endswith("```")
        assert "Live slots: busy=1" in out

    async def test_default_command_is_harbinger_status(self) -> None:
        seen = {}

        async def fake_runner(cmd, timeout_s):
            seen["cmd"] = cmd
            return (0, "ok", "")

        await probe_harbinger_status(runner=fake_runner)
        assert seen["cmd"] == ["harbinger", "status"]

    async def test_nonzero_exit_reports_failure(self) -> None:
        async def fake_runner(cmd, timeout_s):
            return (1, "", "boom: config not found")

        out = await probe_harbinger_status(runner=fake_runner)
        assert "failed" in out.lower()
        assert "boom: config not found" in out

    async def test_timeout_reports_timed_out(self) -> None:
        async def fake_runner(cmd, timeout_s):
            raise TimeoutError()

        out = await probe_harbinger_status(runner=fake_runner, timeout_s=2.0)
        assert "timed out" in out.lower()

    async def test_spawn_error_reports_cleanly(self) -> None:
        async def fake_runner(cmd, timeout_s):
            raise FileNotFoundError("harbinger not on PATH")

        out = await probe_harbinger_status(runner=fake_runner)
        assert "harbinger" in out.lower()
        # Does not raise; returns a string the bot can post.
        assert isinstance(out, str)

    async def test_long_output_truncated_to_discord_limit(self) -> None:
        async def fake_runner(cmd, timeout_s):
            return (0, "x" * 5000, "")

        out = await probe_harbinger_status(runner=fake_runner)
        assert len(out) <= 2000
