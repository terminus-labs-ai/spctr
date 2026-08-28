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
    tail_for_stream,
    should_respond,
    split_for_retry,
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
# /area, /retry, /status area readout (obsidian-fqps.5)
# ---------------------------------------------------------------------------

class TestAreaCommand:
    def test_registered_in_registry(self) -> None:
        assert "area" in get_registered_commands()

    def test_shows_resolved_area(self) -> None:
        ctx = CommandContext(0, "s", 0, area="normandy")
        response = handle_command("area", "", ctx)
        assert response is not None
        assert "normandy" in response

    def test_empty_area_reports_none(self) -> None:
        ctx = CommandContext(0, "s", 0, area="")
        response = handle_command("area", "", ctx)
        assert response is not None
        assert "none" in response.lower()
        assert "`" not in response  # no code-fenced area name

    def test_missing_area_reports_none(self) -> None:
        ctx = CommandContext(0, "s", 0, area=None)
        response = handle_command("area", "", ctx)
        assert response is not None
        assert "none" in response.lower()


class TestStatusAreaReadout:
    def test_status_includes_area_when_present(self) -> None:
        ctx = CommandContext(0, "discord-0", 0, area="normandy")
        response = handle_command("status", "", ctx)
        assert response is not None
        assert "Area" in response
        assert "normandy" in response

    def test_status_shows_none_for_empty_area(self) -> None:
        ctx = CommandContext(0, "discord-0", 0, area="")
        response = handle_command("status", "", ctx)
        assert response is not None
        assert "Area" in response
        assert "none" in response.lower()

    def test_status_omits_area_when_unknown(self) -> None:
        ctx = CommandContext(0, "discord-0", 0, area=None)
        response = handle_command("status", "", ctx)
        assert response is not None
        assert "Area" not in response


class TestRetryRegistration:
    def test_registered_in_registry(self) -> None:
        assert "retry" in get_registered_commands()

    def test_retry_returns_none(self) -> None:
        """/retry produces no sync text — handled async in the interface."""
        assert handle_command("retry", "", _ctx()) is None

    def test_help_lists_area_and_retry(self) -> None:
        response = handle_command("help", "", _ctx())
        assert response is not None
        assert "/area" in response
        assert "/retry" in response


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


class TestSplitForRetry:
    def test_empty_history_returns_none(self) -> None:
        assert split_for_retry([]) is None

    def test_no_user_turn_returns_none(self) -> None:
        assert split_for_retry([_assistant("hi")]) is None

    def test_recalls_last_user_and_trims_from_it(self) -> None:
        history = [_user("first"), _assistant("a1"), _user("second"), _assistant("a2")]
        result = split_for_retry(history)
        assert result is not None
        text, trimmed = result
        assert text == "second"
        # Everything from the recalled user turn onward is dropped.
        assert trimmed == [_user("first"), _assistant("a1")]

    def test_trailing_user_turn_is_recalled(self) -> None:
        history = [_user("only")]
        result = split_for_retry(history)
        assert result is not None
        text, trimmed = result
        assert text == "only"
        assert trimmed == []

    def test_skips_empty_text_user_turn(self) -> None:
        history = [_user("real"), _assistant("a1"), _user("")]
        result = split_for_retry(history)
        assert result is not None
        text, trimmed = result
        assert text == "real"
        assert trimmed == []

    def test_does_not_mutate_input(self) -> None:
        history = [_user("first"), _assistant("a1"), _user("second")]
        original = list(history)
        split_for_retry(history)
        assert history == original


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


# ---------------------------------------------------------------------------
# /model — pure render logic + /status model fields
# ---------------------------------------------------------------------------

from sr2_spectre.interfaces.discord.handler import render_model_command  # noqa: E402


class TestRenderModelCommand:
    MODELS = {"default", "fast"}

    def test_empty_lists_models_and_marks_active(self) -> None:
        out, selection = render_model_command("", self.MODELS, "default", True)
        assert selection is None
        assert "`default`" in out and "`fast`" in out
        assert "← active" in out
        # the marker sits on the active one
        active_line = next(ln for ln in out.splitlines() if "← active" in ln)
        assert "default" in active_line

    def test_unknown_name_errors_without_selection(self) -> None:
        out, selection = render_model_command("ghost", self.MODELS, "default", True)
        assert selection is None
        assert "Unknown model" in out and "ghost" in out

    def test_no_pointer_refuses_switch(self) -> None:
        out, selection = render_model_command("fast", self.MODELS, "default", False)
        assert selection is None
        assert "SR2_ACTIVE_MODEL_FILE" in out

    def test_reselecting_active_is_a_noop(self) -> None:
        out, selection = render_model_command("default", self.MODELS, "default", True)
        assert selection is None
        assert "Already using" in out

    def test_valid_switch_returns_selection(self) -> None:
        out, selection = render_model_command("fast", self.MODELS, "default", True)
        assert selection == "fast"
        assert "Switched to" in out and "fast" in out

    def test_switch_arg_is_stripped(self) -> None:
        out, selection = render_model_command("  fast ", self.MODELS, "default", True)
        assert selection == "fast"


class TestStatusModelFields:
    def test_status_shows_model_and_endpoint_when_present(self) -> None:
        ctx = CommandContext(
            channel_id=1,
            session_id="s",
            message_count=2,
            active_model="fast",
            model_label="m-fast @ http://fast/v1",
        )
        out = handle_command("status", "", ctx)
        assert "**Model:**" in out and "fast" in out
        assert "**Endpoint:**" in out and "http://fast/v1" in out

    def test_status_omits_model_lines_when_absent(self) -> None:
        ctx = CommandContext(channel_id=1, session_id="s", message_count=2)
        out = handle_command("status", "", ctx)
        assert "**Model:**" not in out
        assert "**Endpoint:**" not in out



class TestTailForStream:
    """tail_for_stream() — in-progress progress trims to the LATEST content.

    Regression: the live progress message head-truncated once accumulated
    tool lines first overflowed the limit, freezing the view on stale tool
    lines for the rest of a long run ("40 minutes of no visibility").
    """

    def test_short_text_unchanged(self) -> None:
        assert tail_for_stream("hello", 2000) == "hello"

    def test_text_at_limit_unchanged(self) -> None:
        text = "a" * 2000
        assert tail_for_stream(text, 2000) == text

    def test_keeps_tail_not_head(self) -> None:
        # Old tool lines first, latest activity + text last.
        text = "OLD\n" * 1000 + "LATEST ACTIVITY"
        out = tail_for_stream(text, 2000)
        assert out.endswith("LATEST ACTIVITY")  # newest content survives
        assert len(out) <= 2000
        # Most of the stale head is dropped, not frozen in view.
        assert out.count("OLD") < text.count("OLD")

    def test_result_within_limit(self) -> None:
        out = tail_for_stream("x" * 5000, 2000)
        assert len(out) <= 2000

    def test_marks_truncation(self) -> None:
        out = tail_for_stream("x" * 5000, 2000)
        assert out.startswith("[…]")


# ---------------------------------------------------------------------------
# /status token readout (spc-82)
# ---------------------------------------------------------------------------

class TestStatusTokenReadout:
    def test_status_shows_tokens_when_reported(self) -> None:
        ctx = CommandContext(
            0, "discord-0", 3, tokens_in=1200, tokens_out=350
        )
        response = handle_command("status", "", ctx)
        assert response is not None
        assert "Tokens" in response
        assert "1,200" in response
        assert "350" in response

    def test_status_shows_context_estimate_always(self) -> None:
        ctx = CommandContext(0, "discord-0", 3, context_tokens=4521)
        response = handle_command("status", "", ctx)
        assert response is not None
        assert "Context" in response
        assert "4,521" in response

    def test_status_omits_token_lines_when_zero(self) -> None:
        """Fresh channel (no usage, no context) shows neither line."""
        ctx = CommandContext(0, "discord-0", 0)
        response = handle_command("status", "", ctx)
        assert response is not None
        assert "Tokens" not in response
        assert "Context" not in response

    def test_status_shows_context_without_real_usage(self) -> None:
        """Endpoint silent on usage: context estimate still renders."""
        ctx = CommandContext(0, "discord-0", 2, context_tokens=128)
        response = handle_command("status", "", ctx)
        assert response is not None
        assert "Tokens" not in response
        assert "Context" in response
        assert "128" in response
