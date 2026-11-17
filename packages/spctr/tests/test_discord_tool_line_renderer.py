"""spc-1022 — Discord renders tool lines and progress through the renderer.

AC3 (Discord half): with the default ``DiscordConfig`` the placeholder sent
when a turn starts, every tool line and the whole in-progress message are
byte-for-byte what they were before the extraction — the literals pinned
here are the pre-bead strings copied out of the interface module
(``f"▶ \\`{name}\\` - \\`{inputs}\\`"``, the ✓/✖ result icons, the 80-char
cut, ``"⏳ Thinking..."``).

AC2 (Discord half): the same shapes rendered through an overridden
``DiscordConfig`` prove the templates, the width and the placeholder flow
from config (YAML → pydantic → rendered line); they cannot pass while the
interface hardcodes them.

Harness: the real ``DiscordInterface`` over a mock adapter and a mock agent
streaming scripted events, mirroring
``test_discord_interface.py::test_tool_events_merged_into_thinking_message``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from spctr.events import AgentDone, AgentTextDelta, AgentToolResult, AgentToolStart
from spctr.interfaces.discord.config import DiscordConfig
from spctr.interfaces.discord.interface import DiscordInterface

#: Pre-bead literals — the byte-for-byte contract with the old code.
THINKING = "⏳ Thinking..."
CHANNEL = 12345
PLACEHOLDER_ID = 888
GREP_START = "▶ `grep` - `{'pattern': 'test'}`"
GREP_OK = "✓ `grep` - `found it`"
GREP_ERROR = "✖ `grep` - `boom`"

#: The interface under test, so the scripted stream can flush its progress.
_live: list[DiscordInterface] = []


def _make_mock_agent(events: list) -> MagicMock:
    agent = MagicMock()
    agent.history = []
    agent.session_id = "discord-renderer-test"
    agent.stream_message = _scripted_stream(events)
    return agent


def _scripted_stream(events: list, flush_after: int | None = None):
    """An agent stream over scripted events.

    ``flush_after`` re-renders the progress state the interface accumulated
    for this channel, after the *n*-th event, the way a mid-turn flush does.
    The live interface instance is handed in by ``_drive``.
    """

    async def _stream(text: str) -> Any:
        for index, ev in enumerate(events):
            yield ev
            if flush_after is not None and index == flush_after - 1:
                _live[0]._render_progress(CHANNEL, PLACEHOLDER_ID)

    return _stream


def _make_mock_message(content: str, channel_id: int) -> MagicMock:
    message = MagicMock()
    message.content = content
    channel = MagicMock()
    channel.id = channel_id
    message.channel = channel
    author = MagicMock()
    author.id = 99999
    message.author = author
    return message


def _make_mock_adapter() -> MagicMock:
    mock_adapter = MagicMock()
    mock_adapter.bot_id = 11111
    mock_adapter.bot_mentions = ["<@11111>"]
    mock_adapter.start = AsyncMock()
    mock_adapter.stop = AsyncMock()
    mock_adapter.send_message = AsyncMock(return_value=MagicMock(id=888))
    mock_adapter.edit_message = AsyncMock()
    mock_adapter.send_embed = AsyncMock()
    mock_adapter.set_message_handler = MagicMock()
    mock_adapter.set_slash_handler = MagicMock()
    mock_adapter.interaction_send = AsyncMock()
    mock_adapter.interaction_defer = AsyncMock()
    mock_adapter.access_allowed = MagicMock(return_value=True)
    mock_adapter.is_thread_channel = MagicMock(return_value=False)
    mock_adapter.area_channel = MagicMock(return_value=(None, None))
    typing_ctx = AsyncMock()
    typing_ctx.__aenter__ = AsyncMock()
    typing_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_adapter.channel_typing = MagicMock(return_value=typing_ctx)
    return mock_adapter


async def _drive(
    events: list,
    config: DiscordConfig,
    progress_after: int | None = None,
) -> tuple[MagicMock, list[str], list[str]]:
    """Run one agent turn over a mock adapter.

    Returns the adapter, the bodies sent (the first is the turn's
    placeholder) and the bodies edited (progress and finalization alike).
    """
    interface = DiscordInterface(config)
    _live.clear()
    _live.append(interface)
    agent = MagicMock()
    agent.history = []
    agent.session_id = "discord-renderer-test"
    agent.stream_message = _scripted_stream(events, progress_after)
    with patch("spctr.interfaces.discord.interface.DiscordBotAdapter") as MockAdapter:
        mock_adapter = _make_mock_adapter()
        MockAdapter.return_value = mock_adapter
        await interface.start(agent)
        await interface._process_message(
            _make_mock_message(content="go", channel_id=CHANNEL)
        )
    sends = [c[0][1] for c in mock_adapter.send_message.call_args_list]
    edits = [c[0][2] for c in mock_adapter.edit_message.call_args_list]
    return mock_adapter, sends, edits


# ---------------------------------------------------------------------------
# AC3 — defaults render the pre-bead bytes
# ---------------------------------------------------------------------------


def test_default_config_supplies_the_pre_bead_literals():
    """The config defaults ARE the strings the interface used to hardcode."""
    config = DiscordConfig()
    assert config.tool_line_max_chars == 80
    assert config.tool_start_line == "▶ `{name}` - `{input}`"
    assert config.tool_result_ok_line == "✓ `{name}` - `{content}`"
    assert config.tool_result_error_line == "✖ `{name}` - `{content}`"
    assert config.thinking_message == "⏳ Thinking..."


def test_new_fields_are_yaml_config_fields():
    """YAML → pydantic plumbing for every new field (no code path needed)."""
    config = DiscordConfig.model_validate(
        yaml.safe_load(
            """
            tool_line_max_chars: 12
            tool_start_line: "RUN {name} {input}"
            tool_result_ok_line: "DONE {name} {content}"
            tool_result_error_line: "BAD {name} {content}"
            thinking_message: "…"
            """
        )
    )
    assert config.tool_line_max_chars == 12
    assert config.tool_start_line == "RUN {name} {input}"
    assert config.tool_result_ok_line == "DONE {name} {content}"
    assert config.tool_result_error_line == "BAD {name} {content}"
    assert config.thinking_message == "…"


def test_new_fields_are_documented():
    """AC2: every new field is a documented YAML parameter, per the project
    mandate — same check ``test_zulip_handoff`` runs for the Zulip block."""
    root = Path(__file__).resolve().parents[1]
    example = (root / "config.example.yaml").read_text(encoding="utf-8")
    reference = (root / "docs/CONFIG-REFERENCE.md").read_text(encoding="utf-8")
    defaults = {
        "tool_line_max_chars": "80",
        "tool_start_line": '"▶ `{name}` - `{input}`"',
        "tool_result_ok_line": '"✓ `{name}` - `{content}`"',
        "tool_result_error_line": '"✖ `{name}` - `{content}`"',
        "thinking_message": '"⏳ Thinking..."',
    }
    for name, default in defaults.items():
        assert f"#   {name}: {default}" in example, (
            f"config.example.yaml does not show `discord.{name}`"
        )
        assert f"| `{name}` |" in reference, (
            f"docs/CONFIG-REFERENCE.md does not document `discord.{name}`"
        )
        assert default in reference, (
            f"docs/CONFIG-REFERENCE.md does not state the default of "
            f"`discord.{name}`: {default}"
        )


@pytest.mark.asyncio
async def test_default_placeholder_send_is_byte_identical():
    _, sends, _ = await _drive([AgentDone()], DiscordConfig())
    assert sends[0] == "⏳ Thinking..."


@pytest.mark.asyncio
async def test_default_tool_lines_are_byte_identical():
    """The accumulated tool-start and tool-result lines, rendered mid-turn
    before any text arrives, are exactly the pre-bead bytes."""
    _, _, edits = await _drive(
        [
            AgentToolStart(tool_id="t1", name="grep", input={"pattern": "test"}),
            AgentToolResult(
                tool_id="t1", name="grep", content="found it", is_error=False
            ),
            AgentDone(),
        ],
        DiscordConfig(),
        progress_after=2,
    )
    tool_edits = [c for c in edits if "▶" in c]
    assert tool_edits, f"no tool line was rendered: {edits!r}"
    assert tool_edits[-1] == (
        f"{GREP_START}\n{GREP_OK}\n\n{THINKING}"
    )


@pytest.mark.asyncio
async def test_default_result_line_flattens_newlines_and_names_an_unnamed_tool():
    _, _, edits = await _drive(
        [
            AgentToolResult(
                tool_id="t1", name=None, content="l1\nl2", is_error=True
            ),
            AgentTextDelta(text="Done"),
            AgentDone(),
        ],
        DiscordConfig(),
    )
    assert f"✖ `tool` - `l1|l2`\n\n{THINKING}" in edits


@pytest.mark.asyncio
async def test_default_truncation_matches_the_old_80_char_cut():
    rendered_input = str({"k": "x" * 200})
    assert len(rendered_input) == 209
    _, _, edits = await _drive(
        [
            AgentToolStart(tool_id="t1", name="grep", input={"k": "x" * 200}),
            AgentToolResult(
                tool_id="t1", name="grep", content="y" * 120, is_error=False
            ),
            AgentTextDelta(text="Done"),
            AgentDone(),
        ],
        DiscordConfig(),
    )
    assert (
        f"▶ `grep` - `{rendered_input[:80]}...`\n"
        f"✓ `grep` - `{'y' * 80}...`\n\n{THINKING}"
    ) in edits


@pytest.mark.asyncio
async def test_default_progress_message_with_text_is_byte_identical():
    """Tool lines, a blank line, then the streamed text plus '...' — the
    text path of the same renderer."""
    _, _, edits = await _drive(
        [
            AgentToolStart(tool_id="t1", name="grep", input={"pattern": "test"}),
            AgentTextDelta(text="partial"),
            AgentDone(),
        ],
        DiscordConfig(edit_stream_interval=0.0),
        progress_after=2,
    )
    assert f"{GREP_START}\n\npartial..." in edits
    assert edits[-1] == "partial"


# ---------------------------------------------------------------------------
# AC2 — templates, width and placeholder come from config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_templates_and_thinking_placeholder_come_from_config():
    _, sends, edits = await _drive(
        [
            AgentToolStart(tool_id="t1", name="grep", input={"pattern": "test"}),
            AgentToolResult(
                tool_id="t1", name="grep", content="found it", is_error=False
            ),
            AgentToolResult(
                tool_id="t1", name="grep", content="boom", is_error=True
            ),
            AgentDone(),
        ],
        DiscordConfig(
            tool_start_line="RUN {name} {input}",
            tool_result_ok_line="DONE {name} {content}",
            tool_result_error_line="BAD {name} {content}",
            thinking_message="hmm",
        ),
        progress_after=3,
    )
    assert sends[0] == "hmm"
    assert (
        "RUN grep {'pattern': 'test'}\nDONE grep found it\nBAD grep boom\n\nhmm"
    ) in edits
    assert not any(
        icon in "".join(sends + edits) for icon in ("▶", "✓", "✖", "⏳")
    )


@pytest.mark.asyncio
async def test_line_limit_comes_from_config():
    _, _, edits = await _drive(
        [
            AgentToolStart(tool_id="t1", name="grep", input={"pattern": "test"}),
            AgentDone(),
        ],
        DiscordConfig(tool_line_max_chars=7),
        progress_after=1,
    )
    tool_edits = [c for c in edits if "grep" in c]
    assert tool_edits[-1] == "▶ `grep` - `{'patte...`\n\n⏳ Thinking..."


@pytest.mark.asyncio
async def test_message_length_limit_applies_to_progress():
    """The progress edit goes through the shared renderer, so a message
    longer than the config limit arrives tailed: exactly the limit in
    length, the tail marker, the last tool line kept whole, then the end
    of the streamed text and its in-progress marker."""
    tool_line = "▶ `grep` - `{'pattern': 'test'}`"
    text = "".join(str(i) for i in range(300))
    _, _, edits = await _drive(
        [
            AgentToolStart(tool_id="t1", name="grep", input={"pattern": "test"}),
            AgentTextDelta(text=text),
            AgentDone(),
        ],
        DiscordConfig(edit_stream_interval=0.0, max_message_length=60),
        progress_after=2,
    )
    marker = "[…]\n"
    progress = [c for c in edits if c.startswith(marker)]
    assert progress, f"progress was never tailed: {edits!r}"
    for content in progress:
        assert len(content) == 60
    tail = ("{pattern': 'test'}`\n\n" + text + "...")
    assert progress[0] == marker + tail[-(60 - len(marker)):]
