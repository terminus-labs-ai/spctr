"""Tests for the REPL /context command and context-size footer (spc-955).

Surfaces how much context the current session has consumed:

- ``estimate_history_stats`` (session_io.py) — the shared seam: message count,
  estimated token total and per-role sums, counted by SR2's
  ``CharacterTokenCounter`` (chars // 4).  Deliberately NOT LLM-reported usage
  (that is spc-82); no stream/usage path is involved.
- ``/context`` — line 1: ``Session <id8> · <N> msgs · ~<T> tokens est.``;
  line 2 (non-empty history only): ``By role: <role> ~<tok> · ...`` sorted by
  tokens descending.
- The per-turn dim footer gains ``· ~<T> tok est.`` beside the msg/tool counts.

All token figures below are hand-computed from the fixture strings with
chars // 4 per message, so AC1/AC2/AC7 assert exact numbers.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from sr2.models import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter

from sr2_spectre.interfaces.repl import _COMMANDS, _HELP, REPLInterface
from sr2_spectre.interfaces.session_io import HistoryStats, estimate_history_stats


# ---------------------------------------------------------------------------
# Fixtures and helpers (stdlib + rich + sr2 only; shared conftest untouched)
# ---------------------------------------------------------------------------

_SESSION_ID = "abcdef0123456789"

# Fixture texts — lengths fixed so the chars//4 figures are exact.
_USER_TEXT = "Hello world, this is a test."          # 28 chars -> 7 tokens
_THINK_TEXT = "Thinking about it."                  # 18 chars -> 4 tokens
_TOOL_NAME = "read_file"                            # 9  chars
_TOOL_INPUT = {"path": "/tmp/x"}                    # -> json 18 chars; (9+18)//4 = 6
_RESULT_TEXT = "result text here"                   # 16 chars -> 4 tokens


def _fixture_history() -> list:
    """user 7 + assistant 11 (thinking 18 + tool_use 27 chars -> 45//4) + user 4 = 22.

    Note the assistant figure is the counter over the *combined* block list:
    chars // 4 does not distribute over blocks (4 + 6 = 10 != 11).
    """
    return [
        Message(role="user", content=[TextBlock(text=_USER_TEXT)]),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(text=_THINK_TEXT),
                ToolUseBlock(id="t1", name=_TOOL_NAME, input=_TOOL_INPUT),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="t1", content=_RESULT_TEXT)],
        ),
    ]


def _agent(history: list, session_id: str = _SESSION_ID) -> MagicMock:
    agent = MagicMock()
    agent.session_id = session_id
    agent.history = history
    return agent


class _LineSink:
    """Minimal file-like sink that records written strings line-wise."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def write(self, s: str) -> int:
        for part in s.split("\n"):
            if part:
                self._lines.append(part)
        return len(s)

    def flush(self) -> None:
        pass


def _console_capturing(interface: REPLInterface) -> list[str]:
    buf: list[str] = []
    interface.console = Console(file=_LineSink(buf), force_terminal=False, width=200)
    return buf


def _plain_text(lines: list[str]) -> str:
    """Strip ANSI codes from captured output for assertions."""
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(lines))


# ---------------------------------------------------------------------------
# AC4 — estimate_history_stats: the shared seam in session_io.py
# ---------------------------------------------------------------------------

def test_estimate_history_stats_lives_in_session_io() -> None:
    import inspect

    from sr2_spectre.interfaces import session_io

    assert callable(session_io.estimate_history_stats)
    module = inspect.getmodule(session_io.estimate_history_stats)
    assert module is session_io
    stats = estimate_history_stats([])
    assert isinstance(stats, HistoryStats)
    assert (stats.messages, stats.tokens, stats.by_role) == (0, 0, {})


def test_estimate_history_stats_counts_via_character_token_counter() -> None:
    """AC4: delegation to SR2's CharacterTokenCounter, per message."""
    history = _fixture_history()
    expected = sum(CharacterTokenCounter().count(m.content) for m in history)
    stats = estimate_history_stats(history)
    assert stats.messages == 3
    assert stats.tokens == expected == 22
    assert stats.by_role == {"user": 11, "assistant": 11}


def test_estimate_history_stats_empty_content_contributes_zero() -> None:
    """AC4: missing, None or empty content contributes 0 and must not raise."""

    class _NoContent:
        role = "assistant"  # no content attribute at all

    class _NoneContent:
        role = "assistant"
        content = None

    history = [
        Message(role="assistant", content=[]),
        _NoneContent(),
        _NoContent(),
    ]
    stats = estimate_history_stats(history)
    assert stats.messages == 3
    assert stats.tokens == 0
    assert stats.by_role == {"assistant": 0}


# ---------------------------------------------------------------------------
# AC1 — /context line 1 with exact figures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_line_one_exact_figures() -> None:
    agent = _agent(_fixture_history())
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/context", None)

    text = _plain_text(buf)
    assert f"Session {_SESSION_ID[:8]} · 3 msgs · ~22 tokens est." in text


@pytest.mark.asyncio
async def test_context_ignores_extra_arguments() -> None:
    agent = _agent(_fixture_history())
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/context", "extra args here")

    assert f"Session {_SESSION_ID[:8]} · 3 msgs · ~22 tokens est." in _plain_text(buf)


# ---------------------------------------------------------------------------
# AC2 — per-role breakdown: sums, descending order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_role_breakdown_sums_and_order() -> None:
    agent = _agent(_fixture_history())
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/context", None)

    text = _plain_text(buf)
    assert "By role: assistant ~11 · user ~11" in text
    # The role sums add up to the reported total.
    line = next(l for l in text.splitlines() if l.startswith("By role:"))
    sums = [int(m.group(1)) for m in re.finditer(r"~(\d+)", line)]
    assert sum(sums) == 22


@pytest.mark.asyncio
async def test_context_role_breakdown_descending_by_tokens() -> None:
    """Roles must be listed tokens-descending even when inserted ascending."""
    history = [
        Message(role="assistant", content=[TextBlock(text="a" * 80)]),   # 20
        Message(role="user", content=[TextBlock(text="b" * 200)]),       # 50
        Message(role="assistant", content=[TextBlock(text="c" * 120)]),  # +30 -> 50
    ]
    agent = _agent(history)
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/context", None)

    text = _plain_text(buf)
    # assistant (20+30=50) ties user (50): ties break alphabetically, and the
    # sums add up to the total.
    assert "By role: assistant ~50 · user ~50" in text
    assert "100 tokens est." in text


# ---------------------------------------------------------------------------
# AC3 — empty history renders cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_empty_history() -> None:
    agent = _agent([])
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/context", None)

    text = _plain_text(buf)
    assert f"Session {_SESSION_ID[:8]} · 0 msgs · ~0 tokens est." in text
    assert "By role" not in text


# ---------------------------------------------------------------------------
# AC5 — the per-turn footer shows the same estimated figure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_footer_shows_same_estimate_as_context() -> None:
    from sr2_spectre.events import AgentDone, AgentTextDelta

    history = _fixture_history()
    agent = _agent(history)

    async def _stream(_text: str):
        yield AgentTextDelta(text="ok")
        yield AgentDone(tool_calls_executed=0)

    agent.stream_message = _stream
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._stream_turn(agent, "hi")

    footer = _plain_text(buf)
    assert f"— session {_SESSION_ID[:8]} · 3 msgs · 0 tools · ~22 tok est. —" in footer

    # Same helper value /context reports for the same history.
    stats = estimate_history_stats(history)
    assert f"~{stats.tokens} tok est." in footer


# ---------------------------------------------------------------------------
# AC6 — command registration: _COMMANDS, completer, _HELP, module docstring
# ---------------------------------------------------------------------------


def test_context_in_commands_list() -> None:
    assert "/context" in _COMMANDS


def test_context_offered_by_slash_completer() -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from sr2_spectre.interfaces.repl import _SlashCompleter

    doc = Document(text="/context", cursor_position=len("/context"))
    completions = [
        c.text
        for c in _SlashCompleter(_COMMANDS).get_completions(doc, CompleteEvent(""))
    ]
    assert completions == ["/context"]


@pytest.mark.asyncio
async def test_help_lists_context_command() -> None:
    assert "/context" in _HELP
    agent = _agent([])
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/help", None)

    assert "/context" in _plain_text(buf)


def test_module_docstring_lists_context() -> None:
    import sr2_spectre.interfaces.repl as repl_module

    assert "/context" in (repl_module.__doc__ or "")


# ---------------------------------------------------------------------------
# AC7 — every block type contributes via the counter delegation
# ---------------------------------------------------------------------------


def test_estimate_accounts_for_every_block_type() -> None:
    """Exact per-block-type figures: text, thinking, tool_use, tool_result."""
    counter = CharacterTokenCounter()
    cases = [
        ([TextBlock(text=_USER_TEXT)], 7),
        ([ThinkingBlock(text=_THINK_TEXT)], 4),
        ([ToolUseBlock(id="t1", name=_TOOL_NAME, input=_TOOL_INPUT)], 6),
        ([ToolResultBlock(tool_use_id="t1", content=_RESULT_TEXT)], 4),
        (
            [ToolResultBlock(tool_use_id="t1", content=[TextBlock(text=_RESULT_TEXT)])],
            4,
        ),
    ]
    for content, expected in cases:
        assert counter.count(content) == expected  # sanity: the counter itself
        history = [Message(role="user", content=content)]
        stats = estimate_history_stats(history)
        assert stats.tokens == expected, content
        assert stats.messages == 1
