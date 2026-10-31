"""Tests for per-tool-result truncation guard (obsidian-tqc, layer 1).

Prevents a single oversized tool result from flooding the context window
and crashing the SR2/LLM request with exceed_context_size_error.

Covers:
  A. AgentConfig.tool_result_max_bytes default value
  B. _execute_tool truncates oversized results
  C. _execute_tool does NOT truncate results under the cap
  D. Truncated content includes a clear marker with size info
  E. Error results also get truncated
  F. Configurable cap via AgentConfig
  G. The cap and the marker's original size are UTF-8 bytes (obsidian-ejjc)
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.models import ToolResultBlock, ToolUseBlock
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**agent_kwargs) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test", **agent_kwargs),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={"layers": [
            {"name": "system", "target": "system", "resolvers": [
                {"type": "static", "config": {"text": "You are helpful."}}
            ]},
        ]},
    )


# ---------------------------------------------------------------------------
# A. AgentConfig.tool_result_max_bytes default
# ---------------------------------------------------------------------------

class TestToolResultMaxBytesDefault:
    def test_default_is_64kb(self):
        """Default tool_result_max_bytes is 64KB (65536)."""
        cfg = AgentConfig()
        assert cfg.tool_result_max_bytes == 65536

    def test_custom_value_accepted(self):
        """Custom tool_result_max_bytes is accepted."""
        cfg = AgentConfig(tool_result_max_bytes=1024)
        assert cfg.tool_result_max_bytes == 1024


# ---------------------------------------------------------------------------
# B. _execute_tool truncates oversized results
# ---------------------------------------------------------------------------

class TestExecuteToolTruncation:
    @pytest.mark.asyncio
    async def test_truncation_kicks_in_at_cap(self):
        """A tool result exactly at the cap is NOT truncated."""
        from sr2_spectre.agent import Agent

        cap = 100
        cfg = _make_config(tool_result_max_bytes=cap)

        # Register a tool that returns exactly cap bytes
        tool_output = "x" * cap

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        # Register a tool that returns known content
        agent.register_tool(
            "big_tool",
            "Returns big output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="big_tool", input={})
        result = await agent._execute_tool(block)

        # Should NOT be truncated — exactly at cap
        assert "truncated" not in result.content.lower()
        assert len(result.content) == cap

    @pytest.mark.asyncio
    async def test_truncation_kicks_in_above_cap(self):
        """A tool result exceeding the cap IS truncated."""
        from sr2_spectre.agent import Agent

        cap = 100
        cfg = _make_config(tool_result_max_bytes=cap)

        # Register a tool that returns content exceeding the cap
        tool_output = "x" * 200

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "big_tool",
            "Returns big output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="big_tool", input={})
        result = await agent._execute_tool(block)

        # Result content must be truncated
        assert "truncated" in result.content.lower()
        assert len(result.content) <= cap + 150  # cap + margin for the marker text

    @pytest.mark.asyncio
    async def test_truncation_preserves_prefix(self):
        """Truncated content preserves the beginning of the original output."""
        from sr2_spectre.agent import Agent

        cap = 50
        cfg = _make_config(tool_result_max_bytes=cap)

        # Content with meaningful prefix
        tool_output = "HEADER: important data at the start" + "x" * 500

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "big_tool",
            "Returns big output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="big_tool", input={})
        result = await agent._execute_tool(block)

        # The beginning of the content should be preserved
        assert "HEADER: important data at the start" in result.content

    @pytest.mark.asyncio
    async def test_truncation_marker_includes_original_size(self):
        """The truncation marker includes the original output size for debugging."""
        from sr2_spectre.agent import Agent

        cap = 50
        cfg = _make_config(tool_result_max_bytes=cap)

        tool_output = "x" * 1000

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "big_tool",
            "Returns big output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="big_tool", input={})
        result = await agent._execute_tool(block)

        # Marker should reference the original size
        assert "1000" in result.content or "1024" in result.content or "bytes" in result.content

    @pytest.mark.asyncio
    async def test_error_results_also_truncated(self):
        """Error results from tools are also subject to truncation."""
        from sr2_spectre.agent import Agent

        cap = 50
        cfg = _make_config(tool_result_max_bytes=cap)

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        # Register a tool that raises a huge exception
        def failing_tool():
            raise ValueError("x" * 1000)

        agent.register_tool(
            "fail_tool",
            "Fails with huge error",
            {},
            failing_tool,
        )

        block = ToolUseBlock(id="tu1", name="fail_tool", input={})
        result = await agent._execute_tool(block)

        assert result.is_error is True
        assert "truncated" in result.content.lower() or len(result.content) <= cap + 50

    @pytest.mark.asyncio
    async def test_small_results_unchanged(self):
        """Results well under the cap pass through unchanged."""
        from sr2_spectre.agent import Agent

        cfg = _make_config(tool_result_max_bytes=65536)

        tool_output = "small result"

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "small_tool",
            "Returns small output",
            {},
            lambda: tool_output,
        )

        block = ToolUseBlock(id="tu1", name="small_tool", input={})
        result = await agent._execute_tool(block)

        assert result.content == "small result"
        assert "truncated" not in result.content.lower()

    @pytest.mark.asyncio
    async def test_regression_oversized_tool_result_continues(self):
        """Regression test for spc-3 scenario: oversized tool result must NOT crash the run.

        Before this fix, a ~2.16M-token grep blob injected raw into the next
        SR2/LLM request caused exceed_context_size_error and crashed the run.
        With truncation, the tool returns a bounded result and the loop continues.
        """
        from sr2_spectre.agent import Agent

        # Simulate a very tight cap to reproduce the overflow scenario
        cap = 1000
        cfg = _make_config(tool_result_max_bytes=cap)

        # Simulate a massive tool result (like the grep blob from spc-3)
        massive_output = "x" * (10 * 1024 * 1024)  # 10MB

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            agent = Agent(config=cfg, session_id="s1")

        agent.register_tool(
            "mega_grep",
            "Returns massive output",
            {},
            lambda: massive_output,
        )

        block = ToolUseBlock(id="tu1", name="mega_grep", input={})
        result = await agent._execute_tool(block)

        # Must NOT raise — the run should continue
        assert result is not None
        # Result must be bounded
        assert len(result.content) <= cap + 200
        # Must have a truncation marker
        assert "truncated" in result.content.lower()


# ---------------------------------------------------------------------------
# G. The cap is measured in UTF-8 bytes, not characters (obsidian-ejjc)
# ---------------------------------------------------------------------------

_ORIGINAL_SIZE_RE = re.compile(r"original size:\s*(\d+)\s*bytes", re.IGNORECASE)


def _agent_with_tool(cap: int, fn):
    from sr2_spectre.agent import Agent

    cfg = _make_config(tool_result_max_bytes=cap)
    with patch("sr2_spectre.session.SR2") as MockSR2:
        MockSR2.return_value = MagicMock()
        agent = Agent(config=cfg, session_id="s1")
    agent.register_tool("mb_tool", "Returns multibyte output", {}, fn)
    return agent


async def _run(cap: int, fn) -> ToolResultBlock:
    agent = _agent_with_tool(cap, fn)
    return await agent._execute_tool(ToolUseBlock(id="tu1", name="mb_tool", input={}))


def _kept_prefix(result: str, original: str) -> str:
    """The leading part of ``result`` that is copied verbatim from ``original``."""
    n = 0
    limit = min(len(result), len(original))
    while n < limit and result[n] == original[n]:
        n += 1
    return result[:n]


def _reported_original_size(content: str) -> int:
    matches = _ORIGINAL_SIZE_RE.findall(content)
    assert len(matches) == 1, f"expected one 'original size: N bytes' in marker: {content[-300:]!r}"
    return int(matches[0])


def _assert_cut_is_mid_character(text: str, cap: int) -> None:
    """Precondition: a raw byte cut at ``cap`` splits a multibyte character."""
    with pytest.raises(UnicodeDecodeError):
        text.encode("utf-8")[:cap].decode("utf-8")


# Every case is chosen so that, at cap=100, the byte cut lands inside a
# multibyte character (checked by _assert_cut_is_mid_character in each test).
MULTIBYTE_CASES = [
    pytest.param("€" * 1000, id="3-byte-euro"),
    pytest.param("ab" + "€" * 1000, id="ascii-offset-then-3-byte"),
    pytest.param("a" + "é" * 1000, id="ascii-offset-then-2-byte"),
    pytest.param("a" + "🙂" * 1000, id="ascii-offset-then-4-byte"),
    pytest.param("xyz" + "€é🙂" * 500, id="mixed-widths"),
]


class TestByteBasedTruncation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("original", MULTIBYTE_CASES)
    async def test_multibyte_body_is_within_byte_cap(self, original):
        """AC1: kept body is <= max_bytes UTF-8 bytes, and the cut is not wasteful."""
        cap = 100
        _assert_cut_is_mid_character(original, cap)
        assert len(original.encode("utf-8")) > cap

        result = await _run(cap, lambda: original)

        assert "truncated" in result.content.lower()
        body = _kept_prefix(result.content, original)
        body_bytes = len(body.encode("utf-8"))
        assert body_bytes <= cap
        # Cutting on bytes loses at most one partial character (< 4 bytes).
        assert body_bytes > cap - 4

    @pytest.mark.asyncio
    @pytest.mark.parametrize("original", MULTIBYTE_CASES)
    async def test_whole_result_is_bounded_in_bytes(self, original):
        """AC1: the whole truncated result is the byte cap plus a bounded marker."""
        cap = 100
        _assert_cut_is_mid_character(original, cap)
        result = await _run(cap, lambda: original)
        assert len(result.content.encode("utf-8")) <= cap + 200

    @pytest.mark.asyncio
    @pytest.mark.parametrize("original", MULTIBYTE_CASES)
    async def test_cut_does_not_introduce_partial_characters(self, original):
        """AC2: no replacement character and no partial character at the cut."""
        cap = 100
        _assert_cut_is_mid_character(original, cap)
        result = await _run(cap, lambda: original)

        assert "\ufffd" not in result.content
        # Round-trips cleanly as UTF-8 (no lone surrogates or broken sequences).
        assert result.content.encode("utf-8").decode("utf-8") == result.content
        body = _kept_prefix(result.content, original)
        # The body is a whole-character prefix of the original output.
        assert original.startswith(body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("original", MULTIBYTE_CASES)
    async def test_marker_reports_original_utf8_byte_size(self, original):
        """AC3: the marker's original size equals the UTF-8 byte count."""
        cap = 100
        _assert_cut_is_mid_character(original, cap)
        result = await _run(cap, lambda: original)
        assert _reported_original_size(result.content) == len(original.encode("utf-8"))

    @pytest.mark.asyncio
    async def test_bead_example_70k_euros(self):
        """AC1/AC3: the bead's example — 70,000 x '€' under the default 64 KiB cap."""
        original = "€" * 70_000
        cap = AgentConfig().tool_result_max_bytes
        _assert_cut_is_mid_character(original, cap)
        result = await _run(cap, lambda: original)

        body = _kept_prefix(result.content, original)
        assert len(body.encode("utf-8")) <= cap
        assert len(result.content.encode("utf-8")) <= cap + 200
        assert _reported_original_size(result.content) == 210_000

    @pytest.mark.asyncio
    async def test_under_cap_in_chars_but_over_cap_in_bytes_is_truncated(self):
        """AC4: 50 chars of '€' is 150 bytes, so a 100-byte cap must truncate it."""
        cap = 100
        original = "€" * 50
        assert len(original) <= cap < len(original.encode("utf-8"))

        result = await _run(cap, lambda: original)

        assert result.content != original
        assert "truncated" in result.content.lower()
        assert len(_kept_prefix(result.content, original).encode("utf-8")) <= cap
        assert _reported_original_size(result.content) == 150

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "original",
        [
            pytest.param("€" * 33, id="99-bytes"),
            pytest.param("€" * 33 + "a", id="exactly-100-bytes"),
            pytest.param("🙂" * 25, id="exactly-100-bytes-emoji"),
            pytest.param("", id="empty"),
        ],
    )
    async def test_multibyte_at_or_under_byte_cap_is_verbatim(self, original):
        """AC4: content at or under the cap by bytes passes through unchanged."""
        cap = 100
        assert len(original.encode("utf-8")) <= cap
        result = await _run(cap, lambda: original)
        assert result.content == original
        assert result.is_error is not True

    @pytest.mark.asyncio
    async def test_error_path_uses_byte_based_truncation(self):
        """AC5: a raising tool's multibyte error text is cut and reported in bytes."""
        message = "x" + "€" * 1000

        def failing():
            raise ValueError(message)

        # The untruncated error result, as the tool path renders it.
        full = (await _run(10 * 1024 * 1024, failing)).content
        assert "truncated" not in full.lower()

        cap = 100
        _assert_cut_is_mid_character(full, cap)
        result = await _run(cap, failing)

        assert result.is_error is True
        assert "truncated" in result.content.lower()
        assert "\ufffd" not in result.content
        assert result.content.encode("utf-8").decode("utf-8") == result.content
        body = _kept_prefix(result.content, full)
        assert len(body.encode("utf-8")) <= cap
        assert len(body.encode("utf-8")) > cap - 4
        assert _reported_original_size(result.content) == len(full.encode("utf-8"))

    @pytest.mark.asyncio
    async def test_error_under_chars_over_bytes_is_truncated(self):
        """AC4/AC5: an error under the cap in characters but over it in bytes is cut."""
        def failing():
            raise ValueError("x" + "€" * 40)

        full = (await _run(10 * 1024 * 1024, failing)).content
        cap = 100
        assert len(full) <= cap < len(full.encode("utf-8"))
        _assert_cut_is_mid_character(full, cap)

        result = await _run(cap, failing)

        assert result.is_error is True
        assert "truncated" in result.content.lower()
        assert "\ufffd" not in result.content
        assert len(_kept_prefix(result.content, full).encode("utf-8")) <= cap
        assert _reported_original_size(result.content) == len(full.encode("utf-8"))
