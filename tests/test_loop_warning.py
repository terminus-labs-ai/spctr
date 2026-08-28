"""Tests for agent.tool_loop_warnings (obsidian-6qb3).

Escalating tool-loop warnings: when a tool round completes and the
rounds-remaining before pipeline.max_tool_iterations matches a configured
key, that message is injected verbatim (with a system-reminder marker) onto
the tool-result feedback SR2 re-feeds to the next iteration — giving the
model runway to wrap up before the hard ToolLoopLimitError.

Covers the acceptance criteria:
  A. Config map parses and merges via the existing config system.
  B. Warning fires exactly once per configured threshold at the correct
     rounds-remaining.
  C. Injected text is verbatim from config.
  D. Feature off when map absent; existing ToolLoopLimitError behavior
     unchanged.
  E. Escalation order (higher thresholds fire first) + per-turn reset.

The injection mechanism relies on SR2 reusing the exact ToolResultBlock
objects it yields on ``tool_result_received`` in the next iteration's
CompletionRequest (verified in sr2.orchestrator.SR2.turn). The mock SR2
below simulates that by yielding round objects whose blocks are inspected
after the stream ends.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from sr2.config.models import ToolLoopLimitError
from sr2.models import ToolResultBlock, ToolUseBlock
from sr2.protocols.llm import StreamEvent
from sr2_spectre.config import AgentConfig, SpectreConfig, load_config, merge_configs
from sr2_spectre.session import _LOOP_WARNING_MARKER, Session
from sr2_spectre.tools.registry import ToolRegistry
from sr2_spectre.events import AgentDone, AgentToolResult

MARKER_TEXT = _LOOP_WARNING_MARKER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline_dict(max_tool_iterations: int = 25) -> dict:
    return {
        "layers": [
            {"name": "system", "target": "system",
             "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}]},
            {"name": "conversation", "target": "messages",
             "resolvers": [{"type": "session"}, {"type": "input"}]},
        ],
        "max_tool_iterations": max_tool_iterations,
    }


def _make_config(max_tool_iterations: int = 25, **agent_kwargs) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="test", **agent_kwargs),
        models={"default": {"model": "test-model", "base_url": "http://test:8000"}},
        pipeline=_pipeline_dict(max_tool_iterations),
    )


def _tool_use(tu_id: str) -> ToolUseBlock:
    return ToolUseBlock(id=tu_id, name="probe", input={})


def _tool_result(tu_id: str, content: str) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=tu_id, content=content)


def _tool_round(tu_id: str, content: str) -> list[StreamEvent]:
    """One tool round exactly as SR2's turn() yields it."""
    return [
        StreamEvent(type="tool_use_emitted", tool_uses=[_tool_use(tu_id)]),
        StreamEvent(type="tool_result_received",
                    tool_results=[_tool_result(tu_id, content)]),
        StreamEvent(type="iteration_complete", iteration=0),
    ]


def _final_round(text: str = "done") -> list[StreamEvent]:
    return [StreamEvent(type="text", text=text), StreamEvent(type="end")]


class _MockSR2:
    """Mock SR2 whose turn() yields fixed rounds.

    ``rounds`` is a list of lists of StreamEvents. Round objects are shared
    (not copied) so that post-run inspection of the ToolResultBlocks reflects
    in-place mutation — the same guarantee SR2's real turn() provides when it
    appends ``tool_result_blocks`` to the next CompletionRequest.
    """

    def __init__(self, rounds: list[list[StreamEvent]],
                 raise_loop_limit: bool = False) -> None:
        self.rounds = rounds
        self.raise_loop_limit = raise_loop_limit
        self.seed_session = MagicMock()

    async def turn(self, user_input: list) -> AsyncIterator[StreamEvent]:
        for round_events in self.rounds:
            for ev in round_events:
                yield ev
        if self.raise_loop_limit:
            raise ToolLoopLimitError("tool loop iteration limit reached")


def _make_session(mock_sr2: MagicMock, config: SpectreConfig) -> Session:
    session = Session(
        frame_id="f1",
        config=config,
        llm=None,
        registry=ToolRegistry(),
    )
    session.sr2 = mock_sr2
    return session


def _run_turn(session: Session, text: str = "go") -> list:
    import asyncio
    return asyncio.run(_drain(session, text))


async def _drain(session: Session, text: str) -> list:
    return [ev async for ev in session.stream_message(text)]


def _blocks_from_rounds(rounds: list) -> list[ToolResultBlock]:
    """The ToolResultBlock objects from each tool round (mock shares objects)."""
    out: list[ToolResultBlock] = []
    for round_events in rounds:
        for ev in round_events:
            if isinstance(ev, StreamEvent) and ev.type == "tool_result_received":
                out.extend(ev.tool_results)
    return out


# ---------------------------------------------------------------------------
# A. Config parsing / merging / coercion
# ---------------------------------------------------------------------------

class TestConfig:
    def test_absent_defaults_to_empty_map(self):
        cfg = _make_config()
        assert cfg.agent.tool_loop_warnings == {}

    def test_explicit_map_parses(self):
        cfg = _make_config(tool_loop_warnings={
            5: "prepare to wrap up",
            2: "wrap up now",
            1: "you will fail on the next round",
        })
        assert cfg.agent.tool_loop_warnings == {
            5: "prepare to wrap up",
            2: "wrap up now",
            1: "you will fail on the next round",
        }

    def test_empty_map_means_feature_off(self):
        cfg = _make_config(tool_loop_warnings={})
        assert cfg.agent.tool_loop_warnings == {}

    def test_string_integer_keys_coerced(self):
        """A hand-typed '5' key behaves identically to 5."""
        cfg = SpectreConfig(
            agent={"name": "t", "tool_loop_warnings": {"5": "warn at 5"}},
            models={"default": {"model": "m"}},
            pipeline=_pipeline_dict(),
        )
        assert cfg.agent.tool_loop_warnings == {5: "warn at 5"}

    def test_unparseable_keys_dropped_with_warning(self, caplog):
        import logging
        cfg = SpectreConfig(
            agent={"name": "t", "tool_loop_warnings": {"abc": "nope", 3: "keep"}},
            models={"default": {"model": "m"}},
            pipeline=_pipeline_dict(),
        )
        assert cfg.agent.tool_loop_warnings == {3: "keep"}
        assert any("abc" in rec.getMessage() for rec in caplog.records
                   if rec.levelno == logging.WARNING)

    def test_non_positive_keys_dropped(self, caplog):
        import logging
        cfg = SpectreConfig(
            agent={"name": "t", "tool_loop_warnings": {0: "zero", -1: "neg", 2: "keep"}},
            models={"default": {"model": "m"}},
            pipeline=_pipeline_dict(),
        )
        assert cfg.agent.tool_loop_warnings == {2: "keep"}
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("non-positive" in w or "always >= 1" in w for w in warnings)

    def test_yaml_load_via_load_config(self, tmp_path):
        """The map survives the real YAML load path (int keys from YAML)."""
        cfg_file = tmp_path / "agent.yaml"
        cfg_file.write_text(yaml.dump({
            "agent": {
                "name": "t",
                "tool_loop_warnings": {
                    5: "you have 5 rounds remaining, prepare to wrap up",
                    2: "wrap up and return now, you only have 2 rounds remaining",
                    1: "you will fail if you try one more round",
                },
            },
            "models": {"default": {"model": "m"}},
            "pipeline": _pipeline_dict(max_tool_iterations=6),
        }))
        cfg = load_config(str(cfg_file))
        assert cfg.agent.tool_loop_warnings == {
            5: "you have 5 rounds remaining, prepare to wrap up",
            2: "wrap up and return now, you only have 2 rounds remaining",
            1: "you will fail if you try one more round",
        }

    def test_tier_merge_adds_map(self):
        """Existing 4-tier merge: agent section deep-merges, map survives."""
        parent = {
            "agent": {"name": "t", "tool_loop_warnings": {5: "five"}},
            "models": {"default": {"model": "m"}},
            "pipeline": _pipeline_dict(),
        }
        child = {"agent": {"tool_loop_warnings": {2: "two"}}}
        merged = merge_configs(parent, child)
        assert merged["agent"]["tool_loop_warnings"] == {5: "five", 2: "two"}

        cfg = SpectreConfig(**merged)
        assert cfg.agent.tool_loop_warnings == {5: "five", 2: "two"}

    def test_tier_merge_child_can_override_value(self):
        parent = {"agent": {"tool_loop_warnings": {5: "old"}}}
        child = {"agent": {"tool_loop_warnings": {5: "new"}}}
        merged = merge_configs(parent, child)
        assert merged["agent"]["tool_loop_warnings"][5] == "new"


# ---------------------------------------------------------------------------
# B/C. Firing: threshold, verbatim text, exactly once
# ---------------------------------------------------------------------------

class TestWarningFiring:
    def test_fires_at_configured_threshold_with_verbatim_text(self):
        """max=6; after round 1, remaining=5 → the '5' message is injected verbatim."""
        warn5 = "you have 5 rounds remaining, prepare to wrap up"
        warn2 = "wrap up and return now, you only have 2 rounds remaining"
        cfg = _make_config(max_tool_iterations=6,
                           tool_loop_warnings={5: warn5, 2: warn2})

        # 4 tool rounds + final text.
        rounds = [
            _tool_round("tu1", "r1"),
            _tool_round("tu2", "r2"),
            _tool_round("tu3", "r3"),
            _tool_round("tu4", "r4"),
            _final_round(),
        ]
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        events = _run_turn(session)

        # AgentDone last, 4 tool rounds counted.
        assert isinstance(events[-1], AgentDone)
        assert events[-1].tool_calls_executed == 4

        results = _blocks_from_rounds(rounds)
        # Round 1 (remaining=5) carries the verbatim warning.
        assert warn5 in results[0].content
        assert MARKER_TEXT in results[0].content
        # Rounds 2,3,4 (remaining=4,3,2... wait: round 3 → remaining=3, round 4 → 2)
        # round2 remaining=4 → no key; round3 remaining=3 → no key;
        # round4 remaining=2 → '2' fires.
        assert warn2 not in results[1].content
        assert warn2 not in results[2].content
        assert warn2 in results[3].content

    def test_only_matching_keys_fire(self):
        """Keys that never match rounds-remaining never fire."""
        cfg = _make_config(max_tool_iterations=10, tool_loop_warnings={7: "seven"})
        # 2 rounds → remaining 9, 8 — 7 never reached.
        rounds = [_tool_round("tu1", "r1"), _tool_round("tu2", "r2"), _final_round()]
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        events = _run_turn(session)
        results = _blocks_from_rounds(rounds)
        assert all("seven" not in r.content for r in results)

    def test_each_key_fires_exactly_once_per_turn(self):
        """A configured key fires once even if the remaining value recurs."""
        cfg = _make_config(max_tool_iterations=6, tool_loop_warnings={5: "once"})
        # Round 1 → remaining=5 (fires).
        rounds = [_tool_round("tu1", "r1"), _final_round()]
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        _run_turn(session)
        # A second turn in the same session re-fires (per-turn, not global).
        mock2 = _MockSR2([_tool_round("tu2", "r2"), _final_round()])
        session.sr2 = mock2
        _run_turn(session)

        # First turn: fired exactly once (single tool round).
        assert "once" in mock.rounds[0][1].tool_results[0].content
        # Second turn: fired again (reset per turn).
        results2 = mock2.rounds[0][1].tool_results
        assert "once" in results2[0].content

    def test_threshold_at_one(self):
        """remaining==1 fires on the round just before the hard limit."""
        cfg = _make_config(max_tool_iterations=3, tool_loop_warnings={1: "last one"})
        rounds = [_tool_round("tu1", "r1"), _tool_round("tu2", "r2"), _final_round()]
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        _run_turn(session)
        r1, r2 = (rounds[0][1].tool_results[0], rounds[1][1].tool_results[0])
        assert "last one" not in r1.content          # remaining=2
        assert "last one" in r2.content              # remaining=1

    def test_multiple_warnings_do_not_leak_into_other_rounds(self):
        cfg = _make_config(max_tool_iterations=3,
                           tool_loop_warnings={2: "W2", 1: "W1"})
        rounds = [
            _tool_round("tu1", "r1"),
            _tool_round("tu2", "r2"),
            _tool_round("tu3", "r3"),
            _final_round(),
        ]
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        _run_turn(session)
        blocks = [rounds[i][1].tool_results[0] for i in range(3)]
        # remaining after rounds: 2, 1, 0
        assert "W2" in blocks[0].content and "W1" not in blocks[0].content
        assert "W1" in blocks[1].content and "W2" not in blocks[1].content
        assert "W2" not in blocks[2].content and "W1" not in blocks[2].content


# ---------------------------------------------------------------------------
# D. Off by default + ToolLoopLimitError behavior unchanged
# ---------------------------------------------------------------------------

class TestFeatureOffAndHardLimit:
    def test_feature_off_when_map_absent(self):
        """No map, no warnings, blocks untouched."""
        cfg = _make_config(max_tool_iterations=3)
        rounds = [
            _tool_round("tu1", "r1"),
            _tool_round("tu2", "r2"),
            _final_round(),
        ]
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        events = _run_turn(session)
        blocks = [rounds[i][1].tool_results[0] for i in range(2)]
        assert blocks[0].content == "r1"
        assert blocks[1].content == "r2"
        assert all(MARKER_TEXT not in r.content for r in blocks)

    def test_empty_map_is_off(self):
        cfg = _make_config(max_tool_iterations=3, tool_loop_warnings={})
        rounds = [_tool_round("tu1", "r1"), _final_round()]
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        _run_turn(session)
        assert rounds[0][1].tool_results[0].content == "r1"

    def test_tool_loop_limit_error_graceful_stop_unchanged(self):
        """spc-33 behavior: ToolLoopLimitError still caught, AgentDone last,
        notice emitted — with the feature ON."""
        warn = "wrap up now"
        cfg = _make_config(max_tool_iterations=6, tool_loop_warnings={5: warn})
        rounds = [_tool_round("tu1", "r1"), _tool_round("tu2", "r2")]
        mock = _MockSR2(rounds, raise_loop_limit=True)
        session = _make_session(mock, cfg)
        events = _run_turn(session)

        assert isinstance(events[-1], AgentDone)
        assert events[-1].tool_calls_executed == 2
        # Hard-limit notice still present (wording not pinned, per spc-33).
        texts = [ev.text for ev in events
                 if type(ev).__name__ == "AgentTextDelta"]
        assert any(("limit" in t) or ("iteration" in t) or ("stopped" in t)
                   for t in texts)
        # Soft warning also fired on round 1 (remaining=5).
        assert warn in rounds[0][1].tool_results[0].content

    def test_tool_loop_limit_error_unaffected_when_feature_off(self):
        cfg = _make_config(max_tool_iterations=2)
        rounds = [_tool_round("tu1", "r1"), _tool_round("tu2", "r2")]
        mock = _MockSR2(rounds, raise_loop_limit=True)
        session = _make_session(mock, cfg)
        events = _run_turn(session)
        assert isinstance(events[-1], AgentDone)
        assert events[-1].tool_calls_executed == 2
        assert rounds[0][1].tool_results[0].content == "r1"
        assert rounds[1][1].tool_results[0].content == "r2"

    def test_user_facing_events_carry_original_content(self):
        """The AgentToolResult event yielded to the user shows the original
        tool output, not the injected warning (warning rides in the blocks
        re-fed to SR2)."""
        cfg = _make_config(max_tool_iterations=2, tool_loop_warnings={1: "W"})
        rounds = [_tool_round("tu1", "original"), _final_round()]
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        events = _run_turn(session)
        tool_results = [ev for ev in events if isinstance(ev, AgentToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].content == "original"
        # ...but the block fed to the next iteration carries the warning.
        assert "W" in rounds[0][1].tool_results[0].content


# ---------------------------------------------------------------------------
# E. Escalation order
# ---------------------------------------------------------------------------

class TestEscalationOrder:
    def test_escalation_fires_in_decreasing_remaining_order(self):
        """5 → 2 → 1: higher thresholds fire on earlier rounds, verbatim."""
        cfg = _make_config(
            max_tool_iterations=6,
            tool_loop_warnings={
                5: "you have 5 rounds remaining, prepare to wrap up",
                2: "wrap up and return now, you only have 2 rounds remaining",
                1: "you will fail if you try one more round",
            },
        )
        # 5 tool rounds → remaining after each: 5, 4, 3, 2, 1
        rounds = [_tool_round(f"tu{i}", f"r{i}") for i in range(1, 6)]
        rounds.append(_final_round())
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        _run_turn(session)

        blocks = [rounds[i][1].tool_results[0] for i in range(5)]
        fired_order = []
        for block in blocks:
            for key, msg in sorted(cfg.agent.tool_loop_warnings.items(),
                                   reverse=True):
                if msg in block.content:
                    fired_order.append(key)
        # Warnings appear on rounds 1, 4, 5 → thresholds 5, 2, 1 in that order.
        assert fired_order == [5, 2, 1]

    def test_warning_text_not_rewritten(self):
        """Injected text is byte-for-byte the configured string (plus marker)."""
        msg = "wrap up and return now, you only have 2 rounds remaining"
        cfg = _make_config(max_tool_iterations=3, tool_loop_warnings={2: msg})
        rounds = [_tool_round("tu1", "orig"), _final_round()]
        mock = _MockSR2(rounds)
        session = _make_session(mock, cfg)
        _run_turn(session)
        content = rounds[0][1].tool_results[0].content
        assert msg in content
        assert content.endswith("orig")


# ---------------------------------------------------------------------------
# F. End-to-end through the real SR2 orchestrator
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Prove the injection reaches the LLM's next request via real SR2.

    This is the load-bearing mechanism: SR2 must reuse the ToolResultBlock
    objects Spectre mutates when it builds the next CompletionRequest.
    """

    @pytest.mark.asyncio
    async def test_warning_reaches_next_llm_request(self):
        from sr2.pipeline.token_counting import CharacterTokenCounter
        from sr2.orchestrator import SR2

        warn5 = "you have 5 rounds remaining, prepare to wrap up"
        cfg = _make_config(max_tool_iterations=6, tool_loop_warnings={5: warn5})

        captured: list[Any] = []

        class LLM:
            async def stream(self, request: Any) -> AsyncIterator[StreamEvent]:
                captured.append(request)
                n = len(captured)
                if n <= 3:
                    yield StreamEvent(
                        type="tool_use",
                        tool_use_id=f"call_{n:03d}",
                        tool_name="probe",
                        tool_input={},
                    )
                else:
                    yield StreamEvent(type="text", text="wrapped up")

        async def executor(block: ToolUseBlock) -> ToolResultBlock:
            return ToolResultBlock(tool_use_id=block.id, content=f"res{block.id}")

        session = _make_session(MagicMock(), cfg)
        sr2 = SR2(
            pipeline_config=cfg.pipeline,
            llm={"default": LLM()},
            token_counter=CharacterTokenCounter(),
            session_id="e2e",
            tool_executor=executor,
        )
        session.sr2 = sr2

        await _drain(session, "go")

        assert len(captured) == 4
        # First request: no tool results yet.
        assert "tool_result" not in str(captured[0].messages)
        # Second request: the round-1 result carries the injected warning.
        second = captured[1]
        result_blocks = [
            b for m in second.messages for b in m.content
            if getattr(b, "type", None) == "tool_result"
        ]
        assert len(result_blocks) == 1
        assert warn5 in result_blocks[0].content
        assert MARKER_TEXT in result_blocks[0].content
        # Third request: round-2 result has remaining=4 (no key) — untouched.
        third = captured[2]
        r3 = [
            b for m in third.messages for b in m.content
            if getattr(b, "type", None) == "tool_result"
        ]
        assert r3[-1].content == "rescall_002"
        # Fourth request: round-3 result has remaining=3 (no key) — untouched.
        fourth = captured[3]
        r4 = [
            b for m in fourth.messages for b in m.content
            if getattr(b, "type", None) == "tool_result"
        ]
        assert r4[-1].content == "rescall_003"
