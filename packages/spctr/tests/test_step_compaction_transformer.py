"""Tests for StepCompactionTransformer (obsidian-aal.2).

Covers acceptance criteria from specs/step-compaction.md (FR1-7):
  - Burns content blocks tagged with matching frame id on plan_step_completed
  - Inserts breadcrumb TextBlock per burned frame
  - No-op when no matching blocks (content=None)
  - No LLM call required (build works with no LLM in deps)
  - Fires only on plan_step_completed — other events produce no burn
  - Provenance entries: transformer-origin, non-empty sources, source_layer="step_compaction"
  - Idempotent: burning an already-burned frame is a clean no-op
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from sr2.config.models import TransformerConfig
from sr2.models import TextBlock, ToolUseBlock, ToolResultBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.models import TransformationResult
from sr2.pipeline.provenance import EntryOrigin

from sr2_spectre.planning.transformer import StepCompactionTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transformer() -> StepCompactionTransformer:
    """Build a StepCompactionTransformer with minimal config and no LLM."""
    config = TransformerConfig(
        type="step_compaction",
        config={},
        subscriptions=[
            {"event": "plan_step_completed", "phase": "completed"},
        ],
    )
    deps = Dependencies(
        llm=None,
        session_id="test-session",
        active_frame_provider=None,
    )
    return StepCompactionTransformer.build(config, deps)


def _make_block(
    text: str = "some content",
    frame: str | None = None,
    block_type: str = "text",
) -> TextBlock | ToolUseBlock | ToolResultBlock:
    """Create a content block with optional frame tag."""
    if block_type == "text":
        block = TextBlock(text=text)
    elif block_type == "tool_use":
        block = ToolUseBlock(id="tool-1", name="test_tool", input={})
    else:
        block = ToolResultBlock(tool_use_id="tool-1", content=text, is_error=False)

    if frame is not None:
        block.meta["frame"] = frame
    return block


def _make_completed_event(
    frame: str = "plan:test-plan/01-slug",
    plan: str = "test-plan",
    task: str = "01-slug",
    order: int = 1,
) -> Event:
    """Create a plan_step_completed event."""
    return Event(
        name="plan_step_completed",
        phase=EventPhase.COMPLETED,
        source_layer="plan",
        data={
            "frame": frame,
            "plan": plan,
            "task": task,
            "order": order,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuild:
    def test_build_without_llm(self) -> None:
        """StepCompactionTransformer does NOT require an LLM (no summarization)."""
        config = TransformerConfig(type="step_compaction", config={})
        deps = Dependencies(llm=None, session_id="test")
        transformer = StepCompactionTransformer.build(config, deps)
        assert transformer.name == "step_compaction"

    def test_build_has_subscription(self) -> None:
        """Transformer subscribes to plan_step_completed by default."""
        config = TransformerConfig(
            type="step_compaction",
            config={},
            subscriptions=[{"event": "plan_step_completed", "phase": "completed"}],
        )
        deps = Dependencies(llm=None, session_id="test")
        transformer = StepCompactionTransformer.build(config, deps)
        event_names = [sub.event_name for sub in transformer.subscriptions]
        assert "plan_step_completed" in event_names


class TestBurn:
    async def test_burns_matching_blocks(self) -> None:
        """Blocks tagged with the event's frame id are removed from content."""
        transformer = _make_transformer()

        content = [
            _make_block("system prompt", frame=None),  # No frame — kept
            _make_block("task 01 tool use", frame="plan:p/01-task"),  # Burned
            _make_block("task 01 tool result", frame="plan:p/01-task"),  # Burned
            _make_block("task 02 tool use", frame="plan:p/02-task"),  # Different frame — kept
        ]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is not None
        remaining_texts = [b.text for b in result.content if isinstance(b, TextBlock)]

        # System prompt kept
        assert "system prompt" in remaining_texts
        # Task 01 blocks burned
        assert "task 01 tool use" not in remaining_texts
        assert "task 01 tool result" not in remaining_texts
        # Task 02 kept
        assert "task 02 tool use" in remaining_texts

    async def test_inserts_breadcrumb(self) -> None:
        """A breadcrumb TextBlock replaces the burned span."""
        transformer = _make_transformer()

        content = [
            _make_block("before", frame=None),
            _make_block("burned 1", frame="plan:p/01-task"),
            _make_block("burned 2", frame="plan:p/01-task"),
            _make_block("after", frame=None),
        ]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is not None
        texts = [b.text for b in result.content if isinstance(b, TextBlock)]

        breadcrumb = "[task 01-task completed — see plan]"
        assert breadcrumb in texts
        assert "before" in texts
        assert "after" in texts

    async def test_preserves_block_order(self) -> None:
        """Non-burned blocks maintain their original relative order."""
        transformer = _make_transformer()

        content = [
            _make_block("first", frame=None),
            _make_block("burned", frame="plan:p/01-task"),
            _make_block("second", frame="plan:p/02-task"),
            _make_block("burned2", frame="plan:p/01-task"),
            _make_block("third", frame=None),
        ]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is not None
        # Filter out breadcrumb to check order of original blocks
        original_texts = [
            b.text for b in result.content
            if isinstance(b, TextBlock) and "completed" not in b.text
        ]
        assert original_texts == ["first", "second", "third"]


class TestNoOp:
    async def test_no_matching_blocks_is_noop(self) -> None:
        """When no blocks match the frame id, return content=None (no-op)."""
        transformer = _make_transformer()

        content = [
            _make_block("block 1", frame="plan:p/02-task"),
            _make_block("block 2", frame=None),
        ]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is None
        assert result.events == []
        assert result.entries == []

    async def test_empty_content_is_noop(self) -> None:
        """Empty content list returns no-op."""
        transformer = _make_transformer()

        result = await transformer.transform(
            [],
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is None


class TestEventFiltering:
    async def test_ignores_non_matching_events(self) -> None:
        """Only fires on plan_step_completed — other events produce no burn."""
        transformer = _make_transformer()

        content = [
            _make_block("block", frame="plan:p/01-task"),
        ]

        result = await transformer.transform(
            content,
            [Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")],
        )

        # No burn — block should still be there (no-op)
        assert result.content is None

    async def test_ignores_empty_events_list(self) -> None:
        """No events means no burn."""
        transformer = _make_transformer()

        content = [_make_block("block", frame="plan:p/01-task")]

        result = await transformer.transform(content, [])

        assert result.content is None


class TestProvenance:
    async def test_entries_are_transformer_origin(self) -> None:
        """Burn entries are transformer-origin with source_layer='step_compaction'."""
        transformer = _make_transformer()

        content = [
            _make_block("burned", frame="plan:p/01-task"),
        ]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is not None
        assert len(result.entries) == 1

        entry = result.entries[0]
        assert entry.origin.kind == "transformer"
        assert entry.origin.name == "step_compaction"
        assert entry.layer == "step_compaction"

    async def test_entries_have_non_empty_sources(self) -> None:
        """Transformer-origin entries must have non-empty sources."""
        transformer = _make_transformer()

        content = [
            _make_block("burned 1", frame="plan:p/01-task"),
            _make_block("burned 2", frame="plan:p/01-task"),
        ]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is not None
        assert len(result.entries) == 1

        entry = result.entries[0]
        assert len(entry.sources) > 0


class TestIdempotency:
    async def test_burning_same_frame_twice_is_noop_second_time(self) -> None:
        """Second burn of the same frame (blocks already removed) is a clean no-op."""
        transformer = _make_transformer()

        content = [
            _make_block("block", frame="plan:p/01-task"),
        ]

        # First burn: blocks removed
        result1 = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )
        assert result1.content is not None
        texts1 = [b.text for b in result1.content if isinstance(b, TextBlock)]
        assert "block" not in texts1

        # Second burn: nothing to burn (blocks already gone)
        result2 = await transformer.transform(
            result1.content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )
        assert result2.content is None  # no-op — no matching blocks


class TestMultipleFrames:
    async def test_single_event_burns_one_frame(self) -> None:
        """A single plan_step_completed event burns only that frame's blocks."""
        transformer = _make_transformer()

        content = [
            _make_block("frame a", frame="plan:p/01-task"),
            _make_block("frame b", frame="plan:p/02-task"),
            _make_block("frame a 2", frame="plan:p/01-task"),
        ]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is not None
        texts = [b.text for b in result.content if isinstance(b, TextBlock)]

        assert "frame a" not in texts
        assert "frame a 2" not in texts
        assert "frame b" in texts  # Different frame preserved

    async def test_multiple_completed_events_in_one_call(self) -> None:
        """Multiple plan_step_completed events in one transform call each burn their frame."""
        transformer = _make_transformer()

        content = [
            _make_block("frame a", frame="plan:p/01-task"),
            _make_block("frame b", frame="plan:p/02-task"),
            _make_block("frame c", frame=None),
        ]

        events = [
            _make_completed_event(frame="plan:p/01-task", task="01-task"),
            _make_completed_event(frame="plan:p/02-task", task="02-task"),
        ]

        result = await transformer.transform(content, events)

        assert result.content is not None
        texts = [b.text for b in result.content if isinstance(b, TextBlock)]

        assert "frame a" not in texts
        assert "frame b" not in texts
        assert "frame c" in texts  # Untagged block preserved

        # Should have two breadcrumbs
        breadcrumbs = [b for b in result.content if isinstance(b, TextBlock) and "completed" in b.text]
        assert len(breadcrumbs) == 2

    async def test_breadcrumb_position(self) -> None:
        """Breadcrumb is inserted at the position of the first burned block in each group."""
        transformer = _make_transformer()

        content = [
            _make_block("before", frame=None),
            _make_block("burned", frame="plan:p/01-task"),
            _make_block("middle", frame="plan:p/02-task"),
            _make_block("burned2", frame="plan:p/01-task"),
            _make_block("after", frame=None),
        ]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is not None
        texts = [b.text for b in result.content if isinstance(b, TextBlock)]
        breadcrumb = "[task 01-task completed — see plan]"

        # Breadcrumb should appear between "before" and "middle"
        breadcrumb_idx = texts.index(breadcrumb)
        assert texts[breadcrumb_idx - 1] == "before"
        assert texts[breadcrumb_idx + 1] == "middle"


class TestNonTextBlocks:
    async def test_burns_non_text_blocks(self) -> None:
        """ToolUseBlock and ToolResultBlock with matching frame are burned."""
        transformer = _make_transformer()

        content: list[TextBlock | ToolUseBlock | ToolResultBlock] = [
            _make_block(frame="plan:p/01-task", block_type="tool_use"),
            _make_block("result", frame="plan:p/01-task", block_type="tool_result"),
            _make_block("kept", frame=None),
        ]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.content is not None
        # Only the untagged text block should remain (plus breadcrumb)
        non_breadcrumb = [
            b for b in result.content
            if not (isinstance(b, TextBlock) and "completed" in b.text)
        ]
        assert len(non_breadcrumb) == 1
        assert isinstance(non_breadcrumb[0], TextBlock)
        assert non_breadcrumb[0].text == "kept"


class TestMalformedEvents:
    async def test_event_without_frame_data_is_noop(self) -> None:
        """Malformed event (missing frame key) is a logged no-op, not a crash."""
        transformer = _make_transformer()

        content = [_make_block("block", frame="plan:p/01-task")]

        event = Event(
            name="plan_step_completed",
            phase=EventPhase.COMPLETED,
            source_layer="plan",
            data={"plan": "test", "task": "01"},  # No "frame" key
        )

        result = await transformer.transform(content, [event])

        # Should be a no-op — no crash
        assert result.content is None


class TestSourceLayer:
    async def test_result_has_correct_source_layer(self) -> None:
        """TransformationResult uses source_layer='step_compaction'."""
        transformer = _make_transformer()

        content = [_make_block("burned", frame="plan:p/01-task")]

        result = await transformer.transform(
            content,
            [_make_completed_event(frame="plan:p/01-task", task="01-task")],
        )

        assert result.source_layer == "step_compaction"
        assert result.transformer_name == "step_compaction"
