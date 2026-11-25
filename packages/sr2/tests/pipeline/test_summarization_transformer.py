"""Tests for SummarizationTransformer.

Acceptance Criteria covered:
  AC3:  keep_last_n=3 with 10 turns → [TextBlock(summary), turn8, turn9, turn10]
  AC4:  fewer turns than keep_last_n → no-op (content=None, events=[], entries=[], no LLM call)
  AC5:  emits summarization_complete event with event.data = [TextBlock(summary_text)]
  AC6:  (sr2-127) the transformer emits no entries of its own: no fabricated
        source ids; the source-lineage omission is declared in provenance
  AC9:  model key in config is read by the transformer (routing tested at orchestrator layer)
  AC10: keep_within_tokens keeps only most-recent turns fitting within token budget
  AC11: two summarizations in same session → two LLM calls, two events, two
        results each carrying their own declared provenance

FR3:  SummarizationTransformer implements the Transformer protocol.
FR4:  keep_last_n and keep_within_tokens strategies.
FR5:  LLM is injected at construction (mock in tests).

Non-functional:
  - No-op safety: never calls LLM when content <= keep threshold.
  - LLM failure propagates as-is.
  - sr2-127 (spec FR5, "Synthetic dangling source IDs are invalid"): the
    transformer mints no ids at all. It returns no entries; the Layer
    synthesizes the summary block against real stored entry ids.

Note on AC9 (model routing): CompletionRequest has no `model` field — the LLM client
itself handles model selection. The transformer reads `config["model"]` to select which
LLM callable to use (injected by orchestrator). End-to-end model routing is tested at
the orchestrator integration level, not here.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from sr2.models import TextBlock, TokenUsage
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent, TransformationResult
from sr2.pipeline.protocols import Transformer
from sr2.pipeline.provenance import Entry, EntryOrigin, InMemoryProvenanceStore
from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore
from sr2.protocols.llm import CompletionRequest, CompletionResponse


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    def __init__(self, response_text: str = "This is a summary."):
        self.response_text = response_text
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return CompletionResponse(
            id="mock",
            content=[TextBlock(text=self.response_text)],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest):
        # not used by transformer
        return
        yield  # make it a generator


class FailingLLM:
    """LLM that always raises on complete()."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise RuntimeError("LLM backend unavailable")

    async def stream(self, request: CompletionRequest):
        return
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_text_blocks(n: int, prefix: str = "turn") -> list[TextBlock]:
    """Return n TextBlocks with distinguishable text."""
    return [TextBlock(text=f"{prefix}_{i}") for i in range(n)]


def make_config(
    *,
    keep_strategy: str = "keep_last_n",
    keep_last_n: int = 3,
    keep_tokens: int | None = None,
    subscriptions: list[dict] | None = None,
    model: str | None = None,
    max_executions: int = 10,
) -> object:
    """Build a TransformerConfig for SummarizationTransformer.

    ``model`` defaults to None (no model key in config), which means build()
    will use deps.llm["default"].  Pass an explicit model string only when the
    test specifically exercises named-key resolution.
    """
    from sr2.config.models import EventSubscriptionConfig, TransformerConfig

    sub_configs = subscriptions or [{"event": "turn_start"}]
    subs = [EventSubscriptionConfig(**s) for s in sub_configs]
    inner: dict = {
        "keep_strategy": keep_strategy,
        "keep_last_n": keep_last_n,
    }
    if model is not None:
        inner["model"] = model
    if keep_tokens is not None:
        inner["keep_tokens"] = keep_tokens
    return TransformerConfig(
        type="summarization",
        subscriptions=subs,
        config=inner,
        max_executions=max_executions,
    )


def make_events() -> list[Event]:
    return [Event(name="turn_start", phase=EventPhase.COMPLETED, source_layer="engine")]


def build_transformer(config=None, llm=None):
    """Convenience factory that imports the real implementation."""
    from sr2.pipeline.transformers.summarization import SummarizationTransformer

    cfg = config or make_config()
    lm = llm or MockLLM()
    return SummarizationTransformer(config=cfg, llm=lm)


# ---------------------------------------------------------------------------
# FR3: Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """FR3: SummarizationTransformer satisfies the Transformer protocol."""

    def test_fr3_implements_transformer_protocol(self):
        """FR3: isinstance check against the runtime-checkable Transformer protocol."""
        transformer = build_transformer()
        assert isinstance(transformer, Transformer)

    def test_fr3_has_subscriptions_attribute(self):
        """FR3: .subscriptions is a list of EventSubscription."""
        transformer = build_transformer()
        assert isinstance(transformer.subscriptions, list)
        assert all(isinstance(s, EventSubscription) for s in transformer.subscriptions)

    def test_fr3_has_max_executions_attribute(self):
        """FR3: .max_executions is an int."""
        transformer = build_transformer()
        assert isinstance(transformer.max_executions, int)

    def test_fr3_subscriptions_configurable_via_yaml(self):
        """FR3: subscriptions come from TransformerConfig.subscriptions."""
        config = make_config(subscriptions=[{"event": "memory_updated"}, {"event": "turn_start"}])
        transformer = build_transformer(config=config)
        names = [s.event_name for s in transformer.subscriptions]
        assert "memory_updated" in names
        assert "turn_start" in names

    def test_fr3_transform_is_coroutine(self):
        """FR3: transform() is a coroutine function (async def)."""
        import asyncio

        transformer = build_transformer()
        coro = transformer.transform([], [])
        assert asyncio.iscoroutine(coro)
        coro.close()  # clean up without running


# ---------------------------------------------------------------------------
# AC4: No-op when content <= keep threshold
# ---------------------------------------------------------------------------


class TestNoOp:
    """AC4: When content count <= keep_last_n, return no-op result without calling LLM."""

    @pytest.mark.asyncio
    async def test_ac4_empty_content_is_noop(self):
        """AC4: Empty content list → no-op result, no LLM call."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)

        result = await transformer.transform([], make_events())

        assert isinstance(result, TransformationResult)
        assert result.content is None
        assert result.entries == []
        assert result.events == [] or result.events is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_ac4_exactly_keep_last_n_is_noop(self):
        """AC4: Exactly keep_last_n turns → no-op, no LLM call."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(3)

        result = await transformer.transform(content, make_events())

        assert result.content is None
        assert result.entries == []
        assert result.events == [] or result.events is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_ac4_fewer_than_keep_last_n_is_noop(self):
        """AC4: Fewer turns than keep_last_n → no-op, no LLM call."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=5), llm=llm)
        content = make_text_blocks(2)

        result = await transformer.transform(content, make_events())

        assert result.content is None
        assert result.entries == []
        assert result.events == [] or result.events is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_ac4_noop_result_has_correct_transformer_name(self):
        """AC4: No-op result still carries transformer_name."""
        transformer = build_transformer()
        result = await transformer.transform([], make_events())

        assert isinstance(result.transformer_name, str)
        assert result.transformer_name != ""

    @pytest.mark.asyncio
    async def test_ac4_keep_last_n_one_turn_is_noop(self):
        """AC4: keep_last_n=1 with exactly 1 turn is no-op."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=1), llm=llm)
        content = make_text_blocks(1)

        result = await transformer.transform(content, make_events())

        assert result.content is None
        assert llm.calls == []


# ---------------------------------------------------------------------------
# AC3: keep_last_n summarization
# ---------------------------------------------------------------------------


class TestKeepLastN:
    """AC3: keep_last_n=3 with 10 turns → [TextBlock(summary), turn7, turn8, turn9]."""

    @pytest.mark.asyncio
    async def test_ac3_basic_keep_last_n(self):
        """AC3: 10 turns, keep_last_n=3 → summary block + last 3 turns."""
        llm = MockLLM(response_text="Summary of turns 0-6.")
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert len(result.content) == 4  # 1 summary + 3 kept
        summary_block = result.content[0]
        assert isinstance(summary_block, TextBlock)
        assert summary_block.text == "Summary of turns 0-6."
        # Last 3 turns preserved
        assert result.content[1] is content[7]
        assert result.content[2] is content[8]
        assert result.content[3] is content[9]

    @pytest.mark.asyncio
    async def test_ac3_llm_was_called_once(self):
        """AC3: LLM called exactly once with the turns to summarize."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        await transformer.transform(content, make_events())

        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_ac3_llm_called_with_7_turns_to_summarize(self):
        """AC3: LLM prompt contains content from the 7 turns being summarized."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        await transformer.transform(content, make_events())

        request = llm.calls[0]
        # Extract all text from messages and system prompt to check turns are included
        all_text_parts: list[str] = []
        if request.system:
            all_text_parts.extend(b.text for b in request.system)
        for msg in request.messages:
            all_text_parts.extend(
                block.text for block in msg.content if isinstance(block, TextBlock)
            )
        all_text = " ".join(all_text_parts)
        for i in range(7):
            assert f"turn_{i}" in all_text

    @pytest.mark.asyncio
    async def test_ac3_one_more_than_keep_last_n_triggers_summarization(self):
        """AC3 boundary: keep_last_n+1 turns → summarize 1, keep last keep_last_n."""
        llm = MockLLM(response_text="Brief summary.")
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(4)  # 1 to summarize, 3 to keep

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert len(result.content) == 4  # 1 summary + 3 kept
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Brief summary."
        assert result.content[1] is content[1]
        assert result.content[2] is content[2]
        assert result.content[3] is content[3]

    @pytest.mark.asyncio
    async def test_ac3_keep_last_n_1_summarizes_all_but_last(self):
        """AC3: keep_last_n=1 with 5 turns → summarize 4, keep last 1."""
        llm = MockLLM(response_text="All-but-last summary.")
        transformer = build_transformer(config=make_config(keep_last_n=1), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert len(result.content) == 2  # 1 summary + 1 kept
        assert result.content[0].text == "All-but-last summary."
        assert result.content[1] is content[4]


# ---------------------------------------------------------------------------
# AC5: summarization_complete event
# ---------------------------------------------------------------------------


class TestSummarizationCompleteEvent:
    """AC5: Transform emits summarization_complete event with data=[TextBlock(summary)]."""

    @pytest.mark.asyncio
    async def test_ac5_emits_summarization_complete(self):
        """AC5: result.events contains a summarization_complete event."""
        llm = MockLLM(response_text="The summary text.")
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        assert result.events is not None
        assert len(result.events) >= 1
        event = next((e for e in result.events if e.name == "summarization_complete"), None)
        assert event is not None, "Expected a 'summarization_complete' event"

    @pytest.mark.asyncio
    async def test_ac5_event_phase_is_completed(self):
        """AC5: summarization_complete event has phase=COMPLETED."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        event = next(e for e in result.events if e.name == "summarization_complete")
        assert event.phase == EventPhase.COMPLETED

    @pytest.mark.asyncio
    async def test_ac5_event_data_is_list_with_text_block(self):
        """AC5: event.data is a list containing one TextBlock with the summary text."""
        summary_text = "Precise summary output."
        llm = MockLLM(response_text=summary_text)
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        event = next(e for e in result.events if e.name == "summarization_complete")
        assert isinstance(event.data, list)
        assert len(event.data) == 1
        assert isinstance(event.data[0], TextBlock)
        assert event.data[0].text == summary_text

    @pytest.mark.asyncio
    async def test_ac5_event_source_layer_is_nonempty(self):
        """AC5: event.source_layer is a non-empty string."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        event = next(e for e in result.events if e.name == "summarization_complete")
        assert isinstance(event.source_layer, str)
        assert len(event.source_layer) > 0

    @pytest.mark.asyncio
    async def test_ac5_no_op_emits_no_summarization_complete_event(self):
        """AC4/AC5: No-op path emits no summarization_complete event."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=5), llm=llm)
        content = make_text_blocks(3)  # fewer than keep_last_n

        result = await transformer.transform(content, make_events())

        events = result.events or []
        summarization_events = [e for e in events if e.name == "summarization_complete"]
        assert summarization_events == []


# ---------------------------------------------------------------------------
# AC6 (sr2-127): no fabricated lineage — omission declared explicitly
# ---------------------------------------------------------------------------


class TestProvenanceEntries:
    """AC6 (sr2-127, spec FR5): the transformer declares no source lineage of
    its own and records the omission in its provenance declaration.

    The pre-sr2-127 shape — one entry per summarized block carrying a
    freshly generated ``source_id`` — was a synthetic dangling source id and
    is gone. ``transform(content, events)`` receives content blocks with no
    entry identity, so the transformer emits no entries at all; the Layer
    synthesizes the compiled summary block against the real stored entry ids
    it holds at flush time (see test_r1_*/test_r2_* below).
    """

    @pytest.mark.asyncio
    async def test_ac6_no_entries_emitted(self):
        """sr2-127 / AC1: summarizing 7 turns produces zero transformer entries."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        assert result.entries == []

    @pytest.mark.asyncio
    async def test_ac6_transformer_mints_no_ids(self):
        """sr2-127 / AC1: no entry id (or source id) shape appears anywhere in the result

        except ids the test itself stored. The fabricated-ULID path had no
        observable id of its own beyond entries, so a result with zero
        entries and no declared sources carries no id the transformer made up.
        """
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        assert result.entries == []
        declaration = result.provenance[0]
        # The declaration states the omission rather than naming an id.
        assert "sources" not in declaration, (
            f"transformer declared sources it cannot resolve: {declaration!r}"
        )
        for block in result.content or []:
            assert isinstance(block, TextBlock)

    @pytest.mark.asyncio
    async def test_ac6_omission_declared_explicitly(self):
        """sr2-127 / AC1: the result's provenance records the omission explicitly."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        assert result.provenance is not None
        assert len(result.provenance) == 1
        declaration = result.provenance[0]
        assert declaration.get("sources_omitted") is True
        reason = declaration.get("sources_omitted_reason")
        assert isinstance(reason, str)
        assert reason.strip() != ""

    @pytest.mark.asyncio
    async def test_ac6_declaration_carries_no_causal_override(self):
        """sr2-127: the omission declaration changes no causal field the Layer derives.

        The Layer's _causal_for_item lets a declaration override reason /
        authority / lifecycle. The summarizer's declaration must not touch
        them (nor id / session_id), so the stored summary entry keeps the
        Layer's default transformer causal shape.
        """
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        declaration = result.provenance[0]
        for key in ("reason", "authority", "lifecycle", "id", "session_id"):
            assert key not in declaration, (
                f"summarization declaration must not override {key!r}: {declaration!r}"
            )

    @pytest.mark.asyncio
    async def test_ac6_noop_produces_no_entries(self):
        """AC4/AC6: No-op path returns empty entries list."""
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=5), llm=llm)
        content = make_text_blocks(3)

        result = await transformer.transform(content, make_events())

        assert result.entries == []


# ---------------------------------------------------------------------------
# AC10: keep_within_tokens strategy
# ---------------------------------------------------------------------------


class TestKeepWithinTokens:
    """AC10: keep_within_tokens keeps the most-recent turns fitting within keep_tokens."""

    @pytest.mark.asyncio
    async def test_ac10_basic_keep_within_tokens(self):
        """AC10: keep_within_tokens=20 with varied turn sizes — keeps most-recent fitting items."""
        # Each TextBlock "turn_N" is 6 chars = 1 token (6//4 = 1).
        # 20 tokens budget: we can fit 20 * 4 = 80 chars.
        # content items: 10 blocks each "turn_N" (6 chars) → each = 1 token.
        # 20 tokens means all 10 fit → no summarization (keep all).
        # Use tighter budget to force summarization.
        llm = MockLLM(response_text="Token-budget summary.")
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=4)
        transformer = build_transformer(config=config, llm=llm)
        # Each "turn_N" is 6 chars → 1 token each (6//4=1).
        # Budget of 4 tokens keeps up to 4 most-recent turns.
        content = make_text_blocks(8)  # 8 turns

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        # First block is the summary
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Token-budget summary."
        # "turn_N" = 6 chars, CharacterTokenCounter counts per-block: 6//4 = 1 token each.
        # Budget=4 tokens, 8 blocks × 1 token each → keep last 4, summarize first 4.
        kept = result.content[1:]
        assert len(kept) == 4
        for i, block in enumerate(kept):
            assert block is content[4 + i]

    @pytest.mark.asyncio
    async def test_ac10_all_fit_within_tokens_is_noop(self):
        """AC10: If all items fit within keep_tokens budget, no-op (no LLM call)."""
        llm = MockLLM()
        # 1000 tokens is more than enough for 5 short turns
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=1000)
        transformer = build_transformer(config=config, llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        assert result.content is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_ac10_zero_budget_keeps_nothing(self):
        """AC10: keep_tokens=0 → all items summarized, kept list is empty."""
        llm = MockLLM(response_text="All summarized.")
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=0)
        transformer = build_transformer(config=config, llm=llm)
        content = make_text_blocks(3)

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert len(result.content) == 1  # just the summary, no kept items
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "All summarized."

    @pytest.mark.asyncio
    async def test_ac10_kept_blocks_are_most_recent(self):
        """AC10: The kept blocks are always the most-recent (tail), not random."""
        llm = MockLLM(response_text="Older turns summary.")
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=8)
        transformer = build_transformer(config=config, llm=llm)
        # "turn_N" = 6 chars. CharacterTokenCounter counts per-block: 6//4 = 1 token each.
        # Budget=8 tokens, 10 blocks × 1 token each → keep last 8, summarize first 2.
        content = make_text_blocks(10)

        result = await transformer.transform(content, make_events())

        # Should keep turns 2..9 (last 8), summarize turns 0..1
        kept = result.content[1:]
        assert len(kept) == 8
        for i, block in enumerate(kept):
            assert block is content[2 + i]

    @pytest.mark.asyncio
    async def test_ac10_summarized_items_declare_no_fabricated_entries(self):
        """AC10/sr2-127: token-budget summarization emits no entries and declares the omission."""
        llm = MockLLM()
        config = make_config(keep_strategy="keep_within_tokens", keep_tokens=4)
        transformer = build_transformer(config=config, llm=llm)
        content = make_text_blocks(8)

        result = await transformer.transform(content, make_events())

        assert result.entries == []
        declaration = result.provenance[0]
        assert declaration.get("sources_omitted") is True
        assert "sources" not in declaration


# ---------------------------------------------------------------------------
# AC11: Two summarizations in same session
# ---------------------------------------------------------------------------


class TestTwoSummarizations:
    """AC11: Two calls each produce their own result and declared provenance.

    sr2-127: entry identity is no longer the transformer's business (the
    Layer content-addresses the stored summary), so this class asserts what
    the transformer still owns: one LLM call, one event and one provenance
    declaration per summarization.
    """

    @pytest.mark.asyncio
    async def test_ac11_two_summarizations_each_declare_their_own_provenance(self):
        """AC11: Two separate transform() calls each carry their own provenance declaration."""
        llm = MockLLM(response_text="Round summary.")
        config = make_config(keep_last_n=2, max_executions=10)
        transformer = build_transformer(config=config, llm=llm)

        content_a = make_text_blocks(5, prefix="alpha")
        content_b = make_text_blocks(5, prefix="beta")

        result_a = await transformer.transform(content_a, make_events())
        result_b = await transformer.transform(content_b, make_events())

        # Both results carry a provenance declaration; neither carries entries.
        assert result_a.entries == []
        assert result_b.entries == []
        assert result_a.provenance is not None and len(result_a.provenance) == 1
        assert result_b.provenance is not None and len(result_b.provenance) == 1
        # Distinct result objects — the declarations are not shared state.
        assert result_a.provenance[0] is not result_b.provenance[0]
        # Each summarization compiled its own summary block.
        assert result_a.content[0].text == "Round summary."
        assert result_b.content[0].text == "Round summary."
        assert result_a.content[0] is not result_b.content[0]

    @pytest.mark.asyncio
    async def test_ac11_two_summarizations_two_llm_calls(self):
        """AC11: Two summarizations → LLM called twice."""
        llm = MockLLM()
        config = make_config(keep_last_n=2, max_executions=10)
        transformer = build_transformer(config=config, llm=llm)

        content_a = make_text_blocks(5, prefix="alpha")
        content_b = make_text_blocks(5, prefix="beta")

        await transformer.transform(content_a, make_events())
        await transformer.transform(content_b, make_events())

        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_ac11_two_summarizations_two_events(self):
        """AC11: Each summarization emits its own summarization_complete event."""
        llm = MockLLM(response_text="Repeated summary.")
        config = make_config(keep_last_n=2, max_executions=10)
        transformer = build_transformer(config=config, llm=llm)

        content = make_text_blocks(5)

        result_a = await transformer.transform(content, make_events())
        result_b = await transformer.transform(content, make_events())

        events_a = [e for e in (result_a.events or []) if e.name == "summarization_complete"]
        events_b = [e for e in (result_b.events or []) if e.name == "summarization_complete"]

        assert len(events_a) == 1
        assert len(events_b) == 1


# ---------------------------------------------------------------------------
# sr2-127 AC1 seam (case r1/r2/r3): real Layer + real ProvenanceStore producer
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run ``coro`` to completion without touching the ambient event-loop policy.

    ``asyncio.run`` closes the loop and unsets it from the policy; other
    suites' sync tests later call ``asyncio.get_event_loop()`` and would
    break. A fresh loop that is closed but never set is safe.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_layer_with(transformer, provenance_store) -> "Layer":
    """Real sr2 Layer driving the transformer under test (sr2-127 seam)."""
    from sr2.pipeline.compilation import AppendStrategy
    from sr2.pipeline.layer import Layer
    from sr2.pipeline.models import CompilationTarget
    from sr2.pipeline.token_counting import CharacterTokenCounter

    return Layer(
        name="conversation",
        target=CompilationTarget.MESSAGES,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[transformer],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
        provenance_store=provenance_store,
        session_id="sr2-127-session",
    )


def rows_from_db(db_path):
    """Every ledger row read straight from the SQLite file.

    Raw rows only — no Entry reconstruction — so sweeping the ledger never
    trips the Entry invariants and cannot be defeated by a half-valid row.
    """
    import sqlite3

    with sqlite3.connect(str(db_path)) as db:
        rows = db.execute(
            "SELECT id, content_json, origin_kind, origin_name FROM entries"
        ).fetchall()
        links = db.execute(
            "SELECT entry_id, source_id FROM entry_sources"
        ).fetchall()
    return rows, links


def transformer_entry_ids_transformer_minted(result) -> list[str]:
    """Ids the transformer itself put on emitted entries (sr2-127: none)."""
    return [e.id for e in result.entries]


class TestRealStoreSeam:
    """sr2-127 AC1 (case r1/r2/r3): transform() run against content previously
    stored through the real Layer, so real entry ids exist in the provenance
    store, asserted against that same store.

    Both stores are the real production classes. ``Layer.process_pending``
    (the real producer of stored ids) populates them; the store-level FR5
    check is exercised through the stores' own validation paths — SQLite FK
    enforcement over the raw ledger, and ``InMemoryProvenanceStore.write``,
    which raises ValueError for a source id absent from the store.
    """

    def test_r1_sqlite_all_source_ids_resolve(self, tmp_path):
        """r1 / AC1 + FR5: every source id in the ledger resolves to a stored entry.

        Sweeps the whole SQLite ledger after a summarization turn. The
        fabricated-ULID path cannot survive this sweep: it emitted one entry
        per summarized block whose single source was a freshly generated id
        that no write ever stored.
        """
        db_path = tmp_path / "r1.db"

        async def _run():
            store = SQLiteProvenanceStore(path=db_path)
            await store.connect()
            try:
                llm = MockLLM(response_text="R1 summary.")
                transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
                layer = make_layer_with(transformer, store)

                # Store 10 turns through the real Layer so real entry ids exist.
                turns = make_text_blocks(10)
                layer.add_content(
                    ResolvedContent(
                        resolver_name="session", source_layer="conversation", content=turns
                    )
                )
                await layer.process_pending()

                # Fire summarization over that same, already-stored content.
                layer.handle_event(make_events()[0])
                await layer.process_pending()
            finally:
                await store.close()

        run_async(_run())

        rows, links = rows_from_db(db_path)
        stored_ids = {row[0] for row in rows}

        # The 10 summarized/kept blocks are stored entries.
        assert len(stored_ids) >= 10

        # FR5 at store level: no source link points at an id that is not a row.
        dangling = sorted({sid for _, sid in links if sid not in stored_ids})
        assert dangling == [], f"dangling source ids in the ledger: {dangling}"

        # FR1: the compiled summary block maps to exactly one stored entry, of
        # transformer origin, whose sources are all stored rows.
        def _row_text(row) -> str:
            return json.loads(row[1])["text"]

        summary_rows = [
            row for row in rows
            if _row_text(row) == "R1 summary."
        ]
        assert len(summary_rows) == 1, (
            f"the summary block must map to exactly one stored entry, got {len(summary_rows)}"
        )
        summary = summary_rows[0]
        assert summary[2] == "transformer", f"stored origin_kind: {summary[2]!r}"
        assert summary[3] == "summarization", f"stored origin_name: {summary[3]!r}"
        summary_sources = sorted(sid for eid, sid in links if eid == summary[0])
        assert summary_sources, "the summary entry must have stored source links"
        for source_id in summary_sources:
            assert source_id in stored_ids, (
                f"source id {source_id!r} is not a stored entry — dangling lineage"
            )

    def test_r1_declared_source_id_is_carried_when_it_resolves(self, tmp_path):
        """r1 / AC1: a summarized block whose real stored id is visible is chained to it.

        The summarized block is written to the real store *before* it joins
        the layer content — at summarize time it is stored content with a
        known id, the case-table run — and carries that id in its provenance
        meta. The stored summary entry's source links must be exactly that id,
        never an id the store has not seen.
        """
        real_id = "SR2127REALID0000000000000A"

        async def _drive(store):
            if isinstance(store, SQLiteProvenanceStore):
                await store.connect()
            try:
                # Stored first, through the store's own write path, so the id
                # is a real ledger row before summarize time.
                await store.write(
                    Entry(
                        id=real_id,
                        content=TextBlock(text="block being summarized"),
                        sources=(),
                        origin=EntryOrigin(kind="resolver", name="fixture"),
                        layer="conversation",
                        session_id="sr2-127-session",
                        created_at=datetime.now(tz=timezone.utc),
                    )
                )

                llm = MockLLM(response_text="Declared summary.")
                transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
                transformer._provenance_store = store
                layer = make_layer_with(transformer, store)

                # The summarized block carries its real stored id; the rest of
                # the content has no visible id, so the declared one is the
                # only source the transformer can legitimately name.
                annotated = TextBlock(
                    text="block being summarized",
                    meta={
                        "provenance": {
                            "source_ids": ["SR2127FAKEID0000000000000B", real_id]
                        }
                    },
                )
                filler = make_text_blocks(6, prefix="tail")
                layer.add_content(
                    ResolvedContent(
                        resolver_name="session",
                        source_layer="conversation",
                        content=[annotated, *filler],
                    )
                )
                await layer.process_pending()

                layer.handle_event(make_events()[0])
                await layer.process_pending()
            finally:
                if isinstance(store, SQLiteProvenanceStore):
                    await store.close()

        # InMemory store
        store = InMemoryProvenanceStore()
        run_async(_drive(store))

        async def _sources_in_memory():
            entries = []
            for sid in ("sr2-127-session", "default", ""):
                entries.extend(await store.get_session(sid))
            return entries

        entries = run_async(_sources_in_memory())
        summaries = [
            e for e in entries
            if isinstance(e.content, TextBlock) and e.content.text == "Declared summary."
        ]
        assert len(summaries) == 1, "summary block must map to a stored entry (FR1)"
        assert summaries[0].origin.kind == "transformer"
        assert summaries[0].sources == (real_id,), (
            "the summary entry must chain to the summarized block's real stored "
            f"id; got {summaries[0].sources!r}"
        )

        # SQLite store — asserted over raw ledger rows.
        db_path = tmp_path / "decl.db"
        run_async(_drive(SQLiteProvenanceStore(path=db_path)))
        rows, links = rows_from_db(db_path)
        stored_ids = {row[0] for row in rows}
        summary_rows = [
            row for row in rows
            if json.loads(row[1]).get("text") == "Declared summary."
        ]
        assert len(summary_rows) == 1, "summary block must map to a stored entry (FR1)"
        summary = summary_rows[0]
        assert summary[2] == "transformer"
        sources = sorted(sid for eid, sid in links if eid == summary[0])
        assert sources == [real_id], (
            "the summary entry's stored source links must be exactly the "
            f"block's real stored id; got {sources!r}"
        )
        assert all(sid in stored_ids for sid in sources)

    def test_r2_summary_lineage_reaches_original_turn_entries(self):
        """r2 / AC1 + FR1 + FR5: the summary block maps to a stored entry whose
        sources resolve, and lineage walks back to the original turn entries.

        Also exercises the store-level FR5 check: re-writing the summary entry
        through ``InMemoryProvenanceStore.write`` (which raises ValueError for
        a source id absent from the store) must not raise.
        """
        store = InMemoryProvenanceStore()

        async def _run():
            llm = MockLLM(response_text="R2 summary.")
            transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
            layer = make_layer_with(transformer, store)

            turns = make_text_blocks(10)
            layer.add_content(
                ResolvedContent(
                    resolver_name="session", source_layer="conversation", content=turns
                )
            )
            await layer.process_pending()
            before = {e.id for e in await store.get_session("sr2-127-session")}

            layer.handle_event(make_events()[0])
            await layer.process_pending()
            after = await store.get_session("sr2-127-session")
            new_entries = [e for e in after if e.id not in before]
            return new_entries

        new_entries = run_async(_run())

        # Exactly one new stored entry: the layer's raw-path synthesis of the
        # summary block (unchanged by this repair).
        assert len(new_entries) == 1
        summary = new_entries[0]
        assert isinstance(summary.content, TextBlock)
        assert summary.content.text == "R2 summary."
        assert summary.origin.kind == "transformer"
        assert summary.origin.name == "summarization"
        assert summary.sources

        # Store-level FR5 validation on the single-write path.
        async def _rewrite():
            await store.write(summary)

        run_async(_rewrite())

        # FR5: lineage traversal from the summary reaches the original turns.
        lineage = run_async(store.get_lineage(summary.id))
        lineage_by_id = {e.id: e for e in lineage}
        for source_id in summary.sources:
            assert source_id in lineage_by_id, (
                f"source {source_id!r} does not resolve in lineage traversal"
            )
        reached = {
            e.content.text
            for e in lineage_by_id.values()
            if isinstance(e.content, TextBlock)
        }
        for i in range(7):
            assert f"turn_{i}" in reached, (
                f"lineage from the summary must reach the original turn_{i} entry"
            )
        assert "R2 summary." in reached

    def test_r2_sources_fall_back_to_prior_run_stored_ids(self, tmp_path):
        """r2 / AC1: with no same-turn writes, sources chain to prior-run stored ids.

        The summary is compiled in a turn where nothing else is written, so
        the layer's raw-path synthesis chains to the previous turn's stored
        ids — still real, still stored, never generated.
        """
        db_path = tmp_path / "fallback.db"

        async def _run():
            store = SQLiteProvenanceStore(path=db_path)
            await store.connect()
            try:
                llm = MockLLM(response_text="Fallback summary.")
                transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
                layer = make_layer_with(transformer, store)

                turns = make_text_blocks(10)
                layer.add_content(
                    ResolvedContent(
                        resolver_name="session", source_layer="conversation", content=turns
                    )
                )
                await layer.process_pending()
                stored_ids = {
                    e.id for e in await store.get_session("sr2-127-session")
                }

                # Second turn: the transformer fires, but no new content is
                # added — only its output is compiled.
                layer.handle_event(make_events()[0])
                await layer.process_pending()
                return stored_ids
            finally:
                await store.close()

        stored_ids = run_async(_run())

        rows, links = rows_from_db(db_path)
        summary_rows = [
            row for row in rows
            if json.loads(row[1]).get("text") == "Fallback summary."
        ]
        assert len(summary_rows) == 1, "summary block must still map to a stored entry"
        summary = summary_rows[0]
        assert summary[2] == "transformer"
        sources = sorted(sid for eid, sid in links if eid == summary[0])
        assert sources, "the summary entry must have stored source links"
        for source_id in sources:
            assert source_id in stored_ids, (
                f"fallback source {source_id!r} is not a stored entry"
            )

    @pytest.mark.asyncio
    async def test_r3_noop_path_unchanged_through_real_layer(self, tmp_path):
        """r3 / AC1: content <= keep_last_n through the real Layer — no entries, no writes, no LLM."""
        store = InMemoryProvenanceStore()
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        layer = make_layer_with(transformer, store)

        turns = make_text_blocks(3)
        layer.add_content(
            ResolvedContent(
                resolver_name="session", source_layer="conversation", content=turns
            )
        )
        await layer.process_pending()
        before = {e.id for e in await store.get_session("sr2-127-session")}

        layer.handle_event(make_events()[0])
        changed = await layer.process_pending()

        after = {e.id for e in await store.get_session("sr2-127-session")}
        assert changed is False, "a no-op summarization turn generates no new events"
        assert after == before, "no-op summarization must write nothing new"
        assert llm.calls == [], "no-op summarization must not call the LLM"
        assert layer.get_content() == turns



# ---------------------------------------------------------------------------
# LLM failure propagation
# ---------------------------------------------------------------------------



class TestLLMFailurePropagation:
    """Non-functional: LLM failures propagate out of transform()."""

    @pytest.mark.asyncio
    async def test_llm_failure_propagates_as_exception(self):
        """If LLM raises, transform() raises the same exception."""
        failing_llm = FailingLLM()
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=failing_llm)
        content = make_text_blocks(5)

        with pytest.raises(RuntimeError, match="LLM backend unavailable"):
            await transformer.transform(content, make_events())

    @pytest.mark.asyncio
    async def test_llm_failure_not_suppressed(self):
        """LLM failure must not be silently caught and replaced with empty result."""
        failing_llm = FailingLLM()
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=failing_llm)
        content = make_text_blocks(5)

        raised = False
        try:
            await transformer.transform(content, make_events())
        except Exception:
            raised = True

        assert raised, "Expected an exception to propagate from LLM failure"


# ---------------------------------------------------------------------------
# Constructor and attribute tests
# ---------------------------------------------------------------------------


class TestConstructorAndAttributes:
    """Verify construction, attribute types, and injected LLM."""

    def test_constructor_accepts_config_and_llm(self):
        """SummarizationTransformer(config, llm) constructs without error."""
        transformer = build_transformer()
        assert transformer is not None

    def test_subscriptions_come_from_config(self):
        """Subscriptions are derived from TransformerConfig.subscriptions."""
        config = make_config(subscriptions=[{"event": "context_overflow"}])
        transformer = build_transformer(config=config)
        names = [s.event_name for s in transformer.subscriptions]
        assert "context_overflow" in names

    def test_max_executions_comes_from_config(self):
        """max_executions is taken from TransformerConfig.max_executions."""
        config = make_config(max_executions=7)
        transformer = build_transformer(config=config)
        assert transformer.max_executions == 7

    def test_execution_count_starts_at_zero(self):
        """execution_count is 0 before any transform calls."""
        transformer = build_transformer()
        assert transformer.execution_count == 0

    @pytest.mark.asyncio
    async def test_ac9_injected_llm_is_called_with_model_config(self):
        """AC9: The injected LLM is called when transform fires, regardless of model key.

        Orchestrator maps config["model"] → LLM callable and injects it here.
        At this layer, we verify the injected LLM is what gets called.
        """
        mock_llm = MockLLM(response_text="Model-config summary.")
        config = make_config(model="summarization", keep_last_n=2)
        transformer = build_transformer(config=config, llm=mock_llm)
        content = make_text_blocks(5)

        await transformer.transform(content, make_events())

        assert len(mock_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_execution_count_increments_on_actual_summarization(self):
        """execution_count is NOT incremented by the transformer itself on no-op; only on real runs.

        Note: The Layer increments execution_count after each call. However, the
        SummarizationTransformer (like PassiveTransformer) does NOT self-increment.
        This test verifies that execution_count stays 0 before Layer involvement,
        i.e., the transformer doesn't do it internally on no-op.
        """
        llm = MockLLM()
        transformer = build_transformer(config=make_config(keep_last_n=5), llm=llm)
        content = make_text_blocks(3)  # no-op path

        await transformer.transform(content, make_events())

        # Transformer should not self-increment
        assert transformer.execution_count == 0


# ---------------------------------------------------------------------------
# Result structure validation
# ---------------------------------------------------------------------------


class TestResultStructure:
    """Verify TransformationResult fields are correctly populated."""

    @pytest.mark.asyncio
    async def test_result_has_transformer_name(self):
        """TransformationResult.transformer_name is set."""
        transformer = build_transformer()
        content = make_text_blocks(10)
        result = await transformer.transform(content, make_events())
        assert isinstance(result.transformer_name, str)
        assert result.transformer_name != ""

    @pytest.mark.asyncio
    async def test_result_has_source_layer(self):
        """TransformationResult.source_layer is a string."""
        transformer = build_transformer()
        content = make_text_blocks(10)
        result = await transformer.transform(content, make_events())
        assert isinstance(result.source_layer, str)

    @pytest.mark.asyncio
    async def test_result_content_starts_with_summary_block(self):
        """When summarization fires, result.content[0] is the TextBlock from LLM."""
        summary = "Exactly this text."
        llm = MockLLM(response_text=summary)
        transformer = build_transformer(config=make_config(keep_last_n=3), llm=llm)
        content = make_text_blocks(5)

        result = await transformer.transform(content, make_events())

        assert result.content is not None
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == summary

    @pytest.mark.asyncio
    async def test_result_content_order_summary_then_kept(self):
        """result.content = [summary] + kept_items (summary first, then oldest-to-newest kept)."""
        llm = MockLLM(response_text="S")
        transformer = build_transformer(config=make_config(keep_last_n=2), llm=llm)
        content = make_text_blocks(4)  # summarize [0,1], keep [2,3]

        result = await transformer.transform(content, make_events())

        assert len(result.content) == 3
        assert result.content[0].text == "S"
        assert result.content[1] is content[2]
        assert result.content[2] is content[3]

    @pytest.mark.asyncio
    async def test_result_entries_is_list(self):
        """result.entries is always a list (not None)."""
        transformer = build_transformer()
        content = make_text_blocks(5)
        result = await transformer.transform(content, make_events())
        assert isinstance(result.entries, list)

    @pytest.mark.asyncio
    async def test_result_events_is_list_or_none(self):
        """result.events is a list (not some other type)."""
        transformer = build_transformer()
        content = make_text_blocks(10)
        result = await transformer.transform(content, make_events())
        assert isinstance(result.events, (list, type(None)))


# ---------------------------------------------------------------------------
# TestSummarizationTransformerBuild: build(cls, config, deps) classmethod
# ---------------------------------------------------------------------------


class TestSummarizationTransformerBuild:
    """FR1-FR10: SummarizationTransformer.build(cls, config, deps) classmethod.

    Covers:
    - FR1: build is a classmethod
    - FR2: build accepts (cls, config: TransformerConfig, deps: Dependencies) -> Self
    - FR3: deps.llm is None raises ConfigError mentioning 'summarize' or transformer type
    - FR4: no config["model"] → uses deps.llm["default"]
    - FR5: config["model"] present and key in deps.llm → uses that key
    - FR6: config["model"] present but key absent → falls back to deps.llm["default"]
    - FR7: config.config is None or empty → uses deps.llm["default"]
    - FR8: returns an instance of SummarizationTransformer
    - FR9: returned instance satisfies isinstance(result, Transformer)
    - FR10: existing __init__(config, llm) still works unchanged
    """

    # ------------------------------------------------------------------
    # Imports / fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def transformer_cls(self):
        from sr2.pipeline.transformers.summarization import SummarizationTransformer
        return SummarizationTransformer

    @pytest.fixture
    def default_llm(self):
        return MockLLM(response_text="default llm summary")

    @pytest.fixture
    def named_llm(self):
        return MockLLM(response_text="named llm summary")

    def make_deps(self, default=None, extras: dict | None = None):
        """Build a Dependencies with llm dict, or None."""
        from sr2.pipeline.dependencies import Dependencies

        if default is None and extras is None:
            return Dependencies(llm=None)
        llm_dict: dict = {}
        if default is not None:
            llm_dict["default"] = default
        if extras:
            llm_dict.update(extras)
        return Dependencies(llm=llm_dict)

    # ------------------------------------------------------------------
    # FR1: build is a classmethod
    # ------------------------------------------------------------------

    def test_fr1_build_is_classmethod(self, transformer_cls):
        """FR1: build is accessible as a classmethod on SummarizationTransformer."""
        import inspect
        assert isinstance(
            inspect.getattr_static(transformer_cls, "build"),
            classmethod,
        ), "build must be a classmethod"

    # ------------------------------------------------------------------
    # FR2: build accepts correct signature
    # ------------------------------------------------------------------

    def test_fr2_build_accepts_config_and_deps(self, transformer_cls, default_llm):
        """FR2: build(config, deps) is callable with TransformerConfig and Dependencies."""
        config = make_config()
        deps = self.make_deps(default=default_llm)
        # Should not raise
        result = transformer_cls.build(config, deps)
        assert result is not None

    # ------------------------------------------------------------------
    # FR3: deps.llm is None → ConfigError
    # ------------------------------------------------------------------

    def test_fr3_raises_config_error_when_llm_is_none(self, transformer_cls):
        """FR3: build raises ConfigError (not TypeError) when deps.llm is None."""
        from sr2.config.models import ConfigError
        from sr2.pipeline.dependencies import Dependencies

        config = make_config()
        deps = Dependencies(llm=None)

        with pytest.raises(ConfigError):
            transformer_cls.build(config, deps)

    def test_fr3_error_is_config_error_not_type_error(self, transformer_cls):
        """FR3: The exception type must be ConfigError, not TypeError or AttributeError."""
        from sr2.config.models import ConfigError
        from sr2.pipeline.dependencies import Dependencies

        config = make_config()
        deps = Dependencies(llm=None)

        exc = None
        try:
            transformer_cls.build(config, deps)
        except Exception as e:
            exc = e

        assert exc is not None, "Expected an exception"
        assert isinstance(exc, ConfigError), (
            f"Expected ConfigError, got {type(exc).__name__}: {exc}"
        )

    def test_fr3_error_message_mentions_summarize_or_type(self, transformer_cls):
        """FR3: ConfigError message mentions 'summarize' or the transformer type."""
        from sr2.config.models import ConfigError
        from sr2.pipeline.dependencies import Dependencies

        config = make_config()
        deps = Dependencies(llm=None)

        with pytest.raises(ConfigError) as exc_info:
            transformer_cls.build(config, deps)

        message = str(exc_info.value).lower()
        assert "summariz" in message or "summarization" in message or "llm" in message, (
            f"ConfigError message '{exc_info.value}' should mention the transformer type or llm requirement"
        )

    # ------------------------------------------------------------------
    # FR4: no config["model"] → uses deps.llm["default"]
    # ------------------------------------------------------------------

    def test_fr4_no_model_key_uses_default_llm(self, transformer_cls, default_llm):
        """FR4: When config.config has no 'model' key, uses deps.llm['default']."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        # Build config without a 'model' key in inner dict
        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"keep_strategy": "keep_last_n", "keep_last_n": 3},
            max_executions=10,
        )
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        # The injected LLM should be the default one
        assert result._llm is default_llm

    # ------------------------------------------------------------------
    # FR5: config["model"] present and key in deps.llm → uses that key
    # ------------------------------------------------------------------

    def test_fr5_named_model_key_uses_named_llm(self, transformer_cls, default_llm, named_llm):
        """FR5: config['model'] present and key exists in deps.llm → uses that LLM."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"keep_strategy": "keep_last_n", "keep_last_n": 3, "model": "haiku"},
            max_executions=10,
        )
        deps = self.make_deps(default=default_llm, extras={"haiku": named_llm})

        result = transformer_cls.build(config, deps)

        assert result._llm is named_llm

    def test_fr5_named_model_does_not_use_default(self, transformer_cls, default_llm, named_llm):
        """FR5: When named model key exists, default LLM is not selected."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"model": "opus"},
            max_executions=1,
        )
        deps = self.make_deps(default=default_llm, extras={"opus": named_llm})

        result = transformer_cls.build(config, deps)

        assert result._llm is not default_llm
        assert result._llm is named_llm

    # ------------------------------------------------------------------
    # FR6: config["model"] key absent from deps.llm → raises ConfigError
    # (sr2-14: silent fallback was the bug; explicit error is the fix)
    # ------------------------------------------------------------------

    def test_fr6_missing_model_key_raises_config_error(self, transformer_cls, default_llm):
        """FR6: config['model'] set but key absent from deps.llm → raises ConfigError."""
        from sr2.config.models import ConfigError, TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"model": "nonexistent-model"},
            max_executions=1,
        )
        deps = self.make_deps(default=default_llm)  # only "default" key, not "nonexistent-model"

        with pytest.raises((ConfigError, KeyError, ValueError)):
            transformer_cls.build(config, deps)

    def test_fr6_named_key_present_does_not_raise(self, transformer_cls, default_llm, named_llm):
        """FR6: When named key IS present in deps.llm, build() succeeds."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"model": "haiku"},
            max_executions=1,
        )
        deps = self.make_deps(default=default_llm, extras={"haiku": named_llm})

        result = transformer_cls.build(config, deps)
        assert result is not None
        assert result._llm is named_llm

    # ------------------------------------------------------------------
    # FR7: config.config is None or empty → uses deps.llm["default"]
    # ------------------------------------------------------------------

    def test_fr7_empty_config_dict_uses_default_llm(self, transformer_cls, default_llm):
        """FR7: config.config = {} (empty dict) → uses deps.llm['default']."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={},
            max_executions=1,
        )
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert result._llm is default_llm

    def test_fr7_default_config_dict_uses_default_llm(self, transformer_cls, default_llm):
        """FR7: TransformerConfig with default config (no explicit config arg) → uses default LLM."""
        from sr2.config.models import TransformerConfig

        # TransformerConfig.config defaults to {}
        config = TransformerConfig(type="summarization")
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert result._llm is default_llm

    # ------------------------------------------------------------------
    # FR8: returns instance of SummarizationTransformer
    # ------------------------------------------------------------------

    def test_fr8_returns_summarization_transformer_instance(self, transformer_cls, default_llm):
        """FR8: build() returns an instance of SummarizationTransformer."""
        config = make_config()
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert isinstance(result, transformer_cls)

    def test_fr8_returned_instance_has_correct_config(self, transformer_cls, default_llm):
        """FR8: The returned instance holds the provided config."""
        config = make_config(keep_last_n=7, max_executions=5)
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert result.max_executions == 5
        assert result._config is config

    # ------------------------------------------------------------------
    # FR9: returned instance satisfies Transformer protocol
    # ------------------------------------------------------------------

    def test_fr9_returned_instance_satisfies_transformer_protocol(
        self, transformer_cls, default_llm
    ):
        """FR9: isinstance(result, Transformer) is True."""
        from sr2.pipeline.protocols import Transformer

        config = make_config()
        deps = self.make_deps(default=default_llm)

        result = transformer_cls.build(config, deps)

        assert isinstance(result, Transformer)

    # ------------------------------------------------------------------
    # FR10: existing __init__(config, llm) unchanged — existing tests still work
    # ------------------------------------------------------------------

    def test_fr10_direct_init_still_works(self, transformer_cls):
        """FR10: SummarizationTransformer(config, llm) still constructs without error."""
        llm = MockLLM()
        config = make_config()
        # This is the existing construction path — must not be broken
        instance = transformer_cls(config=config, llm=llm)
        assert instance is not None
        assert isinstance(instance, transformer_cls)

    def test_fr10_direct_init_retains_correct_llm(self, transformer_cls):
        """FR10: Directly constructed instance has the injected LLM."""
        llm = MockLLM(response_text="direct init llm")
        config = make_config()
        instance = transformer_cls(config=config, llm=llm)
        assert instance._llm is llm

    @pytest.mark.asyncio
    async def test_fr10_direct_init_transform_still_works(self, transformer_cls):
        """FR10: Directly constructed instance can still call transform() correctly."""
        llm = MockLLM(response_text="still works")
        config = make_config(keep_last_n=2)
        instance = transformer_cls(config=config, llm=llm)
        content = make_text_blocks(5)

        result = await instance.transform(content, make_events())

        assert result.content is not None
        assert result.content[0].text == "still works"

    # ------------------------------------------------------------------
    # Integration: build() → transform() end-to-end
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_build_then_transform_uses_injected_llm(self, transformer_cls, default_llm):
        """Integration: build() then transform() calls the correct injected LLM."""
        config = make_config(keep_last_n=2)
        deps = self.make_deps(default=default_llm)

        instance = transformer_cls.build(config, deps)
        content = make_text_blocks(5)

        result = await instance.transform(content, make_events())

        assert result.content is not None
        assert result.content[0].text == "default llm summary"
        assert len(default_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_build_with_named_model_then_transform_uses_named_llm(
        self, transformer_cls, default_llm, named_llm
    ):
        """Integration: build() with named model key, then transform() uses named LLM."""
        from sr2.config.models import TransformerConfig, EventSubscriptionConfig

        subs = [EventSubscriptionConfig(event="turn_start")]
        config = TransformerConfig(
            type="summarization",
            subscriptions=subs,
            config={"keep_strategy": "keep_last_n", "keep_last_n": 2, "model": "sonnet"},
            max_executions=10,
        )
        deps = self.make_deps(default=default_llm, extras={"sonnet": named_llm})

        instance = transformer_cls.build(config, deps)
        content = make_text_blocks(5)

        result = await instance.transform(content, make_events())

        assert result.content is not None
        assert result.content[0].text == "named llm summary"
        assert len(named_llm.calls) == 1
        assert len(default_llm.calls) == 0


# ---------------------------------------------------------------------------
# spc-1119 AC7 / AC8: Message content (FR 9-11)
# ---------------------------------------------------------------------------

import logging

from sr2.models import Message, ToolResultBlock, ToolUseBlock


def _U(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


def _A_text(text: str) -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)])


def _A_tc(n: int) -> Message:
    # name "read" (4) + json {"path": "fN"} (14) = 18 chars -> 4 tokens
    return Message(
        role="assistant",
        content=[ToolUseBlock(id=f"tc{n}", name="read", input={"path": f"f{n}"})],
    )


def _R_tc(n: int, chars: int) -> Message:
    return Message(
        role="user",
        content=[ToolResultBlock(tool_use_id=f"tc{n}", content=f"r{n}" + "x" * (chars - 2))],
    )


def _msg_config(keep_tokens: int, **extra) -> object:
    cfg = make_config(keep_strategy="keep_within_tokens", keep_tokens=keep_tokens)
    cfg.config.update(extra)
    return cfg


def _r1_content() -> list[Message]:
    return [
        _U("P" * 400),
        _A_tc(1), _R_tc(1, 4000),
        _A_tc(2), _R_tc(2, 4000),
        _A_tc(3), _R_tc(3, 400),
    ]


class ScriptedLLM:
    """Fake LLM: returns `text`, `content`, or raises `exc`."""

    def __init__(self, text: str | None = "S", exc: Exception | None = None, content=None):
        self.text, self.exc, self.content = text, exc, content
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        if self.exc is not None:
            raise self.exc
        content = self.content if self.content is not None else [TextBlock(text=self.text)]
        return CompletionResponse(id="f", content=content, stop_reason="end_turn", usage=TokenUsage())


def _summary_text(msg) -> str:
    assert isinstance(msg, Message) and msg.role == "user"
    assert len(msg.content) == 1 and isinstance(msg.content[0], TextBlock)
    return msg.content[0].text


class TestMessageSummarization:
    @pytest.mark.asyncio
    async def test_ac7_r1_span_replaced_by_one_headed_summary(self):
        """AC7-r1: [U(prompt), U(header+S), A[tc3], R[tc3]], header count=4."""
        llm = ScriptedLLM("S")
        content = _r1_content()
        t = build_transformer(config=_msg_config(110), llm=llm)
        result = await t.transform(content, make_events())
        assert len(llm.calls) == 1
        new = result.content
        assert len(new) == 4
        assert new[0] == content[0]
        assert _summary_text(new[1]) == "[Summary of 4 earlier messages]\n\nS"
        assert new[2] == content[5] and new[3] == content[6]

    @pytest.mark.asyncio
    async def test_ac7_r2_kept_window_extended_to_keep_tool_pair(self):
        """AC7-r2: keep_tokens fits only R[tc3]; window extends back to A[tc3]."""
        content = _r1_content()
        t = build_transformer(config=_msg_config(102), llm=ScriptedLLM("S"))
        result = await t.transform(content, make_events())
        new = result.content
        assert new[0] == content[0]
        assert _summary_text(new[1]).startswith("[Summary of 4 earlier messages]")
        assert new[2:] == [content[5], content[6]]

    @pytest.mark.asyncio
    async def test_ac7_r3_protected_prompt_is_first_real_user_text(self):
        """AC7-r3: a leading tool-result-only user message is in the span, not the prompt."""
        lead = _R_tc(0, 400)
        prompt = _U("P" * 400)
        content = [lead, prompt, _A_tc(1), _R_tc(1, 4000), _A_tc(3), _R_tc(3, 400)]
        llm = ScriptedLLM("S")
        t = build_transformer(config=_msg_config(110), llm=llm)
        result = await t.transform(content, make_events())
        new = result.content
        assert new[0] == prompt
        assert _summary_text(new[1]) == "[Summary of 3 earlier messages]\n\nS"
        assert new[2:] == content[4:]
        rendered = llm.calls[0].messages[0].content[0].text
        assert "r0" in rendered  # the leading message was summarised

    @pytest.mark.asyncio
    async def test_ac7_r4_empty_span_is_noop(self):
        """AC7-r4: everything fits keep_tokens -> no LLM call, content unchanged."""
        llm = ScriptedLLM("S")
        content = [_U("P" * 400), _A_tc(1), _R_tc(1, 40)]
        t = build_transformer(config=_msg_config(10_000), llm=llm)
        result = await t.transform(content, make_events())
        assert llm.calls == []
        assert result.content is None

    @pytest.mark.asyncio
    async def test_ac7_r5_second_compaction_folds_previous_summary(self):
        """AC7-r5: summary1 is inside the new span; exactly one summary remains."""
        llm = ScriptedLLM("S2")
        summary1 = _U("[Summary of 4 earlier messages]\n\nS1")
        content = [_U("P" * 400), summary1, _A_tc(3), _R_tc(3, 4000), _A_tc(4), _R_tc(4, 400)]
        t = build_transformer(config=_msg_config(110), llm=llm)
        result = await t.transform(content, make_events())
        new = result.content
        assert summary1 not in new
        summaries = [m for m in new if isinstance(m, Message) and "[Summary of" in _summary_text_or_empty(m)]
        assert len(summaries) == 1
        assert _summary_text(new[1]) == "[Summary of 3 earlier messages]\n\nS2"
        assert "S1" in llm.calls[0].messages[0].content[0].text

    @pytest.mark.asyncio
    async def test_ac7_r6_request_render(self):
        """AC7-r6: system = summarization_prompt; one user message with role, text,
        tool name + JSON args, tool-result text; no tools."""
        llm = ScriptedLLM("S")
        content = [
            _U("P" * 400),
            _A_text("thinking aloud ALPHA"),
            Message(role="assistant", content=[ToolUseBlock(id="t9", name="grep_files", input={"pattern": "BETA"})]),
            Message(role="user", content=[ToolResultBlock(tool_use_id="t9", content="GAMMA" + "x" * 4000)]),
            _A_tc(3), _R_tc(3, 400),
        ]
        cfg = _msg_config(110, summarization_prompt="CUSTOM PROMPT")
        t = build_transformer(config=cfg, llm=llm)
        await t.transform(content, make_events())
        req = llm.calls[0]
        assert [b.text for b in req.system] == ["CUSTOM PROMPT"]
        assert len(req.messages) == 1 and req.messages[0].role == "user"
        text = req.messages[0].content[0].text
        for needle in ("assistant", "user", "thinking aloud ALPHA", "grep_files",
                       json.dumps({"pattern": "BETA"}), "GAMMA"):
            assert needle in text, needle
        assert not req.tools

    @pytest.mark.asyncio
    async def test_ac7_r7_default_and_custom_header_and_textblock_path_unchanged(self):
        """AC7-r7: summary_header unset -> default with {count}; custom header honoured;
        TextBlock-only content keeps today's bare summary block."""
        t = build_transformer(config=_msg_config(110, summary_header="<<{count} msgs>>"), llm=ScriptedLLM("S"))
        result = await t.transform(_r1_content(), make_events())
        assert _summary_text(result.content[1]) == "<<4 msgs>>\n\nS"

        llm = MockLLM("plain")
        t = build_transformer(config=make_config(keep_strategy="keep_within_tokens", keep_tokens=4), llm=llm)
        result = await t.transform(make_text_blocks(8), make_events())
        assert result.content[0] == TextBlock(text="plain")
        assert len(result.content) == 5


def _summary_text_or_empty(m: Message) -> str:
    return m.content[0].text if m.content and isinstance(m.content[0], TextBlock) else ""


class TestMessageSummarizationFailures:
    async def _fail(self, llm, caplog, content=None, **extra):
        content = content or _r1_content()
        t = build_transformer(config=_msg_config(110, **extra), llm=llm)
        with caplog.at_level(logging.WARNING, logger="sr2.pipeline.transformers.summarization"):
            result = await t.transform(content, make_events())
        assert result.content is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        return t, warnings[0]

    @pytest.mark.asyncio
    async def test_ac8_r1_llm_raises(self, caplog):
        """AC8-r1: raise -> no change, one WARNING naming timeout, nothing escapes."""
        _, rec = await self._fail(ScriptedLLM(exc=RuntimeError("timeout")), caplog)
        assert "timeout" in rec.getMessage()

    @pytest.mark.asyncio
    async def test_ac8_r2_empty_summary(self, caplog):
        """AC8-r2: no TextBlock / empty text -> no change, WARNING."""
        await self._fail(ScriptedLLM(content=[]), caplog)
        caplog.clear()
        await self._fail(ScriptedLLM(text="   "), caplog)

    @pytest.mark.asyncio
    async def test_ac8_r3_summary_not_smaller_than_span(self, caplog):
        """AC8-r3: counted summary >= span -> no change, WARNING."""
        content = [_U("P" * 400), _A_tc(1), _R_tc(1, 40), _A_tc(3), _R_tc(3, 400)]
        await self._fail(ScriptedLLM("L" * 4000), caplog, content=content)

    @pytest.mark.asyncio
    async def test_ac8_r4_no_retry_before_growth(self, caplog):
        """AC8-r4: after a failure, growth < retry_growth_tokens -> no LLM call."""
        llm = ScriptedLLM(exc=RuntimeError("timeout"))
        t, _ = await self._fail(llm, caplog, retry_growth_tokens=1000)
        llm.exc = None
        grown = _r1_content() + [_A_text("y" * 3996)]  # +999 tokens
        result = await t.transform(grown, make_events())
        assert len(llm.calls) == 1
        assert result.content is None

    @pytest.mark.asyncio
    async def test_ac8_r5_retry_after_growth(self, caplog):
        """AC8-r5: after a failure, growth >= retry_growth_tokens -> LLM called again."""
        llm = ScriptedLLM(exc=RuntimeError("timeout"))
        t, _ = await self._fail(llm, caplog, retry_growth_tokens=1000)
        llm.exc = None
        grown = _r1_content() + [_A_text("y" * 4000), _A_tc(5), _R_tc(5, 40)]  # +1000+ tokens
        result = await t.transform(grown, make_events())
        assert len(llm.calls) == 2
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_ac8_default_retry_growth_is_10000(self, caplog):
        """AC8 default: retry_growth_tokens unset -> 10000; +9999 tokens does not retry."""
        llm = ScriptedLLM(exc=RuntimeError("timeout"))
        t, _ = await self._fail(llm, caplog)
        llm.exc = None
        await t.transform(_r1_content() + [_A_text("y" * 39996)], make_events())
        assert len(llm.calls) == 1
        await t.transform(_r1_content() + [_A_text("y" * 40000)], make_events())
        assert len(llm.calls) == 2
