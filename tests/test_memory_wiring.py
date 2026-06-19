"""Tests for spc-49: in-memory memory subsystem wiring.

Wires sr2's existing memory components (InMemoryMemoryStore, MemoryResolver,
MemoryExtractionTransformer) into Spectre's Runtime → Session → SR2 path.

Covers:
  A. Runtime constructs a shared InMemoryMemoryStore
  B. Runtime → Session → SR2 threading of memory_store
  C. Session.__init__ accepts memory_store and forwards it to SR2
  D. The store is shared across sessions (one per Runtime)
  E. extract → inject roundtrip (transformer saves, resolver retrieves)
  F. resolver-only-injection invariant: resolve() reads, never writes

PERSISTENCE CAVEAT: this bead wires the in-memory path only. InMemoryMemoryStore
is dict-backed and lost on process restart. A persistent MemoryStore impl is
obsidian-cor, a follow-on.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sr2.config.models import ResolverConfig, TransformerConfig
from sr2.memory import (
    InMemoryMemoryStore,
    MemoryExtractionTransformer,
    MemoryResolver,
)
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig


def _make_config(**kwargs) -> SpectreConfig:
    """Build a minimal SpectreConfig, allowing overrides."""
    overrides = dict(
        agent=AgentConfig(name="test"),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={"layers": [
            {"name": "system", "target": "system", "resolvers": [
                {"type": "static", "config": {"text": "You are helpful."}}
            ]},
        ]},
    )
    overrides.update(kwargs)
    return SpectreConfig(**overrides)


# ---------------------------------------------------------------------------
# A. Runtime constructs a shared InMemoryMemoryStore
# ---------------------------------------------------------------------------

class TestRuntimeMemoryStoreConstruction:
    def test_runtime_has_memory_store(self):
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=_make_config())

        assert runtime._memory_store is not None
        assert isinstance(runtime._memory_store, InMemoryMemoryStore)

    def test_memory_store_available_before_initialize(self):
        """Unlike the async provenance store, the in-memory store exists at
        construction time — no initialize() round-trip required."""
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=_make_config())

        assert isinstance(runtime._memory_store, InMemoryMemoryStore)


# ---------------------------------------------------------------------------
# B. Runtime → Session → SR2 threading
# ---------------------------------------------------------------------------

class TestMemoryStoreThreading:
    def test_new_session_passes_store_to_sr2(self):
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                MockSR2.return_value = MagicMock()
                runtime = Runtime(config=_make_config())
                runtime.new_session(frame_id="test-frame")

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["memory_store"] is runtime._memory_store

    def test_new_session_passes_non_none_store(self):
        """The store is always present (constructed in __init__), so SR2 never
        receives None — distinct from the lazily-connected provenance store."""
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                MockSR2.return_value = MagicMock()
                runtime = Runtime(config=_make_config())
                runtime.new_session(frame_id="test-frame")

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["memory_store"] is not None


# ---------------------------------------------------------------------------
# C. Session.__init__ accepts and forwards memory_store
# ---------------------------------------------------------------------------

class TestSessionMemoryStoreParam:
    def test_session_forwards_memory_store_to_sr2(self):
        from sr2_spectre.session import Session

        mock_store = MagicMock()

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            Session(
                frame_id="f1",
                config=_make_config(),
                llm=MagicMock(),
                registry=MagicMock(),
                memory_store=mock_store,
            )

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["memory_store"] is mock_store

    def test_session_memory_store_defaults_to_none(self):
        """Sessions built without a store (e.g. legacy callers) pass None."""
        from sr2_spectre.session import Session

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            Session(
                frame_id="f1",
                config=_make_config(),
                llm=MagicMock(),
                registry=MagicMock(),
            )

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["memory_store"] is None


# ---------------------------------------------------------------------------
# D. One store shared across all sessions
# ---------------------------------------------------------------------------

class TestSharedStoreAcrossSessions:
    def test_two_sessions_share_one_store(self):
        from sr2_spectre.runtime import Runtime

        captured: list[object] = []

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                MockSR2.return_value = MagicMock()
                runtime = Runtime(config=_make_config())
                runtime.new_session(frame_id="frame-a")
                runtime.new_session(frame_id="frame-b")
                for call in MockSR2.call_args_list:
                    captured.append(call.kwargs["memory_store"])

        assert captured[0] is captured[1]
        assert captured[0] is runtime._memory_store


# ---------------------------------------------------------------------------
# E. extract → inject roundtrip (within one process)
# ---------------------------------------------------------------------------

class TestExtractInjectRoundtrip:
    @pytest.mark.asyncio
    async def test_fact_stated_in_one_turn_is_retrievable_later(self):
        """A fact extracted from an assistant response is injected on a later
        user turn — through the real transformer + resolver + shared store."""
        store = InMemoryMemoryStore()

        # Transformer extracts from a completed assistant response.
        transformer = MemoryExtractionTransformer.build(
            TransformerConfig(type="memory_extraction"),
            Dependencies(memory_store=store),
        )
        assistant_event = Event(
            name="assistant_response",
            phase=EventPhase.COMPLETED,
            source_layer="conversation",
            data=[TextBlock(text="I prefer dark mode in the editor")],
        )
        await transformer.transform([], [assistant_event])

        # Store now holds the extracted preference.
        assert store.get_all(), "transformer should have saved a memory"

        # Resolver injects it on a later user turn whose text overlaps.
        resolver = MemoryResolver.build(
            ResolverConfig(type="memory"),
            Dependencies(memory_store=store),
        )
        # InMemoryMemoryStore.search is substring match (query in content), so
        # the query text must appear within the stored memory's content.
        user_event = Event(
            name="user_input",
            phase=EventPhase.STARTING,
            source_layer="conversation",
            data=[TextBlock(text="dark mode in the editor")],
        )
        resolved = await resolver.resolve([user_event])

        assert resolved.content, "resolver should inject the remembered fact"
        injected_text = " ".join(
            b.text for b in resolved.content if isinstance(b, TextBlock)
        )
        assert "dark mode" in injected_text

    @pytest.mark.asyncio
    async def test_unrelated_query_injects_nothing(self):
        """A user turn with no overlap to stored memories injects nothing."""
        store = InMemoryMemoryStore()

        transformer = MemoryExtractionTransformer.build(
            TransformerConfig(type="memory_extraction"),
            Dependencies(memory_store=store),
        )
        await transformer.transform(
            [],
            [Event(
                name="assistant_response",
                phase=EventPhase.COMPLETED,
                source_layer="conversation",
                data=[TextBlock(text="I prefer dark mode in the editor")],
            )],
        )

        resolver = MemoryResolver.build(
            ResolverConfig(type="memory"),
            Dependencies(memory_store=store),
        )
        resolved = await resolver.resolve([Event(
            name="user_input",
            phase=EventPhase.STARTING,
            source_layer="conversation",
            data=[TextBlock(text="what is the weather in Helsinki")],
        )])

        assert resolved.content == []


# ---------------------------------------------------------------------------
# F. resolver-only-injection invariant
# ---------------------------------------------------------------------------

class TestResolverOnlyInjection:
    @pytest.mark.asyncio
    async def test_resolve_does_not_mutate_store(self):
        """MemoryResolver only reads. resolve() must never call save()."""
        store = InMemoryMemoryStore()
        transformer = MemoryExtractionTransformer.build(
            TransformerConfig(type="memory_extraction"),
            Dependencies(memory_store=store),
        )
        await transformer.transform(
            [],
            [Event(
                name="assistant_response",
                phase=EventPhase.COMPLETED,
                source_layer="conversation",
                data=[TextBlock(text="I prefer dark mode in the editor")],
            )],
        )
        before = len(store.get_all())

        resolver = MemoryResolver.build(
            ResolverConfig(type="memory"),
            Dependencies(memory_store=store),
        )
        with patch.object(store, "save", side_effect=AssertionError("resolver wrote to store")):
            await resolver.resolve([Event(
                name="user_input",
                phase=EventPhase.STARTING,
                source_layer="conversation",
                data=[TextBlock(text="dark mode in the editor")],
            )])

        assert len(store.get_all()) == before
