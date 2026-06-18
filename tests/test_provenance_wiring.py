"""Tests for spc-50: Persistent SQLiteProvenanceStore wiring.

Covers:
  A. SpectreConfig.provenance_store_path field
  B. Runtime._resolve_provenance_path() — default, custom, disabled
  C. Runtime.initialize() connects SQLiteProvenanceStore
  D. Runtime.aclose() closes the store
  E. Runtime → Session → SR2 threading
  F. Persistence across reconnect (simulated restart)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.models import TextBlock
from sr2.pipeline.provenance import Entry, EntryOrigin
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
# A. Config field
# ---------------------------------------------------------------------------

class TestConfigProvenanceStorePath:
    def test_defaults_to_none(self):
        config = _make_config()
        assert config.provenance_store_path is None

    def test_accepts_custom_path(self):
        config = _make_config(provenance_store_path="/custom/path.db")
        assert config.provenance_store_path == "/custom/path.db"

    def test_accepts_empty_string_to_disable(self):
        config = _make_config(provenance_store_path="")
        assert config.provenance_store_path == ""

    def test_accepts_tilde_path(self):
        config = _make_config(provenance_store_path="~/.my-spectre/provenance.db")
        assert config.provenance_store_path == "~/.my-spectre/provenance.db"


# ---------------------------------------------------------------------------
# B. Runtime._resolve_provenance_path()
# ---------------------------------------------------------------------------

class TestResolveProvenancePath:
    def test_default_returns_sr2_spectre_path(self):
        from sr2_spectre.runtime import Runtime

        config = _make_config()
        path = Runtime._resolve_provenance_path(config)
        assert path is not None
        assert "provenance.db" in path
        assert ".sr2-spectre" in path

    def test_custom_path_is_resolved(self):
        from sr2_spectre.runtime import Runtime

        config = _make_config(provenance_store_path="/tmp/test-provenance.db")
        path = Runtime._resolve_provenance_path(config)
        assert path == "/tmp/test-provenance.db"

    def test_empty_string_returns_none(self):
        from sr2_spectre.runtime import Runtime

        config = _make_config(provenance_store_path="")
        path = Runtime._resolve_provenance_path(config)
        assert path is None

    def test_tilde_expanded(self):
        from sr2_spectre.runtime import Runtime

        config = _make_config(provenance_store_path="~/custom.db")
        path = Runtime._resolve_provenance_path(config)
        assert not path.startswith("~")
        assert str(Path.home()) in path


# ---------------------------------------------------------------------------
# C. Runtime.initialize() connects store
# ---------------------------------------------------------------------------

class TestRuntimeInitializeProvenance:
    @pytest.mark.asyncio
    async def test_initialize_connects_store(self):
        from sr2_spectre.runtime import Runtime

        config = _make_config()
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=config)

        assert runtime._provenance_store is None
        assert runtime._provenance_store_path is not None

        # Patch where the import resolves — the sqlite module
        with patch(
            "sr2.pipeline.stores.sqlite.SQLiteProvenanceStore"
        ) as MockStore:
            mock_instance = AsyncMock()
            MockStore.return_value = mock_instance
            await runtime.initialize()

        MockStore.assert_called_once_with(db_path=runtime._provenance_store_path)
        mock_instance.connect.assert_awaited_once()
        assert runtime._provenance_store is mock_instance

    @pytest.mark.asyncio
    async def test_initialize_skips_when_disabled(self):
        from sr2_spectre.runtime import Runtime

        config = _make_config(provenance_store_path="")
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=config)

        assert runtime._provenance_store_path is None

        with patch(
            "sr2.pipeline.stores.sqlite.SQLiteProvenanceStore"
        ) as MockStore:
            await runtime.initialize()

        MockStore.assert_not_called()
        assert runtime._provenance_store is None


# ---------------------------------------------------------------------------
# D. Runtime.aclose() closes store
# ---------------------------------------------------------------------------

class TestRuntimeACloseProvenance:
    @pytest.mark.asyncio
    async def test_aclose_closes_store(self):
        from sr2_spectre.runtime import Runtime

        config = _make_config()
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=config)

        mock_store = AsyncMock()
        runtime._provenance_store = mock_store

        await runtime.aclose()

        mock_store.close.assert_awaited_once()
        assert runtime._provenance_store is None

    @pytest.mark.asyncio
    async def test_aclose_noop_when_no_store(self):
        from sr2_spectre.runtime import Runtime

        config = _make_config(provenance_store_path="")
        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=config)

        await runtime.aclose()  # must not raise


# ---------------------------------------------------------------------------
# E. Runtime → Session → SR2 threading
# ---------------------------------------------------------------------------

class TestProvenanceStoreThreading:
    def test_new_session_passes_store_to_session(self):
        from sr2_spectre.runtime import Runtime

        mock_store = MagicMock()

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                MockSR2.return_value = MagicMock()
                runtime = Runtime(config=_make_config())
                runtime._provenance_store = mock_store
                runtime.new_session(frame_id="test-frame")

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["provenance_store"] is mock_store

    def test_new_session_passes_none_when_not_initialized(self):
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                MockSR2.return_value = MagicMock()
                runtime = Runtime(config=_make_config())
                # _provenance_store is None before initialize()
                runtime.new_session(frame_id="test-frame")

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["provenance_store"] is None

    def test_session_accepts_provenance_store_param(self):
        """Session.__init__ accepts provenance_store and passes it to SR2."""
        from sr2_spectre.session import Session

        mock_store = MagicMock()
        mock_llm = MagicMock()

        with patch("sr2_spectre.session.SR2") as MockSR2:
            MockSR2.return_value = MagicMock()
            Session(
                frame_id="f1",
                config=_make_config(),
                llm=mock_llm,
                registry=MagicMock(),
                provenance_store=mock_store,
            )

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["provenance_store"] is mock_store


# ---------------------------------------------------------------------------
# F. Persistence across reconnect (simulated restart)
# ---------------------------------------------------------------------------

class TestPersistenceAcrossRestart:
    @pytest.mark.asyncio
    async def test_entries_survive_reconnect(self):
        """Write an entry, close store, reopen, verify entry persists.

        Simulates a process restart: close() → new instance → connect() → get().
        """
        from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # First "process": write entry
            store1 = SQLiteProvenanceStore(db_path=db_path)
            await store1.connect()

            entry = Entry(
                id="test-entry-1",
                session_id="test-session",
                layer="system",
                origin=EntryOrigin(kind="resolver", name="static"),
                content=TextBlock(text="Hello, persistent provenance"),
                sources=(),
                created_at=datetime.now(timezone.utc),
            )
            await store1.write(entry)
            await store1.close()

            # Second "process": read entry
            store2 = SQLiteProvenanceStore(db_path=db_path)
            await store2.connect()

            retrieved = await store2.get("test-entry-1")
            assert retrieved is not None
            assert retrieved.id == "test-entry-1"
            assert retrieved.session_id == "test-session"
            assert isinstance(retrieved.content, TextBlock)
            assert retrieved.content.text == "Hello, persistent provenance"

            await store2.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_multiple_sessions_share_store(self):
        """Two sessions writing to the same store can read each other's entries."""
        from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = SQLiteProvenanceStore(db_path=db_path)
            await store.connect()

            # Session A writes
            entry_a = Entry(
                id="entry-a",
                session_id="session-a",
                layer="system",
                origin=EntryOrigin(kind="resolver", name="static"),
                content=TextBlock(text="From session A"),
                sources=(),
                created_at=datetime.now(timezone.utc),
            )
            await store.write(entry_a)

            # Session B writes
            entry_b = Entry(
                id="entry-b",
                session_id="session-b",
                layer="system",
                origin=EntryOrigin(kind="resolver", name="static"),
                content=TextBlock(text="From session B"),
                sources=(),
                created_at=datetime.now(timezone.utc),
            )
            await store.write(entry_b)

            # Both entries are queryable
            sessions_a = await store.get_session("session-a")
            assert len(sessions_a) == 1
            assert sessions_a[0].id == "entry-a"

            sessions_b = await store.get_session("session-b")
            assert len(sessions_b) == 1
            assert sessions_b[0].id == "entry-b"

            await store.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_runtime_full_lifecycle(self):
        """End-to-end: Runtime constructs store, initialize connects, aclose closes."""
        from sr2_spectre.runtime import Runtime

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            config = _make_config(provenance_store_path=db_path)

            with patch("sr2_spectre.runtime.LiteLLMCallable"):
                runtime = Runtime(config=config)

            # Before initialize: store not connected
            assert runtime._provenance_store is None
            assert runtime._provenance_store_path == db_path

            # After initialize: store connected
            await runtime.initialize()
            assert runtime._provenance_store is not None

            # DB file should exist after connect
            assert Path(db_path).exists()

            # After aclose: store disconnected
            await runtime.aclose()
            assert runtime._provenance_store is None
        finally:
            Path(db_path).unlink(missing_ok=True)
