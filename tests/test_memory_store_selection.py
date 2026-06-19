"""Tests for spc-49 cycle-2: config-driven memory store backend selection.

The Runtime must select the memory store backend from config/env rather than
always using InMemoryMemoryStore:

  1. Default (no DSN, no env) -> InMemoryMemoryStore   (existing behavior)
  2. config.memory_store_dsn set -> PostgresMemoryStore(dsn)
  3. env SPECTRE_MEMORY_DSN set (config absent) -> PostgresMemoryStore(env dsn)
  4. config DSN wins over env DSN
  5. memory_store_dsn == "" -> explicitly disabled -> InMemoryMemoryStore
     (mirrors the provenance "" convention in Runtime._resolve_provenance_path)
  6. aclose() closes the store iff it is closeable (Postgres has close())
  7. The selected store (whatever backend) is threaded to sessions / SR2
  8. One real-DB integration test proving end-to-end construction

PATCH-TARGET CONTRACT (IMPORTANT for the implementer):
  These unit tests patch ``sr2_spectre.runtime.PostgresMemoryStore``. For that
  target to exist and be the symbol the Runtime actually instantiates, the
  implementer MUST import PostgresMemoryStore at the TOP of runtime.py:

      from sr2.memory import PostgresMemoryStore

  i.e. bind it into the runtime module namespace (NOT a lazy
  ``from sr2.memory import PostgresMemoryStore`` inside __init__, which would
  make ``patch("sr2_spectre.runtime.PostgresMemoryStore")`` a no-op). This is a
  hard contract the tests depend on.

These tests are written BEFORE the implementation. Expected red failures at
this stage: SpectreConfig rejects ``memory_store_dsn`` and/or the Runtime has
no DSN-based selection logic.
"""

from __future__ import annotations

import os

import pytest

from sr2.memory import InMemoryMemoryStore, PostgresMemoryStore

# unittest.mock imported after project imports for symmetry with the existing
# test_memory_wiring.py style.
from unittest.mock import MagicMock, patch  # noqa: E402

from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig  # noqa: E402


def _make_config(**kwargs) -> SpectreConfig:
    """Build a minimal SpectreConfig, allowing overrides.

    Copied from tests/test_memory_wiring.py. By default leaves
    ``memory_store_dsn`` UNSET so the default-in-memory path is exercised
    unless a test explicitly passes one.
    """
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


_PG_DSN = "postgresql://user:pw@db.example:5432/mem"
_ENV_DSN = "postgresql://user:pw@env.example:5432/envmem"


# ---------------------------------------------------------------------------
# 1. Default -> in-memory
# ---------------------------------------------------------------------------

class TestDefaultIsInMemory:
    def test_no_dsn_no_env_selects_in_memory(self, monkeypatch):
        """No config DSN + no env var -> InMemoryMemoryStore (legacy behavior)."""
        monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=_make_config())

        assert isinstance(runtime._memory_store, InMemoryMemoryStore)


# ---------------------------------------------------------------------------
# 2. Config DSN -> Postgres
# ---------------------------------------------------------------------------

class TestConfigDsnSelectsPostgres:
    def test_config_dsn_constructs_postgres_with_that_dsn(self, monkeypatch):
        """memory_store_dsn in config -> PostgresMemoryStore(<dsn>) once."""
        monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"), \
                patch("sr2_spectre.runtime.PostgresMemoryStore") as MockPg:
            MockPg.return_value = MagicMock()
            runtime = Runtime(config=_make_config(memory_store_dsn=_PG_DSN))

        MockPg.assert_called_once_with(_PG_DSN)
        assert runtime._memory_store is MockPg.return_value


# ---------------------------------------------------------------------------
# 3. Env DSN (config absent) -> Postgres
# ---------------------------------------------------------------------------

class TestEnvDsnSelectsPostgres:
    def test_env_dsn_selects_postgres_when_config_dsn_absent(self, monkeypatch):
        """env SPECTRE_MEMORY_DSN + no config DSN -> Postgres(env dsn)."""
        monkeypatch.setenv("SPECTRE_MEMORY_DSN", _ENV_DSN)
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"), \
                patch("sr2_spectre.runtime.PostgresMemoryStore") as MockPg:
            MockPg.return_value = MagicMock()
            runtime = Runtime(config=_make_config())  # memory_store_dsn unset (None)

        MockPg.assert_called_once_with(_ENV_DSN)
        assert runtime._memory_store is MockPg.return_value


# ---------------------------------------------------------------------------
# 4. Config DSN wins over env DSN
# ---------------------------------------------------------------------------

class TestConfigDsnPrecedence:
    def test_config_dsn_takes_precedence_over_env(self, monkeypatch):
        """Both set -> the config DSN is used, not the env DSN."""
        monkeypatch.setenv("SPECTRE_MEMORY_DSN", _ENV_DSN)
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"), \
                patch("sr2_spectre.runtime.PostgresMemoryStore") as MockPg:
            MockPg.return_value = MagicMock()
            runtime = Runtime(config=_make_config(memory_store_dsn=_PG_DSN))

        MockPg.assert_called_once_with(_PG_DSN)


# ---------------------------------------------------------------------------
# 5. Empty-string DSN -> disabled -> in-memory
# ---------------------------------------------------------------------------

class TestEmptyStringDsnDisabled:
    def test_empty_string_config_dsn_falls_back_to_in_memory(self, monkeypatch):
        """memory_store_dsn == "" means explicitly disabled -> InMemory.

        Mirrors Runtime._resolve_provenance_path: "" == disabled, NOT a path.
        Postgres must NOT be constructed.
        """
        monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"), \
                patch("sr2_spectre.runtime.PostgresMemoryStore") as MockPg:
            runtime = Runtime(config=_make_config(memory_store_dsn=""))

        MockPg.assert_not_called()
        assert isinstance(runtime._memory_store, InMemoryMemoryStore)

    def test_empty_string_config_disables_even_when_env_set(self, monkeypatch):
        """Explicit "" disable beats a populated env var.

        Mirrors the provenance "" convention: an explicit empty string is a hard
        disable, not "unset". A set SPECTRE_MEMORY_DSN must NOT resurrect Postgres
        when the config explicitly disables it.
        """
        monkeypatch.setenv("SPECTRE_MEMORY_DSN", _ENV_DSN)
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"), \
                patch("sr2_spectre.runtime.PostgresMemoryStore") as MockPg:
            runtime = Runtime(config=_make_config(memory_store_dsn=""))

        MockPg.assert_not_called()
        assert isinstance(runtime._memory_store, InMemoryMemoryStore)


# ---------------------------------------------------------------------------
# 6. aclose() closes the store iff closeable
# ---------------------------------------------------------------------------

class TestAcloseClosesStore:
    @pytest.mark.asyncio
    async def test_aclose_closes_postgres_store(self, monkeypatch):
        """When a Postgres store is selected, aclose() calls its close()."""
        monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"), \
                patch("sr2_spectre.runtime.PostgresMemoryStore") as MockPg:
            mock_store = MagicMock()
            MockPg.return_value = mock_store
            runtime = Runtime(config=_make_config(memory_store_dsn=_PG_DSN))

            await runtime.aclose()

        mock_store.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose_does_not_raise_for_in_memory_store(self, monkeypatch):
        """In-memory store has no close(); aclose() must not raise."""
        monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"):
            runtime = Runtime(config=_make_config())  # default -> in-memory

        # Must complete cleanly even though InMemoryMemoryStore has no close().
        await runtime.aclose()


# ---------------------------------------------------------------------------
# 7. Selected store is threaded to sessions -> SR2
# ---------------------------------------------------------------------------

class TestSelectedStoreThreadedToSessions:
    def test_new_session_forwards_postgres_store_to_sr2(self, monkeypatch):
        """A Postgres-selected store is the store passed to SR2 by new_session."""
        monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.runtime.LiteLLMCallable"), \
                patch("sr2_spectre.runtime.PostgresMemoryStore") as MockPg, \
                patch("sr2_spectre.session.SR2") as MockSR2:
            MockPg.return_value = MagicMock()
            MockSR2.return_value = MagicMock()
            runtime = Runtime(config=_make_config(memory_store_dsn=_PG_DSN))
            runtime.new_session(frame_id="test-frame")

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs["memory_store"] is runtime._memory_store
        assert call_kwargs["memory_store"] is MockPg.return_value


# ---------------------------------------------------------------------------
# 8. Real-DB integration test (skips if DB unreachable)
# ---------------------------------------------------------------------------

_DEFAULT_TEST_DSN = (
    "postgresql://postgres:postgres@192.168.50.117:5432/spectre_memory_test"
)
_TEST_DSN = os.environ.get("SPECTRE_MEMORY_TEST_DSN", _DEFAULT_TEST_DSN)


def _db_reachable(dsn: str) -> bool:
    """Quick connectivity probe (mirrors sr2 tests/test_pg_memory_store.py)."""
    try:
        import psycopg
    except Exception:  # pragma: no cover - import guard
        return False
    try:
        conn = psycopg.connect(dsn, connect_timeout=3)
    except Exception:
        return False
    else:
        conn.close()
        return True


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _db_reachable(_TEST_DSN),
    reason=f"Test Postgres unreachable at {_TEST_DSN}; skipping real-DB integration test.",
)
async def test_real_postgres_store_constructed_and_closed():
    """End-to-end: real config DSN -> real PostgresMemoryStore -> clean aclose.

    Proves real construction works, not just the mock. No LLM is needed, so
    LiteLLMCallable is still patched to keep the test hermetic.
    """
    from sr2_spectre.runtime import Runtime

    with patch("sr2_spectre.runtime.LiteLLMCallable"):
        runtime = Runtime(config=_make_config(memory_store_dsn=_TEST_DSN))

    assert isinstance(runtime._memory_store, PostgresMemoryStore)

    await runtime.aclose()
