"""Pytest configuration for sr2-spectre."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_memory_dsn_env(monkeypatch):
    """Strip SPECTRE_MEMORY_DSN from the environment for every test by default.

    Memory-store selection (obsidian-cor) reads this env var to pick the
    Postgres backend. A value leaking in from the developer's shell or CI would
    silently flip Runtime construction to Postgres and break tests that expect
    the in-memory default (e.g. test_memory_wiring.py). Tests that exercise the
    env path set it explicitly via monkeypatch, which overrides this default.
    """
    monkeypatch.delenv("SPECTRE_MEMORY_DSN", raising=False)


@pytest.fixture
def event_loop_policy():
    """Use the default asyncio event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
