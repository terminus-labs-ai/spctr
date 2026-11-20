"""Continuity index candidate registry for the versioned benchmark (sr2-98 / AC17).

Both candidates receive the same policy-eligible records, query, and token
budget. Store truth remains in the immutable ledger; index state is
rebuildable. This module publishes raw candidate metrics, not a selection
verdict.

The benchmark runner (sr2-91, sr2-100) imports this registry and runs each
candidate against identical scenarios with no candidate-specific inputs or
budgets.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from sr2.continuity.index import ContinuityIndex
from sr2.continuity.indexes.graph import GraphIndex
from sr2.continuity.indexes.structured import StructuredIndex
from sr2.continuity.indexes.vector_hybrid import VectorHybridIndex

# Deterministic embedding replay (sr2-163 / AC27): installing it at import
# of this registry — which every candidate-selection path imports before any
# embedding-boundary spy is in place — guarantees the replay is the active
# ``litellm.aembedding`` by the time a deterministic run drives the embedding
# candidate, so committed fixture vectors are served with no live call.
from evals.continuity.runner import _install_embedding_replay

_install_embedding_replay()


def _make_structured() -> ContinuityIndex:
    return StructuredIndex()


def _make_vector_hybrid() -> ContinuityIndex:
    return VectorHybridIndex()


def _make_graph() -> ContinuityIndex:
    return GraphIndex()


#: Prefix for the throwaway cache directory an unguided ``CANDIDATES[
#: "embedding"]()`` call owns (sr2-175). Pinned by the lifecycle tests.
EMBED_CACHE_DIR_PREFIX = "sr2-embedding-replay-"


def _make_embedding(
    workdir: str | Path | None = None,
    *,
    query_cache: dict[str, list[float]] | None = None,
) -> ContinuityIndex:
    """The FR40 embedding candidate (sr2-163 / AC27): a
    ``ContinuityIndex`` adapter over ``EmbeddingContinuityIndex`` configured
    for deterministic fixture replay — the committed recorded vectors are
    served at the embedding boundary (no network) and a miss in the fixture
    is an error. The runner drives it through the replayed embedding
    boundary; the factory is built lazily so the fixture (and its throwaway
    SQLite vector cache) exist only for the run that uses it.

    Cache-directory lifecycle (sr2-175): the SQLite vector cache lives at
    ``<workdir>/replay_vectors.sqlite3``. A caller that owns a run directory
    (the FR40 runner via ``_fr40_build_index``) passes *workdir* and decides
    when the cache dies; every repetition of one run shares the path, so a
    rebuild keeps the cache warm. Called bare through the registry, the
    factory owns a ``tempfile.TemporaryDirectory`` attached to the returned
    index — the directory is removed once the index is released, never left
    behind per call."""
    from evals.continuity.runner import _ReplayEmbeddingIndex, load_vector_fixture

    fixture = load_vector_fixture()
    if workdir is not None:
        db_path = Path(workdir) / "replay_vectors.sqlite3"
        return _ReplayEmbeddingIndex(fixture, db_path, query_cache=query_cache)
    owned = tempfile.TemporaryDirectory(prefix=EMBED_CACHE_DIR_PREFIX)
    index = _ReplayEmbeddingIndex(
        fixture,
        Path(owned.name) / "replay_vectors.sqlite3",
        query_cache=query_cache,
    )
    # Attached last on purpose: CPython clears instance attributes in
    # insertion order, so the owned directory is released only after every
    # object below it (and any sqlite handle it may open) is gone.
    index._owned_cache_dir = owned
    return index


#: Registry of candidate index factories. Each entry is a zero-argument
#: callable that produces a fresh ContinuityIndex instance. The benchmark
#: runner calls each factory once per scenario to ensure isolation.
CANDIDATES: dict[str, Callable[[], ContinuityIndex]] = {
    "structured": _make_structured,
    "vector_hybrid": _make_vector_hybrid,
    "graph": _make_graph,
    "embedding": _make_embedding,
}
