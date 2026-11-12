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


def _make_embedding() -> ContinuityIndex:
    """The FR40 embedding candidate (sr2-163 / AC27): a
    ``ContinuityIndex`` adapter over ``EmbeddingContinuityIndex`` configured
    for deterministic fixture replay — the committed recorded vectors are
    served at the embedding boundary (no network) and a miss in the fixture
    is an error. The runner drives it through the replayed embedding
    boundary; the factory is built lazily so the fixture (and a throwaway
    SQLite cache) exist only for the run that uses it."""
    from evals.continuity.runner import _ReplayEmbeddingIndex, load_vector_fixture

    fixture = load_vector_fixture()
    import tempfile

    cache_dir = Path(tempfile.mkdtemp(prefix="sr2-163-embedding-"))
    db_path = cache_dir / "replay_vectors.sqlite3"
    return _ReplayEmbeddingIndex(fixture, db_path)


#: Registry of candidate index factories. Each entry is a zero-argument
#: callable that produces a fresh ContinuityIndex instance. The benchmark
#: runner calls each factory once per scenario to ensure isolation.
CANDIDATES: dict[str, Callable[[], ContinuityIndex]] = {
    "structured": _make_structured,
    "vector_hybrid": _make_vector_hybrid,
    "graph": _make_graph,
    "embedding": _make_embedding,
}
