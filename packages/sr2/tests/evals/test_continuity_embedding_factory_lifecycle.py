"""sr2-175 — the embedding registry factory owns its cache dir, never leaks it.

``_make_embedding`` used to call ``tempfile.mkdtemp`` on every registry
invocation and never remove the directory: each ``CANDIDATES["embedding"]()``
call stranded one temp dir holding a SQLite vector cache. The FR40 runner
never routed through the registry (it built the index itself under the run
directory), so the leak stayed latent — but any future bakeoff / sr2-100
caller of the registry entry paid it.

The factory now takes an optional runner-provided *workdir* (the
``_fr40_build_index`` contract) and, called bare, attaches an owned
``tempfile.TemporaryDirectory`` to the returned index so the directory dies
with it.

Rows:

r1 — a bare registry call creates exactly one cache dir under the system
     temp matching the pinned prefix, that dir holds the SQLite cache path
     while the index is referenced, and it is gone after the index is
     released; a second call + release returns the count to baseline (no
     per-call leak).
r2 — a caller-provided workdir places the SQLite cache at
     ``<workdir>/replay_vectors.sqlite3`` and creates nothing under the
     system temp, leaving the caller in charge of cleanup.
r3 — the FR40 runner arm builds through the same factory, so the
     per-repetition rebuild still gets the workdir-fixed cache path
     (warm-cache continuity across repetitions stays pinned by the
     existing AC27 cage and the sr2-177 isolation rows).
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

from evals.continuity import candidates as cands
from evals.continuity import runner as crunner


def _replay_cache_dirs() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(
        cands.EMBED_CACHE_DIR_PREFIX + "*"))


def test_r1_bare_registry_call_leaves_no_cache_dir_behind():
    before = _replay_cache_dirs()

    index = cands.CANDIDATES["embedding"]()
    created = _replay_cache_dirs() - before
    assert len(created) == 1, f"expected one owned cache dir, got {created}"
    (cache_dir,) = created
    assert cache_dir.is_dir()
    # The SQLite vector cache resolves inside the owned directory.
    db_path = Path(index._index._path)
    assert db_path.parent == cache_dir

    del index
    gc.collect()
    assert not cache_dir.exists(), (
        "owned cache dir survived release of its index")

    # Second generation: the count returns to baseline rather than
    # growing by one per call (the mkdtemp leak shape).
    index2 = cands.CANDIDATES["embedding"]()
    assert len(_replay_cache_dirs() - before) == 1
    del index2
    gc.collect()
    assert _replay_cache_dirs() == before


def test_r2_runner_provided_workdir_owns_no_system_temp(tmp_path):
    before = _replay_cache_dirs()

    index = cands._make_embedding(tmp_path)

    assert Path(index._index._path) == tmp_path / "replay_vectors.sqlite3"
    assert not hasattr(index, "_owned_cache_dir")
    assert _replay_cache_dirs() == before, (
        "a workdir-supplied call must not create a system temp dir")


def test_r3_fr40_build_index_routes_through_the_factory(tmp_path):
    index = crunner._fr40_build_index("embedding", tmp_path)

    assert type(index) is crunner._ReplayEmbeddingIndex
    assert Path(index._index._path) == tmp_path / "replay_vectors.sqlite3"
    assert not hasattr(index, "_owned_cache_dir")
    assert not _replay_cache_dirs(), (
        "the FR40 runner arm must not touch the system temp")
