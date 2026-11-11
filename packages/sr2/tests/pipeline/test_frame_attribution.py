"""sr2-158 — frame attribution for pipeline-layer provenance entries (unit layer).

Bug: every non-turn_archive provenance entry persists with
``session_id == "default"`` (``layer.py`` line 353:
``session_id=(declared or {}).get("session_id") or "default"``), so
compiled-context provenance cannot be attributed to the conversation it
served (breaks sr2-90 AC1 attribution).

Required behavior (bead sr2-158, confirmed case table; precedence fixed by
the orchestrator amendment as
**declared session_id > active frame id > "default"**):

  R1 — a turn compiled while an active-frame provider resolves a frame id for
       the turn's origin stamps every newly persisted non-turn_archive entry
       with that frame id, not "default";
  R2 — with no session frame (no provider configured, or provider returns
       None — headless/CLI) entries keep ``session_id == "default"``;
  R3 — a resolver provenance declaration carrying an explicit ``session_id``
       keeps winning over the active frame id;
  R4 — turn_archive entries keep the archivist's own session id and their
       meta turn_seq/seq behavior unchanged under an active frame;
  R5 — under one active frame, several distinct non-turn_archive layer kinds
       (static/config text, session layer, user input, markdown-file content,
       mode/topic-style guidance layers) each contribute entries in one run
       and none falls back to "default".

Grounding rules honored here:
  - Tests drive the public entry point only: ``SR2(active_frame_provider=...,
    provenance_store=...)`` construction plus ``SR2.turn(..., origin=...)``
    — the existing frame/session dependencies the bead names ("wire the
    existing active_frame_provider/session_id dependencies ... rather than
    inventing new plumbing"). Expected outcomes assert persisted ``Entry``
    fields read back from the store (``InMemoryProvenanceStore`` /
    ``SQLiteProvenanceStore``).
  - Mechanism is NOT pinned: where the frame is resolved per turn, how it
    reaches ``_synthesize_entries`` (constructor field, ``wire()``, shared
    holder), and the provider's call site are the implementer's choices.
    Only persisted session ids (plus R4's pre-existing turn_archive fields)
    are asserted.
  - No new behavior is invented: every assertion maps to R1..R5.
  - Red-run honesty (base commit, before implementation): R1's two tests,
    R3's precedence test and R5's two framed tests fail for the right
    reason — entries persist with ``session_id == "default"`` while a frame
    is active. R2's two tests, R3's declared-wins control and R4 pass today
    by design and must keep passing: R2 pins the unchanged fallback (as the
    case table's note allows); the R3 control pins the preserved
    declared-declaration semantics the case table names (the row itself only
    becomes observable against a frame via the precedence test); R4 pins the
    unchanged turn_archive behavior — any fix that wrongly stamps
    turn_archive entries with the frame id turns it red.
"""

from __future__ import annotations

from pathlib import Path

from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
    TransformerConfig,
)
from sr2.models import Message, TextBlock
from sr2.pipeline.provenance import (
    TURN_ARCHIVE_LAYER,
    Entry,
    InMemoryProvenanceStore,
    read_turn_archive,
)
from sr2.pipeline.token_counting import CharacterTokenCounter

FRAME = "zulip:miranda:realm#stream#topic"
ORIGIN = "zulip"
SR2_SESSION_ID = "01ARCHIVISTSESSION0000000000"
DECLARED_SESSION = "declared-session-42"

# Layers written by the pipeline during the scripted runs below. Anything
# else in a store (e.g. post-process "memory" entries) is out of scope for
# the non-turn_archive assertions of this file.
# Entries persist with the RESOLVER's source_layer (input → "input",
# static_text → "static_text", mode guidance → "mode_guidance", ...), not
# the config layer name. Anything outside this set (e.g. post-process
# "memory" entries) is out of scope for this file's assertions.
EXPECTED_LAYERS = {
    "static_text",
    "session",
    "input",
    "markdown_file",
    "mode_guidance",
    "topic_area",
    "declared",
    "undeclared",
}

# Run context for the multi-kind turn: an area with an eligible topic under
# the fixture topics_root, and a known mode for the guidance layer.
RUN_CONTEXT = {"area": "frame-attribution", "mode": "interactive"}


def _is_run_entry(entry: Entry) -> bool:
    """Entry persisted by this test run (test-created layers only)."""
    return entry.layer in EXPECTED_LAYERS


def _is_run_entry_by_layer(layer: str) -> bool:
    """Same filter for raw SQLite rows."""
    return layer in EXPECTED_LAYERS


# ---------------------------------------------------------------------------
# Raw-content stub resolver (drives layer synthesis of provenance entries)
# ---------------------------------------------------------------------------


class RawResolver:
    """Resolver emitting raw content (optionally with a provenance declaration).

    Stands in for raw resolvers whose entries the layer synthesizes at the
    store boundary (sr2-92 — no raw-content/no-store path). Used by R3,
    where a provenance declaration carrying an explicit ``session_id`` must
    be exercised — a shape no production resolver emits today. Fires on
    ``turn_start``.
    """

    def __init__(
        self,
        name: str,
        source_layer: str,
        content: list,
        provenance: list[dict] | None = None,
    ) -> None:
        from sr2.pipeline.events import EventPhase, EventSubscription

        self.name = name
        self.max_executions: int = 1
        self.execution_count: int = 0
        self.subscriptions = [
            EventSubscription(event_name="turn_start", phase=EventPhase.COMPLETED)
        ]
        self._source_layer = source_layer
        self._content = list(content)
        self._provenance = provenance

    async def resolve(self, events) -> "ResolvedContent":
        from sr2.pipeline.models import ResolvedContent

        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer=self._source_layer,
            content=list(self._content),
            provenance=self._provenance,
        )


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


async def _sqlite_store(tmp_path, name: str):
    from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

    db_path = Path(tmp_path) / name
    store = SQLiteProvenanceStore(db_path=str(db_path))
    await store.connect()
    return store


async def _sqlite_rows(store) -> list[dict]:
    """Raw persisted rows of a connected SQLiteProvenanceStore.

    Read as plain rows (not Entry objects) so the assertions observe exactly
    what was persisted — including rows whose source references exist only
    inside the store's own entry_sources table.
    """
    db = store._db
    assert db is not None, "store must be connected"
    cursor = await db.execute(
        "SELECT id, session_id, layer, origin_kind, origin_name FROM entries"
    )
    cursor.row_factory = None
    rows = await cursor.fetchall()
    return [
        {"id": r[0], "session_id": r[1], "layer": r[2], "origin_kind": r[3],
         "origin_name": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# SR2 orchestrator helpers
# ---------------------------------------------------------------------------


def _make_pipeline_config() -> PipelineConfig:
    """Config whose layers exercise config-text, session and input kinds."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static_text",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[
                    ResolverConfig(type="session"),
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(
                                event="user_input", phase="completed"
                            )
                        ],
                    ),
                ],
            ),
        ]
    )


async def _run_sr2_turn(
    store,
    *,
    active_frame_provider=None,
    config=None,
    run_context=None,
    after_build=None,
    seed=None,
    disable_post_process=False,
):
    """One scripted text turn through SR2 with the shared store.

    ``after_build`` (optional) receives the SR2 instance after
    construction, before the turn runs — used where a test swaps in stub
    resolvers. Returns the SR2 instance (already aclosed) so callers can
    read ``session_id`` and the engine turn sequence.
    """
    from conftest import MockLLM, make_user_input
    from sr2.orchestrator import SR2

    kwargs = {}
    if active_frame_provider is not None:
        kwargs["active_frame_provider"] = active_frame_provider
    if run_context is not None:
        kwargs["run_context_provider"] = lambda: dict(run_context)
    sr2 = SR2(
        pipeline_config=config if config is not None else _make_pipeline_config(),
        llm={"default": MockLLM()},
        token_counter=CharacterTokenCounter(),
        session_id=SR2_SESSION_ID,
        provenance_store=store,
        **kwargs,
    )
    if after_build is not None:
        after_build(sr2)
    if disable_post_process:
        _no_extra_store_writes(sr2)
    if seed:
        sr2.seed_session(list(seed))
    events = [e async for e in sr2.turn(make_user_input("hello there"), origin=ORIGIN)]
    assert any(e.type == "end" for e in events), "turn must complete"
    await sr2.aclose()
    return sr2


# ===========================================================================
# R1 — framed turn: non-turn_archive entries carry the frame id
# ===========================================================================


class TestR1FramedTurnStampsFrameId:
    async def test_orchestrator_framed_turn_entries_carry_frame_id(self):
        """R1: a turn compiled while an active-frame provider resolves a
        frame id for the turn's origin persists every newly persisted
        non-turn_archive entry with session_id == the frame id."""
        store = InMemoryProvenanceStore()
        await _run_sr2_turn(store, active_frame_provider=lambda origin: FRAME)

        entries = [
            e
            for e in store._store.values()
            if _is_run_entry(e) and e.layer != TURN_ARCHIVE_LAYER
        ]
        # The row's outcome is per-entry: every new non-turn_archive entry
        # carries the frame id. (Layer-kind coverage is the R5 row's job;
        # this fixture always yields at least the static_text and input
        # layers' entries, so entries is non-empty.)
        assert entries, "the framed turn must persist provenance entries"
        for entry in entries:
            assert entry.session_id == FRAME, (
                f"entry in layer {entry.layer!r} carries "
                f"session_id={entry.session_id!r}; expected the active frame "
                f"id {FRAME!r}, not a fallback"
            )

    async def test_framed_turn_entries_persist_frame_id_in_sqlite(self, tmp_path):
        """R1 on the durable store: the frame id is what lands in SQLite."""
        store = await _sqlite_store(tmp_path, "r1.db")
        try:
            await _run_sr2_turn(store, active_frame_provider=lambda origin: FRAME)
            rows = [
                r
                for r in await _sqlite_rows(store)
                if _is_run_entry_by_layer(r["layer"])
                and r["layer"] != TURN_ARCHIVE_LAYER
            ]
            assert rows, "the framed turn must persist provenance entries"
            for row in rows:
                assert row["session_id"] == FRAME, (
                    f"layer {row['layer']!r} persisted session_id="
                    f"{row['session_id']!r}; expected the active frame id"
                )
        finally:
            await store.close()


# ===========================================================================
# R2 — no session frame: entries keep "default"
# ===========================================================================


class TestR2NoFrameKeepsDefault:
    async def test_no_provider_entries_stay_default(self):
        """R2 (no provider configured — headless/CLI): persisted entries keep
        session_id == "default" (unchanged fallback)."""
        store = InMemoryProvenanceStore()
        await _run_sr2_turn(store)

        entries = [
            e
            for e in store._store.values()
            if _is_run_entry(e) and e.layer != TURN_ARCHIVE_LAYER
        ]
        assert entries, "the turn must persist provenance entries"
        for entry in entries:
            assert entry.session_id == "default", (
                f"entry in layer {entry.layer!r} carries "
                f"session_id={entry.session_id!r}; without a session frame "
                f"the fallback must remain 'default'"
            )

    async def test_provider_returning_none_entries_stay_default(self):
        """R2 (provider returns None — no frame open for this origin):
        persisted entries keep session_id == "default"."""
        store = InMemoryProvenanceStore()
        await _run_sr2_turn(store, active_frame_provider=lambda origin: None)

        entries = [
            e
            for e in store._store.values()
            if _is_run_entry(e) and e.layer != TURN_ARCHIVE_LAYER
        ]
        assert entries, "the turn must persist provenance entries"
        for entry in entries:
            assert entry.session_id == "default", (
                f"entry in layer {entry.layer!r} carries "
                f"session_id={entry.session_id!r}; a provider resolving no "
                f"frame must leave the fallback at 'default'"
            )


# ===========================================================================
# R3 — declared session_id wins over the active frame id
# ===========================================================================


def _declared_pair_config() -> PipelineConfig:
    """One layer whose placeholder resolver R3 replaces with two stub
    resolvers (declared declaration vs undeclared), so the declaration
    channel of the layer's entry construction (layer.py) is exercised."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="declared",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static_text", config={"text": "placeholder-d"}
                    )
                ],
            ),
        ]
    )


def _inject_stub_resolvers(sr2) -> None:
    """Replace the placeholders with the R3 stub resolvers.

    The engine wired subscriptions for the ``turn_start`` event already at
    construction; the stubs subscribe to the same event, so no rewiring is
    needed (re-subscribing would double-deliver events — see the
    ``turn_start`` duplication comment in ``PipelineEngine._wire_layers``).
    Component identity is not pinned; only persisted Entries are asserted.
    """
    layers = sr2._engine.layers
    layers[0].resolvers = [
        RawResolver(
            "declared_resolver",
            "declared",
            [TextBlock(text="declared item")],
            provenance=[{"session_id": DECLARED_SESSION}],
        ),
        RawResolver(
            "plain_resolver", "undeclared", [TextBlock(text="undeclared item")]
        ),
    ]


class TestR3DeclaredSessionIdWins:
    async def test_declared_session_id_wins(self):
        """R3 (preserved semantics), per item, no frame: the entry built from
        the declaration carrying an explicit session_id keeps its declared
        id — the declaration wins over the no-frame fallback — while the
        undeclared sibling entry keeps the 'default' fallback (R2's
        territory)."""
        store = InMemoryProvenanceStore()
        await _run_sr2_turn(
            store, config=_declared_pair_config(), after_build=_inject_stub_resolvers
        )

        by_text = {
            e.content.text: e
            for e in store._store.values()
            if _is_run_entry(e)
            and e.layer != TURN_ARCHIVE_LAYER
            and isinstance(e.content, TextBlock)
        }
        assert by_text, "the compiled turn must persist provenance entries"
        declared = by_text.get("declared item")
        assert declared is not None, "the declared item's entry is missing"
        assert declared.session_id == DECLARED_SESSION, (
            "the declared session_id must win over the no-frame fallback; "
            f"got {declared.session_id!r}"
        )
        sibling = by_text.get("undeclared item")
        assert sibling is not None, "the undeclared sibling's entry is missing"
        assert sibling.session_id == "default", (
            "without a frame the undeclared entry keeps the 'default' "
            f"fallback; got {sibling.session_id!r}"
        )

    async def test_declared_item_wins_and_sibling_takes_frame(self):
        """R3 precedence: on a framed turn, an item whose declaration carries
        an explicit session_id persists that declared id while an
        undeclared sibling entry takes the frame id — declared > frame,
        'nothing else changes'. (The sibling assertion is what makes the
        row observable: today there is no frame to beat.)"""
        store = InMemoryProvenanceStore()
        await _run_sr2_turn(
            store,
            config=_declared_pair_config(),
            active_frame_provider=lambda origin: FRAME,
            after_build=_inject_stub_resolvers,
        )

        by_text = {
            e.content.text: e
            for e in store._store.values()
            if _is_run_entry(e)
            and isinstance(e.content, TextBlock)
            and e.content.text in {"declared item", "undeclared item"}
        }
        assert set(by_text) == {"declared item", "undeclared item"}, (
            "both stub items must persist one entry each"
        )
        assert by_text["declared item"].session_id == DECLARED_SESSION, (
            "the declared session_id must win over the active frame id; got "
            f"{by_text['declared item'].session_id!r}"
        )
        assert by_text["undeclared item"].session_id == FRAME, (
            "an entry without a declared session_id must take the active "
            f"frame id; got {by_text['undeclared item'].session_id!r}"
        )


# ===========================================================================
# R4 — turn_archive behavior unchanged under an active frame
# ===========================================================================


class TestR4TurnArchiveUnchanged:
    async def test_turn_archive_keeps_archivist_session_and_seq(self):
        """R4: with a frame active for the turn's origin, turn_archive
        entries keep session_id == the archivist's own session id and their
        existing meta turn_seq/seq behavior (read back via
        read_turn_archive). Regression guard: a fix that stamps turn_archive
        with the frame id turns this red."""
        store = InMemoryProvenanceStore()
        sr2 = await _run_sr2_turn(store, active_frame_provider=lambda origin: FRAME)
        turn_seq = sr2._engine.turn_seq

        archived = await read_turn_archive(store, sr2.session_id, turn_seq)
        assert archived, "the turn must archive at least the user input"
        seqs = [e.meta["seq"] for e in archived]
        assert seqs == list(range(len(archived))), (
            f"per-turn meta['seq'] must remain monotonic from 0; got {seqs!r}"
        )
        for entry in archived:
            assert entry.layer == TURN_ARCHIVE_LAYER
            assert entry.session_id == SR2_SESSION_ID, (
                "turn_archive entries must keep the archivist's own session "
                f"id {SR2_SESSION_ID!r}; got {entry.session_id!r}"
            )
            assert entry.meta["turn_seq"] == turn_seq
        first = archived[0]
        assert isinstance(first.content, Message) and first.content.role == "user"


# ===========================================================================
# R5 — every non-turn_archive layer kind carries the frame id
# ===========================================================================


_MULTI_KIND_SET = {
    "static_text",
    "session",
    "input",
    "markdown_file",
    "mode_guidance",
    "topic_area",
}


def _no_extra_store_writes(sr2) -> None:
    """Neutralize the deferred post-process for a deterministic fixture.

    The shared provenance store would otherwise receive post-process writes
    (memory extraction etc.) whose presence and ids are not part of any
    case-table row, making the persisted-set equality checks flaky. The
    turn's own layer + turn_archive writes are unaffected.
    """

    async def _noop(*args, **kwargs) -> None:
        return None

    sr2.post_process = _noop  # type: ignore[method-assign]


def _make_multi_kind_config(tmp_path) -> PipelineConfig:
    """Six layers, one per distinct non-turn_archive kind named by the case
    table, using the production resolvers themselves: static/config text
    (static_text), the session layer, user input (input), markdown-style
    file content (markdown_file over a fixture file), and the mode_guidance
    / topic_area guidance layers (the spctr resolvers the live pipeline
    runs). Content is unique per layer; entry ids are content-addressed, so
    no collisions across layers.

    The session-kind layer is driven through a SessionResolver seed plus the
    production session resolver: on the turn's FIRST firing the resolver
    emits the prior history, so seeding makes the "session" layer kind
    contribute deterministically in a single-turn fixture."""
    md_dir = tmp_path / "mdsrc"
    md_dir.mkdir(parents=True, exist_ok=True)
    (md_dir / "NOTES.md").write_text("# Notes\nmarkdown file style body\n")

    # Eligible topic for the run context's area (RUN_CONTEXT["area"]):
    # README.md carries the topic identity metadata; the requested
    # AGENTS.md carries the injected body.
    topic = tmp_path / "topics" / "provenance-topic"
    topic.mkdir(parents=True, exist_ok=True)
    (topic / "README.md").write_text(
        '---\nkind: topic\nid: "topic:frame-attribution"\n---\n\n# Topic\n'
    )
    (topic / "AGENTS.md").write_text("topic area guidance document body\n")
    # A NOW.md continuity marker is required for the topic to be eligible.
    (topic / "NOW.md").write_text(
        '---\nkind: continuity\nfor: "topic:frame-attribution"\nmode: source\n---\n\n# Now\n'
    )

    return PipelineConfig(
        layers=[
            LayerConfig(
                name="static_text",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static_text",
                        config={"text": "static config guidance for the frame"},
                    )
                ],
            ),
            LayerConfig(
                name="session",
                target="messages",
                resolvers=[ResolverConfig(type="session")],
            ),
            LayerConfig(
                name="input",
                target="messages",
                resolvers=[
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(
                                event="user_input", phase="completed"
                            )
                        ],
                    )
                ],
            ),
            LayerConfig(
                name="markdown_file",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="markdown_file",
                        config={"path": "NOTES.md", "declaring_dir": str(md_dir)},
                    )
                ],
            ),
            LayerConfig(
                name="mode_guidance",
                target="system",
                resolvers=[ResolverConfig(type="mode_guidance")],
            ),
            LayerConfig(
                name="topic_area",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="topic_area",
                        config={
                            "topics_root": str(tmp_path / "topics"),
                            "filename": "AGENTS.md",
                        },
                    )
                ],
            ),
        ]
    )


class TestR5AllLayerKindsStamped:
    async def test_distinct_layer_kinds_contribute_entries(self, tmp_path):
        """R5 contribution control (runs on every route): each distinct
        non-turn_archive layer kind contributes at least one persisted entry
        in one run (no kind is skipped or store-bypassed)."""
        store = InMemoryProvenanceStore()
        await _run_sr2_turn(
            store,
            config=_make_multi_kind_config(tmp_path),
            run_context=RUN_CONTEXT,
            seed=[Message(role="user", content=[TextBlock(text="seeded prior turn")])],
            disable_post_process=True,
        )

        layers_seen = {
            e.layer
            for e in store._store.values()
            if _is_run_entry(e) and e.layer != TURN_ARCHIVE_LAYER
        }
        assert layers_seen == _MULTI_KIND_SET, (
            "each distinct layer kind must contribute at least one entry; "
            f"seen {sorted(layers_seen)!r}"
        )

    async def test_distinct_layer_kinds_all_carry_frame_id(self, tmp_path):
        """R5: under an active frame, no layer kind falls back to "default" —
        all carry the active frame id."""
        store = InMemoryProvenanceStore()
        await _run_sr2_turn(
            store,
            config=_make_multi_kind_config(tmp_path),
            active_frame_provider=lambda origin: FRAME,
            run_context=RUN_CONTEXT,
            seed=[Message(role="user", content=[TextBlock(text="seeded prior turn")])],
            disable_post_process=True,
        )

        entries = [
            e
            for e in store._store.values()
            if _is_run_entry(e) and e.layer != TURN_ARCHIVE_LAYER
        ]
        layers_seen = {e.layer for e in entries}
        assert layers_seen == _MULTI_KIND_SET, (
            "each distinct layer kind must contribute at least one entry; "
            f"seen {sorted(layers_seen)!r}"
        )
        for entry in entries:
            assert entry.session_id == FRAME, (
                f"layer kind {entry.layer!r} fell back to "
                f"session_id={entry.session_id!r}; expected the active frame "
                f"id {FRAME!r}"
            )

    async def test_mixed_layer_kinds_persist_frame_id_in_sqlite(self, tmp_path):
        """R5 on the durable store: every layer kind persists the frame id
        through SQLite."""
        store = await _sqlite_store(tmp_path, "r5.db")
        try:
            await _run_sr2_turn(
                store,
                config=_make_multi_kind_config(tmp_path),
                active_frame_provider=lambda origin: FRAME,
                run_context=RUN_CONTEXT,
                seed=[
                    Message(role="user", content=[TextBlock(text="seeded prior turn")])
                ],
                disable_post_process=True,
            )
            rows = [
                r
                for r in await _sqlite_rows(store)
                if _is_run_entry_by_layer(r["layer"])
                and r["layer"] != TURN_ARCHIVE_LAYER
            ]
            assert {r["layer"] for r in rows} == _MULTI_KIND_SET
            for row in rows:
                assert row["session_id"] == FRAME, (
                    f"layer kind {row['layer']!r} persisted "
                    f"session_id={row['session_id']!r} in SQLite; expected "
                    f"the active frame id"
                )
        finally:
            await store.close()
