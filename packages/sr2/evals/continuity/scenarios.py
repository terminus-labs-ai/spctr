"""Versioned benchmark corpus for the continuity baseline (sr2-91 / AC16).

Fixed inputs and fixed budgets for the ten FR31 scenarios plus the binding
composition fixture pinned by the AC16 cage (``cage-composition``). All
content is literal and deterministic: no clocks, no randomness, no I/O.

Every retrieval turn declares its machine-readable gold per the Benchmark
Metric Contract: exact expected answer strings, current vs superseded
values, policy-tagged blocks (must never reach a restricted agent) and
distractor-tagged blocks. Superseded values carry an older declared
``observed_at`` than their current replacements — the recency inputs the
perfect-reader double resolves conflicts with (stale reads stay measurable
without changing the pinned aggregate outcomes). The degraded-mode
scenarios (``restart``, ``continuity-service-failure``) replay the full
history, matching current behavior.
"""

from __future__ import annotations

from sr2.continuity.consolidation import _content_words

from evals.continuity.models import Block, Scenario, ScenarioTurn

_REPLAY = "continuity_replay"

#: The FR40 scenario id: A (a settled backend decision), its finishing turn,
#: B (an unrelated trip), then a paraphrased reference to A sharing no
#: content word with A's records. Its gold labels declare the gold close
#: turn (``gold_close_turn``) and A's decision string as the expected answer
#: on the reference turn. The recorded embedding vectors for its inputs are
#: committed under ``fixtures/episode_close_paraphrase_return/``.
EPISODE_CLOSE_PARAPHRASE_RETURN = "episode_close_paraphrase_return"

#: Tag vocabularies. Tag values never reach the delivered answer text —
#: tags drive policy-leak / irrelevant-context attribution only.
POLICY_TAG = "restricted-from-agent"
DISTRACTOR_TAG = "noise"

#: Declared sourcing times (recency contract): supersession happened here.
EARLY_AT = "2026-06-01T00:00:00Z"
LATE_AT = "2026-08-01T00:00:00Z"


def _overheard(secret: str) -> Block:
    """A policy-tagged block: a personal fact that must stay withheld."""
    return Block(text=f"overheard: {secret}", policy_tag=POLICY_TAG,
                 layer=_REPLAY, observed_at=LATE_AT)


def _noise(text: str) -> Block:
    """A distractor-tagged block: irrelevant restored context."""
    return Block(text=text, distractor_tag=DISTRACTOR_TAG,
                 layer=_REPLAY, observed_at=LATE_AT)


def _old(text: str) -> Block:
    """Plain restored context, observed before the current values."""
    return Block(text=text, layer=_REPLAY, observed_at=EARLY_AT)


def _scenario(
    sid: str,
    turns: list[ScenarioTurn],
    *,
    token_budget: int = 4000,
    prompt_budget: int = 1000,
    agent_policy: str = "restricted",
    runner_mode: str = "full_history",
) -> Scenario:
    return Scenario(
        id=sid,
        turns=turns,
        token_budget=token_budget,
        prompt_budget=prompt_budget,
        agent_policy=agent_policy,
        runner_mode=runner_mode,
    )


def _turn(
    question: str,
    gold: tuple[str, ...],
    current: tuple[str, ...],
    superseded: tuple[str, ...] = (),
    *,
    extras: tuple[Block, ...] = (),
    delivered: tuple[str, ...] | None = None,
    meta: dict | None = None,
) -> ScenarioTurn:
    """One retrieval turn: its question plus its gold declaration.

    ``extras`` carries tagged restored-context blocks; the current and
    superseded value blocks are appended in observation order
    (superseded-then-current, newest source last). ``meta`` adds
    scenario-specific declarations (e.g. the FR40 gold close turn) on top
    of the default ``{"question": ...}``.
    """
    blocks: list[Block] = [_old(v) for v in superseded]
    blocks += [Block(text=v, layer=_REPLAY) for v in current]
    blocks += list(extras)
    turn_meta = {"question": question}
    if meta:
        turn_meta.update(meta)
    return ScenarioTurn(
        gold_answers=gold,
        current_values=current,
        superseded_values=superseded,
        blocks=blocks,
        delivered=delivered,
        meta=turn_meta,
    )


# ---------------------------------------------------------------------------
# The corpus — ten FR31 scenarios + the pinned composition fixture.
# ---------------------------------------------------------------------------

_SCENARIOS: list[Scenario] = [
    # 1. Project resumption with intervening live changes.
    _scenario("project-resumption-intervening-live-changes", [
        _turn(
            "Which database backend is selected?",
            gold=("database backend: managed PostgreSQL",),
            current=("database backend: managed PostgreSQL",),
            superseded=("database backend: sqlite",),
            extras=(
                _noise("unrelated checkout flow chatter from the sprint room"),
                _overheard("favorite snack: salted licorice"),
            ),
        ),
    ]),
    # 2. Long topic switching and return.
    _scenario("long-topic-switching-and-return", [
        _turn(
            "After the topic switches, which release train are we on?",
            gold=("release train: biweekly Thursday",),
            current=("release train: biweekly Thursday",),
            superseded=("release train: monthly first Monday",),
            extras=(
                _noise("tangents covered: kitchen tiles, kayaking trips"),
                _overheard("dream: flying over a lavender field"),
            ),
        ),
    ]),
    # 3. Personal preference across projects.
    _scenario("personal-preference-across-projects", [
        _turn(
            "Which test runner preference applies to the data tool?",
            gold=("test runner preference: pytest",),
            current=("test runner preference: pytest",),
            superseded=("test runner preference: unittest",),
            extras=(
                _noise("shopping list: oat milk, mustard, flax seeds"),
                _overheard("restored hobby: watercolor painting"),
            ),
        ),
    ]),
    # 4. Per-agent policy isolation.
    _scenario("per-agent-policy-isolation", [
        _turn(
            "Which drafting voice applies to the drafting agent?",
            gold=("drafting voice: formal",),
            current=("drafting voice: formal",),
            superseded=("drafting voice: casual",),
            extras=(
                _overheard("private note for the coding agent: window seat"),
                _noise("background hum about staplers and paperclips"),
            ),
        ),
    ]),
    # 5. Correction / supersession.
    _scenario("correction-supersession", [
        _turn(
            "Which retry interval stands after the correction?",
            gold=("retry interval: 30 seconds",),
            current=("retry interval: 30 seconds",),
            superseded=("retry interval: 5 seconds",),
            extras=(
                _noise("static about warehouse shelving and pallets"),
                _overheard("favorite podcast: true crime midnight"),
            ),
        ),
    ]),
    # 6. Cross-project dependency.
    _scenario("cross-project-dependency", [
        _turn(
            "Which contract version does the relay depend on?",
            gold=("relay contract version: v3",),
            current=("relay contract version: v3",),
            superseded=("relay contract version: v2",),
            extras=(
                _noise("gossip about the office espresso machine"),
                _overheard("birthday gift idea: wool scarf"),
            ),
        ),
    ]),
    # 7. Relevant episodic recovery.
    _scenario("relevant-episodic-recovery", [
        _turn(
            "Which recovery plan did the outage episode end with?",
            gold=("recovery plan: snapshot restore",),
            current=("recovery plan: snapshot restore",),
            superseded=("recovery plan: manual replay",),
            extras=(
                _noise("episode filler about parking permits and umbrellas"),
                _overheard("comfort food after incidents: chicken soup"),
            ),
        ),
    ]),
    # 8. Bounded-context endurance (500-turn scenario, fixed inputs).
    _scenario("bounded-context-endurance-500-turn", [
        _turn(
            "After hundreds of turns, which scaling decision stands?",
            gold=("scaling decision: vertical first",),
            current=("scaling decision: vertical first",),
            superseded=("scaling decision: horizontal first",),
            extras=tuple(
                _noise(f"filler passage {i}: idle chatter segment")
                for i in range(24)
            ) + (_overheard("standing lunch order: lentil soup"),),
        ),
    ], token_budget=8000),
    # 9. Restart — degraded service mode; replay restores the full history.
    _scenario("restart", [
        _turn(
            "After the restart, which session token policy is in force?",
            gold=("session token policy: sliding expiry",),
            current=("session token policy: sliding expiry",),
            superseded=("session token policy: fixed expiry",),
            extras=(
                _noise("startup log noise about daemon flags"),
                _overheard("personal holiday plan: cabin by the lake"),
            ),
        ),
    ], runner_mode="degraded_service"),
    # 10. Continuity-service failure — degraded service mode.
    _scenario("continuity-service-failure", [
        _turn(
            "With the continuity service down, which escalation route holds?",
            gold=("escalation route: pager rotation",),
            current=("escalation route: pager rotation",),
            superseded=("escalation route: email queue",),
            extras=(
                _noise("static about garden gnomes and hose adapters"),
                _overheard("favorite board game: stone harvest"),
            ),
        ),
    ], runner_mode="degraded_service"),
    # Binding composition fixture pinned by the AC16 cage (module docstring
    # of tests/evals/test_continuity_baseline.py). Exact content — the
    # pinned strings, tags, budgets and deliveries; do not edit without a
    # test proposal.
    _scenario("cage-composition", [
        _turn(
            "Which database backend is selected?",
            gold=("backend: PostgreSQL",),
            current=("backend: PostgreSQL",),
            superseded=("backend: SQLite",),
            extras=(
                Block(text="favorite food: vanilla ice cream",
                      policy_tag=POLICY_TAG, layer=_REPLAY),
                Block(text="alpha beta gamma delta epsilon zeta",
                      distractor_tag=DISTRACTOR_TAG, layer=_REPLAY),
            ),
            delivered=None,
        ),
        _turn(
            "Which database backend is selected?",
            gold=("backend: PostgreSQL",),
            current=("backend: PostgreSQL",),
            superseded=("backend: SQLite",),
            extras=(
                Block(text="favorite food: vanilla ice cream",
                      policy_tag=POLICY_TAG, layer=_REPLAY),
                Block(text="alpha beta gamma delta epsilon zeta",
                      distractor_tag=DISTRACTOR_TAG, layer=_REPLAY),
            ),
            delivered=("backend: PostgreSQL",),
        ),
    ]),
    # FR40 — paraphrase-return scenario (sr2-163 / AC27). Seven turns:
    #   A (turns 1-3): a settled storage-backend decision — the decision
    #     string "Postgres" is settled in the A record's text;
    #   the finishing turn (turn 4): the gold close (declared in meta
    #     ``gold_close_turn`` = 4); A's verbatim turns leave the window here;
    #   B (turns 5-6): an unrelated trip, sharing no content word with A;
    #   the reference turn (turn 7): a retrieval turn whose gold answer is A's
    #     decision string while its own text shares no content word with A's
    #     records — the paraphrased mention only the semantic (embedding)
    #     candidate can match.
    # A and B records are plain context blocks (no tags) so the double reads
    # them as facts; the reference turn declares no delivered values — its
    # decision string must reach the request through the index.
    _scenario(
        EPISODE_CLOSE_PARAPHRASE_RETURN,
        [
            _turn(
                "We settled the side project backend: Postgres, a managed "
                "instance with nightly backups.",
                gold=(),
                current=(),
                extras=(
                    Block(text="backend: Postgres", layer=_REPLAY,
                          observed_at=EARLY_AT),
                ),
            ),
            _turn(
                "That settles it: Postgres is the database for the side "
                "project; the storage layout can wait.",
                gold=(),
                current=(),
                extras=(
                    Block(text="database: Postgres", layer=_REPLAY,
                          observed_at=EARLY_AT),
                ),
            ),
            _turn(
                "Decided: Postgres for the side project, a managed instance; "
                "the storage layout can wait.",
                gold=(),
                current=(),
                extras=(
                    Block(text="decision: Postgres", layer=_REPLAY,
                          observed_at=LATE_AT),
                ),
            ),
            _turn(
                "Okay, that's settled. On to the next thing.",
                gold=(),
                current=(),
                meta={"gold_close_turn": 4,
                      "episode_title": "side project backend"},
            ),
            _turn(
                "Planning the trip: which flight window works best?",
                gold=(),
                current=(),
                extras=(
                    Block(text="flight window: midweek morning", layer=_REPLAY,
                          observed_at=EARLY_AT),
                ),
            ),
            _turn(
                "Booking the lodging near the airport, not the old town.",
                gold=(),
                current=(),
                extras=(
                    Block(text="lodging: airport side", layer=_REPLAY,
                          observed_at=EARLY_AT),
                ),
            ),
            # The reference turn declares A's decision as its gold answer and
            # current value. In the plain full-history baseline sweep (run_
            # baseline) that value travels as a delivered block; in the FR40
            # candidate run it re-enters through the selected index instead
            # (the paraphrase shares no content word with A's records, so the
            # lexical candidates cannot match it and only the embedding
            # candidate delivers it — AC27's point).
            _turn(
                "Why did we pick that option from before, again?",
                gold=("decision: Postgres",),
                current=("decision: Postgres",),
                superseded=("decision: SQLite",),
                extras=(
                    _overheard("favorite snack: salted licorice"),
                    _noise("unrelated chatter from the sprint room"),
                ),
            ),
        ],
    ),
]

#: The versioned corpus, keyed by scenario id (FR31 ids + composition).
CORPUS: dict[str, Scenario] = {s.id: s for s in _SCENARIOS}


def gold_turns(scenario: Scenario) -> list[ScenarioTurn]:
    """The retrieval turns of *scenario*: turns declaring gold answers."""
    return [t for t in scenario.turns if tuple(t.gold_answers)]


def _scenario_a_record_texts(scenario: Scenario) -> list[str]:
    """A's record texts for the FR40 content-word contract: every turn whose
    own text or blocks state the decision, plus the reference turn's blocks
    (what a candidate delivered to the reference, if the scenario declares
    any)."""
    texts: list[str] = []
    for turn in scenario.turns:
        own = _turn_text(turn)
        if "Postgres" in own:
            texts.append(own)
            texts.extend(_turn_block_texts(turn))
    return texts


def _turn_text(turn: ScenarioTurn) -> str:
    question = (turn.meta or {}).get("question")
    return question if isinstance(question, str) and question else ""


def _turn_block_texts(turn: ScenarioTurn) -> list[str]:
    return [b.text for b in turn.blocks
            if isinstance(b.text, str) and b.text]


def _reference_turns(scenario: Scenario) -> list[ScenarioTurn]:
    """The retrieval turns whose gold names A's decision while their own
    text does not — the paraphrased reference turns."""
    return [
        turn for turn in scenario.turns
        if turn.gold_answers
        and any("Postgres" in g for g in turn.gold_answers)
        and "Postgres" not in _turn_text(turn)
    ]


def validate_scenario(scenario: Scenario) -> list[str]:
    """Corpus validator (FR40).

    For a scenario that declares a paraphrased reference to a settled
    decision (a retrieval turn whose gold names the decision while its own
    text does not), prove the reference shares no content word with A's
    records: the paraphrase contract is that a semantic match — never a
    shared surface word — is what brings A back. Returns an empty list when
    the scenario satisfies the contract; on violation, one error naming each
    shared content word (the shared words are the exact overlap the mutation
    must surface). Scenarios with no such reference turn validate empty."""
    references = _reference_turns(scenario)
    if not references:
        return []
    a_words = set().union(*(_content_words(t) for t in
                            _scenario_a_record_texts(scenario)))
    errors: list[str] = []
    for turn in references:
        ref_words = set(_content_words(_turn_text(turn)))
        shared = sorted(a_words & ref_words)
        if shared:
            errors.append(
                f"{scenario.id}: the reference turn shares content word(s) "
                f"{shared} with A's records — the paraphrase must share no "
                "content word with A's records (FR40)"
            )
    return errors
