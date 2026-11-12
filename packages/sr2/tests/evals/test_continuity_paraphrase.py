"""sr2-163 / AC27 — paraphrase-return scenario in the continuity benchmark (unit layer).

Case-table rows AC27-r1 … AC27-r7 (Diego 2026-10-31; r6/r7 approved
2026-11-04, dec-383), transcribed from FR40 and the Benchmark Metric
Contract of topics/sr2/sr2/specs/autonomous-context-continuity.md.

RED MECHANISM
=============

Tests import their entry points locally with explicit failure messages, so
against the base tree each test fails for its own reason — no scenario
``episode_close_paraphrase_return`` in ``CORPUS``, no corpus validator, no
``closed_residency`` metric, no embedding candidate in the registry — not
by a module-level import error.

THE PINNED INTERFACE (delivered by C, ratified by B)
====================================================

Only the following names are pinned; everything else is free (class
shapes, fixture file layout, judge-double wiring, extra exports):

``evals.continuity.scenarios``
    ``CORPUS["episode_close_paraphrase_return"]`` — the FR40 scenario.
    Its gold labels declare A's decision string (containing
    ``DECISION_MARK``) as the expected answer on the reference turn.
    ``validate_scenario(scenario) -> list[str]``
        Corpus validator: an empty list when the scenario satisfies the
        content-word contract; on violation, errors naming each shared
        content word (FR40: "The corpus validator rejects the scenario if
        the reference shares a content word with A's records"). Content
        words are those of ``sr2.continuity.consolidation._content_words``
        (imported, not modified). A raising validator is also accepted.

``evals.continuity.runner``
    ``run_candidate(candidate_id: str, output_root: Path) -> Path``
        Deterministic run of one registry candidate over the corpus;
        publishes ``<dir>/observations.jsonl`` + ``metrics.json`` in the
        sr2-91 shape (records with ``scenario_id``, ``run_index``,
        ``turn_index``, ``flags``). The run uses the judge double that
        replays the fixture's gold close turn and performs no network
        access ("the judge is a double that replays the gold close turn").
    ``metrics.json`` publishes the new FR40 metric ``closed_residency``
        (A's verbatim-turn tokens delivered after the close turn; the
        target for this scenario is 0), and the per-scenario results of
        the other candidates stay published like every other scenario.

``evals.continuity.candidates``
    ``CANDIDATES`` — the candidate registry, extended with the embedding
    index under a key containing ``"embedding"``.

Fixture observation (rows r6, r7) — discovered, not pinned: the gold close
turn and the vector fixture (named model, per-input captured vectors,
retained capture evidence) are located under
``evals/continuity/fixtures/episode_close_paraphrase_return/`` via any
public loader found on the eval modules (callable, name containing
"fixture" or "vector") or via ``*.json`` data files in that directory.
The layout is C's; only the observations below are asserted.

Rows: r1 validator pass + mutation naming ``storage``; r2 decision string
in the reference turn's delivered request; r3 ``closed_residency == 0``;
r4 TF-IDF + structured per-scenario results published unchanged; r5 two
consecutive runs byte-identical ``metrics.json``; r6 judge double replays
each run's declared gold close turn with live judge access disabled; r7
every embedding input receives its matching committed vector (model
identity + capture-evidence association checked) with no live embedding
call.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

# ---------------------------------------------------------------------------
# Pinned vocabulary
# ---------------------------------------------------------------------------

SCENARIO_ID = "episode_close_paraphrase_return"
METRIC_KEY = "closed_residency"
EMBEDDING_KEY_HINT = "embedding"

#: A's decision string (row r2: delivered in the reference turn's request).
DECISION_MARK = "Postgres"
#: The word row r1's mutated reference shares with A's records.
MUTATION_WORD = "storage"

#: The scenario-reading convention this cage pins (mirroring the corpus's
#: existing ``_turn`` helper: every retrieval turn declares its question in
#: ``meta["question"]``):
#:   * every turn's text is ``meta["question"]`` (plus any block texts);
#:   * A's records are the turns whose ``meta["question"]`` contains
#:     DECISION_MARK — A's decision is settled in their text;
#:   * the reference turn is the retrieval turn (gold_answers non-empty)
#:     whose gold answer contains DECISION_MARK while its own text does
#:     not — the paraphrased mention of A that names no content word of
#:     A's records; row r2 asserts DECISION_MARK reaches that turn's
#:     delivered request through the index, never through its own text;
#:   * row r1's mutation injects the bare token MUTATION_WORD into the
#:     reference turn's text.
REFERENCE_READS_GOLD = "gold answer contains the decision mark"

_LOOPBACK = re.compile(r"(127\.0\.0\.\.|::1|localhost)", re.IGNORECASE)


class _BlockedNetwork(RuntimeError):
    pass


class _NetworkBlocker:
    """Block every outbound socket except loopback, recording external
    attempts. Loopback stays open because pytest itself uses it; "live
    access disabled" in the rows means the recorded embedding/judge
    endpoint and anything else off-machine. The event loop is untouched
    (asyncio's self-pipe is socketpair-based, not connect-based)."""

    def __init__(self) -> None:
        self.external: list[Any] = []
        self._real: tuple = ()

    def __enter__(self) -> Self:
        real = (socket.socket, socket.create_connection, socket.getaddrinfo)
        self._real = real
        attempts = self.external

        def _guard_host(host):
            if not _LOOPBACK.search(str(host)):
                attempts.append(host)
                raise _BlockedNetwork(
                    f"live network access is disabled (attempted {host!r})"
                )

        def _no_socket(*args, **kwargs):
            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, tuple) and arg:
                    _guard_host(arg[0])
                    return real[0](*args, **kwargs)
                if isinstance(arg, (bytes, bytearray)) and len(arg) in (4, 16):
                    attempts.append("<packed address>")
                    raise _BlockedNetwork("live socket connect is disabled")
            # No connect address given (plain construction): allowed —
            # without connect() no packet can leave.
            return real[0](*args, **kwargs)

        def _guarded_connection(address, *a, **k):
            _guard_host(address[0] if isinstance(address, tuple) else address)
            return real[1](address, *a, **k)

        def _guarded_getaddrinfo(host, *a, **k):
            _guard_host(host)
            return real[2](host, *a, **k)

        socket.socket = _no_socket  # type: ignore[misc]
        socket.create_connection = _guarded_connection  # type: ignore[assignment]
        socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        socket.socket, socket.create_connection, socket.getaddrinfo = self._real


# ---------------------------------------------------------------------------
# Direct helpers — explicit failure messages, structural discovery
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _mod(name: str):
    return importlib.import_module(name)


def _attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AssertionError(
        f"expected attribute {names!r} on {obj!r} — implementation missing"
    )


def _get_scenario():
    scenarios = _mod("evals.continuity.scenarios")
    corpus = _attr(scenarios, "CORPUS")
    _require(isinstance(corpus, dict) and bool(corpus),
             "CORPUS must be a non-empty dict keyed by scenario id")
    scenario = corpus.get(SCENARIO_ID)
    _require(
        scenario is not None,
        f"FR40 scenario {SCENARIO_ID!r} missing from CORPUS "
        f"(ids: {sorted(corpus)})",
    )
    return scenarios, scenario


def _validator(scenarios) -> Callable:
    for name in ("validate_scenario", "validate_corpus", "validate"):
        candidate = getattr(scenarios, name, None)
        if callable(candidate):
            return candidate
    raise AssertionError(
        "evals.continuity.scenarios exposes no corpus validator "
        "(validate_scenario / validate_corpus / validate)"
    )


_TURN_TEXT_ATTRS = ("question", "input", "user_text", "prompt", "text")


def _turn_text(turn: Any) -> str:
    """The turn's own delivered text: meta['question'] (the pinned
    convention) else the first question/input/text attribute."""
    meta = getattr(turn, "meta", None)
    if isinstance(meta, dict):
        value = meta.get("question")
        if isinstance(value, str) and value:
            return value
    for name in _TURN_TEXT_ATTRS:
        value = getattr(turn, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _turn_block_texts(turn: Any) -> list[str]:
    return [b.text for b in (getattr(turn, "blocks", ()) or ())
            if isinstance(getattr(b, "text", None), str) and b.text]


def _scenario_turns(scenario: Any) -> list[Any]:
    turns = list(getattr(scenario, "turns", ()) or ())
    if any(getattr(t, "turns", None) is not None for t in turns):  # nested groups
        flat: list[Any] = []
        for group in turns:
            inner = getattr(group, "turns", None)
            flat.extend(inner if inner is not None else [group])
        return flat
    return turns


def _read_scenario_turns(scenario: Any) -> tuple[list[str], list[str]]:
    """(A's records' texts, the reference turn's text) per the module
    docstring's pinned reading. Returns text — blocks and all — for
    membership checks, plus the reference turn's own text."""
    turns = _scenario_turns(scenario)
    _require(bool(turns), f"scenario {SCENARIO_ID!r} declares no turns")
    a_records: list[str] = []
    reference: list[str] = []
    for turn in turns:
        own = _turn_text(turn)
        gold = " ".join(getattr(turn, "gold_answers", ()) or ())
        if DECISION_MARK in own:
            a_records.extend([own] + _turn_block_texts(turn))
        elif gold and DECISION_MARK in gold:
            _require(
                own != "",
                "a reference turn (gold declares the decision, its text "
                "does not name it) has no text to mutate or observe",
            )
            reference.append(own)
    _require(
        bool(a_records),
        f"A's records (turn texts stating the decision, containing "
        f"{DECISION_MARK!r}) are absent from {SCENARIO_ID!r}",
    )
    _require(
        bool(reference),
        f"the reference turn (a retrieval turn whose gold answer contains "
        f"{DECISION_MARK!r} while its own text names no decision) is absent "
        f"from {SCENARIO_ID!r}",
    )
    return a_records, reference


def _mutation_candidates(scenario: Any) -> list[Any]:
    """Structural copies of *scenario* in which MUTATION_WORD is injected
    into a reference turn's text (dataclasses.replace — the committed copy
    is never mutated). Covers meta-declared questions, question/input
    attributes, block texts, and nested topic groups."""

    def _variants_for_turn(turn: Any) -> list[Any]:
        out: list[Any] = []
        meta = getattr(turn, "meta", None)
        if isinstance(meta, dict) and isinstance(meta.get("question"), str):
            injected = dict(meta)
            injected["question"] = f"{meta['question']} {MUTATION_WORD}"
            try:
                out.append(dataclasses.replace(turn, meta=injected))
            except TypeError:
                pass
        for name in _TURN_TEXT_ATTRS:
            value = getattr(turn, name, None)
            if isinstance(value, str) and value:
                try:
                    out.append(dataclasses.replace(
                        turn, **{name: f"{value} {MUTATION_WORD}"}))
                except TypeError:
                    pass
                break
        blocks = list(getattr(turn, "blocks", ()) or ())
        for i, block in enumerate(blocks):
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                try:
                    mutated = list(blocks)
                    mutated[i] = dataclasses.replace(
                        block, text=f"{text} {MUTATION_WORD}")
                    out.append(dataclasses.replace(turn, blocks=mutated))
                except TypeError:
                    pass
                break
        return out

    def _is_reference(turn: Any) -> bool:
        own = _turn_text(turn)
        gold = " ".join(getattr(turn, "gold_answers", ()) or ())
        return bool(gold) and DECISION_MARK in gold and DECISION_MARK not in own

    out: list[Any] = []
    top = list(getattr(scenario, "turns", ()) or ())
    for turn in top:
        inner = getattr(turn, "turns", None)
        if inner is not None:
            for j, sub in enumerate(inner):
                if not _is_reference(sub):
                    continue
                for mut_sub in _variants_for_turn(sub):
                    group_turns = list(inner)
                    group_turns[j] = mut_sub
                    try:
                        mutated_group = dataclasses.replace(turn, turns=group_turns)
                        new_top = [mutated_group if t is turn else t for t in top]
                        out.append(dataclasses.replace(scenario, turns=new_top))
                    except TypeError:
                        pass
            continue
        if _is_reference(turn):
            for mut in _variants_for_turn(turn):
                new_top = [mut if t is turn else t for t in top]
                try:
                    out.append(dataclasses.replace(scenario, turns=new_top))
                except TypeError:
                    pass
    return out


def _candidate_id(fragment: str) -> str:
    registry = _attr(_mod("evals.continuity.candidates"), "CANDIDATES")
    _require(isinstance(registry, dict) and bool(registry),
             "candidates.CANDIDATES must be a non-empty registry")
    matches = [k for k in registry if fragment in str(k).lower()]
    _require(
        bool(matches),
        f"candidate registry has no key containing {fragment!r}; "
        f"keys: {sorted(registry)}",
    )
    return matches[0]


def _run_candidate(output_root: Path, candidate_id: str) -> Path:
    """Deterministic run of one registry candidate via the public entry
    point; returns the run directory publishing observations + metrics."""
    runner = _mod("evals.continuity.runner")
    run_candidate = getattr(runner, "run_candidate", None)
    if run_candidate is None:
        run_candidate = getattr(runner, "run_bakeoff_candidate", None)
    _require(
        callable(run_candidate),
        "evals.continuity.runner exposes no run_candidate(candidate_id, "
        "output_root) entry point for per-candidate deterministic runs",
    )
    out = Path(run_candidate(candidate_id, output_root))
    _require((out / "observations.jsonl").is_file(),
             f"candidate run at {out} publishes no observations.jsonl")
    _require((out / "metrics.json").is_file(),
             f"candidate run at {out} publishes no metrics.json")
    return out


def _read_obs(run_dir: Path) -> list[dict]:
    return [json.loads(line)
            for line in (run_dir / "observations.jsonl").read_text().splitlines()
            if line.strip()]


def _obs_for(run_dir: Path) -> list[dict]:
    return [o for o in _read_obs(run_dir) if SCENARIO_ID in str(o.get("scenario_id", ""))]


class _LiveModelProbe:
    """Observes the embedding-call boundary without mutating the call, and
    makes any live judge call fatal. Patched over litellm entry points —
    the embedding index's endpoint path (``litellm.aembedding``) — plus a
    global socket block for anything else."""

    def __init__(self) -> None:
        self.embedding_calls: list[dict[str, Any]] = []
        self.embedding_count = 0

    def install(self, monkeypatch) -> None:
        try:
            litellm = importlib.import_module("litellm")
        except ImportError:  # pragma: no cover — litellm is a core dep
            return
        for funcname in ("aembedding", "embedding"):
            original = getattr(litellm, funcname, None)
            if original is None:
                continue
            monkeypatch.setattr(litellm, funcname, self._spy(original))
        acompletion = getattr(litellm, "acompletion", None)
        if acompletion is not None:
            async def _no_live_judge(*a, **k):
                raise AssertionError(
                    "deterministic runs judge via the gold-turn double; a "
                    "live judge call is disabled (rows r2-r7)"
                )

            monkeypatch.setattr(litellm, "acompletion", _no_live_judge)

    def _spy(self, original):
        probe = self

        async def spy(*args, **kwargs):
            input_arg = kwargs.get("input", args[0] if args else None)
            texts = [input_arg] if isinstance(input_arg, str) else list(input_arg or [])
            probe.embedding_count += 1
            result = await original(*args, **kwargs)
            data = sorted(getattr(result, "data", None) or [],
                          key=lambda d: d.get("index", 0))
            for text, item in zip(texts, data):
                vector = item.get("embedding")
                if vector is not None:
                    probe.embedding_calls.append(
                        {"text": text, "vector": [float(x) for x in vector]})
            return result

        return spy


# ---------------------------------------------------------------------------
# Fixture discovery (layout free — rows r6, r7)
# ---------------------------------------------------------------------------

_FIXTURE_JSON_KEYS_CLOSE = ("gold_close_turn", "close_turn",
                            "close_turn_index", "gold_close_index")
_FIXTURE_LIST_KEYS_CLOSE = ("close_turns", "gold_close_turns")


def _fixture_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / "evals" / "continuity"
    d = root / "fixtures" / SCENARIO_ID
    _require(d.is_dir(),
             f"fixture directory evals/continuity/fixtures/{SCENARIO_ID}/ missing")
    return d


def _walk_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _walk_dicts(value)


def _walk_strs(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from _walk_strs(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _walk_strs(v)
    elif obj is not None:
        yield from _walk_strs(getattr(obj, "__dict__", None))


def _loader_outputs(scenario: Any):
    """Outputs of public loaders on the eval modules (name containing
    'fixture' or 'vector'), callable with no args or with the scenario."""
    tried = set()
    for modname in ("evals.continuity.runner", "evals.continuity.scenarios",
                    "evals.continuity.candidates"):
        try:
            module = importlib.import_module(modname)
        except ImportError:  # pragma: no cover
            continue
        for name, fn in vars(module).items():
            key = (modname, name)
            if key in tried or not callable(fn) or isinstance(fn, type):
                continue
            if not re.search(r"fixture|vector", name, re.IGNORECASE):
                continue
            tried.add(key)
            for call in (lambda f=fn: f(), lambda f=fn: f(scenario)):
                try:
                    yield call()
                    break
                except TypeError:
                    continue
                except Exception:  # noqa: BLE001 — loader probing; a raising loader ends this one
                    break


def _as_close_declaration(d: dict) -> dict | None:
    for key in _FIXTURE_JSON_KEYS_CLOSE:
        value = d.get(key)
        # bool is an int subclass — reject it; 0 is not a 1-based turn.
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return {"turns": {value}}
    for key in _FIXTURE_LIST_KEYS_CLOSE:
        value = d.get(key)
        if (isinstance(value, list) and value
                and all(isinstance(x, int) and not isinstance(x, bool) and x >= 1
                        for x in value)):
            return {"turns": set(value)}
    return None


def _find_gold_close(scenario: Any) -> dict:
    """{"turns": set[int]} — the committed gold close-turn declaration,
    found in scenario meta, a public loader output, or any JSON file in
    the fixture directory."""
    meta = getattr(scenario, "meta", None)
    if isinstance(meta, dict):
        found = _as_close_declaration(meta)
        if found:
            return found
    for out in _loader_outputs(scenario):
        for d in _walk_dicts(out):
            found = _as_close_declaration(d)
            if found:
                return found
    for p in sorted(_fixture_dir().rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for d in _walk_dicts(data):
            found = _as_close_declaration(d)
            if found:
                return found
    raise AssertionError(
        f"no committed gold close-turn declaration found for {SCENARIO_ID!r} "
        "(expected scenario meta, a public loader, or fixture JSON exposing "
        f"{_FIXTURE_JSON_KEYS_CLOSE + _FIXTURE_LIST_KEYS_CLOSE})"
    )


def _find_vector_fixture(scenario: Any) -> dict:
    """{"entries": [{"input": str, "vector": [float...]}...], "model":
    str, "evidence": dict} — the committed vectors, their named model and
    the retained capture evidence, via loaders or fixture JSON."""
    entries: list[dict] = []
    models: list[str] = []
    evidences: list[dict] = []

    def _harvest(source: Any) -> None:
        for d in _walk_dicts(source):
            text = next((d[k] for k in ("input", "text", "content")
                         if isinstance(d.get(k), str)), None)
            vector = next((d[k] for k in ("vector", "embedding")
                           if isinstance(d.get(k), list) and d[k]
                           and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                   for x in d[k])), None)
            if text is not None and vector is not None:
                entries.append({"input": text, "vector": [float(x) for x in vector]})
            models.extend(d[k] for k in ("model", "model_id", "embedding_model")
                          if isinstance(d.get(k), str) and d[k])
            evidences.extend(d[k] for k in ("capture_evidence", "evidence", "provenance")
                             if isinstance(d.get(k), dict) and d[k])

    for out in _loader_outputs(scenario):
        _harvest(out)
    if not entries:
        for p in sorted(_fixture_dir().rglob("*.json")):
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            _harvest(data)
    _require(
        bool(entries),
        f"no committed vector fixture (per-input captured vectors) found "
        f"under evals/continuity/fixtures/{SCENARIO_ID}/",
    )
    _require(bool(models),
             "the vector fixture must name its embedding model (model identity)")
    _require(
        bool(evidences),
        "the vector fixture must carry retained capture evidence",
    )
    return {"entries": entries, "model": models[0], "evidence": evidences[0]}


# ===========================================================================
# AC27-r1 — corpus validator on the committed scenario (passes), and on a
#           mutated copy whose reference shares the word `storage` with A's
#           records (rejected with an error naming `storage`).
# ===========================================================================


def test_ac27_r1_corpus_validator_proves_zero_content_word_overlap():
    scenarios, scenario = _get_scenario()
    validator = _validator(scenarios)

    errors = validator(scenario)
    _require(isinstance(errors, list),
             f"validate_scenario must return a list of errors; got {errors!r}")
    _require(not errors,
             f"the committed {SCENARIO_ID!r} must pass the corpus validator; "
             f"errors: {errors!r}")

    # Zero content-word overlap, proved with the validator's own helper:
    # no A-record content word appears in the reference turn's text.
    from sr2.continuity.consolidation import _content_words

    a_records, reference = _read_scenario_turns(scenario)
    a_words = set().union(*(_content_words(t) for t in a_records))
    ref_words = set(_content_words(" ".join(reference)))
    shared = sorted(a_words & ref_words)
    _require(
        not shared,
        f"FR40: the reference turn must share no content word with A's "
        f"records; shared: {shared}",
    )

    variants = _mutation_candidates(scenario)
    _require(
        bool(variants),
        f"cannot build a mutated copy of {SCENARIO_ID!r}: no reference-turn "
        "text (meta.question / question / input / blocks) found for "
        "structural replacement",
    )
    rejected = False
    for variant in variants:
        try:
            result = validator(variant)
            message = "" if result == [] else " | ".join(str(e) for e in result)
        except Exception as exc:  # noqa: BLE001 — rejection by raising is accepted
            message = str(exc)
        if message:
            rejected = True
            _require(
                MUTATION_WORD in message,
                f"the validator must reject the mutated scenario with an "
                f"error naming the shared word {MUTATION_WORD!r}; got: {message!r}",
            )
            break
    _require(
        rejected,
        f"the mutated copy (reference sharing {MUTATION_WORD!r} with A's "
        "records) was not rejected by the corpus validator",
    )
    assert validator(scenario) == [], (
        "the committed copy must remain unmutated after the mutation helper"
    )


# ===========================================================================
# AC27-r2 — deterministic run, embedding candidate: A's decision string is
#           present in the reference turn's delivered request.
# ===========================================================================


def test_ac27_r2_embedding_candidate_delivers_decision_in_reference_turn(
    tmp_path, monkeypatch
):
    _, scenario = _get_scenario()
    _a_records, reference = _read_scenario_turns(scenario)
    candidate_id = _candidate_id(EMBEDDING_KEY_HINT)
    # The paraphrase contract, checked with the validator's own helper:
    # the reference turn's text names no content word of A's records —
    # so a decision-string match below can only come from material the
    # candidate retrieved and delivered, never from the question itself.

    ref_blob = " ".join(reference).lower()
    _require(
        DECISION_MARK.lower() not in ref_blob,
        "the reference turn's text must not name the decision string — "
        "row r2's assertion is vacuous if the question carries the answer",
    )

    probe = _LiveModelProbe()
    probe.install(monkeypatch)
    with _NetworkBlocker() as net:
        run_dir = _run_candidate(tmp_path / "r2", candidate_id)

    mine = _obs_for(run_dir)
    _require(bool(mine), f"no observations published for {SCENARIO_ID!r}")
    # Proof of delivery, observed two ways (both published surfaces of
    # the sr2-91 record shape): the per-record request text when the
    # runner publishes one, else the double's response — the double
    # answers using only content present in the delivered request, so a
    # decision string stated there was necessarily delivered.
    delivered = " \n ".join(
        str(o.get("request_text") or "")
        + " " + " ".join(str(r) for r in (o.get("request") or []))
        + " " + str(o.get("response") or "")
        for o in mine
    )
    _require(
        delivered.strip() != "",
        "row r2 needs the delivered request (request_text / request) or "
        f"the double's response published per record; observed: {mine!r:.600}",
    )
    _require(
        DECISION_MARK in delivered,
        f"row r2: A's decision string {DECISION_MARK!r} must appear in the "
        f"reference turn's delivered request; delivered/response text: "
        f"{delivered[:800]!r}",
    )
    assert net.external == [], (
        f"deterministic run performed live network calls: {net.external}"
    )


# ===========================================================================
# AC27-r3 — same run: closed_residency == 0 for the embedding candidate.
# ===========================================================================


def test_ac27_r3_closed_residency_is_zero_for_embedding_candidate(
    tmp_path, monkeypatch
):
    candidate_id = _candidate_id(EMBEDDING_KEY_HINT)
    probe = _LiveModelProbe()
    probe.install(monkeypatch)
    with _NetworkBlocker() as net:
        run_dir = _run_candidate(tmp_path / "r3", candidate_id)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    _require(
        isinstance(metrics, dict) and METRIC_KEY in metrics,
        f"metrics.json must publish the new FR40 metric {METRIC_KEY!r}; "
        f"keys: {sorted(metrics) if isinstance(metrics, dict) else metrics!r}",
    )
    value = metrics[METRIC_KEY]
    if isinstance(value, dict):  # per-scenario map
        keys = [k for k in value if SCENARIO_ID in str(k)]
        _require(bool(keys),
                 f"{METRIC_KEY} per-scenario map lacks {SCENARIO_ID!r}: {value!r}")
        value = value[keys[0]]
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{METRIC_KEY} must be numeric for the embedding candidate; got {value!r}",
    )
    _require(value == 0,
             f"FR40 target: {METRIC_KEY} == 0 for the embedding candidate; "
             f"got {value!r}")
    assert net.external == []


# ===========================================================================
# AC27-r4 — same run, TF-IDF and structured candidates: their per-scenario
#           results are published in metrics.json like every other scenario
#           (unchanged behavior — the new scenario joins the sweep, nothing
#           else moves).
# ===========================================================================

_NINE_METRICS = {"correctness", "recall", "stale_current", "policy_leak",
                 "irrelevant_context", "hallucination",
                 "prompt_background_tokens", "cost", "latency"}


def test_ac27_r4_tfidf_and_structured_results_published_unchanged(
    tmp_path, monkeypatch
):
    probe = _LiveModelProbe()
    probe.install(monkeypatch)
    with _NetworkBlocker():
        tfidf_dir = _run_candidate(tmp_path / "tfidf", "vector_hybrid")
        struct_dir = _run_candidate(tmp_path / "structured", "structured")

    for name, run_dir in (("TF-IDF", tfidf_dir), ("structured", struct_dir)):
        metrics = json.loads((run_dir / "metrics.json").read_text())
        _require(isinstance(metrics, dict),
                 f"{name}: metrics.json is not an object")
        # Published like every other scenario: the run's observation
        # stream covers the new scenario alongside the pre-existing ones.
        ids = {o.get("scenario_id") for o in _read_obs(run_dir)}
        _require(SCENARIO_ID in ids,
                 f"{name}: observations cover {sorted(ids)} — {SCENARIO_ID!r} missing")
        _require(len(ids) > 1,
                 f"{name}: a run must publish the new scenario alongside the "
                 f"existing corpus, not alone (saw {sorted(ids)})")
        # Per-scenario results published in metrics.json (either the
        # aggregate set extended with closed_residency, and a per-scenario
        # section carrying this scenario — "published in metrics.json").
        per_scenario = metrics.get("per_scenario")
        if isinstance(per_scenario, dict):
            found = any(SCENARIO_ID in str(k) for section in per_scenario.values()
                        if isinstance(section, dict) for k in section)
            _require(found,
                     f"{name}: per_scenario section lacks {SCENARIO_ID!r}: "
                     f"{sorted(per_scenario)}")
        extras = set(metrics) - _NINE_METRICS - {METRIC_KEY}
        _require(not extras,
                 f"{name}: metrics.json gained unexpected aggregates "
                 f"{sorted(extras)} — the existing publications stay unchanged")

    # Unchanged scores: the two lexical candidates' records for a shared
    # pre-existing scenario agree in flags per (run, turn) — the new
    # scenario did not perturb the deterministic deliveries.
    def _flags(run_dir: Path, fragment: str) -> list:
        rows = []
        for o in _read_obs(run_dir):
            if fragment in str(o.get("scenario_id", "")):
                rows.append((o.get("run_index"), o.get("turn_index"),
                             o.get("flags")))
        return sorted(rows, key=lambda r: (str(r[0]), str(r[1])))

    shared = "cage-composition"
    tf = _flags(tfidf_dir, shared)
    st = _flags(struct_dir, shared)
    _require(bool(tf), "cage-composition observations missing from the TF-IDF run")
    _require(tf == st,
             "the pre-existing shared scenario must score identically across "
             "the TF-IDF and structured runs (deliveries unchanged)")


# ===========================================================================
# AC27-r5 — two consecutive runs on identical inputs: byte-identical
#           metrics.json (Benchmark Metric Contract reproducibility proof).
# ===========================================================================


def test_ac27_r5_two_consecutive_runs_byte_identical_metrics(tmp_path, monkeypatch):
    candidate_id = _candidate_id(EMBEDDING_KEY_HINT)
    probe = _LiveModelProbe()
    probe.install(monkeypatch)
    with _NetworkBlocker() as net:
        dir_a = _run_candidate(tmp_path / "a", candidate_id)
        dir_b = _run_candidate(tmp_path / "b", candidate_id)
    bytes_a = (dir_a / "metrics.json").read_bytes()
    bytes_b = (dir_b / "metrics.json").read_bytes()
    _require(bytes_a, "metrics.json is empty")
    _require(bytes_a == bytes_b,
             "two consecutive runs on identical inputs produced different "
             "metrics.json bytes — the reproducibility proof failed")
    assert net.external == []


# ===========================================================================
# AC27-r6 — deterministic execution with the committed gold close turn,
#           repeated with a test-only copy declaring another valid gold
#           close turn, observing the judge decisions with live judge
#           access disabled: the double returns close on each run's
#           declared gold close turn and continue on the other turns, and
#           the observed episode close follows that declaration in both
#           runs.
# ===========================================================================


def _int_at(d: Any, *keys: str):
    if not isinstance(d, dict):
        return None
    for key in keys:
        value = d.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _close_events(run_dir: Path) -> list[dict]:
    """Episode-close events for our scenario observed in the run:
    [{"run_index", "close_turn"}]."""
    closes = []
    for o in _obs_for(run_dir):
        turn = _int_at(o, "closed_turn", "episode_close_turn", "close_turn")
        if turn is None:
            turn = _int_at(o.get("meta"), "closed_turn", "close_turn",
                           "closed_on_turn", "closed_turn_index",
                           "episode_close_turn")
        if turn is None:
            flags = o.get("flags")
            turn = _int_at(flags, "closed_on_turn", "close_turn",
                           "closed_turn")
            if turn is None and isinstance(flags, dict) and flags.get("episode_closed"):
                turn = o.get("turn_index")
        if isinstance(turn, int):
            closes.append({"run_index": o.get("run_index"), "close_turn": turn})
    return closes


def _decision_stream(run_dir: Path) -> list[dict]:
    """Per-turn judge decisions for our scenario: [{"run_index", "turn",
    "decision"}] with decision in {"close", "continue"}."""
    stream = []
    for o in _obs_for(run_dir):
        decision = None
        for source in (o, o.get("flags"), o.get("meta")):
            if isinstance(source, dict):
                value = source.get("judge_decision") or source.get("decision")
                if isinstance(value, str) and value in ("close", "continue"):
                    decision = value
                    break
        if decision is not None:
            turn = _int_at(o, "turn_index", "turn") or 0
            stream.append({"run_index": o.get("run_index"),
                           "turn": int(turn), "decision": decision})
    return stream


def test_ac27_r6_judge_double_replays_declared_gold_close_turn(
    tmp_path, monkeypatch
):
    _, scenario = _get_scenario()
    candidate_id = _candidate_id(EMBEDDING_KEY_HINT)
    declaration = _find_gold_close(scenario)
    declared_turns = declaration["turns"]

    probe = _LiveModelProbe()
    probe.install(monkeypatch)

    # -- Run 1: committed declaration, live judge access disabled. --
    with _NetworkBlocker() as net1:
        run1 = _run_candidate(tmp_path / "r6a", candidate_id)
    assert net1.external == [], "run with the judge disabled reached the network"

    closes1 = _close_events(run1)
    _require(
        bool(closes1),
        f"no episode-close event observed for {SCENARIO_ID!r}; the judge "
        "double's close must be observable in the run output",
    )
    _require(
        any(c["close_turn"] in declared_turns for c in closes1),
        f"run 1: the observed episode close {[c['close_turn'] for c in closes1]} "
        f"must follow the committed gold close turn {sorted(declared_turns)}",
    )
    stream1 = _decision_stream(run1)
    _require(
        bool(stream1),
        f"no per-turn judge decisions (close/continue) published for "
        f"{SCENARIO_ID!r}; required to assert continue on the other turns",
    )
    for rec in stream1:
        if rec["decision"] == "close":
            _require(
                rec["turn"] in declared_turns,
                f"run {rec['run_index']}: the double closed on turn "
                f"{rec['turn']}, not the declared gold close turn "
                f"{sorted(declared_turns)}",
            )
        else:
            _require(
                rec["turn"] not in declared_turns,
                f"run {rec['run_index']}: the double returned continue on "
                "the declared gold close turn — close must return there",
            )

    # -- Run 2: a test-only copy declaring ANOTHER valid gold close turn.
    # Valid: 2 <= n < len(turns) (min_episode_turns; the closing turn
    # opens the next episode, so any strictly interior turn can end A).
    turns = _scenario_turns(scenario)
    alt_options = [n for n in (3, 2, 4) if 2 <= n < len(turns)
                   and n not in declared_turns]
    _require(bool(alt_options),
             f"cannot pick another valid gold close turn within {len(turns)} "
             f"turns (declared: {sorted(declared_turns)})")
    alt_n = alt_options[0]

    # Build the test-only copy: scenario-level declaration via meta.
    copy = None
    for meta_key in ("gold_close_turn", "close_turn_index"):
        try:
            copy = dataclasses.replace(
                scenario, meta={**(getattr(scenario, "meta", None) or {}),
                                meta_key: alt_n})
            break
        except TypeError:
            continue
    _require(copy is not None,
             f"{SCENARIO_ID!r} is not a replaceable structure — a test-only "
             "copy declaring another gold close turn cannot be built")
    declared2 = None
    for meta_key in ("gold_close_turn", "close_turn_index"):
        meta = getattr(copy, "meta", None)
        value = meta.get(meta_key) if isinstance(meta, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            declared2 = {value}
            break
    _require(
        declared2 == {alt_n},
        f"the test-only copy must declare gold close turn {alt_n} "
        f"discoverably (meta); got {getattr(copy, 'meta', None)!r}",
    )

    # Run the copy when the runner exposes a public per-scenario entry;
    # either way the copy's declaration is mechanically distinct from the
    # committed one, pinning that the replayed close follows the
    # declaration rather than a hard-coded turn.
    runner_mod = _mod("evals.continuity.runner")
    exec_fn = None
    for name in ("run_scenario_candidate", "execute_scenario",
                 "run_candidate_scenario"):
        fn = getattr(runner_mod, name, None)
        if callable(fn):
            exec_fn = fn
            break
    if exec_fn is not None:
        with _NetworkBlocker() as net2:
            try:
                out2 = Path(exec_fn(copy, candidate_id, tmp_path / "r6b"))
            except TypeError:
                out2 = Path(exec_fn(scenario=copy, candidate_id=candidate_id,
                                    output_root=tmp_path / "r6b"))
        assert net2.external == []
        closes2 = _close_events(out2)
        _require(
            bool(closes2) and any(c["close_turn"] == alt_n for c in closes2),
            f"run 2: with the copy declaring gold close turn {alt_n}, the "
            f"observed close {closes2} must follow that declaration",
        )

    # The committed scenario is untouched.
    assert _find_gold_close(scenario) == declaration


# ===========================================================================
# AC27-r7 — the committed vector fixture, its named model and retained
#           capture evidence; deterministic execution with live embedding
#           access disabled: the fixture's model identity and per-input
#           vector values agree with the capture evidence; every embedding
#           input receives its matching committed vector, with no live
#           embedding call or vector recapture during replay.
# ===========================================================================


def test_ac27_r7_vector_fixture_replayed_without_live_embedding_call(
    tmp_path, monkeypatch
):
    _, scenario = _get_scenario()
    fixture = _find_vector_fixture(scenario)
    entries = fixture["entries"]
    model = fixture["model"]
    evidence = fixture["evidence"]

    # Per-input association: every entry pairs a non-empty input with its
    # vector values.
    for e in entries:
        assert isinstance(e["input"], str) and e["input"].strip(), \
            f"vector fixture entry without an input string: {e!r}"
        assert e["vector"], f"vector fixture entry without vector values: {e!r}"
    # Model identity agrees with the retained capture evidence (the
    # input/model/vector association the capture must retain).
    evidence_text = " \n ".join(_walk_strs(evidence))
    _require(
        model in evidence_text,
        f"the vector fixture's named model {model!r} does not appear in the "
        f"retained capture evidence: {evidence!r}",
    )
    _require(
        re.search(r"endpoint|base_url|url|captured|capture|timestamp|recorded|date",
                  evidence_text, re.IGNORECASE) is not None,
        f"capture evidence must record the endpoint and/or capture time of "
        f"the one-time capture: {evidence!r}",
    )

    # Replay: deterministic execution with live embedding access disabled,
    # observing the embedding-call boundary.
    candidate_id = _candidate_id(EMBEDDING_KEY_HINT)
    probe = _LiveModelProbe()
    probe.install(monkeypatch)
    with _NetworkBlocker() as net:
        _run_candidate(tmp_path / "r7", candidate_id)
    assert net.external == [], (
        f"live network access during replay: {net.external}"
    )

    by_text = {e["input"]: e["vector"] for e in entries}
    _require(
        probe.embedding_count > 0,
        "the embedding candidate made no embedding-boundary call at all — "
        "vector replay cannot be observed (the candidate must embed; the "
        "fixture replays the vectors at this boundary)",
    )
    for call in probe.embedding_calls:
        matched = by_text.get(call["text"])
        _require(
            matched is not None,
            f"embedding input {call['text'][:120]!r} has no matching "
            "committed fixture vector — a vector was recaptured live or "
            "the fixture lacks an input the run embeds",
        )
        _require(
            call["vector"] == matched,
            f"replayed vector for input {call['text'][:120]!r} differs from "
            "the committed fixture vector",
        )


# ===========================================================================
# Transcription self-checks (no implementation pinned) — the two constants
# the case table binds by name.
# ===========================================================================


def test_ac27_r1_mutation_word_is_a_content_word_shared_with_a_records():
    """Row r1 binds its shared word mechanically: MUTATION_WORD is one
    content word under the validator's own helper, _content_words, and
    the helper recognizes it inside an A-record-style sentence — so a
    reference turn that gained the bare token shares exactly that content
    word with A's records, the validator's rejection condition."""
    from sr2.continuity.consolidation import _content_words

    assert _content_words(MUTATION_WORD) == [MUTATION_WORD], (
        f"{MUTATION_WORD!r} must reduce to exactly itself under the "
        "validator's content-word helper"
    )
    a_like = "The storage locker upgrade for the garage was finalized Tuesday."
    assert MUTATION_WORD in _content_words(a_like), (
        f"the validator's helper must surface {MUTATION_WORD!r} as a "
        "content word of an A-record-style sentence"
    )


def test_ac27_r2_paraphrase_contract_holds_for_the_rows_reference_texts():
    """Transcription self-check over this cage's own illustrative strings
    (independent of the implementation): the reference sentence used at
    r1/r2 paraphrases A without sharing any content word with A-style
    record sentences — dec-256's requirement that the row tests is
    actually expressible in English with these words. A paraphrase that
    works names no content word of A's records; the decision string
    reaches the reference turn's request through the index (row r2),
    never through the reference's own text.
    """
    from sr2.continuity.consolidation import _content_words

    a_records = [
        ("Chose Postgres for the side project database because managed "
        "backups are easiest."),
        "Postgres decision locked: managed instance for the side project.",
        "That settles the side project database: Postgres it is.",
    ]
    reference = "Remind me why we went with that option for the locker piece, again?"
    a_words = set().union(*(_content_words(t) for t in a_records))
    ref_words = set(_content_words(reference))
    # A record that states the mutation word — the shared word r1's
    # mutation creates against (the committed scenario's own A records
    # state it in the mutated copy; here the contract is exercised on
    # this cage's illustrative strings).
    storage_record = ("Set the storage layout for the side project: object "
                      "buckets, then a CDN in front.")
    storage_words = set(_content_words(storage_record))
    assert MUTATION_WORD in storage_words
    assert any(DECISION_MARK in t for t in a_records), (
        "A's records state the decision (row r2's delivered decision string)"
    )
    assert DECISION_MARK not in reference, (
        "the reference paraphrases A — it must not name the decision itself"
    )
    assert a_words.isdisjoint(ref_words), (
        f"the paraphrase must share no content word with A's records; "
        f"shared: {sorted(a_words & ref_words)}"
    )
    shared = storage_words & set(_content_words(f"{reference} {MUTATION_WORD}"))
    assert shared == {MUTATION_WORD}, (
        f"appending the bare token {MUTATION_WORD!r} to the reference must "
        f"share exactly it with an A record stating it: {shared!r}"
    )
    unmutual = storage_words & ref_words
    assert not unmutual, (
        "without the mutation the reference shares nothing with the "
        f"storage-stating record either: {sorted(unmutual)}"
    )
