"""Tests for the review package builder and single-call voter (spc-960).

Unit tests run with no LLM and no network: the voter is exercised against a
fake ``complete`` that records the request bodies, and the AC1 determinism
test uses the frozen kodiak-18-c1 inputs when present (skipping cleanly
when absent). The AC7 live test hits the local relay once per model and is
deselected by default via the ``live`` marker.

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_review.py -q
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sr2_spectre.review import (
    DEFAULT_RELAY_BASE_URL,
    DEFAULT_SECRET_PATTERNS,
    JUDGE_NAMES,
    PackageInput,
    PackageTooLargeError,
    QuorumConfigError,
    RelayCompletionClient,
    RelaySchemaRejected,
    RotationState,
    SecretPathError,
    VOTER_SCHEMA,
    _error_text,
    build_package,
    judge_name,
    quorum_report,
    run_quorum,
    run_voter,
    voter_request_body,
    voter_rows,
)
from sr2_spectre.run_log import SessionLogManager

FROZEN = Path("/data/artifacts/homelab-ansible/scenarios/kodiak-18-c1")
R5 = FROZEN / "decision" / "d-review-r5"
R5_WT = Path("/data/worktrees/homelab-ansible/scenario-kodiak-18-c1-r5")
R5_BASE = "3fb3425"
R5_HEAD = "5cc4276"

CASE_TABLE = """row_id | input | expected | clause | layer
AC2-c1 | {a:1} metadata | exit 64, nothing run | FR 17c | unit
AC2-c2 | --allowlist-file | exit 64 | FR 17a | unit
AC2-c3 | {"a":1} metadata | passes through | FR 17c | unit
"""

SYSTEM_PROMPT = "You are a reviewer. Answer with the voter JSON only."


def _write_case_table(tmp_path: Path) -> Path:
    path = tmp_path / "case_table.md"
    path.write_text(CASE_TABLE, encoding="utf-8")
    return path


def _make_inputs(tmp_path: Path) -> dict[str, Path]:
    test_file = tmp_path / "test_flag.py"
    test_file.write_text("def test_x():\n    pass\n", encoding="utf-8")
    diff = tmp_path / "diff.patch"
    diff.write_text("diff --git a/x b/x\n+x\n", encoding="utf-8")
    run_txt = tmp_path / "run.txt"
    run_txt.write_text("OK (3 tests)\n", encoding="utf-8")
    return {"test_file": test_file, "diff": diff, "run_txt": run_txt}


def _build(tmp_path: Path, **kwargs) -> "object":
    inputs = _make_inputs(tmp_path)
    files = [PackageInput(inputs["test_file"], "tests/test_flag.py")]
    kwargs.setdefault(
        "case_table", _write_case_table(tmp_path)
    )
    kwargs.setdefault("diff", inputs["diff"])
    kwargs.setdefault("run_output", inputs["run_txt"])
    kwargs.setdefault("files", files)
    kwargs.setdefault("out_dir", tmp_path / "pkg")
    return build_package(bead="spc-960", base="base", head="head", **kwargs)


# ---------------------------------------------------------------------------
# Fake completion transport for the voter
# ---------------------------------------------------------------------------


class FakeComplete:
    """Records every request body; returns a canned answer or raises."""

    def __init__(self, answer: str | None, *, fail: Exception | None = None):
        self.answer = answer
        self.fail = fail
        self.bodies: list[dict] = []

    async def __call__(self, body: dict) -> str:
        self.bodies.append(body)
        if self.fail is not None:
            raise self.fail
        return self.answer or ""


def _good_answer(rows: list[str] | None = None) -> str:
    ids = rows if rows is not None else ["AC2-c1", "AC2-c2", "AC2-c3"]
    return json.dumps({"rows": [
        {"row_id": row_id, "holds": True, "reason": "ok"} for row_id in ids
    ]})


def _log_events(log_dir: Path) -> list[dict]:
    events = []
    for path in sorted(log_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# AC1 — build_package determinism and the size limit
# ---------------------------------------------------------------------------


def test_package_layout_and_manifest_contents(tmp_path):
    pkg = _build(tmp_path)
    for name in ("manifest.json", "case_table.md", "diff.patch", "run.txt",
                 "tests/test_flag.py"):
        assert (pkg.directory / name).is_file(), name
    manifest = pkg.manifest
    assert manifest["bead"] == "spc-960"
    assert manifest["base"] == "base"
    assert manifest["head"] == "head"
    paths = [e["path"] for e in manifest["files"]]
    assert paths == ["case_table.md", "diff.patch", "run.txt", "tests/test_flag.py"]
    for entry in manifest["files"]:
        assert len(entry["sha256"]) == 64
    assert isinstance(manifest["rendered_tokens_estimate"], int)
    assert manifest["rendered_tokens_estimate"] > 0


def test_package_hash_is_sha256_of_manifest(tmp_path):
    import hashlib

    pkg = _build(tmp_path)
    manifest_bytes = (pkg.directory / "manifest.json").read_bytes()
    assert pkg.package_hash == hashlib.sha256(manifest_bytes).hexdigest()


def test_package_hash_identical_across_two_runs(tmp_path):
    pkg1 = _build(tmp_path, out_dir=tmp_path / "pkg1")
    pkg2 = _build(tmp_path, out_dir=tmp_path / "pkg2")
    assert pkg1.package_hash == pkg2.package_hash
    assert (pkg1.directory / "manifest.json").read_bytes() == (
        pkg2.directory / "manifest.json").read_bytes()


def test_build_package_refuses_over_limit_package(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 5000, encoding="utf-8")
    inputs = _make_inputs(tmp_path)
    with pytest.raises(PackageTooLargeError, match="exceeds limit"):
        build_package(
            bead="spc-960", base="base", head="head",
            case_table=_write_case_table(tmp_path),
            files=[PackageInput(big, "evidence/big.txt")],
            diff=inputs["diff"], run_output=inputs["run_txt"],
            out_dir=tmp_path / "pkg",
            max_rendered_bytes=1000,
        )


def test_size_estimate_covers_the_voter_message(tmp_path):
    """The estimate and the voter user message come from one rendering.

    ``voter_request_body``'s user message is the rendered package (the same
    text the manifest's size estimate recorded) plus a fixed instruction
    prefix, so the check can never disagree with what the voter receives.
    """
    pkg = _build(tmp_path)
    body = voter_request_body(pkg, SYSTEM_PROMPT, "qwen3.8-flash")
    user_message = body["messages"][1]["content"]
    # The manifest entries all appear in the message, and the message is
    # within a small fixed overhead of the estimated rendering.
    for entry in pkg.manifest["files"]:
        assert f"=== {entry['path']} ===" in user_message
    estimate_chars = pkg.manifest["rendered_tokens_estimate"] * 4
    assert len(user_message) <= estimate_chars + 200


def test_build_package_refuses_secret_input_naming_the_file(tmp_path):
    for pattern, name in [
        ("*.pem", "cert.pem"),
        ("*.key", "id_rsa.key"),
        (".env*", ".env"),
        ("*credentials*", "aws_credentials.json"),
        ("*secret*", "my-secret.md"),
        ("*token*", "token.txt"),
        ("*.zuliprc", ".zuliprc"),
    ]:
        secret = tmp_path / name
        secret.write_text("value", encoding="utf-8")
        inputs = _make_inputs(tmp_path)
        with pytest.raises(SecretPathError, match=name):
            build_package(
                bead="spc-960", base="base", head="head",
                case_table=_write_case_table(tmp_path),
                files=[PackageInput(secret, name)],
                diff=inputs["diff"], run_output=inputs["run_txt"],
                out_dir=tmp_path / "pkg",
            )
        assert pattern in DEFAULT_SECRET_PATTERNS


def test_builder_reads_only_named_paths(tmp_path, monkeypatch):
    """AC5: build_package reads no path outside those its inputs name.

    The builder reads each source once (for its bytes) and each placed
    file once (for the render). Every read is therefore either one of the
    named input paths or a path the builder itself wrote inside the
    package directory: a read of anything else, or a second read of an
    input the builder already consumed, fails here.
    """
    inputs = _make_inputs(tmp_path)
    files = [PackageInput(inputs["test_file"], "tests/test_flag.py")]
    case_table = _write_case_table(tmp_path)
    sources = {Path(p).resolve() for p in
               [case_table, inputs["test_file"], inputs["diff"], inputs["run_txt"]]}

    read: list[Path] = []
    original = Path.read_bytes
    pkg_dir = (tmp_path / "pkg").resolve()

    def tracking_read(self: Path, *a, **k):
        resolved = Path(self).resolve()
        read.append(resolved if pkg_dir in resolved.parents else resolved)
        return original(self, *a, **k)

    # The render reads through Path.read_text, so track both read paths.
    text_read: list[Path] = []
    original_text = Path.read_text

    def tracking_text(self: Path, *a, **k):
        resolved = Path(self).resolve()
        text_read.append(resolved if pkg_dir in resolved.parents else resolved)
        return original_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_bytes", tracking_read)
    monkeypatch.setattr(Path, "read_text", tracking_text)
    build_package(
        bead="spc-960", base="base", head="head",
        case_table=case_table, files=files,
        diff=inputs["diff"], run_output=inputs["run_txt"],
        out_dir=tmp_path / "pkg",
    )

    input_reads = [p for p in read + text_read if pkg_dir not in p.parents]
    placed_reads = [p for p in read + text_read if pkg_dir in p.parents]
    assert set(input_reads) == sources, "read a path no input named"
    assert len(input_reads) == len(sources), "read an input twice"
    assert sorted(p.relative_to(pkg_dir).as_posix() for p in placed_reads) == [
        "case_table.md", "diff.patch", "run.txt", "tests/test_flag.py",
    ]


@pytest.mark.skipif(
    not (FROZEN / "inputs").is_dir() or not (R5 / "run.txt").is_file(),
    reason="frozen kodiak-18-c1 inputs not present",
)
def test_ac1_frozen_r5_inputs_hash_stable(tmp_path):
    """AC1: the frozen r5 inputs produce an identical manifest hash across runs."""
    kit = FROZEN / "inputs"
    files = [
        PackageInput(kit / name, f"inputs/{name}")
        for name in sorted(p.name for p in kit.iterdir() if p.is_file())
    ]
    evidence = [PackageInput(R5 / "run.txt", "evidence/r5-run.txt")]

    # The diff is an input (FR 2: the builder runs no command). Supply it
    # from the r5 worktree when available; otherwise a fixed placeholder.
    diff_path = tmp_path / "diff.patch"
    if R5_WT.is_dir():
        import subprocess

        out = subprocess.run(
            ["git", "diff", R5_BASE, R5_HEAD],
            cwd=R5_WT, capture_output=True, text=True,
        )
        diff_path.write_text(out.stdout, encoding="utf-8")
    else:
        diff_path.write_text("diff placeholder\n", encoding="utf-8")

    case_table = _write_case_table(tmp_path)

    def build(out_dir: Path):
        return build_package(
            bead="kodiak-18-c1", base=R5_BASE, head=R5_HEAD,
            case_table=case_table,
            files=files,
            diff=diff_path,
            run_output=R5 / "run.txt",
            evidence=evidence,
            out_dir=out_dir,
        )

    pkg1 = build(tmp_path / "pkg1")
    pkg2 = build(tmp_path / "pkg2")
    assert pkg1.package_hash == pkg2.package_hash


# ---------------------------------------------------------------------------
# AC2 — no tools, unparseable / missing rows
# ---------------------------------------------------------------------------


async def test_voter_run_offers_no_tools(tmp_path):
    pkg = _build(tmp_path)
    complete = FakeComplete(_good_answer())
    verdict = await run_voter(
        pkg, complete, system_prompt=SYSTEM_PROMPT, model="qwen3.8-flash",
    )
    assert len(complete.bodies) == 1
    body = complete.bodies[0]
    # spc-966: offering no tools means the field is absent entirely — some
    # backends reject "tools": [] with HTTP 400.
    assert "tools" not in body
    assert verdict.request_sent is True
    assert all(row["holds"] for row in verdict.rows)


async def test_voter_unparseable_answer_denies_all_rows(tmp_path):
    pkg = _build(tmp_path)
    complete = FakeComplete("I think the code looks fine, but the quoting is off.")
    verdict = await run_voter(
        pkg, complete, system_prompt=SYSTEM_PROMPT, model="qwen3.8-flash",
    )
    assert len(complete.bodies) == 1
    assert len(verdict.rows) == 3
    for row in verdict.rows:
        assert row["holds"] is False
        assert row["reason"] == "unparseable"
    assert verdict.raw_answer.startswith("I think")


async def test_voter_missing_row_denied_with_reason_missing(tmp_path):
    pkg = _build(tmp_path)
    answer = json.dumps({"rows": [
        {"row_id": "AC2-c1", "holds": True, "reason": "ok"},
        {"row_id": "AC2-c3", "holds": True, "reason": "ok"},
    ]})
    complete = FakeComplete(answer)
    verdict = await run_voter(
        pkg, complete, system_prompt=SYSTEM_PROMPT, model="qwen3.8-flash",
    )
    by_id = {row["row_id"]: row for row in verdict.rows}
    assert by_id["AC2-c1"]["holds"] is True
    assert by_id["AC2-c2"]["holds"] is False
    assert by_id["AC2-c2"]["reason"] == "missing"
    assert by_id["AC2-c3"]["holds"] is True


async def test_voter_transport_failure_denies_all_rows_one_call(tmp_path):
    """A non-400 failure denies every row and retries nothing."""
    pkg = _build(tmp_path)
    complete = FakeComplete(None, fail=ConnectionError("connection reset"))
    verdict = await run_voter(
        pkg, complete, system_prompt=SYSTEM_PROMPT, model="qwen3.8-flash",
    )
    assert len(complete.bodies) == 1, "exactly one call per voter run"
    assert all(row["holds"] is False for row in verdict.rows)
    assert all(row["reason"] == "unparseable" for row in verdict.rows)


# ---------------------------------------------------------------------------
# AC7 — the schema rides on every request, whatever the model name
# ---------------------------------------------------------------------------


def _all_objects_pin_additional_properties(node) -> bool:
    if isinstance(node, dict):
        if node.get("type") == "object":
            if node.get("additionalProperties") is not False:
                return False
        return all(_all_objects_pin_additional_properties(v) for v in node.values())
    if isinstance(node, list):
        return all(_all_objects_pin_additional_properties(v) for v in node)
    return True


def test_voter_schema_pins_additional_properties_on_every_object():
    assert _all_objects_pin_additional_properties(VOTER_SCHEMA)


@pytest.mark.parametrize("model", ["qwen3.8-flash", "qwen3.8-27b", "gemma-4-31b"])
def test_request_carries_schema_whatever_the_model_name(tmp_path, model):
    """AC7: json_schema, strict, additionalProperties: false on every object,
    for every model name — no backend is inferred from the name."""
    pkg = _build(tmp_path)
    body = voter_request_body(pkg, SYSTEM_PROMPT, model)
    assert body["model"] == model
    # 966-c1: no tools offered — the field is absent, not an empty array.
    assert "tools" not in body
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema == VOTER_SCHEMA
    assert _all_objects_pin_additional_properties(schema)
    # max_tokens roomy enough that whitespace padding cannot truncate.
    assert body["max_tokens"] >= 200


async def test_voter_sends_the_schema_for_an_unfamiliar_model_name(tmp_path):
    """A model name unlike any known backend still gets the constrained call."""
    pkg = _build(tmp_path)
    complete = FakeComplete(_good_answer())
    verdict = await run_voter(
        pkg, complete, system_prompt=SYSTEM_PROMPT,
        model="some-desktop-lm-studio-thing",
    )
    assert len(complete.bodies) == 1
    assert complete.bodies[0]["response_format"]["json_schema"]["strict"] is True
    assert verdict.request_sent is True


# ---------------------------------------------------------------------------
# AC5 — rejected schema constraint fails closed; secret refusal; run log
# ---------------------------------------------------------------------------


async def test_voter_http400_denies_all_no_structured_output_no_retry(tmp_path):
    pkg = _build(tmp_path)
    complete = FakeComplete(
        None,
        fail=RelaySchemaRejected(
            "no node that fits the context supports: structured "
            "(capability_not_supported)"
        ),
    )
    verdict = await run_voter(
        pkg, complete, system_prompt=SYSTEM_PROMPT, model="qwen3.8-flash",
    )
    assert len(complete.bodies) == 1, "no unconstrained retry may be sent"
    assert verdict.request_sent is False
    assert len(verdict.rows) == 3
    for row in verdict.rows:
        assert row["holds"] is False
        assert row["reason"] == "no-structured-output"


async def test_voter_invalid_response_format_400_fails_closed(tmp_path):
    pkg = _build(tmp_path)
    complete = FakeComplete(
        None,
        fail=RelaySchemaRejected(
            "JSON Schema: objects require properties and additionalProperties: false "
            "(invalid_response_format)"
        ),
    )
    verdict = await run_voter(
        pkg, complete, system_prompt=SYSTEM_PROMPT, model="gemma-4-31b",
    )
    assert len(complete.bodies) == 1
    assert all(r["reason"] == "no-structured-output" for r in verdict.rows)
    assert "invalid_response_format" in verdict.error


# The live r5 rejection (spc-961-r3 finding 2): an HTTP 400 that is not a
# schema-constraint rejection. It must not reach the no-structured-output
# path; it is the existing other-failure path (unparseable, request_sent).
_EMPTY_TOOLS_400 = json.dumps(
    {
        "error": {
            "message": "tools must not be an empty array. Either provide at "
            "least one tool or omit the field entirely.",
            "type": "invalid_request_error",
        }
    }
).encode()


async def test_voter_non_schema_400_denies_unparseable_not_no_structured_output(
    tmp_path,
):
    """966-c2: a 400 that is not a schema rejection -> unparseable, no retry."""
    import aiohttp

    from sr2_spectre.review import _is_schema_constraint_rejection

    # The HTTP-boundary classification (the same predicate
    # RelayCompletionClient.complete applies, tested end-to-end below in
    # test_relay_client_flags_400_as_schema_rejected): this 400 is an
    # ordinary request failure, not a schema rejection.
    assert not _is_schema_constraint_rejection(_EMPTY_TOOLS_400)

    # run_voter treats it as any other failure: every row denied
    # unparseable, error text kept, request_sent True, one call only.
    pkg = _build(tmp_path)
    complete = FakeComplete(
        None,
        fail=aiohttp.ClientResponseError(
            MagicMock(), (), status=400, message=_error_text(_EMPTY_TOOLS_400)
        ),
    )
    verdict = await run_voter(
        pkg, complete, system_prompt=SYSTEM_PROMPT, model="qwen3.8-flash",
    )
    assert len(complete.bodies) == 1, "no retry may be sent"
    assert verdict.request_sent is True
    assert len(verdict.rows) == 3
    for row in verdict.rows:
        assert row["holds"] is False
        assert row["reason"] == "unparseable"
    assert "empty array" in verdict.error


async def test_voter_schema_rejection_codes_still_no_structured_output(tmp_path):
    """966-c3: capability_not_supported / invalid_response_format unchanged."""
    pkg = _build(tmp_path)
    for code in ("capability_not_supported", "invalid_response_format"):
        complete = FakeComplete(None, fail=RelaySchemaRejected(f"({code})"))
        verdict = await run_voter(
            pkg, complete, system_prompt=SYSTEM_PROMPT, model="qwen3.8-flash",
        )
        assert len(complete.bodies) == 1, "no unconstrained retry may be sent"
        assert all(r["reason"] == "no-structured-output" for r in verdict.rows), code
        assert code in verdict.error


def test_relay_client_flags_400_as_schema_rejected():
    """The HTTP boundary: 400 -> RelaySchemaRejected; other >=400 -> raise."""
    import asyncio

    from unittest.mock import AsyncMock, MagicMock, patch

    import aiohttp

    def _response(status: int, payload: bytes):
        response = MagicMock()
        response.status = status
        response.read = AsyncMock(return_value=payload)
        response.request_info = MagicMock()
        response.history = ()
        return response

    client = RelayCompletionClient(base_url="http://relay.invalid")

    async def run_with(status: int, payload: bytes):
        response = _response(status, payload)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=response)
        cm.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post = MagicMock(return_value=cm)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=ctx):
            return await client.complete({"model": "m"})

    body_400 = json.dumps(
        {"error": {"message": "nope", "code": "capability_not_supported"}}
    ).encode()
    with pytest.raises(RelaySchemaRejected, match="capability_not_supported"):
        asyncio.run(run_with(400, body_400))

    # spc-966: a 400 that is not a schema-constraint rejection (the live r5
    # "tools must not be an empty array" case) is an ordinary request
    # failure, not RelaySchemaRejected.
    empty_tools_400 = json.dumps(
        {
            "error": {
                "message": "tools must not be an empty array. Either provide "
                "at least one tool or omit the field entirely.",
                "type": "invalid_request_error",
            }
        }
    ).encode()
    with pytest.raises(aiohttp.ClientResponseError) as excinfo:
        asyncio.run(run_with(400, empty_tools_400))
    assert not isinstance(excinfo.value, RelaySchemaRejected)
    assert "empty array" in str(excinfo.value)

    # A schema-constraint 400 without a code but naming response_format is
    # still a schema rejection.
    named_400 = json.dumps(
        {"error": {"message": "response_format is not supported by this node"}}
    ).encode()
    with pytest.raises(RelaySchemaRejected):
        asyncio.run(run_with(400, named_400))

    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(run_with(503, b'{"error": {"message": "busy"}}'))

    ok_body = json.dumps(
        {"choices": [{"message": {"content": '{"rows": []}'}}]}
    ).encode()
    assert asyncio.run(run_with(200, ok_body)) == '{"rows": []}'


async def test_voter_writes_raw_answer_and_served_node_to_session_log(
    tmp_path, monkeypatch
):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(log_dir))
    pkg = _build(tmp_path)
    answer = json.dumps({"rows": [
        {"row_id": "AC2-c1", "holds": True, "reason": "ok"},
        {"row_id": "AC2-c2", "holds": False, "reason": "lenient edit"},
        {"row_id": "AC2-c3", "holds": True, "reason": "ok"},
    ]})
    complete = FakeComplete(answer)
    log = SessionLogManager().open_session("voter-test", "role-voter")
    try:
        await run_voter(
            pkg, complete, system_prompt=SYSTEM_PROMPT,
            model="qwen3.8-flash", served_node="gufo", session_log=log,
        )
    finally:
        log.close()

    events = [
        e for e in _log_events(log_dir) if e["event"] == "voter.answer"
    ]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["served_node"] == "gufo"
    assert data["model"] == "qwen3.8-flash"
    assert data["raw_answer"] == answer
    assert data["request_sent"] is True
    rows = {r["row_id"]: r for r in data["rows"]}
    assert rows["AC2-c2"]["holds"] is False
    assert rows["AC2-c2"]["reason"] == "lenient edit"


async def test_voter_no_structured_output_denial_also_logs(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(log_dir))
    pkg = _build(tmp_path)
    complete = FakeComplete(None, fail=RelaySchemaRejected("rejected (capability_not_supported)"))
    log = SessionLogManager().open_session("voter-test", "role-voter")
    try:
        await run_voter(
            pkg, complete, system_prompt=SYSTEM_PROMPT,
            model="qwen3.8-flash", served_node="gufo", session_log=log,
        )
    finally:
        log.close()
    events = [e for e in _log_events(log_dir) if e["event"] == "voter.answer"]
    assert len(events) == 1
    data = events[0]["data"]
    assert data["served_node"] == "gufo"
    assert data["request_sent"] is False
    assert all(r["reason"] == "no-structured-output" for r in data["rows"])


def test_voter_rows_parses_case_table(tmp_path):
    pkg = _build(tmp_path)
    assert voter_rows(pkg) == ["AC2-c1", "AC2-c2", "AC2-c3"]


# ---------------------------------------------------------------------------
# AC7 live check — one call per model through the relay (deselected by default)
# ---------------------------------------------------------------------------

LIVE_MODELS = ["qwen3.8-flash", "qwen3.8-27b", "gemma-4-31b"]


@pytest.mark.live
@pytest.mark.parametrize("model", LIVE_MODELS)
def test_ac7_live_relay_returns_schema_valid_answer(tmp_path, model):
    """AC7: one live call per model returns an answer that parses against the
    voter schema. glm-4.7-flash is not served yet and is not included."""
    import asyncio
    import re

    pkg = _build(tmp_path)
    body = voter_request_body(pkg, SYSTEM_PROMPT, model)
    client = RelayCompletionClient(base_url=DEFAULT_RELAY_BASE_URL, timeout=300.0)
    raw = asyncio.run(client.complete(body))

    data = json.loads(raw)  # must parse at all
    assert isinstance(data.get("rows"), list) and data["rows"], raw[:200]
    for row in data["rows"]:
        assert isinstance(row["row_id"], str)
        assert isinstance(row["holds"], bool)
        assert isinstance(row["reason"], str)
    # Reason text stays one line (a schema deviation if broken; a padding
    # backend that ignores the "one line" ask fails here loudly).
    for row in data["rows"]:
        assert re.fullmatch(r"[^\n]*", row["reason"])


# ---------------------------------------------------------------------------
# AC3 / AC6 — the quorum runner
# ---------------------------------------------------------------------------

POOL = ["qwen3.8-flash", "qwen3.8-27b", "gemma-4-31b", "glm-4.7-flash"]
NODE_OF = {
    "qwen3.8-flash": "b65",
    "qwen3.8-27b": "b65",
    "gemma-4-31b": "xtx",
    "glm-4.7-flash": "xtx",
}


def _node_of(model: str) -> str:
    return NODE_OF[model]


class QuorumComplete:
    """Per-voter fake transport: answers each request from its own row order.

    ``denier`` is the ordinal (0-based) of the request that denies one row;
    ``unparseable`` the ordinal of one whose answer is not JSON; ``missing``
    the ordinal of one whose answer omits one row. ``codes`` maps a model
    name to the answer for that model, taking precedence. Records every
    body in call order.
    """

    def __init__(self, *, denier: int | None = None,
                 unparseable: int | None = None,
                 missing: int | None = None,
                 codes: dict[str, str] | None = None):
        self.denier = denier
        self.unparseable = unparseable
        self.missing = missing
        self.codes = codes or {}
        self.bodies: list[dict] = []

    def _rows_of(self, body: dict) -> list[str]:
        """The rows this voter must answer, in the order its request
        demands: the per-voter order block when the quorum appended one,
        else the case table (rendered last, table order)."""
        msg = body["messages"][1]["content"]
        marker = "in exactly the order listed here"
        if marker in msg:
            block = msg.split(marker, 1)[1].split(":", 1)[1]
            ids = []
            for line in block.splitlines():
                first = line.strip()
                if "|" in first or first.lower() in ("row_id", "id"):
                    break
                if first:
                    ids.append(first)
            assert ids, msg[-200:]
            return ids
        table = msg.split("=== case_table.md ===", 1)[1]
        ids = []
        for line in table.splitlines():
            first = line.split("|", 1)[0].strip()
            if "-" in first and first.lower() not in ("row_id", "id"):
                ids.append(first)
        assert ids, table[:200]
        return ids

    async def __call__(self, body: dict) -> str:
        n = len(self.bodies)
        self.bodies.append(body)
        rows = self._rows_of(body)
        code = self.codes.get(body["model"])
        if code is not None:
            return json.dumps({"rows": [
                {"row_id": row_id, "holds": holds,
                 "reason": "ok" if holds else "lenient edit"}
                for row_id, holds in _apply_code(rows, code)
            ]})
        if n == self.unparseable:
            return "looks fine to me"
        if n == self.missing:
            # Drop the row at a per-request position so each fake run
            # denies exactly one different row.
            drop = n % len(rows)
            return _good_answer(
                [r for i, r in enumerate(rows) if i != drop]
            )
        if n == self.denier:
            return json.dumps({"rows": [
                {"row_id": rows[0], "holds": False, "reason": "lenient edit"},
                *[{"row_id": r, "holds": True, "reason": "ok"} for r in rows[1:]],
            ]})
        return _good_answer(rows)


def _apply_code(rows: list[str], code: str):
    """Answer rows with *code*: 'pass' holds on all, 'deny' denies all,
    'deny_first' denies only the first row."""
    if code == "deny_first":
        return [(rows[0], False), *[(r, True) for r in rows[1:]]]
    return [(row_id, code != "deny") for row_id in rows]


def _table_with_rows(tmp_path: Path, count: int) -> Path:
    lines = ["row_id | input | expected | clause | layer"]
    for i in range(1, count + 1):
        lines.append(f"AC9-c{i} | input {i} | expected {i} | FR 9 | unit")
    path = tmp_path / "big_table.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_many(tmp_path: Path, rows: int, name: str = "pkg"):
    """A package whose case table has *rows* rows (N=7 needs > 3! = 6)."""
    return _build(tmp_path, case_table=_table_with_rows(tmp_path, rows),
                  out_dir=tmp_path / name)


async def test_quorum_all_pass_writes_quorum_json(tmp_path):
    """AC3-c1: N=3, every voter passes every row."""
    pkg = _build(tmp_path)
    complete = QuorumComplete()
    result = await run_quorum(
        pkg, complete, system_prompt=SYSTEM_PROMPT,
        voters=3, models=POOL, node_of=_node_of,
    )
    data = json.loads((pkg.directory / "quorum.json").read_text())
    assert data == result
    assert result["approved"] is True
    assert result["pass_counts"] == {
        "AC2-c1": 3, "AC2-c2": 3, "AC2-c3": 3,
    }
    assert result["denied_rows"] == []
    assert len(result["voter_results"]) == 3
    for vr in result["voter_results"]:
        assert vr["raw_answer"]  # per-voter raw answer recorded
        assert vr["served_node"] in NODE_OF.values()
        assert vr["model"] in POOL


async def test_quorum_one_denial_approved_false_with_voter_and_reason(tmp_path):
    """AC3-c2: N=3, one voter denies one row."""
    pkg = _build(tmp_path)
    complete = QuorumComplete(denier=1)
    result = await run_quorum(
        pkg, complete, system_prompt=SYSTEM_PROMPT,
        voters=3, models=POOL, node_of=_node_of,
    )
    denied = result["denied_rows"]
    assert len(denied) == 1
    assert denied[0]["row_id"] == "AC2-c2"  # voter 1's shuffled first row
    assert result["pass_counts"]["AC2-c2"] == 2
    assert result["approved"] is False
    deniers = denied[0]["denied_by"]
    assert deniers == [{"voter_index": 1, "reason": "lenient edit"}]


async def test_quorum_unparseable_voter_denies_its_rows(tmp_path):
    """AC3-c3: an unparseable answer denies that voter's rows."""
    pkg = _build(tmp_path)
    complete = QuorumComplete(unparseable=0)
    result = await run_quorum(
        pkg, complete, system_prompt=SYSTEM_PROMPT,
        voters=3, models=POOL, node_of=_node_of,
    )
    assert result["approved"] is False
    assert all(count == 2 for count in result["pass_counts"].values())
    for entry in result["denied_rows"]:
        assert entry["denied_by"] == [{"voter_index": 0, "reason": "unparseable"}]


async def test_quorum_missing_row_denied_with_reason_missing(tmp_path):
    """AC3-c3: a parseable answer omitting a row denies it with 'missing'."""
    pkg = _build(tmp_path)
    complete = QuorumComplete(missing=2)
    result = await run_quorum(
        pkg, complete, system_prompt=SYSTEM_PROMPT,
        voters=3, models=POOL, node_of=_node_of,
    )
    assert result["approved"] is False
    missing_rows = [
        e for e in result["denied_rows"]
        if e["denied_by"] == [{"voter_index": 2, "reason": "missing"}]
    ]
    assert len(missing_rows) == 1  # exactly the one omitted row
    # The fake drops one row of voter 2's own order; the missing row is in
    # that voter's order and denied only by voter 2.
    assert missing_rows[0]["row_id"] in result["voter_results"][2]["row_order"]


@pytest.mark.parametrize("n", [2, 4])
async def test_quorum_even_n_config_error_before_any_call(tmp_path, n):
    """AC3-c4: an even N is a configuration error before any model call."""
    pkg = _build(tmp_path)
    complete = QuorumComplete()
    with pytest.raises(QuorumConfigError, match="odd"):
        await run_quorum(
            pkg, complete, system_prompt=SYSTEM_PROMPT,
            voters=n, models=POOL, node_of=_node_of,
        )
    assert complete.bodies == []
    assert not (pkg.directory / "quorum.json").exists()


def test_package_carries_row_ids_no_reread(tmp_path):
    """spc-960 G F1: row ids ride on the Package; the table is not re-read."""
    pkg = _build(tmp_path)
    assert pkg.row_ids == ("AC2-c1", "AC2-c2", "AC2-c3")
    assert voter_rows(pkg) == ["AC2-c1", "AC2-c2", "AC2-c3"]
    (pkg.directory / "case_table.md").unlink()
    assert voter_rows(pkg) == ["AC2-c1", "AC2-c2", "AC2-c3"]


@pytest.mark.parametrize("n", [5, 7])
async def test_quorum_full_pool_coverage_distinct_orders_one_adversarial(
    tmp_path, n
):
    """AC6-c1: N>=pool size: every model serves, orders pairwise distinct,
    exactly one adversarial voter."""
    pkg = _build_many(tmp_path, 5)
    complete = QuorumComplete()
    result = await run_quorum(
        pkg, complete, system_prompt=SYSTEM_PROMPT,
        voters=n, models=POOL, node_of=_node_of,
    )
    served = {vr["model"] for vr in result["voter_results"]}
    assert served == set(POOL)
    orders = [tuple(vr["row_order"]) for vr in result["voter_results"]]
    assert len(set(orders)) == n
    assert sum(1 for vr in result["voter_results"] if vr["adversarial"]) == 1


@pytest.mark.parametrize("diversify", [True, False])
async def test_quorum_rotation_covers_all_four_omissions(
    tmp_path, diversify
):
    """AC6-c2: N=3 over four consecutive runs: 3 distinct models each run,
    the omitted model differs run to run and covers all four, in both
    modes (at cursor c both omit pool[c % len(pool)])."""
    pkg = _build(tmp_path)
    rotation = RotationState()
    omitted = []
    for _ in range(4):
        complete = QuorumComplete()
        result = await run_quorum(
            pkg, complete, system_prompt=SYSTEM_PROMPT,
            voters=3, models=POOL, node_of=_node_of, rotation=rotation,
            diversify=diversify,
        )
        models = [vr["model"] for vr in result["voter_results"]]
        assert len(set(models)) == 3
        omitted.append(({*POOL} - set(models)).pop())
    assert len(set(omitted)) == 4
    assert set(omitted) == set(POOL)


async def test_quorum_calls_grouped_by_model_contiguous(tmp_path):
    """AC6-c3: calls run grouped by model, each model's group contiguous;
    quorum.json records model, served node and wall time per group."""
    pkg = _build_many(tmp_path, 5)
    complete = QuorumComplete()
    result = await run_quorum(
        pkg, complete, system_prompt=SYSTEM_PROMPT,
        voters=5, models=POOL, node_of=_node_of,
    )
    call_sequence = [b["model"] for b in complete.bodies]
    groups = result["groups"]
    assert sum(len(g["voter_indexes"]) for g in groups) == len(call_sequence)
    seen: list[str] = []
    for model in call_sequence:
        if not seen or seen[-1] != model:
            seen.append(model)
    assert len(seen) == len(set(seen)), "a model's calls were not contiguous"
    for g in groups:
        assert g["served_nodes"], g
        assert g["wall_time_s"] >= 0.0


async def test_quorum_group_order_places_models_as_given(tmp_path):
    """The caller orders the groups (xtx loads gemma once, glm once)."""
    pkg = _build_many(tmp_path, 5)
    complete = QuorumComplete()
    result = await run_quorum(
        pkg, complete, system_prompt=SYSTEM_PROMPT,
        voters=5, models=POOL, node_of=_node_of,
        group_order=list(reversed(POOL)),
    )
    assert [g["model"] for g in result["groups"]] == list(reversed(POOL))


async def test_quorum_wall_time_from_the_clock_seam(tmp_path):
    """Per-group wall time is measured with the injected clock."""
    pkg = _build(tmp_path)
    ticks = iter([0.0, 1.5, 2.0, 7.25, 9.0, 9.5])
    result = await run_quorum(
        pkg, QuorumComplete(), system_prompt=SYSTEM_PROMPT,
        voters=3, models=POOL, node_of=_node_of, clock=lambda: next(ticks),
    )
    assert [g["wall_time_s"] for g in result["groups"]] == [1.5, 5.25, 0.5]


@pytest.mark.parametrize(
    "short_pool",
    [
        ["qwen3.8-flash", "qwen3.8-27b", "gemma-4-31b"],       # no glm
        ["qwen3.8-flash", "gemma-4-31b", "glm-4.7-flash"],     # no 27b
        ["qwen3.8-27b", "gemma-4-31b", "glm-4.7-flash"],       # no flash
        ["qwen3.8-flash", "qwen3.8-27b", "glm-4.7-flash"],     # no gemma
    ],
)
async def test_quorum_short_pool_config_error_before_any_call(tmp_path, short_pool):
    """AC6-c4: a pool lacking any required model fails before any call."""
    pkg = _build(tmp_path)
    complete = QuorumComplete()
    with pytest.raises(QuorumConfigError, match="missing required model"):
        await run_quorum(
            pkg, complete, system_prompt=SYSTEM_PROMPT,
            voters=3, models=short_pool, node_of=_node_of,
        )
    assert complete.bodies == []


async def test_quorum_voters_see_the_same_row_set(tmp_path):
    """AC6-c5: every voter gets the same rows; only order and framing differ."""
    pkg = _build(tmp_path)
    complete = QuorumComplete()
    result = await run_quorum(
        pkg, complete, system_prompt=SYSTEM_PROMPT,
        voters=3, models=POOL, node_of=_node_of,
    )
    table = set(pkg.row_ids)
    for vr in result["voter_results"]:
        assert set(vr["row_order"]) == table
    messages = [b["messages"][1]["content"] for b in complete.bodies]
    for row_id in table:
        for msg in messages:
            assert f"| {row_id} |" in msg or msg.count(row_id) >= 1
    adversarial = [
        vr for vr in result["voter_results"] if vr["adversarial"]
    ]
    assert len(adversarial) == 1
    idx = adversarial[0]["index"]
    framed = [i for i, m in enumerate(messages) if "adversarial" in m.lower()]
    assert framed == [idx]


async def test_quorum_writes_the_session_log_per_voter(tmp_path, monkeypatch):
    """Every voter run logs its raw answer and served node (run_log seam)."""
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(log_dir))
    pkg = _build(tmp_path)
    log = SessionLogManager().open_session("quorum-test", "role-voter")
    try:
        await run_quorum(
            pkg, QuorumComplete(), system_prompt=SYSTEM_PROMPT,
            voters=3, models=POOL, node_of=_node_of, session_log=log,
        )
    finally:
        log.close()
    events = [e for e in _log_events(log_dir) if e["event"] == "voter.answer"]
    assert len(events) == 3
    assert {e["data"]["served_node"] for e in events} <= set(NODE_OF.values())
    assert all(e["data"]["raw_answer"] for e in events)


# ---------------------------------------------------------------------------
# AC4 — the measurement report
# ---------------------------------------------------------------------------


async def test_quorum_report_covers_packages_counts_and_models(tmp_path):
    """AC4-c1: report per package and count: caught, denied, unanimity;
    per model catch and deny rates. The fake answers per model: gemma
    always denies (catches the defect row, denies correct rows), glm
    denies only its first row, the Qwen models always pass."""
    defect = _build_many(tmp_path, 5, name="r5")
    correct = _build_many(tmp_path, 5, name="r7d")
    codes = {
        "gemma-4-31b": "deny",
        "glm-4.7-flash": "deny_first",
        "qwen3.8-flash": "pass",
        "qwen3.8-27b": "pass",
    }
    out = tmp_path / "report.md"
    report = await quorum_report(
        {"r5": {"package": defect, "defect_rows": ["AC9-c2"]},
         "fixed": {"package": correct, "defect_rows": []}},
        counts=(3, 5, 7),
        complete=QuorumComplete(codes=codes),
        system_prompt=SYSTEM_PROMPT,
        models=POOL,
        node_of=_node_of,
        out_path=out,
    )
    assert out.read_text(encoding="utf-8") == report
    # Coverage: both packages, all three counts, both run modes.
    for heading in ("## r5", "## fixed"):
        assert heading in report
    for count in (3, 5, 7):
        assert f"- N={count} diversified:" in report
        assert f"- N={count} plain:" in report
    # Per-model catch and deny rates.
    for model in POOL:
        assert f"{model}: voters" in report
    assert "catch rate" in report and "deny rate" in report
    r5_section = report.split("## r5", 1)[1].split("## fixed", 1)[0]
    fixed_section = report.split("## fixed", 1)[1]
    # Unanimity verdicts: r5 denied (gemma denies every row it sees).
    assert "approved=False" in r5_section
    # The defect row is caught in every r5 run: gemma serves every run
    # (N >= pool size covers the pool; with the fresh cursor 0 the r5
    # N=3 runs omit qwen3.8-flash (diversified, cursor 0) and
    # qwen3.8-27b (plain, cursor 1), so gemma is never omitted) and
    # denies every row.
    assert r5_section.count("defect rows caught 1/1 (AC9-c2)") == 6
    # Per-model rates on the r5 package, N=7 plain: 7 voters over the
    # 4-model pool round-robin — qwen3.8-flash, qwen3.8-27b and
    # gemma-4-31b get 2 voters, glm-4.7-flash 1. Gemma's 2 voters both
    # catch (deny the defect row) and deny a correct row; the Qwen
    # voters never deny.
    n7_plain = r5_section.split("- N=7 plain:")[1].split("- N=")[0]
    assert "gemma-4-31b: voters 2; catch rate 2/2 (100%)" in n7_plain
    assert "qwen3.8-flash: voters 2; catch rate 0/2 (0%)" in n7_plain
    assert "deny rate 0/2 (0%)" in n7_plain
    assert "qwen3.8-27b: voters 2; catch rate 0/2 (0%)" in n7_plain
    assert "glm-4.7-flash: voters 1" in n7_plain
    # The fixed package: no defect row exists, so catch is 0/0; the same
    # denying models deny correct rows there too.
    assert "defect rows caught 0/0" in fixed_section
    assert "correct rows denied" in fixed_section
    assert "approved=False" in fixed_section


async def test_quorum_report_correct_package_with_no_defect_rows(tmp_path):
    """A correct-code package has no defect rows: catch 0/0, no denies,
    unanimity approve."""
    correct = _build_many(tmp_path, 5, name="correct")
    passing = {m: "pass" for m in POOL}
    report = await quorum_report(
        {"fixed": {"package": correct, "defect_rows": []}},
        counts=(3,),
        complete=QuorumComplete(codes=passing),
        system_prompt=SYSTEM_PROMPT,
        models=POOL,
        node_of=_node_of,
        out_path=tmp_path / "report.md",
    )
    assert "defect rows: (none)" in report
    assert "defect rows caught 0/0" in report
    assert "correct rows denied 0" in report
    assert "approved=True" in report


# ---------------------------------------------------------------------------
# Regression — the 29-row table must not materialise 29! permutations
# (the OOM that killed the first kodiak-18 measurement run)
# ---------------------------------------------------------------------------


async def test_quorum_29_row_table_no_permutation_blowup(tmp_path):
    """A real-size case table (29 rows, kodiak-18) runs at counts 3, 5, 7.

    The first measurement run died because the row-order step built
    list(itertools.permutations(rows)) — 29! entries — before any model
    call. Orders for a large table must be drawn without materialising
    the permutation space: pairwise distinct (AC6), same row set per
    voter (AC6-c5), no model call needed beyond the per-voter fake.
    """
    pkg = _build_many(tmp_path, 29, name="kodiak18")
    assert len(pkg.row_ids) == 29
    for n in (3, 5, 7):
        complete = QuorumComplete()
        result = await run_quorum(
            pkg, complete, system_prompt=SYSTEM_PROMPT,
            voters=n, models=POOL, node_of=_node_of,
        )
        orders = [tuple(vr["row_order"]) for vr in result["voter_results"]]
        assert len(set(orders)) == n, f"orders not distinct at N={n}"
        for order in orders:
            assert set(order) == set(pkg.row_ids)
        assert result["approved"] is True


async def test_distinct_orders_large_table_stays_bounded(tmp_path):
    """_distinct_orders over 29 rows returns quickly, memory-bounded.

    Guards the exact defect: the count below the permutation total took
    the rng.sample(list(permutations(rows)), count) path. The call must
    finish in well under a second and the orders must be distinct.
    """
    import random
    import time

    from sr2_spectre.review import _distinct_orders

    rows = [f"AC21-c{i}" for i in range(1, 30)]
    started = time.monotonic()
    orders = _distinct_orders(rows, 7, random.Random(0))
    assert time.monotonic() - started < 1.0
    assert len(orders) == 7
    assert len({tuple(o) for o in orders}) == 7
    for order in orders:
        assert sorted(order) == sorted(rows)


async def test_quorum_report_29_row_table_completes(tmp_path):
    """quorum_report over a 29-row package at counts 3, 5, 7 — the call
    pattern the orchestrator's measurement run makes — completes and
    writes the report."""
    pkg = _build_many(tmp_path, 29, name="r5big")
    defect = pkg.row_ids[16]  # a real row id from the built table
    out = tmp_path / "report.md"
    report = await quorum_report(
        {"r5": {"package": pkg, "defect_rows": [defect]}},
        counts=(3, 5, 7),
        complete=QuorumComplete(),
        system_prompt=SYSTEM_PROMPT,
        models=POOL,
        node_of=_node_of,
        out_path=out,
    )
    assert out.exists()
    for count in (3, 5, 7):
        assert f"- N={count} diversified:" in report
        assert f"- N={count} plain:" in report
    # All-passing fake: no row denied, unanimity approve.
    assert "approved=True" in report
    assert "defect rows caught 0/1" in report


# ---------------------------------------------------------------------------
# Regression — rotation without caller bookkeeping (AC6-c2), and the served
# node recorded for the call that was served (AC3, AC6, llm-relay-1)
#
# Both caught defects hid behind the seam the tests used: the rotation test
# passed a RotationState explicitly, so nothing proved the default rotated;
# every served-node assertion read a static model -> node map, so a node
# read before the call (or shifted by one) looked correct.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_rotation(monkeypatch):
    """Give every test a fresh process-wide rotation cursor.

    The state that makes AC6-c2 hold across successive ``run_quorum`` and
    ``quorum_report`` calls is process-wide by design, so without this no
    test may depend on the cursor an earlier test left behind.
    """
    from sr2_spectre import review

    monkeypatch.setattr(review, "_DEFAULT_ROTATION", RotationState())


@pytest.mark.parametrize("diversify", [True, False])
async def test_quorum_rotation_across_successive_runs_without_caller_state(
    tmp_path, diversify
):
    """AC6-c2: four default N=3 runs omit all four pool models in order,
    in both modes.

    The measurement driver calls run_quorum once per process or once per
    count and keeps no state of its own. Passing a RotationState is an
    opt-out for a caller that persists the cursor, not a requirement: with
    none passed, four consecutive N=3 runs must still omit four different
    models. At cursor c both modes omit pool[c % len(pool)], so the
    sequence is the pool order.
    """
    pkg = _build(tmp_path)
    omitted = []
    for _ in range(4):
        result = await run_quorum(
            pkg, QuorumComplete(), system_prompt=SYSTEM_PROMPT,
            voters=3, models=POOL, node_of=_node_of, diversify=diversify,
        )
        models = [vr["model"] for vr in result["voter_results"]]
        assert len(set(models)) == 3, models
        omitted.append(({*POOL} - set(models)).pop())
    assert omitted == POOL, (
        f"the omitted model did not rotate across runs: {omitted}"
    )


async def test_quorum_report_rotates_the_omitted_model(tmp_path):
    """AC6-c2 / AC4: successive default quorum_report calls rotate the model.

    The report drives run_quorum once per count and mode and keeps no
    rotation bookkeeping of its own, so it must share the same state the
    rotation is measured across. Two default reports at counts=(3,) make
    four N=3 runs (diversified, plain, diversified, plain); at cursor c
    both modes omit pool[c % len(pool)], so the omitted models are the
    pool order.
    """
    pkg = _build_many(tmp_path, 5, name="rot")
    omitted = []
    for _ in range(2):
        await quorum_report(
            {"r5": {"package": pkg, "defect_rows": ["AC9-c2"]}},
            counts=(3,),
            complete=QuorumComplete(),
            system_prompt=SYSTEM_PROMPT,
            models=POOL,
            node_of=_node_of,
            out_path=tmp_path / "report.md",
        )
        for mode in ("diversified", "plain"):
            quorum = json.loads(
                (pkg.directory / f"quorum-3-{mode}.json").read_text(
                    encoding="utf-8"
                )
            )
            models = {vr["model"] for vr in quorum["voter_results"]}
            assert len(models) == 3, models
            omitted.append(({*POOL} - models).pop())
    assert omitted == POOL, (
        f"quorum_report did not rotate the omitted model: {omitted}"
    )


async def test_served_node_follows_the_call_not_the_model(tmp_path):
    """AC3/AC6: a node that changes per call is recorded per voter and per group.

    The relay names the serving node in the ``x-relay-node`` response
    header, so it is only known after the call (llm-relay-1). A fake whose
    node changes call to call catches both shapes of the old defect: a
    node read before the call recorded None first and then the previous
    call's node, shifted by one.
    """
    nodes = ["node-a", "node-b", "node-c"]

    class ShiftingClient:
        """Stands in for RelayCompletionClient: last_served_node is set
        by complete(), so it is None before the first call."""

        def __init__(self) -> None:
            self.last_served_node: str | None = None
            self.calls = 0

        async def complete(self, body: dict) -> str:
            self.last_served_node = nodes[self.calls % len(nodes)]
            self.calls += 1
            return await QuorumComplete()(body)

        def node_of(self, model: str) -> str | None:
            return self.last_served_node

    client = ShiftingClient()
    pkg = _build(tmp_path)
    result = await run_quorum(
        pkg, client.complete, system_prompt=SYSTEM_PROMPT,
        voters=3, models=POOL, node_of=client.node_of,
    )
    # Per voter: the node of that voter's own call, in call order.
    served = [vr["served_node"] for vr in result["voter_results"]]
    assert all(s is not None for s in served), served
    assert sorted(served) == sorted(nodes), served
    # Per group: the groups partition the voters, so their nodes are the
    # same set with no repeats — a shifted read would put one node twice
    # and lose another. And each group records exactly the nodes of its
    # own voters' calls.
    groups = result["groups"]
    assert sum(len(g["served_nodes"]) for g in groups) == 3, groups
    for g in groups:
        assert len(g["served_nodes"]) == len(g["voter_indexes"]), g
    by_index = {vr["index"]: vr for vr in result["voter_results"]}
    for g in groups:
        assert set(g["served_nodes"]) == {
            by_index[i]["served_node"] for i in g["voter_indexes"]
        }, g


async def test_run_voter_reads_node_of_after_the_call(tmp_path, monkeypatch):
    """AC3: run_voter's session-log event carries the node of its own call.

    The quorum aggregates after the calls, so a verdict whose node is
    patched afterwards cannot fix the log event: the voter itself must
    read node_of once the completion has returned.
    """
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(log_dir))
    pkg = _build(tmp_path)
    rows = voter_rows(pkg)
    answer = json.dumps({"rows": [
        {"row_id": r, "holds": True, "reason": "ok"} for r in rows
    ]})
    client = RelayCompletionClient.__new__(RelayCompletionClient)
    client.last_served_node = None

    async def complete(body: dict) -> str:
        # The relay reports the node in the response header: only known
        # once the call has returned.
        client.last_served_node = "node-late"
        return answer

    log = SessionLogManager().open_session("voter-node", "role-voter")
    try:
        verdict = await run_voter(
            pkg, complete, system_prompt=SYSTEM_PROMPT,
            model="qwen3.8-flash", node_of=lambda _m: client.last_served_node,
            session_log=log,
        )
    finally:
        log.close()
    events = [
        e for e in _log_events(log_dir) if e["event"] == "voter.answer"
    ]
    assert len(events) == 1
    assert events[0]["data"]["served_node"] == "node-late"
    assert verdict.served_node == "node-late"



# ---------------------------------------------------------------------------
# spc-967 — non-judging voters are listed apart, never counted as judgement
#
# The r5 N=3 plain run read qwen3.8-27b as catch 1/1 and deny 1/1 because
# the voter whose call failed (HTTP 400) denies every row without any
# judgement, and the rate counted that denial as the model's call. A voter
# that judged nothing is listed in its own line and kept out of the rates;
# it still blocks unanimity (fail closed, unchanged).
# ---------------------------------------------------------------------------


class QuorumCompleteFailing(QuorumComplete):
    """QuorumComplete where a named model never gives a judgement.

    ``raise_for`` maps a model name to the exception its call raises;
    ``empty_rows_for`` maps a model name to the answer ``{"rows": []}``
    (every row missing); ``garbage_for`` maps a model name to a non-JSON
    answer (every row unparseable). Every other model passes every row.
    """

    def __init__(self, *, raise_for: dict[str, Exception] | None = None,
                 empty_rows_for: tuple[str, ...] = (),
                 garbage_for: tuple[str, ...] = ()):
        super().__init__()
        self.raise_for = raise_for or {}
        self.empty_rows_for = empty_rows_for
        self.garbage_for = garbage_for

    async def __call__(self, body: dict) -> str:
        model = body["model"]
        self.bodies.append(body)
        failure = self.raise_for.get(model)
        if failure is not None:
            raise failure
        if model in self.empty_rows_for:
            return json.dumps({"rows": []})
        if model in self.garbage_for:
            return "looks fine to me"
        return _good_answer(self._rows_of(body))


def _report_run(tmp_path, complete, name="r5", defect_rows=("AC9-c2",)):
    return quorum_report(
        {name: {"package": _build_many(tmp_path, 5, name=name),
                "defect_rows": list(defect_rows)}},
        counts=(3,),
        complete=complete,
        system_prompt=SYSTEM_PROMPT,
        models=POOL,
        node_of=_node_of,
        out_path=tmp_path / "report.md",
    )


def _plain_voters(tmp_path, name="r5") -> dict[str, int]:
    """model -> voter index for the plain N=3 run the report just wrote.

    At N below the pool size one model is omitted and the assignment is
    shuffled, so a test reads the indexes back from ``quorum-3-plain.json``
    instead of naming them.
    """
    quorum = json.loads(
        (tmp_path / name / "quorum-3-plain.json").read_text(encoding="utf-8")
    )
    return {vr["model"]: vr["index"] for vr in quorum["voter_results"]}


async def test_report_error_voter_listed_apart_and_out_of_rates(tmp_path):
    """967-c1: a voter whose call failed is listed as non-judging and
    excluded from its model's catch and deny rates."""
    error_text = 'HTTP 400: {"code": "capability_not_supported"}'
    report = await _report_run(
        tmp_path,
        QuorumCompleteFailing(
            raise_for={"qwen3.8-flash": RuntimeError(error_text)}
        ),
    )
    # Listed apart: package, count, voter index, model, cause, error text.
    assert (
        f"- non-judging voter: r5 N=3 plain, "
        f"voter {_plain_voters(tmp_path)['qwen3.8-flash']}, "
        "model qwen3.8-flash, cause error, error " + error_text
        + " (excluded from the catch and deny rates below)" in report
    )
    # Excluded from the rates: the N=3 plain run gives qwen3.8-flash one
    # voter, so its rates read 0 judged, never the 1/1 the failed call
    # used to produce.
    plain = report.split("- N=3 plain:")[1].split("- N=")[0]
    assert "qwen3.8-flash: voters 0; catch rate n/a (0 judged)" in plain
    assert "deny rate n/a (0 judged)" in plain
    assert "catch rate 1/1" not in plain and "deny rate 1/1" not in plain
    # Fail closed, unchanged: the failed voter still blocks unanimity.
    assert "approved=False" in report


async def test_report_no_answer_voter_listed_as_non_judging(tmp_path):
    """967-c4: an empty answer (every row missing) and an unparseable
    response (every row unparseable) are non-judging with `error` empty."""
    report = await _report_run(
        tmp_path,
        QuorumCompleteFailing(
            empty_rows_for=("gemma-4-31b",), garbage_for=("glm-4.7-flash",)
        ),
    )
    voters = _plain_voters(tmp_path)
    # spc-972 folds in spc-984: spc-967 row 967-c4 (confirmed) names the
    # empty-answer cause `no-answer`; the unparseable voter keeps `unparseable`.
    for model, cause in (
        ("gemma-4-31b", "no-answer"),
        ("glm-4.7-flash", "unparseable"),
    ):
        assert (
            f"- non-judging voter: r5 N=3 plain, voter {voters[model]}, "
            f"model {model}, cause {cause} (excluded from the catch and "
            "deny rates below)" in report
        )
    plain = report.split("- N=3 plain:")[1].split("- N=")[0]
    assert "gemma-4-31b: voters 0; catch rate n/a (0 judged)" in plain
    assert "glm-4.7-flash: voters 0; catch rate n/a (0 judged)" in plain
    assert "catch rate 1/1" not in plain and "deny rate 1/1" not in plain
    # Fail closed, unchanged.
    assert "approved=False" in report


async def test_report_model_with_only_non_judging_voters_reads_na(tmp_path):
    """967-c2: every voter of one model non-judging -> `n/a (0 judged)`,
    not 1/1 and not the 0/0 dash."""
    report = await _report_run(
        tmp_path,
        QuorumCompleteFailing(
            raise_for={m: RuntimeError("boom") for m in POOL},
            empty_rows_for=POOL,
            garbage_for=POOL,
        ),
    )
    plain = report.split("- N=3 plain:")[1].split("- N=")[0]
    # Only the models serving that run report; the pool model it omits
    # has no line at all.
    for model in _plain_voters(tmp_path):
        assert f"{model}: voters 0; catch rate n/a (0 judged)" in plain
        assert "deny rate n/a (0 judged)" in plain
    # Not the 0/0 dash, and no rate built from a judged voter.
    assert "0/0" not in plain
    rates = "\n".join(
        line for line in plain.splitlines() if ": voters " in line
    )
    assert "1/1" not in rates
    # Every non-judging voter of the run is listed.
    assert report.count("- non-judging voter: r5 N=3 plain") == 3
    # Fail closed, unchanged.
    assert "approved=False" in report


async def test_report_unchanged_when_no_voter_failed(tmp_path):
    """967-c3: with every voter answering, the report reads as before spc-967."""
    codes = {
        "gemma-4-31b": "deny",
        "glm-4.7-flash": "deny_first",
        "qwen3.8-flash": "pass",
        "qwen3.8-27b": "pass",
    }
    report = await _report_run(tmp_path, QuorumComplete(codes=codes))
    assert "non-judging" not in report
    assert "n/a (0 judged)" not in report
    plain = report.split("- N=3 plain:")[1].split("- N=")[0]
    # The plain N=3 run at cursor 0 serves gemma, qwen3.8-flash and glm
    # (qwen3.8-27b is omitted); each gets one voter. Gemma denies every
    # row (catches the defect row, denies no correct row beyond it), glm
    # denies only its first row (AC9-c1, a correct row), qwen3.8-flash
    # passes every row.
    assert "gemma-4-31b: voters 1; catch rate 1/1 (100%)" in plain
    assert "glm-4.7-flash: voters 1; catch rate 0/1 (0%)" in plain
    assert "qwen3.8-flash: voters 1; catch rate 0/1 (0%)" in plain
    assert "qwen3.8-27b" not in plain


# ---------------------------------------------------------------------------
# spc-972 — the report announces every denial with OBJECTION!!! (AC8)
#
# Each voter gets a judge name fixed by its index (Phoenix, Edgeworth,
# Franziska, Godot, Mia, Gumshoe, Maya, cycling past seven) and every
# denied row gets an `OBJECTION!!!` line naming judge, voter index, model,
# row and reason. Presentation only: the name never enters a request body
# or quorum.json, and a non-judging voter (spc-967) never objects.
# ---------------------------------------------------------------------------


async def test_report_denial_announced_by_the_voters_judge(tmp_path):
    """AC8-c1: N=3, voter 1 denies row X with reason R -> the line naming
    Edgeworth (voter 1), that voter's model, the row and the reason."""
    report = await _report_run(tmp_path, QuorumComplete(denier=1))
    # The fake denies by request ordinal, so read the run voter 1 served
    # in (the diversified run) and the row, model and reason it recorded.
    quorum = json.loads(
        (tmp_path / "r5" / "quorum-3-diversified.json").read_text(encoding="utf-8")
    )
    denied = quorum["denied_rows"][0]
    voter1 = next(vr for vr in quorum["voter_results"] if vr["index"] == 1)
    assert [d["voter_index"] for d in denied["denied_by"]] == [1]
    assert (
        f"OBJECTION!!! Edgeworth (voter 1, {voter1['model']}) on "
        f"{denied['row_id']}: {denied['denied_by'][0]['reason']}" in report
    )
    # One line per (row, denying voter); exactly one denial in the run.
    assert report.count("OBJECTION!!!") == 1


async def test_report_no_denials_prints_no_objection(tmp_path):
    """AC8-c2: every voter passes every row -> no OBJECTION!!! line."""
    report = await _report_run(
        tmp_path, QuorumComplete(codes={m: "pass" for m in POOL})
    )
    assert "OBJECTION!!!" not in report
    assert "approved=True" in report


async def test_report_judge_flavor_is_presentation_only(tmp_path):
    """AC8-c3: the same quorum run with fake voters sends request bodies
    and writes quorum.json byte-identical to the pre-flavor output: no
    judge name anywhere in them."""
    pkg = _build_many(tmp_path, 5, name="flavor")
    complete = QuorumComplete(denier=1)
    report = await quorum_report(
        {"r5": {"package": pkg, "defect_rows": ["AC9-c2"]}},
        counts=(3,),
        complete=complete,
        system_prompt=SYSTEM_PROMPT,
        models=POOL,
        node_of=_node_of,
        out_path=tmp_path / "report.md",
    )
    # The flavor did render: the report carries the judge names.
    assert "OBJECTION!!!" in report
    # Every request body sent is plain: no judge name in any of them.
    bodies = json.dumps(complete.bodies)
    for name in JUDGE_NAMES:
        assert name not in bodies
    # quorum.json (each run's file) carries no judge name either.
    for quorum_file in sorted(pkg.directory.glob("quorum-*.json")):
        text = quorum_file.read_text(encoding="utf-8")
        for name in JUDGE_NAMES:
            assert name not in text
    # The body each voter receives is exactly what voter_request_body
    # builds for it — the renderer adds nothing on the way to the model.
    # Requests run grouped by model and the report runs both modes, so
    # match each body to the run (mode) whose row order it carries.
    runs = {
        mode: json.loads(
            (pkg.directory / f"quorum-3-{mode}.json").read_text(encoding="utf-8")
        )
        for mode in ("diversified", "plain")
    }
    # Every body a voter received is exactly what voter_request_body
    # builds for some voter of one of the two runs — the renderer adds
    # nothing on the way to the model — and every voter of both runs has
    # its body among those sent.
    def _expected(mode: str) -> list[dict]:
        quorum = runs[mode]
        return [
            voter_request_body(
                pkg, SYSTEM_PROMPT, vr["model"], rows=vr["row_order"],
                adversarial=vr["adversarial"],
            )
            for vr in quorum["voter_results"]
        ]

    all_expected = _expected("diversified") + _expected("plain")
    assert len(all_expected) == len(complete.bodies)
    for body in complete.bodies:
        assert body in all_expected, body["model"]
    for body in all_expected:
        assert body in complete.bodies


async def test_report_non_judging_voter_does_not_object(tmp_path):
    """AC8-c4: a non-judging voter (call failed, empty answer, unparseable
    answer) is listed as non-judging and gets no OBJECTION!!! line."""
    report = await _report_run(
        tmp_path,
        QuorumCompleteFailing(
            raise_for={"qwen3.8-flash": RuntimeError("boom")},
            empty_rows_for=("gemma-4-31b",),
            garbage_for=("glm-4.7-flash",),
        ),
    )
    assert report.count("- non-judging voter: r5 N=3 plain") == 3
    assert "OBJECTION!!!" not in report


def test_judge_names_by_index_cycle_past_seven():
    """The seven judges in order, cycling from index 7 on."""
    assert JUDGE_NAMES == (
        "Phoenix", "Edgeworth", "Franziska", "Godot", "Mia", "Gumshoe", "Maya",
    )
    assert [judge_name(i) for i in range(9)] == [
        "Phoenix", "Edgeworth", "Franziska", "Godot", "Mia", "Gumshoe", "Maya",
        "Phoenix", "Edgeworth",
    ]
