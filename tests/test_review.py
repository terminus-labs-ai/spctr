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

import pytest

from sr2_spectre.review import (
    DEFAULT_RELAY_BASE_URL,
    DEFAULT_SECRET_PATTERNS,
    PackageInput,
    PackageTooLargeError,
    RelayCompletionClient,
    RelaySchemaRejected,
    SecretPathError,
    VOTER_SCHEMA,
    build_package,
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


def _good_answer() -> str:
    return json.dumps({"rows": [
        {"row_id": "AC2-c1", "holds": True, "reason": "ok"},
        {"row_id": "AC2-c2", "holds": True, "reason": "ok"},
        {"row_id": "AC2-c3", "holds": True, "reason": "ok"},
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
    assert not body["tools"]
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
