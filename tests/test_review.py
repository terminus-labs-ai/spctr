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
    QuorumConfigError,
    RelayCompletionClient,
    RelaySchemaRejected,
    RotationState,
    SecretPathError,
    VOTER_SCHEMA,
    build_package,
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


async def test_quorum_rotation_covers_all_four_omissions(tmp_path):
    """AC6-c2: N=3 over four consecutive runs: 3 distinct models each run,
    the omitted model differs run to run and covers all four."""
    pkg = _build(tmp_path)
    rotation = RotationState()
    omitted = []
    for _ in range(4):
        complete = QuorumComplete()
        result = await run_quorum(
            pkg, complete, system_prompt=SYSTEM_PROMPT,
            voters=3, models=POOL, node_of=_node_of, rotation=rotation,
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
    # (N >= pool size covers the pool; N=3 plain fills round-robin and
    # lands on gemma at index 2) and denies every row.
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
