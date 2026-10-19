"""Tests for the review package builder and single-call voter (spc-960).

Unit tests run with no LLM and no network: the voter is exercised against a
fake backend, and the AC1 determinism test uses the frozen kodiak-18-c1
inputs when present (skipping cleanly when absent). One e2e test hits the
local relay and is deselected by default via the ``live`` marker.

Run: PYTHONPATH=src .venv/bin/python -m pytest tests/test_review.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sr2 import TextBlock, TokenUsage
from sr2.protocols.llm import CompletionResponse
from sr2_spectre.review import (
    DEFAULT_SECRET_PATTERNS,
    PackageInput,
    PackageTooLargeError,
    SecretPathError,
    build_package,
    run_voter,
    structured_output_mode,
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
# Fake backend for the voter
# ---------------------------------------------------------------------------


class FakeLLM:
    """One-shot LLM stub: records the request, returns a canned answer."""

    def __init__(self, answer: str | None, *, fail: Exception | None = None):
        self.answer = answer
        self.fail = fail
        self.requests: list = []
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        return CompletionResponse(
            id="fake",
            content=[TextBlock(text=self.answer or "")],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request):
        raise NotImplementedError


def _log_dir(tmp_path: Path) -> Path:
    return SessionLogManager().directory


def _log_events(tmp_path: Path) -> list[dict]:
    events = []
    for path in sorted(_log_dir(tmp_path).glob("*.jsonl")):
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
    """AC5: build_package reads no path outside those its inputs name."""
    inputs = _make_inputs(tmp_path)
    files = [PackageInput(inputs["test_file"], "tests/test_flag.py")]
    case_table = _write_case_table(tmp_path)
    sources = [case_table, inputs["test_file"], inputs["diff"], inputs["run_txt"]]

    read: list[Path] = []
    original = Path.read_bytes

    def tracking_read(self: Path, *a, **k):
        read.append(Path(self).resolve())
        return original(self, *a, **k)

    monkeypatch.setattr(Path, "read_bytes", tracking_read)
    try:
        build_package(
            bead="spc-960", base="base", head="head",
            case_table=case_table, files=files,
            diff=inputs["diff"], run_output=inputs["run_txt"],
            out_dir=tmp_path / "pkg",
        )
    finally:
        monkeypatch.undo()

    allowed = {Path(p).resolve() for p in sources}
    pkg_dir = (tmp_path / "pkg").resolve()
    for path in read:
        assert path in allowed or pkg_dir in path.parents, path


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
    llm = FakeLLM(json.dumps({"rows": [
        {"row_id": "AC2-c1", "holds": True, "reason": "ok"},
        {"row_id": "AC2-c2", "holds": True, "reason": "ok"},
        {"row_id": "AC2-c3", "holds": True, "reason": "ok"},
    ]}))
    verdict = await run_voter(
        pkg, llm, system_prompt=SYSTEM_PROMPT, model="b65-vllm-qwen3.8",
    )
    assert llm.calls == 1
    request = llm.requests[0]
    assert request.tools is None
    assert verdict.request_sent is True
    assert all(row["holds"] for row in verdict.rows)


async def test_voter_unparseable_answer_denies_all_rows(tmp_path):
    pkg = _build(tmp_path)
    llm = FakeLLM("I think the code looks fine, but the quoting is off.")
    verdict = await run_voter(
        pkg, llm, system_prompt=SYSTEM_PROMPT, model="b65-vllm-qwen3.8",
    )
    assert llm.calls == 1
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
    llm = FakeLLM(answer)
    verdict = await run_voter(
        pkg, llm, system_prompt=SYSTEM_PROMPT, model="b65-vllm-qwen3.8",
    )
    by_id = {row["row_id"]: row for row in verdict.rows}
    assert by_id["AC2-c1"]["holds"] is True
    assert by_id["AC2-c2"]["holds"] is False
    assert by_id["AC2-c2"]["reason"] == "missing"
    assert by_id["AC2-c3"]["holds"] is True


async def test_voter_call_failure_denies_all_rows(tmp_path):
    pkg = _build(tmp_path)
    llm = FakeLLM(None, fail=RuntimeError("connection reset"))
    verdict = await run_voter(
        pkg, llm, system_prompt=SYSTEM_PROMPT, model="b65-vllm-qwen3.8",
    )
    assert llm.calls == 1
    assert all(row["holds"] is False for row in verdict.rows)
    assert all(row["reason"].startswith("unparseable") for row in verdict.rows)


# ---------------------------------------------------------------------------
# AC5 — no-structured-output fail-closed, secret refusal, session log
# ---------------------------------------------------------------------------


def test_structured_output_mode_by_model_name():
    assert structured_output_mode("b65-vllm-qwen3.8") == "guided-json"
    assert structured_output_mode("xtx-llama-gemma-4-31b") == "gbnf"
    # Unknown backend (e.g. desktop LM Studio): no mode, fail closed.
    assert structured_output_mode("qwen3.8-flash") is None


async def test_voter_no_structured_output_denies_all_and_sends_no_request(tmp_path):
    pkg = _build(tmp_path)
    llm = FakeLLM("{}")
    verdict = await run_voter(
        pkg, llm, system_prompt=SYSTEM_PROMPT,
        model="qwen3.8-flash", served_node="gufo",
    )
    assert llm.calls == 0, "no request must be sent without a structured-output mode"
    assert verdict.request_sent is False
    assert len(verdict.rows) == 3
    for row in verdict.rows:
        assert row["holds"] is False
        assert row["reason"] == "no-structured-output"


async def test_voter_writes_raw_answer_and_served_node_to_session_log(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(log_dir))
    pkg = _build(tmp_path)
    answer = json.dumps({"rows": [
        {"row_id": "AC2-c1", "holds": True, "reason": "ok"},
        {"row_id": "AC2-c2", "holds": False, "reason": "lenient edit"},
        {"row_id": "AC2-c3", "holds": True, "reason": "ok"},
    ]})
    llm = FakeLLM(answer)
    log = SessionLogManager().open_session("voter-test", "role-voter")
    try:
        await run_voter(
            pkg, llm, system_prompt=SYSTEM_PROMPT,
            model="b65-vllm-qwen3.8", served_node="b65", session_log=log,
        )
    finally:
        log.close()

    events = _log_events(tmp_path)
    voter_events = [e for e in events if e["event"] == "voter.answer"]
    assert len(voter_events) == 1
    data = voter_events[0]["data"]
    assert data["served_node"] == "b65"
    assert data["model"] == "b65-vllm-qwen3.8"
    assert data["raw_answer"] == answer
    assert data["request_sent"] is True
    rows = {r["row_id"]: r for r in data["rows"]}
    assert rows["AC2-c2"]["holds"] is False
    assert rows["AC2-c2"]["reason"] == "lenient edit"


async def test_voter_no_structured_output_also_logs(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(log_dir))
    pkg = _build(tmp_path)
    llm = FakeLLM("{}")
    log = SessionLogManager().open_session("voter-test", "role-voter")
    try:
        await run_voter(
            pkg, llm, system_prompt=SYSTEM_PROMPT,
            model="qwen3.8-flash", served_node="gufo", session_log=log,
        )
    finally:
        log.close()
    events = _log_events(tmp_path)
    voter_events = [e for e in events if e["event"] == "voter.answer"]
    assert len(voter_events) == 1
    data = voter_events[0]["data"]
    assert data["request_sent"] is False
    assert all(r["reason"] == "no-structured-output" for r in data["rows"])


def test_voter_rows_parses_case_table(tmp_path):
    pkg = _build(tmp_path)
    assert voter_rows(pkg) == ["AC2-c1", "AC2-c2", "AC2-c3"]


# ---------------------------------------------------------------------------
# E2E: real relay call behind the live marker (deselected by default)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    not Path("/home/shepard/git/sr2-spectre/.venv").is_dir(),
    reason="main checkout venv not present",
)
def test_voter_e2e_live_relay(tmp_path, monkeypatch):
    """One live call through the relay against the real LiteLLMCallable.

    Proves the single-call path end to end: one request, no tools, a raw
    answer and served node in the session log. The relay's structured-output
    support is unverified, so the assertion on the answer is lenient: the
    call must succeed and be logged, not that the answer is valid JSON.
    """
    import asyncio

    from sr2.integrations.litellm import LiteLLMCallable

    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SR2_SESSION_LOG_DIR", str(log_dir))
    pkg = _build(tmp_path)
    llm = LiteLLMCallable(
        model="qwen3.8-flash",
        base_url="http://localhost:11434/v1",
        api_key="dummy",
    )
    log = SessionLogManager().open_session("voter-e2e", "role-voter")
    try:
        verdict = asyncio.run(run_voter(
            pkg, llm, system_prompt=SYSTEM_PROMPT,
            model="b65-vllm-qwen3.8", served_node="gufo", session_log=log,
        ))
    finally:
        log.close()

    assert verdict.request_sent is True
    assert len(verdict.rows) == 3
    for row in verdict.rows:
        assert set(row) == {"row_id", "holds", "reason"}
        assert isinstance(row["holds"], bool)
    events = _log_events(tmp_path)
    voter_events = [e for e in events if e["event"] == "voter.answer"]
    assert len(voter_events) == 1
    assert voter_events[0]["data"]["served_node"] == "gufo"
    assert voter_events[0]["data"]["raw_answer"].strip()
