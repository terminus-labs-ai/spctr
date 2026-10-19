"""Review package builder and single-call voter (spc-960, spec slice 1).

Two halves of the local-role-runs spec (``local-role-runs.md`` FR 1-4, NFR):

* ``build_package`` — writes a deterministic package directory (``manifest.json``
  plus the files the manifest names), records a rendered-size estimate, and
  refuses over-limit packages and secret-looking input paths. It reads only
  the paths its inputs name, and runs no command: the diff between base and
  head is an input like the run output.
* ``run_voter`` — one model call with no tools offered, whose answer is
  constrained to the per-row voter schema. A response that does not parse is
  a denial of every row with reason ``unparseable``; a parseable response
  missing a row is a denial of that row with reason ``missing``. A backend
  with no structured-output mode fails closed: every row denied with reason
  ``no-structured-output`` and no request sent. Every run writes its raw
  answer and served node to the session log through the ``run_log`` seam.

The quorum runner (FR 5-7) is spc-961; this module only provides the single
voter call it groups.
"""
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sr2 import Message, TextBlock
from sr2.protocols.llm import CompletionRequest, LLMCallable
from sr2_spectre.run_log import SessionLog

__all__ = [
    "DEFAULT_SECRET_PATTERNS",
    "DEFAULT_MAX_RENDERED_BYTES",
    "PackageTooLargeError",
    "SecretPathError",
    "build_package",
    "run_voter",
    "structured_output_mode",
    "voter_rows",
]

# ---------------------------------------------------------------------------
# Package builder
# ---------------------------------------------------------------------------

# Default secret pattern list (spec NFR; Diego, 2026-10-19).
DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
    ".env*",
    "*.pem",
    "*.key",
    "*.zuliprc",
    "*credentials*",
    "*secret*",
    "*token*",
)

# Configured rendered-size limit, in bytes of the rendered prompt.
DEFAULT_MAX_RENDERED_BYTES: int = 256 * 1024

# Token estimate convention, same as run_log.summarize_request.
_CHARS_PER_TOKEN = 4

# Case table: one line per row, "row_id | input | expected | clause | layer".
_ROW_ID_RE = re.compile(r"^\S+$")


class PackageTooLargeError(RuntimeError):
    """A package whose rendered size exceeds the configured limit."""


class SecretPathError(RuntimeError):
    """An input file whose path matches a configured secret pattern."""


@dataclass(frozen=True)
class PackageInput:
    """A file supplied to the package builder.

    ``path`` is the source to copy; ``dest`` is its path inside the package
    directory (e.g. ``tests/test_bd_gateway_flags.py``).
    """

    path: Path
    dest: str


@dataclass(frozen=True)
class Package:
    """A built review package directory."""

    directory: Path
    manifest: dict[str, Any]
    package_hash: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_id(row: str) -> str:
    """The first cell of a case-table row line (``row_id | ...``)."""
    return row.split("|", 1)[0].strip()


def _rendered_prompt(package_dir: Path, manifest: dict[str, Any]) -> str:
    """Render the voter prompt: every package file with a header.

    The same rendering is used for the size estimate recorded in the
    manifest and (by spc-961) for the user message of the voter call.
    """
    parts: list[str] = [
        "Review package",
        f"bead: {manifest['bead']}",
        f"base: {manifest['base']}",
        f"head: {manifest['head']}",
        "",
    ]
    for entry in manifest["files"]:
        parts.append(f"=== {entry['path']} ===")
        parts.append(
            (package_dir / entry["path"]).read_text(encoding="utf-8", errors="replace")
        )
        parts.append("")
    return "\n".join(parts)


def build_package(
    bead: str,
    base: str,
    head: str,
    case_table: Path,
    files: list[PackageInput],
    diff: Path,
    run_output: Path,
    evidence: list[PackageInput] = (),
    out_dir: Path | None = None,
    max_rendered_bytes: int = DEFAULT_MAX_RENDERED_BYTES,
    secret_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS,
) -> Package:
    """Build a deterministic review package directory (spec FR 1-2).

    Reads only the paths named by its inputs, runs no command (the diff is
    supplied), refuses input files whose path matches a secret pattern, and
    refuses a package whose rendered size exceeds ``max_rendered_bytes``.
    The package hash is the SHA-256 of ``manifest.json``.
    """
    sources = (
        [case_table, diff, run_output]
        + [fi.path for fi in files]
        + [fi.path for fi in evidence]
    )
    for path in sources:
        name = Path(path).name
        for pattern in secret_patterns:
            if fnmatch.fnmatch(name, pattern):
                raise SecretPathError(
                    f"refusing input file matching secret pattern "
                    f"'{pattern}': {path}"
                )

    directory = out_dir or Path("review-package")
    if directory.exists():
        # Never mix a stale tree with a new one.
        for child in sorted(directory.iterdir(), reverse=True):
            child.unlink()

    entries: list[dict[str, str]] = []
    placed: list[tuple[Path, str]] = [
        (case_table, "case_table.md"),
        (diff, "diff.patch"),
        (run_output, "run.txt"),
    ] + [(fi.path, fi.dest) for fi in files] + [(fi.path, fi.dest) for fi in evidence]
    for src, dest in placed:
        target = directory / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(src).read_bytes())
        entries.append({"path": dest, "sha256": _sha256_file(target)})

    manifest: dict[str, Any] = {
        "bead": bead,
        "base": base,
        "head": head,
        "files": entries,
        # Filled in after the render, below.
        "rendered_tokens_estimate": 0,
    }
    rendered = _rendered_prompt(directory, manifest)
    rendered_bytes = len(rendered.encode("utf-8"))
    if rendered_bytes > max_rendered_bytes:
        raise PackageTooLargeError(
            f"rendered size {rendered_bytes} bytes exceeds limit "
            f"{max_rendered_bytes} bytes"
        )
    manifest["rendered_tokens_estimate"] = rendered_bytes // _CHARS_PER_TOKEN

    manifest_path = directory / "manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return Package(
        directory=directory,
        manifest=manifest,
        package_hash=hashlib.sha256(manifest_bytes).hexdigest(),
    )


# ---------------------------------------------------------------------------
# Single-call voter
# ---------------------------------------------------------------------------

# Per-row voter schema (spec Interfaces): {"rows": [{"row_id", "holds", "reason"}]}
VOTER_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_id": {"type": "string"},
                    "holds": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["row_id", "holds", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}

# llama.cpp GBNF grammar for the same schema (compact; no nested objects).
VOTER_GBNF_GRAMMAR = """
root   ::= open rows close
open   ::= \\"{\\" \\"rows\\" \\":\\" \\"[\\"
rows   ::= row (comma row)*
row    ::= \\"{\\" \\"row_id\\" \\":\\" rid comma \\"holds\\" \\":\\" bool comma \\"reason\\" \\":\\" reason \\"\\"
comma  ::= \",\\"
rid    ::= rch+
bool   ::= \\"true\\" | \\"false\\"
reason ::= rch+
rch    ::= [\\x20-\\x7e]
close  ::= \\"\\"\\"
"""


def structured_output_mode(model: str) -> str | None:
    """The backend's structured-output mode, picked from the relay model name.

    ``"guided-json"`` for vLLM, ``"gbnf"`` for llama.cpp, ``None`` when the
    backend is unknown and no structured-output mode is available (fail
    closed; spec FR 3).
    """
    name = model.lower()
    if "vllm" in name:
        return "guided-json"
    if "llama" in name:
        return "gbnf"
    return None


def voter_rows(package: Package) -> list[str]:
    """Row ids from the package's case table, in table order."""
    text = (package.directory / "case_table.md").read_text(encoding="utf-8")
    rows: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        first = line.split("|", 1)[0].strip()
        if first.lower() in ("row_id", "id") or not _ROW_ID_RE.match(first):
            continue  # header line
        rows.append(first)
    return rows


@dataclass
class VoterVerdict:
    """The outcome of one single-call voter run."""

    rows: list[dict[str, Any]]
    raw_answer: str
    served_node: str | None
    model: str
    mode: str | None
    request_sent: bool


def _row_verdict(row_id: str, holds: bool, reason: str) -> dict[str, Any]:
    return {"row_id": row_id, "holds": bool(holds), "reason": reason}


def _parse_answer(raw: str, expected_rows: list[str]) -> list[dict[str, Any]] | None:
    """Parse a raw answer into per-row verdicts, or None if unparseable."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        return None
    by_id: dict[str, dict[str, Any]] = {}
    for entry in data["rows"]:
        if not isinstance(entry, dict):
            return None
        row_id = entry.get("row_id")
        if not isinstance(row_id, str) or "holds" not in entry:
            return None
        reason = entry.get("reason", "")
        by_id[row_id] = _row_verdict(row_id, entry["holds"], str(reason))
    # A missing row counts as denied (spec FR 4).
    return [
        by_id.get(row_id, _row_verdict(row_id, False, "missing"))
        for row_id in expected_rows
    ]


def _render_package_prompt(package: Package) -> str:
    manifest = package.manifest
    parts: list[str] = [
        "Review the package below. Answer with the voter JSON only.",
        f"bead: {manifest['bead']}",
        f"base: {manifest['base']}",
        f"head: {manifest['head']}",
        "",
    ]
    for entry in manifest["files"]:
        parts.append(f"=== {entry['path']} ===")
        parts.append(
            (package.directory / entry["path"]).read_text(encoding="utf-8", errors="replace")
        )
        parts.append("")
    return "\n".join(parts)


async def run_voter(
    package: Package,
    llm: LLMCallable,
    *,
    system_prompt: str,
    model: str,
    served_node: str | None = None,
    session_log: SessionLog | None = None,
    timeout: float = 120.0,
) -> VoterVerdict:
    """One tool-less model call constrained to the voter schema (spec FR 3-4).

    The structured-output mode is picked from the relay model name
    (``structured_output_mode``). With no mode, every row is denied with
    reason ``no-structured-output`` and no request is sent. The raw answer
    and served node are written to the session log through the ``run_log``
    seam (``session_log`` may be None for a no-op).
    """
    expected = voter_rows(package)

    def _log(data: Mapping[str, Any]) -> None:
        if session_log is not None:
            session_log.append("voter.answer", dict(data))

    def _deny(reason: str, *, mode: str | None = None, request_sent: bool = False) -> VoterVerdict:
        _log(
            {
                "model": model,
                "served_node": served_node,
                "raw_answer": "",
                "request_sent": request_sent,
                "rows": [
                    _row_verdict(row_id, False, reason) for row_id in expected
                ],
            }
        )
        return VoterVerdict(
            rows=[_row_verdict(r, False, reason) for r in expected],
            raw_answer="",
            served_node=served_node,
            model=model,
            mode=mode,
            request_sent=request_sent,
        )

    mode = structured_output_mode(model)
    if mode is None:
        return _deny("no-structured-output", mode=None)

    # No tools are offered: the request carries none and the structured
    # output mode constrains the answer to the voter schema.
    request = CompletionRequest(
        system=[TextBlock(text=system_prompt)],
        messages=[
            Message(role="user", content=[TextBlock(text=_render_package_prompt(package))]),
        ],
        tools=None,
    )
    # The structured-output payload is available on the VoterVerdict (mode
    # field) and the module-level VOTER_SCHEMA / VOTER_GBNF_GRAMMAR
    # constants; a backend wrapper (spc-961) uses them to emit
    # backend-specific kwargs (vLLM guided JSON / llama.cpp GBNF).

    started = time.monotonic()
    try:
        response = await _bounded(llm.complete(request), timeout)
    except Exception as exc:  # noqa: BLE001 — a failed call denies every row
        return _deny(f"unparseable: {exc}", mode=mode, request_sent=True)
    elapsed = time.monotonic() - started

    raw = "".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    )
    verdicts = _parse_answer(raw, expected)
    if verdicts is None:
        # A response that does not parse counts as a denial of every row.
        _log(
            {
                "model": model,
                "served_node": served_node,
                "raw_answer": raw,
                "request_sent": True,
                "duration_s": round(elapsed, 3),
                "rows": [
                    _row_verdict(row_id, False, "unparseable") for row_id in expected
                ],
            }
        )
        return VoterVerdict(
            rows=[_row_verdict(r, False, "unparseable") for r in expected],
            raw_answer=raw,
            served_node=served_node,
            model=model,
            mode=mode,
            request_sent=True,
        )

    _log(
        {
            "model": model,
            "served_node": served_node,
            "raw_answer": raw,
            "request_sent": True,
            "duration_s": round(elapsed, 3),
            "rows": verdicts,
        }
    )
    return VoterVerdict(
        rows=verdicts,
        raw_answer=raw,
        served_node=served_node,
        model=model,
        mode=mode,
        request_sent=True,
    )


def _bounded(coro, timeout: float):
    return asyncio.wait_for(coro, timeout)
