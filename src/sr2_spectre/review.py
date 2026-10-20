"""Review package builder and single-call voter (spc-960, spec slice 1).

Two halves of the local-role-runs spec (``local-role-runs.md`` FR 1-4, NFR):

* ``build_package`` — writes a deterministic package directory (``manifest.json``
  plus the files the manifest names), records a rendered-size estimate, and
  refuses over-limit packages and secret-looking input paths. It reads only
  the paths its inputs name, and runs no command: the diff between base and
  head is an input like the run output.
* ``run_voter`` — one model call with no tools offered, whose answer is
  constrained to the per-row voter schema. The schema goes on every request
  as an OpenAI ``response_format`` of type ``json_schema`` (strict,
  ``additionalProperties: false`` on every object) whatever the model name;
  no backend is inferred from the name (spec AC7). The default transport
  POSTs the body directly to the relay's ``/v1/chat/completions``: sr2's
  model-call layer cannot carry ``response_format``. A response that does
  not parse is a denial of every row with reason ``unparseable``; a
  parseable response missing a row is a denial of that row with reason
  ``missing``. A schema constraint the relay or backend rejects (HTTP 400,
  e.g. ``capability_not_supported`` or ``invalid_response_format``) fails
  closed: every row denied with reason ``no-structured-output`` and no
  unconstrained retry sent. Every run writes its raw answer and served node
  to the session log through the ``run_log`` seam.

The quorum runner (FR 5-7) is spc-961; this module only provides the single
voter call it groups.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
from sr2_spectre.run_log import SessionLog

__all__ = [
    "DEFAULT_SECRET_PATTERNS",
    "DEFAULT_MAX_RENDERED_BYTES",
    "DEFAULT_RELAY_BASE_URL",
    "PackageTooLargeError",
    "SecretPathError",
    "RelayCompletionClient",
    "build_package",
    "run_voter",
    "voter_request_body",
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rendered_prompt(package_dir: Path, manifest: dict[str, Any]) -> str:
    """Render the package body: every manifest file under a path header.

    The single rendering shared by the size estimate recorded in the
    manifest and (via ``voter_request_body``) the voter's user message, so
    the two can never disagree.
    """
    parts: list[str] = [
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

    entries: list[dict[str, str]] = []
    placed: list[tuple[Path, str]] = [
        (case_table, "case_table.md"),
        (diff, "diff.patch"),
        (run_output, "run.txt"),
    ] + [(fi.path, fi.dest) for fi in files] + [(fi.path, fi.dest) for fi in evidence]

    directory = out_dir or Path("review-package")
    directory.mkdir(parents=True, exist_ok=True)
    # Only files the builder itself places are ever removed, so an out_dir
    # holding anything else is left alone rather than wiped.
    known = {"manifest.json", *(dest for _, dest in placed)}
    for child in directory.rglob("*"):
        if child.is_file() and child.relative_to(directory).as_posix() not in known:
            child.unlink()

    for src, dest in placed:
        payload = Path(src).read_bytes()  # the one read of each input source
        target = directory / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        entries.append({"path": dest, "sha256": _sha256_bytes(payload)})

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


# Output room: enough that a backend padding the JSON with whitespace cannot
# truncate it (relay probe 2026-10-19: max_tokens 60 truncated, 200 stopped
# cleanly on a two-row answer; 128 per row with a 2048 floor).
VOTER_MAX_TOKENS_PER_ROW: int = 128
VOTER_MAX_TOKENS_FLOOR: int = 2048

# The relay's OpenAI-compatible root (spec Dependencies). Until llm-relay-1
# reports the serving node on every completion, ``served_node`` stays a
# caller-supplied value.
DEFAULT_RELAY_BASE_URL: str = "http://localhost:11434"


def voter_request_body(
    package: Package, system_prompt: str, model: str
) -> dict[str, Any]:
    """The OpenAI chat-completions body for one voter call (spec AC7).

    Carries the voter schema as a ``response_format`` of type
    ``json_schema`` with ``strict: true`` whatever the model name, offers no
    tools — the ``tools`` field is absent entirely, since some backends
    reject an empty tools array with HTTP 400 — and asks for enough output
    tokens that whitespace padding cannot truncate the answer. Thinking is
    off (the workflow's role calls run without it).
    """
    expected = voter_rows(package)
    user_message = (
        "Review the package below against its case table. Answer with the "
        "voter JSON only.\n\n"
        + _rendered_prompt(package.directory, package.manifest)
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max(
            VOTER_MAX_TOKENS_FLOOR, VOTER_MAX_TOKENS_PER_ROW * len(expected)
        ),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "voter_answer",
                "strict": True,
                "schema": VOTER_SCHEMA,
            },
        },
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }


class RelayCompletionClient:
    """Minimal OpenAI-compatible client that carries ``response_format``.

    sr2's model-call layer cannot send ``response_format``, so the voter
    POSTs through the relay's ``/v1/chat/completions`` directly (contract:
    Implementation notes). ``complete`` returns the completion text, or
    raises :class:`RelaySchemaRejected` when the relay or backend answers
    the constrained request with an HTTP 400 that rejects the schema
    constraint itself (spc-966). Any other 400 — e.g. "tools must not be an
    empty array" — is an ordinary request failure raised as
    ``aiohttp.ClientResponseError``, not a schema rejection. It sends
    exactly one request per call: no retry, constrained or not.
    """

    def __init__(
        self, base_url: str = DEFAULT_RELAY_BASE_URL, timeout: float = 180.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def complete(self, body: dict[str, Any]) -> str:
        """POST *body* to ``/v1/chat/completions``; return the answer text."""
        client_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions", json=body
            ) as response:
                payload = await response.read()
                if response.status == 400:
                    if _is_schema_constraint_rejection(payload):
                        raise RelaySchemaRejected(
                            _error_text(payload), _error_code(payload)
                        )
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=_error_text(payload),
                    )
                if response.status >= 400:
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=_error_text(payload),
                    )
                data = json.loads(payload)
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""


class RelaySchemaRejected(RuntimeError):
    """The relay or backend rejected the schema constraint (HTTP 400).

    ``error_code`` is the relay's error code when the body carried one
    (e.g. ``capability_not_supported``, ``invalid_response_format``).
    """

    def __init__(self, message: str, error_code: str = "") -> None:
        super().__init__(message)
        self.error_code = error_code


def _error_text(payload: bytes) -> str:
    try:
        error = json.loads(payload).get("error") or {}
    except (json.JSONDecodeError, AttributeError):
        return payload.decode("utf-8", errors="replace")[:500]
    message = str(error.get("message", ""))[:500]
    code = str(error.get("code", ""))
    return f"{message} ({code})" if code else message


def _error_dict(payload: bytes) -> dict[str, Any]:
    """The ``error`` object of an error response body, or {} if unreadable."""
    try:
        error = json.loads(payload).get("error")
    except (json.JSONDecodeError, AttributeError):
        return {}
    return error if isinstance(error, dict) else {}


def _error_code(payload: bytes) -> str:
    return str(_error_dict(payload).get("code", ""))


# Error codes the relay or backend uses to reject the schema constraint
# itself (spc-966; relay probe 2026-10-20).
_SCHEMA_REJECTION_CODES = frozenset(
    {"capability_not_supported", "invalid_response_format"}
)


def _is_schema_constraint_rejection(payload: bytes) -> bool:
    """Does this HTTP 400 body reject the schema constraint (spec AC5, AC7)?

    True only when the error names a schema-constraint rejection: one of
    the relay's codes (``capability_not_supported``,
    ``invalid_response_format``) or a message naming ``response_format``
    or the JSON schema. Any other 400 (e.g. "tools must not be an empty
    array") is an ordinary request failure, not a schema rejection, and
    must not route the voter run down the ``no-structured-output`` path.
    """
    error = _error_dict(payload)
    if str(error.get("code", "")) in _SCHEMA_REJECTION_CODES:
        return True
    message = str(error.get("message", "")).lower()
    return "response_format" in message or "json schema" in message


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
    request_sent: bool
    error: str = ""


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


async def run_voter(
    package: Package,
    complete: Callable[[dict[str, Any]], Awaitable[str]],
    *,
    system_prompt: str,
    model: str,
    served_node: str | None = None,
    session_log: SessionLog | None = None,
) -> VoterVerdict:
    """One tool-less model call constrained to the voter schema (FR 3-4, AC5, AC7).

    ``complete`` takes the OpenAI body from :func:`voter_request_body` and
    returns the completion text — production passes a
    :class:`RelayCompletionClient`; tests pass a fake that records bodies.
    The schema constraint rides on every request whatever the model name;
    nothing is inferred from the name. The request carries no tools.

    A schema constraint the relay or backend rejects (HTTP 400, surfaced as
    :class:`RelaySchemaRejected`) denies every row with reason
    ``no-structured-output`` and sends no unconstrained retry. Any other
    failure from ``complete`` (transport, timeout) denies every row with
    reason ``unparseable`` and retries nothing: one call per voter run. A
    response that does not parse denies every row with reason
    ``unparseable``; a parseable response missing a row denies that row with
    ``missing``. The raw answer and served node are written to the session
    log through the ``run_log`` seam (``session_log`` may be None for a
    no-op).
    """
    expected = voter_rows(package)

    def _verdict(
        rows: list[dict[str, Any]],
        raw: str,
        request_sent: bool,
        error: str,
        elapsed: float,
    ) -> VoterVerdict:
        if session_log is not None:
            session_log.append(
                "voter.answer",
                {
                    "model": model,
                    "served_node": served_node,
                    "raw_answer": raw,
                    "request_sent": request_sent,
                    "duration_s": round(elapsed, 3),
                    "error": error,
                    "rows": rows,
                },
            )
        return VoterVerdict(
            rows=rows,
            raw_answer=raw,
            served_node=served_node,
            model=model,
            request_sent=request_sent,
            error=error,
        )

    def _deny(reason: str, error: str = "", request_sent: bool = False) -> VoterVerdict:
        return _verdict(
            [_row_verdict(row_id, False, reason) for row_id in expected],
            "",
            request_sent,
            error,
            0.0,
        )

    body = voter_request_body(package, system_prompt, model)
    started = time.monotonic()
    try:
        raw = await complete(body)
    except RelaySchemaRejected as exc:
        # Fail closed (AC5): the schema constraint itself was rejected.
        # Every row is denied; no unconstrained retry is ever sent.
        return _deny("no-structured-output", error=str(exc))
    except Exception as exc:  # noqa: BLE001 — any other failure denies every row
        return _deny("unparseable", error=str(exc), request_sent=True)
    elapsed = time.monotonic() - started

    verdicts = _parse_answer(raw, expected)
    if verdicts is None:
        # A response that does not parse denies every row (spec FR 3).
        return _verdict(
            [_row_verdict(row_id, False, "unparseable") for row_id in expected],
            raw,
            True,
            "",
            elapsed,
        )
    return _verdict(verdicts, raw, True, "", elapsed)
