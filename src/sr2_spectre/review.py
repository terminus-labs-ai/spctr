"""Review package builder, single-call voter and quorum runner (spec slices 1-2).

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

The quorum runner (FR 5-7, spc-961) groups the single voter call: N odd
voters, each with a distinct row order and exactly one with the adversarial
framing (both rendered into the request, since role cards are out of scope),
calls grouped by model with each group contiguous, the served node read from
the relay's ``x-relay-node`` response header (llm-relay-1), per-group wall
time, and ``quorum.json`` beside the package. ``quorum_report`` tabulates
catch and deny rates across packages and voter counts for the kodiak-18-c1
measurement.
"""
from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import random
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
    "REQUIRED_POOL_MODELS",
    "QuorumConfigError",
    "PackageTooLargeError",
    "SecretPathError",
    "RelayCompletionClient",
    "RelaySchemaRejected",
    "RotationState",
    "build_package",
    "quorum_report",
    "run_quorum",
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
    # Row ids from the case table, in table order. Carried on the Package so
    # row handling never re-reads the file (spc-960 G finding F1).
    row_ids: tuple[str, ...] = ()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rendered_prompt(package_dir: Path, manifest: dict[str, Any]) -> str:
    """Render the package body: every manifest file under a path header.

    The single rendering shared by the size estimate recorded in the
    manifest and (via ``voter_request_body``) the voter's user message, so
    the two can never disagree. The case table renders last so the rows to
    answer are the tail of the message, whatever order a voter sees them in.
    """
    parts: list[str] = [
        f"bead: {manifest['bead']}",
        f"base: {manifest['base']}",
        f"head: {manifest['head']}",
        "",
    ]
    entries = sorted(
        manifest["files"],
        key=lambda e: 1 if e["path"] == "case_table.md" else 0,
    )
    for entry in entries:
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
    case_payload = b""
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
        if src is case_table:
            case_payload = payload
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
        # Row ids come from the bytes already read for the copy: row
        # handling never re-reads the case table (spc-960 G finding F1).
        row_ids=tuple(_parse_case_table_rows(
            case_payload.decode("utf-8", errors="replace")
        )),
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

# The relay's OpenAI-compatible root (spec Dependencies). The serving node
# rides the ``x-relay-node`` response header (llm-relay-1, closed); the
# ``served_node`` keyword remains as a caller override for a pinned node.
DEFAULT_RELAY_BASE_URL: str = "http://localhost:11434"


_VOTER_INSTRUCTION = (
    "Review the package below against its case table. Answer with the "
    "voter JSON only. Answer one entry per case-table row, in exactly the "
    "order the rows appear in the table."
)


def _package_message(package: Package) -> str:
    """The package body as the voter sees it: render plus a fixed prefix."""
    return _VOTER_INSTRUCTION + "\n\n" + _rendered_prompt(
        package.directory, package.manifest
    )


def _rows_block(rows: list[str]) -> str:
    """The per-voter rows block: the same rows, this voter's order."""
    return (
        "Answer one entry per case-table row, in exactly the order listed "
        "here (this order is yours alone):\n"
        + "\n".join(rows)
        + "\n"
    )


_ADVERSARIAL_HEADER = (
    "You are the adversarial voter. Assume at least one case-table row "
    "below is violated by this package. For every row, hunt specifically "
    "for a violation of the row's clause: deny a row the moment the "
    "package fails its expected outcome, and name the violating row in "
    "your reasons. Only answer holds=true on a row you could not break."
)


def _adversarial_footer(rows: list[str]) -> str:
    return (
        "Adversarial check before you answer: re-read every row above. "
        "Name in a reason any row whose expected outcome the package "
        f"violates. Rows: {', '.join(rows)}"
    )


def _voter_user_message(
    package: Package, rows: list[str], adversarial: bool
) -> str:
    """Render one voter's user message (FR 6 diversification content).

    The row order and the adversarial framing are request content: every
    voter receives the same rows, only the order and the framing differ.
    """
    if adversarial:
        return (
            _ADVERSARIAL_HEADER
            + "\n\n"
            + _package_message(package)
            + "\n\n"
            + _rows_block(rows)
            + "\n"
            + _adversarial_footer(rows)
        )
    if rows == voter_rows(package):
        # Table order: the plain package message already states it.
        return _package_message(package)
    return _package_message(package) + "\n\n" + _rows_block(rows)


def voter_request_body(
    package: Package,
    system_prompt: str,
    model: str,
    *,
    rows: list[str] | None = None,
    adversarial: bool = False,
) -> dict[str, Any]:
    """The OpenAI chat-completions body for one voter call (spec AC7).

    Carries the voter schema as a ``response_format`` of type
    ``json_schema`` with ``strict: true`` whatever the model name, offers no
    tools — the ``tools`` field is absent entirely, since some backends
    reject an empty tools array with HTTP 400 — and asks for enough output
    tokens that whitespace padding cannot truncate the answer. Thinking is
    off (the workflow's role calls run without it). ``rows`` overrides the
    case-table order this voter answers in (FR 6 diversification);
    ``adversarial`` adds the adversarial framing.
    """
    expected = list(rows) if rows is not None else voter_rows(package)
    user_message = _voter_user_message(package, expected, adversarial)
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
    exactly one request per call: no retry, constrained or not. The serving
    node the relay reports in the ``x-relay-node`` response header
    (llm-relay-1) lands in ``last_served_node`` after each call, success or
    error; ``node_of`` is the seam :func:`run_voter` reads after each call,
    so the node recorded for a voter is the one that served that voter's
    call.
    """

    def __init__(
        self, base_url: str = DEFAULT_RELAY_BASE_URL, timeout: float = 180.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Caller-settable: a quorum package of 1,000+ lines through a local
        # model can exceed the default on a cold model load.
        self.timeout = timeout
        self.last_served_node: str | None = None

    def node_of(self, _model: str) -> str | None:
        """The node that served the last call (``x-relay-node``).

        The seam :func:`run_voter` reads after each call, so the node
        recorded for a voter is the one that served that voter's call
        (llm-relay-1). The model argument is unused: the relay reports one
        node per response, whatever the model in the request.
        """
        return self.last_served_node

    async def complete(self, body: dict[str, Any]) -> str:
        """POST *body* to ``/v1/chat/completions``; return the answer text.

        The serving node the relay reports in the ``x-relay-node``
        response header (llm-relay-1) lands in ``last_served_node`` after
        the call, success or error.
        """
        client_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions", json=body
            ) as response:
                payload = await response.read()
                self.last_served_node = response.headers.get("x-relay-node")
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


def _parse_case_table_rows(text: str) -> list[str]:
    """Row ids from case-table text, in table order."""
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


def voter_rows(package: Package) -> list[str]:
    """Row ids from the package's case table, in table order.

    A Package built by :func:`build_package` carries its ``row_ids``; the
    case table is only re-read for a Package hand-constructed without them
    (spc-960 G finding F1).
    """
    if package.row_ids:
        return list(package.row_ids)
    return _parse_case_table_rows(
        (package.directory / "case_table.md").read_text(encoding="utf-8")
    )


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
    node_of: Callable[[str], str | None] | None = None,
    session_log: SessionLog | None = None,
    rows: list[str] | None = None,
    adversarial: bool = False,
) -> VoterVerdict:
    """One tool-less model call constrained to the voter schema (FR 3-4, AC5, AC7).

    ``complete`` takes the OpenAI body from :func:`voter_request_body` and
    returns the completion text — production passes a
    :class:`RelayCompletionClient`; tests pass a fake that records bodies.
    The schema constraint rides on every request whatever the model name;
    nothing is inferred from the name. The request carries no tools.

    The node that served this call is recorded two ways. ``served_node``
    is a caller override for a node known before the call (a pinned node).
    ``node_of`` is read *after* the call and wins over it: the relay
    reports the serving node in ``x-relay-node`` only once the response is
    in (llm-relay-1), so a node that changes per call is recorded for the
    call it served, never shifted to the next voter. Both the session-log
    event and the returned verdict carry the resolved node.

    A schema constraint the relay or backend rejects (HTTP 400, surfaced as
    :class:`RelaySchemaRejected`) denies every row with reason
    ``no-structured-output`` and sends no unconstrained retry. Any other
    failure from ``complete`` (transport, timeout) denies every row with
    reason ``unparseable`` and retries nothing: one call per voter run. A
    response that does not parse denies every row with reason
    ``unparseable``; a parseable response missing a row denies that row with
    ``missing``. The raw answer and served node are written to the session
    log through the ``run_log`` seam (``session_log`` may be None for a
    no-op). ``rows`` and ``adversarial`` pass through to
    :func:`voter_request_body` (the quorum's per-voter row order and
    framing); the parsed verdict is normalised to case-table order.
    """
    expected = voter_rows(package)
    answer_order = list(rows) if rows is not None else list(expected)

    def _verdict(
        row_verdicts: list[dict[str, Any]],
        raw: str,
        request_sent: bool,
        error: str,
        elapsed: float,
        node: str | None = None,
    ) -> VoterVerdict:
        # The node recorded for this voter is the one that served this
        # voter's call: ``node_of`` is read after the call (the relay
        # reports it in ``x-relay-node`` once the response is in,
        # llm-relay-1) and wins over the pre-call override.
        served = node if node is not None else served_node
        if session_log is not None:
            session_log.append(
                "voter.answer",
                {
                    "model": model,
                    "served_node": served,
                    "raw_answer": raw,
                    "request_sent": request_sent,
                    "duration_s": round(elapsed, 3),
                    "error": error,
                    "rows": row_verdicts,
                },
            )
        return VoterVerdict(
            rows=row_verdicts,
            raw_answer=raw,
            served_node=served,
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

    body = voter_request_body(
        package, system_prompt, model, rows=rows, adversarial=adversarial
    )
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
    # Read after the call: the relay names the serving node in the
    # response header, so this is the node that served *this* call even
    # when it changes call to call (llm-relay-1).
    served = node_of(model) if node_of is not None else None

    verdicts = _parse_answer(raw, answer_order)
    if verdicts is None:
        # A response that does not parse denies every row (spec FR 3).
        return _verdict(
            [_row_verdict(row_id, False, "unparseable") for row_id in expected],
            raw,
            True,
            "",
            elapsed,
            served,
        )
    # Rows come back in the order the voter was given; the verdict is
    # normalised to case-table order so voters aggregate row by row.
    verdicts.sort(key=lambda v: expected.index(v["row_id"]))
    return _verdict(verdicts, raw, True, "", elapsed, served)


# ---------------------------------------------------------------------------
# Quorum runner (FR 5-7, spc-961)
# ---------------------------------------------------------------------------

# The voter pool must contain all four (Diego, 2026-10-19; spec FR 6).
REQUIRED_POOL_MODELS: tuple[str, ...] = (
    "qwen3.8-flash",
    "qwen3.8-27b",
    "gemma-4-31b",
    "glm-4.7-flash",
)


class QuorumConfigError(ValueError):
    """A quorum configuration error, raised before any model call: an even
    N, a pool missing one of the four required models, or N above the
    number of distinct row orders (more voters than the case table has
    row permutations, FR 6)."""


@dataclass
class RotationState:
    """Rotation seam for runs below the pool size (FR 6, AC6-c2).

    A run omits one pool model; successive runs omit successive models so
    the omitted model rotates and covers the whole pool. One instance
    holds the cursor across the calls that make it; ``run_quorum`` and
    ``quorum_report`` share a process-wide default instance when no state
    is passed, so the rotation holds across successive calls within one
    process. A caller that runs the quorum in a fresh process per run (the
    measurement driver) persists ``omit_cursor`` beside its artifacts and
    constructs the state from it.
    """

    omit_cursor: int = 0

    def next_omit(self, pool: list[str]) -> str:
        model = pool[self.omit_cursor % len(pool)]
        self.omit_cursor += 1
        return model

    def advance(self) -> int:
        """Consume one rotation step and return the cursor before it.

        The plain (undiversified) fill uses this as its round-robin offset
        so the model that drops out of the last group rotates across runs
        too (FR 6): without it every plain run omits the same model and
        the rotation stalls.
        """
        cursor = self.omit_cursor
        self.omit_cursor += 1
        return cursor


# Process-wide default: the rotation holds across successive run_quorum
# and quorum_report calls within one process (FR 6, AC6-c2).
_DEFAULT_ROTATION = RotationState()


def _validate_quorum_config(
    package: Package,
    voters: int,
    models: list[str],
    group_order: list[str] | None = None,
) -> None:
    if voters % 2 == 0:
        raise QuorumConfigError(
            f"voters must be odd (spec FR 5), got {voters}"
        )
    if voters < 1:
        raise QuorumConfigError(f"voters must be at least 1, got {voters}")
    pool = list(dict.fromkeys(models))
    if not pool:
        raise QuorumConfigError("the voter pool is empty")
    missing = [m for m in REQUIRED_POOL_MODELS if m not in pool]
    if missing:
        raise QuorumConfigError(
            "voter pool is missing required model(s): "
            + ", ".join(missing)
            + "; the pool must contain "
            + ", ".join(REQUIRED_POOL_MODELS)
        )
    n_orders = 1
    for k in range(2, len(package.row_ids) + 1):
        n_orders *= k
        if n_orders >= voters:
            break
    else:
        n_orders = max(1, len(package.row_ids))
    if voters > n_orders:
        raise QuorumConfigError(
            f"{voters} voters need {voters} distinct row orders (FR 6) but "
            f"the case table has {len(package.row_ids)} row(s)"
        )
    if group_order is not None:
        unknown = [m for m in group_order if m not in pool]
        if unknown:
            raise QuorumConfigError(
                "group order names model(s) outside the voter pool: "
                + ", ".join(unknown)
            )


def _assign_models(
    voters: int,
    pool: list[str],
    rng: random.Random,
    rotation: RotationState | None,
    diversify: bool = True,
) -> list[str]:
    """One model per voter index.

    Diversified, N at or above the pool size: every pool model serves at
    least one voter (Diego's note: at least one voter per model). N below
    it: N distinct models serve, the omitted model rotating across runs
    (FR 6). Without diversification the pool fills round-robin from the
    rotation cursor, so the model that drops out of the last group rotates
    too and successive runs of either kind advance one shared cursor.
    """
    if voters >= len(pool):
        assignment = list(pool)
        assignment += [pool[i % len(pool)] for i in range(voters - len(pool))]
    elif diversify:
        omitted = (rotation or RotationState()).next_omit(pool)
        assignment = [m for m in pool if m != omitted]
    else:
        # Round-robin starting at the rotation cursor: the model the last
        # group runs short of rotates across runs (FR 6) instead of every
        # plain run omitting pool[0].
        offset = (rotation or RotationState()).advance()
        assignment = [pool[(offset + i) % len(pool)] for i in range(voters)]
    rng.shuffle(assignment)
    return assignment


def _order_cap(total: int) -> int:
    """Largest count materialised as an explicit permutation list."""
    return max(total, 40320)


def _distinct_orders(rows: list[str], count: int, rng: random.Random) -> list[list[str]]:
    """``count`` pairwise-distinct permutations of *rows*.

    Small tables (at most ``_ORDER_CAP`` permutations) deal the whole
    permutation list in a shuffled fixed order rather than drawing by
    rejection sampling: with few rows (3 rows, 6 permutations) sampling
    the last distinct order can take arbitrarily long, and a small table
    with more voters than permutations raises :class:`QuorumConfigError`
    in constant time instead.

    A large table (a real case table has tens of rows; 29! is ~10^31)
    cannot materialise ``len(rows)!`` permutations — building that list is
    what OOM-killed the first measurement run — so each order is an
    independent shuffle. A collision among a handful of voters over a
    factorially large order space is a birthday event (~10^-26 for 7
    voters over 29 rows); the loop re-shuffles on the rare collision and
    gives up with :class:`QuorumConfigError` rather than looping forever.
    """
    total = 1
    for k in range(1, len(rows) + 1):
        total *= k
    if count > total:
        raise QuorumConfigError(
            f"{count} voters need {count} distinct row orders (FR 6) but "
            f"{len(rows)} row(s) have only {total} permutation(s)"
        )
    if total <= _order_cap(len(rows)):
        dealt = list(itertools.permutations(rows))
        if count < total:
            dealt = rng.sample(dealt, count)
        rng.shuffle(dealt)
        return [list(order) for order in dealt]
    orders: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(orders) < count:
        if attempts >= 10 * count:
            raise QuorumConfigError(
                f"could not draw {count} distinct row orders from "
                f"{len(rows)} rows without repetition"
            )
        attempts += 1
        order = list(rows)
        rng.shuffle(order)
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        orders.append(order)
    return orders


def _plan_voters(
    package: Package,
    voters: int,
    pool: list[str],
    rng: random.Random,
    rotation: RotationState,
) -> list[dict[str, Any]]:
    """Assign each voter a model, a distinct row order, and the framing.

    Exactly one voter — chosen by the seeded rng — gets the adversarial
    framing; every voter gets a pairwise-distinct row order (FR 6).
    """
    assignment = _assign_models(voters, pool, rng, rotation)
    orders = _distinct_orders(voter_rows(package), voters, rng)
    adversarial_index = rng.randrange(voters)
    return [
        {
            "index": i,
            "model": assignment[i],
            "rows": orders[i],
            "adversarial": i == adversarial_index,
        }
        for i in range(voters)
    ]


def _group_order(models: list[str], order: list[str] | None) -> list[str]:
    """Models in group order: the caller's order when given (the xtx
    swap-cost note), else first-use order of the voter assignment. The
    caller's order is validated against the pool in
    :func:`_validate_quorum_config`, before any call."""
    if order is None:
        return list(dict.fromkeys(models))
    return list(order)


async def run_quorum(
    package: Package,
    complete: Callable[[dict[str, Any]], Awaitable[str]],
    *,
    system_prompt: str,
    voters: int = 3,
    models: list[str],
    node_of: Callable[[str], str | None] | None = None,
    diversify: bool = True,
    seed: int = 0,
    rotation: RotationState | None = None,
    group_order: list[str] | None = None,
    session_log: SessionLog | None = None,
    out_path: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run N single-call voters over a package and write ``quorum.json`` (FR 5-6).

    Configuration errors — an even N, a pool missing any of the four
    required models, N above the number of distinct row orders — raise
    :class:`QuorumConfigError` before any model call. With ``diversify``
    each voter gets a distinct row order and exactly one the adversarial
    framing (rendered into the request: the row order and the framing are
    request content, the role card is out of scope); ``diversify=False``
    gives every voter the table order and no framing (the measurement
    baseline). Calls run grouped by model, each model's group contiguous;
    ``group_order`` orders the groups so a node that swaps models loads
    each model once per quorum. The served node comes from the fake or
    client through ``node_of`` (the relay reports it in ``x-relay-node``
    after the call, llm-relay-1); it is read after each voter's call, so
    the node recorded for a voter is the one that served that voter's
    call. Per-group wall time is recorded. The result is written to
    ``out_path`` (default ``quorum.json`` beside the package) and
    returned.

    With ``diversify`` and N below the pool size, the omitted model
    rotates across runs (FR 6, AC6-c2): a caller-supplied ``rotation``
    drives it, and without one the process-wide default
    :class:`RotationState` does, so the rotation holds for successive
    calls within one process.
    """
    _validate_quorum_config(package, voters, models, group_order)
    pool = list(dict.fromkeys(models))
    rng = random.Random(seed)
    if diversify:
        plan = _plan_voters(
            package, voters, pool, rng, rotation or _DEFAULT_ROTATION
        )
    else:
        rows = voter_rows(package)
        assignment = _assign_models(voters, pool, rng, rotation, diversify=False)
        plan = [
            {
                "index": i,
                "model": assignment[i],
                "rows": list(rows),
                "adversarial": False,
            }
            for i in range(voters)
        ]

    by_model: dict[str, list[dict[str, Any]]] = {}
    for entry in plan:
        by_model.setdefault(entry["model"], []).append(entry)
    groups: list[dict[str, Any]] = []
    for model in _group_order(list(by_model), group_order):
        started = clock()
        for entry in by_model[model]:
            # run_voter reads node_of after the call, so the node recorded
            # for this voter (verdict, quorum.json and session log) is the
            # one that served this voter's call (llm-relay-1).
            verdict = await run_voter(
                package,
                complete,
                system_prompt=system_prompt,
                model=model,
                node_of=node_of,
                session_log=session_log,
                rows=entry["rows"],
                adversarial=entry["adversarial"],
            )
            entry["verdict"] = verdict
        groups.append(
            {
                "model": model,
                "voter_indexes": [e["index"] for e in by_model[model]],
                "served_nodes": sorted(
                    {
                        str(e["verdict"].served_node)
                        for e in by_model[model]
                        if e["verdict"].served_node is not None
                    }
                ),
                "wall_time_s": round(clock() - started, 3),
            }
        )

    rows = voter_rows(package)
    per_row = {
        row_id: sum(1 for e in plan if _holds(e, row_id)) for row_id in rows
    }
    denied_rows = [
        {
            "row_id": row_id,
            "denied_by": [
                {
                    "voter_index": e["index"],
                    "reason": _reason(e, row_id),
                }
                for e in plan
                if not _holds(e, row_id)
            ],
        }
        for row_id in rows
        if per_row[row_id] < voters
    ]
    result: dict[str, Any] = {
        "package": str(package.directory),
        "package_hash": package.package_hash,
        "voters": voters,
        "models": pool,
        "diversify": diversify,
        "approved": not denied_rows,
        "rows": rows,
        "pass_counts": per_row,
        "voter_results": [
            {
                "index": e["index"],
                "model": e["model"],
                "served_node": e["verdict"].served_node,
                "adversarial": e["adversarial"],
                "row_order": e["rows"],
                "raw_answer": e["verdict"].raw_answer,
                "request_sent": e["verdict"].request_sent,
                "error": e["verdict"].error,
                "rows": e["verdict"].rows,
            }
            for e in plan
        ],
        "denied_rows": denied_rows,
        "groups": groups,
    }
    destination = out_path or (package.directory / "quorum.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _holds(entry: dict[str, Any], row_id: str) -> bool:
    for row in entry["verdict"].rows:
        if row["row_id"] == row_id:
            return bool(row["holds"])
    return False


def _reason(entry: dict[str, Any], row_id: str) -> str:
    for row in entry["verdict"].rows:
        if row["row_id"] == row_id:
            return str(row["reason"])
    return "missing"


# ---------------------------------------------------------------------------
# Measurement report (FR 7, AC4)
# ---------------------------------------------------------------------------

def _row_denials(result: dict[str, Any]) -> dict[str, int]:
    """Denied-row id -> number of denying voters, from a quorum result."""
    return {
        entry["row_id"]: len(entry["denied_by"])
        for entry in result["denied_rows"]
    }


def _rate(denials: int, total: int) -> str:
    return f"{denials}/{total} ({denials / total:.0%})" if total else "-"


async def quorum_report(
    packages: dict[str, dict[str, Any]],
    counts: tuple[int, ...] = (3, 5, 7),
    *,
    complete: Callable[[dict[str, Any]], Awaitable[str]],
    system_prompt: str,
    models: list[str],
    node_of: Callable[[str], str | None] | None = None,
    rotation: RotationState | None = None,
    out_path: Path = Path("report.md"),
) -> str:
    """Run the quorum at each count on each package and write ``report.md`` (FR 7, AC4).

    ``packages`` maps a package name to
    ``{"package": Package, "defect_rows": [row_id, ...]}`` — the defect
    rows are the case-table rows known to be violated (the r5 package's
    AC21-c17); every other row is a correct row. Per package and count the
    report records the defect rows caught (at least one deny), the correct
    rows denied, and the unanimity verdict; per model the catch and deny
    rates. Each count runs diversified once and undiversified once: the
    undiversified run is the no-framing baseline that separates a row
    denied because it is violated from one denied because of the framing.
    The report is accepted on completeness, not on the quorum catching the
    defect (contract: thresholds are decided on the report).

    The omitted-model rotation (FR 6, AC6-c2) holds across every
    ``run_quorum`` call the report makes (each count and mode is one run)
    and across successive report calls: with no ``rotation`` passed the
    report shares the process-wide :class:`RotationState`.
    """
    shared_rotation = rotation or _DEFAULT_ROTATION
    lines = [
        "# Quorum measurement report",
        "",
        "Per package and voter count: defect rows caught (a defect row is"
        " caught when at least one voter denies it), correct rows denied,"
        " the unanimity verdict (approved only when every voter passed"
        " every row). Per model: catch rate on defect rows and deny rate"
        " on correct rows. Diversified = shuffled row order plus one"
        " adversarially framed voter; plain = table order, no framing.",
        "",
    ]
    for name, spec in packages.items():
        pkg: Package = spec["package"]
        all_rows = voter_rows(pkg)
        defect_rows = [r for r in spec.get("defect_rows", []) if r in set(all_rows)]
        correct_total = len(all_rows) - len(defect_rows)
        lines += [f"## {name}", "", f"- package: `{pkg.directory}`",
                  f"- package_hash: `{pkg.package_hash}`",
                  f"- defect rows: {', '.join(defect_rows) or '(none)'}",
                  f"- correct rows: {correct_total}", ""]
        for count in counts:
            for mode, diversify in (("diversified", True), ("plain", False)):
                result = await run_quorum(
                    pkg, complete, system_prompt=system_prompt,
                    voters=count, models=models, node_of=node_of,
                    diversify=diversify, rotation=shared_rotation,
                    out_path=pkg.directory / f"quorum-{count}-{mode}.json",
                )
                denials = _row_denials(result)
                caught = sorted(set(defect_rows) & set(denials))
                correct_denied = sorted(set(denials) - set(defect_rows))
                lines.append(
                    f"- N={count} {mode}: approved={result['approved']}; "
                    f"defect rows caught {len(caught)}/{len(defect_rows)}"
                    + (f" ({', '.join(caught)})" if caught else "")
                    + f"; correct rows denied {len(correct_denied)}"
                    + (f" ({', '.join(correct_denied)})" if correct_denied else "")
                )
                per_model: dict[str, dict[str, int]] = {}
                for vr in result["voter_results"]:
                    stats = per_model.setdefault(
                        vr["model"],
                        {"catchers": 0, "deniers": 0, "voters": 0},
                    )
                    stats["voters"] += 1
                    row_state = {r["row_id"]: r["holds"] for r in vr["rows"]}
                    if defect_rows and any(
                        not row_state.get(r, True) for r in defect_rows
                    ):
                        stats["catchers"] += 1
                    if any(
                        not holds
                        for row_id, holds in row_state.items()
                        if row_id not in defect_rows
                    ):
                        stats["deniers"] += 1
                for model in sorted(per_model):
                    stats = per_model[model]
                    voters_here = stats["voters"]
                    lines.append(
                        f"  - {model}: voters {voters_here}; catch rate "
                        f"{_rate(stats['catchers'], voters_here)}"
                        " (voters denying a defect row), deny rate "
                        f"{_rate(stats['deniers'], voters_here)}"
                        " (voters denying a correct row)"
                    )
        lines.append("")
    report = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return report
