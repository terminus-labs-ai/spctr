"""spc-1062: same-failure tripwire for the tool loop (agent-workflow-v3 D38).

Watches the test runs of one turn. When a failing run shows the same failure
as a run at least ``min_span_calls`` calls earlier, with no progress between
them, that is a detection: the first ``notes_before_stop`` detections get a
harness note appended to the run's result, the next one ends the turn with a
harness-written escalation.

"Same failure": both runs carry pytest assertion lines (``E   assert ...``,
``E   SomeError: ...``) and those sets are equal; otherwise both carry failing
test ids (``FAILED``/``ERROR`` lines) and those sets are equal. "Progress": a
run of the same selection (the pytest arguments) with fewer failures than that
selection's best so far; it clears the history. The rule is tuned against the
replay fixtures in ``tests/tripwire/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sr2.config.models import TripwireConfig

_FAILING_ID = re.compile(r"^(?:\d+[:-])?(?:FAILED|ERROR) (\S+\.py(?:::\S+)?)(?:\s|$)")
_FAILURE_LINE = re.compile(
    r"^(?:\d+[:-])?E\s+(?:AssertionError: )?((?:assert\b|[\w.]+(?:Error|Exception)\b).*)$"
)
_SUMMARY_COUNT = re.compile(r"(\d+) (failed|passed|errors?)\b")
_SUMMARY_LINE = re.compile(r" in [\d.]+s\b")
_SHELL_BREAK = re.compile(r"\s(?:\d?>|\||;|&&)")
_QUIET_ARG = re.compile(r"^(?:-q+|-v+|-s|--tb=\S+|--no-header)$")


@dataclass(frozen=True)
class _Run:
    call: int
    ids: frozenset[str]
    failures: frozenset[str]

    def same_failure(self, other: "_Run") -> bool:
        if self.failures and other.failures:
            return self.failures == other.failures
        return bool(self.ids and other.ids) and self.ids == other.ids


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(getattr(block, "text", "") for block in content)


def _failed_count(lines: list[str]) -> int | None:
    """Failed + errors over every pytest summary line, or None without one."""
    counts = [
        _SUMMARY_COUNT.findall(line) for line in lines if _SUMMARY_LINE.search(line)
    ]
    counts = [c for c in counts if c]
    if not counts:
        return None
    return sum(int(n) for c in counts for n, kind in c if kind != "passed")


class Tripwire:
    """Per-turn tripwire state. ``observe`` every executed call in order."""

    def __init__(self, config: TripwireConfig) -> None:
        self._config = config
        self._command = re.compile(config.command_pattern)
        self.call_count = 0
        self.detections = 0
        self._runs: list[_Run] = []
        self._best: dict[str, int] = {}
        self.first_call = 0
        self.failures: list[str] = []

    def observe(self, name: str, arguments: dict, content: Any) -> str | None:
        """Number one executed call; return ``"note"``, ``"stop"`` or None."""
        self.call_count += 1
        command = arguments.get(self._config.command_argument)
        if name not in self._config.tools or not isinstance(command, str):
            return None
        match = self._command.search(command)
        if match is None:
            return None
        lines = _text(content).splitlines()
        failed = _failed_count(lines)
        if failed is not None:
            selection = self._selection(command[match.end():])
            best = self._best.get(selection)
            if best is not None and failed < best:
                self._runs.clear()
            self._best[selection] = failed if best is None else min(best, failed)
        run = _Run(
            call=self.call_count,
            ids=frozenset(m.group(1) for line in lines if (m := _FAILING_ID.match(line))),
            failures=frozenset(
                " ".join(re.sub(r"0x[0-9a-f]+", "0x", m.group(1)).split())
                for line in lines
                if (m := _FAILURE_LINE.match(line))
            ),
        )
        if not (run.ids or run.failures):
            return None
        earlier = next((r for r in self._runs if r.same_failure(run)), None)
        self._runs.append(run)
        if earlier is None or run.call - earlier.call < self._config.min_span_calls:
            return None
        self.detections += 1
        self.first_call = earlier.call
        self.failures = sorted(earlier.ids | run.ids) or sorted(run.failures)
        return "stop" if self.detections > self._config.notes_before_stop else "note"

    def message(self, template: str) -> str:
        """``note`` or ``escalation`` filled for the latest detection."""
        return template.format(
            first_call=self.first_call,
            call=self.call_count,
            failures="\n".join(f"- {f}" for f in self.failures),
        )

    @staticmethod
    def _selection(args: str) -> str:
        """The pytest arguments of a command, minus output-only flags."""
        args = _SHELL_BREAK.split(" " + args, maxsplit=1)[0]
        tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', args)
        return " ".join(t.strip("\"'") for t in tokens if not _QUIET_ARG.match(t))
