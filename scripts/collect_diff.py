#!/usr/bin/env python3
"""Diff collected pytest node IDs between a base ref and the current head (spc-1066).

Why this exists: pass/fail counts and sorted FAIL-set equality cannot detect
silent test loss — an added test and a deleted test cancel in the totals
(spctr spc-1058, Agent G finding r1). This tool collects every workspace
member's pytest node-ID set at a base ref and at the head (the working
tree, or a second ref), reports the added and removed sets, and fails the
gate if and only if some removal is not explicitly sanctioned. Additions
never fail the gate.

Usage (from the repo root):

    uv run --frozen python scripts/collect_diff.py <base-ref> [options]

Options:
    --head REF            Compare against a ref instead of the working tree
                          (that side also runs in a detached worktree).
    --repo-root PATH      Repository root (default: current directory).
    --members M [M ...]   Members to collect, mirroring scripts/run_tests.sh
                          (default: packages/sr2 packages/spctr).
    --allow ENTRY         Sanction one expected removal. Exact node ID, a
                          file-or-directory prefix, or a prefix ending in
                          '::'. Repeatable. A sanctioned removal is reported
                          but does not fail the gate.
    --pytest-cmd CMD      Collection command prefix (default:
                          'uv run --frozen python -m pytest'); the member
                          path and '--collect-only -q' are appended per
                          member, matching the per-member invocations of
                          scripts/run_tests.sh.
    --keep-worktree       Keep the scratch worktrees for debugging.

Exit codes:
    0  no unsanctioned removals
    1  one or more collected test nodes removed relative to base
    2  the check itself could not run (bad ref, collection error, ...)

Base/head refs are collected in `git worktree add --detach` scratch
worktrees under $TMPDIR, never copies of the live worktree (a copy shares
the gitdir and would stage into the build index).
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MEMBERS = ("packages/sr2", "packages/spctr")
DEFAULT_PYTEST_CMD = "uv run --frozen python -m pytest"

EXIT_OK = 0
EXIT_REMOVED = 1
EXIT_ERROR = 2

_TAG = "[collect-diff]"


class CollectError(RuntimeError):
    """A member's pytest collection failed or returned an unexpected code."""

    def __init__(self, member: str, returncode: int, output: str) -> None:
        tail = "\n".join(output.splitlines()[-25:])
        super().__init__(f"{member}: pytest exited {returncode}\n{tail}")


def _run_git(repo_root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args], check=False, capture_output=True, text=True
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}):\n{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def resolve_ref(repo_root: Path, ref: str) -> str:
    """Resolve a ref to its commit sha (raises on unknown refs)."""
    try:
        return _run_git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except RuntimeError:
        raise RuntimeError(f"unknown base/head ref: {ref!r}") from None


def parse_collected_node_ids(output: str) -> set[str]:
    """Extract node IDs from ``pytest --collect-only -q`` stdout.

    The only non-node-ID line that reaches stdout is the trailing summary
    (``N tests collected in …s`` / ``no tests ran in …s``); everything else
    (headers, error sections, warnings) either carries '=' framing, lacks
    the ``::`` separator, or matches the summary shape.
    """
    ids: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        if _looks_like_summary(line):
            continue
        if "::" not in line:
            continue
        ids.add(line)
    return ids


def _looks_like_summary(line: str) -> bool:
    # "3309 tests collected in 12.71s", "1 test collected, 1 error in 0.10s",
    # "no tests ran in 0.00s"
    words = line.split()
    return (
        bool(words) and words[0].isdigit() and any(w in ("collected", "ran") for w in words)
    ) or line.startswith("no tests ran")


def diff_node_sets(base: set[str], head: set[str]) -> tuple[set[str], set[str]]:
    """Return (added, removed): present at head absent at base / the reverse."""
    return head - base, base - head


def is_sanctioned(node: str, allow_entries: list[str]) -> bool:
    """True when ``node`` is covered by one of the --allow entries.

    An entry matches when it equals the node, is a path prefix of it
    (``packages/spctr/tests/test_foo.py`` sanctions the whole file), or is
    a prefix ending in '::' (``tests/test_foo.py::TestClass::`` sanctions a
    class).
    """
    for raw in allow_entries:
        entry = raw.strip()
        if not entry:
            continue
        if (
            node == entry
            or node.startswith((entry + "::", entry + "/"))
            or (entry.endswith("::") and node.startswith(entry))
        ):
            return True
    return False


@dataclass
class GateVerdict:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    sanctioned: list[str] = field(default_factory=list)
    exit_code: int = EXIT_OK

    @classmethod
    def evaluate(
        cls, base: set[str], head: set[str], allow: list[str] | None = None
    ) -> GateVerdict:
        allow = allow or []
        added_set, removed_set = diff_node_sets(base, head)
        unsanctioned = {n for n in removed_set if not is_sanctioned(n, allow)}
        sanctioned = sorted(removed_set - unsanctioned)
        return cls(
            added=sorted(added_set),
            removed=sorted(unsanctioned),
            sanctioned=sanctioned,
            exit_code=EXIT_REMOVED if unsanctioned else EXIT_OK,
        )


def collect_tree(
    tree: Path, members: list[str], pytest_cmd: str
) -> dict[str, set[str]]:
    """Collect the per-member node-ID sets of one checkout.

    Node IDs come back rootdir-relative; they carry the member path only
    when pytest's rootdir is the workspace root (vs. the member directory),
    so the member prefix is applied idempotently — cross-member IDs stay
    distinct either way.
    """
    result: dict[str, set[str]] = {}
    for member in members:
        cmd = [*shlex.split(pytest_cmd), member, "--collect-only", "-q"]
        proc = subprocess.run(cmd, check=False, cwd=str(tree), capture_output=True, text=True)
        if proc.returncode not in (0, 5):  # 5 = "no tests ran"
            raise CollectError(member, proc.returncode, proc.stdout + "\n" + proc.stderr)
        result[member] = {
            node if node.startswith(f"{member}/") else f"{member}/{node}"
            for node in parse_collected_node_ids(proc.stdout)
        }
    return result


class Worktree:
    """A detached ``git worktree add --detach`` scratch checkout."""

    def __init__(self, repo_root: Path, ref: str, keep: bool = False) -> None:
        self.repo_root = repo_root
        self.ref = ref
        self.keep = keep
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="collect-diff-"))
        try:
            _run_git(self.repo_root, "worktree", "add", "--detach", str(self.path), self.ref)
        except Exception:
            self._remove()
            raise
        return self.path

    def __exit__(self, *exc_info: object) -> None:
        if self.path is not None:
            self._remove()

    def _remove(self) -> None:
        if self.keep or self.path is None:
            return
        _run_git(self.repo_root, "worktree", "prune", check=False)
        try:
            _run_git(self.repo_root, "worktree", "remove", "--force", str(self.path), check=False)
        finally:
            shutil.rmtree(self.path, ignore_errors=True)
            self.path = None


def build_report(
    base_sha: str,
    head_label: str,
    base_ids: dict[str, set[str]],
    head_ids: dict[str, set[str]],
    verdict: GateVerdict,
) -> str:
    lines = [
        f"{_TAG} base={base_sha[:12]} head={head_label}",
    ]
    for member in sorted(base_ids):
        lines.append(
            f"{_TAG} {member}: base {len(base_ids.get(member, ()))} "
            f"head {len(head_ids.get(member, ()))}"
        )
    lines.append(f"{_TAG} added ({len(verdict.added)}):")
    lines.extend(f"  + {node}" for node in verdict.added)
    lines.append(f"{_TAG} removed ({len(verdict.removed)}):")
    lines.extend(f"  - {node}" for node in verdict.removed)
    lines.append(f"{_TAG} sanctioned ({len(verdict.sanctioned)}):")
    lines.extend(f"  ~ {node}" for node in verdict.sanctioned)
    if verdict.exit_code == EXIT_OK:
        lines.append(f"{_TAG} result: OK — no unsanctioned removals vs base")
    else:
        lines.append(
            f"{_TAG} result: FAIL — {len(verdict.removed)} collected test node(s) "
            f"removed vs base; each needs justification against the bead's ACs in the "
            f"evidence record or explicit sanction via --allow"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="base ref (branch, tag or commit)")
    parser.add_argument("--head", help="head ref; default: the current working tree")
    parser.add_argument("--repo-root", default=".", help="repository root (default: cwd)")
    parser.add_argument(
        "--members", nargs="+", default=list(DEFAULT_MEMBERS),
        help=f"workspace members to collect (default: {' '.join(DEFAULT_MEMBERS)})",
    )
    parser.add_argument("--allow", action="append", default=[],
                        help="sanction one expected removal (repeatable)")
    parser.add_argument("--pytest-cmd", default=DEFAULT_PYTEST_CMD,
                        help="collection command prefix per member")
    parser.add_argument("--keep-worktree", action="store_true",
                        help="keep scratch worktrees for debugging")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / ".git").exists():
        print(f"{_TAG} error: {repo_root} is not a git repository", file=sys.stderr)
        return EXIT_ERROR

    try:
        base_sha = resolve_ref(repo_root, args.base)
        head_is_worktree = args.head is None
        head_label = "working-tree"
        if head_is_worktree:
            head_tree = repo_root
            head_head = _run_git(repo_root, "rev-parse", "HEAD")
            head_label = f"working-tree (HEAD {head_head[:12]})"
        else:
            head_label = resolve_ref(repo_root, args.head)[:12]

        if head_is_worktree:
            head_ids = collect_tree(head_tree, args.members, args.pytest_cmd)
        else:
            with Worktree(repo_root, head_label, keep=args.keep_worktree) as head_wt:
                head_ids = collect_tree(head_wt, args.members, args.pytest_cmd)

        with Worktree(repo_root, base_sha, keep=args.keep_worktree) as base_wt:
            base_ids = collect_tree(base_wt, args.members, args.pytest_cmd)
    except (RuntimeError, CollectError) as exc:
        print(f"{_TAG} error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    all_base = {n for s in base_ids.values() for n in s}
    all_head = {n for s in head_ids.values() for n in s}
    verdict = GateVerdict.evaluate(all_base, all_head, args.allow)
    print(build_report(base_sha, head_label, base_ids, head_ids, verdict))
    return verdict.exit_code


if __name__ == "__main__":
    sys.exit(main())
