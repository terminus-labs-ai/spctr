#!/usr/bin/env python3
"""Scripted rename sr2_spectre -> spctr (spc-1039, spec spctr-public-v0 FR3).

One deterministic pass over the repository, re-runnable and idempotent:

1. Every tracked text file gets the product-name rewrite below.
2. ``packages/spctr/src/sr2_spectre`` moves to ``packages/spctr/src/spctr``
   with ``git mv`` so file history follows across the rename.

The rewrite maps, case- and separator-aware:

- ``sr2_spectre`` / ``sr2-spectre`` -> ``spctr``: import and distribution
  names, the CLI, prose. Dotted and path joins keep their separator:
  ``sr2_spectre.cli`` -> ``spctr.cli``, ``src/sr2_spectre/config.py`` ->
  ``src/spctr/config.py``. A hyphen seam is kept once:
  ``sr2-spectre-github-tool`` -> ``spctr-github-tool``.
- Product word ``spectre`` -> ``spctr``: prose ``Spectre`` -> ``spctr``,
  class names ``SpectreConfig`` -> ``SpctrConfig``, UPPER_SNAKE segments
  ``SPECTRE_MEMORY_DSN`` -> ``SPCTR_MEMORY_DSN``, config tiers
  ``spectre.yaml`` / ``.spectre.yaml`` -> ``spctr.yaml`` / ``.spctr.yaml``,
  secret files ``*.spectre.zuliprc`` -> ``*.spctr.zuliprc``, state dir
  ``~/.sr2-spectre`` -> ``~/.spctr``, entry-point name ``spectre_tools``
  -> ``spctr_tools``, and names like ``build_spectre_config_source``.

SR2-core names are untouched: the ``sr2`` package and distribution, the
``sr2.*`` entry-point groups, ``~/.sr2`` and ``SR2_HOME``. There are no
aliases or fallbacks to the old names.

Idempotency: every replacement is ``spctr`` in some case and no pattern
alternative matches inside ``spctr``, so a second run changes nothing.

Usage (from the repository root, with git available)::

    python scripts/rename_spctr.py

Exit 0 when done (a second run changes nothing); non-zero if git fails.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# src package move, applied with git mv after the content rewrite.
SRC_OLD = "packages/spctr/src/sr2_spectre"
SRC_NEW = "packages/spctr/src/spctr"

# Skipped by the content rewrite (paths relative to the repo root):
# - uv.lock: regenerated with `uv lock` after the manifests rename; a
#   token rewrite could not keep its lockfile hashes valid anyway.
# - this script and the tests that guard the rename: they name the old
#   tokens by design (docstrings, residue probes, rewrite samples).
SKIP_PATHS = (
    "uv.lock",
    "scripts/rename_spctr.py",
    "packages/spctr/tests/test_entry_points.py",
    "packages/spctr/tests/test_rename_residue.py",
    "packages/spctr/tests/test_rename_script.py",
)

# External environment names, kept verbatim wherever they appear: they
# name things outside this repository (a Postgres database on the test
# host and the variable that overrides its DSN), not the product. Renaming
# them would point sr2's pg memory-store tests at a database that does not
# exist and silently skip them.
PRESERVE = ("spectre_memory_test", "SPECTRE_MEMORY_TEST_DSN")
_PRESERVE_RE = re.compile("|".join(re.escape(t) for t in PRESERVE))

# Canonical replacements per exact spelling, longest token first in the
# pattern. Case rules: lowercase is canonical (spec name ``spctr``); an
# all-caps segment stays all-caps (env vars); ``Spectre`` joined forward
# inside a name keeps title case (class names); every other spelling of
# the product word collapses to the canonical lowercase name.
_CANON = {
    "sr2_spectre": "spctr",
    "sr2-spectre": "spctr",
    "SR2_SPECTRE": "SPCTR",
    "SR2-SPECTRE": "SPCTR",
    "spectre": "spctr",
    "Spectre": "spctr",
    "SPECTRE": "SPCTR",
}

# Two single-pass rewrites, run in this order; pass 2 consumes what pass 1
# emits. Each pass scans left-to-right once, so a replacement can never
# interact with text rewritten earlier in the same pass, and no
# alternative in either pass matches inside ``spctr`` (idempotency).
# re.IGNORECASE is deliberately not used: each spelling carries its own
# case rule.

# Pass 1 — the old two-part name as one token.
_FULL = re.compile(
    r"""
      (?P<mod>                           # dotted module name anywhere, or
        (?<!\w)sr2[-_]spectre(?=\.)      # the CLI at a path/command head:
      | (?:^|(?<=[/~ ]))                 # the following separator belongs
        sr2[-_]spectre(?=[/])            # to what comes next and stays
      )
    | (?P<hy>[Ss][Rr]2[-_][Ss][Pp][Ee][Cc][Tt][Rr][Ee]-(?=\w))
                                         # hyphen seam (any case: tool
                                         # names are also written
                                         # sr2-Spectre-github)
    | (?P<full>                          # old name standing on its own:
        (?<!\w)[Ss][Rr]2[-_][Ss][Pp][Ee][Cc][Tt][Rr][Ee](?!\w)
      )                                  # module, dist, CLI, prose, and
                                         # ".sr2-spectre" (a dot is no
                                         # barrier); the hyphen-joined
                                         # tails matched above win here
    """,
    re.VERBOSE,
)


def _replace_full(match: re.Match[str]) -> str:
    if match.group("hy") is not None:
        return "spctr-"
    token = match.group("mod") or match.group("full")
    return _CANON[token.lower()]


# Pass 2 — the bare product word. The sr2 spelling is listed explicitly so
# "SR2-SPECTRE" collapses whole (the SR2 prefix is part of the product
# name in that spelling), while pass-1 output like "spctr-github-tool"
# matches nothing here.
_WORD = re.compile(
    r"""
      (?P<fullcaps>S[Rr]2[-_]SPECTRE(?![\w]))  # SR2-SPECTRE, SR2_SPECTRE
    | (?P<caps>SPECTRE(?![a-z]))               # all-caps segment: SPECTRE_,
                                               # CONTENT_SPECTRE,
                                               # _MINIMAL_SPECTRE
    | (?P<tcjoin>Spectre(?=\w))                # joined forward: SpectreConfig
    | (?P<low>spectre)                         # lowercase word or fragment:
                                               # .spectre.yaml,
                                               # *.spectre.zuliprc,
                                               # spectre_tools, a-spectre-b
    | (?P<title>Spectre)                 # any other title-case spelling:
                                         # prose "Spectre", identifier tail
    """,
    re.VERBOSE,
)


def _replace_word(match: re.Match[str]) -> str:
    if match.group("fullcaps") is not None:
        return "SPCTR"
    if match.group("tcjoin") is not None:
        return "Spctr"
    if match.group("low") is not None:
        return "spctr"
    if match.group("caps") is not None:
        return "SPCTR"
    # bare title-case Spectre: joined back to a separator keeps the title
    # (note-Spectre -> note-Spctr); standing alone it canonicalises (prose
    # "Spectre" -> spctr).
    start = match.start()
    if start and match.string[start - 1] in "_-":
        return "Spctr"
    return "spctr"


def _rewrite_segment(text: str) -> str:
    text = _FULL.sub(_replace_full, text)
    return _WORD.sub(_replace_word, text)


def rewrite_text(text: str) -> str:
    """Rewrite every segment between PRESERVE tokens; keep the tokens."""
    out: list[str] = []
    pos = 0
    for match in _PRESERVE_RE.finditer(text):
        out.append(_rewrite_segment(text[pos:match.start()]))
        out.append(match.group(0))
        pos = match.end()
    out.append(_rewrite_segment(text[pos:]))
    return "".join(out)


def tracked_files() -> list[str]:
    """Every tracked file, as POSIX paths relative to the repo root."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return sorted(p for p in out.split("\0") if p)


def rewrite_contents(files: list[str]) -> int:
    """Apply the rewrite to every tracked text file; return files changed."""
    changed = 0
    for rel in files:
        if rel.startswith(SKIP_PATHS):
            continue
        path = REPO_ROOT / rel
        if not path.is_file():  # deleted-but-tracked edge case
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: leave alone
        new = rewrite_text(text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    return changed


def move_src_package() -> bool:
    """git mv src/sr2_spectre -> src/spctr. Returns True if it moved."""
    if (REPO_ROOT / SRC_OLD).is_dir():
        subprocess.run(
            ["git", "mv", SRC_OLD, SRC_NEW],
            cwd=REPO_ROOT,
            check=True,
        )
        return True
    return False


def main() -> int:
    files = tracked_files()
    changed = rewrite_contents(files)
    moved = move_src_package()
    print(
        f"rename_spctr: {changed} file(s) rewritten; "
        + (
            f"moved {SRC_OLD} -> {SRC_NEW}"
            if moved
            else "src package already renamed"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
