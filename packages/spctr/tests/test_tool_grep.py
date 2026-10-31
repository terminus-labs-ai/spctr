"""Tests for GrepTool."""
import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_grep_class_attributes() -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    assert isinstance(GrepTool.name, str) and GrepTool.name
    assert isinstance(GrepTool.description, str) and GrepTool.description
    assert isinstance(GrepTool.input_schema, dict)


def test_grep_name_is_grep() -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    assert GrepTool.name == "grep"


def test_grep_input_schema_properties() -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    props = GrepTool.input_schema.get("properties", {})
    assert "pattern" in props
    assert "path" in props
    assert "glob" in props
    assert "regex" in props


def test_grep_input_schema_required() -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    required = GrepTool.input_schema.get("required", [])
    # pattern is the only required input
    assert "pattern" in required
    assert "path" not in required
    assert "glob" not in required
    assert "regex" not in required
    assert required == ["pattern"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_grep_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path("sr2_spectre.tools.builtins.grep.GrepTool")
    assert "grep" in reg


# ---------------------------------------------------------------------------
# Literal single-file match
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_literal_single_file_match(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    target.write_text(
        "alpha line\nneedle here\ngamma line\n",
        encoding="utf-8",
    )

    tool = GrepTool()
    result = await tool(pattern="needle", path=str(target), regex=False)

    # The matched line is on line 2 (1-based) with text "needle here".
    assert "2:needle here" in result
    # The file's name appears in the output.
    assert "file.txt" in result


# ---------------------------------------------------------------------------
# Regex match
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_regex_match(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    # "foo" and "fao" both match r"f.o"; "literal" does not.
    target.write_text("foo\nfao\nliteral\n", encoding="utf-8")

    tool = GrepTool()
    # regex=True is the default.
    result = await tool(pattern=r"f.o", path=str(target))

    # Regex dot matched real characters, not requiring a literal "f.o".
    assert "1:foo" in result
    assert "2:fao" in result
    # A literal search of "f.o" would match neither line, but regex does.
    assert "f.o" not in result


# ---------------------------------------------------------------------------
# Literal mode treats metacharacters literally
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_literal_mode_treats_metachars_literally(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    # Line 1 is the literal "a.b"; line 2 is "axb" which a regex "a.b" WOULD match.
    target.write_text("a.b\naxb\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="a.b", path=str(target), regex=False)

    # Only the literal "a.b" line matches.
    assert "1:a.b" in result
    # "axb" must NOT be returned in literal mode.
    assert "axb" not in result


# ---------------------------------------------------------------------------
# Recursive directory search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_recursive_directory_search(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    nested_file = nested / "file.txt"
    nested_file.write_text("nothing\nfindme please\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="findme", path=str(tmp_path), regex=False)

    assert "file.txt" in result
    assert "2:findme please" in result


# ---------------------------------------------------------------------------
# Multiple matches across multiple files
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_multiple_files_both_appear(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    file_one = tmp_path / "one.txt"
    file_two = tmp_path / "two.txt"
    file_one.write_text("token is here\n", encoding="utf-8")
    file_two.write_text("also token\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="token", path=str(tmp_path), regex=False)

    assert "one.txt" in result
    assert "two.txt" in result


# ---------------------------------------------------------------------------
# Multiple matches within one file (different lines)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_multiple_matches_within_one_file(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    target.write_text(
        "hit one\nnothing\nhit two\n",
        encoding="utf-8",
    )

    tool = GrepTool()
    result = await tool(pattern="hit", path=str(target), regex=False)

    # Match on line 1 and line 3.
    assert "1:hit one" in result
    assert "3:hit two" in result


# ---------------------------------------------------------------------------
# glob scope
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_glob_scopes_files(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    py_file = tmp_path / "a.py"
    txt_file = tmp_path / "b.txt"
    py_file.write_text("the pattern lives here\n", encoding="utf-8")
    txt_file.write_text("the pattern also here\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(
        pattern="pattern", path=str(tmp_path), regex=False, glob="*.py"
    )

    assert "a.py" in result
    assert "b.txt" not in result


# ---------------------------------------------------------------------------
# No-match
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_no_match_returns_string_not_raise(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tool = GrepTool()
    # Should not raise on zero matches.
    result = await tool(pattern="absent_pattern", path=str(target), regex=False)

    # Non-error string that signals the absence of matches. Accept common
    # phrasings ("no match", "no matches", "nothing matched") without pinning
    # exact wording. The absent pattern must NOT appear as a reported match.
    lowered = result.lower()
    assert "no match" in lowered or "nothing match" in lowered
    assert ":absent_pattern" not in result


# ---------------------------------------------------------------------------
# Single-file path (path points directly at a file)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_single_file_path(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "solo.txt"
    target.write_text("first\nmatch_target\nthird\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="match_target", path=str(target), regex=False)

    # Path passed in points directly at the file; that path appears in output.
    assert str(target) in result
    assert "2:match_target" in result


# ---------------------------------------------------------------------------
# Binary file is skipped, not fatal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_binary_file_skipped(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    binary = tmp_path / "binary.bin"
    binary.write_bytes(b"\xff\xfe\x00\x01")

    text_file = tmp_path / "text.txt"
    text_file.write_text("normal\nhas pattern\n", encoding="utf-8")

    tool = GrepTool()
    # Must not raise on the undecodable binary file.
    result = await tool(pattern="pattern", path=str(tmp_path), regex=False)

    assert "text.txt" in result
    assert "2:has pattern" in result


# ---------------------------------------------------------------------------
# Line numbers are 1-based
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_line_numbers_are_one_based(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    # The match is on the 3rd line.
    target.write_text("one\ntwo\nNEEDLE\nfour\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="NEEDLE", path=str(target), regex=False)

    # 1-based: the match must be reported on line 3, formatted ":3:".
    assert ":3:NEEDLE" in result
    # And it must NOT be reported on a 0-based or off-by-one ":2:".
    assert ":2:NEEDLE" not in result


# ---------------------------------------------------------------------------
# Bounded-output behaviors (bead obsidian-1yh)
#
# The unbounded grep crashed the agent: a repo-root search walked a 221MB
# .venv and returned ~2.16M tokens. These tests pin the four robustness levers
# from the bead's FIX section:
#   1. default-ignore directories (.venv, .git, node_modules, __pycache__,
#      dist, build) + a configurable, MERGED custom ignore set
#   2. per-line length cap with a truncation marker
#   3. total-output cap (max matches) with a suppression notice
#   4. robust binary detection via NUL-byte sniff (not just UnicodeDecodeError)
# ---------------------------------------------------------------------------

# Suppression-notice indicator vocabulary: signals that the TOTAL number of
# matches was capped and the remainder omitted. Deliberately excludes
# "truncat" — that token belongs to the per-line truncation signal, a
# distinct concept (see _PER_LINE_TRUNC_TOKENS).
_SUPPRESSION_TOKENS = ("suppress", "omit", "more match", "limit")

# Per-line truncation indicator vocabulary: signals that a single overlong
# line was shortened.
_PER_LINE_TRUNC_TOKENS = ("truncat", "…", "...", "[")


@pytest.mark.asyncio
async def test_grep_default_ignores_venv(tmp_path) -> None:
    """A directory walk must prune the built-in default ignore dirs (.venv)."""
    from sr2_spectre.tools.builtins.grep import GrepTool

    venv_dir = tmp_path / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "pkg.txt").write_text("hidden needle\n", encoding="utf-8")

    keep = tmp_path / "keep.txt"
    keep.write_text("visible needle\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="needle", path=str(tmp_path), regex=False)

    # The file outside .venv is found.
    assert "keep.txt" in result
    # The file inside .venv is pruned by the default ignore set.
    assert "pkg.txt" not in result


@pytest.mark.asyncio
async def test_grep_default_ignores_common_dirs(tmp_path) -> None:
    """Each canonical default ignore dir is pruned during a recursive walk."""
    from sr2_spectre.tools.builtins.grep import GrepTool

    default_dirs = [
        ".venv",
        ".git",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
    ]
    for name in default_dirs:
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / "f.txt").write_text("needle inside\n", encoding="utf-8")

    keep = tmp_path / "keep.txt"
    keep.write_text("needle outside\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="needle", path=str(tmp_path), regex=False)

    assert "keep.txt" in result
    # None of the default-ignored directories should contribute a match.
    for name in default_dirs:
        assert name not in result


@pytest.mark.asyncio
async def test_grep_custom_ignore_dirs_merge_with_defaults(tmp_path) -> None:
    """A custom ignore_dirs is MERGED with defaults, never replaces them.

    Resolved spec decision: passing a custom set ADDS to the built-in
    defaults. There is no escape hatch to disable defaults, so a default-
    ignored directory (.venv) must STILL be pruned even when a custom set
    is supplied.
    """
    from sr2_spectre.tools.builtins.grep import GrepTool

    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "in_venv.txt").write_text("needle\n", encoding="utf-8")

    custom_dir = tmp_path / "custom_skip"
    custom_dir.mkdir()
    (custom_dir / "in_custom.txt").write_text("needle\n", encoding="utf-8")

    keep = tmp_path / "keep.txt"
    keep.write_text("needle\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(
        pattern="needle",
        path=str(tmp_path),
        regex=False,
        ignore_dirs={"custom_skip"},
    )

    # 1. Ordinary file outside any ignored dir IS found.
    assert "keep.txt" in result
    # 2. Custom-ignored dir IS pruned.
    assert "in_custom.txt" not in result
    # 3. Default-ignored dir is STILL pruned despite a custom set being passed
    #    (merge semantics, not replace).
    assert "in_venv.txt" not in result


@pytest.mark.asyncio
async def test_grep_caps_per_line_length_with_truncation_marker(tmp_path) -> None:
    """An overlong matching line is truncated and marked, not dumped whole."""
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "long.txt"
    # One match on a single, very long line (simulates minified/vendored data).
    long_line = "needle" + ("x" * 10_000)
    target.write_text(long_line + "\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="needle", path=str(target), regex=False)

    # The match is reported.
    assert "needle" in result
    # The full 10k-char line must NOT be emitted verbatim.
    assert len(result) < 1000
    # A truncation marker of some kind is present.
    assert any(tok in result for tok in _PER_LINE_TRUNC_TOKENS)


@pytest.mark.asyncio
async def test_grep_caps_total_matches_with_suppression_notice(tmp_path) -> None:
    """When matches exceed the total cap, output is bounded + a notice appears."""
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "many.txt"
    # Far more matches than any reasonable cap; lines are short so this
    # exercises the TOTAL-match cap, not the per-line cap.
    lines = "\n".join(f"needle {i}" for i in range(5_000))
    target.write_text(lines + "\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="needle", path=str(target), regex=False)

    # Output is bounded: not all 5000 matches are present.
    assert result.count("needle") <= 100
    # A suppression notice signals that matches were omitted. Uses the
    # suppression vocabulary only (no per-line "truncat" token here).
    lowered = result.lower()
    assert any(tok in lowered for tok in _SUPPRESSION_TOKENS)


@pytest.mark.asyncio
async def test_grep_under_cap_emits_all_without_notice(tmp_path) -> None:
    """When matches are under the cap, all appear and NO suppression notice."""
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "few.txt"
    target.write_text(
        "needle one\nneedle two\nneedle three\n", encoding="utf-8"
    )

    tool = GrepTool()
    result = await tool(pattern="needle", path=str(target), regex=False)

    # All three matches present.
    assert result.count("needle") == 3
    # No suppression notice when nothing was suppressed.
    lowered = result.lower()
    assert not any(tok in lowered for tok in _SUPPRESSION_TOKENS)


@pytest.mark.asyncio
async def test_grep_nul_byte_file_skipped(tmp_path) -> None:
    """A file with a NUL byte is treated as binary and skipped (NUL sniff).

    The original code only skipped on UnicodeDecodeError; a UTF-8-decodable
    file containing a NUL byte slipped through and got dumped. Robust binary
    detection must skip it.
    """
    from sr2_spectre.tools.builtins.grep import GrepTool

    # Contains a NUL byte but is otherwise valid UTF-8 (decodes cleanly), so
    # only a NUL-byte sniff — not UnicodeDecodeError — catches it.
    nul_file = tmp_path / "data.bin"
    nul_file.write_bytes(b"needle\x00more needle here\n")

    text_file = tmp_path / "text.txt"
    text_file.write_text("needle in text\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="needle", path=str(tmp_path), regex=False)

    # The real text file is searched.
    assert "text.txt" in result
    # The NUL-containing binary is skipped, contributing no matches.
    assert "data.bin" not in result
