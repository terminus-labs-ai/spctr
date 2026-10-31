"""Tests for frontmatter parsing (FR5).

Covers:
  - extract_raw_frontmatter: boundary cases for the --- --- extraction.
  - parse_frontmatter: all three kinds, tolerance, error handling.
  - parse_file: disk I/O wrapping.
  - Models: frozen dataclasses, enum values.

Acceptance criteria:
  - Files lacking a recognized 'kind' or that fail to parse are skipped (logged),
    never crash resolve.
  - YAML parser reuses SR2-bundled PyYAML (no new dep).
"""

import logging
import pytest
import yaml as _yaml
from pathlib import Path
from textwrap import dedent

from sr2_spectre.planning.frontmatter import (
    _FRONTMATTER_PARSERS,
    _build_frontmatter,
    _coerce_str,
    _parse_knowledge,
    _parse_plan,
    _parse_task,
    extract_raw_frontmatter,
    parse_file,
    parse_frontmatter,
    split_frontmatter,
)
from sr2_spectre.planning.models import (
    KnowledgeFrontmatter,
    PlanFrontmatter,
    PlanStatus,
    RECOGNIZED_KINDS,
    TaskFrontmatter,
    TaskStatus,
    get_frontmatter_class,
)


# =========================================================================
# 1. extract_raw_frontmatter
# =========================================================================


class TestExtractRawFrontmatter:
    def test_simple_block(self):
        """Basic frontmatter block between --- delimiters."""
        text = "---\nkind: task\norder: 1\n---\n# Body\n"
        raw = extract_raw_frontmatter(text)
        assert raw is not None
        assert "kind: task" in raw
        assert "order: 1" in raw

    def test_no_frontmatter(self):
        """Text without --- prefix returns None."""
        text = "# Just a heading\n\nSome body text.\n"
        assert extract_raw_frontmatter(text) is None

    def test_single_dash_no_close(self):
        """Opening --- but no closing --- returns None."""
        text = "---\nkind: task\norder: 1\n# No close\n"
        assert extract_raw_frontmatter(text) is None

    def test_empty_block(self):
        """Empty frontmatter (--- followed immediately by ---) returns None."""
        text = "---\n---\n# Body\n"
        assert extract_raw_frontmatter(text) is None

    def test_whitespace_before_block(self):
        """Leading whitespace is stripped before checking for ---."""
        text = "\n\n---\nkind: plan\n---\n# Body\n"
        raw = extract_raw_frontmatter(text)
        assert raw is not None

    def test_multiline_yaml(self):
        """Multi-line YAML values are extracted correctly."""
        text = "---\nkind: task\nverify: |\n  uv run pytest\n  uv run mypy\n---\n# Body\n"
        raw = extract_raw_frontmatter(text)
        assert raw is not None
        assert "uv run pytest" in raw

    def test_block_with_horizontal_rules(self):
        """Body contains --- (horizontal rules) after frontmatter closes."""
        text = "---\nkind: task\n---\n# Body\n\n---\n\nMore text\n"
        raw = extract_raw_frontmatter(text)
        assert raw is not None
        assert "kind: task" in raw


# =========================================================================
# 1b. split_frontmatter (shared boundary scanner)
# =========================================================================


class TestSplitFrontmatter:
    """Tests for the canonical frontmatter boundary scanner.

    This is the shared helper that extract_raw_frontmatter, resolver, and
    complete_step all delegate to.
    """

    def test_simple_block(self):
        """Returns (frontmatter_block, body) for a standard file."""
        text = "---\nkind: task\norder: 1\n---\n# Body\n"
        result = split_frontmatter(text)
        assert result is not None
        fm_block, body = result
        assert fm_block == "---\nkind: task\norder: 1\n---"
        # trailing \n stripped by internal strip(); body starts with \n (paragraph break)
        assert body == "\n# Body"

    def test_no_frontmatter(self):
        """Text without --- prefix returns None."""
        text = "# Just a heading\n\nSome body text.\n"
        assert split_frontmatter(text) is None

    def test_single_dash_no_close(self):
        """Opening --- but no closing --- returns None."""
        text = "---\nkind: task\norder: 1\n# No close\n"
        assert split_frontmatter(text) is None

    def test_empty_block(self):
        """Empty frontmatter (--- followed immediately by ---) still splits.

        split_frontmatter is a low-level boundary scanner — it returns the
        split even when the YAML block is empty.  extract_raw_frontmatter
        (the consumer) treats an empty block as None.
        """
        text = "---\n---\n# Body\n"
        result = split_frontmatter(text)
        assert result is not None
        fm_block, body = result
        assert fm_block == "---\n---"
        assert body == "\n# Body"

    def test_whitespace_before_block(self):
        """Leading whitespace is handled — frontmatter still found."""
        text = "\n\n---\nkind: plan\n---\n# Body\n"
        result = split_frontmatter(text)
        assert result is not None
        fm_block, body = result
        assert fm_block == "---\nkind: plan\n---"
        assert body == "\n# Body"

    def test_multiline_yaml(self):
        """Multi-line YAML values preserved in frontmatter block."""
        text = "---\nkind: task\nverify: |\n  uv run pytest\n  uv run mypy\n---\n# Body\n"
        result = split_frontmatter(text)
        assert result is not None
        fm_block, body = result
        assert "uv run pytest" in fm_block
        assert "uv run mypy" in fm_block

    def test_body_with_horizontal_rules(self):
        """Body contains --- after frontmatter closes — only first block used."""
        text = "---\nkind: task\n---\n# Body\n\n---\n\nMore text\n"
        result = split_frontmatter(text)
        assert result is not None
        fm_block, body = result
        assert fm_block == "---\nkind: task\n---"
        assert "# Body" in body
        assert "More text" in body

    def test_fm_block_includes_delimiters(self):
        """Frontmatter block includes opening and closing --- delimiters."""
        text = "---\nkind: task\n---\n# Body\n"
        fm_block, _body = split_frontmatter(text)
        assert fm_block.startswith("---\n")
        assert fm_block.endswith("---")

    def test_reconstruct_original(self):
        """fm_block + body reconstructs the stripped original text."""
        text = "---\nkind: task\norder: 1\n---\n# Body\n"
        fm_block, body = split_frontmatter(text)
        assert fm_block + body == text.strip()

    def test_multiline_yaml_roundtrip(self):
        """Round-trip works for complex YAML with block scalars."""
        text = "---\nkind: task\nverify: |\n  pytest tests/\n---\n## Task\n\nDo the thing.\n"
        fm_block, body = split_frontmatter(text)
        assert fm_block + body == text.strip()
        assert fm_block == "---\nkind: task\nverify: |\n  pytest tests/\n---"
        assert body == "\n## Task\n\nDo the thing."

    def test_no_body_after_closing(self):
        """File ends right after closing --- with newline."""
        text = "---\nkind: plan\n---\n"
        result = split_frontmatter(text)
        assert result is not None
        fm_block, body = result
        assert fm_block == "---\nkind: plan\n---"
        assert body == ""

    def test_body_preserves_leading_newline(self):
        """If there's a blank line after closing ---, body has the paragraph break."""
        text = "---\nkind: task\n---\n\n# Body starts after blank line\n"
        fm_block, body = split_frontmatter(text)
        assert body.startswith("\n\n")

    def test_empty_string(self):
        """Empty string returns None."""
        assert split_frontmatter("") is None

    def test_only_dashes(self):
        """Just --- with no body returns None."""
        assert split_frontmatter("---") is None

    def test_status_in_fm_block(self):
        """Status field appears in fm_block (for _flip_status use case)."""
        text = "---\nkind: task\nstatus: pending\n---\n# Body\n"
        fm_block, _body = split_frontmatter(text)
        assert "status: pending" in fm_block

    def test_exported_from_planning_package(self):
        """split_frontmatter is importable from the planning package."""
        from sr2_spectre.planning import split_frontmatter as pkg_split
        assert pkg_split is split_frontmatter


# =========================================================================
# 2. parse_frontmatter — task kind
# =========================================================================


class TestParseTaskFrontmatter:
    def _task_text(self, yaml_block: str, body: str = "# Body\n") -> str:
        return f"---\n{yaml_block}\n---\n{body}"

    def test_minimal_task(self):
        """Only kind: task — all other fields get defaults."""
        text = self._task_text("kind: task")
        result = parse_frontmatter(text)
        assert isinstance(result, TaskFrontmatter)
        assert result.kind == "task"
        assert result.plan == ""
        assert result.order == 0
        assert result.status == TaskStatus.PENDING
        assert result.verify == ""
        assert result.title == ""

    def test_full_task(self):
        """All fields populated."""
        text = self._task_text(
            "kind: task\nplan: rename-plugin\norder: 2\n"
            "status: pending\nverify: uv run pytest\n"
            'title: "Rename Plugin to Interface"'
        )
        result = parse_frontmatter(text)
        assert isinstance(result, TaskFrontmatter)
        assert result.plan == "rename-plugin"
        assert result.order == 2
        assert result.status == TaskStatus.PENDING
        assert result.verify == "uv run pytest"
        assert result.title == "Rename Plugin to Interface"

    def test_done_status(self):
        """status: done is parsed correctly."""
        text = self._task_text("kind: task\nstatus: done")
        result = parse_frontmatter(text)
        assert result.status == TaskStatus.DONE

    def test_invalid_status_defaults_to_pending(self):
        """Invalid status value falls back to pending (tolerance)."""
        text = self._task_text("kind: task\nstatus: banana")
        result = parse_frontmatter(text)
        assert result is not None
        assert result.status == TaskStatus.PENDING

    def test_missing_optional_fields(self):
        """Missing optional fields don't crash — get defaults."""
        text = self._task_text("kind: task\norder: 5")
        result = parse_frontmatter(text)
        assert result.order == 5
        assert result.plan == ""
        assert result.verify == ""

    def test_negative_order(self):
        """Negative order is accepted (int)."""
        text = self._task_text("kind: task\norder: -1")
        result = parse_frontmatter(text)
        assert result.order == -1

    def test_non_integer_order_defaults_to_zero(self):
        """Non-integer order falls back to 0 (tolerance)."""
        text = self._task_text("kind: task\norder: abc")
        result = parse_frontmatter(text)
        assert result.order == 0


# =========================================================================
# 3. parse_frontmatter — plan kind
# =========================================================================


class TestParsePlanFrontmatter:
    def _plan_text(self, yaml_block: str) -> str:
        return f"---\n{yaml_block}\n---\n# Body\n"

    def test_minimal_plan(self):
        text = self._plan_text("kind: plan")
        result = parse_frontmatter(text)
        assert isinstance(result, PlanFrontmatter)
        assert result.kind == "plan"
        assert result.slug == ""
        assert result.status == PlanStatus.OPEN
        assert result.goal == ""

    def test_full_plan(self):
        text = self._plan_text(
            "kind: plan\nslug: rename-plugin\nstatus: open\n"
            'goal: "Rename Plugin to Interface across 9 files"'
        )
        result = parse_frontmatter(text)
        assert result.slug == "rename-plugin"
        assert result.status == PlanStatus.OPEN
        assert result.goal == "Rename Plugin to Interface across 9 files"

    def test_done_plan(self):
        text = self._plan_text("kind: plan\nstatus: done")
        result = parse_frontmatter(text)
        assert result.status == PlanStatus.DONE

    def test_invalid_status_defaults_to_open(self):
        text = self._plan_text("kind: plan\nstatus: woot")
        result = parse_frontmatter(text)
        assert result is not None
        assert result.status == PlanStatus.OPEN


# =========================================================================
# 4. parse_frontmatter — project-knowledge kind
# =========================================================================


class TestParseKnowledgeFrontmatter:
    def _knowledge_text(self, yaml_block: str) -> str:
        return f"---\n{yaml_block}\n---\n# Body\n"

    def test_minimal_knowledge(self):
        text = self._knowledge_text("kind: project-knowledge")
        result = parse_frontmatter(text)
        assert isinstance(result, KnowledgeFrontmatter)
        assert result.kind == "project-knowledge"
        assert result.project == ""

    def test_full_knowledge(self):
        text = self._knowledge_text(
            "kind: project-knowledge\nproject: sr2-spectre"
        )
        result = parse_frontmatter(text)
        assert result.project == "sr2-spectre"

    def test_knowledge_matches_project(self):
        """Knowledge project field is used for matching in resolver."""
        text = self._knowledge_text("kind: project-knowledge\nproject: my-project")
        result = parse_frontmatter(text)
        assert result.project == "my-project"


# =========================================================================
# 5. Tolerance — skip, never crash
# =========================================================================


class TestParseFrontmatterTolerance:
    def test_no_frontmatter_returns_none(self):
        """Files without frontmatter return None (skip)."""
        result = parse_frontmatter("# Just a heading\n\nBody text.\n")
        assert result is None

    def test_missing_kind_returns_none(self):
        """Frontmatter without 'kind' returns None."""
        text = "---\norder: 1\nstatus: pending\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert result is None

    def test_unrecognized_kind_returns_none(self):
        """Unrecognized kind returns None (skip, no crash)."""
        text = "---\nkind: recipe\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert result is None

    def test_invalid_yaml_returns_none(self):
        """Malformed YAML returns None (skip, no crash)."""
        text = "---\nkind: task\n  order: [invalid yaml\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert result is None

    def test_non_mapping_yaml_returns_none(self):
        """YAML that parses to a non-dict (e.g., a list) returns None."""
        text = "---\n- item1\n- item2\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        result = parse_frontmatter("")
        assert result is None

    def test_only_dashes_returns_none(self):
        """Just `---` with no body returns None."""
        result = parse_frontmatter("---")
        assert result is None

    def test_frontmatter_with_body_preserved_in_text(self):
        """parse_frontmatter only extracts frontmatter; the body remains in the
        original text. The function doesn't strip or modify the input."""
        text = "---\nkind: task\n---\n# Important Body\n"
        result = parse_frontmatter(text)
        assert result is not None
        assert "# Important Body" in text  # input unchanged

    def test_case_insensitive_kind(self):
        """Kind matching is case-insensitive."""
        text = "---\nkind: TASK\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert isinstance(result, TaskFrontmatter)

    def test_kind_with_whitespace(self):
        """Kind with surrounding whitespace is trimmed."""
        text = "---\nkind:  task  \n---\n# Body\n"
        result = parse_frontmatter(text)
        assert isinstance(result, TaskFrontmatter)


# =========================================================================
# 6. parse_file — disk I/O
# =========================================================================


class TestParseFile:
    def test_parses_valid_file(self, tmp_path):
        content = "---\nkind: task\norder: 1\nstatus: pending\n---\n# Body\n"
        file = tmp_path / "01-test.md"
        file.write_text(content)
        result = parse_file(file)
        assert isinstance(result, TaskFrontmatter)
        assert result.order == 1

    def test_missing_file_returns_none(self, tmp_path):
        result = parse_file(tmp_path / "nonexistent.md")
        assert result is None

    def test_unreadable_file_returns_none(self, tmp_path):
        file = tmp_path / "locked.md"
        file.write_text("---\nkind: task\n---\n")
        file.chmod(0o000)
        try:
            result = parse_file(file)
            assert result is None
        finally:
            file.chmod(0o644)  # cleanup for deletion

    def test_file_without_frontmatter_returns_none(self, tmp_path):
        file = tmp_path / "readme.md"
        file.write_text("# Just a README\n\nNo frontmatter here.\n")
        result = parse_file(file)
        assert result is None


# =========================================================================
# 7. Models — frozen dataclasses & enums
# =========================================================================


class TestModels:
    def test_task_frontmatter_frozen(self):
        """TaskFrontmatter is frozen (immutable)."""
        fm = TaskFrontmatter(plan="test", order=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            fm.plan = "other"

    def test_plan_frontmatter_frozen(self):
        fm = PlanFrontmatter(slug="test")
        with pytest.raises(Exception):
            fm.slug = "other"

    def test_knowledge_frontmatter_frozen(self):
        fm = KnowledgeFrontmatter(project="test")
        with pytest.raises(Exception):
            fm.project = "other"

    def test_recommended_kinds_set(self):
        """RECOGNIZED_KINDS contains all three kinds."""
        assert "task" in RECOGNIZED_KINDS
        assert "plan" in RECOGNIZED_KINDS
        assert "project-knowledge" in RECOGNIZED_KINDS

    def test_get_frontmatter_class(self):
        assert get_frontmatter_class("task") is TaskFrontmatter
        assert get_frontmatter_class("plan") is PlanFrontmatter
        assert get_frontmatter_class("project-knowledge") is KnowledgeFrontmatter
        assert get_frontmatter_class("unknown") is None

    def test_task_status_enum(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.DONE.value == "done"

    def test_plan_status_enum(self):
        assert PlanStatus.OPEN.value == "open"
        assert PlanStatus.DONE.value == "done"

    def test_task_status_from_string(self):
        assert TaskStatus("pending") == TaskStatus.PENDING
        assert TaskStatus("done") == TaskStatus.DONE

    def test_plan_status_from_string(self):
        assert PlanStatus("open") == PlanStatus.OPEN
        assert PlanStatus("done") == PlanStatus.DONE


# =========================================================================
# 8. Logging — warnings emitted on skip
# =========================================================================


class TestLoggingOnSkip:
    def test_warning_on_unrecognized_kind(self, caplog):
        """A warning is logged when kind is unrecognized."""
        caplog.set_level(logging.WARNING, logger="sr2_spectre.planning.frontmatter")
        text = "---\nkind: recipe\n---\n"
        result = parse_frontmatter(text, file_path=Path("/test.md"))
        assert result is None
        assert "Unrecognized kind" in caplog.text

    def test_warning_on_missing_kind(self, caplog):
        """A warning is logged when kind is missing."""
        caplog.set_level(logging.WARNING, logger="sr2_spectre.planning.frontmatter")
        text = "---\norder: 1\n---\n"
        result = parse_frontmatter(text, file_path=Path("/test.md"))
        assert result is None
        assert "No 'kind'" in caplog.text

    def test_warning_on_yaml_error(self, caplog):
        """A warning is logged on YAML parse errors."""
        caplog.set_level(logging.WARNING, logger="sr2_spectre.planning.frontmatter")
        text = "---\nkind: [broken yaml\n---\n"
        result = parse_frontmatter(text, file_path=Path("/test.md"))
        assert result is None
        assert "YAML parse error" in caplog.text

    def test_debug_on_no_frontmatter(self, caplog):
        """Debug-level log when no frontmatter found."""
        caplog.set_level(logging.DEBUG, logger="sr2_spectre.planning.frontmatter")
        text = "# Just a heading\n"
        result = parse_frontmatter(text, file_path=Path("/test.md"))
        assert result is None
        assert "No frontmatter" in caplog.text


# =========================================================================
# 9. PyYAML reuse (no new dependency)
# =========================================================================


class TestPyYAMLReuse:
    def test_uses_builtin_yaml(self):
        """The frontmatter module uses yaml from the existing PyYAML dep."""
        from sr2_spectre.planning import frontmatter
        # Verify yaml module is the standard PyYAML
        assert frontmatter.yaml is not None
        # Verify it can parse (smoke test)
        data = frontmatter.yaml.safe_load("key: value")
        assert data == {"key": "value"}

    def test_no_external_deps_needed(self):
        """frontmatter.py only imports yaml (PyYAML) which is already a dep."""
        import ast
        import inspect

        source = inspect.getsource(
            __import__("sr2_spectre.planning.frontmatter", fromlist=[""])
        )
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        # The only external (non-sr2_spectre) import should be yaml
        external = [i for i in imports if not i.startswith("sr2_spectre.") and i not in (
            "__future__", "logging", "pathlib", "typing"
        )]
        assert external == ["yaml"], f"Unexpected external deps: {external}"


# =========================================================================
# 10. _coerce_str helper — DRY field coercion (spc-19)
# =========================================================================


class TestCoerceStr:
    """Tests for the _coerce_str helper that replaces the repeated
    ``data.get(k, d); if x is None: x = d; x = str(x).strip()`` pattern."""

    def test_present_value_stripped(self):
        """Present non-None value is converted to string and stripped."""
        assert _coerce_str({"key": "  hello  "}, "key", "") == "hello"

    def test_missing_key_returns_default(self):
        """Missing key returns the default."""
        assert _coerce_str({}, "missing", "default") == "default"

    def test_none_value_returns_default(self):
        """Explicitly None value is treated as missing — returns default."""
        assert _coerce_str({"key": None}, "key", "fallback") == "fallback"

    def test_non_string_value_coerced(self):
        """Non-string values (int, float) are coerced to string."""
        assert _coerce_str({"key": 42}, "key", "") == "42"
        assert _coerce_str({"key": 3.14}, "key", "") == "3.14"

    def test_empty_string_stays_empty(self):
        """Empty string value is preserved (not treated as None)."""
        assert _coerce_str({"key": ""}, "key", "default") == ""

    def test_default_is_stripped(self):
        """Default value is also stripped."""
        assert _coerce_str({}, "missing", "  default  ") == "default"


# =========================================================================
# 11. OCP dispatch — _FRONTMATTER_PARSERS & _build_frontmatter
# =========================================================================


class TestFrontmatterDispatch:
    """Tests for the kind→parser dispatch table (spc-16)."""

    def test_parser_registry_contains_all_kinds(self):
        """The dispatch table covers every recognized kind."""
        assert set(_FRONTMATTER_PARSERS.keys()) == RECOGNIZED_KINDS

    def test_task_parser_registered(self):
        assert _FRONTMATTER_PARSERS["task"] is _parse_task

    def test_plan_parser_registered(self):
        assert _FRONTMATTER_PARSERS["plan"] is _parse_plan

    def test_knowledge_parser_registered(self):
        assert _FRONTMATTER_PARSERS["project-knowledge"] is _parse_knowledge

    def test_build_frontmatter_delegates_to_parser(self):
        """_build_frontmatter dispatches through the table, not if/elif."""
        from sr2_spectre.planning.models import TaskFrontmatter
        result = _build_frontmatter(
            TaskFrontmatter,
            {"plan": "test", "order": 1, "status": "pending"},
            "task",
            "label",
        )
        assert isinstance(result, TaskFrontmatter)
        assert result.plan == "test"
        assert result.order == 1

    def test_build_frontmatter_unknown_kind_raises(self):
        """Unknown kind raises ValueError (shouldn't reach here in normal flow
        because parse_frontmatter filters via RECOGNIZED_KINDS first)."""
        from sr2_spectre.planning.models import TaskFrontmatter
        with pytest.raises(ValueError, match="Unknown kind"):
            _build_frontmatter(
                TaskFrontmatter,
                {},
                "nonexistent-kind",
                "label",
            )

    def test_build_frontmatter_plan_kind(self):
        from sr2_spectre.planning.models import PlanFrontmatter
        result = _build_frontmatter(
            PlanFrontmatter,
            {"slug": "my-plan", "status": "open", "goal": "test"},
            "plan",
            "label",
        )
        assert isinstance(result, PlanFrontmatter)
        assert result.slug == "my-plan"

    def test_build_frontmatter_knowledge_kind(self):
        from sr2_spectre.planning.models import KnowledgeFrontmatter
        result = _build_frontmatter(
            KnowledgeFrontmatter,
            {"project": "my-project"},
            "project-knowledge",
            "label",
        )
        assert isinstance(result, KnowledgeFrontmatter)
        assert result.project == "my-project"

    def test_no_if_elif_chain_in_build_frontmatter(self):
        """Verify _build_frontmatter source contains no if/elif kind dispatch.

        This is an architectural test — if someone reverts to an if/elif
        chain, this test will fail and call it out.
        """
        import inspect
        source = inspect.getsource(_build_frontmatter)
        # The function should use _FRONTMATTER_PARSERS, not kind == "..."
        assert "_FRONTMATTER_PARSERS" in source, "Should use dispatch table"
        # Check body lines (skip docstring) for elif kind == pattern
        lines = source.split("\n")
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if '"""' in stripped:
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            # In executable code, there should be no "elif kind ==" pattern
            assert "elif" not in stripped, f"Should not have if/elif kind chain: {stripped}"
            assert 'kind ==' not in stripped, f"Should not have if/elif kind chain: {stripped}"
