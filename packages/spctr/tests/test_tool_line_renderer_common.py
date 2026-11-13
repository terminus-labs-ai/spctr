"""spc-1022 — the ONE tool-line/progress renderer lives in the shared module.

AC1: ``render_tool_line`` and ``render_progress_message`` have exactly one
implementation, in ``spctr.interfaces.common``; neither interface carries
its own copy of the tool-start / tool-result / progress formatting.
AC2: no format literal or width constant remains in either interface module
— every template, the line limit and the thinking placeholder come from the
interface's config object.

The per-interface golden and config-override tests live beside the
interfaces: ``test_discord_tool_line_renderer.py`` (Discord) and the AC9
rows of ``test_zulip_progress.py`` (Zulip, pre-existing, unchanged).
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from spctr.interfaces import common
from spctr.interfaces.common import render_progress_message, render_tool_line

#: The pre-bead literals this extraction must reproduce byte-for-byte.
START_LINE = "▶ `terminal` - `{'command': 'ls'}`"
OK_LINE = "✓ `terminal` - `a|b`"
ERROR_LINE = "✖ `terminal` - `bad`"
THINKING = "⏳ Thinking..."

_INTERFACES_ROOT = Path(inspect.getsourcefile(common)).parent


# ---------------------------------------------------------------------------
# render_tool_line — one implementation, truncation semantics intact
# ---------------------------------------------------------------------------


def test_render_tool_line_fills_name_and_value():
    assert render_tool_line(
        "▶ `{name}` - `{input}`", "terminal", "{'command': 'ls'}", 80
    ) == START_LINE
    assert render_tool_line(
        "✓ `{name}` - `{content}`", "terminal", "a|b", 80
    ) == OK_LINE
    assert render_tool_line(
        "✖ `{name}` - `{content}`", "terminal", "bad", 80
    ) == ERROR_LINE


def test_render_tool_line_cuts_value_to_max_chars_plus_ellipsis():
    line = render_tool_line("▶ `{name}` - `{input}`", "terminal", "x" * 81, 80)
    assert line == f"▶ `terminal` - `{'x' * 80}...`"
    # exactly at the limit: no cut, no marker
    assert render_tool_line("{input}", "t", "y" * 80, 80) == "y" * 80


def test_render_tool_line_single_pass_never_refills_embedded_braces():
    assert render_tool_line("{name} {input}", "{tool}", "{name}", 80) == (
        "{tool} {name}"
    )


def test_render_tool_line_defined_only_in_common():
    """AC1: the one tool-line formatter is common.py's."""
    assert inspect.getmodule(render_tool_line) is common


def test_render_progress_message_defined_only_in_common():
    """AC1: the one progress composition is common.py's."""
    assert inspect.getmodule(render_progress_message) is common


def test_no_interface_module_or_package_exposes_a_renderer():
    """AC1: neither interface defines a renderer of its own, and neither
    interface package re-exports one — the only seam is
    ``spctr.interfaces.common``. (The interface *modules* import the shared
    helpers, so their module globals necessarily name them; what is checked
    here is that the object behind the name is the shared function, and that
    no interface package publishes either name itself.)"""
    modules = [
        pytest.importorskip(name)
        for name in (
            "spctr.interfaces.discord",
            "spctr.interfaces.discord.interface",
            "spctr.interfaces.zulip",
            "spctr.interfaces.zulip.interface",
        )
    ]
    for module in modules:
        for name, shared in (
            ("render_tool_line", render_tool_line),
            ("render_progress_message", render_progress_message),
        ):
            value = getattr(module, name, None)
            if value is not None:
                assert value is shared, (
                    f"{module.__name__} carries its own {name}"
                )
    for pkg in ("spctr.interfaces.discord", "spctr.interfaces.zulip"):
        exported = __import__(pkg, fromlist=["__all__"]).__all__
        assert "render_tool_line" not in exported
        assert "render_progress_message" not in exported


# ---------------------------------------------------------------------------
# render_progress_message — one implementation of the progress composition
# ---------------------------------------------------------------------------


def test_render_progress_message_lines_then_text():
    assert render_progress_message(
        [START_LINE, OK_LINE], "partial", THINKING, 2000
    ) == f"{START_LINE}\n{OK_LINE}\n\npartial..."


def test_render_progress_message_thinking_placeholder_before_text():
    assert render_progress_message([], "", THINKING, 2000) == THINKING
    assert render_progress_message(
        [START_LINE], "", THINKING, 2000
    ) == f"{START_LINE}\n\n{THINKING}"


def test_render_progress_message_overridden_placeholder_flows_through():
    """Anti-vacuity: the placeholder text is the caller's config value, not
    a built-in — overriding it changes the rendered bytes."""
    assert render_progress_message([], "", "hmm", 2000) == "hmm"
    assert render_progress_message(
        ["L1"], "", "hmm", 2000
    ) == "L1\n\nhmm"
    assert THINKING not in render_progress_message(["L1"], "", "hmm", 2000)


def test_render_progress_message_tails_to_max_length():
    big = "x" * 3000
    message = render_progress_message(["LINE"], big, THINKING, 60)
    assert len(message) == 60
    assert message == "[…]\n" + ("x" * 3000 + "...")[-(60 - len("[…]\n")):]


# ---------------------------------------------------------------------------
# AC2 — no format literal or width constant left in an interface module
# ---------------------------------------------------------------------------

#: The templates and the width the interfaces used to hardcode. They may now
#: appear only in a config-default assignment, never anywhere else.
_TEMPLATES = (
    "▶ `{name}` - `{input}`",
    "✓ `{name}` - `{content}`",
    "✖ `{name}` - `{content}`",
    "⏳ Thinking...",
)


def _non_default_strings(tree: ast.AST) -> list[str]:
    """Every string constant in *tree* except class-level annotated
    assignments (which is how the config models declare their defaults)."""
    skip_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AnnAssign)):
            for target in getattr(node, "targets", [getattr(node, "target", None)]):
                if target is not None:
                    skip_ids.add(id(target))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip_ids
    ]


@pytest.mark.parametrize(
    "rel",
    ["discord/interface.py", "zulip/interface.py", "discord/config_source.py"],
)
def test_interface_modules_carry_no_tool_line_format_literals(rel: str):
    """AC2: no tool-line template, thinking placeholder or icon literal is
    built inside an interface module — only the config model's defaults
    name them."""
    source = (_INTERFACES_ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(source)
    strings = _non_default_strings(tree)
    for template in _TEMPLATES:
        assert template not in strings, (
            f"{rel} still hardcodes the message string {template!r}"
        )
    for icon in ("▶", "✓", "✖", "⏳"):
        assert not any(icon in value for value in strings), (
            f"{rel} still hardcodes an icon inside a string: {icon!r}"
        )


@pytest.mark.parametrize(
    "rel, names",
    [
        ("discord/interface.py", ["_render_tool_start", "_render_tool_result"]),
        ["zulip/interface.py", ["tool_start", "tool_result"]],
    ],
)
def test_tool_line_paths_read_templates_from_config(rel: str, names: list[str]):
    """AC2: each interface's tool-start and tool-result renderers pass their
    config's template and ``tool_line_max_chars`` into the shared renderer —
    no width constant, no inline format string."""
    source = (_INTERFACES_ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(source)

    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert len(methods) == len(names), (
        f"{rel}: expected a renderer per name in {names}, found "
        f"{[m.name for m in methods]}"
    )
    for method in methods:
        method_source = ast.get_source_segment(source, method)
        assert method_source is not None
        assert "render_tool_line(" in method_source, (
            f"{rel}::{method.name} does not call render_tool_line"
        )
        assert "tool_line_max_chars" in method_source, (
            f"{rel}::{method.name} passes no configured width"
        )
        if method.name.endswith("start"):
            assert "tool_start_line" in method_source, (
                f"{rel}::{method.name} reads no configured start template"
            )
        else:
            assert "tool_result_ok_line" in method_source, (
                f"{rel}::{method.name} reads no configured ok template"
            )
            assert "tool_result_error_line" in method_source, (
                f"{rel}::{method.name} reads no configured error template"
            )
        # No width constant and no inline format string survived.
        for node in ast.walk(method):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                raise AssertionError(
                    f"{rel}::{method.name} still holds the width constant "
                    f"{node.value!r}"
                )
            if isinstance(node, ast.JoinedStr):
                raise AssertionError(
                    f"{rel}::{method.name} still formats a line inline"
                )


@pytest.mark.parametrize(
    "rel", ["discord/interface.py", "zulip/interface.py"]
)
def test_progress_paths_compose_through_the_shared_renderer(rel: str):
    """AC2: the in-progress composition (and the placeholder it shows) is
    the shared renderer's, fed by config, in both interfaces."""
    tree = ast.parse((_INTERFACES_ROOT / rel).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "render_progress_message"
    ]
    assert calls, f"{rel} never calls render_progress_message"
    for call in calls:
        rendered = ast.dump(call)
        assert "thinking_message" in rendered, (
            f"{rel} does not read the placeholder from config"
        )
        assert "max_message_length" in rendered, (
            f"{rel} does not read the length limit from config"
        )
        assert not any(
            isinstance(value, ast.Constant) and isinstance(value.value, int)
            for value in ast.walk(call)
        ), f"{rel} passes a width constant to the progress renderer"
