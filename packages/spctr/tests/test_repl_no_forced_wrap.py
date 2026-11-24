"""spc-1116: the REPL inserts no line breaks; the terminal wraps.

Every case renders at width 40 and asserts on the raw captured output.
"""

from __future__ import annotations

import io
import re

from rich.console import Console

from spctr.interfaces.repl import REPLInterface, render_markdown

WIDTH = 40
LONG = " ".join(f"word{n:03d}" for n in range(25))  # 199 chars
COLOR_SGR = re.compile(r"\x1b\[(?:[0-9;]*;)?(?:3[0-9]|4[0-9]|9[0-7]|10[0-7])[;m]")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _render(markdown: str, *, color: bool = True) -> str:
    """Print a reply exactly as the REPL commits one, return raw output."""
    buf = io.StringIO()
    console = Console(
        file=buf, width=WIDTH, force_terminal=True, no_color=not color, soft_wrap=True
    )
    console.print(render_markdown(markdown), soft_wrap=True)
    return buf.getvalue()


def _lines(raw: str) -> list[str]:
    return [ANSI.sub("", line) for line in raw.split("\n") if ANSI.sub("", line)]


def test_ac1_paragraph_is_one_unpadded_line() -> None:
    assert len(LONG) > 4 * WIDTH
    assert _lines(_render(LONG)) == [LONG]


def test_ac1_list_item_is_one_line_untruncated() -> None:
    assert _lines(_render(f"- {LONG}")) == [f"• {LONG}"]


def test_ac1_blockquote_is_one_line() -> None:
    assert _lines(_render(f"> {LONG}")) == [f"▌ {LONG}"]


def test_ac1_heading_is_one_line() -> None:
    heading = LONG[:119]
    assert len(heading) == 119 and not heading.endswith(" ")
    assert _lines(_render(f"## {heading}")) == [heading]


def test_ac2_gmap_launch_line_is_verbatim_without_osc8() -> None:
    url = "gmap://start/a/b?harness=claude"
    raw = _render(f"Launch: [{url}]({url}) or !gmap launch '{url}'")
    assert "\x1b]8;" not in raw
    assert _lines(raw) == [f"Launch: {url} or !gmap launch '{url}'"]


def test_ac2_copydraft_line_is_verbatim_without_osc8() -> None:
    raw = _render("Copy: [copydraft://my-slug](copydraft://my-slug)")
    assert "\x1b]8;" not in raw
    assert _lines(raw) == ["Copy: copydraft://my-slug"]


def test_ac3_code_fence_keeps_its_lines() -> None:
    raw = _render("```\nalpha = 1\nbeta = 2\ngamma = 3\n```")
    lines = [ln.strip() for ln in _lines(raw) if ln.strip()]
    assert lines == ["alpha = 1", "beta = 2", "gamma = 3"]


def test_ac3_table_keeps_its_structure() -> None:
    lines = _lines(_render("| h1 | h2 |\n|----|----|\n| a1 | a2 |\n| b1 | b2 |"))
    assert len(lines) >= 4
    row = {cell: i for i, line in enumerate(lines) for cell in line.split()}
    assert row["h1"] == row["h2"] < row["a1"] == row["a2"] < row["b1"] == row["b2"]


def test_ac5_no_color_emits_no_colour_escapes() -> None:
    raw = _render(f"# Title\n\nSome **bold** and `code` text {LONG}", color=False)
    # NO_COLOR strips colour, as before; Rich still emits bold/italic.
    assert not COLOR_SGR.search(raw)
    assert f"Some bold and code text {LONG}" in _lines(raw)


def test_ac6_status_line_from_default_repl_console_is_one_line() -> None:
    iface = REPLInterface()
    buf = io.StringIO()
    iface.console.file = buf
    iface.console.width = WIDTH
    iface.console.print(f"[red]error:[/red] {LONG}")
    assert _lines(buf.getvalue()) == [f"error: {LONG}"]
