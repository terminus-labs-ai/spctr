"""GDScript support in read_symbol.

The tool only understood Python declarations, so on a Godot project it always
answered "symbol not found" — worse than useless for a small model, which
cannot tell "absent" from "unsupported syntax".
"""
import pytest

from sr2_spectre.tools.builtins.read_symbol import ReadSymbolTool, find_symbol

SOURCE = '''extends CharacterBody2D
class_name Player

signal jumped(height: float)

const MAX_SPEED := 400.0
@export var gravity: float = 1200.0

enum State {
\tIDLE,
\tRUNNING,
}

func _ready() -> void:
\tset_physics_process(true)

func jump(height: float) -> void:
\t# push upward
\tvelocity.y = -height
\tjumped.emit(height)

class InnerHelper:
\tvar counter := 0

\tfunc bump() -> void:
\t\tcounter += 1

func _physics_process(delta: float) -> void:
\tmove_and_slide()
'''


@pytest.fixture
def gd_file(tmp_path):
    p = tmp_path / "player.gd"
    p.write_text(SOURCE)
    return str(p)


def test_finds_a_function(gd_file):
    info = find_symbol(gd_file, "jump")
    assert info.kind == "function"
    assert "velocity.y = -height" in info.body
    assert "jumped.emit(height)" in info.body
    # Must stop before the next declaration.
    assert "class InnerHelper" not in info.body


def test_function_body_stops_at_the_next_func(gd_file):
    info = find_symbol(gd_file, "_ready")
    assert "set_physics_process(true)" in info.body
    assert "func jump" not in info.body


def test_finds_a_static_and_indented_method(tmp_path):
    p = tmp_path / "u.gd"
    p.write_text("class_name U\n\nstatic func helper(x):\n\treturn x\n")
    assert find_symbol(str(p), "helper").kind == "function"


def test_inner_class_method_is_a_method(gd_file):
    info = find_symbol(gd_file, "bump")
    assert info.kind == "method"
    assert "counter += 1" in info.body


def test_finds_an_inner_class(gd_file):
    info = find_symbol(gd_file, "InnerHelper")
    assert info.kind == "class"
    assert "var counter" in info.body
    assert "func bump" in info.body


def test_class_name_covers_the_whole_file(gd_file):
    """`class_name Foo` types the file, so the file is the definition."""
    info = find_symbol(gd_file, "Player")
    assert info.kind == "class"
    assert info.start_line == 2
    assert "func _physics_process" in info.body


def test_finds_a_signal(gd_file):
    info = find_symbol(gd_file, "jumped")
    assert info.kind == "signal"
    assert info.start_line == info.end_line
    assert "signal jumped(height: float)" in info.body


def test_finds_a_const(gd_file):
    info = find_symbol(gd_file, "MAX_SPEED")
    assert info.kind == "constant"
    assert "400.0" in info.body


def test_finds_an_exported_var(gd_file):
    info = find_symbol(gd_file, "gravity")
    assert info.kind == "variable"
    assert "@export" in info.body


def test_finds_an_enum(gd_file):
    info = find_symbol(gd_file, "State")
    assert info.kind == "enum"
    assert "RUNNING" in info.body


def test_missing_symbol_says_it_searched_gdscript(gd_file):
    with pytest.raises(ValueError, match="GDScript"):
        find_symbol(gd_file, "no_such_thing")


def test_python_files_are_unaffected(tmp_path):
    """Dispatch is by extension; Python behaviour must not change."""
    p = tmp_path / "m.py"
    p.write_text("class Widget:\n    x: int = 1\n\n\ndef build():\n    return Widget()\n")
    assert find_symbol(str(p), "Widget").kind == "class"
    assert find_symbol(str(p), "build").kind == "function"


@pytest.mark.asyncio
async def test_tool_renders_a_gdscript_symbol(gd_file):
    out = await ReadSymbolTool()(file_path=gd_file, symbol_name="jump")
    assert "Symbol: jump (function)" in out
    assert "velocity.y" in out
