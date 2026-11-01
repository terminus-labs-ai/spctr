"""Entry-point resolution for the renamed spctr distribution (AC4).

AC4-r1 (spec spctr-public-v0 FR3): every entry point declared by the
installed ``spctr`` distribution resolves (``.load()``) to an importable
object, every ``sr2.*`` group target is an ``spctr.`` module path, and
the console script is the bare ``spctr`` — the old ``sr2-spectre``
console script is gone (no alias, dec-280).
"""

from __future__ import annotations

import importlib.metadata as md


def _spctr_entry_points() -> list[md.EntryPoint]:
    dist = md.distribution("spctr")
    return list(dist.entry_points)


def test_ac4_r1_every_spctr_entry_point_loads_from_spctr_modules() -> None:
    """AC4-r1: all declared entry points load; all module paths are spctr.*."""
    entry_points = _spctr_entry_points()
    assert entry_points, "spctr declares no entry points at all"
    groups = {ep.group for ep in entry_points}
    assert "sr2.tool_providers" in groups
    assert "sr2.resolvers" in groups
    assert "sr2.transformers" in groups
    for ep in entry_points:
        module = ep.value.split(":")[0]
        assert module == "spctr" or module.startswith("spctr."), (
            f"entry point {ep.group}/{ep.name} targets {module!r}, not spctr.*"
        )
        obj = ep.load()  # resolves the module and the attribute
        assert obj is not None


def test_console_script_is_spctr_only() -> None:
    """FR3/dec-280: the console script is ``spctr``; no ``sr2-spectre``
    alias remains."""
    console = [
        ep for ep in _spctr_entry_points() if ep.group == "console_scripts"
    ]
    names = {ep.name for ep in console}
    assert names == {"spctr"}, f"console scripts: {sorted(names)}"
    assert console[0].value == "spctr.cli:main"
