"""Tests for TurnResult (core/__init__.py).

TurnResult was moved from core/loop.py (now deleted) into core/__init__.py.
This file tests only the TurnResult dataclass, which is the public export
plugins depend on.
"""

from sr2_spectre.core import TurnResult


def test_turn_result_defaults() -> None:
    r = TurnResult(text="hello")
    assert r.text == "hello"
    assert r.tool_calls_executed == 0
    assert r.total_tokens == 0


def test_turn_result_with_values() -> None:
    r = TurnResult(text="answer", tool_calls_executed=2, total_tokens=200)
    assert r.text == "answer"
    assert r.tool_calls_executed == 2
    assert r.total_tokens == 200
