"""Cage tests for obsidian-c573 AC3: a fatal startup exception exits nonzero.

After the MCP cancellation escaped cli.main(), interpreter shutdown blocked on
a leftover non-daemon thread: the process never reached an exit status, so
systemd (Restart=always) saw a healthy unit for ~8 hours. The contract: a
fatal exception out of the async run must terminate the process with a nonzero
exit status, while normal shutdown keeps its current behavior.

In-process expression of that contract: main() ends the run either by raising
SystemExit(nonzero) or by calling a hard exit (os._exit). Both are intercepted
here — os._exit is patched to raise SystemExit with its code so the test
runner survives — and the assertion is on the exit status, not the mechanism.

Subprocess-level proof against the real process (bounded timeout, no hang with
stray threads) is journey E's tests/test_cli_startup_exit_e2e.py, not this
file. No real bots are started and no ports are bound here.
"""
from __future__ import annotations

import os

import pytest

from sr2_spectre import cli


def _minimal_args() -> list[str]:
    return ["config.yaml", "hello", "--interface", "single_shot"]


def _install_fatal_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make cli.main's async run raise a fatal startup error, and convert any
    direct os._exit hard exit into SystemExit so the in-process contract (the
    exit status) is observable whichever mechanism the fix uses."""

    async def _fatal_run_async(argv: object = None) -> None:
        raise RuntimeError("fatal startup failure")

    def _fake_hard_exit(code: int = 0) -> None:  # mirrors os._exit's contract
        raise SystemExit(code)

    monkeypatch.setattr(cli, "run_async", _fatal_run_async)
    monkeypatch.setattr(os, "_exit", _fake_hard_exit)


def test_main_exits_nonzero_when_run_async_raises_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fatal startup exception raised out of the async run must surface as a
    nonzero exit status (SystemExit), not as a raw exception escaping main().

    Pre-fix, main() let the exception propagate to the interpreter's default
    shutdown path — the path that hung. Only an explicit exit terminates the
    process regardless of a stray non-daemon thread.
    """
    _install_fatal_run(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(_minimal_args())

    assert excinfo.value.code not in (0, None), (
        "a fatal startup exception must exit nonzero so systemd "
        f"Restart=always fires; got exit code {excinfo.value.code!r}"
    )


def test_main_nonzero_exit_keeps_the_failure_visible(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Converting the fatal error to an exit status must not swallow it: the
    failure message stays visible on stderr."""
    _install_fatal_run(monkeypatch)

    with pytest.raises(SystemExit):
        cli.main(_minimal_args())

    err = capsys.readouterr().err
    assert "fatal startup failure" in err, (
        "the fatal error must remain visible on stderr alongside the exit"
    )


def test_main_returns_normally_when_run_async_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal shutdown keeps its current behavior: main() returns without
    raising and without an explicit exit status."""
    calls: list[object] = []

    async def _ok_run_async(argv: object = None) -> None:
        calls.append(argv)

    monkeypatch.setattr(cli, "run_async", _ok_run_async)

    cli.main(_minimal_args())  # must not raise SystemExit

    assert len(calls) == 1
