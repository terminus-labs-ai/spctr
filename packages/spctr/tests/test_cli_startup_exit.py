"""Cage tests for obsidian-c573 AC3: a fatal startup exception exits nonzero.

After the MCP cancellation escaped cli.main(), interpreter shutdown blocked on
a leftover non-daemon thread: the process never reached an exit status, so
systemd (Restart=always) saw a healthy unit for ~8 hours. The contract: a
fatal exception out of the async run must terminate the process with a nonzero
exit status, while normal shutdown keeps its current behavior.

In-process expression of that contract: a fatal startup error must terminate
via an explicit hard exit (os._exit) — the mechanism that bypasses the thread
joins at interpreter shutdown, which is what hung for 8 hours. A plain
SystemExit is NOT sufficient: it still runs interpreter shutdown and hangs on
a stray non-daemon thread. os._exit is patched here to record its code and
raise SystemExit so the test runner survives; the test asserts BOTH the hard
exit fired with a nonzero code and the resulting exit status.

Subprocess-level proof against the real process (bounded timeout, no hang with
stray threads) is journey E's tests/test_cli_startup_exit_e2e.py, not this
file. No real bots are started and no ports are bound here.
"""
from __future__ import annotations

import logging
import os

import pytest

from sr2_spectre import cli


def _minimal_args() -> list[str]:
    return ["config.yaml", "hello", "--interface", "single_shot"]


def _install_fatal_run(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make cli.main's async run raise a fatal startup error.

    Returns the list of codes passed to os._exit: the patched hard exit
    records its call and then raises SystemExit with the same code so the
    test runner survives. A fix that never hard-exits leaves the list empty.
    """
    hard_exits: list[int] = []

    async def _fatal_run_async(argv: object = None) -> None:
        raise RuntimeError("fatal startup failure")

    def _fake_hard_exit(code: int = 0) -> None:  # mirrors os._exit's contract
        hard_exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(cli, "run_async", _fatal_run_async)
    monkeypatch.setattr(os, "_exit", _fake_hard_exit)
    return hard_exits


def test_main_exits_nonzero_when_run_async_raises_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fatal startup exception raised out of the async run must terminate
    via a hard exit (os._exit) with a nonzero code.

    Pre-fix, main() let the exception propagate to the interpreter's default
    shutdown path — the path that hung. A plain SystemExit is not enough: it
    still runs interpreter shutdown, which blocks on a stray non-daemon
    thread. Only os._exit bypasses those joins, so the cage requires the
    hard exit to have actually fired, not merely an exit status.
    """
    hard_exits = _install_fatal_run(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(_minimal_args())

    assert excinfo.value.code not in (0, None), (
        "a fatal startup exception must exit nonzero so systemd "
        f"Restart=always fires; got exit code {excinfo.value.code!r}"
    )
    assert len(hard_exits) == 1 and hard_exits[0] not in (0, None), (
        "the fix must hard-exit via os._exit (which bypasses the thread "
        "joins at interpreter shutdown); a plain SystemExit still hangs on "
        f"a stray non-daemon thread; hard-exit calls: {hard_exits!r}"
    )


def test_main_nonzero_exit_keeps_the_failure_visible(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Converting the fatal error to an exit must not swallow it: the failure
    message stays visible.

    Either channel satisfies this: an ERROR-level logging record whose text
    contains the failure message (production's console StreamHandler puts it
    on stderr) or a direct stderr write. Not both — a fix that logs and also
    prints duplicates the failure.
    """
    _install_fatal_run(monkeypatch)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            cli.main(_minimal_args())

    logged = any(
        rec.levelno >= logging.ERROR and "fatal startup failure" in rec.getMessage()
        for rec in caplog.records
    )
    err = capsys.readouterr().err
    assert logged or "fatal startup failure" in err, (
        "the fatal error must remain visible (an ERROR log record or stderr) "
        "alongside the exit"
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
