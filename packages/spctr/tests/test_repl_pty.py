"""End-to-end REPL rendering tests against a real PTY.

Why a PTY and a terminal emulator instead of the in-memory Rich console the
other REPL tests use: the bug this file guards is *erasure*.  A Rich ``Live``
region erases itself by walking the cursor up N rows and clearing each one,
which only works while all N rows are still on screen.  An in-memory sink
records every byte ever written and performs no erasure at all, so it cannot
tell a frame that was wiped from one that was left behind — the in-memory
tests passed through two failed attempts at this bug.

So the turn runs in a child process attached to a real PTY, and the output is
replayed through ``pyte`` (a terminal emulator) to reconstruct what the user
would actually see: scrollback plus the visible screen.
"""
from __future__ import annotations

import fcntl
import os
import pty
import select
import struct
import subprocess
import sys
import termios

import pytest

pyte = pytest.importorskip("pyte")


_CHILD_TEMPLATE = '''\
import asyncio, sys
sys.path[:0] = {path!r}

from sr2_spectre.interfaces.repl import REPLInterface
from sr2_spectre.events import (
    AgentDone, AgentTextDelta, AgentToolResult, AgentToolStart,
)

NARRATION = {narration!r}
ANSWER = {answer!r}


def _deltas(s, n=8):
    return [s[i:i + n] for i in range(0, len(s), n)]


class FakeAgent:
    session_id = "abcdef123456"
    history = [1, 2, 3]

    async def stream_message(self, text):
        for d in _deltas(NARRATION):
            yield AgentTextDelta(text=d)
            await asyncio.sleep(0.005)
        yield AgentToolStart(name="terminal", input={{"command": "ls"}})
        await asyncio.sleep(0.02)
        yield AgentToolResult(name="terminal", is_error=False)
        await asyncio.sleep(0.02)
        for d in _deltas(ANSWER):
            yield AgentTextDelta(text=d)
            await asyncio.sleep(0.005)
        yield AgentDone(tool_calls_executed=1)


asyncio.run(REPLInterface()._stream_turn(FakeAgent(), "where is foo?"))
'''


def _run_on_pty(script: str, cols: int, rows: int, timeout: float = 60.0) -> str:
    """Run *script* on a PTY of the given size; return scrollback + screen text."""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    env = dict(os.environ, TERM="xterm-256color", COLUMNS=str(cols), LINES=str(rows))
    env.pop("NO_COLOR", None)
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True,
    )
    os.close(slave)

    chunks: list[bytes] = []
    try:
        while True:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
            elif proc.poll() is not None:
                break
    finally:
        os.close(master)
        proc.wait(timeout=timeout)

    assert proc.returncode == 0, b"".join(chunks).decode(errors="replace")

    screen = pyte.HistoryScreen(cols, rows, history=4000, ratio=1.0)
    pyte.ByteStream(screen).feed(b"".join(chunks))

    scrollback = [
        "".join(line[x].data for x in range(screen.columns)).rstrip()
        for line in screen.history.top
    ]
    return "\n".join(scrollback + [line.rstrip() for line in screen.display])


def _long_answer(bullets: int = 40) -> str:
    body = "".join(f"- bullet number {i} carrying enough filler to wrap\n" for i in range(bullets))
    return f"## Summary\n\nThe **answer** is that `foo()` lives in `bar.py`.\n\n{body}\nDone here.\n"


@pytest.mark.parametrize(("cols", "rows"), [(80, 24), (100, 30), (60, 20), (80, 12)])
def test_long_reply_is_not_rendered_twice(cols, rows) -> None:
    """A reply taller than the terminal must reach scrollback exactly once.

    Regression: the streaming Live frame grew past the screen, so the rows that
    scrolled off the top were beyond the reach of its erase sequence.  They
    stayed in scrollback as raw, unparsed text and the Markdown commit printed
    the same content underneath them — the reported "everything appears twice",
    with the second copy Markdown-parsed and the first not.
    """
    narration = "Let me check the repository layout first.\n"
    answer = _long_answer()
    script = _CHILD_TEMPLATE.format(
        path=[str(_src_dir())], narration=narration, answer=answer
    )

    out = _run_on_pty(script, cols, rows)

    for marker in (
        "Let me check the repository layout first.",
        "bullet number 7 carrying enough filler to wrap",
        "bullet number 33 carrying enough filler to wrap",
        "Done here.",
    ):
        assert out.count(marker) == 1, (
            f"{marker!r} appears {out.count(marker)}x at {cols}x{rows}; "
            "the streaming frame leaked into scrollback.\n" + out
        )


def test_interim_narration_survives_the_turn() -> None:
    """Text from a round that ends in a tool call is committed, not discarded.

    The transient Live frame is scratch; if a round's narration is only ever
    shown there it is wiped on exit and the user never sees what the model
    said before it called the tool.
    """
    narration = "Checking the layout before I answer.\n"
    script = _CHILD_TEMPLATE.format(
        path=[str(_src_dir())], narration=narration, answer="All set.\n"
    )

    out = _run_on_pty(script, 80, 24)

    assert out.count("Checking the layout before I answer.") == 1
    assert out.count("All set.") == 1
    # Reading order: narration, then the tool it led to, then the answer.
    assert out.index("Checking the layout") < out.index("⚙ terminal")
    assert out.index("⚙ terminal") < out.index("All set.")


def _src_dir() -> str:
    """The repo's src/ directory, so the child imports the working tree."""
    import sr2_spectre

    return os.path.dirname(os.path.dirname(os.path.abspath(sr2_spectre.__file__)))


# ---------------------------------------------------------------------------
# Row arithmetic behind the frame cap
# ---------------------------------------------------------------------------


def test_rows_counts_wrapped_lines() -> None:
    from sr2_spectre.interfaces.repl import _rows

    assert _rows("", 10) == 1
    assert _rows("abc", 10) == 1
    assert _rows("a" * 25, 10) == 3
    assert _rows("a\nb\nc", 10) == 3


def test_tail_rows_never_exceeds_the_cap() -> None:
    from sr2_spectre.interfaces.repl import _rows, _tail_rows

    text = "\n".join(f"line {i} " + "x" * 30 for i in range(50))
    for width, cap in ((20, 5), (40, 12), (80, 3), (13, 7)):
        tail = _tail_rows(text, width, cap)
        assert _rows(tail, width) <= cap, (width, cap, tail)
        # It is a tail: the very end of the text is always retained.
        assert text.endswith(tail.rsplit("\n", 1)[-1])


def test_tail_rows_crops_a_single_overlong_line() -> None:
    """One unbroken line longer than the cap is cropped from the front."""
    from sr2_spectre.interfaces.repl import _rows, _tail_rows

    text = "y" * 500
    tail = _tail_rows(text, 20, 4)
    assert _rows(tail, 20) <= 4
    assert text.endswith(tail)


def test_tail_rows_passes_short_text_through() -> None:
    from sr2_spectre.interfaces.repl import _tail_rows

    assert _tail_rows("hello\nworld", 80, 10) == "hello\nworld"
