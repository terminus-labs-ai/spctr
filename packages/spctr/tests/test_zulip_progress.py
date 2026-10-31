"""spc-1015 — Zulip live tool and text progress edits (AC9, AC10).

Case-table rows of specs/zulip-channel-access.md, one test each, named
``test_<row>_...``. The real ``ZulipInterface`` poll loop runs over a fake
transport and a fake agent whose Session streams scripted ``(clock, event)``
pairs: before yielding an event it moves a fake clock (the interface's
injected ``clock`` seam) to that time, and after the consumer handles the
event it yields to the loop so a scheduled progress edit can run. The fake
transport answers the placeholder POST with ID 500 and logs every PATCH.
No model, no network, no sleeping on wall time.

Defaults unless a test says otherwise: ``tool_embed_enabled`` true,
``tool_line_max_chars`` 80, the default tool-line templates,
``edit_stream_interval`` 1.0.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from sr2_spectre.events import (
    AgentDone,
    AgentTextDelta,
    AgentToolResult,
    AgentToolStart,
)
from sr2_spectre.interfaces.common import tail_for_stream
from sr2_spectre.interfaces.zulip import ZulipConfig, ZulipInterface
from sr2_spectre.interfaces.zulip.client import ZulipCredentials
from sr2_spectre.resume import ResumeResult

DIEGO_ID = 8
BOT_ID = 9
CHANNEL = 2
THINKING = "⏳ Thinking..."
LS_START_LINE = "▶ `terminal` - `{'command': 'ls'}`"


def _start(**kw: Any) -> AgentToolStart:
    kw.setdefault("name", "terminal")
    kw.setdefault("input", {"command": "ls"})
    return AgentToolStart(tool_id="t1", **kw)


def _result(content: str = "a\nb", is_error: bool = False) -> AgentToolResult:
    return AgentToolResult(tool_id="t1", name="terminal", content=content,
                           is_error=is_error)


def _delta(text: str) -> AgentTextDelta:
    return AgentTextDelta(text=text)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


async def _yield_to_loop(times: int = 3) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


class _FakeSession:
    async def restore_history(self) -> ResumeResult:
        return ResumeResult([], interrupted=False, restore_failed=False)

    def __init__(self, clock: list[float], script: list[tuple[float, Any]]):
        self._clock = clock
        self._script = script

    def set_run_context(self, context: Any) -> None:
        pass

    async def stream_message(self, text: str):
        for at, event in self._script:
            self._clock[0] = at
            yield event
            await _yield_to_loop()
        yield AgentDone(tool_calls_executed=0)

    def close(self) -> None:
        pass


class _FakeAgent:
    def __init__(self, clock: list[float], script: list[tuple[float, Any]]):
        self.config = SimpleNamespace(agent=SimpleNamespace(name="miranda"))
        self._clock = clock
        self._script = script

    def create_session(self, frame_id: str) -> _FakeSession:
        return _FakeSession(self._clock, self._script)

    def apply_config(self, config: Any) -> None:
        pass


class _FakeClient:
    """One scripted message; logs ("PATCH", id, content) when an edit lands.

    With ``hold`` set, an edit whose content is not *final* blocks until
    the final edit has landed, then logs — so a progress edit that was not
    cancelled lands after the answer and shows up last in the log.
    """

    def __init__(self, interface: Any, log: list, hold: bool = False,
                 final: str | None = None):
        self._interface = interface
        self._log = log
        self._batches = [[{
            "id": 101, "type": "stream", "stream_id": CHANNEL,
            "display_recipient": "sandbox", "subject": "t",
            "sender_id": DIEGO_ID, "content": "go",
        }]]
        self._hold = hold
        self._final = final
        self._final_landed = asyncio.Event()

    async def register(self) -> dict:
        return {"queue_id": "q", "last_event_id": 0, "user_id": BOT_ID,
                "full_name": "Miranda",
                "realm_users": [{"user_id": DIEGO_ID, "is_bot": False},
                                {"user_id": BOT_ID, "is_bot": True}]}

    async def get_events(self, queue_id: str, last_event_id: int) -> dict:
        if not self._batches:
            # Let any stray progress edit run before the loop ends.
            await _yield_to_loop(10)
            self._interface._running = False
            return {"events": []}
        messages = self._batches.pop(0)
        return {"events": [{"id": m["id"], "type": "message", "message": m}
                           for m in messages]}

    async def get_topic_messages(self, channel_id: int, topic: str) -> dict:
        return {"messages": []}

    async def send_message(self, **kwargs: Any) -> int:
        self._log.append(("POST", kwargs["content"]))
        return 500

    async def update_message(self, message_id: int, content: str) -> None:
        if self._hold and content != self._final:
            self._log.append(("PATCH-STARTED", message_id, content))
            await self._final_landed.wait()
        self._log.append(("PATCH", message_id, content))
        if content == self._final:
            self._final_landed.set()
            await _yield_to_loop(10)

    async def aclose(self) -> None:
        pass


async def _drive(monkeypatch, script: list[tuple[float, Any]],
                 hold: bool = False, final: str | None = None,
                 **over: Any) -> list:
    """Run the real poll loop over one message; return the ordered log."""
    import sr2_spectre.interfaces.zulip.interface as iface_mod

    clock = [0.0]
    log: list = []
    interface = ZulipInterface(
        ZulipConfig(zuliprc="/unused.zuliprc", users=[DIEGO_ID], **over),
        clock=lambda: clock[0],
    )
    client = _FakeClient(interface, log, hold=hold, final=final)
    monkeypatch.setattr(iface_mod, "load_zulip_credentials",
                        lambda path: ZulipCredentials(
                            email="bot@test.local", api_key="k",
                            site="http://127.0.0.1:1"))
    monkeypatch.setattr(iface_mod, "ZulipClient", lambda **kw: client)
    agent = _FakeAgent(clock, script)
    await interface.start(agent)
    try:
        await interface.run(agent)
    finally:
        await interface.stop()
    return log


def _patches(log: list) -> list[str]:
    return [entry[2] for entry in log if entry[0] == "PATCH" and entry[1] == 500]


# ---------------------------------------------------------------------------
# AC9 — templated tool lines
# ---------------------------------------------------------------------------


async def test_AC9_a_tool_start_appends_start_line(monkeypatch):
    log = await _drive(monkeypatch, [(0.0, _start())])
    progress = _patches(log)[0]
    assert LS_START_LINE in progress
    assert progress == f"{LS_START_LINE}\n\n{THINKING}"


async def test_AC9_b_tool_result_ok_line_flattens_newlines(monkeypatch):
    log = await _drive(monkeypatch, [(0.0, _start()), (0.0, _result("a\nb"))])
    contents = _patches(log)
    assert any("✓ `terminal` - `a|b`" in c for c in contents)
    assert not any("a\nb" in c for c in contents)


async def test_AC9_c_tool_result_error_line(monkeypatch):
    log = await _drive(
        monkeypatch, [(0.0, _start()), (0.0, _result("a\nb", is_error=True))]
    )
    contents = _patches(log)
    assert any("✖ `terminal` - `a|b`" in c for c in contents)
    assert not any("✓" in c for c in contents)


async def test_AC9_d_long_input_and_result_truncated(monkeypatch):
    tool_input = {"k": "x" * 111}
    rendered = str(tool_input)
    assert len(rendered) == 120
    result = "y" * 120
    log = await _drive(
        monkeypatch,
        [(0.0, _start(input=tool_input)), (0.0, _result(result))],
    )
    last_progress = _patches(log)[-2]
    assert last_progress == (
        f"▶ `terminal` - `{rendered[:80]}...`\n"
        f"✓ `terminal` - `{'y' * 80}...`\n\n{THINKING}"
    )


async def test_AC9_d_limit_comes_from_config(monkeypatch):
    log = await _drive(monkeypatch, [(0.0, _start())], tool_line_max_chars=5)
    assert _patches(log)[0] == f"▶ `terminal` - `{{'com...`\n\n{THINKING}"


async def test_AC9_templates_come_from_config(monkeypatch):
    log = await _drive(
        monkeypatch,
        [(0.0, _start()), (0.0, _result("a\nb")),
         (0.0, _result("bad", is_error=True))],
        tool_start_line="START {name}: {input}",
        tool_result_ok_line="OK {name}: {content}",
        tool_result_error_line="ERR {name}: {content}",
    )
    assert _patches(log)[-2] == (
        "START terminal: {'command': 'ls'}\nOK terminal: a|b\n"
        f"ERR terminal: bad\n\n{THINKING}"
    )


async def test_AC9_e_tool_embed_disabled_shows_no_tool_lines(monkeypatch):
    log = await _drive(
        monkeypatch,
        [(0.0, _start()), (0.0, _result("a\nb")), (0.0, _delta("ok"))],
        tool_embed_enabled=False,
    )
    contents = _patches(log)
    assert contents == ["ok...", "ok"]
    assert not any("terminal" in c for c in contents)


# ---------------------------------------------------------------------------
# AC10 — throttled text progress, tail, no edit after the answer
# ---------------------------------------------------------------------------


async def test_AC10_a_text_edits_throttled_by_interval(monkeypatch):
    log = await _drive(
        monkeypatch,
        [(0.0, _delta("a")), (0.2, _delta("b")), (0.5, _delta("c")),
         (1.3, _delta("d"))],
    )
    assert _patches(log) == ["a...", "abcd...", "abcd"]


async def test_AC10_b_interval_zero_disables_text_edits_only(monkeypatch):
    log = await _drive(
        monkeypatch,
        [(0.0, _delta("x")), (0.0, _start()), (5.0, _delta("y"))],
        edit_stream_interval=0.0,
    )
    tool_render = f"{LS_START_LINE}\n\nx..."
    assert _patches(log) == [tool_render, "xy"]


async def test_AC10_c_progress_keeps_tail_within_max_length(monkeypatch):
    text = "".join(f"{i:03d}," for i in range(40))  # 160 characters
    log = await _drive(monkeypatch, [(0.0, _delta(text))],
                       max_message_length=100)
    progress = _patches(log)[0]
    assert progress == tail_for_stream(text + "...", 100)
    assert progress == "[…]\n" + (text + "...")[-96:]
    assert len(progress) <= 100


async def test_AC10_d_pending_progress_edit_never_lands_after_answer(monkeypatch):
    log = await _drive(monkeypatch, [(0.0, _delta("done"))],
                       hold=True, final="done")
    # The progress edit really was in flight when the turn finished ...
    assert ("PATCH-STARTED", 500, "done...") in log
    # ... and the answer is the last edit of the placeholder.
    assert _patches(log)[-1] == "done"
    assert "done..." not in _patches(log)


async def test_AC10_e_tool_lines_then_blank_line_then_text(monkeypatch):
    log = await _drive(monkeypatch, [(0.0, _start()), (2.0, _delta("hi"))])
    assert _patches(log) == [
        f"{LS_START_LINE}\n\n{THINKING}",
        f"{LS_START_LINE}\n\nhi...",
        "hi",
    ]
