"""Zulip handoff display and topic-scoped ``!stop`` behavior (spc-1051)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from spctr.events import (
    AgentHandoffEnd,
    AgentHandoffStart,
    AgentTextDelta,
    AgentToolStart,
)
from spctr.interfaces.zulip.config import ZulipConfig
from spctr.interfaces.zulip.interface import (
    ZulipInterface,
    is_stop_command,
    topic_frame_id,
)


class FakeClient:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.updated: list[tuple[int, str]] = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return len(self.sent)

    async def update_message(self, message_id, content):
        self.updated.append((message_id, content))


class FakeSession:
    def __init__(self, events=(), active_handoff=None) -> None:
        self.events = events
        self.active_handoff = active_handoff
        self.prompts: list[str] = []

    async def stream_message(self, prompt):
        self.prompts.append(prompt)
        for event in self.events:
            yield event
            await asyncio.sleep(0)


class FakeHandoff:
    agent = "role-c"

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def make_interface(**overrides) -> ZulipInterface:
    config = ZulipConfig(
        zuliprc="unused",
        api_url="https://zulip.example",
        edit_stream_interval=1.0,
        **overrides,
    )
    interface = ZulipInterface(config)
    interface._client = FakeClient()
    interface._agent = SimpleNamespace(
        config=SimpleNamespace(agent=SimpleNamespace(name="miranda"))
    )
    interface._sync_reminders = AsyncMock()
    return interface


def message(topic="Build", content="hello", sender_id=8, message_id=1):
    return {
        "id": message_id,
        "type": "stream",
        "stream_id": 2,
        "subject": topic,
        "sender_id": sender_id,
        "content": content,
    }


async def forever() -> None:
    await asyncio.Event().wait()


async def test_handoff_lines_and_foreign_events_stay_out_of_progress():
    """AC6-r1/r2: lines post in order; only caller events edit progress."""
    interface = make_interface()
    session = FakeSession([
        AgentHandoffStart(frame_id="fid", agent="role-c"),
        AgentTextDelta(text="foreign", frame_id="fid"),
        AgentToolStart(name="read", frame_id="fid"),
        AgentTextDelta(text="mine"),
        AgentHandoffEnd(
            frame_id="fid", agent="role-c", status="done", duration_s=1412.3
        ),
    ])

    await interface._run_turn(session, 2, "Build", message(), interface._config)

    assert interface._client.sent == [
        {
            "type": "stream",
            "to": 2,
            "topic": "Build",
            "content": "⏳ Thinking...",
        },
        {
            "type": "stream",
            "to": 2,
            "topic": "Build",
            "content": "→ role-c took over (fid)",
        },
        {
            "type": "stream",
            "to": 2,
            "topic": "Build",
            "content": "← role-c: done in 23m 32s",
        },
    ]
    assert interface._client.updated == [(1, "mine..."), (1, "mine")]


async def test_configured_handoff_templates_are_used():
    """AC6-r3: distinguishable YAML values drive both handoff posts."""
    interface = make_interface(
        handoff_start_message="START {agent}/{frame_id}",
        handoff_end_message="END {agent}/{status}/{duration}",
    )
    session = FakeSession([
        AgentHandoffStart(frame_id="frame-x", agent="role-g"),
        AgentHandoffEnd(
            frame_id="frame-x", agent="role-g", status="failed", duration_s=61
        ),
    ])

    await interface._run_turn(session, 2, "Build", message(), interface._config)

    contents = [call["content"] for call in interface._client.sent]
    assert contents[1:] == ["START role-g/frame-x", "END role-g/failed/1m 1s"]


async def test_stop_command_cancels_handoff_without_cancelling_turn():
    """AC7-r1: active handoff wins and the containing turn keeps running."""
    interface = make_interface(stop_handoff_message="HALT {agent}")
    frame_id = topic_frame_id("miranda", "https://zulip.example", 2, "Build")
    handoff = FakeHandoff()
    interface._sessions[frame_id] = (FakeSession(active_handoff=handoff), [])
    turn = asyncio.create_task(forever())
    interface._topic_tasks[frame_id] = {turn}
    interface._topic_tails[frame_id] = turn

    text = await interface._stop_topic(2, "Build")

    assert text == "HALT role-c"
    assert handoff.cancelled
    assert not turn.cancelled()
    assert interface._client.sent[-1]["content"] == "HALT role-c"
    turn.cancel()
    await asyncio.gather(turn, return_exceptions=True)


async def test_stop_cancels_only_the_topic_turn_and_queue():
    """AC7-r2: current and queued tasks stop without touching another topic."""
    interface = make_interface(stop_turn_message="TURN HALTED")
    first = topic_frame_id("miranda", "https://zulip.example", 2, "Build")
    second = topic_frame_id("miranda", "https://zulip.example", 2, "Other")
    first_tasks = {asyncio.create_task(forever()) for _ in range(2)}
    other_tasks = {asyncio.create_task(forever()) for _ in range(2)}
    interface._topic_tasks = {first: first_tasks, second: other_tasks}
    interface._topic_tails = {
        first: next(iter(first_tasks)), second: next(iter(other_tasks))
    }

    text = await interface._stop_topic(2, "Build")
    await asyncio.sleep(0)

    assert text == "TURN HALTED"
    assert all(task.cancelled() for task in first_tasks)
    assert all(not task.done() for task in other_tasks)
    assert first not in interface._topic_tails
    assert interface._topic_tails[second] in other_tasks
    assert interface._client.sent[-1]["content"] == "TURN HALTED"
    for task in other_tasks:
        task.cancel()
    await asyncio.gather(*other_tasks, return_exceptions=True)


async def test_stop_with_nothing_running_posts_idle_reply():
    """AC7-r3: idle stop is handled without entering a model turn."""
    interface = make_interface(stop_idle_message="IDLE")

    await interface._handle_message(message(content="!stop"), reload=False)
    assert interface._client.sent[-1]["content"] == "IDLE"
    assert interface._sessions == {}


async def test_dispatch_recognizes_exact_stop_before_model_queue():
    """AC7-r3: exact command dispatches stop and never queues a model turn."""
    interface = make_interface()
    interface._stop_topic = AsyncMock(return_value="nothing running")

    interface._dispatch(message(content="  !stop\n"), {1}, interface._config)
    await asyncio.sleep(0)

    interface._stop_topic.assert_awaited_once_with(2, "Build")
    assert interface._topic_tasks == {}
    assert is_stop_command("!stop")
    assert not is_stop_command("!stop now")


async def test_disallowed_stop_is_ignored():
    """AC7-r4: the ordinary allowlist gate precedes command handling."""
    interface = make_interface(users=[8])
    interface._stop_topic = AsyncMock(return_value="nothing running")

    interface._dispatch(message(content="!stop", sender_id=9), {1}, interface._config)
    await asyncio.sleep(0)

    interface._stop_topic.assert_not_awaited()
    assert interface._client.sent == []


def test_handoff_and_stop_message_defaults_and_documentation():
    """AC6-r3: every new message has the specified default and is documented."""
    expected = {
        "handoff_start_message": "→ {agent} took over ({frame_id})",
        "handoff_end_message": "← {agent}: {status} in {duration}",
        "stop_handoff_message": "⏹ stopped {agent}",
        "stop_turn_message": "⏹ stopped",
        "stop_idle_message": "nothing running",
    }
    config = ZulipConfig(zuliprc="unused")
    root = Path(__file__).resolve().parents[1]
    example = (root / "config.example.yaml").read_text()
    reference = (root / "docs/CONFIG-REFERENCE.md").read_text()

    for name, default in expected.items():
        assert getattr(config, name) == default
        assert f"#   {name}: \"{default}\"" in example
        assert f"| `zulip.{name}` |" not in reference
        assert f"| `{name}` |" in reference
        assert f'`"{default}"`' in reference
