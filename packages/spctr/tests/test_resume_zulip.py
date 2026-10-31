"""Zulip archive restoration through real Runtime/Session and fake I/O."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from sr2.models import TextBlock
from sr2.protocols.llm import StreamEvent

from sr2_spectre.agent import Agent
from sr2_spectre.config import ResumeConfig, SpectreConfig
from sr2_spectre.interfaces.zulip.client import ZulipCredentials
from sr2_spectre.interfaces.zulip.config import ZulipConfig
from sr2_spectre.interfaces.zulip.direct import DMessage, ZulipDMInterface
from sr2_spectre.interfaces.zulip.interface import ZulipInterface


MARKER = "[This turn was interrupted by a restart before it finished.]"
NOTICE = "Earlier context for this conversation could not be restored."


class FakeLLM:
    model = "test-model"

    def __init__(self):
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield StreamEvent(type="text", text=f"answer-{len(self.requests)}")
        yield StreamEvent(type="end")


class FakeClient:
    def __init__(self):
        self.messages = {}
        self.prior = []

    async def get_topic_messages(self, channel_id, topic):
        return {"messages": self.prior}

    async def send_message(self, **kwargs):
        message_id = len(self.messages) + 1
        self.messages[message_id] = kwargs["content"]
        return message_id

    async def update_message(self, message_id, content):
        self.messages[message_id] = content

    async def send_direct(self, peer_id, content):
        return await self.send_message(content=content)


def config(path, enabled=True):
    return SpectreConfig(
        agent={"name": "miranda", "resume": {"enabled": enabled}},
        models={"default": {"model": "test-model"}},
        pipeline={"layers": [{"name": "conversation", "target": "messages",
                              "resolvers": [{"type": "session"}, {"type": "input"}]}]},
        memory_store_dsn="", provenance_store_path=str(path),
    )


async def start(monkeypatch, path, mode="topic", enabled=True):
    llm = FakeLLM()
    monkeypatch.setattr("sr2_spectre.live_llm.LiteLLMCallable", lambda **kw: llm)
    agent = Agent(config(path, enabled))
    zulip = ZulipConfig(zuliprc="unused", api_url="https://zulip.example", channels=[2],
                        edit_stream_interval=0, mode="direct" if mode == "dm" else "stream",
                        peer_id=8 if mode == "dm" else None)
    interface = ZulipDMInterface(zulip) if mode == "dm" else ZulipInterface(zulip)
    interface._agent = agent
    interface._client = FakeClient()
    interface._credentials = ZulipCredentials(site="https://zulip.example", email="bot@example", api_key="fake")
    await agent.initialize()
    return agent, interface, llm


async def send(interface, text, index=1, topic="T"):
    if isinstance(interface, ZulipDMInterface):
        await interface._handle_message(DMessage(index, 8, text))
        return interface._session
    await interface._handle_message({"id": index, "type": "stream", "stream_id": 2,
                                     "subject": topic, "sender_id": 8, "content": text},
                                    reload=False)
    return next(entry[0] for entry in interface._sessions.values()
                if entry[0].run_context.source == f"zulip:2:{topic}")


def texts(messages):
    return [block.text for message in messages for block in message.content
            if isinstance(block, TextBlock)]


@pytest.mark.parametrize("mode", ["topic", "dm"], ids=["AC5-c1", "AC5-c2"])
async def test_restart_restores_first_request(monkeypatch, tmp_path, mode):
    path = tmp_path / "provenance.db"
    a, interface, _ = await start(monkeypatch, path, mode)
    try:
        await send(interface, "earlier-one")
        await send(interface, "earlier-two", 2)
    finally:
        await a.aclose()
    b, interface, llm = await start(monkeypatch, path, mode)
    try:
        # Real populated-topic restart must bypass the old cold-topic refusal.
        interface._client.prior = [{"id": 1}, {"id": 2}]
        session = await send(interface, "after-restart", 3)
        request = texts(llm.requests[0].messages)
        assert request == ["earlier-one", "answer-1", "earlier-two", "answer-2", "after-restart"]
        entries = await b._runtime._provenance_store.get_session(session.frame_id, layer="turn_archive")
        current = [entry for entry in entries if entry.meta["turn_seq"] == 2]
        assert texts([entry.content for entry in current]) == ["after-restart", "answer-1"]  # AC5-c4
        assert len(entries) == 6
        prior = max((entry for entry in entries if entry.meta["turn_seq"] == 1), key=lambda e: e.meta["seq"])
        assert min(current, key=lambda e: e.meta["seq"]).sources == (prior.id,)
        # Restoration is once per Session, including a repeated explicit request.
        store = b._runtime._provenance_store
        store.get_session = AsyncMock(wraps=store.get_session)
        await session.restore_history()
        await send(interface, "next", 4)
        store.get_session.assert_not_awaited()
        if mode == "topic":
            interface._client.prior = []
            await send(interface, "new-topic", 5, topic="U")
            assert texts(llm.requests[-1].messages) == ["new-topic"]  # AC5-c3
    finally:
        await b.aclose()


@pytest.mark.parametrize("failure", ["absent", "raises"], ids=["AC6-c1", "AC6-c2"])
async def test_restore_failure_notice_once_outside_history(monkeypatch, tmp_path, caplog, failure):
    path = "" if failure == "absent" else tmp_path / "provenance.db"
    agent, interface, llm = await start(monkeypatch, path)
    agent.config.agent.resume.restore_failed_notice = "custom restore notice"
    if failure == "raises":
        agent._runtime._provenance_store.get_session = AsyncMock(side_effect=OSError("unreadable"))
    try:
        session = await send(interface, "first")
        assert interface._client.messages[1] == "custom restore notice\n\nanswer-1"
        assert texts(llm.requests[0].messages) == ["first"]
        assert session.frame_id in caplog.text
        await send(interface, "second", 2)
        assert interface._client.messages[2] == "answer-2"
        assert texts(session.history) == ["first", "answer-1", "second", "answer-2"]
        assert texts(llm.requests[1].messages) == ["first", "answer-1", "second"]
    finally:
        await agent.aclose()


async def test_disabled_resume_has_no_history_or_notice(monkeypatch, tmp_path):
    """AC6-c3: disabled restore still continues unconditional archive numbering."""
    path = tmp_path / "provenance.db"
    a, interface, _ = await start(monkeypatch, path)
    try:
        await send(interface, "old")
    finally:
        await a.aclose()
    b, interface, llm = await start(monkeypatch, path, enabled=False)
    store = b._runtime._provenance_store
    store.get_session = AsyncMock(wraps=store.get_session)
    try:
        session = await send(interface, "fresh", 2)
        assert texts(llm.requests[0].messages) == ["fresh"]
        assert interface._client.messages[1] == "answer-1"
        assert texts(session.history) == ["fresh", "answer-1"]
        store.get_session.assert_awaited_once_with(session.frame_id, layer="turn_archive")
    finally:
        await b.aclose()


async def test_facade_and_pipeline_rebuild_continue_archive(monkeypatch, tmp_path):
    agent, _, llm = await start(monkeypatch, tmp_path / "provenance.db")
    try:
        await agent.handle_user_message("facade-first")
        agent.new_session(agent.session_id)
        await agent.handle_user_message("facade-rebuilt")
        agent._session.apply_config(agent.config, rebuild_sr2=True)
        await agent.handle_user_message("pipeline-rebuilt")
        entries = await agent._runtime._provenance_store.get_session(agent.session_id, layer="turn_archive")
        assert sorted((entry.meta["turn_seq"], entry.meta["seq"]) for entry in entries) == [
            (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        assert texts(llm.requests[1].messages) == ["facade-rebuilt"]
    finally:
        await agent.aclose()


def test_resume_defaults_and_documentation():
    """AC6-c4, AC8-c1: defaults are explicit in both configuration documents."""
    expected = {"enabled": True, "interrupted_marker": MARKER, "restore_failed_notice": NOTICE}
    assert ResumeConfig().model_dump() == expected
    assert config("").agent.resume.model_dump() == expected
    root = Path(__file__).resolve().parents[1]
    assert yaml.safe_load((root / "config.example.yaml").read_text())["agent"]["resume"] == expected
    reference = (root / "docs/CONFIG-REFERENCE.md").read_text()
    for key, default in expected.items():
        row = next(line for line in reference.splitlines() if f"`agent.resume.{key}` |" in line)
        assert f"`{'true' if default is True else default}`" in row
