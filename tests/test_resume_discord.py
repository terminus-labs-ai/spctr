"""Discord resume against real Runtime/SQLite sessions and fake transport/LLM."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sr2.models import TextBlock
from sr2.protocols.llm import StreamEvent

from sr2_spectre.agent import Agent
from sr2_spectre.config import SpectreConfig
from sr2_spectre.interfaces.discord.config import DiscordConfig
from sr2_spectre.interfaces.discord.interface import DiscordInterface


class FakeLLM:
    model = "test-model"

    def __init__(self):
        self.requests = []
        self.prefix = "answer"

    async def stream(self, request):
        self.requests.append(request)
        yield StreamEvent(type="text", text=f"{self.prefix}-{len(self.requests)}")
        yield StreamEvent(type="end")


class FakeAdapter:
    def __init__(self):
        self.messages = {}

    async def send_message(self, channel_id, content):
        message_id = len(self.messages) + 1
        self.messages[message_id] = content
        return SimpleNamespace(id=message_id)

    async def edit_message(self, channel_id, message_id, content):
        self.messages[message_id] = content

    async def interaction_send(self, interaction, content):
        await self.send_message(interaction.channel_id, content)

    @asynccontextmanager
    async def channel_typing(self, channel_id):
        yield


async def start(monkeypatch, path, enabled=True):
    llm = FakeLLM()
    monkeypatch.setattr("sr2_spectre.live_llm.LiteLLMCallable", lambda **kw: llm)
    config = SpectreConfig(
        agent={"name": "edi", "resume": {"enabled": enabled}},
        models={"default": {"model": "test-model"}},
        pipeline={"layers": [{"name": "conversation", "target": "messages",
                              "resolvers": [{"type": "session"}, {"type": "input"}]}]},
        memory_store_dsn="", provenance_store_path=str(path),
    )
    agent = Agent(config)
    await agent.initialize()
    interface = DiscordInterface(DiscordConfig(edit_stream_interval=0))
    interface._agent = agent
    interface._adapter = FakeAdapter()
    return agent, interface, llm


def texts(messages):
    return [block.text for message in messages for block in message.content
            if isinstance(block, TextBlock)]


async def command(interface, name, native, channel_id=10):
    if native:
        await interface._dispatch_slash(name, "", SimpleNamespace(channel_id=channel_id), channel_id, None)
    else:
        await interface._handle_command(name, "", channel_id)


async def test_restart_restores_prior_turns_once(monkeypatch, tmp_path):
    """AC7-c1: exact first request also rules out duplicate current input."""
    path = tmp_path / "provenance.db"
    a, interface, _ = await start(monkeypatch, path)
    try:
        await interface._run_agent("first", 10)
        await interface._run_agent("second", 10)
    finally:
        await a.aclose()
    b, interface, llm = await start(monkeypatch, path)
    store = b._runtime._provenance_store
    store.get_session = AsyncMock(wraps=store.get_session)
    try:
        await interface._run_agent("third", 10)
        assert texts(llm.requests[0].messages) == ["first", "answer-1", "second", "answer-2", "third"]
        await interface._run_agent("fourth", 10)
        assert texts(llm.requests[1].messages) == ["first", "answer-1", "second", "answer-2", "third", "answer-1", "fourth"]
        # Archive continuation may read on facade replacement; reconstruction is once.
        restores = [call for call in store.get_session.await_args_list if not call.kwargs]
        assert len(restores) == 1
        assert restores[0].args == ("discord-10",)
        await interface._run_agent("new-channel", 20)
        assert texts(llm.requests[-1].messages) == ["new-channel"]
    finally:
        await b.aclose()


@pytest.mark.parametrize("native", [False, True], ids=["text", "native"])
@pytest.mark.parametrize("action", ["reset", "retry"], ids=["AC7-c2", "AC7-c3"])
@pytest.mark.parametrize("cold", [False, True], ids=["live", "cold"])
async def test_rewind_survives_restart(monkeypatch, tmp_path, native, action, cold):
    path = tmp_path / "provenance.db"
    a, interface, llm = await start(monkeypatch, path)
    try:
        await interface._run_agent("first", 10)
        await interface._run_agent("second", 10)
        if cold:
            await a.aclose()
            a, interface, llm = await start(monkeypatch, path)
        # A different channel is now the facade, with a different archive bound.
        await interface._run_agent("other", 20)
        llm.prefix = "rerun"
        await command(interface, action, native)
        entries = await a._runtime._provenance_store.get_session("discord-10", layer="session_rewind")
        assert len(entries) == 1
        assert entries[0].meta == {"keep_through_turn_seq": -1 if action == "reset" else 0,
                                   "through_turn_seq": 1}
        if action == "reset":
            assert interface._session_map.get_or_create(10).history == []
        else:
            # The rerun becomes turn 2, preserving the archive's monotonic sequence.
            archive = await a._runtime._provenance_store.get_session("discord-10", layer="turn_archive")
            assert sorted((e.meta["turn_seq"], e.meta["seq"]) for e in archive) == [
                (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    finally:
        await a.aclose()
    b, interface, llm = await start(monkeypatch, path)
    try:
        await interface._run_agent("after-restart", 10)
        rerun = "rerun-2" if cold else "rerun-4"
        expected = ["after-restart"] if action == "reset" else [
            "first", "answer-1", "second", rerun, "after-restart"]
        assert texts(llm.requests[0].messages) == expected
    finally:
        await b.aclose()


async def test_channel_switch_continues_unique_archive(monkeypatch, tmp_path):
    """AC7-c4: C → D → C retains C's history, numbering, and lineage."""
    agent, interface, llm = await start(monkeypatch, tmp_path / "provenance.db")
    try:
        await interface._run_agent("C-first", 10)
        await interface._run_agent("D-first", 20)
        await interface._run_agent("C-second", 10)
        assert texts(llm.requests[1].messages) == ["D-first"]
        assert texts(llm.requests[2].messages) == ["C-first", "answer-1", "C-second"]
        archive = await agent._runtime._provenance_store.get_session("discord-10", layer="turn_archive")
        archive.sort(key=lambda entry: (entry.meta["turn_seq"], entry.meta["seq"]))
        assert [(e.meta["turn_seq"], e.meta["seq"]) for e in archive] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert archive[2].sources == (archive[1].id,)
    finally:
        await agent.aclose()


@pytest.mark.parametrize("failure", ["absent", "raises"])
async def test_restore_notice_once_and_outside_history(monkeypatch, tmp_path, failure, caplog):
    agent, interface, llm = await start(monkeypatch, "" if failure == "absent" else tmp_path / "provenance.db")
    agent.config.agent.resume.restore_failed_notice = "custom restore notice"
    if failure == "raises":
        agent._runtime._provenance_store.get_session = AsyncMock(side_effect=OSError("unreadable"))
    try:
        await interface._run_agent("first", 10)
        assert interface._adapter.messages[1] == "custom restore notice\n\nanswer-1"
        assert "discord-10" in caplog.text
        await interface._run_agent("other", 20)
        await interface._run_agent("second", 10)
        assert interface._adapter.messages[3] == "answer-3"
        assert texts(llm.requests[-1].messages) == ["first", "answer-1", "second"]
        history = interface._session_map.get_or_create(10).history
        assert [b["text"] for m in history for b in m["content"]] == ["first", "answer-1", "second", "answer-3"]
    finally:
        await agent.aclose()


async def test_disabled_restore_does_not_read_rewinds(monkeypatch, tmp_path):
    path = tmp_path / "provenance.db"
    a, interface, _ = await start(monkeypatch, path)
    try:
        await interface._run_agent("old", 10)
    finally:
        await a.aclose()
    b, interface, llm = await start(monkeypatch, path, enabled=False)
    store = b._runtime._provenance_store
    store.get_session = AsyncMock(wraps=store.get_session)
    try:
        await interface._run_agent("fresh", 10)
        assert texts(llm.requests[0].messages) == ["fresh"]
        assert interface._adapter.messages[1] == "answer-1"
        assert all(call.kwargs == {"layer": "turn_archive"} for call in store.get_session.await_args_list)
    finally:
        await b.aclose()
