"""J1: topic context compiled through the installed plugin and real Session/SR2."""
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from sr2.protocols.llm import StreamEvent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.session import Session
from sr2_spectre.tools.registry import ToolRegistry


def _topic(root: Path, folder: str, area: str, marker: str, mode="source"):
    directory = root / folder
    directory.mkdir(parents=True)
    (directory / "README.md").write_text(
        f"---\nkind: topic\nid: topic:{area}\n---\n", encoding="utf-8"
    )
    (directory / "NOW.md").write_text(
        f"---\nkind: continuity\nfor: topic:{area}\nmode: {mode}\n---\n"
        f"{marker} NOW\n", encoding="utf-8"
    )
    (directory / "AGENTS.md").write_text(f"{marker} AGENTS\n", encoding="utf-8")
    return directory


class _CaptureLLM:
    """Only the external LLM boundary is replaced; retain compiled requests."""
    def __init__(self):
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield StreamEvent(type="text", text="fixture reply")
        yield StreamEvent(type="end")

    async def complete(self, request):
        raise AssertionError("Journey must use the streaming turn entry point")


@pytest.mark.asyncio
async def test_topic_context_compiles_and_refreshes_through_real_session(tmp_path):
    root = tmp_path / "topics"
    selected = _topic(root, "sr2/runtime-hub", "sr2-spectre", "INTENDED")
    _topic(root, "family/sr2-spectre", "sr2-spectre", "VIEW", mode="view")
    _topic(root, "characters/archive/sr2-spectre", "character-decoy", "DECOY")
    _topic(root, "family/henrique (pai)", "henrique", "HENRIQUE")
    # Packaging discovery must work without registering or constructing a resolver.
    assert len([
        ep for ep in entry_points(group="sr2.resolvers") if ep.name == "topic_area"
    ]) == 1
    config = SpectreConfig(
        agent=AgentConfig(name="topic-journey"),
        models={"default": ModelConfig(model="fixture", base_url="http://unused")},
        pipeline={"layers": [
            {"name": "area", "target": "system",
             "degradation_category": "plan_knowledge",
             "resolvers": [
                 {"type": "topic_area", "name": name,
                  "config": {"topics_root": str(root), "filename": filename,
                             "max_tokens": 4000}}
                 for name, filename in [
                     ("area-doc", "AGENTS.md"), ("area-now", "NOW.md")]
             ]},
            {"name": "conversation", "target": "messages",
             "resolvers": [{"type": "session"}, {"type": "input"}]},
        ]},
    )
    llm = _CaptureLLM()
    session = Session("topic-journey", config, llm, ToolRegistry())

    async def turn(area):
        session.set_run_context(RunContext(
            interface="discord", mode=RunMode.INTERACTIVE,
            source="fixture-channel", area=area,
        ))
        before = len(llm.requests)
        events = [event async for event in session.stream_message("compile context")]
        assert events
        assert len(llm.requests) == before + 1
        return "\n".join(block.text for block in llm.requests[-1].system or [])

    try:
        system = await turn("sr2-spectre")
        assert "INTENDED AGENTS" in system and "INTENDED NOW" in system
        assert all(marker not in system for marker in ("VIEW", "DECOY", "HENRIQUE"))

        (selected / "AGENTS.md").write_text("UPDATED AGENTS\n", encoding="utf-8")
        now = selected / "NOW.md"
        now.write_text(now.read_text(encoding="utf-8").replace(
            "INTENDED NOW", "UPDATED NOW"), encoding="utf-8")
        system = await turn("sr2-spectre")
        assert "UPDATED AGENTS" in system and "UPDATED NOW" in system
        assert all(marker not in system for marker in ("INTENDED", "VIEW", "DECOY"))

        system = await turn("henrique")
        assert "HENRIQUE AGENTS" in system and "HENRIQUE NOW" in system
        assert all(marker not in system for marker in (
            "INTENDED", "UPDATED", "VIEW", "DECOY"))

        assert await turn("") == ""
    finally:
        session.close()
