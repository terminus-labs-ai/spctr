"""J1: topic context compiled through the installed plugin and real Session/SR2."""
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from sr2.protocols.llm import StreamEvent
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.core import RunContext, RunMode
from sr2_spectre.session import Session
from sr2_spectre.tools.registry import ToolRegistry


def _topic(root: Path, folder: str, area: str, marker: str, mode="source",
           readme_body=None):
    directory = root / folder
    directory.mkdir(parents=True)
    (directory / "README.md").write_text(
        f"---\nkind: topic\nid: topic:{area}\n---\n"
        + (f"{readme_body}\n" if readme_body else ""), encoding="utf-8"
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


@pytest.mark.asyncio
async def test_readme_context_compiles_and_refreshes_through_real_session(tmp_path):
    """J1/AC1: README selection compiles, refreshes on edit and survives a move."""
    root = tmp_path / "topics"
    selected = _topic(root, "sr2/runtime-hub", "sr2-spectre", "INTENDED",
                      readme_body="INTENDED README")
    _topic(root, "family/sr2-spectre", "sr2-spectre", "VIEW", mode="view",
           readme_body="VIEW README")
    _topic(root, "characters/archive/sr2-spectre", "character-decoy", "DECOY",
           readme_body="DECOY README")
    _topic(root, "family/henrique (pai)", "henrique", "HENRIQUE",
           readme_body="HENRIQUE README")
    # Same discovery guarantee as the AGENTS journey: installed plugin entry point.
    assert len([
        ep for ep in entry_points(group="sr2.resolvers") if ep.name == "topic_area"
    ]) == 1
    config = SpectreConfig(
        agent=AgentConfig(name="readme-journey"),
        models={"default": ModelConfig(model="fixture", base_url="http://unused")},
        pipeline={"layers": [
            {"name": "area", "target": "system",
             "degradation_category": "plan_knowledge",
             "resolvers": [
                 {"type": "topic_area", "name": name,
                  "config": {"topics_root": str(root), "filename": filename,
                             "max_tokens": 4000}}
                 for name, filename in [
                     ("area-readme", "README.md"), ("area-now", "NOW.md")]
             ]},
            {"name": "conversation", "target": "messages",
             "resolvers": [{"type": "session"}, {"type": "input"}]},
        ]},
    )
    llm = _CaptureLLM()
    session = Session("readme-journey", config, llm, ToolRegistry())

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
        # Intended README and paired source NOW compile; decoys and views stay out.
        # Fails if README is not selectable, if AGENTS leaks instead, or if the
        # duplicate-basename/family-view topics win the identity gate.
        system = await turn("sr2-spectre")
        assert "INTENDED README" in system and "INTENDED NOW" in system
        assert "INTENDED AGENTS" not in system and "AGENTS" not in system
        assert all(marker not in system for marker in ("VIEW", "DECOY", "HENRIQUE"))

        # Live reread: both documents edited on disk, next turn shows new content
        # and no stale body (guards against caching or partial refresh).
        readme = selected / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8").replace(
            "INTENDED README", "UPDATED README"), encoding="utf-8")
        now = selected / "NOW.md"
        now.write_text(now.read_text(encoding="utf-8").replace(
            "INTENDED NOW", "UPDATED NOW"), encoding="utf-8")
        system = await turn("sr2-spectre")
        assert "UPDATED README" in system and "UPDATED NOW" in system
        assert "INTENDED" not in system
        assert all(marker not in system for marker in ("VIEW", "DECOY", "HENRIQUE"))

        # J1 move requirement: relocating the topic folder never changes its
        # identity; the next turn still resolves README/NOW at the new home.
        relocated = root / "family/moved-runtime-hub"
        selected.rename(relocated)
        assert not (root / "sr2/runtime-hub").exists()
        system = await turn("sr2-spectre")
        assert "UPDATED README" in system and "UPDATED NOW" in system
        assert all(marker not in system for marker in ("VIEW", "DECOY", "HENRIQUE"))

        # Selection is per-identity, not per-basename: another topic gets only its
        # own README/NOW; the moved topic's content must not leak here.
        system = await turn("henrique")
        assert "HENRIQUE README" in system and "HENRIQUE NOW" in system
        assert all(marker not in system for marker in (
            "INTENDED", "UPDATED", "VIEW", "DECOY"))

        # Interface area semantics unchanged: no area means no topic context.
        assert await turn("") == ""
    finally:
        session.close()
