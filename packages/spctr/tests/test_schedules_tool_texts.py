"""spc-1097: every schedule-tool string is a YAML parameter.

The YAML-parameter mandate (packages/spctr/AGENTS.md) covers the tool
description, the input-schema descriptions and every result/error string
the ``schedule`` tool shows a user or sends a model. These tests pin two
halves:

1. the defaults are the spec strings from agent-reminders.md verbatim —
   pinned as literals here, NOT read from the config model, so drift in
   ``ScheduleTextsConfig`` defaults fails;
2. overrides flow: constructor ``texts`` change the advertised schema and
   every rendered string, and a Runtime registers the overridden schema.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from spctr.config import (
    AgentConfig,
    ModelConfig,
    ScheduleTextsConfig,
    SchedulesConfig,
    SpctrConfig,
)
from spctr.reminders.tool import ChannelMatch
from spctr.schedules import ScheduleStore
from spctr.tools.schedule_tool import (
    AGENT_INSTRUCTION_MAX,
    COMMAND_MAX,
    ScheduleOrigin,
    ScheduleTool,
    ScheduleTurnScope,
    schedule_turn_scope,
)

UTC = timezone.utc
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)  # Monday 07:00 America/Phoenix


def _discord_origin() -> ScheduleOrigin:
    return ScheduleOrigin("miranda", "discord", "42", "<@42>", "100", "general")


def _zulip_origin(default_topic: str = "schedules") -> ScheduleOrigin:
    return ScheduleOrigin(
        agent="miranda",
        platform="zulip",
        requester_id="8",
        requester_mention="@8",
        channel_id="42",
        channel_name="dev-notes",
        default_topic=default_topic,
    )


def _make_tool(
    tmp_path,
    texts: ScheduleTextsConfig | None = None,
    resolver: Any = None,
    engine: Any = None,
) -> ScheduleTool:
    store = ScheduleStore(tmp_path / "schedules", agent="miranda")
    tool = ScheduleTool(
        store=store,
        timezone="America/Phoenix",
        agent_name="miranda",
        origin_getter=lambda: None,
        clock=lambda: NOW,
        engine=engine,
        texts=texts,
    )
    if resolver is not None:
        tool.set_channel_resolver(resolver)
    return tool


def _open_scope(origin: ScheduleOrigin):
    """Seat a turn scope; pass the token to schedule_turn_scope.reset()."""
    return schedule_turn_scope.set(ScheduleTurnScope(origin=origin))


async def _create(tool: ScheduleTool, **kw: Any) -> str:
    args = {"action": "create", "cron": "0 9 * * 1", "type": "command", "command": "echo x"}
    args.update(kw)
    return await tool(**args)


# ---------------------------------------------------------------------------
# 1. Defaults are the spec strings, verbatim
# ---------------------------------------------------------------------------


class TestDefaultsAreSpecStrings:
    async def test_tool_description(self, tmp_path):
        tool = _make_tool(tmp_path)
        assert tool.description == (
            "Create, cancel, or manually run a recurring scheduled job "
            "(command or agent payload) that fires on a cron schedule."
        )

    async def test_schema_property_descriptions_and_structure(self, tmp_path):
        tool = _make_tool(tmp_path)
        props = tool.input_schema["properties"]
        assert props["cron"]["description"] == "5-field cron, e.g. '0 7 * * *' or '0 9 * * 1'"
        assert (
            props["agent"]["description"]
            == "Target agent for an agent payload; omit for the creating agent"
        )
        assert (
            props["channel"]["description"]
            == "Same-platform channel name; omit for this channel"
        )
        assert props["topic"]["description"] == "Zulip post topic; omit for the default"
        assert props["label"]["description"] == "Short name shown in posts and list"
        # Structure is not text: enums and required stay fixed (FR24/FR31).
        assert props["action"]["enum"] == ["create", "list", "cancel", "run_now"]
        assert props["type"]["enum"] == ["command", "agent"]
        assert tool.input_schema["required"] == ["action"]

    async def test_created_result_form(self, tmp_path):
        tool = _make_tool(tmp_path)
        token = _open_scope(_discord_origin())
        try:
            result = await _create(tool)
        finally:
            schedule_turn_scope.reset(token)
        assert result == "Schedule s1 set: Mon 09:00 MST in #general (command: echo x)"

    async def test_zulip_destination_with_topic(self, tmp_path):
        tool = _make_tool(tmp_path)
        token = _open_scope(_zulip_origin())
        try:
            result = await _create(tool)
            listing = await tool(action="list")
        finally:
            schedule_turn_scope.reset(token)
        assert result == (
            "Schedule s1 set: Mon 09:00 MST in #dev-notes > schedules "
            "(command: echo x)"
        )
        # The bare tool has no namer seam: the reloaded file shows the
        # stored destination ID (spc-996/AC47-r4), next Monday is the 26th.
        assert listing == (
            "s1 Mon 2026-10-26 09:00 MST cron=0 9 * * 1 "
            "#42 > schedules command: echo x [never run]"
        )

    async def test_empty_list_and_cancel_forms(self, tmp_path):
        tool = _make_tool(tmp_path)
        token = _open_scope(_discord_origin())
        try:
            assert await tool(action="list") == "No schedules."
            await _create(tool)
            assert await tool(action="cancel", id="s1") == "Cancelled schedule s1"
        finally:
            schedule_turn_scope.reset(token)

    async def test_executed_form(self, tmp_path):
        class Engine:
            async def execute_now(self, row):
                pass

        tool = _make_tool(tmp_path, engine=Engine())
        token = _open_scope(_discord_origin())
        try:
            await _create(tool)
            assert await tool(action="run_now", id="s1") == (
                "Schedule s1 executed. Next run remains as scheduled."
            )
        finally:
            schedule_turn_scope.reset(token)

    async def test_error_defaults_verbatim(self, tmp_path):
        async def none_resolver(platform: str, name: str):
            return []

        async def two(platform: str, name: str):
            return [ChannelMatch(1, "a"), ChannelMatch(2, "b")]

        empty = _make_tool(tmp_path)
        token = _open_scope(_discord_origin())
        try:
            assert await empty(action="bogus") == "Error: unknown action 'bogus'"
            assert await _create(empty, cron="") == (
                "Error: 'cron' is required and must not be empty"
            )
            assert (
                await _create(empty, cron="not a cron")
            ).startswith("Error: invalid cron expression: ")
            assert await _create(empty, type=None) == (
                "Error: 'type' is required (command or agent)"
            )
            assert await _create(empty, type="script") == (
                "Error: unknown payload type 'script'"
            )
            assert await _create(empty, type="command", command="  ") == (
                "Error: 'command' is required for a command payload"
            )
            assert await _create(empty, type="command", command="x" * 4001) == (
                "Error: 'command' exceeds 4000 characters"
            )
            assert await _create(empty, type="agent", instruction="") == (
                "Error: 'instruction' is required for an agent payload"
            )
            assert await _create(empty, type="agent", instruction="y" * 4001) == (
                "Error: 'instruction' exceeds 4000 characters"
            )
            assert await empty(action="cancel", id=None) == (
                "Error: 'id' is required for cancel"
            )
            assert await empty(action="run_now", id=None) == (
                "Error: 'id' is required for run_now"
            )
            assert await empty(action="cancel", id="s9") == (
                "Error: schedule 's9' not found"
            )
            assert await empty(action="run_now", id="s9") == (
                "Error: schedule 's9' not found"
            )
            await _create(empty)
            assert await empty(action="run_now", id="s1") == (
                "Error: schedule engine not available"
            )

            # Channel-resolution errors need a registered resolver (or its
            # absence) — separate tools keep the store clean.
            t_res = _make_tool(tmp_path / "a", resolver=none_resolver)
            assert await _create(t_res, channel="ghost") == (
                "Error: channel 'ghost' not found on this platform"
            )
            t_two = _make_tool(tmp_path / "b", resolver=two)
            assert await _create(t_two, channel="dup") == (
                "Error: channel 'dup' is ambiguous on this platform"
            )
            assert await _create(t_two, channel="zulip#dup") == (
                "Error: Cross-platform schedules are not supported"
            )
            assert await _create(empty, channel="whatever") == (
                "Error: named channels are not available in this build"
            )
        finally:
            schedule_turn_scope.reset(token)

    async def test_no_interface_error_default(self, tmp_path):
        tool = _make_tool(tmp_path)  # no scope, origin_getter -> None
        assert await tool(action="list") == (
            "Error: Schedules need the Discord interface"
        )

# ---------------------------------------------------------------------------
# 2. Overrides flow through schema and every render site
# ---------------------------------------------------------------------------


class TestOverrides:
    async def test_advertised_schema_uses_overrides(self, tmp_path):
        texts = ScheduleTextsConfig(
            tool_description="Job scheduler.",
            cron_description="cron!",
            agent_description="who runs it",
            channel_description="where",
            topic_description="topic?",
            label_description="name",
        )
        tool = _make_tool(tmp_path, texts=texts)
        assert tool.description == "Job scheduler."
        props = tool.input_schema["properties"]
        assert props["cron"]["description"] == "cron!"
        assert props["agent"]["description"] == "who runs it"
        assert props["channel"]["description"] == "where"
        assert props["topic"]["description"] == "topic?"
        assert props["label"]["description"] == "name"
        # Structure untouched by text overrides.
        assert props["action"]["enum"] == ["create", "list", "cancel", "run_now"]
        assert tool.input_schema["required"] == ["action"]

    async def test_created_and_list_templates_override(self, tmp_path):
        texts = ScheduleTextsConfig(
            created_format="SET {id}@{next}->{destination}:{payload_type}/{summary}",
            destination_format="<{channel}>",
            destination_topic_format="<{channel}|{topic}>",
            list_row_format="{id}|{next}|{cron}|{destination}|{payload_type}|{summary}|{last_status}",
            never_run_label="-",
            empty_list_message="EMPTY",
        )
        tool = _make_tool(tmp_path, texts=texts)
        token = _open_scope(_discord_origin())
        try:
            assert await _create(tool) == (
                "SET s1@Mon 09:00 MST-><general>:command/echo x"
            )
            ztoken = _open_scope(_zulip_origin())
            try:
                assert await _create(tool) == (
                    "SET s2@Mon 09:00 MST-><dev-notes|schedules>:command/echo x"
                )
            finally:
                schedule_turn_scope.reset(ztoken)
            rows = (await tool(action="list")).split("\n")
            assert rows[0] == (
                "s1|Mon 2026-10-26 09:00 MST|0 9 * * 1|<100>|command|echo x|-"
            )
            assert rows[1] == (
                "s2|Mon 2026-10-26 09:00 MST|0 9 * * 1|<42|schedules>"
                "|command|echo x|-"
            )
        finally:
            schedule_turn_scope.reset(token)

    async def test_empty_cancel_execute_messages_override(self, tmp_path):
        class Engine:
            async def execute_now(self, row):
                pass

        texts = ScheduleTextsConfig(
            empty_list_message="nothing scheduled",
            cancelled_message="gone {id}",
            executed_message="ran {id}",
        )
        tool = _make_tool(tmp_path, texts=texts, engine=Engine())
        token = _open_scope(_discord_origin())
        try:
            assert await tool(action="list") == "nothing scheduled"
            await _create(tool)
            assert await tool(action="run_now", id="s1") == "ran s1"
            assert await tool(action="cancel", id="s1") == "gone s1"
        finally:
            schedule_turn_scope.reset(token)

    async def test_error_template_placeholders(self, tmp_path):
        texts = ScheduleTextsConfig(
            need_interface_error="no interface",
            unknown_action_error="what is {action}?",
            invalid_cron_error="bad cron {reason}",
            unknown_payload_type_error="no type {type}",
            command_too_long_error="max {max}",
            instruction_too_long_error="max {max}",
            channel_not_found_error="no channel {channel}",
            channel_ambiguous_error="many {channel}",
            cross_platform_error="wrong platform",
            no_resolver_error="no resolver",
            resolver_unusable_error="unusable",
            schedule_not_found_error="gone '{id}'",
            already_cancelled_error="{id} twice",
            cancelled_schedule_error="{id} dead",
            id_required_cancel_error="need id for cancel",
            id_required_run_now_error="need id for run_now",
            cron_required_error="need cron",
            type_required_error="need type",
            command_required_error="need command",
            instruction_required_error="need instruction",
            engine_unavailable_error="no engine",
        )

        async def none_resolver(platform: str, name: str):
            return []

        async def two(platform: str, name: str):
            return [ChannelMatch(1, "a"), ChannelMatch(2, "b")]

        bare = _make_tool(tmp_path, texts=texts)
        assert await bare(action="list") == "no interface"

        token = _open_scope(_discord_origin())
        try:
            assert await bare(action="nope") == "what is nope?"
            assert await _create(bare, cron="") == "need cron"
            assert (await _create(bare, cron="x x x")).startswith("bad cron ")
            assert await _create(bare, type=None) == "need type"
            assert await _create(bare, type="zz") == "no type zz"
            assert await _create(bare, type="command", command="") == "need command"
            assert await _create(bare, type="command", command="x" * 4001) == f"max {COMMAND_MAX}"
            assert await _create(bare, type="agent", instruction="") == "need instruction"
            assert (
                await _create(bare, type="agent", instruction="y" * 4001)
                == f"max {AGENT_INSTRUCTION_MAX}"
            )
            assert await bare(action="cancel", id=None) == "need id for cancel"
            assert await bare(action="run_now", id=None) == "need id for run_now"
            assert await bare(action="cancel", id="s9") == "gone 's9'"
            assert await bare(action="run_now", id="s9") == "gone 's9'"
            await _create(bare)
            assert await bare(action="run_now", id="s1") == "no engine"

            t_none = _make_tool(tmp_path / "a", texts=texts, resolver=none_resolver)
            assert await _create(t_none, channel="ghost") == "no channel ghost"
            t_two = _make_tool(tmp_path / "b", texts=texts, resolver=two)
            assert await _create(t_two, channel="dup") == "many dup"
            assert await _create(t_two, channel="zulip#dup") == "wrong platform"
        finally:
            schedule_turn_scope.reset(token)

    async def test_unusable_resolver_override(self, tmp_path):
        async def bad_resolver(platform: str, name: str):
            return [{"not": "a match"}]

        texts = ScheduleTextsConfig(resolver_unusable_error="unusable")
        tool = _make_tool(tmp_path / "u", texts=texts, resolver=bad_resolver)
        token = _open_scope(_discord_origin())
        try:
            assert await _create(tool, channel="x") == "unusable"
        finally:
            schedule_turn_scope.reset(token)


# ---------------------------------------------------------------------------
# Wiring: the Runtime registers the configured schema (spc-1097)
# ---------------------------------------------------------------------------


def _pipeline_dict() -> dict:
    return {
        "layers": [
            {
                "name": "system",
                "target": "system",
                "resolvers": [{"type": "static", "config": {"text": "hi"}}],
            }
        ]
    }


class TestRuntimeRegistration:
    def test_registry_carries_configured_texts(self, tmp_path):
        from spctr.runtime import Runtime

        texts = ScheduleTextsConfig(tool_description="Registered description.")
        config = SpctrConfig(
            agent=AgentConfig(
                name="wire",
                schedules=SchedulesConfig(
                    enabled=True,
                    timezone="America/Phoenix",
                    store_path=str(tmp_path / "schedules"),
                    texts=texts,
                ),
            ),
            models={"default": ModelConfig(model="m", base_url="http://x:1/v1")},
            pipeline=_pipeline_dict(),
        )
        with patch("spctr.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=config)
        assert "schedule" in runtime.registry
        spec = runtime.registry._tools["schedule"]
        assert spec.description == "Registered description."
        assert (
            spec.input_schema["properties"]["topic"]["description"]
            == ScheduleTextsConfig().topic_description
        )
