"""Tests for auto-injection of complete_step when PlanResolver is configured.

When a pipeline layer contains a resolver with type=='plan', the Runtime
auto-registers CompleteStepTool with the resolver's plans_root — so the
agent doesn't need to declare it explicitly in tools[].

Covers:
  A. Auto-inject when plan resolver present
  B. No auto-inject when no plan resolver
  C. No duplicate when complete_step already registered explicitly
  D. plans_root passed through to CompleteStepTool
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.tools.builtins.complete_step import CompleteStepTool


def _config_with_plan_resolver(plans_root: str | None = None) -> SpectreConfig:
    """Build a config with a plan resolver in one of the pipeline layers."""
    plan_resolver_config = {"project": "test-project"}
    if plans_root:
        plan_resolver_config["plans_root"] = plans_root

    return SpectreConfig(
        agent=AgentConfig(name="test", tools=[]),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
                },
                {
                    "name": "plan",
                    "target": "system",
                    "resolvers": [
                        {"type": "plan", "config": plan_resolver_config}
                    ],
                },
                {
                    "name": "tools",
                    "target": "tools",
                    "resolvers": [],
                    "tool_providers": [{"type": "spectre_tools"}],
                },
                {
                    "name": "conversation",
                    "target": "messages",
                    "resolvers": [{"type": "session"}, {"type": "input"}],
                },
            ]
        },
    )


def _config_without_plan_resolver() -> SpectreConfig:
    """Build a config with no plan resolver in any layer."""
    return SpectreConfig(
        agent=AgentConfig(name="test", tools=[]),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
                },
                {
                    "name": "tools",
                    "target": "tools",
                    "resolvers": [],
                    "tool_providers": [{"type": "spectre_tools"}],
                },
                {
                    "name": "conversation",
                    "target": "messages",
                    "resolvers": [{"type": "session"}, {"type": "input"}],
                },
            ]
        },
    )


# ---------------------------------------------------------------------------
# A. Auto-inject when plan resolver present
# ---------------------------------------------------------------------------

class TestAutoInjectWithPlanResolver:
    def test_complete_step_registered_when_plan_resolver_present(self):
        """When a plan resolver exists, complete_step is auto-registered."""
        from sr2_spectre.runtime import Runtime

        cfg = _config_with_plan_resolver()

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "complete_step" in runtime.registry

    def test_complete_step_not_registered_without_plan_resolver(self):
        """When no plan resolver exists, complete_step is NOT auto-registered."""
        from sr2_spectre.runtime import Runtime

        cfg = _config_without_plan_resolver()

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "complete_step" not in runtime.registry

    def test_complete_step_uses_resolver_plans_root(self, tmp_path: Path):
        """The auto-registered CompleteStepTool receives the plans_root from the resolver config."""
        from sr2_spectre.runtime import Runtime

        custom_root = str(tmp_path / "my-plans")
        Path(custom_root).mkdir()
        cfg = _config_with_plan_resolver(plans_root=custom_root)

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        # Verify the registered tool's instance has the correct plans_root
        spec = runtime.registry._tools.get("complete_step")
        assert spec is not None
        # The fn is the __call__ method of the CompleteStepTool instance
        tool_instance = spec.fn.__self__
        assert isinstance(tool_instance, CompleteStepTool)
        assert str(tool_instance._plans_root) == str(Path(custom_root).resolve())

    def test_complete_step_uses_default_plans_root_when_not_specified(self):
        """When plans_root is not in resolver config, CompleteStepTool uses its own default (~/.sr2/plans)."""
        from sr2_spectre.runtime import Runtime

        cfg = _config_with_plan_resolver()  # no plans_root specified

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        spec = runtime.registry._tools.get("complete_step")
        assert spec is not None
        tool_instance = spec.fn.__self__
        assert isinstance(tool_instance, CompleteStepTool)
        # Default plans_root is ~/.sr2/plans
        expected = Path.home() / ".sr2" / "plans"
        assert tool_instance._plans_root == expected.resolve()


# ---------------------------------------------------------------------------
# B. No duplicate when already registered
# ---------------------------------------------------------------------------

class TestNoDuplicateRegistration:
    def test_no_duplicate_when_complete_step_explicit(self):
        """If complete_step is explicitly in tools[], don't register twice."""
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import ToolConfig

        cfg = _config_with_plan_resolver()
        # Explicitly add complete_step to the tools list
        cfg.agent.tools = [
            ToolConfig(
                name="complete_step",
                class_path="sr2_spectre.tools.builtins.complete_step.CompleteStepTool",
                config={},
            )
        ]

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        # Should still have exactly one complete_step
        assert "complete_step" in runtime.registry
        # Verify it's only registered once (one entry in _tools dict)
        assert sum(1 for name in runtime.registry.list_names() if name == "complete_step") == 1

    def test_explicit_complete_step_without_plan_resolver(self):
        """complete_step registered explicitly still works without a plan resolver."""
        from sr2_spectre.runtime import Runtime
        from sr2_spectre.config import ToolConfig

        cfg = _config_without_plan_resolver()
        cfg.agent.tools = [
            ToolConfig(
                name="complete_step",
                class_path="sr2_spectre.tools.builtins.complete_step.CompleteStepTool",
                config={},
            )
        ]

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "complete_step" in runtime.registry


# ---------------------------------------------------------------------------
# C. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_plan_resolver_in_non_first_layer(self):
        """Auto-injection works when plan resolver is in any layer, not just the first."""
        from sr2_spectre.runtime import Runtime

        cfg = SpectreConfig(
            agent=AgentConfig(name="test", tools=[]),
            models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
            pipeline={
                "layers": [
                    {
                        "name": "system",
                        "target": "system",
                        "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
                    },
                    {
                        "name": "tools",
                        "target": "tools",
                        "resolvers": [],
                        "tool_providers": [{"type": "spectre_tools"}],
                    },
                    {
                        "name": "plan-layer",
                        "target": "system",
                        "resolvers": [
                            {"type": "plan", "config": {"project": "test-project"}}
                        ],
                    },
                    {
                        "name": "conversation",
                        "target": "messages",
                        "resolvers": [{"type": "session"}, {"type": "input"}],
                    },
                ]
            },
        )

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "complete_step" in runtime.registry

    def test_multiple_plan_resolvers_uses_first(self):
        """If multiple plan resolvers exist (unusual), use the first one's plans_root."""
        from sr2_spectre.runtime import Runtime

        cfg = SpectreConfig(
            agent=AgentConfig(name="test", tools=[]),
            models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
            pipeline={
                "layers": [
                    {
                        "name": "plan-a",
                        "target": "system",
                        "resolvers": [
                            {"type": "plan", "config": {"project": "proj-a", "plans_root": "/tmp/plans-a"}}
                        ],
                    },
                    {
                        "name": "plan-b",
                        "target": "system",
                        "resolvers": [
                            {"type": "plan", "config": {"project": "proj-b", "plans_root": "/tmp/plans-b"}}
                        ],
                    },
                    {
                        "name": "tools",
                        "target": "tools",
                        "resolvers": [],
                        "tool_providers": [{"type": "spectre_tools"}],
                    },
                    {
                        "name": "conversation",
                        "target": "messages",
                        "resolvers": [{"type": "session"}, {"type": "input"}],
                    },
                ]
            },
        )

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=cfg)

        assert "complete_step" in runtime.registry
        # Should use the first plan resolver found
        spec = runtime.registry._tools["complete_step"]
        tool_instance = spec.fn.__self__
        assert str(tool_instance._plans_root) == "/tmp/plans-a"
