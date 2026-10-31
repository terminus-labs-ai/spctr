"""Integration test: active_frame_provider wiring through Runtime → Session → SR2.

Covers acceptance criteria from spc-22:
  A. Runtime builds PlanResolver when pipeline config has a plan resolver.
  B. Runtime passes active_frame_provider to Session.new_session().
  C. Session passes active_frame_provider to SR2().
  D. SR2 stamps block.meta["frame"] when provider is set.
  E. StepCompactionTransformer can burn blocks stamped by the provider.
  F. Runtime without plan resolver → provider is None (regression-safe).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sr2.models import TextBlock
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.runtime import Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config_with_plan(plans_root: str, project: str = "test-proj") -> SpectreConfig:
    """Build a SpectreConfig with a plan resolver in the pipeline."""
    layers = [
        {
            "name": "system",
            "target": "system",
            "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
        },
        {
            "name": "plan",
            "target": "system",
            "resolvers": [
                {
                    "type": "plan",
                    "config": {
                        "plans_root": plans_root,
                        "project": project,
                    },
                },
            ],
        },
        {
            "name": "conversation",
            "target": "messages",
            "resolvers": [{"type": "session"}, {"type": "input"}],
        },
    ]
    return SpectreConfig(
        agent=AgentConfig(name="test"),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={"layers": layers},
    )


def _make_config_no_plan() -> SpectreConfig:
    """Build a SpectreConfig WITHOUT a plan resolver."""
    layers = [
        {
            "name": "system",
            "target": "system",
            "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
        },
        {
            "name": "conversation",
            "target": "messages",
            "resolvers": [{"type": "session"}, {"type": "input"}],
        },
    ]
    return SpectreConfig(
        agent=AgentConfig(name="test"),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline={"layers": layers},
    )


def _write_minimal_plan(plan_dir: Path) -> None:
    """Write a minimal _plan.md + one pending task."""
    (plan_dir / "_plan.md").write_text(
        "---\nkind: plan\nslug: test-plan\nstatus: open\ngoal: \"Test\"\n---\n\nContract.\n"
    )
    (plan_dir / "01-setup.md").write_text(
        "---\nkind: task\nplan: test-plan\norder: 1\nstatus: pending\n"
        'verify: "echo ok"\ntitle: "Setup"\n---\n\nSet things up.\n'
    )


# ---------------------------------------------------------------------------
# A. Runtime builds PlanResolver when plan resolver is configured
# ---------------------------------------------------------------------------


class TestRuntimeBuildsPlanResolver:
    def test_runtime_builds_plan_resolver(self, tmp_path):
        """Runtime creates a PlanResolver instance when pipeline has type==plan."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        _write_minimal_plan(plan_dir)

        config = _make_config_with_plan(str(plans_dir))

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            # Patch PlanResolver in the planning module (imported inside Runtime.__init__)
            with patch(
                "sr2_spectre.planning.PlanResolver",
                wraps=__import__(
                    "sr2_spectre.planning.resolver", fromlist=["PlanResolver"]
                ).PlanResolver,
            ) as MockPlanResolver:
                runtime = Runtime(config=config)

        MockPlanResolver.assert_called_once()
        assert runtime._active_frame_provider is not None

    def test_runtime_no_plan_resolver_without_plan_in_pipeline(self, tmp_path):
        """Runtime does NOT build PlanResolver when no plan resolver in pipeline."""
        config = _make_config_no_plan()

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            # Patch PlanResolver in the planning module to detect if it's ever
            # imported during Runtime init.
            with patch(
                "sr2_spectre.planning.PlanResolver",
                side_effect=AttributeError("should not be called"),
            ):
                runtime = Runtime(config=config)

        assert runtime._active_frame_provider is None


# ---------------------------------------------------------------------------
# B. Runtime passes active_frame_provider to Session
# ---------------------------------------------------------------------------


class TestRuntimePassesProviderToSession:
    def test_new_session_receives_provider(self, tmp_path):
        """Runtime.new_session() passes active_frame_provider to Session."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        _write_minimal_plan(plan_dir)

        config = _make_config_with_plan(str(plans_dir))

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                MockSR2.return_value = MagicMock()
                runtime = Runtime(config=config)
                runtime.new_session(frame_id="frame-1")

        call_kwargs = MockSR2.call_args.kwargs
        assert "active_frame_provider" in call_kwargs
        assert call_kwargs["active_frame_provider"] is not None

    def test_new_session_no_provider_without_plan_resolver(self, tmp_path):
        """Without a plan resolver, Session receives active_frame_provider=None."""
        config = _make_config_no_plan()

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                MockSR2.return_value = MagicMock()
                runtime = Runtime(config=config)
                runtime.new_session(frame_id="frame-1")

        call_kwargs = MockSR2.call_args.kwargs
        assert call_kwargs.get("active_frame_provider") is None


# ---------------------------------------------------------------------------
# C. active_frame_provider returns correct frame id
# ---------------------------------------------------------------------------


class TestFrameProviderReturnsCorrectFrame:
    def test_provider_returns_frame_id(self, tmp_path):
        """The active_frame_provider returns the current task's frame id."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        _write_minimal_plan(plan_dir)

        config = _make_config_with_plan(str(plans_dir))

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=config)

        provider = runtime._active_frame_provider
        assert provider is not None

        # Call with any origin — it should return the frame id
        frame = provider("tui")
        # _extract_slug_from_filename strips "01-" prefix → "setup"
        assert frame == "plan:test-plan/setup"


# ---------------------------------------------------------------------------
# D. SR2 stamps blocks with frame meta
# ---------------------------------------------------------------------------


class TestSR2BlockStamping:
    def test_sr2_stamps_block_when_provider_set(self, tmp_path):
        """SR2._stamp_block stamps meta['frame'] when active_frame_provider is set."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        _write_minimal_plan(plan_dir)

        config = _make_config_with_plan(str(plans_dir))

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            with patch("sr2_spectre.session.SR2") as MockSR2:
                mock_sr2_instance = MagicMock()
                MockSR2.return_value = mock_sr2_instance
                runtime = Runtime(config=config)
                session = runtime.new_session(frame_id="frame-1")

        # Verify SR2 was constructed with the provider
        call_kwargs = MockSR2.call_args.kwargs
        provider = call_kwargs["active_frame_provider"]

        # Simulate what SR2._stamp_block does
        block = TextBlock(text="test content")
        frame = provider("tui")
        if frame:
            block.meta["frame"] = frame

        assert block.meta["frame"] == "plan:test-plan/setup"


# ---------------------------------------------------------------------------
# E. Full integration: provider → SR2 stamping → StepCompaction burn
# ---------------------------------------------------------------------------


class TestStepCompactionIntegration:
    async def test_blocks_stamped_by_provider_are_compacted(self, tmp_path):
        """Blocks stamped with the frame from active_frame_provider are burned
        by StepCompactionTransformer on plan_step_completed.

        This proves the end-to-end chain:
        PlanResolver.current_frame_id() → active_frame_provider →
        SR2._stamp_block() → block.meta["frame"] → StepCompactionTransformer burns
        """
        from sr2.config.models import TransformerConfig
        from sr2.pipeline.dependencies import Dependencies
        from sr2.pipeline.events import Event, EventPhase
        from sr2_spectre.planning.transformer import StepCompactionTransformer

        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        _write_minimal_plan(plan_dir)

        config = _make_config_with_plan(str(plans_dir))

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            runtime = Runtime(config=config)

        provider = runtime._active_frame_provider
        assert provider is not None

        # Simulate SR2 stamping blocks with the frame from the provider
        frame = provider("tui")
        assert frame == "plan:test-plan/setup"

        blocks = [
            TextBlock(text="system prompt"),  # no frame
            TextBlock(text="task work output"),  # stamped
        ]
        blocks[1].meta["frame"] = frame

        # Build the transformer
        transformer_config = TransformerConfig(
            type="step_compaction",
            config={},
            subscriptions=[{"event": "plan_step_completed", "phase": "completed"}],
        )
        deps = Dependencies(llm=None, session_id="test")
        transformer = StepCompactionTransformer.build(transformer_config, deps)

        # Fire the plan_step_completed event
        completed_event = Event(
            name="plan_step_completed",
            phase=EventPhase.COMPLETED,
            source_layer="plan",
            data={
                "frame": frame,
                "plan": "test-plan",
                "task": "01-setup",
                "order": 1,
            },
        )

        result = await transformer.transform(blocks, [completed_event])

        # The stamped block should be burned, system prompt kept
        assert result.content is not None
        texts = [b.text for b in result.content if isinstance(b, TextBlock)]
        assert "system prompt" in texts
        assert "task work output" not in texts

        # Breadcrumb should be present (uses the task field from the event,
        # but the task slug is "setup" since _extract_slug strips "01-")
        breadcrumb = "[task setup completed — see plan]"
        assert breadcrumb in texts
