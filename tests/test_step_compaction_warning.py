"""Tests for the step_compaction startup warning (spc-28).

When a PlanResolver is configured but no step_compaction transformer is
declared in any pipeline layer, the Runtime emits a startup WARNING.

Covers three states:
  A. Plan resolver + step_compaction present → no warning
  B. Plan resolver present, no step_compaction → warning
  C. Neither plan resolver nor step_compaction → no warning
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.runtime import Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_with_plan_and_compaction() -> SpectreConfig:
    """Config with both a plan resolver AND a step_compaction transformer."""
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
                        {"type": "plan", "config": {"project": "test-project"}}
                    ],
                },
                {
                    "name": "compact",
                    "target": "messages",
                    "resolvers": [],
                    "transformers": [
                        {"type": "step_compaction", "config": {}}
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


def _config_with_plan_no_compaction() -> SpectreConfig:
    """Config with a plan resolver but NO step_compaction transformer."""
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


def _config_no_plan_no_compaction() -> SpectreConfig:
    """Config with neither plan resolver nor step_compaction transformer."""
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
                    "name": "conversation",
                    "target": "messages",
                    "resolvers": [{"type": "session"}, {"type": "input"}],
                },
            ]
        },
    )


# ---------------------------------------------------------------------------
# A. Both plan resolver + step_compaction → no warning
# ---------------------------------------------------------------------------

class TestBothConfigured:
    def test_no_warning_when_both_present(self):
        """When plan resolver AND step_compaction are both configured, no warning is emitted."""
        cfg = _config_with_plan_and_compaction()

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            with patch("sr2_spectre.runtime.logger") as mock_logger:
                runtime = Runtime(config=cfg)

        # Should NOT warn about step_compaction
        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "step_compaction" in str(call)
        ]
        assert len(warning_calls) == 0


# ---------------------------------------------------------------------------
# B. Plan resolver without step_compaction → warning
# ---------------------------------------------------------------------------

class TestPlanWithoutCompaction:
    def test_warning_when_plan_resolver_missing_compaction(self):
        """When a plan resolver exists but no step_compaction transformer, a WARNING is emitted."""
        cfg = _config_with_plan_no_compaction()

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            with patch("sr2_spectre.runtime.logger") as mock_logger:
                runtime = Runtime(config=cfg)

        # Should emit exactly one warning mentioning step_compaction
        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "step_compaction" in str(call)
        ]
        assert len(warning_calls) == 1
        # Verify the warning message mentions the unbounded context risk
        assert "unbounded" in warning_calls[0][0][0]

    def test_warning_mentions_step_compaction(self):
        """The warning message names step_compaction and PlanResolver."""
        cfg = _config_with_plan_no_compaction()

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            with patch("sr2_spectre.runtime.logger") as mock_logger:
                runtime = Runtime(config=cfg)

        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "step_compaction" in str(call)
        ]
        assert len(warning_calls) == 1
        message = warning_calls[0][0][0]
        assert "PlanResolver" in message
        assert "step_compaction" in message


# ---------------------------------------------------------------------------
# C. Neither plan resolver nor step_compaction → no warning
# ---------------------------------------------------------------------------

class TestNeitherConfigured:
    def test_no_warning_without_plan_resolver(self):
        """When there's no plan resolver, the absence of step_compaction is not warned about."""
        cfg = _config_no_plan_no_compaction()

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            with patch("sr2_spectre.runtime.logger") as mock_logger:
                runtime = Runtime(config=cfg)

        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "step_compaction" in str(call)
        ]
        assert len(warning_calls) == 0


# ---------------------------------------------------------------------------
# D. Static helper: _find_step_compaction_transformer
# ---------------------------------------------------------------------------

class TestFindStepCompactionTransformer:
    def test_finds_transformer_in_single_layer(self):
        """Returns True when a step_compaction transformer exists in any layer."""
        cfg = _config_with_plan_and_compaction()
        assert Runtime._find_step_compaction_transformer(cfg) is True

    def test_returns_false_when_absent(self):
        """Returns False when no step_compaction transformer exists."""
        cfg = _config_with_plan_no_compaction()
        assert Runtime._find_step_compaction_transformer(cfg) is False

    def test_returns_false_with_other_transformers(self):
        """Returns False when only non-step_compaction transformers exist."""
        cfg = SpectreConfig(
            agent=AgentConfig(name="test", tools=[]),
            models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
            pipeline={
                "layers": [
                    {
                        "name": "system",
                        "target": "system",
                        "resolvers": [{"type": "static", "config": {"text": "You are helpful."}}],
                        "transformers": [{"type": "other_transformer", "config": {}}],
                    },
                ]
            },
        )
        assert Runtime._find_step_compaction_transformer(cfg) is False

    def test_handles_none_transformers_list(self):
        """Handles layers where transformers is None (default)."""
        cfg = _config_no_plan_no_compaction()
        # Should not raise
        assert Runtime._find_step_compaction_transformer(cfg) is False

    def test_finds_transformer_in_last_layer(self):
        """Finds step_compaction even when it's in a later layer."""
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
                        "name": "conversation",
                        "target": "messages",
                        "resolvers": [{"type": "session"}, {"type": "input"}],
                    },
                    {
                        "name": "compact",
                        "target": "messages",
                        "resolvers": [],
                        "transformers": [{"type": "step_compaction", "config": {}}],
                    },
                ]
            },
        )
        assert Runtime._find_step_compaction_transformer(cfg) is True
