"""sr2-148 — ``retrieval_profiles`` nests under ``continuity`` (spec AC).

The spec's "Policy configuration" example (autonomous-context-continuity.md)
puts ``retrieval_profiles:`` inside the ``continuity:`` block, while the
model historically declared it beside ``continuity`` on ``PipelineConfig``.
No test loaded the spec example, so the mismatch was invisible.

Case rows pinned here:
  - AC1: ``ContinuityConfig.retrieval_profiles`` exists with the profile
    mapping default ``{}``; ``PipelineConfig`` carries no such field.
  - AC2: the spec's example YAML block, copied verbatim into
    ``tests/fixtures/spec_retrieval_profiles.yaml``, loads and the
    ``coding`` profile resolves (default and first rule id); a config
    with a top-level ``retrieval_profiles`` key fails validation with a
    message naming ``continuity.retrieval_profiles`` (no silent
    acceptance of the legacy spelling).

The exception type is not pinned beyond pydantic's ``ValidationError``
umbrella (same convention as ``test_continuity_config.py``): the repo's
``ConfigError`` subclasses ``ValueError`` so pydantic wraps it, and both
land under that umbrella.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sr2.config.models import (
    ContinuityConfig,
    LayerConfig,
    PipelineConfig,
    RetrievalProfileConfig,
    ResolverConfig,
)

_SPEC_EXAMPLE = Path(__file__).resolve().parent.parent / "fixtures" / (
    "spec_retrieval_profiles.yaml"
)


def _pipeline(**kwargs) -> PipelineConfig:
    """Minimal valid PipelineConfig kwargs base for legacy-key probes."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[ResolverConfig(type="session")],
            )
        ],
        **kwargs,
    )


class TestPlacementOnContinuityConfig:
    """AC1 — the mapping lives on ContinuityConfig, not PipelineConfig."""

    def test_continuity_config_has_retrieval_profiles_default_empty(self):
        cfg = ContinuityConfig(recent_token_budget=12000)
        assert cfg.retrieval_profiles == {}

    def test_continuity_config_accepts_profile_mapping(self):
        cfg = ContinuityConfig(
            recent_token_budget=12000,
            retrieval_profiles={"coding": RetrievalProfileConfig()},
        )
        assert cfg.retrieval_profiles["coding"].default == (
            "deny_cross_agent_personal"
        )

    def test_pipeline_config_has_no_retrieval_profiles_field(self):
        assert "retrieval_profiles" not in PipelineConfig.model_fields, (
            "AC1: PipelineConfig must not declare retrieval_profiles; the "
            "mapping is nested under continuity"
        )
        cfg = _pipeline()
        assert not hasattr(cfg, "retrieval_profiles")


class TestSpecExampleLoads:
    """AC2 — the spec's example block resolves through the model."""

    def test_spec_example_coding_profile_resolves(self):
        document = yaml.safe_load(_SPEC_EXAMPLE.read_text())
        raw = document["pipeline"]
        # The fixture is the spec block verbatim (a continuity section); a
        # loadable pipeline additionally needs the required layers field.
        raw["layers"] = [
            {
                "name": "conversation",
                "target": "messages",
                "resolvers": [{"type": "session"}],
            }
        ]
        cfg = PipelineConfig.model_validate(raw)
        profile = cfg.continuity.retrieval_profiles["coding"]
        assert profile.default == "deny_cross_agent_personal"
        assert profile.rules[0].id == "coding-preferences"


class TestLegacyTopLevelRejected:
    """AC2 — top-level retrieval_profiles fails loudly, naming the new home."""

    def test_top_level_retrieval_profiles_rejected_with_populated_mapping(self):
        with pytest.raises(ValidationError) as excinfo:
            _pipeline(retrieval_profiles={"coding": {"default": "allow"}})
        message = str(excinfo.value)
        assert "continuity.retrieval_profiles" in message, (
            "the failure must name the new location so the operator can "
            f"relocate the key; got: {message}"
        )

    def test_top_level_retrieval_profiles_rejected_even_when_empty(self):
        with pytest.raises(ValidationError) as excinfo:
            _pipeline(retrieval_profiles={})
        assert "continuity.retrieval_profiles" in str(excinfo.value)

    def test_failure_message_is_the_repos_own_wording(self):
        """The message is exactly the repo's wording (style of
        ``_ValidatableConfigError`` users): a plain sentence naming the new
        location, not a pydantic-mangled guess."""
        with pytest.raises(ValidationError) as excinfo:
            _pipeline(retrieval_profiles={"coding": {}})
        assert "retrieval_profiles moved to continuity.retrieval_profiles" in (
            str(excinfo.value)
        )

    def test_nested_retrieval_profiles_still_loads(self):
        """Only the TOP-LEVEL key is the error; the spec spelling loads."""
        document = yaml.safe_load(_SPEC_EXAMPLE.read_text())
        document["pipeline"]["layers"] = [
            {
                "name": "conversation",
                "target": "messages",
                "resolvers": [{"type": "session"}],
            }
        ]
        cfg = PipelineConfig.model_validate(document["pipeline"])
        assert "coding" in cfg.continuity.retrieval_profiles
