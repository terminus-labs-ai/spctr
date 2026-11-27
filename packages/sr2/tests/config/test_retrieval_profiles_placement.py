"""spc-1113 — ``retrieval_profiles`` lives under ``continuity:`` (spec example)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sr2.config.models import ContinuityConfig, PipelineConfig

FIXTURE = Path(__file__).parent.parent / "fixtures" / "spec_retrieval_profiles.yaml"


def _load(data: dict) -> PipelineConfig:
  return PipelineConfig.model_validate({"layers": [], **data})


def test_spec_example_nested_under_continuity_loads():
  config = _load(yaml.safe_load(FIXTURE.read_text()))

  profile = config.continuity.retrieval_profiles["coding"]
  assert profile.default == "deny_cross_agent_personal"
  assert profile.rules[0].id == "coding-preferences"
  assert [r.id for r in profile.rules] == [
    "coding-preferences",
    "unrelated-personal",
    "current-project",
  ]


def test_continuity_retrieval_profiles_defaults_empty():
  assert ContinuityConfig(recent_token_budget=1).retrieval_profiles == {}


def test_pipeline_config_has_no_retrieval_profiles_field():
  assert "retrieval_profiles" not in PipelineConfig.model_fields


def test_top_level_retrieval_profiles_rejected():
  data = yaml.safe_load(FIXTURE.read_text())
  data["retrieval_profiles"] = data["continuity"].pop("retrieval_profiles")

  with pytest.raises(ValidationError) as excinfo:
    _load(data)
  assert "retrieval_profiles moved to continuity.retrieval_profiles" in str(excinfo.value)
