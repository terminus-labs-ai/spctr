"""Image scenario config loader + validation.

Loads an image_scenarios.yaml containing reusable fragments
(models/checkpoints, frames, contents, loras) and scenarios
composing them by reference. Validates at load time:
every scenario ref resolves; scenario.model supports its modality.
Fails fast with clear errors — never at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Fragment models
# ---------------------------------------------------------------------------

class ModelFragment(BaseModel):
    """Checkpoint/model definition."""
    file: str
    dialect: str = "natural"
    translate: bool = False
    quality: str = ""
    negative: str = ""
    modalities: list[str] = Field(default_factory=list)


class FrameFragment(BaseModel):
    """Camera framing definition."""
    tags: str = ""
    size: list[int] = Field(default_factory=lambda: [1024, 1024])


class ContentFragment(BaseModel):
    """Content rating + tags."""
    tags: str = ""
    level: str = "sfw"


class LoraFragment(BaseModel):
    """LoRA definition."""
    file: str
    strength: float = 1.0
    clip_strength: float | None = None
    trigger: str = ""


class ModalityFragment(BaseModel):
    """Modality definition (txt2img, img2img, etc.)."""
    template: str = "txt2img.json"
    inputs: list[str] = Field(default_factory=list)
    output: str = "image"


# ---------------------------------------------------------------------------
# Scenario model
# ---------------------------------------------------------------------------

class Scenario(BaseModel):
    """A scenario composing fragments by reference.

    References are validated at load time against the fragment registry.
    """
    modality: str
    model: str
    frame: str
    content: str
    loras: list[str] = Field(default_factory=list)
    extra: str = ""
    translate_hint: str = ""


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class ImageScenariosConfig(BaseModel):
    """Top-level image scenarios configuration.

    Fragments are defined once and referenced by scenarios.
    All references are validated at construction time.
    """
    models: dict[str, ModelFragment] = Field(default_factory=dict)
    frames: dict[str, FrameFragment] = Field(default_factory=dict)
    contents: dict[str, ContentFragment] = Field(default_factory=dict)
    loras: dict[str, LoraFragment] = Field(default_factory=dict)
    modalities: dict[str, ModalityFragment] = Field(default_factory=dict)
    scenarios: dict[str, Scenario] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resolved scenario (post-validation, ready for use)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedScenario:
    """Fully resolved scenario with all fragments dereferenced.

    Produced by ImageScenarioRegistry after load-time validation.
    """
    name: str
    modality: ModalityFragment
    model: ModelFragment
    frame: FrameFragment
    content: ContentFragment
    loras: list[LoraFragment] = field(default_factory=list)
    extra: str = ""
    translate_hint: str = ""


# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------

class ScenarioConfigError(Exception):
    """Raised when image_scenarios.yaml fails validation."""


# ---------------------------------------------------------------------------
# Registry — loads, validates, resolves
# ---------------------------------------------------------------------------

class ImageScenarioRegistry:
    """Loads and validates image scenario configuration.

    Validates at construction:
    - Every scenario reference resolves to an existing fragment
    - Each scenario's model supports the scenario's modality
    - Required fragment sections are non-empty

    Usage:
        registry = ImageScenarioRegistry(path_to_yaml)
        scenario = registry.get("selfie")  # ResolvedScenario
        names = registry.scenario_names()  # list[str]
    """

    def __init__(self, config_path: str | Path) -> None:
        raw = self._load_yaml(config_path)
        self._config = self._parse_and_validate(raw)
        self._resolved: dict[str, ResolvedScenario] = self._resolve_all()

    @staticmethod
    def _load_yaml(config_path: str | Path) -> dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise ScenarioConfigError(f"Scenario config not found: {path}")

        text = path.read_text()
        raw = yaml.safe_load(text)
        if raw is None:
            raise ScenarioConfigError(
                f"Scenario config is empty: {path}"
            )
        if not isinstance(raw, dict):
            raise ScenarioConfigError(
                f"Scenario config must be a YAML mapping, got {type(raw).__name__}: {path}"
            )
        return raw

    @staticmethod
    def _parse_and_validate(raw: dict[str, Any]) -> ImageScenariosConfig:
        try:
            return ImageScenariosConfig(**raw)
        except ValidationError as exc:
            errors = "; ".join(str(e) for e in exc.errors())
            raise ScenarioConfigError(f"Invalid scenario config: {errors}") from exc

    def _resolve_all(self) -> dict[str, ResolvedScenario]:
        resolved: dict[str, ResolvedScenario] = {}
        errors: list[str] = []

        for name, scenario in self._config.scenarios.items():
            try:
                resolved[name] = self._resolve_scenario(name, scenario)
            except ScenarioConfigError as exc:
                errors.append(f"  {name}: {exc}")

        if errors:
            raise ScenarioConfigError(
                "Scenario validation failed:\n" + "\n".join(errors)
            )

        return resolved

    def _resolve_scenario(
        self, name: str, scenario: Scenario
    ) -> ResolvedScenario:
        cfg = self._config

        # Validate modality ref
        if scenario.modality not in cfg.modalities:
            raise ScenarioConfigError(
                f"modality '{scenario.modality}' not found in modalities. "
                f"Available: {sorted(cfg.modalities.keys())}"
            )

        # Validate model ref
        if scenario.model not in cfg.models:
            raise ScenarioConfigError(
                f"model '{scenario.model}' not found in models. "
                f"Available: {sorted(cfg.models.keys())}"
            )

        # Validate frame ref
        if scenario.frame not in cfg.frames:
            raise ScenarioConfigError(
                f"frame '{scenario.frame}' not found in frames. "
                f"Available: {sorted(cfg.frames.keys())}"
            )

        # Validate content ref
        if scenario.content not in cfg.contents:
            raise ScenarioConfigError(
                f"content '{scenario.content}' not found in contents. "
                f"Available: {sorted(cfg.contents.keys())}"
            )

        # Validate model supports modality
        model = cfg.models[scenario.model]
        if scenario.modality not in model.modalities:
            raise ScenarioConfigError(
                f"model '{scenario.model}' does not support modality "
                f"'{scenario.modality}'. Model supports: {model.modalities}"
            )

        # Validate LoRA refs
        loras: list[LoraFragment] = []
        for lora_name in scenario.loras:
            if lora_name not in cfg.loras:
                raise ScenarioConfigError(
                    f"lora '{lora_name}' not found in loras. "
                    f"Available: {sorted(cfg.loras.keys())}"
                )
            loras.append(cfg.loras[lora_name])

        return ResolvedScenario(
            name=name,
            modality=cfg.modalities[scenario.modality],
            model=model,
            frame=cfg.frames[scenario.frame],
            content=cfg.contents[scenario.content],
            loras=loras,
            extra=scenario.extra,
            translate_hint=scenario.translate_hint,
        )

    # -- Public API --

    def get(self, name: str) -> ResolvedScenario:
        """Get a resolved scenario by name.

        Raises:
            KeyError: If scenario name doesn't exist.
        """
        try:
            return self._resolved[name]
        except KeyError:
            available = sorted(self._resolved.keys())
            raise KeyError(
                f"Unknown scenario '{name}'. Available: {available}"
            ) from None

    def scenario_names(self) -> list[str]:
        """Return sorted list of scenario names."""
        return sorted(self._resolved.keys())

    @property
    def config(self) -> ImageScenariosConfig:
        """Return the raw parsed config."""
        return self._config


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def load_image_scenarios(config_path: str | Path) -> ImageScenarioRegistry:
    """Load and validate an image_scenarios.yaml file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        An ImageScenarioRegistry with all scenarios resolved and validated.

    Raises:
        ScenarioConfigError: If the config is invalid or references don't resolve.
    """
    return ImageScenarioRegistry(config_path)
