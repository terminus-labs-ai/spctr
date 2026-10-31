"""Tests for image scenario config loader + validation (spc-63).

Covers:
- Fragment parsing (models, frames, contents, loras, modalities)
- Scenario reference resolution
- Model-modality compatibility validation
- Fail-fast on bad config (missing refs, empty file, wrong type)
- ResolvedScenario data integrity
- Public API (get, scenario_names, config)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from sr2_spectre.tools.image_scenarios import (
    ContentFragment,
    FrameFragment,
    ImageScenarioRegistry,
    ImageScenariosConfig,
    LoraFragment,
    ModalityFragment,
    ModelFragment,
    ResolvedScenario,
    Scenario,
    ScenarioConfigError,
    load_image_scenarios,
)


# ---------------------------------------------------------------------------
# Fixtures — YAML configs written to temp files
# ---------------------------------------------------------------------------

VALID_CONFIG = """\
models:
  dreamshaper:
    file: "SDXL/dreamshaperXL_alpha2Xl10.safetensors"
    dialect: natural
    translate: false
    quality: "masterpiece, best quality, detailed"
    negative: "low quality, blurry, deformed"
    modalities: [txt2img]
  pony:
    file: "Pony/ponyDiffusionV6XL.safetensors"
    dialect: booru
    translate: true
    quality: "score_9, score_8_up, source_anime"
    negative: "score_4, score_5, worst quality"
    modalities: [txt2img]

frames:
  portrait:
    tags: "close-up, face focus"
    size: [832, 1216]
  full_body:
    tags: "full body, standing"
    size: [832, 1216]

contents:
  sfw:
    tags: "sfw"
    level: sfw
  nsfw:
    tags: "nsfw"
    level: nsfw

loras:
  vexa_face:
    file: "vexa_lora.safetensors"
    strength: 0.8
    trigger: "vexatoken"

modalities:
  txt2img:
    template: "txt2img.json"
    inputs: []
    output: image

scenarios:
  selfie:
    modality: txt2img
    model: dreamshaper
    frame: portrait
    content: sfw
    loras: [vexa_face]
    extra: "selfie angle, casual"
  boudoir:
    modality: txt2img
    model: pony
    frame: full_body
    content: nsfw
    loras: [vexa_face]
    extra: "dramatic lighting"
    translate_hint: "emphasize pose, outfit"
"""


@pytest.fixture
def valid_config_path(tmp_path: Path) -> Path:
    p = tmp_path / "image_scenarios.yaml"
    p.write_text(VALID_CONFIG)
    return p


@pytest.fixture
def registry(valid_config_path: Path) -> ImageScenarioRegistry:
    return ImageScenarioRegistry(valid_config_path)


# ---------------------------------------------------------------------------
# Fragment model tests
# ---------------------------------------------------------------------------

class TestModelFragment:
    def test_required_file(self):
        m = ModelFragment(file="test.safetensors")
        assert m.file == "test.safetensors"
        assert m.dialect == "natural"
        assert m.translate is False
        assert m.modalities == []

    def test_full_model(self):
        m = ModelFragment(
            file="pony.safetensors",
            dialect="booru",
            translate=True,
            quality="score_9",
            negative="score_4",
            modalities=["txt2img"],
        )
        assert m.dialect == "booru"
        assert m.translate is True
        assert "txt2img" in m.modalities


class TestFrameFragment:
    def test_defaults(self):
        f = FrameFragment()
        assert f.tags == ""
        assert f.size == [1024, 1024]

    def test_custom(self):
        f = FrameFragment(tags="close-up", size=[832, 1216])
        assert f.tags == "close-up"
        assert f.size == [832, 1216]


class TestContentFragment:
    def test_defaults(self):
        c = ContentFragment()
        assert c.tags == ""
        assert c.level == "sfw"

    def test_nsfw(self):
        c = ContentFragment(tags="nsfw", level="nsfw")
        assert c.level == "nsfw"


class TestLoraFragment:
    def test_required_file(self):
        l = LoraFragment(file="test_lora.safetensors")
        assert l.file == "test_lora.safetensors"
        assert l.strength == 1.0
        assert l.trigger == ""

    def test_full(self):
        l = LoraFragment(file="face.safetensors", strength=0.7, trigger="facetoken")
        assert l.strength == 0.7
        assert l.trigger == "facetoken"

    def test_clip_strength_defaults_to_none(self):
        """clip_strength is None by default (falls back to strength at build time)."""
        l = LoraFragment(file="test_lora.safetensors")
        assert l.clip_strength is None

    def test_clip_strength_set(self):
        """clip_strength can be set independently of strength."""
        l = LoraFragment(file="face.safetensors", strength=1.0, clip_strength=0.5)
        assert l.strength == 1.0
        assert l.clip_strength == 0.5


class TestModalityFragment:
    def test_defaults(self):
        m = ModalityFragment()
        assert m.template == "txt2img.json"
        assert m.inputs == []
        assert m.output == "image"


# ---------------------------------------------------------------------------
# Scenario model tests
# ---------------------------------------------------------------------------

class TestScenario:
    def test_minimal(self):
        s = Scenario(modality="txt2img", model="dreamshaper", frame="portrait", content="sfw")
        assert s.loras == []
        assert s.extra == ""
        assert s.translate_hint == ""

    def test_full(self):
        s = Scenario(
            modality="txt2img",
            model="pony",
            frame="full_body",
            content="nsfw",
            loras=["vexa_face"],
            extra="dramatic lighting",
            translate_hint="emphasize pose",
        )
        assert s.loras == ["vexa_face"]
        assert s.translate_hint == "emphasize pose"


# ---------------------------------------------------------------------------
# Registry — load and validate
# ---------------------------------------------------------------------------

class TestImageScenarioRegistry:
    def test_load_valid_config(self, registry):
        names = registry.scenario_names()
        assert "selfie" in names
        assert "boudoir" in names

    def test_get_scenario(self, registry):
        selfie = registry.get("selfie")
        assert isinstance(selfie, ResolvedScenario)
        assert selfie.name == "selfie"
        assert selfie.model.file == "SDXL/dreamshaperXL_alpha2Xl10.safetensors"
        assert selfie.model.dialect == "natural"
        assert selfie.model.translate is False
        assert selfie.frame.tags == "close-up, face focus"
        assert selfie.frame.size == [832, 1216]
        assert selfie.content.level == "sfw"
        assert len(selfie.loras) == 1
        assert selfie.loras[0].file == "vexa_face_lora.safetensors" or selfie.loras[0].file == "vexa_lora.safetensors"
        assert selfie.extra == "selfie angle, casual"

    def test_get_nsfw_scenario(self, registry):
        boudoir = registry.get("boudoir")
        assert boudoir.model.dialect == "booru"
        assert boudoir.model.translate is True
        assert boudoir.content.level == "nsfw"
        assert boudoir.translate_hint == "emphasize pose, outfit"

    def test_scenario_names_sorted(self, registry):
        names = registry.scenario_names()
        assert names == sorted(names)

    def test_get_unknown_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="Unknown scenario 'nonexistent'"):
            registry.get("nonexistent")

    def test_config_property(self, registry):
        cfg = registry.config
        assert isinstance(cfg, ImageScenariosConfig)
        assert "dreamshaper" in cfg.models
        assert "portrait" in cfg.frames


# ---------------------------------------------------------------------------
# Fail-fast validation
# ---------------------------------------------------------------------------

class TestValidationFailures:
    def test_missing_model_ref(self, tmp_path: Path):
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              selfie:
                modality: txt2img
                model: nonexistent
                frame: portrait
                content: sfw
        """)
        p = tmp_path / "bad.yaml"
        p.write_text(config)

        with pytest.raises(ScenarioConfigError, match="model 'nonexistent' not found"):
            ImageScenarioRegistry(p)

    def test_missing_frame_ref(self, tmp_path: Path):
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              selfie:
                modality: txt2img
                model: dreamshaper
                frame: nonexistent
                content: sfw
        """)
        p = tmp_path / "bad.yaml"
        p.write_text(config)

        with pytest.raises(ScenarioConfigError, match="frame 'nonexistent' not found"):
            ImageScenarioRegistry(p)

    def test_missing_content_ref(self, tmp_path: Path):
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              selfie:
                modality: txt2img
                model: dreamshaper
                frame: portrait
                content: nonexistent
        """)
        p = tmp_path / "bad.yaml"
        p.write_text(config)

        with pytest.raises(ScenarioConfigError, match="content 'nonexistent' not found"):
            ImageScenarioRegistry(p)

    def test_missing_modality_ref(self, tmp_path: Path):
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              selfie:
                modality: img2img
                model: dreamshaper
                frame: portrait
                content: sfw
        """)
        p = tmp_path / "bad.yaml"
        p.write_text(config)

        with pytest.raises(ScenarioConfigError, match="modality 'img2img' not found"):
            ImageScenarioRegistry(p)

    def test_model_modality_mismatch(self, tmp_path: Path):
        """Model doesn't support the scenario's modality."""
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
              img2img:
                template: "img2img.json"
            scenarios:
              selfie:
                modality: img2img
                model: dreamshaper
                frame: portrait
                content: sfw
        """)
        p = tmp_path / "bad.yaml"
        p.write_text(config)

        with pytest.raises(ScenarioConfigError, match="does not support modality"):
            ImageScenarioRegistry(p)

    def test_missing_lora_ref(self, tmp_path: Path):
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            loras:
              vexa_face:
                file: "vexa_lora.safetensors"
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              selfie:
                modality: txt2img
                model: dreamshaper
                frame: portrait
                content: sfw
                loras: [nonexistent_lora]
        """)
        p = tmp_path / "bad.yaml"
        p.write_text(config)

        with pytest.raises(ScenarioConfigError, match="lora 'nonexistent_lora' not found"):
            ImageScenarioRegistry(p)

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(ScenarioConfigError, match="not found"):
            ImageScenarioRegistry(tmp_path / "nonexistent.yaml")

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.yaml"
        p.write_text("")

        with pytest.raises(ScenarioConfigError, match="empty"):
            ImageScenarioRegistry(p)

    def test_yaml_list_instead_of_mapping(self, tmp_path: Path):
        p = tmp_path / "bad.yaml"
        p.write_text("- item1\n- item2\n")

        with pytest.raises(ScenarioConfigError, match="must be a YAML mapping"):
            ImageScenarioRegistry(p)

    def test_multiple_errors_aggregated(self, tmp_path: Path):
        """Multiple bad scenarios report all errors."""
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              bad_model:
                modality: txt2img
                model: nonexistent
                frame: portrait
                content: sfw
              bad_frame:
                modality: txt2img
                model: dreamshaper
                frame: nonexistent
                content: sfw
        """)
        p = tmp_path / "bad.yaml"
        p.write_text(config)

        with pytest.raises(ScenarioConfigError) as exc_info:
            ImageScenarioRegistry(p)

        error_msg = str(exc_info.value)
        assert "bad_model" in error_msg
        assert "bad_frame" in error_msg


# ---------------------------------------------------------------------------
# ResolvedScenario dataclass
# ---------------------------------------------------------------------------

class TestResolvedScenario:
    def test_frozen(self, registry):
        s = registry.get("selfie")
        with pytest.raises(Exception):  # FrozenInstanceError
            s.name = "modified"

    def test_repr(self, registry):
        s = registry.get("selfie")
        assert "selfie" in repr(s)

    def test_loras_preserved(self, registry):
        selfie = registry.get("selfie")
        assert len(selfie.loras) == 1
        assert selfie.loras[0].strength == 0.8
        assert selfie.loras[0].trigger == "vexatoken"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

class TestLoadImageScenarios:
    def test_convenience_function(self, valid_config_path: Path):
        registry = load_image_scenarios(valid_config_path)
        assert isinstance(registry, ImageScenarioRegistry)
        assert "selfie" in registry.scenario_names()

    def test_convenience_fail(self, tmp_path: Path):
        with pytest.raises(ScenarioConfigError):
            load_image_scenarios(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_scenario_without_loras(self, tmp_path: Path):
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              simple:
                modality: txt2img
                model: dreamshaper
                frame: portrait
                content: sfw
        """)
        p = tmp_path / "simple.yaml"
        p.write_text(config)

        registry = ImageScenarioRegistry(p)
        simple = registry.get("simple")
        assert simple.loras == []

    def test_scenario_without_extra(self, tmp_path: Path):
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              simple:
                modality: txt2img
                model: dreamshaper
                frame: portrait
                content: sfw
        """)
        p = tmp_path / "simple.yaml"
        p.write_text(config)

        registry = ImageScenarioRegistry(p)
        simple = registry.get("simple")
        assert simple.extra == ""
        assert simple.translate_hint == ""

    def test_model_with_multiple_modalities(self, tmp_path: Path):
        """A model supporting multiple modalities works with any of them."""
        config = textwrap.dedent("""\
            models:
              multi:
                file: "multi.safetensors"
                modalities: [txt2img, img2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
              img2img:
                template: "img2img.json"
            scenarios:
              txt:
                modality: txt2img
                model: multi
                frame: portrait
                content: sfw
              img:
                modality: img2img
                model: multi
                frame: portrait
                content: sfw
        """)
        p = tmp_path / "multi.yaml"
        p.write_text(config)

        registry = ImageScenarioRegistry(p)
        assert registry.get("txt").modality.template == "txt2img.json"
        assert registry.get("img").modality.template == "img2img.json"

    def test_empty_scenarios_section(self, tmp_path: Path):
        """Config with no scenarios is valid — just empty registry."""
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios: {}
        """)
        p = tmp_path / "empty_scenarios.yaml"
        p.write_text(config)

        registry = ImageScenarioRegistry(p)
        assert registry.scenario_names() == []

    def test_available_shown_on_bad_ref(self, tmp_path: Path):
        """Error message lists available options."""
        config = textwrap.dedent("""\
            models:
              dreamshaper:
                file: "test.safetensors"
                modalities: [txt2img]
              pony:
                file: "pony.safetensors"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              bad:
                modality: txt2img
                model: nonexistent
                frame: portrait
                content: sfw
        """)
        p = tmp_path / "bad.yaml"
        p.write_text(config)

        with pytest.raises(ScenarioConfigError) as exc_info:
            ImageScenarioRegistry(p)

        error_msg = str(exc_info.value)
        assert "dreamshaper" in error_msg
        assert "pony" in error_msg
