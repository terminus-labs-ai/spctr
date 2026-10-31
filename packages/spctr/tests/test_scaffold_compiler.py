"""Tests for Zone-1 deterministic scaffold compiler (spc-66).

Covers:
- Full scaffold assembly order (quality → frame → content → extra → triggers → intent)
- Negative prompt from checkpoint
- LoRA trigger token injection
- Empty/missing fragment fields (quality, tags, extra, triggers)
- FR10: compiled-prompt logging
- CompiledPrompt dataclass (frozen, field integrity)
"""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from sr2_spectre.tools.image_scenarios import (
    ContentFragment,
    FrameFragment,
    ImageScenarioRegistry,
    LoraFragment,
    ModalityFragment,
    ModelFragment,
    ResolvedScenario,
)
from sr2_spectre.tools.scaffold_compiler import (
    CompiledPrompt,
    compile_scaffold,
)


# ---------------------------------------------------------------------------
# Fixtures — resolved scenarios built directly for unit testing
# ---------------------------------------------------------------------------

@pytest.fixture
def full_scenario() -> ResolvedScenario:
    """Scenario with all fields populated."""
    return ResolvedScenario(
        name="selfie",
        modality=ModalityFragment(template="txt2img.json"),
        model=ModelFragment(
            file="SDXL/dreamshaperXL.safetensors",
            dialect="natural",
            translate=False,
            quality="masterpiece, best quality, detailed",
            negative="low quality, blurry, deformed",
            modalities=["txt2img"],
        ),
        frame=FrameFragment(tags="close-up, face focus", size=[832, 1216]),
        content=ContentFragment(tags="sfw", level="sfw"),
        loras=[
            LoraFragment(file="vexa_lora.safetensors", strength=0.8, trigger="vexatoken"),
        ],
        extra="selfie angle, casual",
        translate_hint="",
    )


@pytest.fixture
def minimal_scenario() -> ResolvedScenario:
    """Scenario with empty optional fields."""
    return ResolvedScenario(
        name="bare",
        modality=ModalityFragment(template="txt2img.json"),
        model=ModelFragment(
            file="test.safetensors",
            dialect="natural",
            translate=False,
            quality="",
            negative="",
            modalities=["txt2img"],
        ),
        frame=FrameFragment(tags="", size=[1024, 1024]),
        content=ContentFragment(tags="", level="sfw"),
        loras=[],
        extra="",
        translate_hint="",
    )


@pytest.fixture
def multi_lora_scenario() -> ResolvedScenario:
    """Scenario with multiple LoRAs, some with triggers, some without."""
    return ResolvedScenario(
        name="multi_lora",
        modality=ModalityFragment(template="txt2img.json"),
        model=ModelFragment(
            file="Pony/ponyDiffusionV6XL.safetensors",
            dialect="booru",
            translate=True,
            quality="score_9, score_8_up, source_anime",
            negative="score_4, score_5, worst quality",
            modalities=["txt2img"],
        ),
        frame=FrameFragment(tags="full body, standing", size=[832, 1216]),
        content=ContentFragment(tags="nsfw", level="nsfw"),
        loras=[
            LoraFragment(file="face.safetensors", strength=0.8, trigger="facetoken"),
            LoraFragment(file="outfit.safetensors", strength=0.6, trigger="outfittoken"),
            LoraFragment(file="style.safetensors", strength=0.5, trigger=""),  # no trigger
        ],
        extra="dramatic lighting",
        translate_hint="emphasize pose",
    )


# ---------------------------------------------------------------------------
# Core assembly order
# ---------------------------------------------------------------------------

class TestCompileScaffoldOrder:
    def test_full_assembly_order(self, full_scenario: ResolvedScenario):
        """Verify deterministic order: quality → frame → content → extra → triggers → intent."""
        result = compile_scaffold(full_scenario, "vexa grinning on a rooftop")

        assert result.positive == (
            "masterpiece, best quality, detailed, "
            "close-up, face focus, "
            "sfw, "
            "selfie angle, casual, "
            "vexatoken, "
            "vexa grinning on a rooftop"
        )
        assert result.negative == "low quality, blurry, deformed"

    def test_intent_always_last(self, full_scenario: ResolvedScenario):
        """User intent is always appended at the end of the positive prompt."""
        result = compile_scaffold(full_scenario, "custom intent here")
        assert result.positive.endswith("custom intent here")

    def test_quality_first(self, full_scenario: ResolvedScenario):
        """Checkpoint quality boilerplate is always first."""
        result = compile_scaffold(full_scenario, "test")
        assert result.positive.startswith("masterpiece, best quality, detailed")


# ---------------------------------------------------------------------------
# Negative prompt
# ---------------------------------------------------------------------------

class TestNegativePrompt:
    def test_negative_from_checkpoint(self, full_scenario: ResolvedScenario):
        result = compile_scaffold(full_scenario, "test")
        assert result.negative == "low quality, blurry, deformed"

    def test_empty_negative(self, minimal_scenario: ResolvedScenario):
        """When checkpoint has no negative, result is empty string."""
        result = compile_scaffold(minimal_scenario, "test")
        assert result.negative == ""


# ---------------------------------------------------------------------------
# Empty / missing fields
# ---------------------------------------------------------------------------

class TestEmptyFields:
    def test_minimal_scenario(self, minimal_scenario: ResolvedScenario):
        """Scenario with all optional fields empty — only intent in positive."""
        result = compile_scaffold(minimal_scenario, "just the intent")
        assert result.positive == "just the intent"
        assert result.negative == ""

    def test_empty_intent(self, full_scenario: ResolvedScenario):
        """Empty intent — scaffold only, no trailing comma."""
        result = compile_scaffold(full_scenario, "")
        assert result.positive == (
            "masterpiece, best quality, detailed, "
            "close-up, face focus, "
            "sfw, "
            "selfie angle, casual, "
            "vexatoken"
        )
        # Should NOT end with ", "
        assert not result.positive.endswith(", ")

    def test_empty_quality(self):
        """Missing quality skips that segment."""
        scenario = ResolvedScenario(
            name="no_quality",
            modality=ModalityFragment(),
            model=ModelFragment(
                file="test.safetensors",
                quality="",
                negative="bad",
                modalities=["txt2img"],
            ),
            frame=FrameFragment(tags="portrait"),
            content=ContentFragment(tags="sfw"),
            loras=[],
            extra="",
        )
        result = compile_scaffold(scenario, "subject")
        assert result.positive == "portrait, sfw, subject"
        assert "quality" not in result.positive.lower()

    def test_empty_frame_tags(self):
        """Empty frame tags are skipped."""
        scenario = ResolvedScenario(
            name="no_frame",
            modality=ModalityFragment(),
            model=ModelFragment(
                file="test.safetensors",
                quality="masterpiece",
                negative="bad",
                modalities=["txt2img"],
            ),
            frame=FrameFragment(tags=""),
            content=ContentFragment(tags="sfw"),
            loras=[],
            extra="",
        )
        result = compile_scaffold(scenario, "subject")
        assert result.positive == "masterpiece, sfw, subject"

    def test_empty_content_tags(self):
        """Empty content tags are skipped."""
        scenario = ResolvedScenario(
            name="no_content",
            modality=ModalityFragment(),
            model=ModelFragment(
                file="test.safetensors",
                quality="masterpiece",
                negative="bad",
                modalities=["txt2img"],
            ),
            frame=FrameFragment(tags="portrait"),
            content=ContentFragment(tags=""),
            loras=[],
            extra="",
        )
        result = compile_scaffold(scenario, "subject")
        assert result.positive == "masterpiece, portrait, subject"

    def test_empty_extra(self):
        """Empty scenario extra is skipped."""
        scenario = ResolvedScenario(
            name="no_extra",
            modality=ModalityFragment(),
            model=ModelFragment(
                file="test.safetensors",
                quality="masterpiece",
                negative="bad",
                modalities=["txt2img"],
            ),
            frame=FrameFragment(tags="portrait"),
            content=ContentFragment(tags="sfw"),
            loras=[],
            extra="",
        )
        result = compile_scaffold(scenario, "subject")
        assert result.positive == "masterpiece, portrait, sfw, subject"


# ---------------------------------------------------------------------------
# LoRA triggers
# ---------------------------------------------------------------------------

class TestLoraTriggers:
    def test_single_trigger(self, full_scenario: ResolvedScenario):
        result = compile_scaffold(full_scenario, "test")
        assert "vexatoken" in result.positive

    def test_multiple_triggers(self, multi_lora_scenario: ResolvedScenario):
        """Multiple LoRA triggers joined with comma, in declaration order."""
        result = compile_scaffold(multi_lora_scenario, "test")
        # Triggers should be joined as "facetoken, outfittoken"
        assert "facetoken, outfittoken" in result.positive
        # The no-trigger LoRA should not contribute
        assert "style" not in result.positive.lower() or "style.safetensors" not in result.positive

    def test_lora_without_trigger_skipped(self):
        """LoRA with empty trigger string contributes nothing."""
        scenario = ResolvedScenario(
            name="no_trigger",
            modality=ModalityFragment(),
            model=ModelFragment(
                file="test.safetensors",
                quality="q",
                negative="n",
                modalities=["txt2img"],
            ),
            frame=FrameFragment(tags="f"),
            content=ContentFragment(tags="c"),
            loras=[
                LoraFragment(file="a.safetensors", trigger=""),
                LoraFragment(file="b.safetensors", trigger="tok"),
            ],
            extra="",
        )
        result = compile_scaffold(scenario, "intent")
        assert result.positive == "q, f, c, tok, intent"

    def test_no_loras(self, full_scenario: ResolvedScenario):
        """Scenario with no LoRAs — no trigger segment."""
        scenario_no_lora = ResolvedScenario(
            name=full_scenario.name,
            modality=full_scenario.modality,
            model=full_scenario.model,
            frame=full_scenario.frame,
            content=full_scenario.content,
            loras=[],
            extra=full_scenario.extra,
        )
        result = compile_scaffold(scenario_no_lora, "test")
        assert "vexatoken" not in result.positive


# ---------------------------------------------------------------------------
# CompiledPrompt dataclass
# ---------------------------------------------------------------------------

class TestCompiledPrompt:
    def test_frozen(self, full_scenario: ResolvedScenario):
        result = compile_scaffold(full_scenario, "test")
        with pytest.raises(Exception):  # FrozenInstanceError
            result.positive = "modified"

    def test_metadata(self, full_scenario: ResolvedScenario):
        result = compile_scaffold(full_scenario, "test")
        assert result.scenario_name == "selfie"
        assert result.model_file == "SDXL/dreamshaperXL.safetensors"

    def test_repr(self, full_scenario: ResolvedScenario):
        result = compile_scaffold(full_scenario, "test")
        assert "CompiledPrompt" in repr(result)


# ---------------------------------------------------------------------------
# FR10: Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_compiled_prompt(
        self, full_scenario: ResolvedScenario, caplog: pytest.LogCaptureFixture
    ):
        """compile_scaffold logs intent → compiled positive + negative + model/scenario."""
        caplog.set_level(logging.INFO)

        compile_scaffold(full_scenario, "vexa on rooftop")

        # Verify the log contains key elements
        log_output = caplog.text
        assert "Scaffold compiled" in log_output
        assert "scenario=selfie" in log_output
        assert "model=SDXL/dreamshaperXL.safetensors" in log_output
        assert "natural" in log_output
        assert "vexa on rooftop" in log_output
        assert "masterpiece, best quality, detailed" in log_output
        assert "low quality, blurry, deformed" in log_output

    def test_log_truncates_long_intent(
        self, full_scenario: ResolvedScenario, caplog: pytest.LogCaptureFixture
    ):
        """Long intents are truncated in logs."""
        caplog.set_level(logging.INFO)
        long_intent = "a" * 200
        compile_scaffold(full_scenario, long_intent)

        # The log should contain the truncated intent (120 chars)
        assert "a" * 120 in caplog.text
        assert "a" * 200 not in caplog.text

    def test_log_truncates_long_positive(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Long positive prompts are truncated in logs."""
        scenario = ResolvedScenario(
            name="long",
            modality=ModalityFragment(),
            model=ModelFragment(
                file="test.safetensors",
                quality="q" * 100,
                negative="n",
                modalities=["txt2img"],
            ),
            frame=FrameFragment(tags="f" * 100),
            content=ContentFragment(tags="c" * 100),
            loras=[],
            extra="e" * 100,
        )
        caplog.set_level(logging.INFO)
        compile_scaffold(scenario, "intent")

        # Positive in log should be truncated to 200 chars
        for record in caplog.records:
            if "positive:" in record.message:
                # The positive portion after "positive: " should be ≤ 200
                pos_start = record.message.index("positive: ") + len("positive: ")
                pos_value = record.message[pos_start:].split("\n")[0]
                assert len(pos_value) <= 200


# ---------------------------------------------------------------------------
# Integration with registry
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_compile_from_registry(self, tmp_path: Path):
        """End-to-end: load config → resolve scenario → compile scaffold."""
        config = dedent("""\
            models:
              dreamshaper:
                file: "SDXL/dreamshaperXL.safetensors"
                dialect: natural
                translate: false
                quality: "masterpiece, best quality"
                negative: "low quality, blurry"
                modalities: [txt2img]
            frames:
              portrait:
                tags: "close-up"
                size: [832, 1216]
            contents:
              sfw:
                tags: "sfw"
                level: sfw
            loras:
              face:
                file: "face.safetensors"
                strength: 0.8
                trigger: "facetoken"
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              selfie:
                modality: txt2img
                model: dreamshaper
                frame: portrait
                content: sfw
                loras: [face]
                extra: "casual angle"
        """)
        p = tmp_path / "scenarios.yaml"
        p.write_text(config)

        registry = ImageScenarioRegistry(p)
        scenario = registry.get("selfie")
        result = compile_scaffold(scenario, "character smiling")

        assert result.positive == (
            "masterpiece, best quality, "
            "close-up, "
            "sfw, "
            "casual angle, "
            "facetoken, "
            "character smiling"
        )
        assert result.negative == "low quality, blurry"
        assert result.scenario_name == "selfie"
        assert result.model_file == "SDXL/dreamshaperXL.safetensors"

    def test_compile_natural_dialect(self, tmp_path: Path):
        """Natural dialect scenario — no translation, raw intent appended."""
        config = dedent("""\
            models:
              ds:
                file: "ds.safetensors"
                dialect: natural
                translate: false
                quality: "masterpiece"
                negative: "blurry"
                modalities: [txt2img]
            frames:
              p:
                tags: "portrait"
            contents:
              sfw:
                tags: "sfw"
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              test:
                modality: txt2img
                model: ds
                frame: p
                content: sfw
        """)
        p = tmp_path / "scenarios.yaml"
        p.write_text(config)

        registry = ImageScenarioRegistry(p)
        scenario = registry.get("test")
        result = compile_scaffold(scenario, "a cat on a wall")

        # Natural dialect: intent is appended as-is
        assert result.positive.endswith("a cat on a wall")
        assert result.model_file == "ds.safetensors"

    def test_compile_booru_dialect(self, tmp_path: Path):
        """Booru dialect scenario — scaffold still deterministic, intent appended raw.
        Zone-2 (FR5) would translate the intent, but Zone-1 just appends."""
        config = dedent("""\
            models:
              pony:
                file: "pony.safetensors"
                dialect: booru
                translate: true
                quality: "score_9, score_8_up"
                negative: "score_4"
                modalities: [txt2img]
            frames:
              fb:
                tags: "full body"
            contents:
              nsfw:
                tags: "nsfw"
                level: nsfw
            modalities:
              txt2img:
                template: "txt2img.json"
            scenarios:
              boudoir:
                modality: txt2img
                model: pony
                frame: fb
                content: nsfw
                extra: "dramatic lighting"
        """)
        p = tmp_path / "scenarios.yaml"
        p.write_text(config)

        registry = ImageScenarioRegistry(p)
        scenario = registry.get("boudoir")
        result = compile_scaffold(scenario, "character in evening gown")

        assert result.positive == (
            "score_9, score_8_up, "
            "full body, "
            "nsfw, "
            "dramatic lighting, "
            "character in evening gown"
        )
        assert result.negative == "score_4"
