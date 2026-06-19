"""Tests for per-character content cap enforcement (spc-71).

The GenerateImageTool accepts a max_content cap (sfw|nsfw) from character
config. When the agent requests a scenario whose content level exceeds the
cap, the tool refuses — generating nothing and returning a refusal string.

This is a hard floor, independent of the model's scenario choice.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from sr2_spectre.tools.builtins.generate_image import GenerateImageTool
from sr2_spectre.tools.image_scenarios import ImageScenarioRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENARIO_CONFIG = textwrap.dedent("""\
    models:
      dreamshaper:
        file: "SDXL/dreamshaperXL.safetensors"
        dialect: natural
        translate: false
        quality: "masterpiece, best quality"
        negative: "low quality, blurry"
        modalities: [txt2img]
      pony:
        file: "Pony/ponyDiffusionV6XL.safetensors"
        dialect: booru
        translate: true
        quality: "score_9"
        negative: "score_4"
        modalities: [txt2img]

    frames:
      portrait:
        tags: "close-up"
        size: [832, 1216]
      full_body:
        tags: "full body"
        size: [832, 1216]

    contents:
      sfw:
        tags: "sfw"
        level: sfw
      nsfw:
        tags: "nsfw"
        level: nsfw

    modalities:
      txt2img:
        template: "txt2img.json"

    scenarios:
      selfie:
        modality: txt2img
        model: dreamshaper
        frame: portrait
        content: sfw
      boudoir:
        modality: txt2img
        model: pony
        frame: full_body
        content: nsfw
""")


@pytest.fixture
def scenario_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "image_scenarios.yaml"
    p.write_text(SCENARIO_CONFIG)
    return p


@pytest.fixture
def registry(scenario_yaml: Path) -> ImageScenarioRegistry:
    return ImageScenarioRegistry(scenario_yaml)


@pytest.fixture
def tool_with_registry(registry: ImageScenarioRegistry) -> GenerateImageTool:
    return GenerateImageTool(
        max_content="nsfw",
        scenario_registry=registry,
    )


@pytest.fixture
def sfw_capped_tool(registry: ImageScenarioRegistry) -> GenerateImageTool:
    return GenerateImageTool(
        max_content="sfw",
        scenario_registry=registry,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestContentCapConstructor:
    def test_default_max_content_is_nsfw(self):
        """Default cap is nsfw (most permissive)."""
        tool = GenerateImageTool()
        assert tool.max_content == "nsfw"

    def test_sfw_cap_accepted(self):
        tool = GenerateImageTool(max_content="sfw")
        assert tool.max_content == "sfw"

    def test_nsfw_cap_accepted(self):
        tool = GenerateImageTool(max_content="nsfw")
        assert tool.max_content == "nsfw"

    def test_invalid_cap_raises(self):
        """Unknown content level raises ValueError."""
        with pytest.raises(ValueError, match="max_content"):
            GenerateImageTool(max_content="ecchi")

    def test_registry_optional(self):
        """Tool works without a registry (cap enforcement skipped)."""
        tool = GenerateImageTool(max_content="sfw")
        assert tool.scenario_registry is None


# ---------------------------------------------------------------------------
# Cap enforcement — refusal
# ---------------------------------------------------------------------------

class TestContentCapRefusal:
    @pytest.mark.asyncio
    async def test_sfw_cap_blocks_nsfw_scenario(self, sfw_capped_tool):
        """Character with sfw cap is refused when model picks nsfw scenario."""
        sfw_capped_tool.client.is_available = AsyncMock(return_value=True)

        result = await sfw_capped_tool(
            intent="character in evening gown",
            scenario="boudoir",
        )
        # Must be a refusal string, not a file path
        assert "refused" in result.lower() or "blocked" in result.lower() or "exceeds" in result.lower()
        assert "boudoir" in result.lower()
        assert "nsfw" in result.lower()

    @pytest.mark.asyncio
    async def test_sfw_cap_allows_sfw_scenario(self, sfw_capped_tool, tmp_path):
        """Character with sfw cap can still use sfw scenarios."""
        sfw_capped_tool.output_dir = tmp_path
        sfw_capped_tool.client.is_available = AsyncMock(return_value=True)
        sfw_capped_tool.client.generate = AsyncMock(
            return_value=Path("/tmp/img.png")
        )

        result = await sfw_capped_tool(
            intent="character smiling",
            scenario="selfie",
        )
        # Should proceed normally
        assert "/tmp/img.png" in result

    @pytest.mark.asyncio
    async def test_nsfw_cap_allows_all_scenarios(self, tool_with_registry, tmp_path):
        """Character with nsfw cap allows both sfw and nsfw scenarios."""
        tool_with_registry.output_dir = tmp_path
        tool_with_registry.client.is_available = AsyncMock(return_value=True)
        tool_with_registry.client.generate = AsyncMock(
            return_value=Path("/tmp/img.png")
        )

        # NSFW scenario
        result = await tool_with_registry(
            intent="character in evening gown",
            scenario="boudoir",
        )
        assert "/tmp/img.png" in result

        # SFW scenario
        result = await tool_with_registry(
            intent="character smiling",
            scenario="selfie",
        )
        assert "/tmp/img.png" in result


# ---------------------------------------------------------------------------
# Cap enforcement — no registry fallback
# ---------------------------------------------------------------------------

class TestContentCapNoRegistry:
    @pytest.mark.asyncio
    async def test_no_registry_skips_enforcement(self, tmp_path):
        """Without a registry, cap enforcement is skipped (no scenario lookup)."""
        tool = GenerateImageTool(max_content="sfw")  # sfw cap, but no registry
        tool.output_dir = tmp_path
        tool.client.is_available = AsyncMock(return_value=True)
        tool.client.generate = AsyncMock(return_value=Path("/tmp/img.png"))

        # Even with an nsfw scenario name, without registry we can't look it up
        result = await tool(
            intent="character",
            scenario="boudoir",
        )
        # Should proceed (no registry to enforce against)
        assert "/tmp/img.png" in result


# ---------------------------------------------------------------------------
# Cap enforcement — unknown scenario
# ---------------------------------------------------------------------------

class TestContentCapUnknownScenario:
    @pytest.mark.asyncio
    async def test_unknown_scenario_proceeds(self, sfw_capped_tool, tmp_path):
        """Scenario not in registry proceeds (unknown content level = allow)."""
        sfw_capped_tool.output_dir = tmp_path
        sfw_capped_tool.client.is_available = AsyncMock(return_value=True)
        sfw_capped_tool.client.generate = AsyncMock(
            return_value=Path("/tmp/img.png")
        )

        result = await sfw_capped_tool(
            intent="character",
            scenario="unknown_preset",
        )
        # Unknown scenario -> proceed (can't enforce what we don't know)
        assert "/tmp/img.png" in result


# ---------------------------------------------------------------------------
# Cap enforcement — no scenario specified
# ---------------------------------------------------------------------------

class TestContentCapNoScenario:
    @pytest.mark.asyncio
    async def test_no_scenario_proceeds(self, sfw_capped_tool, tmp_path):
        """When no scenario is specified, cap check is skipped."""
        sfw_capped_tool.output_dir = tmp_path
        sfw_capped_tool.client.is_available = AsyncMock(return_value=True)
        sfw_capped_tool.client.generate = AsyncMock(
            return_value=Path("/tmp/img.png")
        )

        result = await sfw_capped_tool(intent="character smiling")
        assert "/tmp/img.png" in result


# ---------------------------------------------------------------------------
# Refusal string format
# ---------------------------------------------------------------------------

class TestRefusalString:
    @pytest.mark.asyncio
    async def test_refusal_does_not_contain_image_path(self, sfw_capped_tool):
        """Refusal string must not look like a file path."""
        sfw_capped_tool.client.is_available = AsyncMock(return_value=True)
        result = await sfw_capped_tool(intent="test", scenario="boudoir")
        assert not result.endswith(".png")
        assert not result.endswith(".jpg")

    @pytest.mark.asyncio
    async def test_refusal_mentions_scenario_and_cap(self, sfw_capped_tool):
        """Refusal string identifies the scenario and the cap."""
        sfw_capped_tool.client.is_available = AsyncMock(return_value=True)
        result = await sfw_capped_tool(intent="test", scenario="boudoir")
        assert "boudoir" in result.lower()
        assert "sfw" in result.lower()


# ---------------------------------------------------------------------------
# Content level comparison
# ---------------------------------------------------------------------------

class TestContentLevelComparison:
    def test_sfw_below_nsfw(self):
        """sfw level is below nsfw level."""
        from sr2_spectre.tools.builtins.generate_image import _content_level_rank
        assert _content_level_rank("sfw") < _content_level_rank("nsfw")

    def test_same_level_allowed(self):
        """Same level is allowed (cap == scenario level)."""
        from sr2_spectre.tools.builtins.generate_image import _content_level_rank
        assert _content_level_rank("sfw") <= _content_level_rank("sfw")
        assert _content_level_rank("nsfw") <= _content_level_rank("nsfw")
