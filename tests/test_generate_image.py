"""Tests for the GenerateImageTool."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.tools.builtins.generate_image import GenerateImageTool
from sr2_spectre.tools.builtins.comfyui_client import ComfyUIClient, ImageRef


@pytest.fixture
def tool() -> GenerateImageTool:
    return GenerateImageTool(
        comfyui_url="http://192.168.50.233:8188",
        checkpoint="test.safetensors",
        style_prompt="anime style, detailed",
        width=1024,
        height=1024,
        steps=28,
        cfg=7.0,
        output_dir="/tmp/test_images",
    )


# -- Class attributes --

def test_tool_class_attributes():
    assert GenerateImageTool.name == "generate_image"
    assert "Generate" in GenerateImageTool.description or "generate" in GenerateImageTool.description
    assert "prompt" in GenerateImageTool.input_schema["required"]


def test_tool_input_schema():
    schema = GenerateImageTool.input_schema
    assert schema["type"] == "object"
    assert "prompt" in schema["properties"]
    assert "negative_prompt" in schema["properties"]
    assert "style" in schema["properties"]


# -- Prompt assembly --

def test_assemble_prompt_basic(tool):
    result = tool._assemble_prompt("a cat", None)
    assert "a cat" in result
    assert "anime style, detailed" in result  # style_prompt always added


def test_assemble_prompt_with_style(tool):
    result = tool._assemble_prompt("a cat", "selfie")
    assert "a cat" in result
    assert "selfie style" in result
    assert "anime style, detailed" in result


def test_assemble_prompt_unknown_style(tool):
    """Unknown style is silently ignored (only adds prompt + style_prompt)."""
    result = tool._assemble_prompt("a cat", "unknown")
    assert "a cat" in result
    assert "anime style, detailed" in result


def test_assemble_prompt_no_style_prompt():
    """When no style_prompt is set, only user prompt appears."""
    tool = GenerateImageTool(style_prompt="")
    result = tool._assemble_prompt("a cat", None)
    assert result == "a cat"


# -- Workflow builder --

def test_build_text2img_workflow(tool):
    workflow = tool._build_text2img_workflow("positive", "negative", 42)
    
    # Check all expected nodes exist
    assert "3" in workflow  # KSampler
    assert "4" in workflow  # CheckpointLoaderSimple
    assert "5" in workflow  # EmptyLatentImage
    assert "6" in workflow  # CLIPTextEncode (positive)
    assert "7" in workflow  # CLIPTextEncode (negative)
    assert "8" in workflow  # VAEDecode
    assert "9" in workflow  # SaveImage
    
    # Verify class types
    assert workflow["3"]["class_type"] == "KSampler"
    assert workflow["4"]["class_type"] == "CheckpointLoaderSimple"
    assert workflow["5"]["class_type"] == "EmptyLatentImage"
    assert workflow["6"]["class_type"] == "CLIPTextEncode"
    assert workflow["8"]["class_type"] == "VAEDecode"
    assert workflow["9"]["class_type"] == "SaveImage"
    
    # Verify inputs are wired correctly
    sampler = workflow["3"]["inputs"]
    assert sampler["seed"] == 42
    assert sampler["steps"] == 28
    assert sampler["cfg"] == 7.0
    assert sampler["model"] == ["4", 0]
    assert sampler["positive"] == ["6", 0]
    assert sampler["negative"] == ["7", 0]
    
    # Verify checkpoint name
    assert workflow["4"]["inputs"]["ckpt_name"] == "test.safetensors"
    
    # Verify resolution
    assert workflow["5"]["inputs"]["width"] == 1024
    assert workflow["5"]["inputs"]["height"] == 1024
    
    # Verify prompt text
    assert "positive" in workflow["6"]["inputs"]["text"]
    assert "negative" in workflow["7"]["inputs"]["text"]


# -- Call (success) --

@pytest.mark.asyncio
async def test_call_success(tool, tmp_path):
    # Change output dir for test
    tool.output_dir = tmp_path
    
    # Mock client methods
    tool.client.is_available = AsyncMock(return_value=True)
    tool.client.generate = AsyncMock(return_value=Path("/tmp/test_images/result.png"))
    
    result = await tool(prompt="a cat sitting on a wall")
    assert "/tmp/test_images/result.png" in result


@pytest.mark.asyncio
async def test_call_comfyui_unavailable(tool):
    tool.client.is_available = AsyncMock(return_value=False)
    
    result = await tool(prompt="test")
    assert "offline" in result.lower() or "not reachable" in result.lower()


@pytest.mark.asyncio
async def test_call_with_negative_prompt(tool, tmp_path):
    tool.output_dir = tmp_path
    tool.client.is_available = AsyncMock(return_value=True)
    tool.client.generate = AsyncMock(return_value=Path("/tmp/img.png"))
    
    result = await tool(
        prompt="a cat",
        negative_prompt="dog, bird",
        style="portrait",
    )
    assert "/tmp/img.png" in result


# -- Style presets --

def test_style_presets_defined():
    assert "selfie" in GenerateImageTool.STYLE_PRESETS
    assert "portrait" in GenerateImageTool.STYLE_PRESETS
    assert "scene" in GenerateImageTool.STYLE_PRESETS
    assert "illustration" in GenerateImageTool.STYLE_PRESETS


# -- Constructor config --

def test_constructor_defaults():
    tool = GenerateImageTool()
    assert "dreamshaper" in tool.checkpoint.lower() or "sd_xl" in tool.checkpoint.lower()
    assert tool.width == 1024
    assert tool.height == 1024
    assert tool.steps == 28
    assert tool.cfg == 7.0


def test_constructor_custom():
    tool = GenerateImageTool(
        checkpoint="custom.safetensors",
        width=768,
        height=768,
        steps=20,
        cfg=5.0,
    )
    assert tool.checkpoint == "custom.safetensors"
    assert tool.width == 768
    assert tool.steps == 20
    assert tool.cfg == 5.0
