"""Tests for the workflow template loader + structured patch-map."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sr2_spectre.tools.builtins.workflow_loader import (
    PatchMap,
    apply_patches,
    build_workflow,
    load_template,
)


# -- load_template --

def test_load_txt2img_template():
    """Loading txt2img returns a valid workflow dict."""
    wf = load_template("txt2img")
    assert isinstance(wf, dict)
    # All expected nodes present
    for node_id in ("3", "4", "5", "6", "7", "8", "9"):
        assert node_id in wf, f"Node {node_id} missing from txt2img template"


def test_load_template_returns_deep_copy():
    """Two loads return independent dicts."""
    wf1 = load_template("txt2img")
    wf2 = load_template("txt2img")
    wf1["3"]["inputs"]["seed"] = 999
    assert wf2["3"]["inputs"]["seed"] != 999


def test_load_template_unknown_modality():
    with pytest.raises(ValueError, match="Unknown modality"):
        load_template("img2img")


def test_load_template_node_class_types():
    """Verify the template has correct ComfyUI node types."""
    wf = load_template("txt2img")
    expected = {
        "3": "KSampler",
        "4": "CheckpointLoaderSimple",
        "5": "EmptyLatentImage",
        "6": "CLIPTextEncode",
        "7": "CLIPTextEncode",
        "8": "VAEDecode",
        "9": "SaveImage",
    }
    for node_id, cls_type in expected.items():
        assert wf[node_id]["class_type"] == cls_type, (
            f"Node {node_id}: expected {cls_type}, got {wf[node_id]['class_type']}"
        )


# -- apply_patches --

def test_apply_patches_basic():
    """Patch a single input value."""
    wf = load_template("txt2img")
    patched = apply_patches(wf, {("3", "seed"): 42})
    assert patched["3"]["inputs"]["seed"] == 42
    # Original unchanged
    assert wf["3"]["inputs"]["seed"] == 0


def test_apply_patches_multiple():
    """Patch multiple inputs across nodes."""
    patches: PatchMap = {
        ("3", "seed"): 12345,
        ("3", "steps"): 20,
        ("6", "text"): "a cat",
        ("7", "text"): "blurry",
        ("5", "width"): 512,
    }
    patched = apply_patches(load_template("txt2img"), patches)

    assert patched["3"]["inputs"]["seed"] == 12345
    assert patched["3"]["inputs"]["steps"] == 20
    assert patched["6"]["inputs"]["text"] == "a cat"
    assert patched["7"]["inputs"]["text"] == "blurry"
    assert patched["5"]["inputs"]["width"] == 512


def test_apply_patches_does_not_mutate_original():
    """Original workflow is never mutated."""
    wf = load_template("txt2img")
    original_seed = wf["3"]["inputs"]["seed"]
    apply_patches(wf, {("3", "seed"): 999})
    assert wf["3"]["inputs"]["seed"] == original_seed


def test_apply_patches_empty():
    """Empty patch-map returns a deep copy."""
    wf = load_template("txt2img")
    patched = apply_patches(wf, {})
    assert patched == wf
    assert patched is not wf


def test_apply_patches_nonexistent_node():
    with pytest.raises(KeyError, match="doesn't exist in workflow"):
        apply_patches(load_template("txt2img"), {("999", "seed"): 1})


def test_apply_patches_nonexistent_input():
    with pytest.raises(KeyError, match="doesn't exist"):
        apply_patches(load_template("txt2img"), {("3", "nonexistent_key"): 1})


# -- build_workflow --

def test_build_workflow_no_patches():
    """build_workflow with no patches returns template."""
    wf = build_workflow("txt2img")
    assert wf["3"]["inputs"]["seed"] == 0


def test_build_workflow_with_patches():
    """build_workflow applies patches in one call."""
    wf = build_workflow("txt2img", {("3", "seed"): 777, ("6", "text"): "hello"})
    assert wf["3"]["inputs"]["seed"] == 777
    assert wf["6"]["inputs"]["text"] == "hello"


def test_build_workflow_unknown_modality():
    with pytest.raises(ValueError, match="Unknown modality"):
        build_workflow("img2img")


# -- Round-trip equivalence with _build_text2img_workflow --

def test_template_matches_built_workflow():
    """The seeded template + patches should match _build_text2img_workflow output."""
    from sr2_spectre.tools.builtins.generate_image import GenerateImageTool

    tool = GenerateImageTool(
        checkpoint="SDXL\\dreamshaperXL_alpha2Xl10.safetensors",
        width=1024,
        height=1024,
        steps=28,
        cfg=7.0,
    )

    positive = "a test prompt"
    negative = "blurry, ugly"
    seed = 42

    # Build via the old method
    built = tool._build_text2img_workflow(positive, negative, seed)

    # Build via template + patches
    patches: PatchMap = {
        ("3", "seed"): seed,
        ("6", "text"): positive,
        ("7", "text"): negative,
    }
    patched = build_workflow("txt2img", patches)

    assert patched == built, (
        "Template+patches output differs from _build_text2img_workflow. "
        "Diff:\n" + _diff_dicts(patched, built)
    )


def test_template_matches_custom_checkpoint():
    """Template with checkpoint patch matches built workflow with custom checkpoint."""
    from sr2_spectre.tools.builtins.generate_image import GenerateImageTool

    tool = GenerateImageTool(
        checkpoint="custom_model.safetensors",
        width=768,
        height=768,
        steps=20,
        cfg=5.0,
    )

    built = tool._build_text2img_workflow("prompt", "neg", 99)

    patches: PatchMap = {
        ("3", "seed"): 99,
        ("3", "steps"): 20,
        ("3", "cfg"): 5.0,
        ("4", "ckpt_name"): "custom_model.safetensors",
        ("5", "width"): 768,
        ("5", "height"): 768,
        ("6", "text"): "prompt",
        ("7", "text"): "neg",
    }
    patched = build_workflow("txt2img", patches)

    assert patched == built


def _diff_dicts(a: Any, b: Any, path: str = "") -> str:
    """Minimal diff helper for debug output."""
    if a == b:
        return ""
    if type(a) != type(b):
        return f"Type mismatch at {path or 'root'}: {type(a)} vs {type(b)}"
    if isinstance(a, dict):
        all_keys = sorted(set(list(a.keys()) + list(b.keys())))
        diffs = []
        for k in all_keys:
            if k not in b:
                diffs.append(f"  {path}.{k}: present in A, missing in B")
            elif k not in a:
                diffs.append(f"  {path}.{k}: missing in A, present in B")
            else:
                sub = _diff_dicts(a[k], b[k], f"{path}.{k}")
                if sub:
                    diffs.append(sub)
        return "\n".join(diffs)
    if isinstance(a, list):
        if len(a) != len(b):
            return f"List length mismatch at {path}: {len(a)} vs {len(b)}"
        diffs = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            sub = _diff_dicts(ai, bi, f"{path}[{i}]")
            if sub:
                diffs.append(sub)
        return "\n".join(diffs)
    return f"Value mismatch at {path or 'root'}: {a!r} vs {b!r}"
