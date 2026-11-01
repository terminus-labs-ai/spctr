"""ComfyUI workflow template loader with structured patch-map.

Loads workflow JSON templates by modality (txt2img, img2img, etc.) and
applies structured patches keyed by (node_id, input_key).

This replaces string-substitution on raw JSON with a typed, structured
approach. Each patch targets a specific node input field.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Directory containing workflow template JSON files
_TEMPLATE_DIR = Path(__file__).parent / "workflow_templates"

# Supported modalities mapped to their template filenames
_MODALITY_MAP = {
    "txt2img": "txt2img.json",
}


# -- Types --

# Patch key: (node_id, input_key) → new value
# e.g. {("6", "text"): "a cat", ("3", "seed"): 42}
PatchMap = dict[tuple[str, str], Any]


# -- Public API --

def load_template(modality: str) -> dict[str, Any]:
    """Load a ComfyUI workflow template by modality name.

    Args:
        modality: One of the registered modalities (e.g. "txt2img").

    Returns:
        Deep-copy of the workflow dict (safe to mutate).

    Raises:
        ValueError: If modality is not registered.
        FileNotFoundError: If the template file doesn't exist.
    """
    if modality not in _MODALITY_MAP:
        known = ", ".join(sorted(_MODALITY_MAP.keys()))
        raise ValueError(
            f"Unknown modality '{modality}'. Available: {known}"
        )

    template_path = _TEMPLATE_DIR / _MODALITY_MAP[modality]
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found: {template_path} "
            f"(modality={modality!r})"
        )

    logger.debug("Loading workflow template: %s → %s", modality, template_path)
    raw = json.loads(template_path.read_text(encoding="utf-8"))
    # Return a deep copy so callers can mutate safely
    return json.loads(json.dumps(raw))


def apply_patches(
    workflow: dict[str, Any],
    patches: PatchMap,
) -> dict[str, Any]:
    """Apply a structured patch-map to a workflow dict.

    Each patch key is (node_id, input_key). The value replaces the
    corresponding field in workflow[node_id]["inputs"][input_key].

    Args:
        workflow: Workflow dict (will NOT be mutated; returns new dict).
        patches: Mapping of (node_id, input_key) → value.

    Returns:
        New workflow dict with patches applied.

    Raises:
        KeyError: If a patch targets a non-existent node or input key.
    """
    import copy
    result = copy.deepcopy(workflow)

    for (node_id, input_key), value in patches.items():
        if node_id not in result:
            raise KeyError(
                f"Patch targets node '{node_id}' which doesn't exist in workflow. "
                f"Available nodes: {sorted(result.keys())}"
            )
        node = result[node_id]
        if "inputs" not in node:
            raise KeyError(
                f"Node '{node_id}' ({node.get('class_type', '?')}) has no 'inputs' field"
            )
        if input_key not in node["inputs"]:
            raise KeyError(
                f"Patch targets input '{input_key}' on node '{node_id}' "
                f"({node.get('class_type', '?')}), but that input doesn't exist. "
                f"Available inputs: {sorted(node['inputs'].keys())}"
            )
        node["inputs"][input_key] = value
        logger.debug(
            "Patched node %s[%s] = %s", node_id, input_key, repr(value)[:60]
        )

    return result


def build_workflow(
    modality: str,
    patches: PatchMap | None = None,
) -> dict[str, Any]:
    """Load a template and apply patches in one call.

    Convenience wrapper around load_template() + apply_patches().

    Args:
        modality: Template modality (e.g. "txt2img").
        patches: Optional patch-map. If None, returns unmodified template.

    Returns:
        Patched workflow dict.
    """
    workflow = load_template(modality)
    if patches:
        workflow = apply_patches(workflow, patches)
    return workflow
