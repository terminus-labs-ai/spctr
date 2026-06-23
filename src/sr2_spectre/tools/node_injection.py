"""Generic node-stack injection for ComfyUI workflow graphs.

Injects a variable-length node stack at a **named edge** of the graph,
rewiring edge references. The mechanism is generic over edge name so
LoRA, IPAdapter, and ControlNet are later config-adds, not rewrites.

Usage:
    # Define edges for a template
    edges = {
        "model": WorkflowEdge(
            name="model",
            source=EdgeSource(node_id="4", output_index=0),
            consumers=[EdgeConsumer(node_id="3", input_key="model")],
        ),
        "clip": WorkflowEdge(
            name="clip",
            source=EdgeSource(node_id="4", output_index=1),
            consumers=[
                EdgeConsumer(node_id="6", input_key="clip"),
                EdgeConsumer(node_id="7", input_key="clip"),
            ],
        ),
    }

    # Build a LoRA stack
    stack = build_lora_stack([
        LoraFragment(file="face.safetensors", strength=0.8, trigger="facetoken"),
        LoraFragment(file="style.safetensors", strength=0.5, trigger="styletoken"),
    ])

    # Inject
    workflow = inject(workflow, edges, stack)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EdgeSource:
    """Source of a workflow edge (node output)."""
    node_id: str
    output_index: int


@dataclass(frozen=True)
class EdgeConsumer:
    """Consumer of a workflow edge (node input)."""
    node_id: str
    input_key: str


@dataclass(frozen=True)
class WorkflowEdge:
    """A named edge in the workflow graph.

    An edge connects a source node output to zero or more consumer inputs.
    Injection inserts a node stack between source and consumers.
    """
    name: str
    source: EdgeSource
    consumers: list[EdgeConsumer] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Node stack model
# ---------------------------------------------------------------------------

@dataclass
class NodeStack:
    """A stack of nodes to inject at named edges.

    The stack declares which edges it consumes (inputs) and produces (outputs).
    The injector wires edge sources to stack inputs, and rewires edge consumers
    to stack outputs.

    Attributes:
        nodes: List of node dicts with class_type and static input values.
            Wire references (to edge sources) are filled by the injector.
        consumes: Maps edge_name -> (node_idx, input_key). Declares which
            edge feeds which input of which node in the stack.
        produces: Maps edge_name -> (node_idx, output_idx). Declares which
            node output produces which edge after injection.
        internal_wires: Optional list of (src_node_idx, src_output_idx,
            dst_node_idx, dst_input_key) tuples for wiring nodes within
            the stack (e.g. chaining LoRA loaders).
    """
    nodes: list[dict[str, Any]]
    consumes: dict[str, tuple[int, str]]
    produces: dict[str, tuple[int, int]]
    internal_wires: list[tuple[int, int, int, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Injection error
# ---------------------------------------------------------------------------

class InjectionError(Exception):
    """Raised when node injection fails."""


# ---------------------------------------------------------------------------
# Core injection
# ---------------------------------------------------------------------------

def _next_node_id(workflow: dict[str, Any]) -> str:
    """Return the next available numeric node ID."""
    existing = {int(nid) for nid in workflow.keys() if nid.isdigit()}
    if not existing:
        return "10"
    return str(max(existing) + 1)


def inject(
    workflow: dict[str, Any],
    edges: dict[str, WorkflowEdge],
    stack: NodeStack,
) -> dict[str, Any]:
    """Inject a node stack at named edges, rewiring consumers.

    The workflow is NOT mutated. Returns a new workflow dict with the
    stack nodes inserted and edge references rewired.

    Args:
        workflow: ComfyUI workflow dict.
        edges: Named edges available for injection.
        stack: Node stack declaring consumed/produced edges.

    Returns:
        New workflow dict with injection applied.

    Raises:
        InjectionError: If a consumed/produced edge is unknown, or a
            stack node index is out of range.
    """
    result = copy.deepcopy(workflow)

    # Validate consumed edges exist
    for edge_name in stack.consumes:
        if edge_name not in edges:
            available = sorted(edges.keys())
            raise InjectionError(
                f"Stack consumes unknown edge '{edge_name}'. "
                f"Available: {available}"
            )

    # Validate produced edges exist
    for edge_name in stack.produces:
        if edge_name not in edges:
            available = sorted(edges.keys())
            raise InjectionError(
                f"Stack produces unknown edge '{edge_name}'. "
                f"Available: {available}"
            )

    # Validate stack node indices
    for edge_name, (node_idx, _) in stack.consumes.items():
        if node_idx < 0 or node_idx >= len(stack.nodes):
            raise InjectionError(
                f"Stack consumes edge '{edge_name}' from node index {node_idx}, "
                f"but stack has {len(stack.nodes)} nodes (0-{len(stack.nodes) - 1})"
            )

    for edge_name, (node_idx, _) in stack.produces.items():
        if node_idx < 0 or node_idx >= len(stack.nodes):
            raise InjectionError(
                f"Stack produces edge '{edge_name}' from node index {node_idx}, "
                f"but stack has {len(stack.nodes)} nodes (0-{len(stack.nodes) - 1})"
            )

    # Assign new node IDs to stack nodes
    stack_node_ids: list[str] = []
    for _ in stack.nodes:
        nid = _next_node_id(result)
        stack_node_ids.append(nid)
        result[nid] = {"class_type": _["class_type"], "inputs": copy.deepcopy(_["inputs"])}

    logger.debug(
        "Injected %d nodes: %s", len(stack_node_ids), stack_node_ids
    )

    # Wire internal connections within the stack
    for src_idx, src_out_idx, dst_idx, dst_input_key in stack.internal_wires:
        if src_idx < 0 or src_idx >= len(stack_node_ids):
            raise InjectionError(
                f"Internal wire source index {src_idx} out of range "
                f"(stack has {len(stack_node_ids)} nodes)"
            )
        if dst_idx < 0 or dst_idx >= len(stack_node_ids):
            raise InjectionError(
                f"Internal wire dest index {dst_idx} out of range "
                f"(stack has {len(stack_node_ids)} nodes)"
            )
        wire = [stack_node_ids[src_idx], src_out_idx]
        result[stack_node_ids[dst_idx]]["inputs"][dst_input_key] = wire
        logger.debug(
            "Internal wire: [%s, %d] -> %s[%s]",
            stack_node_ids[src_idx], src_out_idx,
            stack_node_ids[dst_idx], dst_input_key,
        )

    # Wire consumed edges: edge source -> stack input
    for edge_name, (node_idx, input_key) in stack.consumes.items():
        edge = edges[edge_name]
        target_nid = stack_node_ids[node_idx]
        wire = [edge.source.node_id, edge.source.output_index]
        result[target_nid]["inputs"][input_key] = wire
        logger.debug(
            "Wired edge '%s' source [%s, %d] -> stack node %s[%s]",
            edge_name, edge.source.node_id, edge.source.output_index,
            target_nid, input_key,
        )

    # Rewire produced edges: stack output -> edge consumers
    for edge_name, (node_idx, output_idx) in stack.produces.items():
        edge = edges[edge_name]
        source_nid = stack_node_ids[node_idx]
        for consumer in edge.consumers:
            wire = [source_nid, output_idx]
            result[consumer.node_id]["inputs"][consumer.input_key] = wire
            logger.debug(
                "Rewired edge '%s' consumer %s[%s] <- stack node [%s, %d]",
                edge_name, consumer.node_id, consumer.input_key,
                source_nid, output_idx,
            )

    return result


# ---------------------------------------------------------------------------
# LoRA stack builder
# ---------------------------------------------------------------------------

def build_lora_stack(
    loras: list,
    *,
    strength_key: str = "strength_model",
    clip_strength_key: str = "strength_clip",
    lora_name_key: str = "lora_name",
) -> NodeStack:
    """Build a NodeStack for a chain of LoRA loaders.

    Each LoraLoader takes (model, clip) as inputs and outputs (model, clip).
    The chain feeds: CheckpointLoader -> LoraLoader1 -> LoraLoader2 -> ... -> KSampler

    The stack consumes "model" and "clip" edges and produces "model" and "clip" edges.

    Args:
        loras: List of LoraFragment-like objects with file, strength, trigger attrs.
        strength_key: Input key for model strength (default: "strength_model").
        clip_strength_key: Input key for clip strength (default: "strength_clip").
        lora_name_key: Input key for LoRA filename (default: "lora_name").

    Returns:
        A NodeStack ready for injection at the "model" and "clip" edges.

    Raises:
        InjectionError: If loras list is empty.
    """
    if not loras:
        raise InjectionError("Cannot build LoRA stack with empty lora list")

    nodes: list[dict[str, Any]] = []
    consumes: dict[str, tuple[int, str]] = {}
    produces: dict[str, tuple[int, int]] = {}
    internal_wires: list[tuple[int, int, int, str]] = []

    for i, lora in enumerate(loras):
        node = {
            "class_type": "LoraLoader",
            "inputs": {
                lora_name_key: lora.file,
                strength_key: lora.strength,
                clip_strength_key: lora.strength,
                # model and clip are wired by the injector
                "model": None,
                "clip": None,
            },
        }
        nodes.append(node)

        if i == 0:
            # First node consumes the original edges
            consumes["model"] = (0, "model")
            consumes["clip"] = (0, "clip")
        else:
            # Chain: wire previous node's outputs to this node's inputs
            # LoraLoader outputs: 0=model, 1=clip
            internal_wires.append((i - 1, 0, i, "model"))
            internal_wires.append((i - 1, 1, i, "clip"))

    # Last node produces the output edges
    # LoraLoader outputs: 0=model, 1=clip
    last_idx = len(nodes) - 1
    produces["model"] = (last_idx, 0)
    produces["clip"] = (last_idx, 1)

    return NodeStack(
        nodes=nodes, consumes=consumes, produces=produces,
        internal_wires=internal_wires,
    )


# ---------------------------------------------------------------------------
# Edge discovery (convenience)
# ---------------------------------------------------------------------------

def discover_edges(workflow: dict[str, Any]) -> dict[str, WorkflowEdge]:
    """Discover injectable edges from a workflow by scanning wire references.

    Scans all node inputs for wire references [node_id, output_idx] and
    builds an edge map. Edges are named by convention based on the source
    node's class_type and output index.

    Known conventions:
    - CheckpointLoaderSimple output 0 -> "model"
    - CheckpointLoaderSimple output 1 -> "clip"
    - CheckpointLoaderSimple output 2 -> "vae"
    - CLIPTextEncode output 0 -> "conditioning"

    Returns:
        Dict mapping edge names to WorkflowEdge with source and consumers.
    """
    # Convention: (class_type, output_idx) -> edge_name
    EDGE_CONVENTIONS: dict[tuple[str, int], str] = {
        ("CheckpointLoaderSimple", 0): "model",
        ("CheckpointLoaderSimple", 1): "clip",
        ("CheckpointLoaderSimple", 2): "vae",
        ("CLIPTextEncode", 0): "conditioning",
    }

    edges: dict[str, WorkflowEdge] = {}

    for node_id, node in workflow.items():
        if "inputs" not in node:
            continue
        for input_key, value in node["inputs"].items():
            if not isinstance(value, list) or len(value) != 2:
                continue

            source_nid, output_idx = value

            # Look up source node class_type
            source_node = workflow.get(str(source_nid), {})
            class_type = source_node.get("class_type", "")

            edge_name = EDGE_CONVENTIONS.get((class_type, output_idx))
            if edge_name is None:
                continue

            if edge_name not in edges:
                edges[edge_name] = WorkflowEdge(
                    name=edge_name,
                    source=EdgeSource(node_id=str(source_nid), output_index=output_idx),
                    consumers=[],
                )

            edges[edge_name].consumers.append(
                EdgeConsumer(node_id=node_id, input_key=input_key)
            )

    return edges
