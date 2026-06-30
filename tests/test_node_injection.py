"""Tests for generic node-stack injection (spc-69 / FR7).

Covers:
- Edge model (WorkflowEdge, EdgeSource, EdgeConsumer)
- Core inject() — wiring, rewiring, immutability
- LoRA stack builder — single and chained
- Edge discovery from workflow templates
- Error cases (unknown edges, empty stack, out-of-range indices)
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sr2_spectre.tools.node_injection import (
    EdgeConsumer,
    EdgeSource,
    InjectionError,
    NodeStack,
    WorkflowEdge,
    build_lora_stack,
    discover_edges,
    inject,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal workflow matching txt2img.json topology
# ---------------------------------------------------------------------------

def make_txt2img_workflow() -> dict:
    """Minimal txt2img workflow matching the template topology."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": 28,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "test.safetensors"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "positive", "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "negative", "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "test", "images": ["8", 0]},
        },
    }


TXT2IMG_EDGES = {
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
    "vae": WorkflowEdge(
        name="vae",
        source=EdgeSource(node_id="4", output_index=2),
        consumers=[EdgeConsumer(node_id="8", input_key="vae")],
    ),
    "conditioning": WorkflowEdge(
        name="conditioning",
        source=EdgeSource(node_id="6", output_index=0),
        consumers=[EdgeConsumer(node_id="3", input_key="positive")],
    ),
}


# Fake LoraFragment for testing.
# Intentionally lacks a `clip_strength` attribute to exercise the
# getattr-style fallback in build_lora_stack (duck-typed objects).
@dataclass
class FakeLora:
    file: str
    strength: float = 1.0
    trigger: str = ""


# Fake LoraFragment that DOES carry an independent clip_strength field.
@dataclass
class FakeLoraClip:
    file: str
    strength: float = 1.0
    trigger: str = ""
    clip_strength: float | None = None


# ---------------------------------------------------------------------------
# Edge model tests
# ---------------------------------------------------------------------------

class TestEdgeModel:
    def test_workflow_edge_fields(self):
        edge = WorkflowEdge(
            name="model",
            source=EdgeSource(node_id="4", output_index=0),
            consumers=[EdgeConsumer(node_id="3", input_key="model")],
        )
        assert edge.name == "model"
        assert edge.source.node_id == "4"
        assert edge.source.output_index == 0
        assert len(edge.consumers) == 1

    def test_frozen_dataclasses(self):
        src = EdgeSource(node_id="4", output_index=0)
        with pytest.raises(Exception):  # FrozenInstanceError
            src.node_id = "5"

    def test_empty_consumers(self):
        edge = WorkflowEdge(
            name="orphan",
            source=EdgeSource(node_id="99", output_index=0),
        )
        assert edge.consumers == []


# ---------------------------------------------------------------------------
# Core inject() tests
# ---------------------------------------------------------------------------

class TestInject:
    def test_single_node_injection(self):
        """Inject a single node at the model edge."""
        workflow = make_txt2img_workflow()

        stack = NodeStack(
            nodes=[
                {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "lora_name": "test.safetensors",
                        "strength_model": 0.8,
                        "clip_strength": 0.8,
                        "model": None,
                        "clip": None,
                    },
                }
            ],
            consumes={"model": (0, "model"), "clip": (0, "clip")},
            produces={"model": (0, 0), "clip": (0, 1)},
        )

        result = inject(workflow, TXT2IMG_EDGES, stack)

        # Original workflow unchanged
        assert workflow["3"]["inputs"]["model"] == ["4", 0]
        assert workflow["6"]["inputs"]["clip"] == ["4", 1]

        # New node added
        assert "10" in result
        assert result["10"]["class_type"] == "LoraLoader"
        assert result["10"]["inputs"]["lora_name"] == "test.safetensors"

        # Stack input wired to original source
        assert result["10"]["inputs"]["model"] == ["4", 0]
        assert result["10"]["inputs"]["clip"] == ["4", 1]

        # Consumers rewired to stack output
        assert result["3"]["inputs"]["model"] == ["10", 0]
        assert result["6"]["inputs"]["clip"] == ["10", 1]
        assert result["7"]["inputs"]["clip"] == ["10", 1]

    def test_injection_does_not_mutate_original(self):
        """Original workflow is never mutated."""
        workflow = make_txt2img_workflow()
        original_model = workflow["3"]["inputs"]["model"]

        stack = NodeStack(
            nodes=[
                {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "lora_name": "test.safetensors",
                        "strength_model": 0.8,
                        "clip_strength": 0.8,
                        "model": None,
                        "clip": None,
                    },
                }
            ],
            consumes={"model": (0, "model"), "clip": (0, "clip")},
            produces={"model": (0, 0), "clip": (0, 1)},
        )

        inject(workflow, TXT2IMG_EDGES, stack)
        assert workflow["3"]["inputs"]["model"] == original_model

    def test_chained_lora_injection(self):
        """Two LoRA nodes chained: checkpoint -> lora1 -> lora2 -> sampler."""
        workflow = make_txt2img_workflow()

        stack = build_lora_stack([
            FakeLora(file="face.safetensors", strength=0.8),
            FakeLora(file="style.safetensors", strength=0.5),
        ])

        result = inject(workflow, TXT2IMG_EDGES, stack)

        # Two new nodes
        assert "10" in result
        assert "11" in result
        assert result["10"]["class_type"] == "LoraLoader"
        assert result["11"]["class_type"] == "LoraLoader"

        # First LoRA consumes original edges
        assert result["10"]["inputs"]["model"] == ["4", 0]
        assert result["10"]["inputs"]["clip"] == ["4", 1]

        # Second LoRA consumes first LoRA (internal chain wiring)
        assert result["11"]["inputs"]["model"] == ["10", 0]
        assert result["11"]["inputs"]["clip"] == ["10", 1]

        # Consumers rewired to last LoRA output
        assert result["3"]["inputs"]["model"] == ["11", 0]
        assert result["6"]["inputs"]["clip"] == ["11", 1]
        assert result["7"]["inputs"]["clip"] == ["11", 1]

    def test_three_lora_chain(self):
        """Three LoRA nodes: checkpoint -> lora1 -> lora2 -> lora3 -> sampler."""
        workflow = make_txt2img_workflow()

        stack = build_lora_stack([
            FakeLora(file="face.safetensors", strength=0.8),
            FakeLora(file="style.safetensors", strength=0.5),
            FakeLora(file="pose.safetensors", strength=0.3),
        ])

        result = inject(workflow, TXT2IMG_EDGES, stack)

        # Three new nodes
        assert "10" in result
        assert "11" in result
        assert "12" in result

        # Chain wiring: 4 -> 10 -> 11 -> 12
        assert result["10"]["inputs"]["model"] == ["4", 0]
        assert result["10"]["inputs"]["clip"] == ["4", 1]
        assert result["11"]["inputs"]["model"] == ["10", 0]
        assert result["11"]["inputs"]["clip"] == ["10", 1]
        assert result["12"]["inputs"]["model"] == ["11", 0]
        assert result["12"]["inputs"]["clip"] == ["11", 1]

        # Consumers rewired to last node
        assert result["3"]["inputs"]["model"] == ["12", 0]
        assert result["6"]["inputs"]["clip"] == ["12", 1]
        assert result["7"]["inputs"]["clip"] == ["12", 1]

    def test_injection_at_conditioning_edge(self):
        """Inject at the conditioning edge (for ControlNet later)."""
        workflow = make_txt2img_workflow()

        stack = NodeStack(
            nodes=[
                {
                    "class_type": "ControlNetApply",
                    "inputs": {
                        "conditioning": None,
                        "control_net": "control.safetensors",
                        "strength": 1.0,
                    },
                }
            ],
            consumes={"conditioning": (0, "conditioning")},
            produces={"conditioning": (0, 0)},
        )

        result = inject(workflow, TXT2IMG_EDGES, stack)

        # New node
        assert "10" in result
        assert result["10"]["class_type"] == "ControlNetApply"

        # Wired to CLIPTextEncode output
        assert result["10"]["inputs"]["conditioning"] == ["6", 0]

        # KSampler rewired
        assert result["3"]["inputs"]["positive"] == ["10", 0]

    def test_multiple_injections(self):
        """Inject at two edges sequentially."""
        workflow = make_txt2img_workflow()

        # First: LoRA at model/clip
        lora_stack = build_lora_stack([
            FakeLora(file="face.safetensors", strength=0.8),
        ])
        result = inject(workflow, TXT2IMG_EDGES, lora_stack)

        # Second: something at conditioning
        cond_stack = NodeStack(
            nodes=[
                {
                    "class_type": "ConditioningCombine",
                    "inputs": {"conditioning": None},
                }
            ],
            consumes={"conditioning": (0, "conditioning")},
            produces={"conditioning": (0, 0)},
        )
        result = inject(result, TXT2IMG_EDGES, cond_stack)

        # Both injections present
        assert "10" in result  # LoRA
        assert "11" in result  # ConditioningCombine

    def test_next_node_id_assignment(self):
        """Node IDs are assigned sequentially after existing max."""
        workflow = make_txt2img_workflow()

        stack = NodeStack(
            nodes=[
                {"class_type": "TestNode", "inputs": {"a": None}},
                {"class_type": "TestNode", "inputs": {"b": None}},
            ],
            consumes={"model": (0, "a"), "clip": (0, "b")},
            produces={"model": (1, 0), "clip": (1, 1)},
        )

        result = inject(workflow, TXT2IMG_EDGES, stack)

        assert "10" in result
        assert "11" in result


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestInjectionErrors:
    def test_unknown_consumed_edge(self):
        workflow = make_txt2img_workflow()
        stack = NodeStack(
            nodes=[{"class_type": "X", "inputs": {"a": None}}],
            consumes={"nonexistent": (0, "a")},
            produces={"model": (0, 0)},
        )
        with pytest.raises(InjectionError, match="unknown edge 'nonexistent'"):
            inject(workflow, TXT2IMG_EDGES, stack)

    def test_unknown_produced_edge(self):
        workflow = make_txt2img_workflow()
        stack = NodeStack(
            nodes=[{"class_type": "X", "inputs": {"a": None}}],
            consumes={"model": (0, "a")},
            produces={"nonexistent": (0, 0)},
        )
        with pytest.raises(InjectionError, match="unknown edge 'nonexistent'"):
            inject(workflow, TXT2IMG_EDGES, stack)

    def test_out_of_range_consume_index(self):
        workflow = make_txt2img_workflow()
        stack = NodeStack(
            nodes=[{"class_type": "X", "inputs": {"a": None}}],
            consumes={"model": (5, "a")},
            produces={"model": (0, 0)},
        )
        with pytest.raises(InjectionError, match="node index 5"):
            inject(workflow, TXT2IMG_EDGES, stack)

    def test_out_of_range_produce_index(self):
        workflow = make_txt2img_workflow()
        stack = NodeStack(
            nodes=[{"class_type": "X", "inputs": {"a": None}}],
            consumes={"model": (0, "a")},
            produces={"model": (5, 0)},
        )
        with pytest.raises(InjectionError, match="node index 5"):
            inject(workflow, TXT2IMG_EDGES, stack)

    def test_out_of_range_internal_wire(self):
        workflow = make_txt2img_workflow()
        stack = NodeStack(
            nodes=[{"class_type": "X", "inputs": {"a": None}}],
            consumes={"model": (0, "a")},
            produces={"model": (0, 0)},
            internal_wires=[(0, 0, 5, "b")],
        )
        with pytest.raises(InjectionError, match="dest index 5"):
            inject(workflow, TXT2IMG_EDGES, stack)

    def test_empty_lora_stack_raises(self):
        with pytest.raises(InjectionError, match="empty lora list"):
            build_lora_stack([])


# ---------------------------------------------------------------------------
# LoRA stack builder tests
# ---------------------------------------------------------------------------

class TestBuildLoraStack:
    def test_single_lora(self):
        stack = build_lora_stack([
            FakeLora(file="face.safetensors", strength=0.8),
        ])

        assert len(stack.nodes) == 1
        assert stack.nodes[0]["class_type"] == "LoraLoader"
        assert stack.nodes[0]["inputs"]["lora_name"] == "face.safetensors"
        assert stack.nodes[0]["inputs"]["strength_model"] == 0.8
        assert stack.nodes[0]["inputs"]["strength_clip"] == 0.8

        # Consumes from first node
        assert stack.consumes["model"] == (0, "model")
        assert stack.consumes["clip"] == (0, "clip")

        # Produces from first (and last) node
        assert stack.produces["model"] == (0, 0)
        assert stack.produces["clip"] == (0, 1)

        # No internal wires for single node
        assert stack.internal_wires == []

    def test_chained_lora(self):
        stack = build_lora_stack([
            FakeLora(file="face.safetensors", strength=0.8),
            FakeLora(file="style.safetensors", strength=0.5),
            FakeLora(file="pose.safetensors", strength=0.3),
        ])

        assert len(stack.nodes) == 3

        # All nodes have correct class type and lora_name
        assert stack.nodes[0]["inputs"]["lora_name"] == "face.safetensors"
        assert stack.nodes[1]["inputs"]["lora_name"] == "style.safetensors"
        assert stack.nodes[2]["inputs"]["lora_name"] == "pose.safetensors"

        # Consumes from first node
        assert stack.consumes["model"] == (0, "model")
        assert stack.consumes["clip"] == (0, "clip")

        # Produces from last node
        assert stack.produces["model"] == (2, 0)
        assert stack.produces["clip"] == (2, 1)

        # Internal wires: 0->1, 1->2
        assert len(stack.internal_wires) == 4  # 2 edges * 2 transitions
        # Wire 0->1: model
        assert (0, 0, 1, "model") in stack.internal_wires
        # Wire 0->1: clip
        assert (0, 1, 1, "clip") in stack.internal_wires
        # Wire 1->2: model
        assert (1, 0, 2, "model") in stack.internal_wires
        # Wire 1->2: clip
        assert (1, 1, 2, "clip") in stack.internal_wires


# ---------------------------------------------------------------------------
# Independent clip strength (spc-70)
# ---------------------------------------------------------------------------

class TestLoraClipStrength:
    def test_clip_strength_none_falls_back_to_strength(self):
        """clip_strength=None → strength_clip equals strength_model."""
        stack = build_lora_stack([
            FakeLoraClip(file="face.safetensors", strength=0.8, clip_strength=None),
        ])

        assert stack.nodes[0]["inputs"]["strength_model"] == 0.8
        assert stack.nodes[0]["inputs"]["strength_clip"] == 0.8

    def test_independent_clip_strength(self):
        """clip_strength=0.5 with strength=1.0 → independent values in node."""
        stack = build_lora_stack([
            FakeLoraClip(file="face.safetensors", strength=1.0, clip_strength=0.5),
        ])

        assert stack.nodes[0]["inputs"]["strength_model"] == 1.0
        assert stack.nodes[0]["inputs"]["strength_clip"] == 0.5

    def test_missing_clip_strength_attr_falls_back(self):
        """Duck-typed object lacking clip_strength attr → falls back to strength."""
        # FakeLora has no clip_strength attribute at all.
        lora = FakeLora(file="face.safetensors", strength=0.7)
        assert not hasattr(lora, "clip_strength")

        stack = build_lora_stack([lora])

        assert stack.nodes[0]["inputs"]["strength_model"] == 0.7
        assert stack.nodes[0]["inputs"]["strength_clip"] == 0.7

    def test_chain_preserves_per_lora_clip_strength(self):
        """Each LoRA in a chain keeps its own model/clip strengths."""
        stack = build_lora_stack([
            FakeLoraClip(file="face.safetensors", strength=1.0, clip_strength=0.5),
            FakeLoraClip(file="style.safetensors", strength=0.8, clip_strength=None),
            FakeLoraClip(file="pose.safetensors", strength=0.3, clip_strength=0.9),
        ])

        # First: independent clip
        assert stack.nodes[0]["inputs"]["strength_model"] == 1.0
        assert stack.nodes[0]["inputs"]["strength_clip"] == 0.5

        # Second: clip falls back to strength
        assert stack.nodes[1]["inputs"]["strength_model"] == 0.8
        assert stack.nodes[1]["inputs"]["strength_clip"] == 0.8

        # Third: independent clip
        assert stack.nodes[2]["inputs"]["strength_model"] == 0.3
        assert stack.nodes[2]["inputs"]["strength_clip"] == 0.9


# ---------------------------------------------------------------------------
# Edge discovery tests
# ---------------------------------------------------------------------------

class TestDiscoverEdges:
    def test_discover_txt2img_edges(self):
        workflow = make_txt2img_workflow()
        edges = discover_edges(workflow)

        assert "model" in edges
        assert "clip" in edges
        assert "vae" in edges
        assert "conditioning" in edges

    def test_model_edge_topology(self):
        workflow = make_txt2img_workflow()
        edges = discover_edges(workflow)

        model_edge = edges["model"]
        assert model_edge.source.node_id == "4"
        assert model_edge.source.output_index == 0
        assert len(model_edge.consumers) == 1
        assert model_edge.consumers[0].node_id == "3"
        assert model_edge.consumers[0].input_key == "model"

    def test_clip_edge_topology(self):
        workflow = make_txt2img_workflow()
        edges = discover_edges(workflow)

        clip_edge = edges["clip"]
        assert clip_edge.source.node_id == "4"
        assert clip_edge.source.output_index == 1
        assert len(clip_edge.consumers) == 2

    def test_conditioning_edge_topology(self):
        """Both CLIPTextEncode nodes (6=positive, 7=negative) produce conditioning edges.

        discover_edges groups by convention name, so both CLIPTextEncode outputs
        map to "conditioning". The edge has two sources' consumers merged.
        """
        workflow = make_txt2img_workflow()
        edges = discover_edges(workflow)

        # The conditioning edge is discovered from node 6 (first CLIPTextEncode
        # encountered). Node 7 also produces conditioning but since discover_edges
        # scans all nodes, both contribute consumers to the same edge name.
        cond_edge = edges["conditioning"]
        assert cond_edge.source.output_index == 0
        # Node 3 consumes conditioning on both "positive" and "negative" inputs
        assert len(cond_edge.consumers) == 2
        consumer_keys = {c.input_key for c in cond_edge.consumers}
        assert consumer_keys == {"positive", "negative"}

    def test_unknown_class_type_ignored(self):
        """Edges from unknown class types are not discovered."""
        workflow = make_txt2img_workflow()
        workflow["99"] = {
            "class_type": "CustomNode",
            "inputs": {"data": ["4", 0]},
        }
        edges = discover_edges(workflow)

        # Only known conventions are discovered
        assert "model" in edges
        # The custom node's consumption of output 0 from node 4
        # should be included as a consumer of the model edge
        model_consumers = [c for c in edges["model"].consumers if c.node_id == "99"]
        assert len(model_consumers) == 1

    def test_empty_workflow(self):
        edges = discover_edges({})
        assert edges == {}
