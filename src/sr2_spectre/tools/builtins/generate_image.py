"""Image generation tool — wraps ComfyUI for SDXL text-to-image.

The agent provides an **intent** (natural-language description of what the
character wants to convey). The tool compiles that intent into a proper
SDXL prompt by combining it with checkpoint-specific scaffolding, style
presets, and character consistency tokens.

The **negative prompt** is tool-owned per checkpoint — the agent does not
set it. A rare override remains available internally.

Supports two modes:
- **text2img** — intent-only generation (default, simplest)
- **photomaker** — character-consistent generation using a reference face

The LLM calls this tool when a character wants to send an image.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from sr2_spectre.tools.builtins.comfyui_client import ComfyUIClient, ComfyUIError

logger = logging.getLogger(__name__)


class GenerateImageTool:
    """Generate images via ComfyUI.

    The agent sends an intent (what the character wants to convey).
    The tool compiles it into a full SDXL prompt using checkpoint-owned
    scaffolding, scenario presets, and character style tokens.

    Config via constructor (passed from character YAML):
        comfyui_url: str          — e.g. "http://192.168.50.233:8188"
        checkpoint: str           — checkpoint filename in models/ (e.g. "DreamShaperXL.safetensors")
        reference_image: str      — path to character face reference (for photomaker mode)
        style_prompt: str         — appended to every prompt for consistency
        negative_prompt: str      — tool-owned negative per checkpoint (rare override)
        width: int                — default 1024
        height: int               — default 1024
        steps: int                — default 28
        cfg: float                — default 7.0
        output_dir: str           — temp dir for generated images
    """

    name = "generate_image"
    description = (
        "Generate an image from a natural-language intent describing what the "
        "character wants to convey. The intent is not a literal SDXL prompt — "
        "the tool compiles it with checkpoint-specific scaffolding, scenario "
        "presets, and character consistency tokens. The negative prompt is "
        "managed by the tool per checkpoint and is not set by the agent. "
        "Returns the path to the generated image file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": (
                    "Natural-language intent describing what the character wants "
                    "to convey — what they are doing, their expression, the setting, "
                    "and mood. This is not a literal SDXL prompt; the tool compiles "
                    "it with checkpoint-specific scaffolding. "
                    "Example: 'Vexa standing on a rainy rooftop at night, neon signs "
                    "reflecting in puddles, cyberpunk aesthetic, cinematic lighting'"
                ),
            },
            "scenario": {
                "type": "string",
                "description": (
                    "Scenario preset that frames the composition. Options: 'selfie', "
                    "'portrait', 'scene', 'illustration'. If omitted, uses character default."
                ),
            },
        },
        "required": ["intent"],
    }

    # Default negative prompt — good for most SDXL character work
    DEFAULT_NEGATIVE = (
        "low quality, blurry, ugly, deformed, bad anatomy, disfigured, "
        "bad proportions, extra limbs, poorly drawn face, poorly drawn hands, "
        "watermark, text, signature, jpeg artifacts"
    )

    # Scenario presets — modify how the prompt is framed
    SCENARIO_PRESETS = {
        "selfie": (
            "selfie style, close-up portrait, casual angle, looking at camera, "
            "shallow depth of field, natural lighting"
        ),
        "portrait": (
            "professional portrait, studio quality, detailed face, dramatic lighting, "
            "sharp focus, high resolution"
        ),
        "scene": (
            "full body shot, dynamic pose, environmental storytelling, "
            "cinematic composition, atmospheric lighting"
        ),
        "illustration": (
            "digital illustration, stylized, clean lines, vibrant colors, "
            "artstation quality, concept art style"
        ),
    }

    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        checkpoint: str = "SDXL\\dreamshaperXL_alpha2Xl10.safetensors",
        reference_image: Optional[str] = None,
        style_prompt: str = "",
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        cfg: float = 7.0,
        seed: int = 0,
        output_dir: str = "/tmp/spectre_images",
    ) -> None:
        self.client = ComfyUIClient(
            base_url=comfyui_url,
            max_poll_time=600.0,
            poll_interval=3.0,
        )
        self.checkpoint = checkpoint
        self.reference_image = reference_image
        self.style_prompt = style_prompt
        self.negative_prompt = negative_prompt  # tool-owned; None → DEFAULT_NEGATIVE
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg = cfg
        self.seed = seed  # 0 = random each time
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- workflow builders --

    def _build_text2img_workflow(
        self,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> dict:
        """Build a basic SDXL text-to-image workflow."""
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": self.steps,
                    "cfg": self.cfg,
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
                "inputs": {
                    "ckpt_name": self.checkpoint,
                },
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": self.width,
                    "height": self.height,
                    "batch_size": 1,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["4", 1],
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1],
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "spectre",
                    "images": ["8", 0],
                },
            },
        }

    # -- prompt assembly --

    def _assemble_prompt(
        self,
        intent: str,
        scenario: Optional[str],
    ) -> str:
        """Combine intent, scenario preset, and character style prompt."""
        parts = [intent]

        # Add scenario preset if specified
        if scenario and scenario in self.SCENARIO_PRESETS:
            parts.append(self.SCENARIO_PRESETS[scenario])

        # Always add character style prompt for consistency
        if self.style_prompt:
            parts.append(self.style_prompt)

        return ", ".join(parts)

    # -- execution --

    async def __call__(
        self,
        intent: str,
        scenario: Optional[str] = None,
    ) -> str:
        """Generate an image and return the path to the saved file.

        Args:
            intent: Natural-language intent (not a literal SDXL prompt).
            scenario: Optional scenario preset ('selfie', 'portrait', 'scene',
                      'illustration').

        Returns a string with the file path on success, or an error message.
        """
        try:
            # Check ComfyUI is alive
            if not await self.client.is_available():
                return "Error: ComfyUI is not reachable. The image generation service is offline."

            # Assemble the positive prompt from intent
            positive = self._assemble_prompt(intent, scenario)

            # Negative prompt is tool-owned per checkpoint
            neg = self.negative_prompt if self.negative_prompt else self.DEFAULT_NEGATIVE

            # Pick a seed (0 = random)
            import random
            seed = random.randint(1, 2**32 - 1) if self.seed == 0 else self.seed

            # Build and submit workflow
            workflow = self._build_text2img_workflow(positive, neg, seed)
            logger.info(
                "Submitting image generation: intent=%s, seed=%d, model=%s",
                intent[:80], seed, self.checkpoint,
            )

            image_path = await self.client.generate(
                workflow, self.output_dir, prefix="spectre"
            )

            return str(image_path)

        except ComfyUIError as exc:
            logger.error("ComfyUI error during image generation: %s", exc)
            return f"Image generation failed: {exc}"
        except TimeoutError:
            logger.error("Image generation timed out")
            return "Image generation timed out. Try again with a simpler intent."
        except Exception as exc:
            logger.exception("Unexpected error during image generation")
            return f"Image generation error: {exc}"
