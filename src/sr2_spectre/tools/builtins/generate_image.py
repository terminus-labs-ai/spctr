"""Image generation tool — wraps ComfyUI for SDXL text-to-image.

Supports two modes:
- **text2img** — prompt-only generation (default, simplest)
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

    Config via constructor (passed from character YAML):
        comfyui_url: str          — e.g. "http://192.168.50.233:8188"
        checkpoint: str           — checkpoint filename in models/ (e.g. "DreamShaperXL.safetensors")
        reference_image: str      — path to character face reference (for photomaker mode)
        style_prompt: str         — appended to every prompt for consistency
        width: int                — default 1024
        height: int               — default 1024
        steps: int                — default 28
        cfg: float                — default 7.0
        output_dir: str           — temp dir for generated images
    """

    name = "generate_image"
    description = (
        "Generate an image based on a prompt. The image reflects the character's "
        "visual style and can depict scenes, selfies, or illustrations described "
        "in the prompt. Returns the path to the generated image file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Descriptive prompt for the image. Be specific about what the "
                    "character is doing, their expression, the setting, and mood. "
                    "Example: 'Vexa standing on a rainy rooftop at night, neon signs "
                    "reflecting in puddles, cyberpunk aesthetic, cinematic lighting'"
                ),
            },
            "negative_prompt": {
                "type": "string",
                "description": (
                    "Things to avoid in the image. Optional — defaults to common "
                    "artifacts filter."
                ),
            },
            "style": {
                "type": "string",
                "description": (
                    "Override the character's default style. Options: 'selfie', "
                    "'portrait', 'scene', 'illustration'. If omitted, uses character default."
                ),
            },
        },
        "required": ["prompt"],
    }

    # Default negative prompt — good for most SDXL character work
    DEFAULT_NEGATIVE = (
        "low quality, blurry, ugly, deformed, bad anatomy, disfigured, "
        "bad proportions, extra limbs, poorly drawn face, poorly drawn hands, "
        "watermark, text, signature, jpeg artifacts"
    )

    # Style presets — modify how the prompt is framed
    STYLE_PRESETS = {
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
        user_prompt: str,
        style: Optional[str],
    ) -> str:
        """Combine user prompt, style preset, and character style prompt."""
        parts = [user_prompt]

        # Add style preset if specified
        if style and style in self.STYLE_PRESETS:
            parts.append(self.STYLE_PRESETS[style])

        # Always add character style prompt for consistency
        if self.style_prompt:
            parts.append(self.style_prompt)

        return ", ".join(parts)

    # -- execution --

    async def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
    ) -> str:
        """Generate an image and return the path to the saved file.

        Returns a string with the file path on success, or an error message.
        """
        try:
            # Check ComfyUI is alive
            if not await self.client.is_available():
                return "Error: ComfyUI is not reachable. The image generation service is offline."

            # Assemble the positive prompt
            positive = self._assemble_prompt(prompt, style)

            # Use default negative if none provided
            neg = negative_prompt if negative_prompt else self.DEFAULT_NEGATIVE

            # Pick a seed (0 = random)
            import random
            seed = random.randint(1, 2**32 - 1) if self.seed == 0 else self.seed

            # Build and submit workflow
            workflow = self._build_text2img_workflow(positive, neg, seed)
            logger.info(
                "Submitting image generation: prompt=%s, seed=%d, model=%s",
                positive[:80], seed, self.checkpoint,
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
            return "Image generation timed out. Try again with a simpler prompt."
        except Exception as exc:
            logger.exception("Unexpected error during image generation")
            return f"Image generation error: {exc}"
