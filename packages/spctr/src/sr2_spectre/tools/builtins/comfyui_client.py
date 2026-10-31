"""Async ComfyUI REST API client — submit workflows, poll, download outputs.

Self-contained. Uses aiohttp (already a Spectre dep). No external requests.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import asyncio
import aiohttp


class ComfyUIError(Exception):
    """Error from ComfyUI execution."""


@dataclass(frozen=True)
class ImageRef:
    """Reference to an image stored in ComfyUI's output."""
    filename: str
    subfolder: str = ""
    output_type: str = "output"

    def __repr__(self) -> str:
        return f"ImageRef({self.filename!r}, subfolder={self.subfolder!r})"


class ComfyUIClient:
    """Async client for ComfyUI's REST API.

    Args:
        base_url: Full ComfyUI URL, e.g. "http://192.168.50.233:8188"
        timeout: Request timeout in seconds (per-operation, not polling).
        max_poll_time: Max seconds to wait for a prompt to complete.
        poll_interval: Seconds between history polls.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        timeout: float = 60.0,
        max_poll_time: float = 600.0,
        poll_interval: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_poll_time = max_poll_time
        self.poll_interval = poll_interval

    # -- internal helpers --

    def _url(self, path: str) -> str:
        return urljoin(self.base_url + "/", path.lstrip("/"))

    async def _get(self, path: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self._url(path), timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _post(self, path: str, payload: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url(path),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    # -- health --

    async def is_available(self) -> bool:
        """Check if ComfyUI server is reachable."""
        try:
            data = await self._get("/system_stats")
            return "system" in data
        except Exception:
            return False

    # -- health (legacy: also accept /prompt as fallback) --

    async def ping(self) -> bool:
        """Lightweight ping. Returns True if the API responds."""
        return await self.is_available()

    # -- queue / prompt --

    async def submit_workflow(self, workflow: dict) -> str:
        """Submit a workflow dict to the prompt queue. Returns prompt_id."""
        client_id = str(uuid.uuid4())
        payload = {
            "prompt": workflow,
            "client_id": client_id,
        }
        data = await self._post("/prompt", payload)
        return data["prompt_id"]

    # -- history --

    async def get_history(self, prompt_id: str) -> Optional[dict]:
        """Get execution history for a prompt_id. None if not yet done."""
        data = await self._get(f"/history/{prompt_id}")
        return data.get(prompt_id)

    # -- polling --

    async def wait_for_completion(self, prompt_id: str) -> dict:
        """Poll until the prompt completes. Returns the history entry."""
        start = time.monotonic()
        while (time.monotonic() - start) < self.max_poll_time:
            history = await self.get_history(prompt_id)
            if history is not None:
                status = history.get("status", {})
                if status.get("completed", False):
                    return history
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise ComfyUIError(f"ComfyUI execution failed: {messages}")
            await asyncio.sleep(self.poll_interval)
        raise TimeoutError(
            f"ComfyUI prompt {prompt_id} did not complete within {self.max_poll_time}s"
        )

    # -- output retrieval --

    @staticmethod
    def extract_images(history: dict) -> list[ImageRef]:
        """Extract ImageRef list from a completed history entry."""
        images: list[ImageRef] = []
        outputs = history.get("outputs", {})
        for _node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                images.append(ImageRef(
                    filename=img["filename"],
                    subfolder=img.get("subfolder", ""),
                    output_type=img.get("type", "output"),
                ))
        return images

    async def download_image(self, image_ref: ImageRef, dest: Path) -> Path:
        """Download a single image from ComfyUI to dest path."""
        params = {
            "filename": image_ref.filename,
            "subfolder": image_ref.subfolder,
            "type": image_ref.output_type,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self._url("/view"),
                params=params,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        f.write(chunk)
        return dest

    async def download_first_output(
        self, history: dict, output_dir: Path, prefix: str = "img"
    ) -> Path:
        """Download the first output image from history. Returns the path."""
        images = self.extract_images(history)
        if not images:
            raise ComfyUIError("No output images found in history")
        img_ref = images[0]
        suffix = Path(img_ref.filename).suffix or ".png"
        dest = output_dir / f"{prefix}{suffix}"
        return await self.download_image(img_ref, dest)

    # -- convenience: full generate cycle --

    async def generate(
        self, workflow: dict, output_dir: Path, prefix: str = "img"
    ) -> Path:
        """Submit a workflow, wait for completion, download first image.

        Returns the path to the downloaded image file.
        """
        prompt_id = await self.submit_workflow(workflow)
        history = await self.wait_for_completion(prompt_id)
        return await self.download_first_output(history, output_dir, prefix=prefix)
