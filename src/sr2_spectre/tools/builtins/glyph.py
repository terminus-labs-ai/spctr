"""Glyph search — the engine reference, through a host-side shim.

Glyph is a RAG knowledge base whose corpus includes the Godot class reference
and the engine documentation, chunked one entry per class, method, property and
signal. That is the material a small model most reliably invents: an agent that
half-remembers a signal name writes code that parses and then does nothing.

The tool talks to a shim rather than to Glyph directly, for one reason: Glyph's
own `search` takes the corpus name as an argument, and the same database holds
private sources that have no business in a shared Discord server. The shim
pins the corpus to an allowlist, pins its version, and caps the result size —
server-side of the container boundary, where a snippet cannot reach it.

One tool, two arguments. The registered tool schemas are re-sent on every turn,
and this agent runs on a 32k context; a broad surface is what made the official
GitHub MCP server unusable here.
"""
from __future__ import annotations

import json
import os

import aiohttp


class GlyphError(RuntimeError):
    """Raised when the Glyph shim rejects a request or is unreachable."""


def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{clipped}\n… truncated at {max_bytes} bytes."


class GlyphSearchTool:
    """Search the pinned engine-reference corpora."""

    name = "glyph_search"
    description = (
        "Search the Godot 4.7 engine reference: class reference (godot-api), "
        "documentation and tutorials (godot-docs), or shader examples "
        "(godot-shaders). Use it to check a real class, method, property or "
        "signal before writing code that calls it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up, in plain words.",
            },
            "source": {
                "type": "string",
                "enum": ["godot-api", "godot-docs", "godot-shaders"],
                "description": "Which corpus. Defaults to godot-api.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        base_url: str,
        token_env: str = "GRINDFORGE_GLYPH_TOKEN",
        timeout: int = 90,
        max_bytes: int = 6000,
    ) -> None:
        """
        Args:
            base_url: Root URL of the Glyph shim.
            token_env: Environment variable holding the shim's bearer token.
                Env rather than a file: CodeExecTool scrubs its child's
                environment but shares the filesystem.
            timeout: Per-request timeout in seconds. Generous on purpose — a
                reranked search runs a second model pass over the candidates,
                and a cold model load costs more than the first result does.
            max_bytes: Cap on the text handed back to the model.
        """
        self.base_url = base_url.rstrip("/")
        self.token_env = token_env
        self.timeout = timeout
        self.max_bytes = max_bytes

    async def __call__(self, query: str, source: str | None = None) -> str:
        token = os.environ.get(self.token_env, "")
        if not token:
            raise GlyphError(
                f"No Glyph token: ${self.token_env} is unset in this process."
            )
        body = {"query": query}
        if source:
            body["source"] = source

        url = f"{self.base_url}/glyph/search"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    url, json=body, headers={"Authorization": f"Bearer {token}"}
                ) as resp:
                    text = await resp.text()
                    if resp.status == 401:
                        raise GlyphError("The Glyph shim rejected this agent's token.")
                    if resp.status >= 500:
                        raise GlyphError(f"The Glyph shim failed (HTTP {resp.status}).")
                    data = json.loads(text)
        except aiohttp.ClientError as exc:
            raise GlyphError(f"Could not reach the Glyph shim: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise GlyphError("The Glyph shim returned malformed output.") from exc

        if not data.get("ok"):
            detail = data.get("error") or "unknown error"
            # Deliberately not an exception: an unavailable knowledge base is
            # something to report and move past, not a reason for the model to
            # retry the same lookup until it runs out of iterations.
            return f"Glyph search is unavailable: {_truncate(str(detail), 500)}"
        return _truncate(data.get("output") or "(no results)", self.max_bytes)
