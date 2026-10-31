"""Web search tool — query a SearXNG instance."""
from __future__ import annotations

from urllib.parse import urlencode

import aiohttp


class WebSearchTool:
    """Search the web via a SearXNG JSON API."""

    name = "web_search"
    description = "Search the web using a SearXNG instance and return formatted results."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, base_url: str, max_results: int = 5) -> None:
        self.base_url = base_url
        self.max_results = max_results

    async def __call__(self, query: str) -> str:
        params = urlencode({"q": query, "format": "json"})
        url = f"{self.base_url}/search?{params}"

        async with aiohttp.ClientSession() as session:
            response = session.get(url)
            async with response as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Search request failed with HTTP {resp.status}"
                    )
                data = await resp.json()

        results = data.get("results", [])[: self.max_results]
        if not results:
            return "No results found."

        lines = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "")
            url_str = item.get("url", "")
            content = item.get("content", "")
            lines.append(f"[{i}] {title}\n{url_str}\n{content}")

        return "\n\n".join(lines)
