"""Beads tools — issue tracking through a host-side shim.

The beads database for a repository is an embedded Dolt DB under `.beads/`,
reachable only by running the `bd` binary against that repository, and bd has
no daemon or server mode. An agent that runs in a container therefore cannot
reach it directly, and mounting the repository in so it could would hand the
container write access to the working tree — which `code_exec` would inherit,
and which risks two bd processes competing over one Dolt DB.

So the host runs a small shim that owns the `bd` calls and exposes a typed
HTTP API, and these tools are its client. Validation of everything the model
supplies happens in the shim, next to the subprocess it protects; these tools
keep the surface small and the responses trimmed.

Five tools rather than one per bd subcommand: the tool schemas are re-sent on
every turn, and a broad surface is exactly what made the official GitHub MCP
server unusable here.
"""
from __future__ import annotations

import json
import os
from typing import Any

import aiohttp


class BeadsError(RuntimeError):
    """Raised when the beads shim rejects a request or is unreachable."""


def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{clipped}\n… truncated at {max_bytes} bytes."


class _BeadsTool:
    """Shared transport for the beads tools."""

    endpoint = ""

    def __init__(
        self,
        base_url: str,
        token_env: str = "GRINDFORGE_BEADS_TOKEN",
        timeout: int = 30,
        max_bytes: int = 8000,
    ) -> None:
        """
        Args:
            base_url: Root URL of the beads shim.
            token_env: Environment variable holding the shim's bearer token.
                Env rather than a file: CodeExecTool scrubs its child's
                environment but shares the filesystem.
            timeout: Per-request timeout in seconds.
            max_bytes: Cap on the text handed back to the model.
        """
        self.base_url = base_url.rstrip("/")
        self.token_env = token_env
        self.timeout = timeout
        self.max_bytes = max_bytes

    async def _post(self, payload: dict[str, Any]) -> str:
        token = os.environ.get(self.token_env, "")
        if not token:
            raise BeadsError(
                f"No beads token: ${self.token_env} is unset in this process."
            )
        # Drop keys the model left empty so the shim sees a clean request.
        body = {k: v for k, v in payload.items() if v is not None}

        url = f"{self.base_url}{self.endpoint}"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    url, json=body, headers={"Authorization": f"Bearer {token}"}
                ) as resp:
                    text = await resp.text()
                    if resp.status == 401:
                        raise BeadsError("The beads shim rejected this agent's token.")
                    if resp.status >= 500:
                        raise BeadsError(f"The beads shim failed (HTTP {resp.status}).")
                    data = json.loads(text)
        except aiohttp.ClientError as exc:
            raise BeadsError(f"Could not reach the beads shim: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BeadsError("The beads shim returned malformed output.") from exc

        if not data.get("ok"):
            detail = data.get("error") or data.get("output") or "unknown error"
            return f"Beads rejected that: {_truncate(str(detail), 1000)}"
        return _truncate(data.get("output") or "(no output)", self.max_bytes)


class BeadsQueryTool(_BeadsTool):
    """Read the issue tracker."""

    name = "beads_query"
    endpoint = "/beads/query"
    description = (
        "Query the studio's issue tracker (beads). Use op=list for open issues, "
        "ready for unblocked work, blocked, show for one issue by id, search by "
        "text, or stats for a summary."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["list", "ready", "blocked", "show", "search", "stats"],
                "description": "Defaults to list.",
            },
            "id": {"type": "string", "description": "Issue id, for op=show."},
            "query": {"type": "string", "description": "Search text, for op=search."},
            "status": {
                "type": "string",
                "enum": ["open", "in_progress", "blocked", "closed", "deferred"],
            },
            "priority": {"type": "string", "description": "0-4, 0 is highest."},
            "type": {
                "type": "string",
                "enum": ["bug", "feature", "task", "epic", "chore", "decision"],
            },
            "labels": {"type": "string", "description": "Comma-separated label filter."},
            "assignee": {"type": "string"},
        },
        "required": [],
    }

    async def __call__(
        self,
        op: str = "list",
        id: str | None = None,
        query: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        type: str | None = None,
        labels: str | None = None,
        assignee: str | None = None,
    ) -> str:
        return await self._post({
            "op": op, "id": id, "query": query, "status": status,
            "priority": priority, "type": type, "labels": labels,
            "assignee": assignee,
        })


class BeadsCreateTool(_BeadsTool):
    """File a new issue."""

    name = "beads_create"
    endpoint = "/beads/create"
    description = "File a new issue in the studio's issue tracker (beads)."
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "One-line summary."},
            "description": {"type": "string"},
            "acceptance": {"type": "string", "description": "Acceptance criteria."},
            "type": {
                "type": "string",
                "enum": ["bug", "feature", "task", "epic", "chore", "decision"],
            },
            "priority": {"type": "string", "description": "0-4, 0 is highest."},
            "labels": {"type": "string", "description": "Comma-separated."},
            "assignee": {"type": "string"},
        },
        "required": ["title"],
    }

    async def __call__(
        self,
        title: str,
        description: str | None = None,
        acceptance: str | None = None,
        type: str | None = None,
        priority: str | None = None,
        labels: str | None = None,
        assignee: str | None = None,
    ) -> str:
        return await self._post({
            "title": title, "description": description, "acceptance": acceptance,
            "type": type, "priority": priority, "labels": labels,
            "assignee": assignee,
        })


class BeadsUpdateTool(_BeadsTool):
    """Change an existing issue, including closing it."""

    name = "beads_update"
    endpoint = "/beads/update"
    description = (
        "Update an issue in the studio's issue tracker (beads). Set status=closed "
        "to close it. Pass only the fields you are changing."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Issue id."},
            "status": {
                "type": "string",
                "enum": ["open", "in_progress", "blocked", "closed", "deferred"],
            },
            "priority": {"type": "string", "description": "0-4, 0 is highest."},
            "type": {
                "type": "string",
                "enum": ["bug", "feature", "task", "epic", "chore", "decision"],
            },
            "title": {"type": "string"},
            "description": {"type": "string"},
            "acceptance": {"type": "string"},
            "notes": {"type": "string", "description": "Appended to existing notes."},
            "assignee": {"type": "string"},
            "add_labels": {"type": "string", "description": "Comma-separated."},
            "remove_labels": {"type": "string", "description": "Comma-separated."},
        },
        "required": ["id"],
    }

    async def __call__(
        self,
        id: str,
        status: str | None = None,
        priority: str | None = None,
        type: str | None = None,
        title: str | None = None,
        description: str | None = None,
        acceptance: str | None = None,
        notes: str | None = None,
        assignee: str | None = None,
        add_labels: str | None = None,
        remove_labels: str | None = None,
    ) -> str:
        return await self._post({
            "id": id, "status": status, "priority": priority, "type": type,
            "title": title, "description": description, "acceptance": acceptance,
            "notes": notes, "assignee": assignee,
            "add_labels": add_labels, "remove_labels": remove_labels,
        })


class BeadsCommentTool(_BeadsTool):
    """Comment on an issue."""

    name = "beads_comment"
    endpoint = "/beads/comment"
    description = "Add a comment to an issue in the studio's issue tracker (beads)."
    input_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Issue id."},
            "body": {"type": "string", "description": "Comment text."},
        },
        "required": ["id", "body"],
    }

    async def __call__(self, id: str, body: str) -> str:
        return await self._post({"id": id, "body": body})


class BeadsDepTool(_BeadsTool):
    """Link or unlink issue dependencies."""

    name = "beads_dep"
    endpoint = "/beads/dep"
    description = (
        "Link or unlink a dependency between two issues in the studio's issue "
        "tracker (beads)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "from": {"type": "string", "description": "Source issue id."},
            "to": {"type": "string", "description": "Target issue id."},
            "op": {"type": "string", "enum": ["link", "unlink"], "description": "Defaults to link."},
            "type": {
                "type": "string",
                "enum": ["blocks", "related", "parent-child", "discovered-from"],
                "description": "Defaults to blocks.",
            },
        },
        "required": ["from", "to"],
    }

    async def __call__(
        self,
        op: str = "link",
        type: str = "blocks",
        **kwargs: Any,
    ) -> str:
        # "from" is a Python keyword, so it cannot be a named parameter.
        payload = {
            "op": op,
            "type": type,
            "from": kwargs.get("from"),
            "to": kwargs.get("to"),
        }
        if not payload["from"] or not payload["to"]:
            return "beads_dep needs both 'from' and 'to' issue ids."
        return await self._post(payload)
