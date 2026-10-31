"""GitHub tools — a narrow, repo-pinned surface over the REST API.

Why not the official MCP server: with ``--toolsets=context,repos,issues,
pull_requests`` it advertises 41 tools whose JSON schemas total ~114KB, about
30,900 tokens. Against a 32k budget on a local 27B that is the whole context
window spent before the first message, and its responses are full API JSON,
which then blows out the tool-result budget too. These tools trade breadth for
a surface a small model can actually hold, and return trimmed text rather than
raw JSON.

The repository is pinned in configuration, not passed by the model. Two
reasons: it keeps ``owner``/``repo`` out of every schema, and it means the
agent cannot reach another repository even if its token's scope were widened
later.

The token is read from an environment variable rather than a file on purpose.
``CodeExecTool`` scrubs its child's environment but shares the filesystem, so a
snippet can read a mounted secret and cannot read an env var. For anything the
agent process holds, env is the more protected channel of the two.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any
from urllib.parse import quote, urlencode

import aiohttp

_API_ROOT = "https://api.github.com"
_ACCEPT = "application/vnd.github+json"
_API_VERSION = "2022-11-28"


class GitHubError(RuntimeError):
    """Raised when the GitHub API rejects a request."""


class _RepoClient:
    """Minimal authenticated client bound to a single repository."""

    def __init__(self, repo: str, token: str, timeout: int) -> None:
        self.repo = repo
        self._token = token
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": _ACCEPT,
            "X-GitHub-Api-Version": _API_VERSION,
            "User-Agent": "sr2-spectre-github-tool",
        }

    async def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        """Call the API and return the decoded JSON body.

        ``path`` is appended to the API root as given, so callers are
        responsible for quoting anything model-supplied.
        """
        url = f"{_API_ROOT}{path}"
        if params:
            url = f"{url}?{urlencode({k: v for k, v in params.items() if v is not None})}"

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.request(
                method, url, headers=self._headers, json=body
            ) as resp:
                text = await resp.text()
                if resp.status == 404:
                    raise GitHubError("Not found.")
                if resp.status == 403 and "rate limit" in text.lower():
                    raise GitHubError("GitHub rate limit reached; try again later.")
                if resp.status in (401, 403):
                    # Name the missing permission. A fine-grained PAT grants
                    # each resource separately, so "refused" without saying
                    # which one leaves the model guessing and retrying.
                    resource = _resource_for(path)
                    raise GitHubError(
                        f"GitHub refused this request (HTTP {resp.status}). The "
                        f"token does not grant {resource} access to this "
                        f"repository. This is a token permission to be fixed by "
                        f"a human, not something to retry or work around."
                    )
                if resp.status >= 400:
                    raise GitHubError(f"GitHub returned HTTP {resp.status}.")
                if not text:
                    return None
                return json.loads(text)


def _resource_for(path: str) -> str:
    """Name the fine-grained PAT permission an API path needs."""
    if "/issues" in path:
        return "Issues (Read and write)"
    if "/pulls" in path:
        return "Pull requests (Read and write)"
    if "/contents" in path or "/git/" in path:
        return "Contents (Read and write)"
    return "the required"


def _truncate(text: str, max_bytes: int) -> str:
    """Cut *text* to *max_bytes*, flagging that it was cut.

    Tool results are re-sent on every subsequent turn, so an untrimmed file or
    issue list is paid for repeatedly, not once.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{clipped}\n… truncated at {max_bytes} bytes."


class _RepoTool:
    """Shared construction for the repo-pinned tools."""

    def __init__(
        self,
        repo: str,
        token_env: str = "GRINDFORGE_GITHUB_TOKEN",
        timeout: int = 20,
        max_bytes: int = 8000,
    ) -> None:
        """
        Args:
            repo: "owner/name". Pinned here so the model never supplies it.
            token_env: Environment variable holding the token.
            timeout: Per-request timeout in seconds.
            max_bytes: Cap on the text handed back to the model.
        """
        if "/" not in repo:
            raise ValueError(f"repo must be 'owner/name', got {repo!r}")
        self.repo = repo
        self.token_env = token_env
        self.timeout = timeout
        self.max_bytes = max_bytes

    @property
    def _client(self) -> _RepoClient:
        token = os.environ.get(self.token_env, "")
        if not token:
            raise GitHubError(
                f"No GitHub token: ${self.token_env} is unset in this process."
            )
        return _RepoClient(self.repo, token, self.timeout)

    def _path(self, suffix: str) -> str:
        return f"/repos/{self.repo}{suffix}"


class GitHubReadFileTool(_RepoTool):
    """Read one file from the pinned repository."""

    name = "github_read_file"
    description = "Read the contents of a file from the studio's GitHub repository."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path within the repo."},
            "ref": {"type": "string", "description": "Branch, tag or commit. Optional."},
        },
        "required": ["path"],
    }

    async def __call__(self, path: str, ref: str | None = None) -> str:
        data = await self._client.request(
            "GET", self._path(f"/contents/{quote(path.lstrip('/'))}"), params={"ref": ref}
        )
        if isinstance(data, list):
            return f"{path} is a directory. Use github_list_files."
        if data.get("encoding") != "base64":
            return f"{path} is not a text file."
        try:
            body = base64.b64decode(data["content"]).decode("utf-8")
        except UnicodeDecodeError:
            return f"{path} is a binary file ({data.get('size', 0)} bytes)."
        return _truncate(f"{path}:\n{body}", self.max_bytes)


class GitHubListFilesTool(_RepoTool):
    """List a directory in the pinned repository."""

    name = "github_list_files"
    description = "List files and directories at a path in the studio's GitHub repository."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path. Omit for the root."},
            "ref": {"type": "string", "description": "Branch, tag or commit. Optional."},
        },
        "required": [],
    }

    async def __call__(self, path: str = "", ref: str | None = None) -> str:
        data = await self._client.request(
            "GET", self._path(f"/contents/{quote(path.strip('/'))}"), params={"ref": ref}
        )
        if not isinstance(data, list):
            return f"{path} is a file. Use github_read_file."
        lines = [
            f"{'dir ' if e['type'] == 'dir' else 'file'}  {e['name']}"
            for e in sorted(data, key=lambda e: (e["type"] != "dir", e["name"]))
        ]
        header = f"{path or '/'} ({len(lines)} entries):"
        return _truncate("\n".join([header, *lines]), self.max_bytes)


class GitHubSearchCodeTool(_RepoTool):
    """Search code within the pinned repository."""

    name = "github_search_code"
    description = "Search for code in the studio's GitHub repository."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms."},
        },
        "required": ["query"],
    }

    async def __call__(self, query: str) -> str:
        data = await self._client.request(
            "GET", "/search/code", params={"q": f"{query} repo:{self.repo}", "per_page": 20}
        )
        items = data.get("items", [])
        if not items:
            return f"No code matches for {query!r}."
        lines = [f"{len(items)} match(es) for {query!r}:"]
        lines += [f"  {i['path']}" for i in items]
        return _truncate("\n".join(lines), self.max_bytes)


class GitHubListIssuesTool(_RepoTool):
    """List issues, excluding pull requests."""

    name = "github_list_issues"
    description = "List issues in the studio's GitHub repository."
    input_schema = {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "enum": ["open", "closed", "all"],
                "description": "Defaults to open.",
            },
            "labels": {"type": "string", "description": "Comma-separated label filter."},
        },
        "required": [],
    }

    async def __call__(self, state: str = "open", labels: str | None = None) -> str:
        data = await self._client.request(
            "GET",
            self._path("/issues"),
            params={"state": state, "labels": labels, "per_page": 30},
        )
        # The issues endpoint returns PRs too; they are a different workflow.
        issues = [i for i in data if "pull_request" not in i]
        if not issues:
            return f"No {state} issues."
        lines = [f"{len(issues)} {state} issue(s):"]
        for i in issues:
            tags = ",".join(l["name"] for l in i.get("labels", []))
            lines.append(f"  #{i['number']} [{i['state']}] {i['title']}" + (f" ({tags})" if tags else ""))
        return _truncate("\n".join(lines), self.max_bytes)


class GitHubReadIssueTool(_RepoTool):
    """Read one issue and its comments."""

    name = "github_read_issue"
    description = "Read an issue from the studio's GitHub repository, including its comments."
    input_schema = {
        "type": "object",
        "properties": {
            "number": {"type": "integer", "description": "Issue number."},
        },
        "required": ["number"],
    }

    async def __call__(self, number: int) -> str:
        client = self._client
        issue = await client.request("GET", self._path(f"/issues/{int(number)}"))
        comments = await client.request(
            "GET", self._path(f"/issues/{int(number)}/comments"), params={"per_page": 20}
        )
        lines = [
            f"#{issue['number']} [{issue['state']}] {issue['title']}",
            f"by {issue['user']['login']}",
            "",
            issue.get("body") or "(no description)",
        ]
        for c in comments or []:
            lines += ["", f"--- {c['user']['login']}:", c.get("body") or ""]
        return _truncate("\n".join(lines), self.max_bytes)


class GitHubWriteIssueTool(_RepoTool):
    """Create a new issue, or update an existing one."""

    name = "github_write_issue"
    description = (
        "Create an issue in the studio's GitHub repository, or update an "
        "existing one by passing its number."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "number": {
                "type": "integer",
                "description": "Issue to update. Omit to create a new issue.",
            },
            "title": {"type": "string", "description": "Issue title."},
            "body": {"type": "string", "description": "Issue description."},
            "state": {
                "type": "string",
                "enum": ["open", "closed"],
                "description": "Only when updating.",
            },
        },
        "required": [],
    }

    async def __call__(
        self,
        number: int | None = None,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
    ) -> str:
        payload = {
            k: v
            for k, v in (("title", title), ("body", body), ("state", state))
            if v is not None
        }
        if not payload:
            return "Nothing to write: pass at least a title, body or state."

        if number is None:
            if not title:
                return "A new issue needs a title."
            issue = await self._client.request("POST", self._path("/issues"), body=payload)
            return f"Created #{issue['number']}: {issue['title']}\n{issue['html_url']}"

        issue = await self._client.request(
            "PATCH", self._path(f"/issues/{int(number)}"), body=payload
        )
        return f"Updated #{issue['number']} [{issue['state']}]: {issue['title']}\n{issue['html_url']}"


class GitHubCommentTool(_RepoTool):
    """Comment on an issue or pull request."""

    name = "github_comment"
    description = (
        "Post a comment on an issue or pull request in the studio's GitHub repository."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "number": {"type": "integer", "description": "Issue or PR number."},
            "body": {"type": "string", "description": "Comment text."},
        },
        "required": ["number", "body"],
    }

    async def __call__(self, number: int, body: str) -> str:
        # PRs are issues as far as the comments endpoint is concerned.
        comment = await self._client.request(
            "POST", self._path(f"/issues/{int(number)}/comments"), body={"body": body}
        )
        return f"Commented on #{number}: {comment['html_url']}"


class GitHubListPullRequestsTool(_RepoTool):
    """List pull requests."""

    name = "github_list_pull_requests"
    description = "List pull requests in the studio's GitHub repository."
    input_schema = {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "enum": ["open", "closed", "all"],
                "description": "Defaults to open.",
            },
        },
        "required": [],
    }

    async def __call__(self, state: str = "open") -> str:
        data = await self._client.request(
            "GET", self._path("/pulls"), params={"state": state, "per_page": 30}
        )
        if not data:
            return f"No {state} pull requests."
        lines = [f"{len(data)} {state} pull request(s):"]
        lines += [
            f"  #{p['number']} [{p['state']}] {p['title']} ({p['head']['ref']} → {p['base']['ref']})"
            for p in data
        ]
        return _truncate("\n".join(lines), self.max_bytes)


class GitHubReadPullRequestTool(_RepoTool):
    """Read a pull request and which files it touches."""

    name = "github_read_pull_request"
    description = (
        "Read a pull request from the studio's GitHub repository, with the list "
        "of files it changes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "number": {"type": "integer", "description": "Pull request number."},
        },
        "required": ["number"],
    }

    async def __call__(self, number: int) -> str:
        client = self._client
        pr = await client.request("GET", self._path(f"/pulls/{int(number)}"))
        files = await client.request(
            "GET", self._path(f"/pulls/{int(number)}/files"), params={"per_page": 50}
        )
        lines = [
            f"#{pr['number']} [{pr['state']}] {pr['title']}",
            f"by {pr['user']['login']}  {pr['head']['ref']} → {pr['base']['ref']}",
            f"+{pr.get('additions', 0)} -{pr.get('deletions', 0)} across {pr.get('changed_files', 0)} file(s)",
            "",
            pr.get("body") or "(no description)",
            "",
            "Files:",
        ]
        lines += [f"  {f['status']:>8}  {f['filename']}" for f in files or []]
        return _truncate("\n".join(lines), self.max_bytes)


class GitHubWriteFileTool(_RepoTool):
    """Create or replace a file in the pinned repository."""

    name = "github_write_file"
    description = (
        "Create or replace a file in the studio's GitHub repository, committing "
        "the change. Provide the file's FULL new contents, not a patch or diff."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path within the repo."},
            "content": {
                "type": "string",
                "description": "The complete new contents of the file.",
            },
            "message": {"type": "string", "description": "Commit message."},
            "branch": {
                "type": "string",
                "description": "Branch to commit to. Omit for the default branch.",
            },
        },
        "required": ["path", "content", "message"],
    }

    async def __call__(
        self,
        path: str,
        content: str,
        message: str,
        branch: str | None = None,
    ) -> str:
        client = self._client
        api_path = self._path(f"/contents/{quote(path.lstrip('/'))}")

        # Updating requires the blob sha of what is being replaced; creating
        # must not send one. Look first and branch on what is actually there.
        sha: str | None = None
        try:
            existing = await client.request("GET", api_path, params={"ref": branch})
            if isinstance(existing, list):
                return f"{path} is a directory, not a file."
            sha = existing.get("sha")
        except GitHubError as exc:
            if "Not found" not in str(exc):
                raise

        body: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        }
        if branch:
            body["branch"] = branch
        if sha:
            body["sha"] = sha

        result = await client.request("PUT", api_path, body=body)
        commit = (result or {}).get("commit", {})
        verb = "Updated" if sha else "Created"
        where = f" on {branch}" if branch else ""
        return f"{verb} {path}{where} in {commit.get('sha', '?')[:7]}: {message}"


class GitHubCreateBranchTool(_RepoTool):
    """Create a branch in the pinned repository."""

    name = "github_create_branch"
    description = "Create a new branch in the studio's GitHub repository."
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "New branch name."},
            "base": {
                "type": "string",
                "description": "Branch to start from. Omit for the default branch.",
            },
        },
        "required": ["name"],
    }

    async def __call__(self, name: str, base: str | None = None) -> str:
        client = self._client
        if base is None:
            repo_info = await client.request("GET", self._path(""))
            base = repo_info.get("default_branch", "main")

        ref = await client.request("GET", self._path(f"/git/ref/heads/{quote(base)}"))
        base_sha = ref["object"]["sha"]

        await client.request(
            "POST",
            self._path("/git/refs"),
            body={"ref": f"refs/heads/{name}", "sha": base_sha},
        )
        return f"Created branch {name} from {base} at {base_sha[:7]}."


class GitHubCreatePullRequestTool(_RepoTool):
    """Open a pull request in the pinned repository."""

    name = "github_create_pull_request"
    description = "Open a pull request in the studio's GitHub repository."
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Pull request title."},
            "head": {"type": "string", "description": "Branch containing the changes."},
            "base": {
                "type": "string",
                "description": "Branch to merge into. Omit for the default branch.",
            },
            "body": {"type": "string", "description": "Description."},
        },
        "required": ["title", "head"],
    }

    async def __call__(
        self,
        title: str,
        head: str,
        base: str | None = None,
        body: str | None = None,
    ) -> str:
        client = self._client
        if base is None:
            repo_info = await client.request("GET", self._path(""))
            base = repo_info.get("default_branch", "main")

        payload: dict[str, Any] = {"title": title, "head": head, "base": base}
        if body:
            payload["body"] = body

        pr = await client.request("POST", self._path("/pulls"), body=payload)
        return f"Opened #{pr['number']}: {pr['title']} ({head} → {base})\n{pr['html_url']}"
