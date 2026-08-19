"""Tests for the repo-pinned GitHub tools.

The HTTP boundary is faked at ``_RepoTool._client``: what matters here is the
repo pinning, the shape handed to a small model, and the trimming, not
aiohttp's behaviour.
"""
import base64
import json
import os
from unittest.mock import patch

import pytest

from sr2_spectre.tools.builtins.github import (
    GitHubCommentTool,
    GitHubError,
    GitHubListFilesTool,
    GitHubListIssuesTool,
    GitHubListPullRequestsTool,
    GitHubReadFileTool,
    GitHubReadIssueTool,
    GitHubReadPullRequestTool,
    GitHubSearchCodeTool,
    GitHubWriteIssueTool,
    _truncate,
)

ALL_TOOLS = (
    GitHubReadFileTool, GitHubListFilesTool, GitHubSearchCodeTool,
    GitHubListIssuesTool, GitHubReadIssueTool, GitHubWriteIssueTool,
    GitHubCommentTool, GitHubListPullRequestsTool, GitHubReadPullRequestTool,
)

REPO = "grindforge/grindsourced"


class FakeClient:
    """Records calls and replays queued responses."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    async def request(self, method, path, params=None, body=None):
        self.calls.append({"method": method, "path": path, "params": params, "body": body})
        if not self._responses:
            raise AssertionError(f"unexpected extra request: {method} {path}")
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def build(tool_cls, *responses, **kwargs):
    tool = tool_cls(repo=REPO, token_env="TEST_GH_TOKEN", **kwargs)
    fake = FakeClient(*responses)
    patcher = patch.object(type(tool), "_client", property(lambda self: fake))
    patcher.start()
    return tool, fake, patcher


# ---------------------------------------------------------------------------
# Repo pinning — the property that keeps the agent off other repositories
# ---------------------------------------------------------------------------

def test_no_tool_lets_the_model_choose_a_repository() -> None:
    """owner/repo must never be model-supplied; it is pinned in config."""
    for tool_cls in ALL_TOOLS:
        props = set(tool_cls.input_schema.get("properties", {}))
        assert not props & {"owner", "repo", "repository", "full_name"}, tool_cls.name


def test_requests_are_scoped_to_the_configured_repo() -> None:
    tool, fake, p = build(GitHubListIssuesTool, [])
    try:
        import asyncio
        asyncio.run(tool())
    finally:
        p.stop()
    assert fake.calls[0]["path"] == f"/repos/{REPO}/issues"


def test_repo_must_be_owner_slash_name() -> None:
    with pytest.raises(ValueError):
        GitHubReadFileTool(repo="grindsourced")


def test_schema_budget_stays_small() -> None:
    """The whole point of these tools: the official MCP surface costs ~30.9k
    tokens, which does not fit a 32k budget alongside anything else."""
    blob = json.dumps([
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in ALL_TOOLS
    ])
    assert len(blob) < 6000, f"schema grew to {len(blob)} bytes"


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_token_is_a_clear_error() -> None:
    tool = GitHubReadFileTool(repo=REPO, token_env="DEFINITELY_UNSET_TOKEN_VAR")
    with pytest.raises(GitHubError, match="DEFINITELY_UNSET_TOKEN_VAR"):
        await tool(path="README.md")


@pytest.mark.asyncio
async def test_token_is_read_from_the_environment_at_call_time() -> None:
    tool = GitHubReadFileTool(repo=REPO, token_env="TEST_GH_TOKEN")
    with patch.dict(os.environ, {"TEST_GH_TOKEN": "t"}):
        client = tool._client
    assert client.repo == REPO


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_file_decodes_base64() -> None:
    payload = {"encoding": "base64",
               "content": base64.b64encode(b"extends Node\n").decode()}
    tool, fake, p = build(GitHubReadFileTool, payload)
    try:
        out = await tool(path="player.gd")
    finally:
        p.stop()
    assert "extends Node" in out
    assert "player.gd" in out


@pytest.mark.asyncio
async def test_read_file_on_a_directory_redirects() -> None:
    tool, fake, p = build(GitHubReadFileTool, [{"name": "a", "type": "file"}])
    try:
        out = await tool(path="src")
    finally:
        p.stop()
    assert "github_list_files" in out


@pytest.mark.asyncio
async def test_read_file_reports_binary_rather_than_dumping_it() -> None:
    payload = {"encoding": "base64",
               "content": base64.b64encode(b"\xff\xfe\x00\x01").decode(), "size": 4}
    tool, fake, p = build(GitHubReadFileTool, payload)
    try:
        out = await tool(path="icon.png")
    finally:
        p.stop()
    assert "binary" in out


@pytest.mark.asyncio
async def test_read_file_passes_the_ref_through() -> None:
    payload = {"encoding": "base64", "content": base64.b64encode(b"x").decode()}
    tool, fake, p = build(GitHubReadFileTool, payload)
    try:
        await tool(path="a.gd", ref="dev")
    finally:
        p.stop()
    assert fake.calls[0]["params"]["ref"] == "dev"


@pytest.mark.asyncio
async def test_list_files_puts_directories_first() -> None:
    entries = [
        {"name": "player.gd", "type": "file"},
        {"name": "scenes", "type": "dir"},
    ]
    tool, fake, p = build(GitHubListFilesTool, entries)
    try:
        out = await tool(path="src")
    finally:
        p.stop()
    assert out.index("scenes") < out.index("player.gd")


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_issues_excludes_pull_requests() -> None:
    """The issues endpoint returns PRs too; they are a different workflow."""
    data = [
        {"number": 1, "state": "open", "title": "Real issue", "labels": []},
        {"number": 2, "state": "open", "title": "A PR", "labels": [],
         "pull_request": {"url": "..."}},
    ]
    tool, fake, p = build(GitHubListIssuesTool, data)
    try:
        out = await tool()
    finally:
        p.stop()
    assert "Real issue" in out
    assert "A PR" not in out
    assert "1 open issue(s)" in out


@pytest.mark.asyncio
async def test_read_issue_includes_comments() -> None:
    issue = {"number": 7, "state": "open", "title": "Jump feels floaty",
             "user": {"login": "friend"}, "body": "gravity too low"}
    comments = [{"user": {"login": "diego"}, "body": "try 1200"}]
    tool, fake, p = build(GitHubReadIssueTool, issue, comments)
    try:
        out = await tool(number=7)
    finally:
        p.stop()
    assert "Jump feels floaty" in out
    assert "gravity too low" in out
    assert "diego" in out and "try 1200" in out


@pytest.mark.asyncio
async def test_write_issue_creates_when_no_number_given() -> None:
    created = {"number": 12, "title": "New", "html_url": "https://x/12"}
    tool, fake, p = build(GitHubWriteIssueTool, created)
    try:
        out = await tool(title="New", body="b")
    finally:
        p.stop()
    assert fake.calls[0]["method"] == "POST"
    assert fake.calls[0]["body"] == {"title": "New", "body": "b"}
    assert "Created #12" in out


@pytest.mark.asyncio
async def test_write_issue_updates_when_number_given() -> None:
    updated = {"number": 12, "state": "closed", "title": "New", "html_url": "https://x/12"}
    tool, fake, p = build(GitHubWriteIssueTool, updated)
    try:
        out = await tool(number=12, state="closed")
    finally:
        p.stop()
    assert fake.calls[0]["method"] == "PATCH"
    assert fake.calls[0]["path"].endswith("/issues/12")
    assert "Updated #12" in out


@pytest.mark.asyncio
async def test_write_issue_with_nothing_to_write_does_not_call_the_api() -> None:
    tool, fake, p = build(GitHubWriteIssueTool)
    try:
        out = await tool()
    finally:
        p.stop()
    assert fake.calls == []
    assert "at least a title" in out


@pytest.mark.asyncio
async def test_new_issue_without_a_title_is_refused() -> None:
    tool, fake, p = build(GitHubWriteIssueTool)
    try:
        out = await tool(body="orphan body")
    finally:
        p.stop()
    assert fake.calls == []
    assert "needs a title" in out


@pytest.mark.asyncio
async def test_comment_posts_to_the_issues_endpoint() -> None:
    tool, fake, p = build(GitHubCommentTool, {"html_url": "https://x/1#c"})
    try:
        out = await tool(number=3, body="looks good")
    finally:
        p.stop()
    assert fake.calls[0]["path"].endswith("/issues/3/comments")
    assert fake.calls[0]["body"] == {"body": "looks good"}
    assert "#3" in out


# ---------------------------------------------------------------------------
# Pull requests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_pull_requests_shows_branches() -> None:
    data = [{"number": 4, "state": "open", "title": "Add dash",
             "head": {"ref": "dash"}, "base": {"ref": "main"}}]
    tool, fake, p = build(GitHubListPullRequestsTool, data)
    try:
        out = await tool()
    finally:
        p.stop()
    assert "#4" in out and "dash" in out and "main" in out


@pytest.mark.asyncio
async def test_read_pull_request_lists_changed_files() -> None:
    pr = {"number": 4, "state": "open", "title": "Add dash",
          "user": {"login": "friend"}, "head": {"ref": "dash"}, "base": {"ref": "main"},
          "additions": 30, "deletions": 2, "changed_files": 1, "body": "adds a dash"}
    files = [{"status": "modified", "filename": "src/player.gd"}]
    tool, fake, p = build(GitHubReadPullRequestTool, pr, files)
    try:
        out = await tool(number=4)
    finally:
        p.stop()
    assert "src/player.gd" in out
    assert "+30 -2" in out


# ---------------------------------------------------------------------------
# Trimming — tool results are re-sent every turn, so size is paid repeatedly
# ---------------------------------------------------------------------------

def test_truncate_marks_what_it_cut() -> None:
    out = _truncate("x" * 100, 20)
    assert out.startswith("x" * 20)
    assert "truncated" in out


def test_truncate_leaves_short_text_alone() -> None:
    assert _truncate("hello", 100) == "hello"


@pytest.mark.asyncio
async def test_large_file_is_trimmed_to_max_bytes() -> None:
    payload = {"encoding": "base64",
               "content": base64.b64encode(b"A" * 50_000).decode()}
    tool, fake, p = build(GitHubReadFileTool, payload, max_bytes=500)
    try:
        out = await tool(path="big.gd")
    finally:
        p.stop()
    assert len(out.encode()) < 700
    assert "truncated" in out
