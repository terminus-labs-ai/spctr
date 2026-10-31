"""Tests for the beads tools.

The shim is faked at the aiohttp boundary: request-shaping and response
handling are this module's job, while validating model input is the shim's
(tested next to the subprocess it protects, in beads-shim/test_beads_shim.py).
"""
import json
import os
from unittest.mock import patch

import pytest

from sr2_spectre.tools.builtins.beads import (
    BeadsCommentTool,
    BeadsCreateTool,
    BeadsDepTool,
    BeadsError,
    BeadsQueryTool,
    BeadsUpdateTool,
    _truncate,
)

ALL_TOOLS = (
    BeadsQueryTool, BeadsCreateTool, BeadsUpdateTool, BeadsCommentTool, BeadsDepTool,
)
BASE = "http://192.168.50.117:8431"


class FakePost:
    """Stands in for session.post, recording the request it was given."""

    def __init__(self, status=200, payload=None, raw=None):
        self.status = status
        self._payload = payload if payload is not None else {"ok": True, "output": "done"}
        self._raw = raw
        self.calls = []

    def __call__(self, url, json=None, headers=None):
        self.calls.append({"url": url, "body": json, "headers": headers})
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return self._raw if self._raw is not None else json.dumps(self._payload)


class FakeSession:
    def __init__(self, post):
        self.post = post

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def run(tool, fake, **kwargs):
    """Invoke *tool*, with aiohttp replaced by *fake*."""
    import asyncio
    with patch.dict(os.environ, {"GRINDFORGE_BEADS_TOKEN": "tok"}), \
         patch("aiohttp.ClientSession", lambda *a, **k: FakeSession(fake)):
        return asyncio.run(tool(**kwargs))


def build(cls, **kwargs):
    kwargs.setdefault("token_env", "GRINDFORGE_BEADS_TOKEN")
    return cls(base_url=BASE, **kwargs)


# --- surface ---------------------------------------------------------------

def test_schema_budget_stays_small() -> None:
    """Tool schemas are re-sent every turn; a broad surface is what made the
    official GitHub MCP server unusable here."""
    blob = json.dumps([
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in ALL_TOOLS
    ])
    assert len(blob) < 6000, f"schema grew to {len(blob)} bytes"


def test_every_tool_has_a_distinct_endpoint() -> None:
    endpoints = [t.endpoint for t in ALL_TOOLS]
    assert len(set(endpoints)) == len(endpoints)
    assert all(e.startswith("/beads/") for e in endpoints)


# --- token -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_token_is_a_clear_error() -> None:
    tool = build(BeadsQueryTool, token_env="DEFINITELY_UNSET_BEADS_VAR")
    with pytest.raises(BeadsError, match="DEFINITELY_UNSET_BEADS_VAR"):
        await tool()


def test_token_is_sent_as_a_bearer_header() -> None:
    fake = FakePost()
    run(build(BeadsQueryTool), fake)
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer tok"


def test_rejected_token_surfaces_plainly() -> None:
    fake = FakePost(status=401)
    with pytest.raises(BeadsError, match="rejected this agent's token"):
        run(build(BeadsQueryTool), fake)


# --- request shaping -------------------------------------------------------

def test_query_defaults_to_list_and_drops_empty_fields() -> None:
    fake = FakePost()
    run(build(BeadsQueryTool), fake)
    assert fake.calls[0]["url"] == f"{BASE}/beads/query"
    assert fake.calls[0]["body"] == {"op": "list"}


def test_unset_optional_fields_are_not_sent() -> None:
    """The shim should see a clean request, not a wall of nulls."""
    fake = FakePost()
    run(build(BeadsCreateTool), fake, title="Jump feels floaty", priority="1")
    assert fake.calls[0]["body"] == {"title": "Jump feels floaty", "priority": "1"}


def test_update_sends_only_changed_fields() -> None:
    fake = FakePost()
    run(build(BeadsUpdateTool), fake, id="grindsourced-1", status="closed")
    assert fake.calls[0]["body"] == {"id": "grindsourced-1", "status": "closed"}


def test_comment_sends_id_and_body() -> None:
    fake = FakePost()
    run(build(BeadsCommentTool), fake, id="grindsourced-1", body="fixed in a546b61")
    assert fake.calls[0]["body"] == {"id": "grindsourced-1", "body": "fixed in a546b61"}


def test_dep_accepts_from_which_is_a_python_keyword() -> None:
    fake = FakePost()
    run(build(BeadsDepTool), fake, **{"from": "grindsourced-1", "to": "grindsourced-2"})
    assert fake.calls[0]["body"] == {
        "op": "link", "type": "blocks", "from": "grindsourced-1", "to": "grindsourced-2",
    }


def test_dep_without_both_ids_does_not_call_the_shim() -> None:
    fake = FakePost()
    out = run(build(BeadsDepTool), fake, **{"from": "grindsourced-1"})
    assert fake.calls == []
    assert "needs both" in out


# --- responses -------------------------------------------------------------

def test_successful_output_is_returned() -> None:
    fake = FakePost(payload={"ok": True, "output": "grindsourced-9 created"})
    assert "grindsourced-9 created" in run(build(BeadsQueryTool), fake)


def test_rejection_is_reported_not_raised() -> None:
    """A bad field should teach the model, not blow up its turn."""
    fake = FakePost(status=400, payload={"ok": False, "error": "priority must be 0-4"})
    out = run(build(BeadsUpdateTool), fake, id="grindsourced-1", priority="9")
    assert "Beads rejected that" in out
    assert "priority must be 0-4" in out


def test_shim_failure_raises() -> None:
    fake = FakePost(status=503)
    with pytest.raises(BeadsError, match="HTTP 503"):
        run(build(BeadsQueryTool), fake)


def test_malformed_shim_output_raises() -> None:
    fake = FakePost(raw="<html>not json</html>")
    with pytest.raises(BeadsError, match="malformed"):
        run(build(BeadsQueryTool), fake)


def test_long_output_is_trimmed() -> None:
    fake = FakePost(payload={"ok": True, "output": "x" * 50_000})
    out = run(build(BeadsQueryTool, max_bytes=500), fake)
    assert len(out.encode()) < 700
    assert "truncated" in out


def test_empty_output_is_labelled() -> None:
    fake = FakePost(payload={"ok": True, "output": ""})
    assert run(build(BeadsQueryTool), fake) == "(no output)"


def test_truncate_leaves_short_text_alone() -> None:
    assert _truncate("hello", 100) == "hello"
