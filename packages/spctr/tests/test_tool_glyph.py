"""Tests for the glyph_search tool.

The shim is faked at the aiohttp boundary: request-shaping and response
handling are this module's job, while pinning the corpus and bounding the
request is the shim's (tested next to the service it protects, in
grindforge-bot/glyph-shim/test_glyph_shim.py).
"""
import json
import os
from unittest.mock import patch

import pytest

from sr2_spectre.tools.builtins.glyph import GlyphError, GlyphSearchTool, _truncate

BASE = "http://192.168.50.117:8432"


class FakePost:
    """Stands in for session.post, recording the request it was given."""

    def __init__(self, status=200, payload=None, raw=None):
        self.status = status
        self._payload = payload if payload is not None else {
            "ok": True, "source": "godot-api", "output": "### Node2D.position",
        }
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
    with patch.dict(os.environ, {"GRINDFORGE_GLYPH_TOKEN": "tok"}), \
         patch("aiohttp.ClientSession", lambda *a, **k: FakeSession(fake)):
        return asyncio.run(tool(**kwargs))


def build(**kwargs):
    return GlyphSearchTool(base_url=BASE, **kwargs)


# --- surface ---------------------------------------------------------------

def test_schema_budget_stays_small() -> None:
    """Tool schemas are re-sent every turn, and this agent runs on 32k."""
    tool = build()
    blob = json.dumps({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    })
    assert len(blob) < 900, f"schema grew to {len(blob)} bytes"


def test_source_is_constrained_to_the_public_corpora() -> None:
    """The model must not be able to name a corpus the shim would refuse; the
    schema and the allowlist have to agree."""
    assert build().input_schema["properties"]["source"]["enum"] == [
        "godot-api", "godot-docs", "godot-shaders",
    ]


# --- request shaping --------------------------------------------------------

def test_query_is_posted_to_the_search_endpoint() -> None:
    fake = FakePost()
    run(build(), fake, query="Node2D position")
    assert fake.calls[0]["url"] == f"{BASE}/glyph/search"
    assert fake.calls[0]["body"] == {"query": "Node2D position"}


def test_source_is_forwarded_when_given() -> None:
    fake = FakePost()
    run(build(), fake, query="canvas_item", source="godot-shaders")
    assert fake.calls[0]["body"]["source"] == "godot-shaders"


def test_absent_source_is_omitted_so_the_shim_applies_its_default() -> None:
    fake = FakePost()
    run(build(), fake, query="x", source=None)
    assert "source" not in fake.calls[0]["body"]


def test_token_is_sent_as_a_bearer_header() -> None:
    fake = FakePost()
    run(build(), fake, query="x")
    assert fake.calls[0]["headers"] == {"Authorization": "Bearer tok"}


def test_missing_token_fails_before_any_request() -> None:
    fake = FakePost()
    tool = build()
    import asyncio
    with patch.dict(os.environ, {}, clear=True), \
         patch("aiohttp.ClientSession", lambda *a, **k: FakeSession(fake)):
        with pytest.raises(GlyphError, match="GRINDFORGE_GLYPH_TOKEN"):
            asyncio.run(tool(query="x"))
    assert fake.calls == []


# --- responses --------------------------------------------------------------

def test_results_are_returned_verbatim() -> None:
    fake = FakePost(payload={"ok": True, "output": "### Node2D.position\nThe position."})
    assert run(build(), fake, query="x") == "### Node2D.position\nThe position."


def test_an_unavailable_backend_is_reported_not_raised() -> None:
    """A knowledge base that is down is something to report and move past. An
    exception here reads to the model as a tool it should try again."""
    fake = FakePost(payload={"ok": False, "error": "could not reach Glyph: timed out"})
    out = run(build(), fake, query="x")
    assert out.startswith("Glyph search is unavailable:")
    assert "timed out" in out


def test_a_rejected_token_raises() -> None:
    fake = FakePost(status=401)
    with pytest.raises(GlyphError, match="rejected this agent's token"):
        run(build(), fake, query="x")


def test_a_shim_crash_raises() -> None:
    fake = FakePost(status=503)
    with pytest.raises(GlyphError, match="HTTP 503"):
        run(build(), fake, query="x")


def test_malformed_output_raises() -> None:
    fake = FakePost(raw="<html>502 Bad Gateway</html>")
    with pytest.raises(GlyphError, match="malformed"):
        run(build(), fake, query="x")


def test_empty_results_have_a_readable_placeholder() -> None:
    fake = FakePost(payload={"ok": True, "output": ""})
    assert run(build(), fake, query="x") == "(no results)"


def test_output_is_capped() -> None:
    fake = FakePost(payload={"ok": True, "output": "x" * 20000})
    out = run(build(max_bytes=500), fake, query="x")
    assert out.endswith("truncated at 500 bytes.")
    assert len(out.encode()) < 600


def test_truncation_never_produces_a_broken_character() -> None:
    assert "�" not in _truncate("é" * 100, 51)
