"""LiteLLM must not use its aiohttp transport in a Spectre process.

LiteLLM caches HTTP clients for an hour and drops expired ones without
closing them; with the aiohttp transport each dropped client logs
``ERROR: Unclosed client session`` when garbage-collected. The httpx
transport has no such finalizer noise.
"""
from __future__ import annotations

import logging

import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from sr2_spectre.cli import _configure_logging


@pytest.fixture
def restore_global_state(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
    monkeypatch.delenv("DISABLE_AIOHTTP_TRANSPORT", raising=False)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_configure_logging_selects_httpx_transport(tmp_path, restore_global_state):
    assert AsyncHTTPHandler._should_use_aiohttp_transport() is True

    _configure_logging("INFO", str(tmp_path / "spectre.log"))

    assert AsyncHTTPHandler._should_use_aiohttp_transport() is False
