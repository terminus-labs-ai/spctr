"""Tests for the async ComfyUI client."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sr2_spectre.tools.builtins.comfyui_client import (
    ComfyUIClient,
    ComfyUIError,
    ImageRef,
)


@pytest.fixture
def client() -> ComfyUIClient:
    return ComfyUIClient(
        base_url="http://192.168.50.233:8188",
        timeout=10.0,
        max_poll_time=30.0,
        poll_interval=0.1,
    )


# -- ImageRef --

def test_image_ref_repr():
    ref = ImageRef(filename="test.png", subfolder="output")
    assert "test.png" in repr(ref)
    assert "output" in repr(ref)


# -- URL building --

def test_url_forward_slash(client):
    assert client._url("/prompt").endswith("/prompt")


def test_url_no_slash(client):
    assert client._url("prompt").endswith("/prompt")


# -- Health --

@pytest.mark.asyncio
async def test_is_available_true(client):
    with patch.object(
        client, "_get", new=AsyncMock(return_value={"system": {"os": "linux"}})
    ):
        result = await client.is_available()
    assert result is True


@pytest.mark.asyncio
async def test_is_available_false(client):
    with patch.object(client, "_get", new=AsyncMock(side_effect=Exception("unreachable"))):
        result = await client.is_available()
    assert result is False


# -- Workflow submission --

@pytest.mark.asyncio
async def test_submit_workflow(client):
    mock_post = AsyncMock(return_value={"prompt_id": "abc123"})
    with patch.object(client, "_post", mock_post):
        prompt_id = await client.submit_workflow({"1": {"class_type": "KSampler"}})
    assert prompt_id == "abc123"

    # Verify it called _post with the right path
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "/prompt"


# -- History --

@pytest.mark.asyncio
async def test_get_history(client):
    history_data = {
        "status": {"completed": True},
        "outputs": {"9": {"images": [{"filename": "img.png", "subfolder": "", "type": "output"}]}},
    }
    with patch.object(
        client, "_get", new=AsyncMock(return_value={"abc123": history_data})
    ):
        result = await client.get_history("abc123")
    assert result is not None
    assert result["status"]["completed"] is True


@pytest.mark.asyncio
async def test_get_history_not_done(client):
    with patch.object(client, "_get", new=AsyncMock(return_value={})):
        result = await client.get_history("abc123")
    assert result is None


# -- Image extraction --

def test_extract_images(client):
    history = {
        "outputs": {
            "9": {
                "images": [
                    {"filename": "img1.png", "subfolder": "", "type": "output"},
                    {"filename": "img2.jpg", "subfolder": "test", "type": "temp"},
                ]
            }
        }
    }
    refs = client.extract_images(history)
    assert len(refs) == 2
    assert refs[0].filename == "img1.png"
    assert refs[0].output_type == "output"
    assert refs[1].filename == "img2.jpg"
    assert refs[1].subfolder == "test"


def test_extract_images_empty():
    history = {"outputs": {}}
    assert ComfyUIClient.extract_images(history) == []


# -- Wait for completion (success) --

@pytest.mark.asyncio
async def test_wait_for_completion_success(client):
    history = {"status": {"completed": True}, "outputs": {}}

    async def mock_get(path):
        return {"p1": history}

    with patch.object(client, "_get", new=AsyncMock(side_effect=mock_get)):
        result = await client.wait_for_completion("p1")
    assert result["status"]["completed"] is True


# -- Wait for completion (error) --

@pytest.mark.asyncio
async def test_wait_for_completion_error(client):
    history = {"status": {"status_str": "error", "messages": ["OOM"]}}

    async def mock_get(path):
        return {"p1": history}

    with patch.object(client, "_get", new=AsyncMock(side_effect=mock_get)):
        with pytest.raises(ComfyUIError, match="OOM"):
            await client.wait_for_completion("p1")


# -- Wait for completion (timeout) --

@pytest.mark.asyncio
async def test_wait_for_completion_timeout(client):
    # Never returns history — simulates a hang
    async def mock_get(path):
        return {}

    client.poll_interval = 0.05  # Fast poll for test
    client.max_poll_time = 0.2   # Short timeout

    with patch.object(client, "_get", new=AsyncMock(side_effect=mock_get)):
        with pytest.raises(TimeoutError):
            await client.wait_for_completion("p1")


# -- Download image --

@pytest.mark.asyncio
async def test_download_image(client, tmp_path):
    # We can't easily mock aiohttp at this level, so test the path logic
    # by checking download_first_output + extract_images work together
    history = {
        "outputs": {
            "9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
        }
    }

    mock_ref = AsyncMock()
    mock_ref.return_value = tmp_path / "test.png"

    with patch.object(client, "download_image", new=mock_ref):
        dest = await client.download_first_output(history, tmp_path, prefix="test")

    assert dest == tmp_path / "test.png"


# -- download_first_output no images --

@pytest.mark.asyncio
async def test_download_first_output_no_images(client, tmp_path):
    history = {"outputs": {}}
    with pytest.raises(ComfyUIError, match="No output images"):
        await client.download_first_output(history, tmp_path)


# -- Full generate cycle --

@pytest.mark.asyncio
async def test_generate_full_cycle(client, tmp_path):
    history = {
        "status": {"completed": True},
        "outputs": {
            "9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
        },
    }

    async def mock_submit(workflow):
        return "p1"

    async def mock_wait(pid):
        return history

    mock_ref = AsyncMock()
    mock_ref.return_value = tmp_path / "test.png"

    with (
        patch.object(client, "submit_workflow", new=mock_submit),
        patch.object(client, "wait_for_completion", new=mock_wait),
        patch.object(client, "download_image", new=mock_ref),
    ):
        path = await client.generate({"1": {}}, tmp_path, prefix="test")

    assert path == tmp_path / "test.png"


# -- Constructor defaults --

def test_client_defaults():
    c = ComfyUIClient()
    assert c.base_url == "http://127.0.0.1:8188"
    assert c.timeout == 60.0
    assert c.max_poll_time == 600.0
    assert c.poll_interval == 2.0


def test_client_base_url_strips_trailing_slash():
    c = ComfyUIClient(base_url="http://host:8188/")
    assert c.base_url == "http://host:8188"
