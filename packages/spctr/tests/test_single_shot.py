"""Tests for SingleShotInterface."""
import pytest
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

from sr2_spectre.interfaces.single_shot import SingleShotInterface
from sr2_spectre.core import TurnResult


def _make_agent(response_text: str = "The answer is 4") -> MagicMock:
    agent = MagicMock()
    agent.handle_user_message = AsyncMock(return_value=TurnResult(
        text=response_text, tool_calls_executed=0, total_tokens=30
    ))
    return agent


@pytest.mark.asyncio
async def test_start_stop_are_noops() -> None:
    interface = SingleShotInterface(prompt="hello")
    agent = _make_agent()
    await interface.start(agent)
    await interface.stop()
    agent.handle_user_message.assert_not_called()


@pytest.mark.asyncio
async def test_run_with_constructor_prompt(capsys: pytest.CaptureFixture) -> None:
    interface = SingleShotInterface(prompt="What is 2+2?")
    agent = _make_agent("4")

    await interface.run(agent)

    agent.handle_user_message.assert_called_once_with("What is 2+2?")
    captured = capsys.readouterr()
    assert captured.out.strip() == "4"


@pytest.mark.asyncio
async def test_run_reads_from_stdin(capsys: pytest.CaptureFixture) -> None:
    interface = SingleShotInterface()
    agent = _make_agent("pong")

    with patch("sys.stdin", StringIO("ping\n")), patch("sys.argv", ["sr2-spectre"]):
        await interface.run(agent)

    agent.handle_user_message.assert_called_once_with("ping")
    captured = capsys.readouterr()
    assert captured.out.strip() == "pong"


@pytest.mark.asyncio
async def test_run_empty_prompt_exits(capsys: pytest.CaptureFixture) -> None:
    interface = SingleShotInterface()
    agent = _make_agent()

    with patch("sys.stdin", StringIO("")), patch("sys.argv", ["sr2-spectre"]):
        with pytest.raises(SystemExit) as exc_info:
            await interface.run(agent)

    assert exc_info.value.code == 1
    agent.handle_user_message.assert_not_called()


def test_load_interface_via_cli() -> None:
    """_load_interface('single_shot') must return the SingleShotInterface class."""
    from sr2_spectre.cli import _load_interface
    cls = _load_interface("single_shot")
    assert cls is SingleShotInterface


def test_load_interface_with_prompt_kwarg() -> None:
    """Prompt kwarg is accepted by the resolved class and set on the instance.

    Note: _load_interface returns the CLASS; run_async instantiates it as
    ``interface_cls()`` (no kwargs), so the prompt actually reaches the
    interface via direct construction here — mirroring how a caller would
    pass it through if wiring changed.
    """
    from sr2_spectre.cli import _load_interface

    cls = _load_interface("single_shot")
    instance = cls(prompt="hello")
    assert isinstance(instance, SingleShotInterface)
    assert instance._prompt == "hello"



