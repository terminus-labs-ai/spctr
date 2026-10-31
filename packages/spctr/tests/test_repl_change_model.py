"""Tests for the REPL /change-model command (spc-956).

Terminal counterpart of Discord's /model: list configured models and switch
the session's active model mid-session.  Covers the pure decision logic
(render_change_model_command) and the REPL handler end to end — listing,
in-process apply through agent.apply_config, pointer-file persistence via
$SR2_ACTIVE_MODEL_FILE, pointer-write failure, unknown names, re-selecting
the active model, and presence in completions/help/docstring.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.interfaces.repl import (
    REPLInterface,
    render_change_model_command,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers (stdlib pytest only; shared conftest untouched)
# ---------------------------------------------------------------------------


def _spectre_with_models(active: str = "default") -> SpectreConfig:
    """A real SpectreConfig with two model entries, like the Discord tests."""
    return SpectreConfig(
        agent=AgentConfig(name="t"),
        models={
            "default": ModelConfig(model="m-default", base_url="http://default/v1"),
            "fast": ModelConfig(model="m-fast", base_url="http://fast/v1"),
        },
        pipeline={"layers": []},
        active_model=active,
    )


def _agent_with_config(config: SpectreConfig) -> MagicMock:
    """Mock agent whose apply_config really updates its config, like Agent.

    ``agent.config`` is deleted from the mock up front so assignment above
    and inside ``_apply`` lands on a real attribute rather than raising
    through MagicMock's reserved-name guard.
    """
    agent = MagicMock()
    del agent.config
    agent.config = config

    def _apply(new: SpectreConfig) -> list[str]:
        agent.config = new
        return ["models"]

    agent.apply_config.side_effect = _apply
    return agent


class _LineSink:
    """Minimal file-like sink that records written strings line-wise."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def write(self, s: str) -> int:
        for part in s.split("\n"):
            if part:
                self._lines.append(part)
        return len(s)

    def flush(self) -> None:
        pass


def _console_capturing(interface: REPLInterface) -> list[str]:
    from rich.console import Console

    buf: list[str] = []
    interface.console = Console(file=_LineSink(buf), force_terminal=False, width=100)
    return buf


def _plain_text(lines: list[str]) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(lines))


def _rendered(lines: list[tuple[str, str]]) -> str:
    """Render a (markup, style) line list through Rich to plain text."""
    from rich.console import Console

    buf: list[str] = []
    console = Console(file=_LineSink(buf), force_terminal=False, width=100)
    for markup, style in lines:
        console.print(markup, style=style or None)
    return _plain_text(buf)


# ---------------------------------------------------------------------------
# Pure decision logic
# ---------------------------------------------------------------------------


def test_render_lists_models_sorted_with_active_marker() -> None:
    lines, selection = render_change_model_command(
        None, {"fast", "default", "big"}, "fast", pointer_configured=True
    )
    assert selection is None
    text = _rendered(lines)
    # One line per model, sorted, each name appearing only on its own line.
    model_lines = [ln for ln in text.splitlines() if any(n in ln for n in ("big", "default", "fast"))]
    assert len(model_lines) == 3
    assert "big" in model_lines[0]
    assert "default" in model_lines[1]
    assert "fast" in model_lines[2]
    assert "← active" in text
    # exactly one marker, on the active model's line
    marked = [ln for ln in text.splitlines() if "← active" in ln]
    assert len(marked) == 1 and "fast" in marked[0]


def test_render_unknown_model_errors_without_selection() -> None:
    lines, selection = render_change_model_command(
        "ghost", {"default", "fast"}, "default", pointer_configured=True
    )
    assert selection is None
    text = _rendered(lines)
    assert "Unknown model" in text
    assert "default" in text and "fast" in text


def test_render_active_model_reports_already_using() -> None:
    lines, selection = render_change_model_command(
        "fast", {"default", "fast"}, "fast", pointer_configured=True
    )
    assert selection is None
    assert "Already using" in _rendered(lines)


def test_render_switch_selects_and_notes_persistence() -> None:
    lines, selection = render_change_model_command(
        "fast", {"default", "fast"}, "default", pointer_configured=True
    )
    assert selection == "fast"
    text = _rendered(lines)
    assert "Switched to" in text and "fast" in text
    assert "SR2_ACTIVE_MODEL_FILE" not in text


def test_render_switch_without_pointer_states_not_persisted() -> None:
    lines, selection = render_change_model_command(
        "fast", {"default", "fast"}, "default", pointer_configured=False
    )
    assert selection == "fast"
    assert "SR2_ACTIVE_MODEL_FILE" in _rendered(lines)


def test_render_matches_model_names_case_sensitively() -> None:
    lines, selection = render_change_model_command(
        "FAST", {"default", "fast"}, "default", pointer_configured=True
    )
    assert selection is None
    assert "Unknown model" in _rendered(lines)


# ---------------------------------------------------------------------------
# AC1 — bare /change-model lists models; nothing changes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_change_model_lists_models_without_side_effects(
    tmp_path, monkeypatch
) -> None:
    pointer = tmp_path / "active_model"
    monkeypatch.setenv("SR2_ACTIVE_MODEL_FILE", str(pointer))
    config = _spectre_with_models("default")
    agent = _agent_with_config(config)
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/change-model", None)

    text = _plain_text(buf)
    assert "default" in text and "fast" in text
    assert "← active" in text
    agent.apply_config.assert_not_called()
    assert agent.config.active_model == "default"
    assert not pointer.exists()


# ---------------------------------------------------------------------------
# AC2 — valid switch applies in-process and persists to the pointer file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switch_applies_config_and_writes_pointer(tmp_path, monkeypatch) -> None:
    pointer = tmp_path / "active_model"
    monkeypatch.setenv("SR2_ACTIVE_MODEL_FILE", str(pointer))
    config = _spectre_with_models("default")
    agent = _agent_with_config(config)
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/change-model", "fast")

    # Applied in-process through apply_config with a copied config
    # (the Runtime re-seats the LLM through LiveLLM.retarget).
    agent.apply_config.assert_called_once()
    applied = agent.apply_config.call_args[0][0]
    assert applied.active_model == "fast"
    assert applied.models is config.models
    assert agent.config.active_model == "fast"
    # Pointer file: same trailing-newline convention as the Discord writer.
    assert pointer.read_text() == "fast\n"
    assert "Switched to" in _plain_text(buf)


@pytest.mark.asyncio
async def test_switch_passes_copied_config_to_apply(tmp_path, monkeypatch) -> None:
    """The REPL must not mutate the config object in place — it copies."""
    pointer = tmp_path / "active_model"
    monkeypatch.setenv("SR2_ACTIVE_MODEL_FILE", str(pointer))
    config = _spectre_with_models("default")
    agent = _agent_with_config(config)
    interface = REPLInterface()
    _console_capturing(interface)

    await interface._handle_command(agent, "/change-model", "fast")

    applied = agent.apply_config.call_args[0][0]
    assert applied is not config
    assert applied.active_model == "fast"
    # The original object is untouched (apply_config received a copy).
    assert config.active_model == "default"


# ---------------------------------------------------------------------------
# AC3 — no pointer file: switch still applies in-session, says so
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_switch_without_pointer_is_session_only(monkeypatch) -> None:
    monkeypatch.delenv("SR2_ACTIVE_MODEL_FILE", raising=False)
    agent = _agent_with_config(_spectre_with_models("default"))
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/change-model", "fast")

    assert agent.config.active_model == "fast"
    text = _plain_text(buf)
    assert "Switched to" in text
    assert "SR2_ACTIVE_MODEL_FILE" in text  # states persistence is unavailable


# ---------------------------------------------------------------------------
# AC4 — pointer-write failure: error, no change, no success claim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pointer_write_failure_changes_nothing(monkeypatch) -> None:
    monkeypatch.setenv("SR2_ACTIVE_MODEL_FILE", "/nonexistent-dir/active_model")
    config = _spectre_with_models("default")
    agent = _agent_with_config(config)
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/change-model", "fast")

    text = _plain_text(buf)
    assert "Could not persist" in text
    assert "Switched to" not in text
    agent.apply_config.assert_not_called()
    assert agent.config.active_model == "default"


# ---------------------------------------------------------------------------
# AC5 — unknown name: error lists available models; nothing changes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_model_lists_available_and_changes_nothing(
    tmp_path, monkeypatch
) -> None:
    pointer = tmp_path / "active_model"
    monkeypatch.setenv("SR2_ACTIVE_MODEL_FILE", str(pointer))
    config = _spectre_with_models("default")
    agent = _agent_with_config(config)
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/change-model", "ghost")

    text = _plain_text(buf)
    assert "Unknown model" in text
    assert "default" in text and "fast" in text
    agent.apply_config.assert_not_called()
    assert agent.config.active_model == "default"
    assert not pointer.exists()


# ---------------------------------------------------------------------------
# AC6 — re-selecting the active model: "already using", no write, no apply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selecting_active_model_is_a_noop(tmp_path, monkeypatch) -> None:
    pointer = tmp_path / "active_model"
    monkeypatch.setenv("SR2_ACTIVE_MODEL_FILE", str(pointer))
    config = _spectre_with_models("default")
    agent = _agent_with_config(config)
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(agent, "/change-model", "default")

    assert "Already using" in _plain_text(buf)
    agent.apply_config.assert_not_called()
    assert not pointer.exists()


# ---------------------------------------------------------------------------
# AC7 — presence in completions, help, and the module docstring
# ---------------------------------------------------------------------------


def test_change_model_in_commands_list() -> None:
    from sr2_spectre.interfaces.repl import _COMMANDS

    assert "/change-model" in _COMMANDS


def test_change_model_offered_by_completer() -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from sr2_spectre.interfaces.repl import _COMMANDS, _SlashCompleter

    doc = Document(text="/change", cursor_position=len("/change"))
    texts = [
        c.text
        for c in _SlashCompleter(_COMMANDS).get_completions(doc, CompleteEvent(""))
    ]
    assert texts == ["/change-model"]


@pytest.mark.asyncio
async def test_help_lists_change_model() -> None:
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(MagicMock(), "/help", None)

    assert "/change-model" in _plain_text(buf)


def test_module_docstring_lists_change_model() -> None:
    import sr2_spectre.interfaces.repl as repl_mod

    assert "/change-model" in (repl_mod.__doc__ or "")


# ---------------------------------------------------------------------------
# Dispatch & env plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_dispatched_case_insensitively() -> None:
    """The run loop lowercases the command token; the handler must match."""
    monkey = _agent_with_config(_spectre_with_models("default"))
    interface = REPLInterface()
    buf = _console_capturing(interface)

    await interface._handle_command(monkey, "/change-model", None)

    assert "Models:" in _plain_text(buf)


def test_pointer_path_reads_env(monkeypatch) -> None:
    from sr2_spectre.interfaces.repl import _active_model_pointer

    monkeypatch.delenv("SR2_ACTIVE_MODEL_FILE", raising=False)
    assert _active_model_pointer() is None

    monkeypatch.setenv("SR2_ACTIVE_MODEL_FILE", "/tmp/active_model")
    assert _active_model_pointer() == Path("/tmp/active_model")
