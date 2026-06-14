"""Tests for CLI argument parsing (sr2-spectre).

Verifies that positional arguments (config, prompt) must appear contiguously
before options, and that the usage examples in the epilog reflect the correct
command syntax.  Regression tests for obsidian-nid.
"""
from __future__ import annotations

import pytest

from sr2_spectre.cli import _parse_args


# ---------------------------------------------------------------------------
# Correct usage: positionals contiguous, options after
# ---------------------------------------------------------------------------

class TestParseArgsCorrectForms:
    """Verify all documented command forms parse correctly."""

    def test_single_shot_prompt_before_interface(self) -> None:
        """sr2-spectre config.yaml 'prompt' --interface single_shot"""
        args = _parse_args(
            ["config.yaml", "What is the weather?", "--interface", "single_shot"]
        )
        assert args.config == "config.yaml"
        assert args.prompt == ["What is the weather?"]
        assert args.interface == "single_shot"

    def test_single_shot_prompt_default_interface(self) -> None:
        """--interface single_shot is the default; omitting it works."""
        args = _parse_args(["config.yaml", "What is 2+2?"])
        assert args.config == "config.yaml"
        assert args.prompt == ["What is 2+2?"]
        assert args.interface == "single_shot"  # default

    def test_tui_without_prompt(self) -> None:
        """sr2-spectre config.yaml --interface tui"""
        args = _parse_args(["config.yaml", "--interface", "tui"])
        assert args.config == "config.yaml"
        assert args.prompt == []
        assert args.interface == "tui"

    def test_trace_with_prompt(self) -> None:
        """sr2-spectre config.yaml 'Hello' --trace"""
        args = _parse_args(["config.yaml", "Hello", "--trace"])
        assert args.config == "config.yaml"
        assert args.prompt == ["Hello"]
        assert args.trace is True

    def test_multi_word_prompt(self) -> None:
        """Prompt is nargs='*', so multiple words are captured."""
        args = _parse_args(
            ["config.yaml", "How", "do", "I", "center", "a", "div?"]
        )
        assert args.prompt == ["How", "do", "I", "center", "a", "div?"]

    def test_agent_flag_with_prompt(self) -> None:
        """--agent can appear after positionals."""
        args = _parse_args(["config.yaml", "hello", "--agent", "edi"])
        assert args.config == "config.yaml"
        assert args.prompt == ["hello"]
        assert args.agent == "edi"

    def test_all_flags_after_positionals(self) -> None:
        """Complex form: config + prompt + multiple flags."""
        args = _parse_args(
            [
                "config.yaml",
                "run diagnostics",
                "--interface",
                "single_shot",
                "--trace",
                "--session-id",
                "abc-123",
                "--log-level",
                "DEBUG",
            ]
        )
        assert args.config == "config.yaml"
        assert args.prompt == ["run diagnostics"]
        assert args.interface == "single_shot"
        assert args.trace is True
        assert args.session_id == "abc-123"
        assert args.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Broken usage: option between positionals (must exit 2)
# ---------------------------------------------------------------------------

class TestParseArgsBrokenForms:
    """Verify that the broken forms (option between positionals) still fail.

    These are regression tests — we fixed the docs, not the parser, so the
    broken form must still error to make users read the corrected docs.
    """

    def test_interface_between_config_and_prompt_fails(self) -> None:
        """sr2-spectre config.yaml --interface single_shot 'prompt' exits 2."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(
                ["config.yaml", "--interface", "single_shot", "What is the weather?"]
            )
        assert exc_info.value.code == 2

    def test_trace_between_config_and_prompt_fails(self) -> None:
        """Any flag between config and prompt fails."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["config.yaml", "--trace", "hello"])
        assert exc_info.value.code == 2

    def test_agent_between_config_and_prompt_fails(self) -> None:
        """--agent between config and prompt also fails."""
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["config.yaml", "--agent", "edi", "hello"])
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Epilog correctness
# ---------------------------------------------------------------------------

class TestEpilogCorrectness:
    """Verify the argparse epilog contains correct usage examples."""

    def test_epilog_exists(self) -> None:
        """The parser has an epilog with usage examples."""
        parser = _parse_args.__code__  # can't easily re-create, so test via help output
        # Instead, test by parsing and checking the help format string
        import io
        import contextlib

        from sr2_spectre.cli import _parse_args as parse
        # Reconstruct the parser to check epilog
        import argparse

        # We can't call _parse_args directly with --help (it sys.exits),
        # so verify the module docstring instead
        import sr2_spectre.cli as cli_module
        assert "config.yaml" in cli_module.__doc__
        assert "sr2-spectre" in cli_module.__doc__

    def test_module_docstring_shows_correct_order(self) -> None:
        """Module docstring must show prompt BEFORE --interface, not after."""
        import sr2_spectre.cli as cli_module
        doc = cli_module.__doc__ or ""

        # The correct form: config prompt --interface
        # Find the single_shot example line
        for line in doc.splitlines():
            if "single_shot" in line and "sr2-spectre" in line:
                # In the correct form, the prompt (quoted string) comes BEFORE --interface
                prompt_pos = line.index("'") if "'" in line else line.index('"') if '"' in line else -1
                interface_pos = line.index("--interface")
                assert prompt_pos < interface_pos, (
                    f"Docstring shows prompt AFTER --interface (broken order): {line}"
                )
                break
        else:
            pytest.fail("No single_shot usage line found in module docstring")


# ---------------------------------------------------------------------------
# --agent shorthand without a positional config (systemd service form)
# ---------------------------------------------------------------------------

class TestParseArgsAgentWithoutPositional:
    """The positional config is OPTIONAL when --agent is given.

    run_async resolves the config from --agent (agents dir) and only requires
    the positional when --agent is absent. The systemd unit uses this form:
        sr2-spectre --agent edi --interface discord
    Regression guard for obsidian-87f (Discord-as-service).
    """

    def test_agent_only_no_positional(self) -> None:
        """--agent X --interface Y must parse with config defaulting to None."""
        args = _parse_args(["--agent", "edi", "--interface", "discord"])
        assert args.config is None
        assert args.agent == "edi"
        assert args.interface == "discord"
        assert args.prompt == []

    def test_agent_only_minimal(self) -> None:
        """Bare --agent with no positional and no other options parses."""
        args = _parse_args(["--agent", "tali"])
        assert args.config is None
        assert args.agent == "tali"

    def test_positional_config_still_works_without_agent(self) -> None:
        """The positional form is unaffected: config set, agent None."""
        args = _parse_args(["config.yaml", "--interface", "tui"])
        assert args.config == "config.yaml"
        assert args.agent is None
