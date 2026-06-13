"""CLI entry point for sr2-spectre.

Usage:
    sr2-spectre config.yaml "What is the weather?" --interface single_shot
    sr2-spectre config.yaml --interface tui
    sr2-spectre config show [--no-provenance] [--sr2-home PATH]

Note: positional arguments (config, prompt) must appear contiguously before
options.  The prompt positional is consumed only by single_shot mode.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LITELLM_LOG", "ERROR")

from sr2.pipeline.tracing import CollectingTracer, render_compiled_request, render_trace
from sr2_spectre.agent import Agent
from sr2_spectre.config import SpectreConfig
from sr2_spectre.config import load_resolved_config as _resolve_merged_config

logger = logging.getLogger(__name__)

_DEFAULT_LOG_FILE = Path.home() / ".sr2-spectre" / "spectre.log"


def _parse_config_show_args(argv: list[str]) -> argparse.Namespace:
    """Parse arguments for 'spectre config show' subcommand."""
    parser = argparse.ArgumentParser(
        prog="sr2-spectre config show",
        description="Show the resolved configuration with provenance annotations.",
        add_help=True,
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        default=False,
        help="Output plain YAML without source annotations",
    )
    parser.add_argument(
        "--sr2-home",
        type=str,
        default=None,
        help="Override SR2_HOME directory",
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default=None,
        help="Optional positional config file (merged at tier 4, highest priority)",
    )
    return parser.parse_args(argv)


def _run_config_show(argv: list[str]) -> int:
    """Execute the 'config show' dry-run command. Returns exit code (0 or 1).

    Uses SpectreConfig (pydantic) as the single validation source of truth,
    matching the real startup path in resolve_config().
    """
    from sr2_spectre.config import (
        format_dry_run,
        load_config_with_provenance,
        load_resolved_config_with_provenance,
    )
    from pydantic import ValidationError

    args = _parse_config_show_args(argv)

    env = dict(os.environ)
    if args.sr2_home:
        env["SR2_HOME"] = args.sr2_home

    try:
        if args.config_file:
            config, provenance = load_resolved_config_with_provenance(
                args.config_file, cwd=Path.cwd(), env=env
            )
        else:
            config, provenance = load_config_with_provenance(cwd=Path.cwd(), env=env)
    except Exception as exc:
        print(f"# Error loading config: {exc}", file=sys.stdout)
        return 1

    # Validate through the same path as real startup: build SpectreConfig.
    # This ensures dry-run and startup agree on validity.
    errors: list[str] = []
    try:
        SpectreConfig(**config)
    except ValidationError as exc:
        errors = [f"{err['loc'][0] if err['loc'] else '?'}: {err['msg']}" for err in exc.errors()]

    output = format_dry_run(
        config=config,
        provenance=provenance,
        errors=errors,
        show_provenance=not args.no_provenance,
    )
    print(output, end="")

    return 1 if errors else 0


def resolve_agent_config_path(
    agent_name: str,
    agents_dir: Path | None = None,
) -> Path:
    """Resolve an agent name to its config YAML path.

    ``--agent edi`` → ``<agents_dir>/edi.yaml``

    Args:
        agent_name: Short agent name (e.g. "edi", "liara").
        agents_dir: Directory containing agent YAML files. Defaults to
            ``~/.sr2/agents/``.

    Returns:
        Absolute resolved Path to the agent's YAML config file.

    Raises:
        FileNotFoundError: If the resolved YAML file does not exist.
    """
    if agents_dir is None:
        agents_dir = Path.home() / ".sr2" / "agents"
    else:
        agents_dir = Path(agents_dir).expanduser().resolve()

    config_path = agents_dir / f"{agent_name}.yaml"
    return config_path.resolve()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sr2-spectre",
        description="SR2 Spectre — full agent runtime powered by SR2",
        epilog=(
            "Examples:\n"
            "  sr2-spectre config.yaml 'What is the weather?' --interface single_shot\n"
            "  sr2-spectre config.yaml --interface tui\n"
            "  sr2-spectre config.yaml 'Hello' --trace\n"
            "\n"
            "Note: positional arguments (config, prompt) must appear before\n"
            "options.  The prompt positional is consumed only by single_shot mode.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML config file (optional when --agent is set)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent name shorthand — resolves to <agents_dir>/<name>.yaml. "
             "Agents dir defaults to ~/.sr2/agents/ (override with --agents-dir).",
    )
    parser.add_argument(
        "--agents-dir",
        type=str,
        default=None,
        help="Directory containing agent YAML files (default: ~/.sr2/agents/)",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default="single_shot",
        help="Interface to run: single_shot, tui (default: single_shot)",
    )

    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Override session ID",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--log-file",
        default=str(_DEFAULT_LOG_FILE),
        help=f"Log file path (default: {_DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text (for single_shot mode)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Print pipeline firing timeline after reply",
    )
    return parser.parse_args(argv)


def _load_interface(interface_name: str, **kwargs: Any) -> Any:
    """Load an interface class by name."""
    known_interfaces = {
        "single_shot": "sr2_spectre.interfaces.single_shot.SingleShotInterface",
        "tui": "sr2_spectre.interfaces.tui.TUIInterface",
        "discord": "sr2_spectre.interfaces.discord.interface.DiscordInterface",
    }

    class_path = known_interfaces.get(interface_name, interface_name)
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)



def _configure_logging(level: str, log_file: str) -> None:
    os.environ.setdefault("LITELLM_LOG", "WARNING")

    numeric = getattr(logging, level)
    file_path = Path(log_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(numeric)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logging.basicConfig(level=numeric, handlers=[file_handler, console_handler], force=True)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


def resolve_config(
    positional_path: str | Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> SpectreConfig:
    """Resolve the unified 4-tier config and build a SpectreConfig.

    Thin typed wrapper over ``config.load_resolved_config`` (which owns the
    merge logic and is exercised by the resolution test suite): merges
    $SR2_HOME/config.yaml, $SR2_HOME/spectre.yaml, <cwd>/.spectre.yaml, then
    the extends-resolved positional file (tier 4, highest priority), and
    validates the result into a SpectreConfig.
    """
    merged = _resolve_merged_config(positional_path, cwd=cwd, env=env)
    return SpectreConfig(**merged)


async def run_async(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log_level, args.log_file)

    # Resolve the effective config path: --agent shorthand overrides positional arg.
    if args.agent:
        config_path = resolve_agent_config_path(
            args.agent, agents_dir=Path(args.agents_dir) if args.agents_dir else None
        )
    elif args.config:
        config_path = args.config
    else:
        print("Error: either --agent or a positional config path is required.", file=sys.stderr)
        sys.exit(2)

    logger.info("SR2 Spectre starting")

    config = resolve_config(
        config_path, cwd=Path.cwd(), env=dict(os.environ)
    )
    logger.info(
        "Agent: %s | model: %s",
        config.agent.name,
        config.models["default"].model,
    )

    tracer = CollectingTracer() if args.trace else None

    if tracer is not None:
        agent = Agent(
            config=config,
            session_id=args.session_id,
            tracer=tracer,
        )
    else:
        agent = Agent(
            config=config,
            session_id=args.session_id,
        )

    interface_name = args.interface

    interface_kwargs: dict[str, Any] = {}
    if interface_name == "single_shot" and args.prompt:
        interface_kwargs["prompt"] = " ".join(args.prompt)
    if interface_name == "discord" and config.discord is not None:
        interface_kwargs["config"] = config.discord

    await agent.initialize()

    try:
        interface = _load_interface(interface_name, **interface_kwargs)

        await interface.start(agent)
        await interface.run(agent)
        await interface.stop()
    finally:
        await agent.aclose()

    if tracer is not None:
        print(render_trace(tracer.get_trace()))
        if tracer.compiled_request is not None:
            print(render_compiled_request(tracer.compiled_request))

    logger.info("SR2 Spectre shutdown")


def main(argv: list[str] | None = None) -> None:
    """Entry point. Dispatches 'config show' before agent startup."""
    args = argv if argv is not None else sys.argv[1:]

    # Dispatch 'config show' subcommand before agent startup
    if len(args) >= 2 and args[0] == "config" and args[1] == "show":
        exit_code = _run_config_show(args[2:])
        sys.exit(exit_code)

    asyncio.run(run_async(argv))


if __name__ == "__main__":
    main()
