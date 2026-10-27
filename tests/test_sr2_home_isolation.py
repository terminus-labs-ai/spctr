"""Guard: no default under ~/.sr2 may be hardcoded — all follow SR2_HOME.

spc-160 routed the reminders/schedules stores through
``resolve_sr2_home()``; spc-1004 does the same for the schedule log
directories, the plan resolver's plans/knowledge roots, the complete_step
plans root and the CLI ``--agent`` directory.

The autouse conftest fixture ``_isolate_sr2_home`` points SR2_HOME at a
per-test temp dir, so a default that resolves through ``resolve_sr2_home()``
lands outside the real ``~/.sr2`` — and a default that still hardcodes
``Path.home() / ".sr2"`` lands inside it, failing every test here.

The ``~/.sr2-spectre`` dash-dir (dispatch log, session.json, provenance db)
is a different directory and deliberately out of scope.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from sr2.config.models import ResolverConfig

from sr2_spectre.cli import resolve_agent_config_path
from sr2_spectre.planning.resolver import PlanResolver
from sr2_spectre.run_log import SessionLogManager
from sr2_spectre.schedules.agent import log_dir_for as agent_log_dir_for
from sr2_spectre.schedules.command import log_dir_for as command_log_dir_for
from sr2_spectre.schedules.command import run_command
from sr2_spectre.tools.builtins.complete_step import CompleteStepTool


def real_home() -> Path:
    """The machine's real ``~/.sr2`` — nothing may be written under it."""
    return Path("~/.sr2").expanduser().resolve()


# ---------------------------------------------------------------------------
# Pre-existing guard (spc-160 class): session logs
# ---------------------------------------------------------------------------

def test_session_log_directory_is_not_real_sr2_home():
    assert not SessionLogManager().directory.is_relative_to(real_home())


# ---------------------------------------------------------------------------
# spc-1004: the defaults themselves resolve outside the real ~/.sr2
# ---------------------------------------------------------------------------

def test_schedules_agent_log_dir_for_follows_sr2_home():
    """agent.log_dir_for('x') roots at $SR2_HOME/logs/schedules, not ~/.sr2."""
    resolved = agent_log_dir_for("x")
    assert not resolved.is_relative_to(real_home())
    assert resolved.parts[-4:-1] == ("logs", "schedules", "x")
    assert resolved.name == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Pinned to the fixture's SR2_HOME, so a path merely *outside* ~/.sr2
    # (e.g. /tmp) could not pass by accident.
    from sr2_spectre.config import resolve_sr2_home

    assert resolved.is_relative_to(resolve_sr2_home().resolve())


def test_schedules_command_log_dir_for_follows_sr2_home():
    """command.log_dir_for('x') roots at $SR2_HOME/logs/schedules, not ~/.sr2."""
    resolved = command_log_dir_for("x")
    assert not resolved.is_relative_to(real_home())
    assert resolved.parts[-4:-1] == ("logs", "schedules", "x")
    assert resolved.name == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from sr2_spectre.config import resolve_sr2_home

    assert resolved.is_relative_to(resolve_sr2_home().resolve())


def test_complete_step_default_plans_root_follows_sr2_home():
    """CompleteStepTool() with no argument resolves outside the real ~/.sr2."""
    resolved = CompleteStepTool()._plans_root
    assert not resolved.is_relative_to(real_home())
    assert resolved == resolve_expected() / "plans"


def test_plan_resolver_default_plans_root_follows_sr2_home():
    """Empty-ish resolver config → plans_root defaults under $SR2_HOME."""
    resolver = PlanResolver(ResolverConfig(type="plan", config={"project": "p1"}))
    resolved = resolver._plans_root
    assert not resolved.is_relative_to(real_home())
    assert resolved == resolve_expected() / "plans"


def test_plan_resolver_default_knowledge_root_follows_sr2_home():
    """Explicit project, no knowledge_root → default under $SR2_HOME."""
    resolver = PlanResolver(ResolverConfig(type="plan", config={"project": "p1"}))
    resolved = resolver._knowledge_root
    assert not resolved.is_relative_to(real_home())
    assert resolved == resolve_expected() / "knowledge" / "p1"


def test_plan_resolver_auto_default_knowledge_root_follows_sr2_home():
    """``__auto__`` project: the per-turn knowledge root also follows SR2_HOME."""
    resolver = PlanResolver(
        ResolverConfig(type="plan", config={"project": "__auto__"})
    )
    resolved = resolver._resolve_knowledge_root("auto-proj")
    assert not resolved.is_relative_to(real_home())
    assert resolved == resolve_expected() / "knowledge" / "auto-proj"


def test_cli_agents_dir_default_follows_sr2_home():
    """resolve_agent_config_path() default directory is $SR2_HOME/agents."""
    path = resolve_agent_config_path("edi")
    assert not path.is_relative_to(real_home())
    assert path == resolve_expected() / "agents" / "edi.yaml"


# ---------------------------------------------------------------------------
# Regression: the eager mkdir no longer escapes SR2_HOME
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_command_does_not_write_under_real_sr2_home(tmp_path):
    """run_command() mkdirs its log dir eagerly; that dir must be in SR2_HOME."""
    probe_agent = "spc1004probe"
    escaped_root = real_home() / "logs" / "schedules" / probe_agent

    result = await run_command("true", agent=probe_agent, cwd=str(tmp_path))

    assert not escaped_root.exists(), (
        f"run_command wrote outside SR2_HOME: {escaped_root}"
    )
    log = Path(result.log_path)
    assert log.exists()
    assert log.is_relative_to(resolve_expected() / "logs" / "schedules" / probe_agent)


def resolve_expected() -> Path:
    """The active SR2_HOME (the fixture's temp dir), resolved."""
    from sr2_spectre.config import resolve_sr2_home

    return resolve_sr2_home().resolve()
