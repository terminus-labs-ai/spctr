"""Host-local agent-config tier (spctr-4, spec AC 1 build half).

An agent config at ``$SR2_HOME/agents/<name>.yaml`` is merged with a sibling
``<name>.local.yaml`` after every other tier (highest precedence) when that
file exists. A positional config outside ``$SR2_HOME/agents/`` never gains the
tier, even when a sibling ``*.local.yaml`` exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spctr.cli import resolve_agent_config_path
from spctr.config import (
    format_dry_run,
    load_resolved_config,
    load_resolved_config_with_provenance,
)

TRACKED = {
    "agent": {
        "name": "miranda",
        "reminders": {"enabled": True, "timezone": "Europe/Lisbon"},
        "schedules": {"enabled": True, "timezone": "Europe/Lisbon"},
    },
    "models": {
        "default": {
            "model": "qwen",
            "base_url": "https://lan.example/v1",
        }
    },
    "active_model": "default",
}

LOCAL = {
    "agent": {
        "reminders": {"enabled": False},
        "schedules": {"enabled": False},
    },
    "models": {
        "default": {"base_url": "http://kodiak.tailnet.example:11434/v1"},
    },
}


@pytest.fixture
def sr2_home(tmp_path: Path) -> Path:
    home = tmp_path / "sr2-home"
    (home / "agents").mkdir(parents=True)
    return home


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    return cwd


def _write(path: Path, data: dict) -> bytes:
    text = yaml.safe_dump(data, sort_keys=False)
    path.write_text(text)
    return text.encode()


def _env(sr2_home: Path) -> dict[str, str]:
    return {"SR2_HOME": str(sr2_home)}


# AC1-r1: local values override the tracked file; tracked keys survive and
# the tracked file on disk is untouched.
def test_local_tier_overrides_tracked_agent_config(sr2_home, workdir, monkeypatch):
    tracked = sr2_home / "agents" / "miranda.yaml"
    tracked_bytes = _write(tracked, TRACKED)
    _write(sr2_home / "agents" / "miranda.local.yaml", LOCAL)

    # Real producer: the CLI's agent-name resolution feeds the loader.
    monkeypatch.setenv("SR2_HOME", str(sr2_home))
    agent_path = resolve_agent_config_path("miranda")
    assert agent_path == tracked.resolve()

    config = load_resolved_config(agent_path, cwd=workdir, env=_env(sr2_home))

    assert config == {
        "agent": {
            "name": "miranda",
            "reminders": {"enabled": False, "timezone": "Europe/Lisbon"},
            "schedules": {"enabled": False, "timezone": "Europe/Lisbon"},
        },
        "models": {
            "default": {
                "model": "qwen",
                "base_url": "http://kodiak.tailnet.example:11434/v1",
            }
        },
        "active_model": "default",
    }
    assert tracked.read_bytes() == tracked_bytes


# AC1-r2: no local file -> resolution is exactly the tracked content.
def test_absent_local_file_falls_back_to_tracked(sr2_home, workdir):
    tracked = sr2_home / "agents" / "miranda.yaml"
    _write(tracked, TRACKED)

    config, provenance = load_resolved_config_with_provenance(
        tracked, cwd=workdir, env=_env(sr2_home)
    )

    assert config == TRACKED
    assert {pv.source for pv in provenance.values()} == {
        f"{tracked.resolve()} (global-spctr)"
    }


# AC1-r3: nested dicts deep-merge; named lists merge by name; other lists
# are replaced.
def test_local_tier_uses_existing_merge_semantics(sr2_home, workdir):
    tracked = sr2_home / "agents" / "miranda.yaml"
    _write(tracked, {
        "agent": {
            "name": "miranda",
            "tools": [
                {"name": "shell", "timeout": 30},
                {"name": "web", "timeout": 10},
            ],
        },
        "models": {"default": {"model": "qwen", "params": {"temperature": 0.7, "top_p": 0.9}}},
        "extra_args": ["--a", "--b"],
    })
    _write(sr2_home / "agents" / "miranda.local.yaml", {
        "agent": {"tools": [{"name": "web", "timeout": 99}]},
        "models": {"default": {"params": {"temperature": 0.1}}},
        "extra_args": ["--c"],
    })

    config = load_resolved_config(tracked, cwd=workdir, env=_env(sr2_home))

    assert config == {
        "agent": {
            "name": "miranda",
            "tools": [
                {"name": "shell", "timeout": 30},
                {"name": "web", "timeout": 99},
            ],
        },
        "models": {"default": {"model": "qwen", "params": {"temperature": 0.1, "top_p": 0.9}}},
        "extra_args": ["--c"],
    }


# AC1-r4: provenance names the local file for every key it overrides, and
# the dry-run labels it host-local, never "(extends)".
def test_local_tier_provenance_and_dry_run_label(sr2_home, workdir):
    tracked = sr2_home / "agents" / "miranda.yaml"
    local = sr2_home / "agents" / "miranda.local.yaml"
    _write(tracked, TRACKED)
    _write(local, LOCAL)

    config, provenance = load_resolved_config_with_provenance(
        tracked, cwd=workdir, env=_env(sr2_home)
    )

    local_label = f"{local.resolve()} (host-local)"
    tracked_label = f"{tracked.resolve()} (global-spctr)"
    assert provenance["agent"].source == local_label
    assert provenance["models"].source == local_label
    assert provenance["active_model"].source == tracked_label

    report = format_dry_run(config, provenance, errors=[])
    assert f"agent:  # ← {local_label}" in report
    assert f"models:  # ← {local_label}" in report
    assert f"active_model: default  # ← {tracked_label}" in report
    assert "(extends)" not in report


# AC1-r6: a positional config outside $SR2_HOME/agents/ ignores its sibling
# *.local.yaml.
def test_local_tier_ignored_outside_agents_dir(sr2_home, tmp_path, workdir):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    pipeline = elsewhere / "pipeline.yaml"
    _write(pipeline, {"agent": {"name": "pipe"}, "active_model": "default"})
    _write(elsewhere / "pipeline.local.yaml", {"active_model": "local-only", "leak": True})

    config, provenance = load_resolved_config_with_provenance(
        pipeline, cwd=workdir, env=_env(sr2_home)
    )

    assert config == {"agent": {"name": "pipe"}, "active_model": "default"}
    assert all("pipeline.local.yaml" not in pv.source for pv in provenance.values())


# Contract step 3: passing the local file itself as the positional config
# merges it once and adds no further tier.
def test_local_file_as_positional_merges_once(sr2_home, workdir):
    local = sr2_home / "agents" / "miranda.local.yaml"
    _write(sr2_home / "agents" / "miranda.yaml", TRACKED)
    _write(local, LOCAL)

    config = load_resolved_config(local, cwd=workdir, env=_env(sr2_home))

    assert config == LOCAL


# A missing agent config still fails, even when its local sibling exists.
def test_missing_agent_config_still_raises_with_local_present(sr2_home, workdir):
    _write(sr2_home / "agents" / "miranda.local.yaml", LOCAL)

    with pytest.raises(FileNotFoundError):
        load_resolved_config(
            sr2_home / "agents" / "miranda.yaml", cwd=workdir, env=_env(sr2_home)
        )
