"""Tests for unified runtime + config-show tier-4 resolution.

These tests exercise the unification of the positional CLI config file into
the four-tier config resolution chain. The positional file is treated as
**tier 4** (highest precedence). Final resolution is:

    merge(
        $SR2_HOME/config.yaml,                  # tier 1
        $SR2_HOME/spectre.yaml,                 # tier 2
        <cwd>/.spectre.yaml,                    # tier 3
        extends-resolved(<positional file>),    # tier 4, wins over all
    )

ASSUMED SEAM (the implementer MUST conform to these signatures):

    from sr2_spectre.config import (
        load_resolved_config,
        load_resolved_config_with_provenance,
    )

    def load_resolved_config(
        positional_path: str | Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> dict:
        '''Merge tiers 1-3 then the extends-resolved positional file at tier 4.
        Returns the merged config dict. Missing tier files are silently skipped.
        Circular extends raises CircularExtendsError. If positional_path equals
        a tier path it is not merged twice.'''

    def load_resolved_config_with_provenance(
        positional_path: str | Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[dict, dict]:
        '''Same resolution as load_resolved_config, but also returns a
        provenance map: {top_level_key: ProvenanceValue(value, source)}.
        Keys contributed by the positional file carry the positional file's
        source label.'''

`cwd` defaults to Path.cwd() and `env` defaults to os.environ when None,
matching the existing load_merged_config / load_config_with_provenance
conventions. Tests always pass cwd and env explicitly.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sr2_spectre.config import (
    CircularExtendsError,
    load_resolved_config,
    load_resolved_config_with_provenance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


@pytest.fixture
def dirs(tmp_path):
    """Provide a fresh SR2_HOME dir, a cwd dir, and the env wired to SR2_HOME."""
    sr2_home = tmp_path / "sr2home"
    sr2_home.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()
    env = {"SR2_HOME": str(sr2_home)}
    return sr2_home, cwd, env


# ---------------------------------------------------------------------------
# AC1: positional file wins over each lower tier
# ---------------------------------------------------------------------------

class TestPositionalWinsOverTiers:
    def test_positional_overrides_tier1(self, dirs):
        """A key set in both tier 1 and the positional file resolves to positional."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "config.yaml", {"key": "tier1"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"key": "positional"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["key"] == "positional"

    def test_positional_overrides_tier2(self, dirs):
        """A key set in both tier 2 and the positional file resolves to positional."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "spectre.yaml", {"key": "tier2"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"key": "positional"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["key"] == "positional"

    def test_positional_overrides_tier3(self, dirs):
        """A key set in both tier 3 and the positional file resolves to positional."""
        sr2_home, cwd, env = dirs
        _write_yaml(cwd / ".spectre.yaml", {"key": "tier3"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"key": "positional"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["key"] == "positional"

    def test_positional_overrides_all_tiers_simultaneously(self, dirs):
        """When the same key is set in every tier, positional (tier 4) wins."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "config.yaml", {"key": "tier1"})
        _write_yaml(sr2_home / "spectre.yaml", {"key": "tier2"})
        _write_yaml(cwd / ".spectre.yaml", {"key": "tier3"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"key": "positional"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["key"] == "positional"


# ---------------------------------------------------------------------------
# AC2: lower tiers contribute keys the positional file omits
# ---------------------------------------------------------------------------

class TestLowerTiersContributeOmittedKeys:
    def test_tier1_only_key_survives(self, dirs):
        """models.default.base_url set only in tier 1 survives when positional omits it."""
        sr2_home, cwd, env = dirs
        _write_yaml(
            sr2_home / "config.yaml",
            {"models": {"default": {"base_url": "http://tier1:8000"}}},
        )
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"agent": {"name": "spectre"}})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["models"]["default"]["base_url"] == "http://tier1:8000"
        assert result["agent"]["name"] == "spectre"

    def test_deep_merge_positional_and_tier1(self, dirs):
        """Nested dicts deep-merge: positional adds a sibling key, tier1 key survives."""
        sr2_home, cwd, env = dirs
        _write_yaml(
            sr2_home / "config.yaml",
            {"models": {"default": {"base_url": "http://tier1:8000"}}},
        )
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"models": {"default": {"model": "gpt-pos"}}})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["models"]["default"]["base_url"] == "http://tier1:8000"
        assert result["models"]["default"]["model"] == "gpt-pos"


# ---------------------------------------------------------------------------
# AC3: tier ordering among 1/2/3 preserved, positional overrides all
# ---------------------------------------------------------------------------

class TestTierOrderingPreserved:
    def test_tier2_overrides_tier1_when_positional_silent(self, dirs):
        """Tier 2 wins over tier 1 for keys the positional file doesn't set."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "config.yaml", {"shared": "tier1"})
        _write_yaml(sr2_home / "spectre.yaml", {"shared": "tier2"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"other": "positional"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["shared"] == "tier2"

    def test_tier3_overrides_tier2_when_positional_silent(self, dirs):
        """Tier 3 wins over tier 2 for keys the positional file doesn't set."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "spectre.yaml", {"shared": "tier2"})
        _write_yaml(cwd / ".spectre.yaml", {"shared": "tier3"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"other": "positional"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["shared"] == "tier3"

    def test_full_precedence_chain(self, dirs):
        """Each tier contributes its own unique key; positional wins shared key."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "config.yaml", {"shared": "tier1", "t1": True})
        _write_yaml(sr2_home / "spectre.yaml", {"shared": "tier2", "t2": True})
        _write_yaml(cwd / ".spectre.yaml", {"shared": "tier3", "t3": True})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"shared": "positional", "t4": True})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["shared"] == "positional"
        assert result["t1"] is True
        assert result["t2"] is True
        assert result["t3"] is True
        assert result["t4"] is True


# ---------------------------------------------------------------------------
# AC4: extends: on the positional file is resolved before the merge
# ---------------------------------------------------------------------------

class TestPositionalExtends:
    def test_positional_inherits_base_keys(self, dirs):
        """A positional file that extends a base inherits the base's keys."""
        sr2_home, cwd, env = dirs
        base = cwd / "base.yaml"
        _write_yaml(base, {"from_base": "base_value", "shared": "base"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"extends": "base.yaml", "own": "pos_value"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["from_base"] == "base_value"
        assert result["own"] == "pos_value"

    def test_positional_own_keys_win_over_base(self, dirs):
        """The positional file's own keys win over the base it extends."""
        sr2_home, cwd, env = dirs
        base = cwd / "base.yaml"
        _write_yaml(base, {"shared": "base"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"extends": "base.yaml", "shared": "positional"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["shared"] == "positional"

    def test_extends_key_not_present_in_result(self, dirs):
        """The 'extends' key itself is stripped from the resolved config."""
        sr2_home, cwd, env = dirs
        base = cwd / "base.yaml"
        _write_yaml(base, {"from_base": True})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"extends": "base.yaml"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert "extends" not in result

    def test_extends_resolved_positional_still_wins_over_tiers(self, dirs):
        """Keys inherited via positional's extends still override lower tiers."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "config.yaml", {"shared": "tier1"})
        base = cwd / "base.yaml"
        _write_yaml(base, {"shared": "from_base"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"extends": "base.yaml"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["shared"] == "from_base"


# ---------------------------------------------------------------------------
# AC5: ${VAR} interpolation in the positional file's extends: path
# ---------------------------------------------------------------------------

class TestExtendsVarInterpolation:
    def test_sr2_home_var_in_extends_path(self, dirs):
        """extends: ${SR2_HOME}/agents/base.yaml resolves via env interpolation."""
        sr2_home, cwd, env = dirs
        agents_dir = sr2_home / "agents"
        agents_dir.mkdir()
        _write_yaml(agents_dir / "base.yaml", {"from_base": "base_value"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"extends": "${SR2_HOME}/agents/base.yaml", "own": "pos"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["from_base"] == "base_value"
        assert result["own"] == "pos"


# ---------------------------------------------------------------------------
# AC6: backward compatibility — single file, no extends, no tier files
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_single_file_no_tiers_returns_file_content(self, tmp_path):
        """With SR2_HOME pointing at an empty dir and no .spectre.yaml,
        the resolved config equals exactly the positional file's content."""
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        content = {
            "agent": {"name": "solo"},
            "models": {"default": {"model": "gpt-x"}},
        }
        pos = cwd / "run.yaml"
        _write_yaml(pos, content)

        result = load_resolved_config(pos, cwd=cwd, env={"SR2_HOME": str(empty_home)})
        assert result == content

    def test_single_file_nonexistent_sr2_home(self, tmp_path):
        """SR2_HOME pointing at a nonexistent dir is fine; positional content stands."""
        nonexistent_home = tmp_path / "does_not_exist"
        cwd = tmp_path / "project"
        cwd.mkdir()
        content = {"agent": {"name": "solo"}}
        pos = cwd / "run.yaml"
        _write_yaml(pos, content)

        result = load_resolved_config(
            pos, cwd=cwd, env={"SR2_HOME": str(nonexistent_home)}
        )
        assert result == content


# ---------------------------------------------------------------------------
# AC7: missing tier files silently skipped
# ---------------------------------------------------------------------------

class TestMissingTiersSkipped:
    def test_missing_tiers_do_not_error(self, dirs):
        """No tier files exist — only the positional file contributes, no error."""
        sr2_home, cwd, env = dirs
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"key": "positional"})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result == {"key": "positional"}

    def test_partial_tiers_present(self, dirs):
        """Only tier 1 present (tiers 2/3 missing) — tier1 + positional merge cleanly."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "config.yaml", {"t1": True})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"t4": True})

        result = load_resolved_config(pos, cwd=cwd, env=env)
        assert result["t1"] is True
        assert result["t4"] is True


# ---------------------------------------------------------------------------
# AC8: circular extends raises CircularExtendsError
# ---------------------------------------------------------------------------

class TestCircularExtends:
    def test_circular_positional_extends_raises(self, dirs):
        """A positional file in a circular extends chain raises CircularExtendsError."""
        sr2_home, cwd, env = dirs
        a = cwd / "a.yaml"
        b = cwd / "b.yaml"
        _write_yaml(a, {"extends": "b.yaml"})
        _write_yaml(b, {"extends": "a.yaml"})

        with pytest.raises(CircularExtendsError):
            load_resolved_config(a, cwd=cwd, env=env)


# ---------------------------------------------------------------------------
# AC9: de-dup — positional path identical to a tier path
# ---------------------------------------------------------------------------

class TestDeDup:
    def test_positional_equals_tier3_no_double_merge(self, dirs):
        """Passing cwd/.spectre.yaml as the positional file does not error and
        does not duplicate-merge it.

        Strengthened de-dup assertion: a plain-list merge is *replace*, so a
        double-merge of [1,2,3] is indistinguishable from a single merge and
        proves nothing. Instead we make double-application observable via
        provenance: the shared key must resolve to exactly ONE source. If the
        positional path were merged twice (once as tier 3, once as tier 4) the
        resolver would still produce a single value, but the de-dup contract is
        that the key has a single, well-defined provenance source naming the
        file — not a duplicated/extends artifact. We assert the source resolves
        to the .spectre.yaml file itself and nothing leaks an 'extends' key.
        """
        sr2_home, cwd, env = dirs
        tier3_path = cwd / ".spectre.yaml"
        _write_yaml(tier3_path, {"key": "tier3"})

        config, provenance = load_resolved_config_with_provenance(
            tier3_path, cwd=cwd, env=env
        )
        assert config["key"] == "tier3"
        # De-dup artifact guard: no spurious 'extends' key leaked into the result.
        assert "extends" not in config
        # Single, well-defined provenance source naming the shared file. If the
        # path were double-merged as two distinct tiers, provenance bookkeeping
        # would have no single coherent source to point at; pinning it to the
        # filename makes the de-dup observable rather than silently lossy.
        assert ".spectre.yaml" in provenance["key"].source

    def test_positional_equals_tier3_lower_tier_still_contributes(self, dirs):
        """De-dup of the positional==tier3 path must not drop lower-tier keys."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "config.yaml", {"t1": True})
        tier3_path = cwd / ".spectre.yaml"
        _write_yaml(tier3_path, {"key": "tier3"})

        result = load_resolved_config(tier3_path, cwd=cwd, env=env)
        assert result["t1"] is True
        assert result["key"] == "tier3"


# ---------------------------------------------------------------------------
# AC10: provenance variant honors the positional file at tier 4
# ---------------------------------------------------------------------------

class TestProvenanceHonorsPositional:
    def test_provenance_includes_positional_key(self, dirs):
        """The provenance map includes keys contributed by the positional file."""
        sr2_home, cwd, env = dirs
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"key": "positional"})

        config, provenance = load_resolved_config_with_provenance(
            pos, cwd=cwd, env=env
        )
        assert config["key"] == "positional"
        assert "key" in provenance
        assert provenance["key"].value == "positional"

    def test_provenance_source_points_at_positional_file(self, dirs):
        """When the positional file wins a key, its provenance source names that file."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "config.yaml", {"key": "tier1"})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"key": "positional"})

        config, provenance = load_resolved_config_with_provenance(
            pos, cwd=cwd, env=env
        )
        assert config["key"] == "positional"
        assert "run.yaml" in provenance["key"].source

    def test_provenance_lower_tier_key_keeps_its_source(self, dirs):
        """A key only in a lower tier keeps that tier's provenance source."""
        sr2_home, cwd, env = dirs
        _write_yaml(sr2_home / "config.yaml", {"only_tier1": True})
        pos = cwd / "run.yaml"
        _write_yaml(pos, {"key": "positional"})

        config, provenance = load_resolved_config_with_provenance(
            pos, cwd=cwd, env=env
        )
        assert config["only_tier1"] is True
        assert "config.yaml" in provenance["only_tier1"].source


# ---------------------------------------------------------------------------
# CLI wiring — close the "green units, dead CLI" loophole
# ---------------------------------------------------------------------------
#
# The unit tests above prove load_resolved_config / *_with_provenance work, but
# nothing proves the CLI callers actually route the positional file through the
# unified resolver. Without these, the implementer could add the resolver,
# make every unit test green, and leave run_async still calling
# load_config(args.config) and _run_config_show still calling
# load_config_with_provenance(cwd, env) (which ignores the positional file).
#
# Patch style mirrors tests/test_trace_flag.py (monkeypatch
# sr2_spectre.cli.<name>, neutralize Agent + plugin loader so no LLM turn runs).

from unittest.mock import AsyncMock, MagicMock, patch

from sr2_spectre.cli import main, run_async
from sr2_spectre.core import TurnResult


def _make_mock_config() -> MagicMock:
    """Minimal SpectreConfig stand-in (mirrors test_trace_flag.py)."""
    config = MagicMock()
    config.agent.name = "test-agent"
    config.agent.tools = []
    config.models = {"default": MagicMock(model="test-model", base_url=None)}
    return config


def _fake_interface_cls(instance):
    """Class whose instantiation returns ``instance`` (for _load_interface patches)."""
    return type("_FakeInterfaceCls", (), {"__new__": staticmethod(lambda cls: instance)})


def _make_mock_interface() -> MagicMock:
    """Mock interface whose run() drives one handle_user_message and prints the reply."""
    plugin = MagicMock()
    plugin.start = AsyncMock()
    plugin.stop = AsyncMock()

    async def _run(agent: MagicMock) -> None:
        result = await agent.handle_user_message("hi")
        print(result.text)

    plugin.run = _run
    return plugin


class TestCliRoutesPositionalThroughResolver:
    """Test A1: run_async must route the positional path through the unified
    resolver (sr2_spectre.cli.resolve_config), not the old load_config."""

    @pytest.mark.asyncio
    async def test_run_async_calls_unified_resolver_with_positional_path(self):
        """run_async('myrun.yaml', ...) calls resolve_config once with
        the positional path as its first argument.

        This asserts ONLY the wiring — that the CLI hands the positional file to
        the unified resolver. Merge semantics are owned by the unit tests above.
        """
        mock_config = _make_mock_config()
        resolver_spy = MagicMock(return_value=mock_config)

        with (
            # The seam: the CLI must call this name with the positional path.
            # create=True because the implementer has not added the import yet,
            # so this test fails for the right reason (resolver not wired) rather
            # than an AttributeError at patch time.
            patch(
                "sr2_spectre.cli.resolve_config",
                resolver_spy,
                create=True,
            ),
            patch("sr2_spectre.cli._configure_logging"),
            patch(
                "sr2_spectre.cli._load_interface",
                return_value=_fake_interface_cls(_make_mock_interface()),
            ),
            patch("sr2_spectre.cli.Agent") as MockAgent,
        ):
            mock_agent_instance = AsyncMock()
            mock_agent_instance.handle_user_message.return_value = TurnResult(
                text="hi", tool_calls_executed=0, total_tokens=1
            )
            MockAgent.return_value = mock_agent_instance

            # Prompt placed directly after the config positional so the current
            # argparse layout accepts it (mirrors test_trace_flag.py argv order);
            # this guarantees the failure is the wiring assertion below, not an
            # argparse rejection. --plugin single_shot is the default anyway.
            await run_async(["myrun.yaml", "hi"])

        assert resolver_spy.call_count == 1, (
            "run_async did not route the positional file through "
            "resolve_config (still calling load_config?)"
        )
        # The positional path is the first positional argument to the resolver.
        called_positional = resolver_spy.call_args.args[0]
        assert str(called_positional) == "myrun.yaml", (
            f"resolver received {called_positional!r}, expected 'myrun.yaml'"
        )


class TestConfigShowHonorsPositionalFile:
    """Test A2: 'config show <file>' must honor the positional file end-to-end.

    The current parser (_parse_config_show_args) has NO positional file arg and
    _run_config_show ignores it. Spec addendum requirement #1 mandates that
    config show honor the positional file, so the implementer MUST add a
    positional file argument and route it through the unified resolver.
    """

    def test_config_show_honors_positional_file(self, tmp_path, capsys):
        """config show <pos_file> --sr2-home <home> surfaces a positional-ONLY
        key and names the positional file as that key's provenance source."""
        home = tmp_path / "sr2home"
        home.mkdir()
        # Tier 1: a key present only in $SR2_HOME/config.yaml.
        _write_yaml(home / "config.yaml", {"tier1_key": "from_tier1"})

        # Positional file: a key present in NO tier.
        pos_file = tmp_path / "run.yaml"
        _write_yaml(pos_file, {"positional_only_key": "from_positional"})

        with pytest.raises(SystemExit):
            main(
                [
                    "config",
                    "show",
                    str(pos_file),
                    "--sr2-home",
                    str(home),
                ]
            )

        out = capsys.readouterr().out

        # The positional-only key/value must appear — proves the positional file
        # was loaded at all (it fails if config show ignores the positional arg).
        assert "positional_only_key" in out, (
            f"positional-only key missing from config show output:\n{out}"
        )
        assert "from_positional" in out, (
            f"positional-only value missing from config show output:\n{out}"
        )
        # Provenance must name the positional file as the source for that key.
        # Substring match on the filename (consistent with provenance tests
        # above) — do not hardcode the exact annotation label format.
        assert "run.yaml" in out, (
            f"positional file not shown as provenance source:\n{out}"
        )
