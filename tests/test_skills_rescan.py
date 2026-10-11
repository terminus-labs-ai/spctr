"""Per-message skills_dirs rescan (spc-110).

``Runtime.apply_config`` runs once per inbound message (Discord:
``_apply_agent_config`` → ``Agent.apply_config`` → ``Runtime.apply_config``),
and for most messages the config object is byte-for-byte unchanged — the
early return fires. Skill *files* however change without the config changing,
so the skills_dirs freshness check must run on EVERY ``apply_config`` call,
before that early return:

- added skill file (bundled ``<dir>/<name>/SKILL.md`` or flat ``<dir>/<name>.md``)
  becomes visible on the next call;
- removed file/folder disappears on the next call;
- edited content is served fresh on the next call;
- nothing changed → the rebuild path is not entered at all (no rebuild, no
  log noise);
- a failed rescan (unreadable directory, unreadable/mid-edit skill file)
  keeps the previous registry exactly in force, logs exactly one warning, and
  ``apply_config`` returns normally;
- the ``skill_registry`` OBJECT identity is stable across rebuilds because
  the auto-injected ``load_skill`` tool holds that object's reference;
- builtin behavior and the existing config-value-change rebuild path are
  preserved.

The seam under test is ``Runtime.apply_config`` called repeatedly with an
UNCHANGED config object, against real on-disk temp skills_dirs. mtimes are
forced with ``os.utime`` (never sleeps). The rebuild path is spied at
``Runtime._rebuild_skills`` — the design constraint names that exact method
as the only rebuild entry point, and the "not entered" / "exactly one
warning" criteria are impossible to pin through any narrower public seam.

Two mechanics in here are load-bearing, not cosmetic:

*Timeline.* Every mtime this file sets comes from a synthetic clock based at
``1_900_000_000`` (2030), strictly ABOVE the real wall clock. ``os.utime`` on
a file bumps that file's *parent directory* mtime to the real "now", so a
timeline below "now" can make a directory's recorded mtime go backwards
mid-test and a (mtime, size) change signature stops being monotonic — an
otherwise correct implementation then correctly decides nothing changed. Above
"now", every later event (synthetic or real) is newer than every earlier one,
and every timestamp in this file is drawn from a disjoint, non-repeating
range, so a signature flips exactly when the test intends it to.

*Explicit frontmatter.* Bundled ``SKILL.md`` files are seeded WITH a
frontmatter ``name`` equal to their directory name. Naming a bare
``SKILL.md`` from its directory is a fallback path inside
``discover_skills_in_dir``; the tests must not depend on a fallback for the
name (and therefore content key) they assert on.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RUNTIME_LOGGER = "sr2_spectre.runtime"

#: Base of the synthetic mtime timeline: 1.9e9 ≈ 2030, i.e. above the real
#: clock, for the monotonicity reason spelled out in the module docstring.
_BASE_MTIME = 1_900_000_000.0

#: Timestamps for edits made outside the ``rescan`` fixture's clock.
#:
#: They MUST NOT live inside the clock's range (``_BASE_MTIME + 1 … + 999``)
#: and must not be "now"-adjacent either: ``os.utime`` bumps the edited
#: file's PARENT DIRECTORY to the real "now", so an edit timestamp that
#: collides with a timestamp the clock hands out later makes the recorded
#: (path, mtime, size) signature non-monotonic and a restore-to-recorded-value
#: leg becomes a coin flip. This band sits above every value the clock can
#: produce and each constant is used exactly once in this file.
_T_EDIT_1 = _BASE_MTIME + 100_000.0  # bundled body edit v1 → v2
_T_EDIT_2 = _BASE_MTIME + 100_001.0  # flat frontmatter rename
_T_UNREADABLE = _BASE_MTIME + 100_003.0  # unreadable-file rewrite (AC 5)
_T_AWAY = _BASE_MTIME + 100_010.0  # "move the mtime away" leg
_T_OVERSHOOT = _BASE_MTIME + 100_011.0  # move again before restoring back


def _skill_text(name: str, body: str = "Body.") -> str:
    """Frontmatter block valid for either on-disk layout.

    Frontmatter is REQUIRED for the flat ``<dir>/<name>.md`` form. It is
    optional for bundled ``<dir>/<name>/SKILL.md`` — but only via a fallback,
    so this file always supplies it (``name`` = directory name) and never
    exercises discovery's fallback naming.
    """
    return (
        "---\n"
        f"name: {name}\n"
        f"description: Skill {name}\n"
        "version: 1.0.0\n"
        "---\n"
        f"{body}\n"
    )


def _skill_name_for(path: Path) -> str:
    """The name a skill path declares: directory name for a bundled
    ``…/<name>/SKILL.md``, file stem for a flat ``<dir>/<name>.md``."""
    return path.parent.name if path.name == "SKILL.md" else path.stem


def _seed_skill(path: Path, *, text: str, mtime: float) -> Path:
    """Write ``text`` to a skill path with an explicit, distinct mtime.

    Setting the mtime via ``os.utime`` keeps the rescan's change signature
    fully under test control — no sleeps, no filesystem timestamp-grain
    races. Prefer :func:`_seed_skill_file`, which adds the frontmatter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _seed_skill_file(path: Path, *, body: str = "Body.", mtime: float) -> Path:
    """Seed a skill whose frontmatter ``name`` comes from the path itself.

    Works for both recognised layouts, so bundled ``SKILL.md`` files carry
    ``name`` = directory name instead of relying on discovery's fallback.
    """
    return _seed_skill(
        path, text=_skill_text(_skill_name_for(path), body), mtime=mtime
    )


def _bump(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


def _pipeline_dict(system_text: str = "You are helpful.") -> dict:
    return {
        "layers": [
            {
                "name": "system",
                "target": "system",
                "resolvers": [{"type": "static", "config": {"text": system_text}}],
            },
            {
                "name": "tools",
                "target": "tools",
                "resolvers": [],
                "tool_providers": [{"type": "spectre_tools"}],
            },
            {
                "name": "conversation",
                "target": "messages",
                "resolvers": [{"type": "session"}, {"type": "input"}],
            },
        ]
    }


def _make_config(
    *,
    skills_dirs: list[str] | None = None,
    default_skills: bool = True,
    system_text: str = "You are helpful.",
) -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(
            name="test",
            tools=[],
            skills=[],
            skills_dirs=skills_dirs or [],
            default_skills=default_skills,
        ),
        models={"default": ModelConfig(model="test-model", base_url="http://test:8000")},
        pipeline=_pipeline_dict(system_text),
    )


def _warning_records(caplog, needle: str | None = None) -> list[logging.LogRecord]:
    """WARNING records from anywhere in the ``sr2_spectre`` logger tree.

    Deliberately scoped to the ``sr2_spectre`` ROOT rather than one logger
    name: the contract says "exactly one warning" about a failed rescan
    without naming which module logs it (runtime vs skills.core both
    qualify), while the prefix filter still excludes third-party warnings —
    e.g. LiteLLM emits a couple at import/setup on some runs, which would
    otherwise make "exactly one" fail for reasons unrelated to this feature.
    Do not narrow this to a single logger name.
    """
    records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name.split(".")[0] == "sr2_spectre"
    ]
    if needle is not None:
        records = [r for r in records if needle in r.getMessage()]
    return records


@pytest.fixture
def runtime_factory():
    """Build Runtimes with the LLM construction stubbed out (existing pattern)."""

    def _build(config: SpectreConfig):
        from sr2_spectre.runtime import Runtime

        with patch("sr2_spectre.live_llm.LiteLLMCallable"):
            return Runtime(config=config)

    return _build


@pytest.fixture
def rescan(runtime_factory, tmp_path):
    """A Runtime whose skills_dirs are live temp dirs, plus rescan helpers.

    ``run()`` drives ``apply_config`` with the SAME config object — exactly
    what the Discord interface does per message when nothing in the config
    changed. ``rebuild_calls`` counts entries into ``Runtime._rebuild_skills``.

    ``seed()`` takes the next tick of a module-wide synthetic clock (base
    ``1_900_000_000``) as the file's mtime and wraps ``body`` in frontmatter
    naming the skill after its path. See the module docstring: both are
    required for the signature-flipping in AC 1–3 and AC 5 to be
    deterministic rather than clock-dependent.

    AC 4 note: the steady-state test below settles on a first message and
    therefore cannot distinguish "startup seeded the signature" from "nothing
    ever rescans". ``TestNonSkillsRuntimeUnaffected::test_first_message_after_startup_without_changes_is_cold``
    is the load-bearing half of AC 4 — it pins the bootstrap-seeding
    requirement (first message after startup costs zero rebuilds). Keep both.
    """
    dir_a = tmp_path / "skills-a"
    dir_b = tmp_path / "skills-b"
    dir_a.mkdir()
    dir_b.mkdir()

    # Synthetic mtime timeline for every file this fixture seeds. Ticks are
    # handed out once each (no collisions) and start above the real clock;
    # timestamps for out-of-fixture edits live in a disjoint band above this
    # range — see the ``_T_*`` constants.
    clock: Iterator[float] = iter([_BASE_MTIME + i for i in range(1, 1000)])

    def seed(path: Path, *, body: str = "Body.") -> Path:
        return _seed_skill_file(path, body=body, mtime=next(clock))

    config = _make_config(skills_dirs=[str(dir_a), str(dir_b)])
    runtime = runtime_factory(config)

    # Spy on the rebuild entry point. Patching the CLASS is safe here because
    # the spy only delegates to the real function and counts entries; the
    # design constraint names ``Runtime._rebuild_skills`` as the sole rebuild
    # path, which is exactly what AC 4 and the failure tests must pin.
    original = type(runtime)._rebuild_skills
    calls: list[int] = []

    def spy(self, cfg: SpectreConfig) -> None:  # noqa: ANN001 - class patch
        calls.append(1)
        original(self, cfg)

    patcher = patch.object(type(runtime), "_rebuild_skills", spy)
    patcher.start()
    try:
        yield {
            "runtime": runtime,
            "config": config,
            "dir_a": dir_a,
            "dir_b": dir_b,
            "seed": seed,
            "run": lambda: runtime.apply_config(config),
            "rebuild_calls": lambda: len(calls),
        }
    finally:
        patcher.stop()


# ---------------------------------------------------------------------------
# AC 1 — Added skill files appear on the next apply_config
# ---------------------------------------------------------------------------


class TestAddedSkillVisible:
    def test_bundled_skill_added_becomes_visible(self, rescan, caplog):
        """<dir>/<name>/SKILL.md dropped in mid-run shows up next call."""
        runtime = rescan["runtime"]
        assert "now-maintenance" not in runtime.skill_registry

        # Message 1: nothing changed — baseline stays clean.
        with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER):
            assert rescan["run"]() == []
        assert rescan["rebuild_calls"]() == 0

        # Operator creates the skill while the bot is running.
        rescan["seed"](
            rescan["dir_a"] / "now-maintenance" / "SKILL.md",
            body="# Maintenance\n\nRestart the bot weekly.\n",
        )

        # Message 2: the very next apply_config must serve it.
        assert rescan["run"]() == []
        assert "now-maintenance" in runtime.skill_registry
        content = runtime.skill_registry.get_content("now-maintenance")
        assert content is not None and "Restart the bot weekly" in content

    def test_flat_skill_added_becomes_visible(self, rescan):
        """<dir>/<name>.md (frontmatter form) in the other dir works too."""
        runtime = rescan["runtime"]
        rescan["run"]()  # settle

        rescan["seed"](
            rescan["dir_b"] / "flat-addition.md", body="Flat body."
        )

        assert rescan["run"]() == []
        content = runtime.skill_registry.get_content("flat-addition")
        assert content is not None
        assert "Flat body" in content


# ---------------------------------------------------------------------------
# AC 2 — Removed skill file/folder disappears on the next apply_config
# ---------------------------------------------------------------------------


class TestRemovedSkillGone:
    def test_deleted_skill_folder_disappears(self, rescan):
        runtime = rescan["runtime"]
        rescan["seed"](
            rescan["dir_a"] / "work-priority-review" / "SKILL.md",
            body="# Review\n\nRank the work.\n",
        )
        rescan["run"]()
        assert "work-priority-review" in runtime.skill_registry

        shutil.rmtree(rescan["dir_a"] / "work-priority-review")

        assert rescan["run"]() == []
        assert "work-priority-review" not in runtime.skill_registry

    def test_deleted_flat_file_disappears(self, rescan):
        runtime = rescan["runtime"]
        victim = rescan["seed"](rescan["dir_b"] / "gone-soon.md", body="Bye.")
        rescan["run"]()
        assert "gone-soon" in runtime.skill_registry

        victim.unlink()

        assert rescan["run"]() == []
        assert "gone-soon" not in runtime.skill_registry


# ---------------------------------------------------------------------------
# AC 3 — Edited SKILL.md content served fresh on the next apply_config
# ---------------------------------------------------------------------------


class TestEditedSkillFresh:
    def test_bundled_body_edit_picked_up(self, rescan):
        runtime = rescan["runtime"]
        skill = rescan["seed"](
            rescan["dir_a"] / "editable" / "SKILL.md", body="# E\n\nv1 body.\n"
        )
        rescan["run"]()
        assert "v1 body" in runtime.skill_registry.get_content("editable")

        # Rewrite, then bump FORWARD along the synthetic timeline: a mtime
        # older than the one just recorded (or older than "now") would let an
        # mtime-based signature miss the edit entirely.
        skill.write_text(_skill_text("editable", "# E\n\nv2 body.\n"), encoding="utf-8")
        _bump(skill, _T_EDIT_1)

        assert rescan["run"]() == []
        content = runtime.skill_registry.get_content("editable")
        assert "v2 body" in content
        assert "v1 body" not in content

    def test_flat_edit_including_frontmatter_name_picked_up(self, rescan):
        runtime = rescan["runtime"]
        skill = rescan["seed"](
            rescan["dir_b"] / "renamer.md", body="Old content."
        )
        rescan["run"]()
        assert "renamer" in runtime.skill_registry

        skill.write_text(
            _skill_text("renamed", "New content."), encoding="utf-8"
        )
        _bump(skill, _T_EDIT_2)

        assert rescan["run"]() == []
        assert "renamed" in runtime.skill_registry
        assert "renamer" not in runtime.skill_registry
        assert "New content." in runtime.skill_registry.get_content("renamed")


# ---------------------------------------------------------------------------
# AC 4 — Steady state: rebuild path NOT entered, zero log noise
# ---------------------------------------------------------------------------


class TestSteadyStateNoRebuild:
    def test_repeated_unchanged_calls_never_rebuild_and_never_log(
        self, rescan, caplog
    ):
        runtime = rescan["runtime"]
        rescan["seed"](
            rescan["dir_a"] / "stable" / "SKILL.md", body="# S\n\nstable.\n"
        )
        rescan["seed"](rescan["dir_b"] / "stable-flat.md", body="Flat stable.")
        # First call: the fs differs from the startup snapshot. An
        # implementation that folds bootstrap into the first scan does zero
        # rebuilds here; one that seeds lazily does one. Pinning "one" would
        # over-specify, so settle and watch from here on. (The bootstrap
        # seeding itself is pinned by
        # TestNonSkillsRuntimeUnaffected.test_first_message_after_startup_without_changes_is_cold.)
        rescan["run"]()
        settled = rescan["rebuild_calls"]()
        names = runtime.skill_registry.list_names()

        # No per-message log noise, and the "no rebuild" claim is about OUR
        # subsystem: filter to spectre-owned loggers so a chatty unrelated
        # library cannot mask or trip the check.
        with caplog.at_level(logging.INFO):
            for _ in range(3):
                assert rescan["run"]() == []
            assert rescan["rebuild_calls"]() == settled

        spectre_logs = [
            r for r in caplog.records if r.name.split(".")[0] == "sr2_spectre"
        ]
        assert spectre_logs == []
        assert runtime.skill_registry.list_names() == names

    def test_mtime_restored_to_recorded_value_does_not_rebuild(self, rescan):
        """A net-zero change costs zero rebuilds.

        The touch moves the mtime away and back *between* two messages: the
        signature the next scan computes is exactly the one it recorded, so
        the rebuild path must stay cold. (A touch that leaves the mtime
        different rebuilds — that is the accepted mtime-based design;
        freshness never depends on content-diffing.)
        """
        runtime = rescan["runtime"]
        skill = rescan["seed"](
            rescan["dir_a"] / "touched" / "SKILL.md", body="# T\n\ntouched.\n"
        )
        rescan["run"]()

        # What the last scan recorded is the mtime the file has RIGHT NOW,
        # before this test moves it. Capture it here: reading it after the
        # "away" bump would restore the wrong value and make the restore a
        # coin flip.
        recorded_mtime = skill.stat().st_mtime

        _bump(skill, _T_AWAY)
        _bump(skill, _T_OVERSHOOT)
        _bump(skill, recorded_mtime)  # back to what the last scan recorded

        before = rescan["rebuild_calls"]()
        assert rescan["run"]() == []
        assert rescan["rebuild_calls"]() == before
        assert "touched" in runtime.skill_registry


# ---------------------------------------------------------------------------
# AC 5 — Failed rescan: last-good registry in force, exactly one warning
# ---------------------------------------------------------------------------


class TestRescanFailureLastGood:
    """Last-good contract for the RESCAN path.

    A failed rescan is pinned as: ``apply_config`` returns normally, exactly
    one WARNING is emitted by the ``sr2_spectre`` tree, and — the real teeth
    — the skills that were in force before the failure are still in force
    byte-for-byte.

    Which failures actually count as "the rescan failed" follows from what
    the discovery layer already swallows on its own:

    * An **unreadable configured directory** cannot be told apart from an
      empty one by a (path, mtime, size) signature over its entries — the
      entries simply stop being enumerable. So the failure must be provoked
      by making a *configured* skills_dir itself unreadable while some other
      change forces the scan to run.
    * A **bundled SKILL.md with broken frontmatter is absorbed leniently**
      by ``discover_skills_in_dir`` (it names the skill from its directory
      and returns it), so a half-written YAML block is not an observable
      rescan failure at all. The "temporarily invalid SKILL.md" case that a
      build-first last-good contract must survive is the one that raises out
      of discovery: a skill file that cannot be **read**.
    """

    def _snapshot(self, runtime) -> dict[str, tuple]:
        return {
            name: (
                s.name,
                s.description,
                s.version,
                s.content,
            )
            for name in runtime.skill_registry.list_names()
            if (s := runtime.skill_registry.get(name)) is not None
        }

    def test_unreadable_skills_dir_keeps_last_good_with_one_warning(
        self, rescan, caplog
    ):
        """Configured skills_dir chmod 000 → last good, one warning, no raise.

        The failure has to be provoked on a *configured* directory: a mode-0
        subdirectory is invisible both to discovery's ``*/SKILL.md`` glob and
        to any signature over enumerable entries, so it can never trigger a
        rescan — failure or otherwise. Locking ``dir_a`` and then flipping
        ``dir_b`` pins the behavior every signature shape owes the contract:
        once the scan attempts ``dir_a`` at all, an unreadable ``dir_a`` is a
        failed rescan — keep last good, warn once, return normally.
        """
        runtime = rescan["runtime"]
        rescan["seed"](
            rescan["dir_a"] / "keeper" / "SKILL.md", body="# K\n\nkeep me.\n"
        )
        rescan["seed"](rescan["dir_b"] / "other-flat.md", body="Other.")
        rescan["run"]()
        assert "keeper" in runtime.skill_registry  # baseline: freshness works
        before = self._snapshot(runtime)

        # Lock the CONFIGURED directory itself (see the test's docstring).
        rescan["dir_a"].chmod(0)
        try:
            # Probe: skip on environments where mode bits are not enforced
            # (e.g. running as root) — there is no failure to provoke then.
            try:
                os.listdir(rescan["dir_a"])
                pytest.skip("filesystem mode bits not enforced for this process")
            except PermissionError:
                pass

            # dir_a can no longer be scanned at all. Whatever the rescan
            # decides, dir_b changes as well so the scan is genuinely
            # attempted (the failure cannot be dodged as "nothing to do"):
            # the registry must not lose what it already had.
            rescan["seed"](rescan["dir_b"] / "latecomer.md", body="Too late.")

            with caplog.at_level(logging.WARNING):
                result = rescan["run"]()  # must not raise

            assert result == []
            assert self._snapshot(runtime) == before
            assert "keeper" in runtime.skill_registry
            assert "latecomer" not in runtime.skill_registry

            warnings = _warning_records(caplog)
            assert len(warnings) == 1, [r.getMessage() for r in warnings]
        finally:
            rescan["dir_a"].chmod(0o755)

    def test_unreadable_skill_file_keeps_last_good_with_one_warning(
        self, rescan, caplog
    ):
        """A skill file whose read() raises OSError must not wipe the registry."""
        runtime = rescan["runtime"]
        rescan["seed"](
            rescan["dir_a"] / "editable" / "SKILL.md",
            body="# Editable\n\nThe real content.\n",
        )
        rescan["run"]()
        assert "editable" in runtime.skill_registry  # baseline: freshness works
        before = self._snapshot(runtime)

        # Mid-edit worst case: the file is there but cannot be read (editor
        # swapped it out / flipped its mode between the scan and the load).
        # Unlike broken YAML, this raises out of discovery, which is what the
        # "keep the previous registry" contract has to absorb.
        locked = rescan["dir_b"] / "locked-file.md"
        locked.write_text(_skill_text("locked-file", "Whatever."), encoding="utf-8")
        _bump(locked, _T_UNREADABLE)
        locked.chmod(0)
        try:
            # Probe: skip where mode bits are not enforced (e.g. root), since
            # then the read succeeds and there is no failure to provoke.
            try:
                locked.read_text(encoding="utf-8")
                pytest.skip("filesystem mode bits not enforced for this process")
            except PermissionError:
                pass

            with caplog.at_level(logging.WARNING):
                result = rescan["run"]()  # must not raise

            assert result == []
            assert self._snapshot(runtime) == before
            assert "editable" in runtime.skill_registry

            warnings = _warning_records(caplog)
            assert len(warnings) == 1, [r.getMessage() for r in warnings]
        finally:
            locked.chmod(0o644)


# ---------------------------------------------------------------------------
# AC (design) — Registry OBJECT identity: load_skill holds the reference
# ---------------------------------------------------------------------------


class TestRegistryObjectIdentity:
    def test_registry_object_survives_rebuilds_and_serves_fresh_content(
        self, rescan
    ):
        """load_skill holds the skill_registry object; rebuilds refill in place."""
        runtime = rescan["runtime"]
        assert "load_skill" in runtime.registry

        registry_id = id(runtime.skill_registry)
        registry_obj = runtime.skill_registry

        # Force real churn between messages; count the rebuilds that happen,
        # then require at least one (exact counts would over-specify whether
        # add+remove coalesce into a single rebuild per message).
        before = rescan["rebuild_calls"]()
        flat = rescan["seed"](rescan["dir_a"] / "churn.md", body="v1.")
        rescan["run"]()

        flat.unlink()
        rescan["seed"](
            rescan["dir_a"] / "churn2" / "SKILL.md", body="# churn2\n\nv2.\n"
        )
        rescan["run"]()
        assert rescan["rebuild_calls"]() > before

        assert id(runtime.skill_registry) == registry_id
        assert runtime.skill_registry is registry_obj

        # The registered load_skill callable must serve the REBUILT contents:
        # if the rebuild had swapped the registry object, the tool would still
        # answer from the stale one.
        spec = runtime.registry._tools["load_skill"]
        out = asyncio.run(spec.fn(skill_name="churn2"))
        assert "v2." in out
        assert "not found" not in out


# ---------------------------------------------------------------------------
# AC 6 — Builtin behavior preserved + value-change rebuild still works
# ---------------------------------------------------------------------------


class TestBuiltinBehaviorPreserved:
    def test_builtins_survive_a_config_value_change_rebuild(self, rescan):
        """Pure preservation pin (AC 6, holds before AND after the rescan
        work): a config-value-change rebuild re-registers DEFAULT_SKILLS."""
        runtime = rescan["runtime"]
        # Builtin identity comes from DEFAULT_SKILLS, imported here rather
        # than hardcoded, so the pin does not rot if the list is edited.
        from sr2_spectre.skills.builtin import DEFAULT_SKILLS

        for builtin in DEFAULT_SKILLS:
            assert builtin.name in runtime.skill_registry

        # A skills_dirs VALUE change is what routes through the existing
        # config-change rebuild branch (asserted just below).
        changed = _make_config(
            skills_dirs=[str(rescan["dir_a"] / "now-removed-from-config")],
        )
        applied = runtime.apply_config(changed)
        assert "agent.skills" in applied
        for builtin in DEFAULT_SKILLS:
            assert builtin.name in runtime.skill_registry

    def test_rescan_rebuild_keeps_dir_skills_over_builtins(self, rescan):
        """Rescan-path companion pin: after a fs-churn rebuild the disk
        skill must still be the one served, not the builtin resurrected."""
        runtime = rescan["runtime"]
        rescan["seed"](
            rescan["dir_a"] / "solid-review" / "SKILL.md",
            body="# Solid review\n\nDisk version wins.\n",
        )
        # First apply_config: fs differs from the startup snapshot, so the
        # new rescan rebuilds — and the value-change path must keep working
        # regardless. Either way the disk skill ends up over the builtin.
        rescan["run"]()
        content = runtime.skill_registry.get_content("solid-review")
        assert "Disk version wins" in content
        # A second churn + unchanged-config call must not regress it.
        rescan["seed"](rescan["dir_b"] / "more.md", body="More.")
        rescan["run"]()
        assert "Disk version wins" in runtime.skill_registry.get_content(
            "solid-review"
        )

    def test_default_skills_false_suppresses_builtins_after_rescan(
        self, runtime_factory, tmp_path
    ):
        """default_skills:false across the rescan path still withholds builtins."""
        skills_dir = tmp_path / "only-disk"
        skills_dir.mkdir()
        _seed_skill_file(
            skills_dir / "solo" / "SKILL.md",
            body="# Solo\n\nsolo only.\n",
            mtime=_BASE_MTIME + 2_000.0,
        )
        config = _make_config(skills_dirs=[str(skills_dir)], default_skills=False)
        runtime = runtime_factory(config)
        assert "sr2-conventions" not in runtime.skill_registry

        churn = _seed_skill_file(
            skills_dir / "solo2.md", body="Churn.", mtime=_BASE_MTIME + 2_001.0
        )
        # Same object, unchanged config — pure rescan calls.
        assert runtime.apply_config(config) == []
        assert "solo" in runtime.skill_registry
        assert "sr2-conventions" not in runtime.skill_registry

        churn.unlink()
        assert runtime.apply_config(config) == []
        assert "solo2" not in runtime.skill_registry
        assert "sr2-conventions" not in runtime.skill_registry


class TestValueChangeRebuildStillWorks:
    def test_skills_dirs_value_change_still_rebuilds(
        self, runtime_factory, tmp_path
    ):
        """Config-value change to skills_dirs keeps its old rebuild path."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _seed_skill_file(
            dir_a / "from-a" / "SKILL.md", body="from a.", mtime=_BASE_MTIME + 2_100.0
        )
        _seed_skill_file(
            dir_b / "from-b.md", body="from b.", mtime=_BASE_MTIME + 2_101.0
        )
        runtime = runtime_factory(_make_config(skills_dirs=[str(dir_a)]))
        assert "from-a" in runtime.skill_registry

        changed = _make_config(skills_dirs=[str(dir_b)])
        applied = runtime.apply_config(changed)
        assert "agent.skills" in applied
        assert "from-b" in runtime.skill_registry
        assert "from-a" not in runtime.skill_registry

    def test_default_skills_value_change_still_rebuilds(self, rescan):
        runtime = rescan["runtime"]
        rescan["run"]()
        assert "sr2-conventions" in runtime.skill_registry

        changed = _make_config(
            skills_dirs=[str(rescan["dir_a"]), str(rescan["dir_b"])],
            default_skills=False,
        )
        applied = runtime.apply_config(changed)
        assert "agent.skills" in applied
        assert "sr2-conventions" not in runtime.skill_registry


# ---------------------------------------------------------------------------
# AC 7 support — rescan must not disturb other apply_config behavior
# ---------------------------------------------------------------------------


class TestNonSkillsRuntimeUnaffected:
    def test_runtime_without_skills_dirs_unchanged(self, runtime_factory):
        """No skills_dirs → repeated identical apply_config stays a pure no-op."""
        config = _make_config()
        runtime = runtime_factory(config)
        names = runtime.skill_registry.list_names()
        assert "sr2-conventions" in names  # builtins, no dirs

        for _ in range(3):
            assert runtime.apply_config(config) == []
        assert runtime.skill_registry.list_names() == names

    def test_first_message_after_startup_without_changes_is_cold(
        self, runtime_factory, tmp_path
    ):
        """Bootstrap seeds the signature: the FIRST message rebuilds nothing.

        Startup already scanned the dirs, so an unchanged filesystem must
        cost exactly zero rebuilds on message one. This is the load-bearing
        AC 4 test (see the ``rescan`` fixture docstring).
        """
        skills_dir = tmp_path / "boot"
        skills_dir.mkdir()
        _seed_skill_file(
            skills_dir / "booted" / "SKILL.md",
            body="# B\n\nbooted.\n",
            mtime=_BASE_MTIME + 2_200.0,
        )
        config = _make_config(skills_dirs=[str(skills_dir)])
        runtime = runtime_factory(config)
        assert "booted" in runtime.skill_registry

        original = type(runtime)._rebuild_skills
        calls: list = []

        def spy(self, cfg):  # noqa: ANN001 - class patch
            calls.append(cfg)
            original(self, cfg)

        with patch.object(type(runtime), "_rebuild_skills", spy):
            assert runtime.apply_config(config) == []
        assert calls == []
