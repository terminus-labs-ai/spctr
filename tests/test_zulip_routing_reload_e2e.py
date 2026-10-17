"""J1C (E2E) — spc-120: Zulip area routing and live whole-config reload.

Runs the real entry point (``python -m sr2_spectre.cli <config> --interface
zulip``) with only the Zulip server and the OpenAI-compatible model endpoint
substituted on loopback (``zulip_e2e_support``).  CLI, config resolution and
reload, the Zulip interface, Agent/Runtime/Session, the SR2 pipeline and the
``topic_area``/session/input resolvers are real.

Journeys / AC:
- test_area_routing_and_isolation → AC2, AC3
- test_live_reload_and_last_good  → AC6

Area context is proven at the compiled model request: each area's AGENTS.md
and source NOW.md carry a marker, so "own marker exactly once, every other
marker absent" is the resolver-context assertion.  The child's cwd is
``<workdir>/cwd-area`` and a topic with that identity exists, so a
working-directory fallback compiles CWD_MARKER.

How the assertions reject wrong behavior:
- area from cwd: CWD_MARKER appears and the own-area marker is missing;
- area from the channel/folder name or a half-eligible topic: BETA's identity
  differs from its folder, and the ``stale`` topic's NEAR_MISS_MARKER leaks;
- every area compiled: MISC_MARKER (eligible, mapped to no channel) appears;
- shared history across topics: another topic's trigger text appears, while a
  topic's own follow-up must still carry its earlier trigger;
- area pinned to a Session's creation: existing topic r1 keeps ALPHA after
  its channel is re-pointed (to BETA, then to "");
- reload a no-op or frozen after the first message: the re-pointed channel
  keeps ALPHA, the newly mapped channel stays inert, the document edit never
  shows, and the endpoint edits never move requests between model servers;
- user filter pinned to startup or only growing: OTHER_ID stays inert after
  being added, or keeps getting turns after being removed;
- channels merged over the startup set or accumulated: a removed startup
  channel (45) or removed late channel (47) still gets a turn;
- no last-good: the malformed file kills the child, fails the turn, or applies
  its routing edit; no recovery: the repaired file's routing never applies.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zulip_e2e_support import (  # noqa: E402
    BOT_ID,
    CONFIGURED_CHANNEL,
    DIEGO_ID,
    OTHER_ID,
    UNCONFIGURED_CHANNEL,
    Harness,
    ModelServer,
    ZulipCliRun,
)

ALPHA_AREA, ALPHA_MARKER = "alpha-area", "SPC120_ALPHA_MARKER"
# Identity deliberately differs from its folder name ("beta").
BETA_AREA, BETA_MARKER = "sr2-spectre-beta", "SPC120_BETA_MARKER"
CWD_AREA, CWD_MARKER = "cwd-area", "SPC120_CWD_MARKER"
# Ineligible: README has no id; NOW.md is for an identity no README claims.
STALE_AREA, NEAR_MISS_AREA = "sr2-stale", "sr2-spectre-near-miss"
NEAR_MISS_MARKER = "SPC120_NEAR_MISS_MARKER"
MISC_AREA, MISC_MARKER = "misc-area", "SPC120_MISC_MARKER"
MARKERS = (ALPHA_MARKER, BETA_MARKER, CWD_MARKER, NEAR_MISS_MARKER, MISC_MARKER)

ALPHA_CHANNEL = CONFIGURED_CHANNEL   # 42 -> ALPHA_AREA
BETA_CHANNEL = 44                    # -> BETA_AREA
NOAREA_CHANNEL = 45                  # -> "" (explicit no-area)
STALE_CHANNEL = 46                   # -> STALE_AREA (no eligible topic)
LATE_CHANNEL = 47                    # mapped while the CLI runs

EDITED_LINE = "SPC120_EDITED_AREA_LINE"
RELOAD_ANSWER = "SPC120_RELOAD_ANSWER"


def _topics(root: Path) -> None:
    def topic(folder: str, readme_id: str | None, now_for: str, marker: str) -> None:
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        rid = f"id: topic:{readme_id}\n" if readme_id else ""
        (d / "README.md").write_text(f"---\nkind: topic\n{rid}---\n",
                                     encoding="utf-8")
        (d / "NOW.md").write_text(
            f"---\nkind: continuity\nfor: topic:{now_for}\nmode: source\n---\n"
            f"{marker} NOW\n", encoding="utf-8")
        (d / "AGENTS.md").write_text(f"{marker} AGENTS\n", encoding="utf-8")

    topic("alpha", ALPHA_AREA, ALPHA_AREA, ALPHA_MARKER)
    topic("beta", BETA_AREA, BETA_AREA, BETA_MARKER)
    topic(CWD_AREA, CWD_AREA, CWD_AREA, CWD_MARKER)
    topic("stale", None, NEAR_MISS_AREA, NEAR_MISS_MARKER)
    topic("misc", MISC_AREA, MISC_AREA, MISC_MARKER)


def _harness(tmp_path: Path, name: str) -> Harness:
    return Harness(
        workdir=tmp_path / name,
        channels={ALPHA_CHANNEL: ALPHA_AREA, BETA_CHANNEL: BETA_AREA,
                  NOAREA_CHANNEL: "", STALE_CHANNEL: STALE_AREA},
        topic_docs=_topics,
    )


def _fail(run: ZulipCliRun, message: str) -> str:
    rc = run.proc.poll() if run.proc is not None else None
    return (f"{message}\nreturncode={rc} model_requests={len(run.model.requests)} "
            f"sends={len(run.zulip.sends)} log_tail={run.log_text[-1500:]!r} "
            f"child_tail={run.child_output()[-1200:]!r}")


def _text(request: dict) -> str:
    return "\n".join(str(m.get("content") or "")
                     for m in request.get("messages", []) if isinstance(m, dict))


def _turn(run: ZulipCliRun, channel: int, topic: str, text: str,
          model: ModelServer | None = None, sender: int = DIEGO_ID) -> str:
    """Send one allowed message; return its single compiled model request."""
    model = model or run.model
    sends, requests = len(run.zulip.sends), len(model.requests)
    run.zulip.push_message(channel, topic, sender, text)
    assert run.wait_idle(expected_replies=sends + 1), _fail(
        run, f"no reply to {text!r} in {channel}/{topic}")
    assert len(model.requests) == requests + 1, _fail(
        run, f"{text!r} did not reach the expected model endpoint exactly once "
             f"({len(model.requests) - requests} requests)")
    compiled = _text(model.requests[-1])
    assert text in compiled, _fail(run, f"newest request is not {text!r}'s turn")
    assert model.answer in run.zulip.sends[-1]["content"], _fail(
        run, f"reply to {text!r} did not come from the expected endpoint")
    return compiled


def _assert_area(run: ZulipCliRun, compiled: str, own: str | None) -> None:
    """``own`` area marker's AGENTS and NOW exactly once; no other marker."""
    for marker in MARKERS:
        if marker == own:
            for doc in ("AGENTS", "NOW"):
                assert compiled.count(f"{marker} {doc}") == 1, _fail(
                    run, f"{marker} {doc} not compiled exactly once: "
                         f"{compiled[:900]!r}")
        else:
            assert marker not in compiled, _fail(
                run, f"{marker} leaked (expected area marker {own}): "
                     f"{compiled[:900]!r}")


def _assert_no_work(run: ZulipCliRun, channel: int, topic: str) -> None:
    """An ignored trigger: no history probe, model request, reply or Session."""
    run.wait_ignorable_delivered()
    run.assert_quiescent(len(run.model.requests), len(run.zulip.sends))
    assert (channel, topic) not in run.zulip.history_calls, _fail(
        run, f"ignored trigger {channel}/{topic} probed topic history")
    assert run.count_identities("miranda", topic) == 0, _fail(
        run, f"ignored trigger {channel}/{topic} created a Session")


def _route(channel: int, old: str, new: str) -> dict[str, str]:
    return {f"    {channel}: {old!r}\n": f"    {channel}: {new!r}\n"}


# ---------------------------------------------------------------------------
# AC2 + AC3
# ---------------------------------------------------------------------------


def test_area_routing_and_isolation(tmp_path):
    """Two mapped channels get their own areas through the real resolvers; the
    explicit no-area channel and an area without an eligible topic get none
    (never the cwd area); topics keep separate histories; bot, disallowed-user
    and unmapped-channel events do no model work."""
    harness = _harness(tmp_path, "j1c-route")
    run, cwd, env = harness.build()   # stops its own endpoints if it fails
    try:
        harness.launch(run, cwd, env)
        assert run.wait_started(), _fail(run, "CLI never started")
        assert run.zulip.history_calls == [], _fail(run, "startup replayed history")

        texts = {
            "a1": _turn(run, ALPHA_CHANNEL, "route-a", "ALPHA TRIGGER"),
            "b1": _turn(run, BETA_CHANNEL, "route-b", "BETA TRIGGER"),
            "n1": _turn(run, NOAREA_CHANNEL, "plain", "NOAREA TRIGGER"),
            "h1": _turn(run, STALE_CHANNEL, "halfmap", "HALFMAP TRIGGER"),
            "a2": _turn(run, ALPHA_CHANNEL, "second-a", "ALPHA SECOND"),
            "a3": _turn(run, ALPHA_CHANNEL, "route-a", "ALPHA FOLLOWUP"),
        }
        areas = {"a1": ALPHA_MARKER, "b1": BETA_MARKER, "n1": None, "h1": None,
                 "a2": ALPHA_MARKER, "a3": ALPHA_MARKER}
        # Each turn's own trigger(s); route-a's follow-up keeps its history.
        own = {"a1": {"ALPHA TRIGGER"}, "b1": {"BETA TRIGGER"},
               "n1": {"NOAREA TRIGGER"}, "h1": {"HALFMAP TRIGGER"},
               "a2": {"ALPHA SECOND"}, "a3": {"ALPHA TRIGGER", "ALPHA FOLLOWUP"}}
        triggers = set().union(*own.values())
        for name, compiled in texts.items():
            _assert_area(run, compiled, areas[name])
            for expected in own[name]:
                assert expected in compiled, _fail(
                    run, f"turn {name} lacks its own history {expected!r}")
            for foreign in triggers - own[name]:
                assert foreign not in compiled, _fail(
                    run, f"{foreign!r} leaked into turn {name}'s request")
        for topic in ("route-a", "route-b", "plain", "halfmap", "second-a"):
            assert run.count_identities("miranda", topic) == 1, _fail(
                run, f"topic {topic!r} does not have exactly one Session")

        # Bot (also allowlisted) and disallowed user on a mapped-area channel,
        # and an allowed user on an unmapped channel: no work at all.
        run.zulip.push_message(ALPHA_CHANNEL, "botfresh", BOT_ID, "bot chatter")
        _assert_no_work(run, ALPHA_CHANNEL, "botfresh")
        run.zulip.push_message(ALPHA_CHANNEL, "unallowed", OTHER_ID, "hi")
        _assert_no_work(run, ALPHA_CHANNEL, "unallowed")
        run.zulip.push_message(UNCONFIGURED_CHANNEL, "offch", DIEGO_ID, "hi")
        _assert_no_work(run, UNCONFIGURED_CHANNEL, "offch")
        assert len(run.model.requests) == 6, _fail(run, "unexpected model work")
    finally:
        harness.close(run)


# ---------------------------------------------------------------------------
# AC6 — next-message reload through the real CLI, last-good on malformed
# ---------------------------------------------------------------------------


def test_live_reload_and_last_good(tmp_path):
    """Routing, area-document and model-endpoint edits made while the CLI
    runs apply on the next message, every time (not only the first); document
    and endpoint edits reach existing topics; a malformed file keeps the
    last-good config and the process alive; the repaired file then applies."""
    harness = _harness(tmp_path, "j1c-reload")
    run, cwd, env = harness.build()   # stops its own endpoints if it fails
    model2 = ModelServer(answer=RELOAD_ANSWER)
    try:
        model2.start()
        harness.launch(run, cwd, env)
        beta_doc = run.workdir / "topics" / "beta" / "AGENTS.md"
        assert run.wait_started(), _fail(run, "CLI never started")
        _assert_area(run, _turn(run, ALPHA_CHANNEL, "r1", "BASE"), ALPHA_MARKER)
        _assert_area(run, _turn(run, ALPHA_CHANNEL, "r1", "BASE TWO"),
                     ALPHA_MARKER)

        # (1) Re-point channel 42 to BETA: the next new topic follows it.
        harness.edit_config(_route(ALPHA_CHANNEL, ALPHA_AREA, BETA_AREA))
        _assert_area(run, _turn(run, ALPHA_CHANNEL, "r2", "ROUTED NEW"),
                     BETA_MARKER)

        # (1b) Spec ruling: every turn uses the channel's current area, so the
        # existing topic r1 (Session created under ALPHA) is re-routed too,
        # keeping its Session and history; re-pointing to "" gives it no area
        # and no cwd fallback.  Then restore BETA for the steps below.
        for area, marker, text in ((BETA_AREA, BETA_MARKER, "OLD TOPIC BETA"),
                                   ("", None, "OLD TOPIC NOAREA")):
            if area == "":
                harness.edit_config(_route(ALPHA_CHANNEL, BETA_AREA, ""))
            rerouted = _turn(run, ALPHA_CHANNEL, "r1", text)
            _assert_area(run, rerouted, marker)
            # Both earlier ALPHA-era turns ("BASE", "BASE TWO") and the BETA one.
            assert rerouted.count("BASE") >= 2 and "BASE TWO" in rerouted and (
                "OLD TOPIC BETA" in rerouted), _fail(
                run, f"re-routed topic r1 lost its history: {rerouted[:900]!r}")
            assert run.count_identities("miranda", "r1") == 1, _fail(
                run, "re-routed topic r1 does not keep exactly one Session")
        harness.edit_config(_route(ALPHA_CHANNEL, "", BETA_AREA))

        # (2) A channel mapped while running: inert before, admitted after.
        run.zulip.push_message(LATE_CHANNEL, "late", DIEGO_ID, "LATE INERT")
        _assert_no_work(run, LATE_CHANNEL, "late")
        harness.edit_config({"  channels:\n":
                             f"  channels:\n    {LATE_CHANNEL}: {ALPHA_AREA!r}\n"})
        _assert_area(run, _turn(run, LATE_CHANNEL, "late2", "LATE MAPPED"),
                     ALPHA_MARKER)

        # (3) Live area-document edit reaches the existing topic, and its
        # removal does too (no resolver cache).
        original_doc = beta_doc.read_text(encoding="utf-8")
        beta_doc.write_text(original_doc + EDITED_LINE + "\n", encoding="utf-8")
        edited = _turn(run, ALPHA_CHANNEL, "r2", "DOC EDIT")
        assert edited.count(EDITED_LINE) == 1, _fail(
            run, "edited area document not compiled exactly once")
        _assert_area(run, edited, BETA_MARKER)
        beta_doc.write_text(original_doc, encoding="utf-8")
        assert EDITED_LINE not in _turn(run, ALPHA_CHANNEL, "r2", "DOC OFF"), (
            _fail(run, "removed document line is still compiled"))

        # (3b) User allowlist reloads too: OTHER_ID is inert on a mapped
        # channel, gets a turn once added on disk, and is inert again once
        # removed.  A filter pinned to the startup block fails the middle
        # step; one that only ever grows fails the last.
        allow = f"  allowed_user_ids: [{DIEGO_ID}, {BOT_ID}]\n"
        allow_other = f"  allowed_user_ids: [{DIEGO_ID}, {BOT_ID}, {OTHER_ID}]\n"
        run.zulip.push_message(ALPHA_CHANNEL, "u1", OTHER_ID, "USER INERT")
        _assert_no_work(run, ALPHA_CHANNEL, "u1")
        harness.edit_config({allow: allow_other})
        _assert_area(run, _turn(run, ALPHA_CHANNEL, "u2", "USER ADDED",
                                sender=OTHER_ID), BETA_MARKER)
        harness.edit_config({allow_other: allow})
        run.zulip.push_message(ALPHA_CHANNEL, "u3", OTHER_ID, "USER REMOVED")
        _assert_no_work(run, ALPHA_CHANNEL, "u3")

        # (3c) Channel removal reloads: drop a startup mapping (no-area 45)
        # and the late one (47).  Merging loaded channels over the startup
        # set keeps 45; accumulating every channel ever loaded keeps 47.
        harness.edit_config({f"    {NOAREA_CHANNEL}: ''\n": "",
                             f"    {LATE_CHANNEL}: {ALPHA_AREA!r}\n": ""})
        run.zulip.push_message(NOAREA_CHANNEL, "gone45", DIEGO_ID, "GONE 45")
        _assert_no_work(run, NOAREA_CHANNEL, "gone45")
        run.zulip.push_message(LATE_CHANNEL, "gone47", DIEGO_ID, "GONE 47")
        _assert_no_work(run, LATE_CHANNEL, "gone47")

        # (4) Whole-config edit: the model endpoint moves to model2 for an
        # existing topic's next turn.
        endpoint_two = {f"    base_url: {run.model.base_url}/v1\n":
                        f"    base_url: {model2.base_url}/v1\n",
                        "    api_key: synthetic-api-key\n":
                        "    api_key: synthetic-api-key-two\n"}
        before = len(run.model.requests)
        harness.edit_config(endpoint_two)
        _assert_area(run, _turn(run, ALPHA_CHANNEL, "r2", "ENDPOINT", model2),
                     BETA_MARKER)

        # (5) Malformed YAML that also carries a routing edit: last-good (BETA,
        # model2) stays in force and the process survives.
        harness.edit_config({"active_model: default\n": "active_model: [unclosed\n",
                             **_route(ALPHA_CHANNEL, BETA_AREA, ALPHA_AREA)})
        _assert_area(run, _turn(run, ALPHA_CHANNEL, "r3", "LASTGOOD", model2),
                     BETA_MARKER)
        assert run.proc.poll() is None, _fail(run, "CLI died on malformed config")
        assert "reload failed" in run.log_text, _fail(
            run, "malformed config was not reported as a failed reload")

        # (6) Repair only the syntax: the next message applies the whole file,
        # including the routing edit that arrived with the broken write.
        harness.edit_config({"active_model: [unclosed\n": "active_model: default\n"})
        _assert_area(run, _turn(run, ALPHA_CHANNEL, "r4", "RECOVER", model2),
                     ALPHA_MARKER)

        # (7) Endpoint back to the original server; model2 gets nothing more.
        harness.edit_config({v: k for k, v in endpoint_two.items()})
        _assert_area(run, _turn(run, ALPHA_CHANNEL, "r4", "FINAL"), ALPHA_MARKER)
        assert len(model2.requests) == 3, _fail(run, "model2 kept receiving turns")
        assert len(run.model.requests) == before + 1, _fail(
            run, "original endpoint received turns while model2 was in force")
        assert run.proc.poll() is None, _fail(run, "CLI died during reload")
    finally:
        try:
            harness.close(run)   # kill+reap the child, stop its endpoints
        finally:
            model2.stop()        # safe even if it never came up
