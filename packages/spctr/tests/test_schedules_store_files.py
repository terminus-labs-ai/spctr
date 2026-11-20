"""Directory-backed ScheduleStore: round-trip fidelity and concurrency (spc-994).

AC37 (async facades), AC38 (round-trip fidelity, key order), AC39
(concurrent writers, kill between temp write and replace, temp sweep).
File tests use tmp_path directories; concurrency uses two real
processes sharing one directory.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from spctr.schedules import Schedule, ScheduleStore

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)
SENTINEL = "9999-12-31T23:59:59+00:00"


def _make_schedule(**kwargs) -> Schedule:
    defaults = dict(
        id="s1",
        agent="miranda",
        platform="discord",
        requester_id="42",
        requester_mention="<@42>",
        dest_channel_id="100",
        dest_channel_name="general",
        label="test",
        cron="0 7 * * *",
        payload_type="command",
        payload=json.dumps({"command": "echo hi", "cwd": None}),
        on_miss="run_late",
        next_run_at=NOW + timedelta(hours=1),
        last_run_at=None,
        created_at=NOW,
        status="active",
        last_status=None,
        last_error=None,
    )
    defaults.update(kwargs)
    return Schedule(**defaults)


def _write_schedule_file(store_dir: Path, filename: str, user_keys: str, body: str) -> Path:
    path = store_dir / filename
    path.write_text(f"---\n{user_keys}\n---\n\n{body}", encoding="utf-8")
    return path


# Runtime keys live in the same frontmatter block as user keys but are
# engine-owned: advance_after_run rewrites them, so they are excluded when
# checking that *user* keys survive unchanged across a rewrite.
_RUNTIME_KEY_PREFIXES = (
    "next_run_at:", "last_run_at:", "last_status:", "last_error:",
)


def _body_after_fence(data: bytes) -> bytes:
    """The body bytes after the closing frontmatter fence, verbatim.

    Mirrors ``store._split_document``: the file opens with ``---\\n`` and
    the body starts after the closing ``---`` line, minus exactly one
    separator newline. Lets a test prove a runtime rewrite touched no body
    byte at all — no strip, no newline rewrite, no unicode change.
    """
    assert data.startswith(b"---\n"), "file must open with a frontmatter fence"
    end = data.find(b"\n---\n", 4)
    assert end != -1, "file has no closing frontmatter fence"
    rest = data[end + len(b"\n---\n") :]
    return rest[1:] if rest.startswith(b"\n") else rest


def _user_key_lines(data: bytes) -> set[str]:
    """The frontmatter user-key lines, excluding fences and runtime keys.

    Reads only the head block (before the closing fence) so a non-empty
    body never leaks into the set; drops engine-owned runtime keys whose
    values a rewrite is expected to change.
    """
    head = data.split(b"\n---\n", 1)[0]
    return {
        line
        for line in head.decode("utf-8").splitlines()
        if line
        and not line.startswith("---")
        and not line.startswith(_RUNTIME_KEY_PREFIXES)
    }


USER_BASE = """platform: discord
requester_id: '42'
requester_mention: <@42>
channel: '100'
label: test
cron: 0 7 * * *
type: command
command: echo hi
on_miss: run_late
"""


# ---------------------------------------------------------------------------
# AC38: round-trip fidelity
# ---------------------------------------------------------------------------

class TestAC38RoundTrip:
    def test_created_file_parses_back_equal(self, tmp_path):
        """AC38-r1: a created file parses back to an equal Schedule.

        FR37 semantics: the channel *name* is never a file key (the seam
        resolves names at load, spc-996), so the parsed name is empty.
        """
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        row = _make_schedule(
            label="wördly ✓ 日本語",
            dest_channel_name="café ☕",
            next_run_at=datetime(2026, 10, 21, 7, 0, tzinfo=UTC),
            payload=json.dumps({"command": "echo héllo — ünïcode ✓", "cwd": "/tmp"}),
        )
        store.create(row)
        parsed = store.get("s1")
        assert parsed is not None
        assert (
            parsed.id, parsed.agent, parsed.platform, parsed.requester_id,
            parsed.requester_mention, parsed.dest_channel_id,
            parsed.dest_channel_name, parsed.label, parsed.cron,
            parsed.payload_type, parsed.payload, parsed.on_miss,
            parsed.next_run_at, parsed.last_run_at, parsed.created_at,
            parsed.status, parsed.last_status, parsed.last_error,
        ) == (
            "s1", row.agent, row.platform, row.requester_id,
            row.requester_mention, row.dest_channel_id,
            "", row.label, row.cron,
            row.payload_type, row.payload, row.on_miss,
            row.next_run_at, row.last_run_at, row.created_at,
            row.status, row.last_status, row.last_error,
        )
        # The instruction/label unicode survives on disk, not just in memory.
        text = (tmp_path / "schedules" / "s1.md").read_text(encoding="utf-8")
        assert "wördly ✓ 日本語" in text
        assert "héllo — ünïcode ✓" in text
        # The resolved channel ID is the file's ``channel`` key (FR37/FR38);
        # a channel name is never persisted.
        assert "channel:" in text
        assert "café" not in text

    def test_created_file_with_unknown_extra_key_parses_back_equal(self, tmp_path):
        """AC38-r1: parse-back equality holds when the frontmatter carries
        an unknown extra key (spc-994 Agent G test-strength finding).

        The extra key is injected after ``store.create`` — the same
        mechanics the runtime-rewrite test uses for ``extra_note`` — so the
        equality check runs against a file the store itself wrote plus one
        user line, not a hand-authored fixture.
        """
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        row = _make_schedule(
            label="wördly ✓ 日本語 🎯",
            next_run_at=datetime(2026, 10, 21, 7, 0, tzinfo=UTC),
            payload=json.dumps({"command": "echo héllo — ünïcode ✓", "cwd": "/tmp"}),
        )
        store.create(row)
        path = tmp_path / "schedules" / "s1.md"
        original = path.read_text(encoding="utf-8")
        # Inject one unknown key line into the frontmatter block, right
        # after the opening fence.
        path.write_text(
            original.replace("---\n", "---\nzz_extra: keep-me 🎯\n", 1),
            encoding="utf-8",
        )
        assert "zz_extra: keep-me 🎯" in path.read_text(encoding="utf-8")

        parsed = store.get("s1")
        assert parsed is not None
        assert (
            parsed.id, parsed.agent, parsed.platform, parsed.requester_id,
            parsed.requester_mention, parsed.dest_channel_id,
            parsed.dest_channel_name, parsed.label, parsed.cron,
            parsed.payload_type, parsed.payload, parsed.on_miss,
            parsed.next_run_at, parsed.last_run_at, parsed.created_at,
            parsed.status, parsed.last_status, parsed.last_error,
        ) == (
            "s1", row.agent, row.platform, row.requester_id,
            row.requester_mention, row.dest_channel_id,
            "", row.label, row.cron,
            row.payload_type, row.payload, row.on_miss,
            row.next_run_at, row.last_run_at, row.created_at,
            row.status, row.last_status, row.last_error,
        )
        # The unknown key is still on disk and its value byte-intact (the
        # emoji is pinned by its UTF-8 bytes, not just the ASCII prefix).
        after = path.read_text(encoding="utf-8")
        assert "zz_extra: keep-me" in after
        assert "zz_extra: keep-me \U0001F3AF" in after  # 🎯 on disk
        # The unicode/emoji label and command survived on disk too.
        assert "wördly ✓ 日本語 🎯" in after
        assert "héllo — ünïcode ✓" in after

    def test_agent_body_survives_byte_for_byte(self, tmp_path):
        """AC38-r1: unicode/emoji agent bodies survive create + parse."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        body = "Summarize the beads 🚂🎉\n\nSecond para — em-dash 日本語 ✓"
        row = _make_schedule(
            payload_type="agent",
            payload=json.dumps({"agent": "miranda", "instruction": body}, ensure_ascii=False),
        )
        store.create(row)
        parsed = store.get("s1")
        assert parsed is not None
        payload = json.loads(parsed.payload)
        assert payload["instruction"] == body
        payload_agent = json.loads(row.payload)
        assert payload_agent["agent"] == "miranda"

    def test_runtime_rewrite_preserves_user_keys_and_body(self, tmp_path):
        """AC38-r2: advance_after_run preserves user keys (incl. unknown
        extras) AND a non-empty body byte-for-byte.

        The body is hand-authored with leading/trailing whitespace, a tab
        indent, runs of blank lines, unicode and an emoji, then survives
        two runtime rewrites (claim, then advance_after_run). Every byte
        after the closing fence must be untouched: this is what kills a
        mutation such as ``extra["body"].strip()`` at store.py:494, which
        the earlier command-file version could not catch (command files
        carry an empty body, so ``echo hi`` matched the frontmatter key,
        never the body). CRLF bodies are spc-1082's scope; the engine
        still normalizes them, so the fixture stays LF-only here.
        """
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        body = "  \nLead \ttabbed line\t\n\n\nAfter blank run ✓ 日本語 🎯\ntrailing   \n   "
        user_keys = (
            "platform: discord\n"
            "label: rewrite-fidelity\n"
            "cron: 0 7 * * *\n"
            "type: agent\n"
            "on_miss: run_late\n"
            "extra_note: keep-me 🎯\n"
            f"next_run_at: '{(NOW - timedelta(seconds=1)).isoformat()}'\n"
        )
        path = _write_schedule_file(tmp_path / "schedules", "s1.md", user_keys, body)
        before = path.read_bytes()
        # The fixture really carries the body verbatim after the fence.
        assert _body_after_fence(before) == body.encode("utf-8")

        assert store.claim("s1") is True
        store.advance_after_run("s1", NOW + timedelta(days=1), NOW, "ok", None)
        after = path.read_bytes()

        # Byte-for-byte body fidelity across both runtime rewrites: the
        # leading spaces, the tab, the blank-line run, the unicode and the
        # trailing whitespace are all preserved exactly.
        assert _body_after_fence(after) == body.encode("utf-8")
        assert _body_after_fence(after) == _body_after_fence(before)

        # The unknown extra user key survives the runtime rewrite too.
        assert b"extra_note: keep-me" in after
        assert b"\xf0\x9f\x8e\xaf" in after  # 🎯 survives
        # Runtime keys moved exactly as requested; user keys are untouched.
        parsed = store.get("s1")
        assert parsed.next_run_at == NOW + timedelta(days=1)
        assert parsed.last_run_at == NOW
        assert parsed.last_status == "ok"
        assert parsed.last_error is None
        # Every user key (known + unknown extra) is preserved byte-for-byte
        # as a line; the runtime keys are the only lines that changed.
        assert _user_key_lines(before) == _user_key_lines(after)

    def test_created_at_survives_claim_and_advance(self, tmp_path):
        """AC38: created_at is persisted and stable across runtime rewrites."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        row = _make_schedule(
            next_run_at=NOW - timedelta(seconds=1),
            created_at=NOW - timedelta(days=7),
        )
        sid = store.create(row)
        store.claim(sid)
        store.advance_after_run(sid, NOW + timedelta(days=1), NOW, "ok", None)
        parsed = store.get(sid)
        assert parsed.created_at == NOW - timedelta(days=7)
        assert parsed.created_at != parsed.next_run_at
        # On disk too: the created_at key survived both rewrites.
        text = (tmp_path / "schedules" / f"{sid}.md").read_text(encoding="utf-8")
        assert "created_at" in text

    def test_render_user_keys_before_runtime_keys(self, tmp_path):
        """AC38-r2: user keys render before runtime keys in the file."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule())
        text = (tmp_path / "schedules" / "s1.md").read_text(encoding="utf-8")
        last_user = max(text.index(k + ":") for k in ("on_miss", "command", "cron", "type"))
        first_runtime = min(text.index(k + ":") for k in ("next_run_at", "last_run_at", "last_status", "last_error"))
        assert last_user < first_runtime
        # Runtime keys in canonical order.
        runtime = [text.index(k + ":") for k in ("next_run_at", "last_run_at", "last_status", "last_error")]
        assert runtime == sorted(runtime)


# ---------------------------------------------------------------------------
# AC38 (spc-1082): line-ending fidelity.  A hand-authored file saved with
# CRLF endings keeps every byte of its body across runtime rewrites; only
# the engine-owned runtime keys change.
# ---------------------------------------------------------------------------

_FENCE_LINE_RE = re.compile(rb"^---[ \t]*\r?\n", re.MULTILINE)

#: Runtime keys a rewrite may legitimately change.
_RUNTIME_PREFIXES = (b"next_run_at:", b"last_run_at:", b"last_status:", b"last_error:")


def _body_of(data: bytes) -> bytes:
    """The bytes after the second fence line, minus one separator line.

    Implemented independently of the store's splitter (a regex over raw
    bytes) so a rewrite that normalizes line endings cannot hide behind
    the code under test.
    """
    fences = list(_FENCE_LINE_RE.finditer(data))
    assert len(fences) == 2, data[:400]
    rest = data[fences[-1].end() :]
    if rest.startswith(b"\r\n"):
        return rest[2:]
    if rest.startswith(b"\n"):
        return rest[1:]
    return rest


def _frontmatter_lines(data: bytes, sep: bytes) -> list[bytes]:
    """Non-empty user frontmatter lines: outside the fences, minus runtime keys."""
    fences = list(_FENCE_LINE_RE.finditer(data))
    assert len(fences) == 2, data[:400]
    head = data[fences[0].end() : fences[1].start()]
    lines = [ln for ln in head.split(sep) if ln]
    return [ln for ln in lines if not ln.startswith(_RUNTIME_PREFIXES)]


def _hand_authored_crlf_file(path: Path, body: str, *, head_eol: str = "\r\n") -> Path:
    """Write a whole schedule file the way a Windows editor would."""
    head = head_eol.join(
        [
            "---",
            "platform: discord",
            "label: crlf-note",
            "cron: 0 7 * * *",
            "type: agent",
            "on_miss: run_late",
            "channel: '100'",
            "extra_note: keep-me 🎯",
            "next_run_at: '2026-10-20T13:59:59+00:00'",
            "---",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((head + body).encode("utf-8"))
    return path


class TestAC38LineEndings:
    def test_crlf_file_is_valid_and_its_body_survives_rewrites(self, tmp_path):
        """AC38-r3: claim + advance_after_run keep a CRLF body byte-for-byte."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        body = "Line one\r\nLine two\r\n\r\nTrailing spaces   \r\n日本語 ✓\r\n"
        path = _hand_authored_crlf_file(tmp_path / "schedules" / "crlf-note.md", body)
        before = path.read_bytes()

        rows, invalid = store.scan()
        assert invalid == []
        assert [row.id for row in rows] == ["crlf-note"]
        assert json.loads(rows[0].payload)["instruction"] == body

        assert store.claim("crlf-note") is True
        store.advance_after_run("crlf-note", NOW + timedelta(days=1), NOW, "ok", None)

        after = path.read_bytes()
        assert _body_of(after) == body.encode("utf-8")
        # Nothing else moved either: the unknown extra key and every other
        # user line are unchanged, and the file kept CRLF throughout (no
        # bare LF anywhere).
        assert _frontmatter_lines(before, b"\r\n") == _frontmatter_lines(after, b"\r\n")
        assert b"extra_note: keep-me \xf0\x9f\x8e\xaf\r\n" in after
        assert b"\n" not in after.replace(b"\r\n", b"")
        parsed = store.get("crlf-note")
        assert parsed.next_run_at == NOW + timedelta(days=1)
        assert parsed.last_status == "ok"

    def test_repeated_rewrites_do_not_drift(self, tmp_path):
        """AC38-r3: a second rewrite with the same values changes no bytes.

        The first rewrite used to convert CRLF to LF; every rewrite after
        that changed the file again as long as line endings were being
        normalized somewhere.
        """
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        path = _hand_authored_crlf_file(
            tmp_path / "schedules" / "crlf-note.md", "One\r\nTwo\r\n"
        )
        nxt, ran = NOW + timedelta(days=1), NOW
        assert store.claim("crlf-note") is True
        store.advance_after_run("crlf-note", nxt, ran, "ok", None)
        first = path.read_bytes()
        assert store.claim("crlf-note") is True
        store.advance_after_run("crlf-note", nxt, ran, "ok", None)
        assert path.read_bytes() == first

    def test_lf_file_is_untouched_in_shape(self, tmp_path):
        """AC38-r3: an LF file still round-trips with LF only (no regression)."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        body = "Line one\nLine two\n"
        path = _hand_authored_crlf_file(
            tmp_path / "schedules" / "lf-note.md", body, head_eol="\n"
        )
        assert store.claim("lf-note") is True
        store.advance_after_run("lf-note", NOW + timedelta(days=1), NOW, "ok", None)
        after = path.read_bytes()
        assert _body_of(after) == body.encode("utf-8")
        assert b"\r" not in after

    def test_mixed_line_endings_keep_the_body_bytes(self, tmp_path):
        """AC38-r3: LF frontmatter with a CRLF body stays byte-exact."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        body = "Line one\r\nLine two\r\n"
        path = _hand_authored_crlf_file(
            tmp_path / "schedules" / "mixed-note.md", body, head_eol="\n"
        )
        rows, invalid = store.scan()
        assert invalid == []
        assert json.loads(rows[0].payload)["instruction"] == body
        assert store.claim("mixed-note") is True
        store.advance_after_run("mixed-note", NOW + timedelta(days=1), NOW, "ok", None)
        after = path.read_bytes()
        assert _body_of(after) == body.encode("utf-8")
        # The structure keeps the style its own fence lines used (LF):
        # a stray CRLF in the body rewrites nothing outside the body.
        fences = list(_FENCE_LINE_RE.finditer(after))
        assert b"\r" not in after[: fences[1].start()]
        # Re-reading the rewritten file still yields the same instruction.
        assert json.loads(store.get("mixed-note").payload)["instruction"] == body

    def test_created_agent_file_with_crlf_instruction_is_all_crlf(self, tmp_path):
        """spc-1100: a created file carries one line style, not case-row r4's.

        A tool-created schedule whose agent instruction contains CRLF used
        to land as LF frontmatter over a CRLF body — the mixed shape. The
        create path now takes the fence/frontmatter style from the body,
        and a runtime rewrite keeps the all-CRLF file byte-stable.
        """
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        body = "Do the thing\r\nsecond line\r\n"
        row = _make_schedule(
            payload_type="agent",
            payload=json.dumps({"instruction": body}),
        )
        sid = store.create(row)
        path = tmp_path / "schedules" / f"{sid}.md"
        data = path.read_bytes()
        assert _body_of(data) == body.encode("utf-8")
        assert b"\n" not in data.replace(b"\r\n", b"")
        # Round trip: the instruction reads back verbatim.
        assert json.loads(store.get(sid).payload)["instruction"] == body
        # A runtime rewrite stays all-CRLF and keeps the body exact.
        assert store.claim(sid) is True
        store.advance_after_run(sid, NOW + timedelta(days=1), NOW, "ok", None)
        after = path.read_bytes()
        assert _body_of(after) == body.encode("utf-8")
        assert b"\n" not in after.replace(b"\r\n", b"")
        assert json.loads(store.get(sid).payload)["instruction"] == body

    def test_created_agent_file_with_lf_instruction_stays_lf(self, tmp_path):
        """spc-1100: an LF instruction still creates an LF-only file."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        body = "Do the thing\nsecond line\n"
        row = _make_schedule(
            payload_type="agent",
            payload=json.dumps({"instruction": body}),
        )
        sid = store.create(row)
        data = (tmp_path / "schedules" / f"{sid}.md").read_bytes()
        assert b"\r" not in data
        assert _body_of(data) == body.encode("utf-8")
        assert json.loads(store.get(sid).payload)["instruction"] == body

    @pytest.mark.parametrize("head_eol", ["\n", "\r\n"])
    def test_lone_fence_line_in_body_is_not_a_fence(self, tmp_path, head_eol):
        """spc-1100: a body line equal to ``---`` stays body, forever.

        The closing fence is the first line-start ``---`` after the head,
        so later ``---`` lines cannot end the frontmatter: G acceptance of
        spc-1082 claimed such a body line truncates at the first runtime
        rewrite; on both the accepted range (788f51b) and main it does
        not, and these pins hold the byte-for-byte promise of AC38 for
        every head line ending crossed with a lone-fence body.
        """
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        body = f"note{head_eol}---{head_eol}tail{head_eol}"
        path = _hand_authored_crlf_file(
            tmp_path / "schedules" / "fence-note.md", body, head_eol=head_eol
        )
        rows, invalid = store.scan()
        assert invalid == []
        assert json.loads(rows[0].payload)["instruction"] == body
        assert store.claim("fence-note") is True
        store.advance_after_run("fence-note", NOW + timedelta(days=1), NOW, "ok", None)
        after = path.read_bytes()
        # Not _body_of(): that helper asserts exactly two fence lines, and
        # a lone --- body line is a third fence-looking line by design.
        # The fixture layout is exact (helper-written prefix + body), so
        # tail equality is the byte-for-byte check here.
        assert after.endswith(body.encode("utf-8"))
        assert json.loads(store.get("fence-note").payload)["instruction"] == body


# ---------------------------------------------------------------------------
# FR37 key semantics (spc-1077): agent is the payload target, channel is
# the destination, the owning agent is the store's (FR36) — never
# frontmatter; payload_agent/dest_channel_* are not file keys.
# ---------------------------------------------------------------------------

# The spec's own example file ("Schedule file (increment 3)"), including
# the comment line the renderer would never write.
# ``platform: discord`` is stamped per FR38 (this agent has exactly one
# schedule-delivery platform enabled; the derivation itself is spc-996).
SPEC_EXAMPLE = """platform: discord
label: daily-digest
cron: 0 7 * * *
type: command
command: python3 /home/shepard/git/daily-digest/digest.py
channel: "1504678682290159728"
on_miss: run_late

# runtime — engine-owned, rewritten atomically after each run
next_run_at: "2026-10-27T07:00:00-07:00"
last_run_at: "2026-10-26T07:00:04-07:00"
last_status: ok
"""


class TestFR37KeySemantics:
    def test_created_command_file_keys_are_fr37(self, tmp_path):
        """A created command file carries exactly the FR37 user keys (plus
        requester/created_at extras); no owning agent, no dest_channel_*,
        no payload_agent."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule())
        text = (tmp_path / "schedules" / "s1.md").read_text(encoding="utf-8")
        keys = [
            line.split(":", 1)[0]
            for line in text.splitlines()
            if line and not line.startswith("---")
            and not line.startswith(("#", "next_run_at:", "last_run_at:", "last_status:", "last_error:"))
        ]
        assert keys == [
            "platform", "label", "cron", "type", "command", "on_miss",
            "channel", "requester_id", "requester_mention", "created_at",
        ]
        assert "dest_channel_id" not in text
        assert "dest_channel_name" not in text
        assert "payload_agent" not in text
        assert "\nagent:" not in text

    def test_channel_key_is_the_destination_id(self, tmp_path):
        """FR37/FR38: the resolved channel ID round-trips through the
        ``channel`` key; the in-memory name is empty on reload."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule(dest_channel_id="1504678682290159728"))
        parsed = store.get("s1")
        assert parsed.dest_channel_id == "1504678682290159728"
        assert parsed.dest_channel_name == ""

    def test_agent_payload_default_target_is_the_store_agent(self, tmp_path):
        """An agent payload targeting the owning agent writes no ``agent``
        key (FR37: optional, default the owning agent) and reloads with
        the store's agent as the target."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule(
            payload_type="agent",
            payload=json.dumps({"agent": "miranda", "instruction": "do it"}),
        ))
        text = (tmp_path / "schedules" / "s1.md").read_text(encoding="utf-8")
        assert "agent:" not in text
        assert "payload_agent" not in text
        parsed = store.get("s1")
        assert json.loads(parsed.payload)["agent"] == "miranda"

    def test_agent_payload_explicit_target_round_trips(self, tmp_path):
        """An agent payload targeting another agent writes ``agent:
        <target>`` (FR37) and reloads with that target."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule(
            payload_type="agent",
            payload=json.dumps({"agent": "ezio", "instruction": "for ezio"}),
        ))
        text = (tmp_path / "schedules" / "s1.md").read_text(encoding="utf-8")
        assert "agent: ezio" in text
        parsed = store.get("s1")
        assert json.loads(parsed.payload)["agent"] == "ezio"
        assert parsed.agent == "miranda"  # owner is the store's, FR36

    def test_spec_example_file_parses_with_channel_and_owner(self, tmp_path):
        """The spec's own daily-digest.md example (FR37 section) parses
        valid: the channel ID is the destination and the owning agent is
        the store's — list_for_agent sees it (the spc-1077 probe)."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        _write_schedule_file(tmp_path / "schedules", "daily-digest.md", SPEC_EXAMPLE, "Daily digest pipeline; see ~/git/daily-digest/README.md for the sections.")
        parsed = store.get("daily-digest")
        assert parsed is not None
        assert parsed.agent == "miranda"
        assert parsed.dest_channel_id == "1504678682290159728"
        assert parsed.platform == "discord"
        assert parsed.payload_type == "command"
        assert json.loads(parsed.payload)["command"] == "python3 /home/shepard/git/daily-digest/digest.py"
        assert parsed.next_run_at == datetime(2026, 10, 27, 7, 0, tzinfo=ZoneInfo("America/Phoenix"))
        assert [s.id for s in store.list_for_agent("miranda")] == ["daily-digest"]


# ---------------------------------------------------------------------------
# AC39: concurrency and crash safety
# ---------------------------------------------------------------------------

WORKER = r"""
import json, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from spctr.schedules import Schedule, ScheduleStore

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)
base = sys.argv[1]
mode = sys.argv[2]
tag = sys.argv[3]
store = ScheduleStore(base, agent="miranda")

def row(i, nra):
    return Schedule(
        id=f"{tag}{i}", agent="miranda", platform="discord",
        requester_id="42", requester_mention="<@42>",
        dest_channel_id="100", dest_channel_name="general", label="t",
        cron="0 7 * * *", payload_type="command",
        payload=json.dumps({"command": "echo hi", "cwd": None}),
        on_miss="run_late", next_run_at=nra, last_run_at=None,
        created_at=NOW, status="active", last_status=None, last_error=None,
    )

try:
    if mode == "writer":
        ids = []
        for i in range(15):
            ids.append(store.create(row(i, NOW - timedelta(seconds=1))))
            if i % 3 == 0:
                store.cancel(ids[i // 3])
            if i % 2 == 0:
                store.advance_after_run(ids[(i // 2) % len(ids)], NOW + timedelta(minutes=i), NOW, "ok", None)
        print("writer-done " + ",".join(ids))
    else:
        for i in range(15):
            due = store.due(NOW + timedelta(hours=1))
            for s in due:
                if store.claim(s.id):
                    store.advance_after_run(s.id, NOW + timedelta(hours=2, minutes=i), NOW, "ok", None)
            time.sleep(0.005)
        print("engine-done")
except Exception as e:
    print(f"worker-error: {type(e).__name__}: {e}")
    sys.exit(1)
"""


def _run_worker(base: str, mode: str, tag: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", WORKER, base, mode, tag],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )
    return proc.stdout.strip()


class TestAC39Concurrency:
    def test_two_processes_no_lost_or_corrupted(self, tmp_path):
        """AC39-r1: two concurrent writers on one directory lose nothing."""
        base = tmp_path / "schedules"
        base.mkdir()
        p1 = subprocess.Popen(
            [sys.executable, "-c", WORKER, str(base), "writer", "w"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        p2 = subprocess.Popen(
            [sys.executable, "-c", WORKER, str(base), "engine", "e"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        outs = []
        for p in (p1, p2):
            out, err = p.communicate(timeout=120)
            outs.append(out.strip())
            assert p.returncode == 0, err
        assert outs[0].startswith("writer-done")
        assert outs[1] == "engine-done"

        store = ScheduleStore(base, agent="miranda")
        # Every surviving file parses; no file is left corrupt.
        valid, invalid = store.scan()
        assert invalid == []
        # No schedule lost and none extra: exactly the schedules the
        # writer created and did NOT cancel survive — the engine only
        # claims/advances.  ids[0..4] is the cancelled set (i % 3 == 0
        # cancels ids[i // 3]); a cancelled stem may be reused by a later
        # create, so compare against the writer's own id list.
        created = outs[0].split(" ", 1)[1].split(",")
        assert len(created) == 15
        cancelled = set(created[:5])
        assert {s.id for s in valid} == set(created[5:]) - cancelled
        assert len(valid) == 10
        # Every mutation landed as a complete file: each surviving file
        # re-parses identically through a fresh store.
        fresh = ScheduleStore(base, agent="miranda")
        for s in valid:
            assert fresh.get(s.id) == s
        # No temp files remain for finished work.
        assert [p.name for p in base.iterdir() if p.name.startswith(".tmp-")] == []


KILL_HELPER = r"""
import os, sys, time
from spctr.schedules.store import ScheduleStore, _render_document, _runtime_list, _user_keys
from spctr.schedules import Schedule
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
NOW = datetime(2026, 10, 20, 14, 0, 0, tzinfo=UTC)
base = sys.argv[1]
store = ScheduleStore(base, agent="miranda")
row = Schedule(
    id="k1", agent="miranda", platform="discord", requester_id="", requester_mention="",
    dest_channel_id="100", dest_channel_name="general", label="test", cron="0 7 * * *",
    payload_type="command", payload=json.dumps({"command": "echo hi", "cwd": None}),
    on_miss="run_late", next_run_at=NOW - timedelta(seconds=1), last_run_at=None,
    created_at=NOW, status="active", last_status=None, last_error=None,
)
store.create(row)
# Simulate: temp written, process dies before os.replace.
text = store._render_row(row)
tmp = store._dir / ".tmp-killed-0001-k1.md"
tmp.write_text(text, encoding="utf-8")
print("killed-ready", flush=True)
time.sleep(30)
"""


class TestAC39KillBetweenWriteAndReplace:
    def test_original_readable_and_temp_swept(self, tmp_path):
        """AC39-r2: kill between temp write and replace leaves the original
        readable; the leftover temp is swept after one poll interval."""
        base = tmp_path / "schedules"
        base.mkdir()
        store = ScheduleStore(base, agent="miranda")
        row = _make_schedule(next_run_at=NOW - timedelta(seconds=1))
        store.create(row)
        original = store.get("s1")
        original_bytes = (base / "s1.md").read_bytes()

        proc = subprocess.Popen(
            [sys.executable, "-c", KILL_HELPER, str(base)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        line = proc.stdout.readline().strip()
        assert line == "killed-ready"
        proc.kill()
        proc.wait(timeout=10)

        # The original file is still readable and uncorrupted: its bytes
        # and parsed row are exactly what the writer committed.
        assert (base / "s1.md").read_bytes() == original_bytes
        fresh = ScheduleStore(base, agent="miranda")
        parsed = fresh.get("s1")
        assert parsed == original
        # The leftover temp is not mistaken for a schedule: the scan shows
        # the two committed files as valid and reports nothing invalid.
        valid_now, invalid_now = fresh.scan()
        assert {s.id for s in valid_now} == {"s1", "s2"}
        assert invalid_now == []

        # The leftover temp is present until one poll interval has passed,
        # then a sweep deletes it.
        temps = [p.name for p in base.iterdir() if p.name.startswith(".tmp-")]
        assert temps == [".tmp-killed-0001-k1.md"]
        # One poll interval later: the temp is past its age and a sweep
        # deletes it. (The test's clock is fake; age is proven by
        # backdating the temp's mtime past one poll interval.)
        import os as _os

        ts = (NOW - timedelta(seconds=35)).timestamp()
        _os.utime(base / ".tmp-killed-0001-k1.md", (ts, ts))
        store.sweep_temp_files(NOW, 30.0)
        assert [p.name for p in base.iterdir() if p.name.startswith(".tmp-")] == []


# ---------------------------------------------------------------------------
# AC37: sequential s<N> assignment; async facades over the directory store
# ---------------------------------------------------------------------------

class TestAC37IdAssignment:
    def test_create_ignores_placeholder_id_and_never_overwrites(self, tmp_path):
        """Two creates with the same placeholder id produce s1 and s2."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        assert store.create(_make_schedule(id="s0")) == "s1"
        assert store.create(_make_schedule(id="s0")) == "s2"
        assert (tmp_path / "schedules" / "s1.md").is_file()
        assert (tmp_path / "schedules" / "s2.md").is_file()
        assert store.get("s1") is not None
        assert store.get("s2") is not None

    def test_create_continues_after_highest_existing_stem(self, tmp_path):
        """A hand-authored s7.md moves the next create to s8."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        _write_schedule_file(
            tmp_path / "schedules", "s7.md",
            USER_BASE + "next_run_at: '2026-10-21T07:00:00+00:00'\n",
            "notes",
        )
        assert store.create(_make_schedule(id="s0")) == "s8"
        # Non-s<N> files do not affect numbering.
        _write_schedule_file(tmp_path / "schedules", "standup.md", USER_BASE, "notes")
        assert store.create(_make_schedule(id="s0")) == "s9"


class TestAC37AsyncFacades:
    @pytest.mark.asyncio
    async def test_async_facades_match_sync_contract(self, tmp_path):
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        assert await store.create_async(
            _make_schedule(next_run_at=NOW - timedelta(seconds=1))
        ) == "s1"

        store2 = ScheduleStore(tmp_path / "schedules", agent="miranda")
        due = await store2.due_async(NOW)
        assert [s.id for s in due] == ["s1"]

        assert await store.claim_async("s1") is True
        assert await store.claim_async("s1") is False  # exactly-once

        nxt = NOW + timedelta(days=1)
        await store.advance_after_run_async("s1", nxt, NOW, "ok", None)
        row = await store.get_async("s1")
        assert row.next_run_at == nxt and row.last_status == "ok"

        assert await store.cancel_async("s1") is True
        assert await store.cancel_async("s1") is False

        rows = await store.list_for_agent_async("miranda")
        assert rows == []
        assert await store.get_async("s1") is None


class TestAC37Recovery:
    def test_claimed_row_is_recovered_by_a_fresh_store(self, tmp_path):
        """AC37-r1: a restart finds the schedule left at the claim sentinel."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule(next_run_at=NOW - timedelta(seconds=1)))
        assert store.claim("s1") is True

        # A restart: a fresh store object over the same directory.
        fresh = ScheduleStore(tmp_path / "schedules", agent="miranda")
        recovered = fresh.recover_claimed(NOW)
        assert [s.id for s in recovered] == ["s1"]
        assert recovered[0].next_run_at == datetime.fromisoformat(SENTINEL)
        assert recovered[0].status == "active"
        assert recovered[0].agent == "miranda"
        assert json.loads(recovered[0].payload)["command"] == "echo hi"

    def test_recover_claimed_scopes_to_platform(self, tmp_path):
        """AC37-r1: recovery honors the platform filter."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule(
            platform="discord", next_run_at=NOW - timedelta(seconds=1)
        ))
        store.create(_make_schedule(
            platform="zulip", next_run_at=NOW - timedelta(seconds=1)
        ))
        assert store.claim("s1") is True
        assert store.claim("s2") is True

        fresh = ScheduleStore(tmp_path / "schedules", agent="miranda")
        assert [s.id for s in fresh.recover_claimed(NOW, platform="discord")] == ["s1"]
        assert [s.id for s in fresh.recover_claimed(NOW, platform="zulip")] == ["s2"]
        assert [s.id for s in fresh.recover_claimed(NOW)] == ["s1", "s2"]

    def test_unclaimed_due_and_future_rows_are_not_recovered(self, tmp_path):
        """AC37-r1: only sentinel-stuck rows are recovered."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule(next_run_at=NOW - timedelta(seconds=1)))  # s1 due
        store.create(_make_schedule(next_run_at=NOW + timedelta(days=1)))     # s2 future

        fresh = ScheduleStore(tmp_path / "schedules", agent="miranda")
        assert fresh.recover_claimed(NOW) == []
        # The due file is still live and due; the future one still waits.
        assert [s.id for s in fresh.due(NOW)] == ["s1"]

    @pytest.mark.asyncio
    async def test_recover_claimed_async_matches_sync(self, tmp_path):
        """AC37-r1: the async facade returns the same recovered row."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule(next_run_at=NOW - timedelta(seconds=1)))
        assert store.claim("s1") is True

        fresh = ScheduleStore(tmp_path / "schedules", agent="miranda")
        recovered = await fresh.recover_claimed_async(NOW)
        assert [s.id for s in recovered] == ["s1"]
        assert recovered[0].next_run_at == datetime.fromisoformat(SENTINEL)

    def test_advance_after_run_clears_the_recovered_row(self, tmp_path):
        """A recovered row leaves the recovery set once the engine advances it."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        store.create(_make_schedule(next_run_at=NOW - timedelta(seconds=1)))
        assert store.claim("s1") is True

        fresh = ScheduleStore(tmp_path / "schedules", agent="miranda")
        assert [s.id for s in fresh.recover_claimed(NOW)] == ["s1"]
        nxt = NOW + timedelta(days=1)
        fresh.advance_after_run("s1", nxt, NOW, "ok", None)
        assert fresh.recover_claimed(NOW) == []
        row = fresh.get("s1")
        assert row.next_run_at == nxt
        assert row.last_status == "ok"


# ---------------------------------------------------------------------------
# AC37-r2: custom schedules.store_path template honored end to end
# ---------------------------------------------------------------------------

class TestAC37CustomStorePath:
    def test_custom_store_path_template(self, tmp_path):
        """AC37-r2: resolved_store_path honors a custom {agent} template."""
        from spctr.config import SchedulesConfig, resolve_sr2_home

        cfg = SchedulesConfig(store_path=str(tmp_path / "custom" / "{agent}"))
        resolved = cfg.resolved_store_path("miranda")
        assert resolved == str(tmp_path / "custom" / "miranda")

        store = ScheduleStore(resolved, agent="miranda")
        assert store.create(_make_schedule()) == "s1"
        assert (tmp_path / "custom" / "miranda" / "s1.md").is_file()
        assert store.get("s1") is not None
        assert [s.id for s in store.list_for_agent("miranda")] == ["s1"]

        # Nothing was written under the default state/{agent}/schedules/.
        state_root = Path(resolve_sr2_home()) / "state"
        assert not (state_root / "miranda" / "schedules").exists()
        leftovers = list(state_root.rglob("*.md")) if state_root.exists() else []
        assert leftovers == []

    def test_tilde_in_store_path_expanded(self):
        """AC37-r2: a '~' store_path resolves under the user's home."""
        from spctr.config import SchedulesConfig

        cfg = SchedulesConfig(store_path="~/sched-x/{agent}")
        resolved = cfg.resolved_store_path("miranda")
        assert resolved == str(Path.home() / "sched-x" / "miranda")
        assert not resolved.startswith("~")


# ---------------------------------------------------------------------------
# Scan primitive: first-seen files, paused files, invalid files (FR39/40/44)
# ---------------------------------------------------------------------------

class TestScanPrimitive:
    def test_first_seen_no_next_run_at_is_valid_and_not_due(self, tmp_path):
        """FR39: a file with no next_run_at is a live schedule the engine
        schedules on its next poll — not an invalid file."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        _write_schedule_file(tmp_path / "schedules", "new.md", USER_BASE, "notes")
        valid, invalid = store.scan()
        assert invalid == []
        assert [s.id for s in valid] == ["new"]
        assert valid[0].next_run_at is None
        assert valid[0].status == "active"
        # Not due and not claimable until the engine stores an occurrence.
        assert store.due(NOW) == []
        assert store.claim("new") is False
        store.advance_after_run("new", NOW + timedelta(days=1), NOW, "ok", None)
        assert store.get("new").next_run_at == NOW + timedelta(days=1)
        assert store.due(NOW) == []
        assert [s.id for s in store.due(NOW + timedelta(days=2))] == ["new"]

    def test_closing_fence_at_eof_without_newline(self, tmp_path):
        """A closing --- fence at EOF (no trailing newline) parses."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        (tmp_path / "schedules" / "eof.md").write_text(
            "---\n" + USER_BASE + "next_run_at: '2026-10-21T07:00:00+00:00'\n---",
            encoding="utf-8",
        )
        row = store.get("eof")
        assert row is not None
        assert row.id == "eof"
        assert json.loads(row.payload)["command"] == "echo hi"

    def test_unquoted_yaml_timestamp_accepted(self, tmp_path):
        """A hand-authored unquoted ISO timestamp parses as a datetime."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        _write_schedule_file(
            tmp_path / "schedules", "ts.md",
            USER_BASE + "next_run_at: 2026-10-21T07:00:00+00:00\n",
            "notes",
        )
        row = store.get("ts")
        assert row is not None
        assert row.next_run_at == datetime(2026, 10, 21, 7, 0, tzinfo=UTC)

    def test_enabled_false_pauses(self, tmp_path):
        """FR44: enabled: false is valid, never due/claimed, cancellable."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        _write_schedule_file(
            tmp_path / "schedules", "parked.md",
            USER_BASE + "enabled: false\nnext_run_at: '2026-10-19T07:00:00+00:00'\n",
            "notes",
        )
        valid, invalid = store.scan()
        assert invalid == []
        assert valid[0].status == "paused"
        # Past due but paused: never due, never claimed.
        assert store.due(NOW) == []
        assert store.claim("parked") is False
        # Hidden from the default list; listed as paused when asked.
        assert store.list_for_agent("miranda") == []
        assert [
            s.id for s in store.list_for_agent("miranda", statuses=("paused",))
        ] == ["parked"]
        # A paused schedule is still a schedule: cancel removes it.
        assert store.cancel("parked") is True
        assert store.get("parked") is None

    def test_invalid_files_reported_with_reasons(self, tmp_path):
        """FR40: every invalid shape lands in scan()'s invalid list."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        base = tmp_path / "schedules"
        fence = "---\n{keys}\n---\n\nnotes\n"
        good_keys = (
            "cron: 0 7 * * *\ntype: command\ncommand: x\nplatform: discord\n"
            "next_run_at: '2026-10-21T07:00:00+00:00'\n"
        )
        bad = {
            "BAD.md": fence.format(keys=good_keys),
            "nofences.md": "just text, no frontmatter\n",
            "badyaml.md": "---\ncron: [unclosed\n---\n",
            "nocron.md": fence.format(
                keys="type: command\ncommand: x\nplatform: discord\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badcron.md": fence.format(
                keys="cron: 0 7 *** *\ntype: command\ncommand: x\nplatform: discord\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badtype.md": fence.format(
                keys="cron: 0 7 * * *\ntype: rocket\nplatform: discord\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badonmiss.md": fence.format(
                keys="cron: 0 7 * * *\ntype: command\ncommand: x\nplatform: discord\n"
                "on_miss: maybe\nnext_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badenabled.md": fence.format(
                keys="cron: 0 7 * * *\ntype: command\ncommand: x\nplatform: discord\n"
                "enabled: banana\nnext_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "badplatform.md": fence.format(
                keys="cron: 0 7 * * *\ntype: command\ncommand: x\nplatform: telegraph\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
            "noplatform.md": fence.format(
                keys="cron: 0 7 * * *\ntype: command\ncommand: x\n"
                "next_run_at: '2026-10-21T07:00:00+00:00'\n"
            ),
        }
        for name, text in bad.items():
            (base / name).write_text(text, encoding="utf-8")
        (base / "big.md").write_text(
            fence.format(keys=good_keys) + "x" * (64 * 1024), encoding="utf-8"
        )
        (base / "bigbody.md").write_text(
            "---\ncron: 0 7 * * *\ntype: agent\nplatform: discord\n---\n\n"
            + "x" * 4001,
            encoding="utf-8",
        )
        valid, invalid = store.scan()
        assert valid == []
        reasons = {e.filename: e.reason for e in invalid}
        assert set(reasons) == set(bad) | {"big.md", "bigbody.md"}
        assert reasons["BAD.md"] == "invalid id stem"
        assert reasons["noplatform.md"] == "platform required"
        assert "cron" in reasons["nocron.md"]
        assert "cron" in reasons["badcron.md"]
        assert "on_miss" in reasons["badonmiss.md"]
        assert "enabled" in reasons["badenabled.md"]
        assert "platform" in reasons["badplatform.md"]
        assert "64 KiB" in reasons["big.md"]
        assert "4000" in reasons["bigbody.md"]

    def test_invalid_file_does_not_block_valid_sibling(self, tmp_path):
        """FR40: an invalid file is skipped entirely; siblings unaffected."""
        store = ScheduleStore(tmp_path / "schedules", agent="miranda")
        (tmp_path / "schedules" / "broken.md").write_text(
            "no fences here", encoding="utf-8"
        )
        store.create(_make_schedule(next_run_at=NOW - timedelta(seconds=1)))
        valid, invalid = store.scan()
        assert [s.id for s in valid] == ["s1"]
        assert [e.filename for e in invalid] == ["broken.md"]
        assert [s.id for s in store.due(NOW)] == ["s1"]
        assert store.claim("s1") is True
        # The invalid file is never due and never claimed; cancel still
        # removes it (cancel is file deletion).
        assert store.claim("broken") is False
        assert [s.id for s in store.due(NOW + timedelta(days=30))] == []
        assert store.cancel("broken") is True
