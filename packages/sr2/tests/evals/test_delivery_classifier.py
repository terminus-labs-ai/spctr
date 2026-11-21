"""obsidian-wk29 — DeliveryClassifier unit tests (direct bead, C-owned).

AC2 rows: input entry stamped with ``window_time``; background entry
stamped from ``window`` when it lacks ``observed_at``; entry with its own
``observed_at`` left alone; grouping order preserved.

The classifier is driven two ways: directly against a recording
``make_entry`` (AC2 unit semantics), and through the REAL producer —
``_ProductDouble._carrier`` — so the seam both ends of which this bead
authored cannot hide a shape mismatch (layer names, stamping, grouping).
"""

from __future__ import annotations

import hashlib
import inspect
from datetime import UTC, datetime

from sr2.models import Message, TextBlock
from sr2.pipeline.provenance import Entry, EntryOrigin

from evals.continuity.delivery import DeliveryClassifier
from evals.continuity.runner import _ProductDouble

CLASSIFIER = DeliveryClassifier()


def _make_entry(text: str, layer: str) -> Entry:
    """Record the (text, layer) pairing while building a real Entry, so a
    test can assert which layer the classifier requested for a value."""
    _calls.append((text, layer))
    return _entry(text, layer)


def _entry(text: str, layer: str, meta: dict | None = None) -> Entry:
    digest = hashlib.sha256(f"{layer}\n{text}".encode()).hexdigest()
    return Entry(
        id=f"evals-{digest[:20]}",
        content=TextBlock(text=text),
        sources=(),
        origin=EntryOrigin(kind="resolver", name="test"),
        layer=layer,
        session_id="test",
        created_at=datetime(2026, 10, 22, tzinfo=UTC),
        meta=dict(meta or {}),
    )


_calls: list[tuple[str, str]] = []


def _reset() -> None:
    _calls.clear()


def test_input_entry_stamped_with_window_time():
    _reset()
    bg, inp = CLASSIFIER.classify(
        ["background value", "current turn input"],
        input_text="current turn input",
        window_time="2026-06-01T12:00:00Z",
        window={},
        make_entry=_make_entry,
    )
    assert inp[0].content.text == "current turn input"
    assert inp[0].meta["observed_at"] == "2026-06-01T12:00:00Z"
    assert _calls == [
        ("background value", "session"),
        ("current turn input", "input"),
    ]


def test_background_entry_stamped_from_window_when_unstamped():
    _reset()
    bg, inp = CLASSIFIER.classify(
        ["value one", "value two", "turn input"],
        input_text="turn input",
        window_time=None,
        window={"value one": "2026-01-01 00:00:00"},
        make_entry=_make_entry,
    )
    assert [e.content.text for e in inp] == ["turn input"]
    assert len(bg) == 2
    assert bg[0].content.text == "value one"
    assert bg[0].meta["observed_at"] == "2026-01-01 00:00:00"
    # Absent from the window: travels unclassified (R4 semantics).
    assert bg[1].content.text == "value two"
    assert "observed_at" not in bg[1].meta


def test_entry_with_own_observed_at_left_alone():
    # A delivery line carrying its own ``[observed …]`` stamp (lifted into
    # meta by the entry builder) must not be re-stamped from the window.
    stamped_text = "[observed 2026-02-02 00:00:00]\nown stamped value"

    def stamped(text: str, layer: str) -> Entry:
        # Mirrors the runner's _stamped_line: an ``[observed …]`` prefix
        # line is lifted into meta["observed_at"] by the entry builder.
        meta = {}
        head, sep, _ = text.partition("\n")
        if sep and head.startswith("[observed ") and head.endswith("]"):
            meta["observed_at"] = head[len("[observed "):-1]
        return _entry(text, layer, meta=meta)

    bg, inp = CLASSIFIER.classify(
        [stamped_text, "turn input"],
        input_text="turn input",
        window_time=None,
        window={stamped_text: "2026-03-03 00:00:00"},
        make_entry=stamped,
    )
    assert [e.content.text for e in inp] == ["turn input"]
    assert len(bg) == 1
    assert bg[0].meta["observed_at"] == "2026-02-02 00:00:00"


def test_grouping_order_preserved():
    _reset()
    bg, inp = CLASSIFIER.classify(
        ["b1", "in1", "b2", "in1"],
        input_text="in1",
        window_time="W",
        window={"b2": "t2"},
        make_entry=_make_entry,
    )
    # Both buckets preserve delivery order; only the delivered text equal
    # to input_text joins the input bucket (every occurrence).
    assert [e.content.text for e in bg] == ["b1", "b2"]
    assert bg[0].content.text == "b1" and "observed_at" not in bg[0].meta
    assert bg[1].content.text == "b2" and bg[1].meta["observed_at"] == "t2"
    assert [e.content.text for e in inp] == ["in1", "in1"]
    assert all(e.meta["observed_at"] == "W" for e in inp)

    # Blank entries (empty or whitespace-only) are dropped from both
    # buckets; order of the survivors is unchanged.
    _reset()
    bg2, inp2 = CLASSIFIER.classify(
        ["", "b1", "   ", "in1", "  \t"],
        input_text="in1",
        window_time="W",
        window={},
        make_entry=_make_entry,
    )
    assert [e.content.text for e in bg2] == ["b1"]
    assert [e.content.text for e in inp2] == ["in1"]
    assert [t for t, _ in _calls] == ["b1", "in1"]

    # Duplicates: every delivered occurrence is kept, in order.
    _reset()
    bg3, inp3 = CLASSIFIER.classify(
        ["b1", "in1", "b1"],
        input_text="in1",
        window_time=None,
        window={},
        make_entry=_make_entry,
    )
    assert [e.content.text for e in bg3] == ["b1", "b1"]
    assert [e.content.text for e in inp3] == ["in1"]


def test_real_carrier_seam_partition_and_stamping():
    """The classifier through the REAL producer: ``_ProductDouble._carrier``
    must hand the classifier the declared input text / window and the
    resulting CompiledRequest must show the partition, layers, stamps and
    grouping the classifier produced. Expected entry ids are the pinned
    ``_entry`` digests (independently computed), not re-derived from the
    code under test."""
    double = _ProductDouble()
    double.input_text = "ask the agent now"
    double.window_time = "2026-06-01T12:00:00Z"
    double.observed_window = {"restored value: 7": "2026-01-01 00:00:00"}

    request = type("Req", (), {
        "system": [TextBlock(
            text="You are the continuity baseline agent.")],
        "messages": [
            Message.model_construct(role="user", content=[
                TextBlock(text="restored value: 7"),
                TextBlock(text="  "),  # blank: dropped
                TextBlock(text="ask the agent now"),
                TextBlock(text="another: 9"),
            ]),
        ],
    })

    carrier = double._carrier(request)
    assert [e.id for e in carrier.system] == [
        "evals-6b3aa8cddc243cd1a558"]
    # Grouping: background message first (delivery order), then the input
    # message; the blank block is dropped.
    assert len(carrier.messages) == 2
    assert [e.id for e in carrier.messages[0].content] == [
        "evals-58849636b40ad3708106",  # restored value: 7
        "evals-d0932b2fb670f0dabc0c",  # another: 9
    ]
    assert [e.id for e in carrier.messages[1].content] == [
        "evals-98e7fe33b3a93ec3f7a7",  # ask the agent now
    ]

    by_text = {e.content.text: e for e in (
        list(carrier.system)
        + [e for m in carrier.messages for e in m.content])}
    assert by_text["ask the agent now"].layer == "input"
    assert by_text["ask the agent now"].meta["observed_at"] \
        == "2026-06-01T12:00:00Z"
    assert by_text["restored value: 7"].layer == "session"
    assert by_text["restored value: 7"].meta["observed_at"] \
        == "2026-01-01 00:00:00"
    assert by_text["another: 9"].layer == "session"
    assert "observed_at" not in by_text["another: 9"].meta


def test_classifier_is_pure_and_stateless():
    """No hidden state: identical calls return equal partitions, and the
    classifier holds no reference to _ProductDouble."""
    _reset()
    a = CLASSIFIER.classify(
        ["x", "in"], input_text="in", window_time="W",
        window={"x": "t"}, make_entry=_make_entry)
    b = CLASSIFIER.classify(
        ["x", "in"], input_text="in", window_time="W",
        window={"x": "t"}, make_entry=_make_entry)
    assert [e.id for e in a[0]] == [e.id for e in b[0]]
    assert [e.id for e in a[1]] == [e.id for e in b[1]]
    assert not any(isinstance(v, _ProductDouble)
                   for v in vars(CLASSIFIER).values())


def test_carrier_delegates_classification_to_classifier():
    """AC1 check on the real module: ``_carrier``'s source contains no
    equality against ``self.input_text`` and no ``observed_at``
    assignment — both live in DeliveryClassifier, which has no reference
    to the SR2 driving loop or to _ProductDouble."""
    from evals.continuity import runner as crunner

    src = inspect.getsource(crunner._ProductDouble._carrier)
    assert "text == self.input_text" not in src
    assert 'meta["observed_at"]' not in src
    assert "observed_at" in inspect.getsource(DeliveryClassifier)
    cls_src = inspect.getsource(DeliveryClassifier)
    assert "_ProductDouble" not in cls_src
    assert "run_baseline" not in cls_src
    assert "SR2" not in cls_src
