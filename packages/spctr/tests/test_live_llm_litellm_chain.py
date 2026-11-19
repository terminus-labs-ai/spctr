"""spc-1008: LoggedLLM → LiveLLM → real LiteLLMCallable → provider, one chain.

The two layers of the usage path were each tested against their own fake:
``FakeInner`` in test_live_llm_usage_and_served_model.py yields scripted
``StreamEvent``s whatever it is handed, and sr2's fake ``acompletion`` used to
return a usage chunk whether or not the request asked for one. Nothing in the
suite chained the two, which is the gap that let obsidian-la2z ship streamed
calls with no usage at all.

This file drives one request through the real producer: a genuine
``LiteLLMCallable`` (built by ``build_llm`` from a ``ModelConfig``, exactly as
startup does) behind ``LiveLLM`` behind ``LoggedLLM``, with only the network
boundary — ``litellm.acompletion`` — replaced by a usage-conditional fake. The
fake answers the way the relay does: it emits the trailing usage-only chunk
only when the request carried ``stream_options={"include_usage": True}``.

So the test fails if the client stops asking for usage, if the client stops
turning the chunk it receives into a ``usage`` event, or if ``LoggedLLM``
stops recording what that event carried. The expected values below are the
fake's own literals, stated independently of anything under test.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from sr2.models import Message, TextBlock
from sr2.protocols.llm import CompletionRequest

from spctr.config import ModelConfig
from spctr.live_llm import LiveLLM, LoggedLLM

REQUEST = CompletionRequest(
    messages=[Message(role="user", content=[TextBlock(text="hi")])]
)

# What the fake wire reports. Every assertion below names one of these
# literals; none is read back from the code under test.
SERVED_MODEL = "qwen3.8-flash-next"
POOL_NAME = "pool-name"
API_KEY = "test-key"
INPUT_TOKENS = 17
OUTPUT_TOKENS = 5
FINISH_REASON = "stop"


def _chunk(content=None, *, model=None, finish_reason=None, usage=None):
    """One streaming chunk in the OpenAI chat-completions shape.

    Mirrors the sr2 member's builder — the two member suites each ship a root
    ``tests`` package, so cross-importing that module from here is fragile.
    The usage-only trailing chunk of that protocol carries no choices.
    """
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    delta.reasoning_content = None
    delta.thinking_blocks = None

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = (
        [] if (content is None and finish_reason is None and usage is not None) else [choice]
    )
    chunk.model = model
    chunk.usage = usage
    return chunk


def _usage(prompt_tokens: int, completion_tokens: int) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    return usage


def _is_usage_only_trailing_chunk(chunk) -> bool:
    """True for the usage-only trailing chunk of the OpenAI include_usage shape.

    A chunk carrying text, a finish_reason or a tool-call delta is content the
    caller needs whatever it asked for, so it is emitted unconditionally; the
    guard only ever holds back the usage record.
    """
    return (
        chunk.usage is not None
        and all(getattr(c, "delta", None) is None or c.delta.content is None for c in chunk.choices)
        and all(getattr(c, "finish_reason", None) is None for c in chunk.choices)
    )


def _conditional_acompletion(chunks):
    """Usage-conditional fake of the wire, recording the kwargs it was sent."""
    calls: list[dict] = []

    async def fake(*args, **kwargs):
        calls.append(kwargs)
        include_usage = bool((kwargs.get("stream_options") or {}).get("include_usage"))
        emitted = [
            chunk
            for chunk in chunks
            if include_usage or not _is_usage_only_trailing_chunk(chunk)
        ]

        async def gen():
            for chunk in emitted:
                yield chunk

        return gen()

    fake.calls = calls
    return fake


def _relay_stream():
    """A completion as the relay sends it: text, finish, then the usage chunk."""
    return [
        _chunk(content="ok", model=SERVED_MODEL),
        _chunk(content=None, model=SERVED_MODEL, finish_reason=FINISH_REASON),
        _chunk(model=SERVED_MODEL, usage=_usage(INPUT_TOKENS, OUTPUT_TOKENS)),
    ]


class RecordingLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict]] = []

    def append(self, event, data=None):
        self.entries.append((event, dict(data or {})))

    def named(self, name: str) -> list[dict]:
        return [data for event, data in self.entries if event == name]


async def _drain(llm) -> list:
    return [event async for event in llm.stream(REQUEST)]


class _SharedWire:
    """One conditional wire to run requests down, for the end-state tests.

    Driving the include_usage True and False halves of the task sentence
    against the *same* scripted relay stream leaves no room for the two
    results to differ in anything but the request.
    """

    def __init__(self) -> None:
        self.fake = _conditional_acompletion(_relay_stream())

    async def end_of(self, **params) -> tuple[dict, dict]:
        """Stream one request; ``params`` become ``ModelConfig.params``.

        Returns the ``model.end`` entry and the kwargs the wire received.
        ``ModelConfig`` → ``build_llm`` → ``LiteLLMCallable`` → patched
        ``litellm.acompletion``: no layer of the chain is faked, and this is
        the construction path production uses.
        """
        cfg = ModelConfig(
            model=POOL_NAME,
            base_url="http://test:18000",
            api_key=API_KEY,
            params=params,
        )
        log = RecordingLog()
        with patch("litellm.acompletion", new=self.fake):
            llm = LoggedLLM(LiveLLM(cfg), lambda: log, lambda: "p")
            events = await _drain(llm)

        assert len(log.named("model.end")) == 1, "exactly one terminal entry"
        assert [e.type for e in events][-1] == "end", "the caller still gets the end event"
        return log.named("model.end")[0], self.fake.calls[-1]


async def test_chain_records_usage_and_served_model_from_the_real_callable():
    """AC2: LoggedLLM.model.end carries the fake's token counts and served model."""
    wire = _SharedWire()
    end, kwargs = await wire.end_of()

    assert len(wire.fake.calls) == 1, "one request must go out"
    assert end["usage"] is not None, (
        "the chain dropped the usage event — LoggedLLM recorded None even "
        "though the request asked for usage"
    )
    assert end["usage"]["input_tokens"] == INPUT_TOKENS
    assert end["usage"]["output_tokens"] == OUTPUT_TOKENS
    assert end["served_model"] == SERVED_MODEL
    assert end["finish_reason"] == FINISH_REASON


async def test_chain_records_null_usage_when_include_usage_is_false():
    """AC2's second half: include_usage=False → model.end records usage: None.

    The same chain over the same scripted relay stream; only the request
    differs. So a pass here proves the null is the request's doing and not an
    absence of usage on the wire.
    """
    wire = _SharedWire()
    requested, _ = await wire.end_of()
    assert requested["usage"] is not None, (
        "the shared wire must be capable of delivering usage, or the False "
        "half below proves nothing"
    )

    end, kwargs = await wire.end_of(stream_options={"include_usage": False})

    assert len(wire.fake.calls) == 2, "two requests went out over one wire"
    assert kwargs["stream_options"] == {"include_usage": False}
    assert end["usage"] is None
    # served_model rides on the content chunks, so it outlives the withheld
    # usage chunk: usage: None here means "not requested", never "stream died".
    assert end["served_model"] == SERVED_MODEL
    assert end["finish_reason"] == FINISH_REASON


async def test_default_chain_request_asks_the_wire_for_usage():
    """Why the first test can pass: the default request carries include_usage.

    Pinned separately so a regression that removes the option from the request
    names itself, instead of surfacing only as a bare ``usage is not None``
    failure one layer up.
    """
    _, kwargs = await _SharedWire().end_of()

    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
