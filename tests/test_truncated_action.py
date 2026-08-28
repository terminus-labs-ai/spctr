"""obsidian-hx00: truncated/malformed structured tool calls must not surface
as opaque JSON parser failures.

Regression coverage for the hrb-36.3 external-executor failure: a tool call
larger than the 3,000-token response allowance produced a structured response
that was cut mid-string, and the runner died with
``JSONDecodeError: Unterminated string`` before making any repository change.

Covers:
  A. action_safety.validate_structured_action — the truncated boundary and
     other malformed shapes, with bounded actionable messages.
  B. LiveLLM.stream — a provider-side JSONDecodeError on the accumulated
     argument stream is converted into TruncatedActionError (no silent
     token-budget increase anywhere on the path).
  C. Session._execute_tool — a raw-string action input (one that slipped
     through a provider's parser) is rejected with a bounded
     ToolResultBlock(is_error=True); the registry is never called.
  D. Session.stream_message — a malformed-action failure mid-turn is
     surfaced as a bounded AgentError event, not an uncaught traceback.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_spectre.action_safety import (
    MalformedActionError,
    TruncatedActionError,
    validate_structured_action,
)
from sr2_spectre.config import AgentConfig, ModelConfig, SpectreConfig
from sr2_spectre.events import AgentDone, AgentError
from sr2_spectre.live_llm import LiveLLM
from sr2_spectre.session import Session

# A truncated argument stream: the exact shape produced when a 3,000-token
# response allowance cuts a tool call inside a string value.
_TRUNCATED_ARGS = '{"path": "src/mod'
_UNTERMINATED_STRING = "Unterminated string"


def _msg_of(exc: Exception) -> str:
    """Extract just the reason text from a MalformedActionError message."""
    # Message format: "<prefix> could not be used as a structured action: <reason>. <remedy>"
    return exc.args[0]


# ---------------------------------------------------------------------------
# A. validate_structured_action — the boundary
# ---------------------------------------------------------------------------


class TestValidateStructuredAction:
    def test_valid_action_parses(self):
        out = validate_structured_action('{"path": "src/mod.py", "content": "x"}')
        assert out == {"path": "src/mod.py", "content": "x"}

    def test_truncated_stream_raises_truncated(self):
        with pytest.raises(TruncatedActionError) as ei:
            validate_structured_action(_TRUNCATED_ARGS, tool_name="file_write")
        msg = _msg_of(ei.value)
        assert "file_write" in msg
        # Bounded + actionable: names the condition and the fix.
        assert "truncated" in msg
        assert "smaller" in msg.lower()
        # The remedy is decomposing the action, not raising the budget.
        assert "token budget" in msg

    def test_unterminated_string_is_the_truncation_case(self):
        # The observed hrb-36.3 failure mode, verbatim: the parser's own
        # message is the unterminated-string kind.
        with pytest.raises(TruncatedActionError):
            validate_structured_action('{"a": "unterminated')
        try:
            json.loads('{"a": "unterminated')
        except json.JSONDecodeError as exc:
            assert exc.msg.startswith(_UNTERMINATED_STRING)

    def test_non_object_decode_rejected(self):
        with pytest.raises(MalformedActionError):
            validate_structured_action('[1, 2, 3]', tool_name="t")
        with pytest.raises(MalformedActionError):
            validate_structured_action('"just a string"')

    def test_excessive_depth_rejected(self):
        node = "x"
        for _ in range(40):
            node = {"nest": node}
        deep = json.dumps(node)
        with pytest.raises(MalformedActionError) as ei:
            validate_structured_action(deep, max_depth=32)
        assert "too deep" in _msg_of(ei.value)


# ---------------------------------------------------------------------------
# B. LiveLLM.stream — provider parse failure becomes a bounded capability error
# ---------------------------------------------------------------------------


def _model_cfg() -> ModelConfig:
    return ModelConfig(model="test-model", base_url="http://test:8000/v1")


class TestLiveLLMStreamTruncation:
    async def test_json_decode_error_becomes_truncated_action_error(self):
        async def _broken_stream(request):
            yield {"type": "tool_use", "tool_name": "file_write"}
            # Simulates sr2/integrations/litellm.py: json.loads of an
            # accumulated argument stream that the 3,000-token allowance
            # cut inside a string.
            raise json.JSONDecodeError(
                _UNTERMINATED_STRING, _TRUNCATED_ARGS, 12
            )
            yield  # pragma: no cover — makes this an async generator

        live = LiveLLM(_model_cfg())
        with patch.object(live, "_inner") as inner:
            inner.stream = _broken_stream
            with pytest.raises(TruncatedActionError) as ei:
                async for _ in live.stream(None):
                    pass
        exc = ei.value
        # The original parser detail is preserved (no opacity added).
        assert _UNTERMINATED_STRING in exc.detail
        # Bounded + actionable: names the condition and the fix.
        msg = _msg_of(exc)
        assert "truncated" in msg
        assert "smaller" in msg.lower()
        assert "token budget" in msg

    async def test_well_formed_stream_is_untouched(self):
        async def _ok_stream(request):
            yield {"type": "text", "text": "hello"}
            yield {"type": "end"}

        live = LiveLLM(_model_cfg())
        with patch.object(live, "_inner") as inner:
            inner.stream = _ok_stream
            seen = [ev async for ev in live.stream(None)]
        assert len(seen) == 2


# ---------------------------------------------------------------------------
# C. Session._execute_tool — malformed action rejected before execution
# ---------------------------------------------------------------------------


def _make_config() -> SpectreConfig:
    return SpectreConfig(
        agent=AgentConfig(name="t"),
        models={"default": ModelConfig(model="m", base_url="http://x/v1")},
        pipeline={"layers": []},
    )


def _make_session() -> Session:
    with patch("sr2_spectre.session.SR2") as mock_sr2:
        mock_sr2.return_value = MagicMock()
        return Session(
            frame_id="f1",
            config=_make_config(),
            llm=MagicMock(),
            registry=MagicMock(),
        )


def _raw_string_block(block_input: str):
    """Build a ToolUseBlock whose input is a raw argument stream.

    A well-formed provider yields a dict; a stream that never made it through
    a parser arrives as a raw string. ToolUseBlock's pydantic validation
    rejects raw strings at construction, so bypass validation to model the
    bypassed-parser case (which is exactly what the session gate guards).
    """
    from sr2.models import ToolUseBlock

    return ToolUseBlock.model_construct(
        id="tu-1", name="file_write", input=block_input
    )


class TestExecuteToolMalformedAction:
    async def test_truncated_string_input_is_rejected_without_execution(self):
        session = _make_session()
        session._registry.execute = AsyncMock()

        result = await session._execute_tool(_raw_string_block(_TRUNCATED_ARGS))

        assert result.is_error is True
        assert "truncated" in result.content
        # Bounded and actionable, not a raw parser traceback.
        assert "JSONDecodeError" not in result.content
        assert "Traceback" not in result.content
        assert "token budget" in result.content
        # The registry was never called — no half-applied action.
        session._registry.execute.assert_not_awaited()

    async def test_valid_string_input_is_parsed_and_executed(self):
        session = _make_session()
        session._registry.execute = AsyncMock(return_value="ok")

        block = _raw_string_block('{"path": "a.py"}')
        block.id = "tu-2"

        result = await session._execute_tool(block)
        assert result.is_error is False
        args = session._registry.execute.await_args.args
        assert args[1] == {"path": "a.py"}

    async def test_dict_input_is_not_touched(self):
        session = _make_session()
        session._registry.execute = AsyncMock(return_value="ok")

        from sr2.models import ToolUseBlock

        block = ToolUseBlock(
            id="tu-3", name="file_write", input={"path": "a.py", "content": "x"}
        )
        result = await session._execute_tool(block)
        assert result.is_error is False
        args = session._registry.execute.await_args.args
        # Dict input passed through unmodified.
        assert args[1] == {"path": "a.py", "content": "x"}


# ---------------------------------------------------------------------------
# D. Session.stream_message — bounded AgentError instead of a traceback
# ---------------------------------------------------------------------------


class TestStreamMessageMalformedAction:
    async def test_truncated_action_surfaces_as_agent_error(self):
        session = _make_session()

        async def _failing_turn(user_input, *, origin: str = ""):
            raise TruncatedActionError(
                "file_write", detail=f"{_UNTERMINATED_STRING} (at char 12)"
            )
            yield  # pragma: no cover

        session.sr2 = MagicMock()
        session.sr2.seed_session = MagicMock()
        session.sr2.turn = _failing_turn

        events = [ev async for ev in session.stream_message("big edit please")]

        # No traceback escaped: the stream completes and reports the failure.
        assert any(isinstance(ev, AgentError) for ev in events)
        err = next(ev for ev in events if isinstance(ev, AgentError))
        assert "truncated" in err.message
        assert "token budget" in err.message
        # Bounded: no opaque parser traceback in the surfaced message.
        assert "Traceback" not in err.message
        assert "JSONDecodeError" not in err.message
        # The turn failed before any tool executed: history holds the user
        # message plus a single assistant message carrying the bounded error
        # notice — no orphan tool entries, so a retry turn starts clean.
        assert [m.role for m in session.history] == ["user", "assistant"]
        from sr2.models import TextBlock

        assistant_blocks = session.history[-1].content
        assert any(
            isinstance(b, TextBlock) and "truncated" in b.text
            for b in assistant_blocks
        )
        # The stream still ends with AgentDone, so consumers don't need
        # exception handling to know the turn terminated.
        assert isinstance(events[-1], AgentDone)
