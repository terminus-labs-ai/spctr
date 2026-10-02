"""LiveLLM — a stable LLM handle whose target can be swapped underneath it.

``SR2`` captures its ``LLMCallable`` at construction and holds it for the life
of the session, so replacing ``Runtime.llm`` on a config reload would only
affect sessions created *after* the reload. Conversations already open would
keep calling the old endpoint until they were rebuilt — which is exactly the
failure this indirection removes: an operator fixing a wrong ``base_url``
expects the next message in the channel they are already talking in to reach
the new endpoint.

So sessions are handed a ``LiveLLM`` instead. It satisfies the ``LLMCallable``
protocol by delegating to an inner ``LiteLLMCallable``, and ``retarget()``
swaps that inner instance atomically. Every session, open or not, follows.

The swap is whole-object: a request already in flight finishes against the
instance it started with, and the next request picks up the new one. Nothing
observes a half-applied endpoint change.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from litellm.exceptions import APIConnectionError, MidStreamFallbackError
from sr2.integrations.litellm import LiteLLMCallable

if TYPE_CHECKING:
    from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent

from sr2_spectre.config import ModelConfig
from sr2_spectre.run_log import SessionLog, summarize_request

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS: int = 32768


class ModelCallGuardError(RuntimeError):
    """A model call was aborted for running too long or going silent."""

    def __init__(
        self, kind: str, limit_seconds: float, elapsed_seconds: float, model: str
    ) -> None:
        super().__init__(
            f"model call aborted: {kind} limit of {limit_seconds:g}s exceeded "
            f"(elapsed {elapsed_seconds:.1f}s, model={model})"
        )
        self.kind = kind
        self.limit_seconds = float(limit_seconds)
        self.elapsed_seconds = float(elapsed_seconds)


def _guard_abort(kind: str, limit: float, started: float, model: str) -> ModelCallGuardError:
    elapsed = time.monotonic() - started
    logger.warning(
        "Model call aborted — kind=%s limit=%gs elapsed=%.1fs model=%s",
        kind, limit, elapsed, model,
    )
    return ModelCallGuardError(kind, limit, elapsed, model)


def build_llm(model_cfg: ModelConfig) -> LiteLLMCallable:
    """Build a LiteLLMCallable from a model config block.

    Single definition of how a ``ModelConfig`` becomes a callable, shared by
    startup and reload so the two can never drift.
    """
    kwargs: dict[str, Any] = {
        "model": model_cfg.model,
        "base_url": model_cfg.base_url,
    }
    if model_cfg.api_key:
        kwargs["api_key"] = model_cfg.api_key
    if model_cfg.params:
        kwargs.update(model_cfg.params)
    if "max_tokens" not in model_cfg.params and "max_completion_tokens" not in model_cfg.params:
        kwargs["max_tokens"] = DEFAULT_MAX_OUTPUT_TOKENS
    return LiteLLMCallable(**kwargs)


class LiveLLM:
    """An ``LLMCallable`` that forwards to a swappable inner callable."""

    def __init__(self, model_cfg: ModelConfig) -> None:
        self._model_cfg = model_cfg
        self._inner = build_llm(model_cfg)

    @property
    def model_config(self) -> ModelConfig:
        """The model config the current target was built from."""
        return self._model_cfg

    @property
    def model(self) -> str:
        """The model id in force, as litellm sees it (provider prefix included)."""
        return self._inner.model

    def retarget(self, model_cfg: ModelConfig) -> bool:
        """Point at a new model config. Returns True if anything changed.

        A no-op when the config is unchanged, so the common case — a reload
        that found nothing new — does not churn the callable or the log.
        """
        if model_cfg == self._model_cfg:
            return False

        self._model_cfg = model_cfg
        self._inner = build_llm(model_cfg)
        logger.info(
            "LLM retargeted — model=%s base_url=%s",
            model_cfg.model,
            model_cfg.base_url,
        )
        return True

    async def complete(self, request: "CompletionRequest") -> "CompletionResponse":
        inner, limit, model = self._inner, self._model_cfg.call_timeout_seconds, self.model
        started = time.monotonic()
        try:
            return await asyncio.wait_for(inner.complete(request), limit)
        except TimeoutError:
            if limit is None:
                raise
            raise _guard_abort("duration", limit, started, model) from None

    async def stream(self, request: "CompletionRequest") -> AsyncIterator["StreamEvent"]:
        # Bound once, up front: a retarget part-way through must not splice two
        # endpoints into a single response. Guard limits bind at the same point.
        inner, model = self._inner, self.model
        call_limit = self._model_cfg.call_timeout_seconds
        stall_limit = self._model_cfg.stall_timeout_seconds
        started = time.monotonic()
        it = inner.stream(request).__aiter__()
        try:
            while True:
                remaining = None if call_limit is None else call_limit - (time.monotonic() - started)
                timeout = min((t for t in (remaining, stall_limit) if t is not None), default=None)
                try:
                    if timeout is not None and timeout <= 0:
                        raise TimeoutError
                    event = await asyncio.wait_for(anext(it), timeout)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    if call_limit is not None and time.monotonic() - started >= call_limit:
                        raise _guard_abort("duration", call_limit, started, model) from None
                    if stall_limit is None:
                        raise
                    raise _guard_abort("stall", stall_limit, started, model) from None
                yield event
        finally:
            aclose = getattr(it, "aclose", None)
            if aclose is not None:
                await aclose()


class LoggedLLM:
    """Per-Session ``LLMCallable`` that logs model calls, then delegates.

    Wraps the shared ``LiveLLM`` so hot retargeting is preserved: every request
    delegates to ``inner`` afresh, and ``LiveLLM.stream`` binds its current
    target once. Requests and responses pass through unchanged.
    """

    def __init__(
        self,
        inner: "LiveLLM",
        log_provider: Callable[[], SessionLog | None],
        profile_provider: Callable[[], str],
    ) -> None:
        self._inner = inner
        self._log_provider = log_provider
        self._profile_provider = profile_provider

    def _start(self, log: SessionLog, request: Any) -> float:
        log.append(
            "model.start",
            {
                "profile": self._profile_provider(),
                "model": self._inner.model,
                **summarize_request(request),
            },
        )
        return time.monotonic()

    async def complete(self, request: "CompletionRequest") -> "CompletionResponse":
        log = self._log_provider()
        if log is None:
            return await self._inner.complete(request)
        started = self._start(log, request)
        try:
            response = await self._inner.complete(request)
        except BaseException as exc:
            log.append("model.error", _error_data(exc, started))
            raise
        log.append(
            "model.end",
            {
                "finish_reason": response.stop_reason,
                "usage": response.usage.model_dump(),
                "duration_ms": _elapsed_ms(started),
            },
        )
        return response

    async def stream(self, request: "CompletionRequest") -> AsyncIterator["StreamEvent"]:
        log = self._log_provider()
        if log is None:
            async for event in self._inner.stream(request):
                yield event
            return
        started = self._start(log, request)
        out_chars = 0
        usage: dict[str, Any] | None = None
        finish: Any = None
        terminal, data = "model.cancel", {}
        try:
            async for event in self._inner.stream(request):
                if event.type in ("text", "thinking") and event.text:
                    out_chars += len(event.text)
                    log.append(
                        "model.progress",
                        {
                            "kind": event.type,
                            "preview": event.text,
                            "output_tokens_estimate": out_chars // 4,
                        },
                    )
                elif event.type == "usage" and event.usage is not None:
                    usage = event.usage.model_dump()
                elif event.type == "end":
                    finish = (event.meta or {}).get("finish_reason")
                yield event
            terminal = "model.end"
            data = {"finish_reason": finish, "usage": usage}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            terminal, data = "model.error", _error_data(exc, started)
            raise
        finally:
            data.setdefault("duration_ms", _elapsed_ms(started))
            log.append(terminal, data)


RETRYABLE_STREAM_ERRORS: tuple[type[BaseException], ...] = (
    MidStreamFallbackError,
    APIConnectionError,
)


class RetryingLLM:
    """Per-Session ``LLMCallable`` that retries a streamed call that dies.

    Each attempt is buffered and yielded only after it finishes, so events from
    a failed attempt never reach SR2. Only transport errors are retried; guard
    aborts, cancellation and every other error propagate after one attempt.
    Retry settings are read at the start of each ``stream()`` call.
    """

    def __init__(
        self, inner: Any, config_provider: Callable[[], ModelConfig]
    ) -> None:
        self._inner = inner
        self._config_provider = config_provider

    async def complete(self, request: "CompletionRequest") -> "CompletionResponse":
        return await self._inner.complete(request)

    async def stream(self, request: "CompletionRequest") -> AsyncIterator["StreamEvent"]:
        cfg = self._config_provider()
        retries, backoff = cfg.stream_retries, cfg.stream_retry_backoff_seconds
        attempt = 0
        while True:
            buffered: list[Any] = []
            try:
                async for event in self._inner.stream(request):
                    buffered.append(event)
            except RETRYABLE_STREAM_ERRORS as exc:
                if attempt >= retries:
                    raise
                delay = backoff * 2**attempt
                attempt += 1
                logger.warning(
                    "Model stream failed — retry %d/%d in %gs error=%s model=%s",
                    attempt, retries, delay, type(exc).__name__, cfg.model,
                )
                await asyncio.sleep(delay)
                continue
            for event in buffered:
                yield event
            return


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


def _error_data(exc: BaseException, started: float) -> dict[str, Any]:
    return {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "duration_ms": _elapsed_ms(started),
    }
