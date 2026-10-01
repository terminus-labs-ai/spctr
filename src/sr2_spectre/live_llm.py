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

from sr2.integrations.litellm import LiteLLMCallable

if TYPE_CHECKING:
    from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent

from sr2_spectre.config import ModelConfig
from sr2_spectre.run_log import SessionLog, summarize_request

logger = logging.getLogger(__name__)


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
        return await self._inner.complete(request)

    async def stream(self, request: "CompletionRequest") -> AsyncIterator["StreamEvent"]:
        # Bound once, up front: a retarget part-way through must not splice two
        # endpoints into a single response.
        inner = self._inner
        async for event in inner.stream(request):
            yield event


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


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


def _error_data(exc: BaseException, started: float) -> dict[str, Any]:
    return {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "duration_ms": _elapsed_ms(started),
    }
