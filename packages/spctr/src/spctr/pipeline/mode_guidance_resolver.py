"""ModeGuidanceResolver: injects headless/interactive behavior guidance.

Reads the execution mode from deps.run_context_provider() and injects the
matching guidance block into the system prompt.  Declared in base.yaml so
all agents inherit it — mode is per-interface, not per-persona.

Hard rule: nothing enters context except through a resolver.  The interface
DECIDES the mode (already does); this resolver READS it and injects guidance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.token_counting import CHARS_PER_TOKEN
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_SUBSCRIPTION = EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)

# ---------------------------------------------------------------------------
# Guidance blocks — defined once, consumed per mode
# ---------------------------------------------------------------------------

_HEADLESS_GUIDANCE = """## Mode: Headless

You are running in a headless environment. The user cannot respond to
clarifying questions — this channel has no input capability.

- **Be proactive and self-resolving.** Make reasonable assumptions and
  act on them rather than asking for clarification.
- **Do not ask clarifying questions.** The user cannot answer.
- **Provide complete, actionable output.** Include all context, code, and
  explanations in your response so the user has everything they need.
- **If critical information is missing**, state your assumptions and
  proceed with the most reasonable interpretation."""

_INTERACTIVE_GUIDANCE = """## Mode: Interactive

You are running in an interactive environment. The user can respond to
clarifying questions and provide additional context.

- **Ask clarifying questions when needed.** It is better to confirm than
  to assume incorrectly.
- **Be conversational.** Engage naturally — the user is present and can
  follow up.
- **You may break complex tasks into steps** and confirm before proceeding."""

# Fallback when run_context_provider is unavailable (regression-safe).
_FALLBACK_GUIDANCE = """## Mode: Unknown

The execution mode could not be determined. Behave conversationally
and ask clarifying questions if you need information from the user."""


# ---------------------------------------------------------------------------
# ModeGuidanceResolver
# ---------------------------------------------------------------------------


class ModeGuidanceResolver:
    """Injects mode-appropriate behavior guidance into the system prompt.

    Reads ``deps.run_context_provider()`` at resolve time to determine whether
    the run is headless or interactive, then returns the corresponding
    guidance block.

    When ``run_context_provider`` is not available (core without harness
    wiring), falls back to a neutral guidance message.  This is safe — the
    agent simply doesn't get mode-specific tuning.

    Config fields
    -------------
    None. This resolver requires no configuration beyond declaration.
    """

    name: str = "mode_guidance"

    def __init__(
        self,
        config: ResolverConfig,
        provider: Callable[[], dict[str, str] | None] | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, [_DEFAULT_SUBSCRIPTION]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, config: ResolverConfig, deps: Dependencies) -> "ModeGuidanceResolver":
        return cls(config, provider=deps.run_context_provider)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        """Resolve the mode guidance block for this turn.

        Reads ``self._provider()`` to get the current mode,
        then returns the matching guidance text.
        """
        self.execution_count += 1

        guidance = self._resolve_guidance()
        tokens = len(guidance) // CHARS_PER_TOKEN

        return ResolvedContent(
            resolver_name=self.name,
            source_layer="mode_guidance",
            content=[TextBlock(text=guidance)],
            token_count=tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_guidance(self) -> str:
        """Determine the guidance block from the run-context provider."""
        if self._provider is None:
            return _FALLBACK_GUIDANCE

        ctx = self._provider()
        if not ctx or not isinstance(ctx, dict):
            return _FALLBACK_GUIDANCE

        mode = ctx.get("mode")
        if mode == "headless":
            return _HEADLESS_GUIDANCE
        if mode == "interactive":
            return _INTERACTIVE_GUIDANCE

        # Unknown mode value — log and fall through.
        logger.warning(
            "ModeGuidanceResolver: unknown mode %r — using fallback guidance",
            mode,
        )
        return _FALLBACK_GUIDANCE
