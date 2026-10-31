"""Zone-1: Deterministic scaffold compiler.

Assembles a positive prompt purely from config fragments — no LLM involved.
Order is deterministic:

    1. Checkpoint quality boilerplate
    2. Frame tags
    3. Content tags
    4. Scenario extra
    5. LoRA trigger tokens
    6. User intent (appended last)

Negative prompt comes directly from the checkpoint's negative field.

FR10: Logs intent → compiled positive + negative + chosen model/scenario
on every call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sr2_spectre.tools.image_scenarios import ResolvedScenario

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledPrompt:
    """Result of deterministic scaffold compilation.

    Attributes:
        positive: Fully assembled positive prompt (scaffold + intent).
        negative: Negative prompt from checkpoint config.
        scenario_name: Resolved scenario identifier.
        model_file: Checkpoint file path on ComfyUI.
    """

    positive: str
    negative: str
    scenario_name: str
    model_file: str


def compile_scaffold(
    scenario: ResolvedScenario,
    intent: str,
) -> CompiledPrompt:
    """Compile a deterministic positive prompt from a resolved scenario.

    Assembles parts in fixed order (quality → frame → content → extra →
    LoRA triggers → intent). No LLM translation — that's Zone-2 (FR5).

    Args:
        scenario: Fully resolved scenario with all fragments dereferenced.
        intent: Natural-language intent from the agent (appended last).

    Returns:
        CompiledPrompt with positive, negative, and metadata.

    Logs:
        INFO: intent → compiled positive + negative + model/scenario.
    """
    parts: list[str] = []

    # 1. Checkpoint quality boilerplate
    if scenario.model.quality:
        parts.append(scenario.model.quality)

    # 2. Frame tags
    if scenario.frame.tags:
        parts.append(scenario.frame.tags)

    # 3. Content tags
    if scenario.content.tags:
        parts.append(scenario.content.tags)

    # 4. Scenario extra
    if scenario.extra:
        parts.append(scenario.extra)

    # 5. LoRA trigger tokens (in declaration order)
    triggers = [lora.trigger for lora in scenario.loras if lora.trigger]
    if triggers:
        parts.append(", ".join(triggers))

    # 6. User intent (always last)
    if intent:
        parts.append(intent)

    positive = ", ".join(parts)
    negative = scenario.model.negative

    # FR10: Log intent → compiled + chosen model/scenario
    logger.info(
        "Scaffold compiled — scenario=%s, model=%s (%s):\n"
        "  intent: %s\n"
        "  positive: %s\n"
        "  negative: %s",
        scenario.name,
        scenario.model.file,
        scenario.model.dialect,
        intent[:120],
        positive[:200],
        negative[:120],
    )

    return CompiledPrompt(
        positive=positive,
        negative=negative,
        scenario_name=scenario.name,
        model_file=scenario.model.file,
    )
