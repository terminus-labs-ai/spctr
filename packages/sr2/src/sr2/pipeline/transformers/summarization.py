"""SummarizationTransformer: compresses older content via LLM summarization."""

from __future__ import annotations

import json
import logging

from sr2.config.models import ConfigError, TransformerConfig
from sr2.models import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import TransformationResult
from sr2.pipeline.protocols import TokenCounter
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions
from sr2.protocols.llm import CompletionRequest, LLMCallable

logger = logging.getLogger(__name__)

_DEFAULT_SUMMARIZATION_PROMPT = "Summarize the following conversation turns concisely."
_DEFAULT_SUMMARY_HEADER = "[Summary of {count} earlier messages]"
_DEFAULT_RETRY_GROWTH_TOKENS = 10000

_SOURCES_OMITTED_REASON = (
    "summarization: transform() receives content blocks without entry identity, "
    "so it declares no per-block source ids; the Layer resolves the summarized "
    "blocks to their real stored entry ids at flush time (FR5)."
)


class SummarizationTransformer:
    """Summarizes older content blocks via an injected LLM callable.

    Supports two keep strategies:
    - keep_last_n: keep the N most-recent blocks, summarize the rest.
    - keep_within_tokens: keep the most-recent blocks that fit within a
      token budget (newest-to-oldest), summarize everything else.
    """

    name: str = "summarization"

    def __init__(
        self,
        config: TransformerConfig,
        llm: LLMCallable,
        token_counter: TokenCounter | None = None,
        session_id: str = "",
    ) -> None:
        self._config = config
        self._llm = llm
        self._token_counter: TokenCounter = token_counter or CharacterTokenCounter()
        self._session_id = session_id
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        # Counted content size at the last failed Message summary (FR 11).
        self._failed_at_tokens: int | None = None

        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, []
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, config: TransformerConfig, deps: Dependencies) -> "SummarizationTransformer":
        """Construct from a TransformerConfig and a Dependencies container."""
        if deps.llm is None:
            raise ConfigError("transformer 'summarize' requires an LLM but none was provided")
        cfg = config.config or {}
        key = cfg.get("model") if cfg.get("model") else None

        if key is not None:
            # Explicit model key: must be present — no silent fallback.
            if key not in deps.llm:
                raise ConfigError(
                    f"transformer 'summarization' configured model={key!r} "
                    f"but that key is absent from deps.llm (available: {list(deps.llm.keys())!r})"
                )
            llm = deps.llm[key]
        else:
            # No model key in config: use "default" if present, else raise.
            if "default" not in deps.llm:
                raise ConfigError(
                    "transformer 'summarization' has no 'model' config key and "
                    f"deps.llm has no 'default' entry (available: {list(deps.llm.keys())!r})"
                )
            llm = deps.llm["default"]
        return cls(config, llm, session_id=deps.session_id)

    # ------------------------------------------------------------------
    # Transformer protocol
    # ------------------------------------------------------------------

    async def transform(
        self, content: list[ContentBlock], events: list[Event]
    ) -> TransformationResult:
        cfg = self._config.config
        strategy = cfg.get("keep_strategy", "keep_last_n")

        if strategy == "keep_within_tokens" and any(isinstance(c, Message) for c in content):
            return await self._transform_messages(content, cfg)

        if strategy == "keep_within_tokens":
            to_summarize, to_keep = self._split_by_tokens(content, cfg)
        else:
            # keep_last_n (default)
            to_summarize, to_keep = self._split_by_count(content, cfg)

        # No-op: nothing to summarize
        if not to_summarize:
            return TransformationResult(
                transformer_name=self.name,
                source_layer="summarization",
                content=None,
                events=[],
                entries=[],
            )

        # Call LLM to summarize
        summary_block = await self._summarize(to_summarize)

        # Provenance for the summarized blocks. `transform(content, events)`
        # sees content blocks, never entry ids, and the blocks it summarizes
        # were themselves stored as provenance entries at ingestion (FR1 has
        # no raw-content/no-store bypass). Re-minting ids here — ULID or
        # content hash — would only ever duplicate what the ledger already
        # holds, and a fresh ULID would be exactly the synthetic dangling
        # source id FR5 forbids. So this transformer declares no per-block
        # source lineage: the omission is recorded in the result's provenance
        # declaration, and the Layer synthesizes the summary block against
        # the real stored entry ids it holds at flush time (FR5).
        declared_sources = await self._resolve_real_source_ids(to_summarize)
        declaration: dict = {
            "sources_omitted": not declared_sources,
            "sources_omitted_reason": _SOURCES_OMITTED_REASON,
        }
        if declared_sources:
            declaration["sources"] = list(declared_sources)

        # Emit event
        event = Event(
            name="summarization_complete",
            phase=EventPhase.COMPLETED,
            source_layer="summarization",
            data=[summary_block],
        )

        new_content: list[ContentBlock] = [summary_block] + list(to_keep)

        return TransformationResult(
            transformer_name=self.name,
            source_layer="summarization",
            content=new_content,
            events=[event],
            entries=[],
            provenance=[declaration],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_by_count(
        self, content: list[ContentBlock], cfg: dict
    ) -> tuple[list[ContentBlock], list[ContentBlock]]:
        """Split content into (to_summarize, to_keep) by keep_last_n."""
        keep_n = cfg.get("keep_last_n", 3)
        if len(content) <= keep_n:
            return [], content
        return list(content[:-keep_n]), list(content[-keep_n:])

    def _split_by_tokens(
        self, content: list[ContentBlock], cfg: dict
    ) -> tuple[list[ContentBlock], list[ContentBlock]]:
        """Split content into (to_summarize, to_keep) by keep_within_tokens.

        Walk newest-to-oldest, accumulate tokens, keep items while cumulative
        count <= budget. Everything not kept is summarized.
        """
        budget = cfg.get("keep_tokens", 0)
        kept: list[ContentBlock] = []
        cumulative = 0
        for block in reversed(content):
            block_tokens = self._token_counter.count([block])
            if cumulative + block_tokens <= budget:
                kept.append(block)
                cumulative += block_tokens
            else:
                break

        # kept is newest-to-oldest; reverse to restore original order
        kept = list(reversed(kept))
        n_kept = len(kept)
        to_summarize = list(content[: len(content) - n_kept])
        return to_summarize, kept

    def _block_to_text(self, block: ContentBlock) -> str:
        """Extract a text representation from any ContentBlock type."""
        if isinstance(block, TextBlock):
            return block.text
        # For non-text blocks, emit a typed placeholder rather than str(block).
        block_type = type(block).__name__
        name_attr = getattr(block, "name", None)
        if name_attr is not None:
            return f"[{block_type}: {name_attr}]"
        return f"[{block_type}]"

    async def _summarize(self, blocks: list[ContentBlock]) -> TextBlock:
        """Build a CompletionRequest and call the LLM, returning the summary TextBlock."""
        return await self._complete("\n".join(self._block_to_text(b) for b in blocks))

    async def _complete(self, turns_text: str) -> TextBlock:
        cfg = self._config.config or {}
        prompt = cfg.get("summarization_prompt") or _DEFAULT_SUMMARIZATION_PROMPT
        request = CompletionRequest(
            system=[TextBlock(text=prompt)],
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text=turns_text)],
                )
            ],
        )
        response = await self._llm.complete(request)
        # Extract first TextBlock from response
        for block in response.content:
            if isinstance(block, TextBlock):
                return block
        return TextBlock(text="")

    # ------------------------------------------------------------------
    # Message content (FR 9-11)
    # ------------------------------------------------------------------

    def _noop(self) -> TransformationResult:
        return TransformationResult(
            transformer_name=self.name,
            source_layer="summarization",
            content=None,
            events=[],
            entries=[],
        )

    def _fail(self, total_tokens: int, reason: str) -> TransformationResult:
        logger.warning("summarization: %s; content left unchanged", reason)
        self._failed_at_tokens = total_tokens
        return self._noop()

    @staticmethod
    def _is_user_text(item: ContentBlock | Message) -> bool:
        return (
            isinstance(item, Message)
            and item.role == "user"
            and any(isinstance(b, TextBlock) and b.text.strip() for b in item.content)
        )

    @staticmethod
    def _has_tool_result(item: ContentBlock | Message) -> bool:
        return isinstance(item, Message) and any(
            isinstance(b, ToolResultBlock) for b in item.content
        )

    def _keep_from(self, content: list, start: int, budget: int) -> int:
        """Index where the kept window begins: newest items fitting ``budget``,
        extended back so a tool result never starts the window (FR 9)."""
        k = len(content)
        cumulative = 0
        while k > start:
            tokens = self._token_counter.count([content[k - 1]])
            if cumulative + tokens > budget:
                break
            cumulative += tokens
            k -= 1
        while start < k < len(content) and self._has_tool_result(content[k]):
            k -= 1
        return k

    def _render_block(self, block: ContentBlock) -> str:
        if isinstance(block, (TextBlock, ThinkingBlock)):
            return block.text
        if isinstance(block, ToolUseBlock):
            return f"[tool call {block.name} {json.dumps(block.input)}]"
        if isinstance(block, ToolResultBlock):
            text = (
                block.content
                if isinstance(block.content, str)
                else "\n".join(tb.text for tb in block.content)
            )
            return f"[tool result] {text}"
        return self._block_to_text(block)

    def _render(self, item: ContentBlock | Message) -> str:
        if isinstance(item, Message):
            return f"{item.role}: " + "\n".join(self._render_block(b) for b in item.content)
        return self._render_block(item)

    async def _transform_messages(
        self, content: list, cfg: dict
    ) -> TransformationResult:
        """Replace the span between the protected prompt and the kept window
        with one headed summary message (FR 9-11)."""
        total = self._token_counter.count(content)
        growth = cfg.get("retry_growth_tokens", _DEFAULT_RETRY_GROWTH_TOKENS)
        if self._failed_at_tokens is not None and total - self._failed_at_tokens < growth:
            return self._noop()

        prompt_idx = next((i for i, m in enumerate(content) if self._is_user_text(m)), None)
        start = 0 if prompt_idx is None else prompt_idx + 1
        keep_from = self._keep_from(content, start, cfg.get("keep_tokens", 0))
        span = [m for i, m in enumerate(content[:keep_from]) if i != prompt_idx]
        if not span:
            return self._noop()

        try:
            summary = await self._complete("\n\n".join(self._render(m) for m in span))
        except Exception as exc:  # noqa: BLE001 - FR 11: never escape the pipeline
            return self._fail(total, f"summary call failed: {exc}")
        if not summary.text.strip():
            return self._fail(total, "summary call returned empty text")

        header = cfg.get("summary_header") or _DEFAULT_SUMMARY_HEADER
        summary_msg = Message(
            role="user",
            content=[TextBlock(text=header.format(count=len(span)) + "\n\n" + summary.text)],
        )
        span_tokens = self._token_counter.count(span)
        summary_tokens = self._token_counter.count([summary_msg])
        if summary_tokens >= span_tokens:
            return self._fail(
                total,
                f"summary ({summary_tokens} tokens) does not shrink span ({span_tokens} tokens)",
            )

        self._failed_at_tokens = None
        prompt = [] if prompt_idx is None else [content[prompt_idx]]
        return TransformationResult(
            transformer_name=self.name,
            source_layer="summarization",
            content=prompt + [summary_msg] + list(content[keep_from:]),
            events=[
                Event(
                    name="summarization_complete",
                    phase=EventPhase.COMPLETED,
                    source_layer="summarization",
                    data=[summary_msg.content[0]],
                )
            ],
            entries=[],
            provenance=[
                {"sources_omitted": True, "sources_omitted_reason": _SOURCES_OMITTED_REASON}
            ],
        )

    async def _resolve_real_source_ids(
        self, blocks: list[ContentBlock]
    ) -> tuple[str, ...]:
        """Real stored entry ids for the summarized blocks, if any is resolvable.

        Only ids already present in the provenance store are declared — this
        transformer never invents an id (FR5). When no block carries a stored
        id this returns empty and the Layer resolves the summarized blocks
        from the turn's write set at flush time.
        """
        store = getattr(self, "_provenance_store", None)
        if store is None:
            # No store handle: nothing can be confirmed to resolve.
            return ()
        for block in blocks:
            meta = getattr(block, "meta", None)
            if not isinstance(meta, dict):
                continue
            declared = meta.get("provenance")
            if not isinstance(declared, dict):
                continue
            candidates = declared.get("source_ids")
            if isinstance(candidates, str):
                candidates = (candidates,)
            if not isinstance(candidates, (list, tuple)):
                continue
            for candidate in candidates:
                if not isinstance(candidate, str) or not candidate:
                    continue
                if await store.get(candidate) is not None:
                    return (candidate,)
        return ()
