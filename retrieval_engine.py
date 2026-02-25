"""
Retrieval Engine — Top-level orchestrator for context retrieval.

Pipeline:  query → MemoryStore (candidates)
                 → LifecycleManager (filter stale / refresh)
                 → ContextManager (score & rank)
                 → formatted output with explanations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from context_manager import ContextManager
from lifecycle_manager import LifecycleManager
from memory_store import MemoryStore
from models import QueryContext, ScoredMemory


@dataclass
class RetrievalResult:
    """The final output of a context retrieval query."""
    query: str
    entity_id: Optional[str]
    scored_memories: List[ScoredMemory] = field(default_factory=list)
    context_summary: Dict = field(default_factory=dict)
    total_candidates: int = 0
    total_returned: int = 0

    def display(self) -> str:
        """Pretty-print the retrieval result for CLI demo."""
        lines = [
            "=" * 70,
            f"  QUERY       : {self.query}",
            f"  ENTITY      : {self.entity_id or '(all)'}",
            f"  CANDIDATES  : {self.total_candidates}",
            f"  RETURNED    : {self.total_returned}",
            "=" * 70,
        ]

        for i, sm in enumerate(self.scored_memories, 1):
            lines.append(f"\n--- Memory #{i} (score: {sm.final_score:.3f}) ---")
            lines.append(f"  Type       : {sm.memory.memory_type.value}")
            lines.append(f"  Status     : {sm.memory.status.value}")
            lines.append(f"  Content    : {sm.memory.content}")
            lines.append(f"  Entity     : {sm.memory.entity_id} ({sm.memory.entity_type})")
            lines.append(f"  Age        : {sm.memory.days_since_creation():.0f} days")
            lines.append(f"  Importance : {sm.memory.importance}/10")
            lines.append(
                f"  Scores     : semantic={sm.semantic_score:.3f}  "
                f"temporal={sm.temporal_score:.3f}  "
                f"relational={sm.relational_score:.3f}"
            )
            if sm.memory.tags:
                lines.append(f"  Tags       : {', '.join(sm.memory.tags)}")
            if "_conflict_note" in sm.memory.metadata:
                lines.append(f"  ⚠ CONFLICT : {sm.memory.metadata['_conflict_note']}")

        # Context summary by type
        if self.context_summary:
            lines.append("\n" + "=" * 70)
            lines.append("  CONTEXT SUMMARY (grouped by memory type)")
            lines.append("=" * 70)
            for mtype, entries in self.context_summary.items():
                lines.append(f"\n  [{mtype.upper()}]")
                for e in entries:
                    age = f"{e['age_days']}d ago"
                    lines.append(f"    • (score={e['score']:.3f}, {age}) {e['content'][:90]}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


class RetrievalEngine:
    """
    Orchestrates the full retrieval pipeline:
      1. Semantic search in MemoryStore → candidates
      2. Lifecycle check (skip archived, note stale)
      3. Context scoring & ranking
      4. Explanation generation
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        context_manager: ContextManager,
        lifecycle_manager: LifecycleManager,
    ):
        self.store = memory_store
        self.context = context_manager
        self.lifecycle = lifecycle_manager

    def query(
        self,
        text: str,
        entity_id: Optional[str] = None,
        top_k: int = 10,
        include_stale: bool = False,
    ) -> RetrievalResult:
        """
        Run the full retrieval pipeline for a natural-language query.

        Parameters
        ----------
        text          : What the agent wants to know (e.g. "quality issues with supplier")
        entity_id     : Optional entity to scope results to
        top_k         : Max results
        include_stale : Whether to include stale memories
        """
        # Build query context
        qctx = QueryContext(
            query_text=text,
            entity_id=entity_id,
            top_k=top_k,
            include_stale=include_stale,
        )

        # Step 1: Semantic search for candidates
        candidates = self.store.search_similar(
            query_text=text,
            top_k=top_k * 3,  # over-fetch so scoring can filter further
            entity_id=entity_id,
            include_stale=include_stale,
        )

        total_candidates = len(candidates)

        # Step 2: Refresh accessed memories (bump access_count)
        for memory, _ in candidates:
            self.lifecycle.refresh_memory(memory)

        # Step 3: Score, rank, and cap
        scored = self.context.build_context(candidates, qctx)

        # Step 4: Summarise
        summary = self.context.summarize_context(scored)

        return RetrievalResult(
            query=text,
            entity_id=entity_id,
            scored_memories=scored,
            context_summary=summary,
            total_candidates=total_candidates,
            total_returned=len(scored),
        )

    def explain(self, scored_memory: ScoredMemory) -> str:
        """
        Generate a human-readable explanation of why a particular memory
        was retrieved and how it scored.
        """
        m = scored_memory.memory
        lines = [
            f"This memory was retrieved because:",
            f"  • Semantic similarity to your query: {scored_memory.semantic_score:.1%}",
            f"  • Recency (temporal) score: {scored_memory.temporal_score:.1%} "
            f"({m.days_since_creation():.0f} days old)",
            f"  • Entity/tag relationship score: {scored_memory.relational_score:.1%}",
            f"  • Combined relevance: {scored_memory.final_score:.1%}",
        ]
        if m.importance >= 8:
            lines.append(f"  • Marked as HIGH IMPORTANCE ({m.importance}/10)")
        if "_conflict_note" in m.metadata:
            lines.append(f"  ⚠ Note: {m.metadata['_conflict_note']}")
        return "\n".join(lines)
