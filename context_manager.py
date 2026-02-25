"""
Context Manager — Relevance scoring and context hierarchy.

Combines three proximity dimensions to rank memories for a given query:
  1. Semantic   – cosine similarity between query and memory embeddings
  2. Temporal   – exponential decay based on memory age
  3. Relational – entity and tag overlap
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

from config import DECAY_RATE, MAX_CONTEXT_ITEMS, MIN_RELEVANCE_SCORE, RELEVANCE_WEIGHTS
from models import Memory, MemoryType, QueryContext, ScoredMemory


class ContextManager:
    """
    Scores and ranks memories by relevance, resolves conflicts,
    and produces structured context summaries.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        decay_rate: float = DECAY_RATE,
        max_items: int = MAX_CONTEXT_ITEMS,
        min_score: float = MIN_RELEVANCE_SCORE,
    ):
        self.weights = weights or RELEVANCE_WEIGHTS
        self.decay_rate = decay_rate
        self.max_items = max_items
        self.min_score = min_score

    # ───────────────── Individual score components ──────────────────────────

    def semantic_score(self, cosine_sim: float) -> float:
        """
        Normalise cosine similarity to [0, 1].
        Sentence-transformer cosine similarities are typically in [-1, 1],
        but after L2 normalisation they are mostly in [0, 1].
        """
        return max(0.0, min(1.0, cosine_sim))

    def temporal_score(self, memory: Memory) -> float:
        """
        Exponential decay: recent memories score close to 1,
        old memories approach 0.

        Formula:  score = exp(-decay_rate × days_since_creation)
        """
        days = memory.days_since_creation()
        return math.exp(-self.decay_rate * days)

    def relational_score(
        self,
        memory: Memory,
        query_entity_id: Optional[str] = None,
        query_tags: Optional[List[str]] = None,
    ) -> float:
        """
        Measures how closely the memory is linked to the queried entity.

        Returns
        -------
        1.0  – memory belongs to the exact queried entity
        0.5  – memory shares at least one tag with the query
        0.0  – no relationship found
        """
        if query_entity_id and memory.entity_id == query_entity_id:
            return 1.0

        if query_tags:
            overlap = set(memory.tags) & set(query_tags)
            if overlap:
                return 0.5

        return 0.0

    # ───────────────── Combined scoring ─────────────────────────────────────

    def score_memory(
        self,
        memory: Memory,
        cosine_sim: float,
        query_entity_id: Optional[str] = None,
        query_tags: Optional[List[str]] = None,
    ) -> ScoredMemory:
        """
        Compute the final weighted score for a single memory.

        final = w_s × semantic + w_t × temporal + w_r × relational
        """
        sem = self.semantic_score(cosine_sim)
        tmp = self.temporal_score(memory)
        rel = self.relational_score(memory, query_entity_id, query_tags)

        final = (
            self.weights["semantic"] * sem
            + self.weights["temporal"] * tmp
            + self.weights["relational"] * rel
        )

        return ScoredMemory(
            memory=memory,
            semantic_score=round(sem, 4),
            temporal_score=round(tmp, 4),
            relational_score=round(rel, 4),
            final_score=round(final, 4),
        )

    # ───────────────── Build context ────────────────────────────────────────

    def build_context(
        self,
        candidates: List[tuple],
        query_context: QueryContext,
    ) -> List[ScoredMemory]:
        """
        Score a list of (Memory, cosine_sim) candidate pairs, filter, sort,
        and return the top-k most relevant scored memories.

        Parameters
        ----------
        candidates    : output of MemoryStore.search_similar()
        query_context : the original query parameters
        """
        scored: List[ScoredMemory] = []

        # Extract tags from the query for relational scoring
        query_tags = query_context.query_text.lower().split()

        for memory, cosine_sim in candidates:
            sm = self.score_memory(
                memory=memory,
                cosine_sim=cosine_sim,
                query_entity_id=query_context.entity_id,
                query_tags=query_tags,
            )
            if sm.final_score >= self.min_score:
                scored.append(sm)

        # Sort by final score (descending)
        scored.sort(key=lambda s: s.final_score, reverse=True)

        # Apply top-k cap
        top_k = min(query_context.top_k, self.max_items)
        scored = scored[:top_k]

        # Resolve conflicts among top results
        scored = self.resolve_conflicts(scored)

        return scored

    # ───────────────── Conflict resolution ──────────────────────────────────

    def resolve_conflicts(self, scored_memories: List[ScoredMemory]) -> List[ScoredMemory]:
        """
        When two memories about the same entity + topic contradict each other,
        prefer the more recent one (with a note about the conflict).

        Heuristic: memories with the same entity_id AND at least one shared tag
        are considered potentially conflicting.  Among those, the newer memory
        gets a small boost and the older one is annotated.
        """
        if len(scored_memories) <= 1:
            return scored_memories

        # Group by (entity_id, first shared tag) — simple conflict key
        groups: Dict[str, List[ScoredMemory]] = defaultdict(list)
        for sm in scored_memories:
            key = sm.memory.entity_id
            if sm.memory.tags:
                key += "|" + sm.memory.tags[0]
            groups[key].append(sm)

        result: List[ScoredMemory] = []
        for group in groups.values():
            if len(group) > 1:
                # Sort within group by created_at (newest first)
                group.sort(key=lambda s: s.memory.created_at, reverse=True)
                # Boost the newest
                group[0].final_score = min(1.0, group[0].final_score * 1.1)
                # Keep all but mark older ones
                for older in group[1:]:
                    older.memory.metadata["_conflict_note"] = (
                        "Newer information available — this memory may be outdated."
                    )
            result.extend(group)

        # Re-sort after conflict adjustments
        result.sort(key=lambda s: s.final_score, reverse=True)
        return result

    # ───────────────── Context summary ──────────────────────────────────────

    def summarize_context(self, scored_memories: List[ScoredMemory]) -> Dict:
        """
        Group scored memories by type and return a structured summary dict.

        Useful for presenting context to downstream decision logic.
        """
        summary: Dict[str, list] = {
            "immediate": [],
            "historical": [],
            "temporal": [],
            "experiential": [],
        }

        for sm in scored_memories:
            entry = {
                "content": sm.memory.content,
                "score": sm.final_score,
                "age_days": round(sm.memory.days_since_creation()),
                "importance": sm.memory.importance,
                "entity": sm.memory.entity_id,
            }
            if "_conflict_note" in sm.memory.metadata:
                entry["conflict_note"] = sm.memory.metadata["_conflict_note"]
            summary[sm.memory.memory_type.value].append(entry)

        # Remove empty categories
        return {k: v for k, v in summary.items() if v}
