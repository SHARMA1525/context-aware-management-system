from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

from config import DECAY_RATE, MAX_CONTEXT_ITEMS, MIN_RELEVANCE_SCORE, RELEVANCE_WEIGHTS
from models import Memory, MemoryType, QueryContext, ScoredMemory


class ContextManager:
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


    def semantic_score(self, cosine_sim: float) -> float:
        return max(0.0, min(1.0, cosine_sim))

    def temporal_score(self, memory: Memory) -> float:
        days = memory.days_since_creation()
        return math.exp(-self.decay_rate * days)

    def relational_score(
        self,
        memory: Memory,
        query_entity_id: Optional[str] = None,
        query_tags: Optional[List[str]] = None,
    ) -> float:
        if query_entity_id and memory.entity_id == query_entity_id:
            return 1.0

        if query_tags:
            overlap = set(memory.tags) & set(query_tags)
            if overlap:
                return 0.5

        return 0.0


    def score_memory(
        self,
        memory: Memory,
        cosine_sim: float,
        query_entity_id: Optional[str] = None,
        query_tags: Optional[List[str]] = None,
    ) -> ScoredMemory:
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


    def build_context(
        self,
        candidates: List[tuple],
        query_context: QueryContext,
    ) -> List[ScoredMemory]:
        scored: List[ScoredMemory] = []

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

        scored.sort(key=lambda s: s.final_score, reverse=True)

        top_k = min(query_context.top_k, self.max_items)
        scored = scored[:top_k]

        scored = self.resolve_conflicts(scored)

        return scored


    def resolve_conflicts(self, scored_memories: List[ScoredMemory]) -> List[ScoredMemory]:
        if len(scored_memories) <= 1:
            return scored_memories

        groups: Dict[str, List[ScoredMemory]] = defaultdict(list)
        for sm in scored_memories:
            key = sm.memory.entity_id
            if sm.memory.tags:
                key += "|" + sm.memory.tags[0]
            groups[key].append(sm)

        result: List[ScoredMemory] = []
        for group in groups.values():
            if len(group) > 1:
                group.sort(key=lambda s: s.memory.created_at, reverse=True)
                group[0].final_score = min(1.0, group[0].final_score * 1.1)
                for older in group[1:]:
                    older.memory.metadata["_conflict_note"] = (
                        "Newer information available — this memory may be outdated."
                    )
            result.extend(group)

        result.sort(key=lambda s: s.final_score, reverse=True)
        return result

    def summarize_context(self, scored_memories: List[ScoredMemory]) -> Dict:
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
        return {k: v for k, v in summary.items() if v}
