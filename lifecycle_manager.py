"""
Lifecycle Manager — Rules engine for memory freshness & archival.

Determines when memories become stale, should be archived, or remain evergreen.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from config import (
    ARCHIVE_INACTIVITY_DAYS,
    EVERGREEN_IMPORTANCE_THRESHOLD,
    EVERGREEN_TAG,
    STALENESS_THRESHOLDS,
    MemoryTypeName,
)
from models import Memory, MemoryStatus


class LifecycleManager:
    """
    Applies lifecycle rules to memories stored in a MemoryStore.

    Rules
    -----
    1. Staleness  : memory older than its type threshold → STALE
    2. Evergreen  : importance ≥ 8 or tagged "evergreen" → never stale
    3. Archive    : stale AND no access for ARCHIVE_INACTIVITY_DAYS → ARCHIVED
    4. Refresh    : accessing a memory bumps updated_at and access_count
    5. Invalidate : external trigger can mark a memory as ARCHIVED with a reason
    """

    def __init__(
        self,
        staleness_thresholds: Dict = None,
        archive_days: int = ARCHIVE_INACTIVITY_DAYS,
        evergreen_threshold: int = EVERGREEN_IMPORTANCE_THRESHOLD,
        evergreen_tag: str = EVERGREEN_TAG,
    ):
        self.staleness_thresholds = staleness_thresholds or STALENESS_THRESHOLDS
        self.archive_days = archive_days
        self.evergreen_threshold = evergreen_threshold
        self.evergreen_tag = evergreen_tag

    # ───────────────────── Individual checks ────────────────────────────────

    def is_evergreen(self, memory: Memory) -> bool:
        """Check if a memory is evergreen (exempt from staleness)."""
        if memory.importance >= self.evergreen_threshold:
            return True
        if self.evergreen_tag in memory.tags:
            return True
        return False

    def check_staleness(self, memory: Memory) -> bool:
        """
        Return True if the memory has exceeded its staleness threshold.
        Evergreen memories always return False.
        """
        if self.is_evergreen(memory):
            return False

        # Map MemoryType value to config key
        type_key = MemoryTypeName(memory.memory_type.value)
        threshold_days = self.staleness_thresholds.get(type_key, 365)

        return memory.days_since_update() > threshold_days

    def should_archive(self, memory: Memory) -> bool:
        """
        A memory should be archived if it is stale AND has not been
        accessed for ARCHIVE_INACTIVITY_DAYS.
        """
        if memory.status != MemoryStatus.STALE:
            return False
        return memory.days_since_update() > self.archive_days

    # ───────────────────── Batch lifecycle sweep ────────────────────────────

    def run_lifecycle_sweep(self, memories: List[Memory]) -> Dict[str, int]:
        """
        Iterate over all memories and apply lifecycle rules.

        Returns a summary dict: {"marked_stale": N, "archived": N, "unchanged": N}
        """
        stats = {"marked_stale": 0, "archived": 0, "unchanged": 0}

        for memory in memories:
            if memory.status == MemoryStatus.ARCHIVED:
                continue  # already done

            if memory.status == MemoryStatus.STALE:
                if self.should_archive(memory):
                    memory.status = MemoryStatus.ARCHIVED
                    memory.metadata["_archived_reason"] = "Inactivity after staleness"
                    stats["archived"] += 1
                else:
                    stats["unchanged"] += 1

            elif memory.status == MemoryStatus.ACTIVE:
                if self.check_staleness(memory):
                    memory.status = MemoryStatus.STALE
                    memory.metadata["_stale_since"] = datetime.now().isoformat()
                    stats["marked_stale"] += 1
                else:
                    stats["unchanged"] += 1

        return stats

    # ───────────────────── Manual actions ───────────────────────────────────

    def refresh_memory(self, memory: Memory) -> Memory:
        """
        Mark a memory as recently accessed — bumps updated_at
        and increments access_count.
        """
        memory.access_count += 1
        memory.updated_at = datetime.now()
        # If it was stale, accessing it can bring it back to active
        if memory.status == MemoryStatus.STALE:
            memory.status = MemoryStatus.ACTIVE
            memory.metadata.pop("_stale_since", None)
        return memory

    def invalidate_memory(self, memory: Memory, reason: str = "") -> Memory:
        """
        Manually mark a memory as archived (e.g. contract renegotiated,
        supplier replaced, etc.).
        """
        memory.status = MemoryStatus.ARCHIVED
        memory.metadata["_invalidated_reason"] = reason or "Manually invalidated"
        memory.updated_at = datetime.now()
        return memory
