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

    def is_evergreen(self, memory: Memory) -> bool:
        if memory.importance >= self.evergreen_threshold:
            return True
        if self.evergreen_tag in memory.tags:
            return True
        return False

    def check_staleness(self, memory: Memory) -> bool:
        if self.is_evergreen(memory):
            return False
        type_key = MemoryTypeName(memory.memory_type.value)
        threshold_days = self.staleness_thresholds.get(type_key, 365)

        return memory.days_since_update() > threshold_days

    def should_archive(self, memory: Memory) -> bool:
        if memory.status != MemoryStatus.STALE:
            return False
        return memory.days_since_update() > self.archive_days

    def run_lifecycle_sweep(self, memories: List[Memory]) -> Dict[str, int]:
        stats = {"marked_stale": 0, "archived": 0, "unchanged": 0}

        for memory in memories:
            if memory.status == MemoryStatus.ARCHIVED:
                continue 

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

    def refresh_memory(self, memory: Memory) -> Memory:
        memory.access_count += 1
        memory.updated_at = datetime.now()
        if memory.status == MemoryStatus.STALE:
            memory.status = MemoryStatus.ACTIVE
            memory.metadata.pop("_stale_since", None)
        return memory

    def invalidate_memory(self, memory: Memory, reason: str = "") -> Memory:
        memory.status = MemoryStatus.ARCHIVED
        memory.metadata["_invalidated_reason"] = reason or "Manually invalidated"
        memory.updated_at = datetime.now()
        return memory
