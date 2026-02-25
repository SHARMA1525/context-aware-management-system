"""
Data models for the Context & Memory Management System.

Defines the core data structures:
  - MemoryType   : Categorizes business context (immediate / historical / temporal / experiential)
  - MemoryStatus : Lifecycle state (active / stale / archived)
  - Memory       : A single unit of business knowledge with its embedding
  - Entity       : A business entity (supplier, customer, product) linked to memories
  - QueryContext : Parameters for a context retrieval query
  - ScoredMemory : A memory annotated with relevance scores
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


# ──────────────────────────────── Enums ─────────────────────────────────────

class MemoryType(str, Enum):
    """Categories of business context — mirrors the assignment taxonomy."""
    IMMEDIATE = "immediate"         # Current transaction details
    HISTORICAL = "historical"       # Past interactions, patterns, outcomes
    TEMPORAL = "temporal"           # Time-sensitive / seasonal information
    EXPERIENTIAL = "experiential"   # Lessons learned from successes & failures


class MemoryStatus(str, Enum):
    """Lifecycle state of a memory."""
    ACTIVE = "active"       # Fresh and usable
    STALE = "stale"         # Past its freshness window but still available
    ARCHIVED = "archived"   # No longer surfaced in queries


# ──────────────────────────────── Memory ────────────────────────────────────

@dataclass
class Memory:
    """
    A single unit of business knowledge.

    Attributes
    ----------
    id            : Unique identifier (UUID).
    content       : Human-readable text describing the knowledge.
    memory_type   : Category (immediate / historical / …).
    status        : Lifecycle state.
    entity_id     : ID of the related business entity (supplier, customer …).
    entity_type   : Type label for the entity (e.g. "supplier", "customer").
    tags          : Free-form labels for cross-linking (e.g. ["quality", "logistics"]).
    importance    : 1–10 rating; ≥ 8 is treated as evergreen.
    created_at    : When the memory was first recorded.
    updated_at    : Last time the memory was modified or accessed.
    access_count  : How many times this memory has been retrieved.
    embedding     : Vector embedding of `content` (set after creation).
    metadata      : Any extra key-value data (amounts, dates, names …).
    """
    content: str
    memory_type: MemoryType
    entity_id: str
    entity_type: str = ""
    tags: List[str] = field(default_factory=list)
    importance: int = 5
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    status: MemoryStatus = MemoryStatus.ACTIVE
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: Dict = field(default_factory=dict)

    # ── helpers ──────────────────────────────────────────────────────────
    def days_since_creation(self) -> float:
        """Number of days elapsed since this memory was created."""
        return (datetime.now() - self.created_at).total_seconds() / 86_400

    def days_since_update(self) -> float:
        """Number of days elapsed since this memory was last updated."""
        return (datetime.now() - self.updated_at).total_seconds() / 86_400

    def to_dict(self) -> Dict:
        """Serialize to a plain dict (embedding stored as list for JSON)."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "status": self.status.value,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "tags": self.tags,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
        }

    @classmethod 
    def from_dict(cls, data: Dict) -> "Memory":
        """Deserialize from a dict (re-creates numpy embedding)."""
        emb = data.get("embedding")
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            status=MemoryStatus(data["status"]),
            entity_id=data["entity_id"],
            entity_type=data.get("entity_type", ""),
            tags=data.get("tags", []),
            importance=data.get("importance", 5),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            access_count=data.get("access_count", 0),
            embedding=np.array(emb) if emb is not None else None,
            metadata=data.get("metadata", {}),
        )


# ──────────────────────────────── Entity ────────────────────────────────────

@dataclass
class Entity:
    """
    A business entity (supplier, customer, product, warehouse …).
    Serves as a hub linking related memories.
    """
    id: str
    name: str
    entity_type: str                          # "supplier", "customer", "product", …
    metadata: Dict = field(default_factory=dict)


# ────────────────────────────── QueryContext ────────────────────────────────

@dataclass
class QueryContext:
    """
    Parameters for a context retrieval query.

    Attributes
    ----------
    query_text         : Natural-language description of the decision context.
    entity_id          : (optional) Restrict to memories linked to this entity.
    memory_types       : (optional) Only include these memory types.
    time_range_days    : (optional) Only include memories created within N days.
    top_k              : Max number of results to return.
    include_stale      : Whether to also consider stale (but not archived) memories.
    """
    query_text: str
    entity_id: Optional[str] = None
    memory_types: Optional[List[MemoryType]] = None
    time_range_days: Optional[int] = None
    top_k: int = 10
    include_stale: bool = False


# ────────────────────────────── ScoredMemory ────────────────────────────────

@dataclass
class ScoredMemory:
    """A Memory annotated with its relevance scores for a given query."""
    memory: Memory
    semantic_score: float = 0.0     # cosine similarity (0–1)
    temporal_score: float = 0.0     # recency via exponential decay (0–1)
    relational_score: float = 0.0   # entity / tag overlap (0–1)
    final_score: float = 0.0        # weighted combination

    def explain(self) -> str:
        """Human-readable explanation of why this memory was retrieved."""
        lines = [
            f"Memory  : {self.memory.content[:100]}...",
            f"Type    : {self.memory.memory_type.value}",
            f"Status  : {self.memory.status.value}",
            f"Age     : {self.memory.days_since_creation():.0f} days",
            f"Scores  : semantic={self.semantic_score:.3f}  "
            f"temporal={self.temporal_score:.3f}  "
            f"relational={self.relational_score:.3f}",
            f"Final   : {self.final_score:.3f}",
        ]
        return "\n".join(lines)
