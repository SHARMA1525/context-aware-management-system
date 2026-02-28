from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

class MemoryType(str, Enum):
    IMMEDIATE = "immediate"         
    HISTORICAL = "historical"      
    TEMPORAL = "temporal"           
    EXPERIENTIAL = "experiential"  


class MemoryStatus(str, Enum):
    ACTIVE = "active"      
    STALE = "stale"      
    ARCHIVED = "archived"  


@dataclass
class Memory:
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

    def days_since_creation(self) -> float:
        return (datetime.now() - self.created_at).total_seconds() / 86_400

    def days_since_update(self) -> float:
        return (datetime.now() - self.updated_at).total_seconds() / 86_400

    def to_dict(self) -> Dict:
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


@dataclass
class Entity:
    id: str
    name: str
    entity_type: str                         
    metadata: Dict = field(default_factory=dict)


@dataclass
class QueryContext:
    query_text: str
    entity_id: Optional[str] = None
    memory_types: Optional[List[MemoryType]] = None
    time_range_days: Optional[int] = None
    top_k: int = 10
    include_stale: bool = False


@dataclass
class ScoredMemory:
    memory: Memory
    semantic_score: float = 0.0   
    temporal_score: float = 0.0     
    relational_score: float = 0.0  
    final_score: float = 0.0       

    def explain(self) -> str:
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
