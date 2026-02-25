"""
Memory Store — Vector-based storage and retrieval of business memories.

Uses sentence-transformers to embed text and numpy for cosine similarity.
All memories live in-memory (dict-backed) with optional JSON persistence.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from models import Entity, Memory, MemoryStatus, MemoryType


class MemoryStore:
    """
    Central store for business memories.

    Responsibilities
    ----------------
    - Embed content using a SentenceTransformer model.
    - Store / retrieve / update / delete memories.
    - Perform semantic search via cosine similarity.
    - Maintain an entity index for fast relationship lookups.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model ID for sentence-transformers.
        """
        print(f"[MemoryStore] Loading embedding model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()

        # Primary storage
        self.memories: Dict[str, Memory] = {}

        # Secondary indexes
        self.entity_index: Dict[str, List[str]] = {}   # entity_id → [memory_id, …]
        self.entities: Dict[str, Entity] = {}           # entity_id → Entity

        print(f"[MemoryStore] Ready. Embedding dimension = {self.embedding_dim}")

    # ────────────────────────── Embedding helpers ───────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Produce a normalised embedding vector for *text*."""
        vec = self.model.encode(text, convert_to_numpy=True)
        # L2-normalise so dot product == cosine similarity
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ────────────────────────── Entity management ──────────────────────────

    def add_entity(self, entity: Entity) -> Entity:
        """Register a business entity (supplier, customer, …)."""
        self.entities[entity.id] = entity
        if entity.id not in self.entity_index:
            self.entity_index[entity.id] = []
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    # ────────────────────────── Memory CRUD ─────────────────────────────────

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        entity_id: str,
        entity_type: str = "",
        tags: Optional[List[str]] = None,
        importance: int = 5,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> Memory:
        """
        Create a new Memory, embed its content, store it, and update indexes.

        Returns the newly created Memory object.
        """
        memory = Memory(
            content=content,
            memory_type=memory_type,
            entity_id=entity_id,
            entity_type=entity_type,
            tags=tags or [],
            importance=importance,
            created_at=created_at or datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {},
        )
        memory.embedding = self._embed(content)

        # Store
        self.memories[memory.id] = memory

        # Update entity index
        if entity_id not in self.entity_index:
            self.entity_index[entity_id] = []
        self.entity_index[entity_id].append(memory.id)

        return memory

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        return self.memories.get(memory_id)

    def update_memory(self, memory_id: str, **kwargs) -> Optional[Memory]:
        """
        Update fields on an existing memory.
        Re-embeds if `content` changes.
        """
        mem = self.memories.get(memory_id)
        if mem is None:
            return None

        for key, value in kwargs.items():
            if hasattr(mem, key):
                setattr(mem, key, value)

        # Re-embed if content changed
        if "content" in kwargs:
            mem.embedding = self._embed(mem.content)

        mem.updated_at = datetime.now()
        return mem

    def delete_memory(self, memory_id: str) -> bool:
        """Remove a memory from the store and indexes."""
        mem = self.memories.pop(memory_id, None)
        if mem is None:
            return False
        # Clean entity index
        if mem.entity_id in self.entity_index:
            self.entity_index[mem.entity_id] = [
                mid for mid in self.entity_index[mem.entity_id] if mid != memory_id
            ]
        return True

    # ────────────────────────── Query helpers ──────────────────────────────

    def get_by_entity(self, entity_id: str) -> List[Memory]:
        """Return all memories linked to a given entity."""
        ids = self.entity_index.get(entity_id, [])
        return [self.memories[mid] for mid in ids if mid in self.memories]

    def get_by_type(self, memory_type: MemoryType) -> List[Memory]:
        """Return all memories of a given type."""
        return [m for m in self.memories.values() if m.memory_type == memory_type]

    def get_all_active(self, include_stale: bool = False) -> List[Memory]:
        """Return all non-archived memories."""
        allowed = {MemoryStatus.ACTIVE}
        if include_stale:
            allowed.add(MemoryStatus.STALE)
        return [m for m in self.memories.values() if m.status in allowed]

    # ────────────────────────── Semantic search ────────────────────────────

    def search_similar(
        self,
        query_text: str,
        top_k: int = 10,
        entity_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        include_stale: bool = False,
    ) -> List[tuple]:
        """
        Embed *query_text* and return the top-k most similar memories.

        Returns a list of (Memory, cosine_similarity) tuples sorted descending.
        """
        query_vec = self._embed(query_text)

        # Gather candidate memories
        candidates = self.get_all_active(include_stale=include_stale)

        if entity_id:
            entity_ids_set = {entity_id}
            candidates = [m for m in candidates if m.entity_id in entity_ids_set]

        if memory_types:
            type_set = set(memory_types)
            candidates = [m for m in candidates if m.memory_type in type_set]

        if not candidates:
            return []

        # Build embedding matrix and compute cosine similarities in one shot
        emb_matrix = np.stack([m.embedding for m in candidates])  # (N, dim)
        similarities = emb_matrix @ query_vec                      # (N,)

        # Pair up and sort
        scored = list(zip(candidates, similarities.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    # ────────────────────────── Persistence ─────────────────────────────────

    def save_to_file(self, filepath: str) -> None:
        """Persist all memories and entities to a JSON file."""
        data = {
            "memories": [m.to_dict() for m in self.memories.values()],
            "entities": [
                {"id": e.id, "name": e.name, "entity_type": e.entity_type, "metadata": e.metadata}
                for e in self.entities.values()
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[MemoryStore] Saved {len(self.memories)} memories to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load memories and entities from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        for edata in data.get("entities", []):
            self.add_entity(Entity(**edata))

        for mdata in data.get("memories", []):
            mem = Memory.from_dict(mdata)
            self.memories[mem.id] = mem
            if mem.entity_id not in self.entity_index:
                self.entity_index[mem.entity_id] = []
            self.entity_index[mem.entity_id].append(mem.id)

        print(f"[MemoryStore] Loaded {len(self.memories)} memories from {filepath}")

    # ────────────────────────── Stats ───────────────────────────────────────

    def stats(self) -> Dict:
        """Return a summary of the store contents."""
        by_type = {}
        by_status = {}
        for m in self.memories.values():
            by_type[m.memory_type.value] = by_type.get(m.memory_type.value, 0) + 1
            by_status[m.status.value] = by_status.get(m.status.value, 0) + 1
        return {
            "total_memories": len(self.memories),
            "total_entities": len(self.entities),
            "by_type": by_type,
            "by_status": by_status,
        }
