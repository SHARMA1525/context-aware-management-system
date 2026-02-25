"""
Tests for the Context & Memory Management System.

Covers: memory CRUD, vector embeddings, semantic search, entity linking,
        relevance scoring, lifecycle management, conflict resolution,
        overload prevention, and end-to-end retrieval.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from config import DECAY_RATE, MAX_CONTEXT_ITEMS
from context_manager import ContextManager
from lifecycle_manager import LifecycleManager
from memory_store import MemoryStore
from models import Entity, Memory, MemoryStatus, MemoryType, QueryContext, ScoredMemory


# ─── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def store():
    """Shared MemoryStore for the test module (model loaded once)."""
    return MemoryStore()


@pytest.fixture
def ctx_mgr():
    return ContextManager()


@pytest.fixture
def lc_mgr():
    return LifecycleManager()


# ════════════════════════════════════════════════════════════════════════
#  1. Memory CRUD
# ════════════════════════════════════════════════════════════════════════

class TestMemoryCRUD:
    """Test basic Create / Read / Update / Delete operations."""

    def test_add_and_get_memory(self, store: MemoryStore):
        mem = store.add_memory(
            content="Test memory for CRUD",
            memory_type=MemoryType.IMMEDIATE,
            entity_id="test-entity",
        )
        assert mem.id is not None
        assert store.get_memory(mem.id) is mem

    def test_update_memory_content(self, store: MemoryStore):
        mem = store.add_memory(
            content="Original content",
            memory_type=MemoryType.HISTORICAL,
            entity_id="test-entity",
        )
        old_embedding = mem.embedding.copy()
        store.update_memory(mem.id, content="Updated content completely different")
        assert mem.content == "Updated content completely different"
        # Embedding should be re-computed
        assert not np.allclose(old_embedding, mem.embedding)

    def test_delete_memory(self, store: MemoryStore):
        mem = store.add_memory(
            content="To be deleted",
            memory_type=MemoryType.TEMPORAL,
            entity_id="test-entity",
        )
        mid = mem.id
        assert store.delete_memory(mid) is True
        assert store.get_memory(mid) is None

    def test_delete_nonexistent(self, store: MemoryStore):
        assert store.delete_memory("nonexistent-id") is False


# ════════════════════════════════════════════════════════════════════════
#  2. Vector Embeddings
# ════════════════════════════════════════════════════════════════════════

class TestEmbeddings:
    """Test that embeddings are generated and cosine similarity works."""

    def test_embedding_created(self, store: MemoryStore):
        mem = store.add_memory(
            content="Vector embedding test",
            memory_type=MemoryType.IMMEDIATE,
            entity_id="emb-test",
        )
        assert mem.embedding is not None
        assert len(mem.embedding.shape) == 1  # 1D vector
        assert mem.embedding.shape[0] > 0

    def test_similar_texts_have_high_similarity(self, store: MemoryStore):
        """Two semantically similar texts should have cosine sim > 0.5."""
        vec_a = store._embed("The product quality is excellent")
        vec_b = store._embed("The quality of goods is very high")
        sim = float(np.dot(vec_a, vec_b))
        assert sim > 0.5

    def test_dissimilar_texts_have_low_similarity(self, store: MemoryStore):
        """Unrelated texts should have lower cosine similarity."""
        vec_a = store._embed("Invoice payment processing")
        vec_b = store._embed("The weather is sunny today")
        sim = float(np.dot(vec_a, vec_b))
        assert sim < 0.5


# ════════════════════════════════════════════════════════════════════════
#  3. Entity Linking
# ════════════════════════════════════════════════════════════════════════

class TestEntityLinking:
    """Test that memories are correctly linked to entities."""

    def test_get_by_entity(self, store: MemoryStore):
        eid = "entity-link-test"
        store.add_entity(Entity(id=eid, name="Test Entity", entity_type="test"))
        store.add_memory(content="Memory A for entity", memory_type=MemoryType.IMMEDIATE, entity_id=eid)
        store.add_memory(content="Memory B for entity", memory_type=MemoryType.HISTORICAL, entity_id=eid)
        memories = store.get_by_entity(eid)
        assert len(memories) >= 2

    def test_get_by_type(self, store: MemoryStore):
        store.add_memory(content="Type filter test", memory_type=MemoryType.EXPERIENTIAL, entity_id="tf-test")
        experiential = store.get_by_type(MemoryType.EXPERIENTIAL)
        assert len(experiential) >= 1


# ════════════════════════════════════════════════════════════════════════
#  4. Semantic Search
# ════════════════════════════════════════════════════════════════════════

class TestSemanticSearch:
    """Test the vector similarity search."""

    def test_search_returns_results(self, store: MemoryStore):
        store.add_memory(
            content="Supplier delivered damaged goods causing production delay",
            memory_type=MemoryType.HISTORICAL,
            entity_id="search-test",
        )
        results = store.search_similar("quality problems with supplier", top_k=5)
        assert len(results) > 0
        # Each result is (Memory, similarity)
        mem, sim = results[0]
        assert isinstance(mem, Memory)
        assert isinstance(sim, float)

    def test_search_top_result_is_most_similar(self, store: MemoryStore):
        results = store.search_similar("quality problems", top_k=5)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]  # sorted descending


# ════════════════════════════════════════════════════════════════════════
#  5. Relevance Scoring
# ════════════════════════════════════════════════════════════════════════

class TestRelevanceScoring:
    """Test the three-axis scoring in ContextManager."""

    def test_semantic_score_bounds(self, ctx_mgr: ContextManager):
        assert ctx_mgr.semantic_score(0.95) == 0.95
        assert ctx_mgr.semantic_score(-0.5) == 0.0
        assert ctx_mgr.semantic_score(1.5) == 1.0

    def test_temporal_score_recent(self, ctx_mgr: ContextManager):
        recent = Memory(
            content="recent", memory_type=MemoryType.IMMEDIATE,
            entity_id="t", created_at=datetime.now(),
        )
        score = ctx_mgr.temporal_score(recent)
        assert score > 0.9  # very recent → close to 1

    def test_temporal_score_old(self, ctx_mgr: ContextManager):
        old = Memory(
            content="old", memory_type=MemoryType.IMMEDIATE,
            entity_id="t", created_at=datetime.now() - timedelta(days=365),
        )
        score = ctx_mgr.temporal_score(old)
        expected = math.exp(-DECAY_RATE * 365)
        assert abs(score - expected) < 0.05

    def test_relational_score_same_entity(self, ctx_mgr: ContextManager):
        mem = Memory(content="x", memory_type=MemoryType.IMMEDIATE, entity_id="ent-1")
        assert ctx_mgr.relational_score(mem, query_entity_id="ent-1") == 1.0

    def test_relational_score_shared_tag(self, ctx_mgr: ContextManager):
        mem = Memory(content="x", memory_type=MemoryType.IMMEDIATE, entity_id="ent-2", tags=["quality"])
        assert ctx_mgr.relational_score(mem, query_entity_id="ent-999", query_tags=["quality"]) == 0.5

    def test_relational_score_no_match(self, ctx_mgr: ContextManager):
        mem = Memory(content="x", memory_type=MemoryType.IMMEDIATE, entity_id="ent-2")
        assert ctx_mgr.relational_score(mem, query_entity_id="ent-999") == 0.0


# ════════════════════════════════════════════════════════════════════════
#  6. Lifecycle Management
# ════════════════════════════════════════════════════════════════════════

class TestLifecycle:
    """Test staleness detection, archival, evergreen, and refresh."""

    def test_old_temporal_memory_is_stale(self, lc_mgr: LifecycleManager):
        mem = Memory(
            content="old temporal",
            memory_type=MemoryType.TEMPORAL,
            entity_id="lc-test",
            updated_at=datetime.now() - timedelta(days=60),
        )
        assert lc_mgr.check_staleness(mem) is True

    def test_recent_memory_not_stale(self, lc_mgr: LifecycleManager):
        mem = Memory(
            content="recent",
            memory_type=MemoryType.TEMPORAL,
            entity_id="lc-test",
            updated_at=datetime.now(),
        )
        assert lc_mgr.check_staleness(mem) is False

    def test_evergreen_memory_never_stale(self, lc_mgr: LifecycleManager):
        mem = Memory(
            content="critical lesson",
            memory_type=MemoryType.EXPERIENTIAL,
            entity_id="lc-test",
            importance=9,
            updated_at=datetime.now() - timedelta(days=999),
        )
        assert lc_mgr.is_evergreen(mem) is True
        assert lc_mgr.check_staleness(mem) is False

    def test_evergreen_tag(self, lc_mgr: LifecycleManager):
        mem = Memory(
            content="tagged evergreen",
            memory_type=MemoryType.HISTORICAL,
            entity_id="lc-test",
            importance=3,
            tags=["evergreen"],
            updated_at=datetime.now() - timedelta(days=999),
        )
        assert lc_mgr.is_evergreen(mem) is True

    def test_lifecycle_sweep(self, lc_mgr: LifecycleManager):
        memories = [
            Memory(content="fresh", memory_type=MemoryType.IMMEDIATE, entity_id="sw",
                   updated_at=datetime.now()),
            Memory(content="old temporal", memory_type=MemoryType.TEMPORAL, entity_id="sw",
                   updated_at=datetime.now() - timedelta(days=60)),
        ]
        stats = lc_mgr.run_lifecycle_sweep(memories)
        assert stats["marked_stale"] >= 1
        assert memories[1].status == MemoryStatus.STALE

    def test_refresh_reactivates(self, lc_mgr: LifecycleManager):
        mem = Memory(content="stale one", memory_type=MemoryType.TEMPORAL, entity_id="r",
                     status=MemoryStatus.STALE)
        lc_mgr.refresh_memory(mem)
        assert mem.status == MemoryStatus.ACTIVE
        assert mem.access_count == 1

    def test_invalidate(self, lc_mgr: LifecycleManager):
        mem = Memory(content="contract replaced", memory_type=MemoryType.IMMEDIATE, entity_id="inv")
        lc_mgr.invalidate_memory(mem, reason="New contract signed")
        assert mem.status == MemoryStatus.ARCHIVED
        assert mem.metadata["_invalidated_reason"] == "New contract signed"


# ════════════════════════════════════════════════════════════════════════
#  7. Overload Prevention
# ════════════════════════════════════════════════════════════════════════

class TestOverloadPrevention:
    """Test that the system respects MAX_CONTEXT_ITEMS."""

    def test_top_k_cap(self, ctx_mgr: ContextManager):
        # Create many fake scored memories
        candidates = []
        for i in range(20):
            mem = Memory(content=f"memory {i}", memory_type=MemoryType.HISTORICAL,
                         entity_id="cap-test", created_at=datetime.now())
            mem.embedding = np.random.randn(384)
            candidates.append((mem, 0.5 + i * 0.01))

        qctx = QueryContext(query_text="test", top_k=5)
        result = ctx_mgr.build_context(candidates, qctx)
        assert len(result) <= 5


# ════════════════════════════════════════════════════════════════════════
#  8. Conflict Resolution
# ════════════════════════════════════════════════════════════════════════

class TestConflictResolution:
    """Test that contradictory memories are handled correctly."""

    def test_newer_memory_boosted(self, ctx_mgr: ContextManager):
        old_mem = Memory(content="supplier bad quality", memory_type=MemoryType.HISTORICAL,
                         entity_id="conflict-ent", tags=["quality"],
                         created_at=datetime.now() - timedelta(days=180))
        new_mem = Memory(content="supplier improved quality", memory_type=MemoryType.HISTORICAL,
                         entity_id="conflict-ent", tags=["quality"],
                         created_at=datetime.now() - timedelta(days=15))

        scored = [
            ScoredMemory(memory=old_mem, final_score=0.7),
            ScoredMemory(memory=new_mem, final_score=0.7),
        ]
        resolved = ctx_mgr.resolve_conflicts(scored)
        # Newer memory should be first (boosted)
        assert resolved[0].memory.content == "supplier improved quality"
        assert resolved[0].final_score > resolved[1].final_score


# ════════════════════════════════════════════════════════════════════════
#  9. Serialization
# ════════════════════════════════════════════════════════════════════════

class TestSerialization:
    """Test Memory to_dict / from_dict round-trip."""

    def test_roundtrip(self):
        mem = Memory(
            content="serialize me",
            memory_type=MemoryType.EXPERIENTIAL,
            entity_id="ser-ent",
            tags=["test", "serialize"],
            importance=7,
        )
        mem.embedding = np.random.randn(384)

        d = mem.to_dict()
        restored = Memory.from_dict(d)

        assert restored.id == mem.id
        assert restored.content == mem.content
        assert restored.memory_type == mem.memory_type
        assert restored.tags == mem.tags
        assert np.allclose(restored.embedding, mem.embedding)
