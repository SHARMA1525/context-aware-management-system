"""
Configuration constants for the Context & Memory Management System.
All tuning knobs are centralized here for easy adjustment.
"""

# ───────────────────────────── Embedding Model ──────────────────────────────
# Lightweight model (~80 MB) that produces 384-dim vectors.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ──────────────────────────── Staleness Thresholds ──────────────────────────
# Number of days after which a memory type is considered "stale".
# Memories tagged as evergreen or with importance >= 8 bypass staleness.
from enum import Enum  # noqa: E402 (needed at module level for the dict below)


class MemoryTypeName(str, Enum):
    """Mirror of models.MemoryType for config use (avoids circular import)."""
    IMMEDIATE = "immediate"
    HISTORICAL = "historical"
    TEMPORAL = "temporal"
    EXPERIENTIAL = "experiential"


STALENESS_THRESHOLDS = {
    MemoryTypeName.IMMEDIATE: 7,       # Current transaction data — stale in 1 week
    MemoryTypeName.HISTORICAL: 365,    # Past patterns — stale after 1 year
    MemoryTypeName.TEMPORAL: 30,       # Time-sensitive info — stale in 1 month
    MemoryTypeName.EXPERIENTIAL: 180,  # Lessons learned — stale in 6 months
}

# ──────────────────────── Archive Rules ─────────────────────────────────────
# A stale memory is archived if it has not been accessed for this many days.
ARCHIVE_INACTIVITY_DAYS = 90

# ──────────────────────── Temporal Decay ────────────────────────────────────
# Used in the formula: temporal_score = exp(-DECAY_RATE * days_since_creation)
# Higher value → older memories lose relevance faster.
DECAY_RATE = 0.01

# ──────────────────────── Relevance Weights ─────────────────────────────────
# Weights for the three scoring dimensions (must sum to 1.0).
RELEVANCE_WEIGHTS = {
    "semantic":   0.50,   # How close the meaning is to the query
    "temporal":   0.25,   # How recent the memory is
    "relational": 0.25,   # How closely tied to the queried entity
}

# ──────────────────────── Retrieval Caps ────────────────────────────────────
# Maximum number of context items returned per query (prevents overload).
MAX_CONTEXT_ITEMS = 10

# Minimum final score to include a memory in results (filters noise).
MIN_RELEVANCE_SCORE = 0.15

# ──────────────────────── Evergreen Rules ───────────────────────────────────
# Memories with importance >= this value are treated as evergreen.
EVERGREEN_IMPORTANCE_THRESHOLD = 8

# Tag that explicitly marks a memory as evergreen regardless of importance.
EVERGREEN_TAG = "evergreen"
