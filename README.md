# Context & Memory Management System for AI Agents

A prototype system that enables AI agents to maintain, retrieve, and utilize contextual business information — using **vector embeddings** for semantic similarity, **rule-based scoring** for relevance ranking, and a **lifecycle manager** for memory freshness.

> Built as a solution to the "Context and Memory Management for AI Agents in Business Environments" design challenge.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL ENGINE                            │
│  (Orchestrates the full query → context pipeline)               │
├────────────┬───────────────────┬────────────────────────────────┤
│            │                   │                                │
│  MEMORY    │   CONTEXT         │   LIFECYCLE                    │
│  STORE     │   MANAGER         │   MANAGER                     │
│            │                   │                                │
│ • Embed    │ • Semantic score  │ • Staleness rules              │
│ • Store    │ • Temporal score  │ • Archival logic               │
│ • Search   │ • Relational      │ • Evergreen detection          │
│ • CRUD     │   score           │ • Refresh on access            │
│            │ • Conflict        │ • Manual invalidation          │
│            │   resolution      │                                │
├────────────┴───────────────────┴────────────────────────────────┤
│                      DATA MODELS                                │
│  Memory | Entity | QueryContext | ScoredMemory                  │
├─────────────────────────────────────────────────────────────────┤
│                    CONFIG (tuning knobs)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Types

| Type | Description | Staleness Threshold | Example |
|---|---|---|---|
| **Immediate** | Current transaction details | 7 days | Invoice, PO, GRN |
| **Historical** | Past interactions, patterns | 365 days | Quality issues, payment disputes |
| **Temporal** | Time-sensitive / seasonal info | 30 days | Monsoon logistics, summer packaging |
| **Experiential** | Lessons learned | 180 days | Inspection best practices |

Memories with **importance ≥ 8** or tagged `"evergreen"` are exempt from staleness.

---

## Relevance Scoring Formula

Each retrieved memory is scored on three dimensions:

```
final_score = 0.50 × semantic + 0.25 × temporal + 0.25 × relational
```

| Dimension | How It Works | Range |
|---|---|---|
| **Semantic** | Cosine similarity between query embedding and memory embedding | 0 – 1 |
| **Temporal** | `exp(-0.01 × days_since_creation)` — recent memories score higher | 0 – 1 |
| **Relational** | 1.0 if same entity, 0.5 if shared tag, 0.0 otherwise | 0 – 1 |

Weights are configurable in `config.py`.

---

## Memory Lifecycle

```
  ACTIVE ──────────────► STALE ──────────────► ARCHIVED
         (exceeds                (no access
          staleness               for 90+ days)
          threshold)

  Evergreen memories (importance ≥ 8 or tag "evergreen") skip staleness.
  Accessing a stale memory reactivates it → ACTIVE.
  External triggers can invalidate any memory → ARCHIVED.
```

---

## Project Structure

```
├── config.py              # Tuning knobs (thresholds, weights, caps)
├── models.py              # Data models (Memory, Entity, QueryContext, ScoredMemory)
├── memory_store.py        # Vector store (embed, CRUD, semantic search)
├── context_manager.py     # Relevance scoring & conflict resolution
├── lifecycle_manager.py   # Staleness, archival, evergreen rules
├── retrieval_engine.py    # Top-level orchestrator
├── seed_data.py           # Pre-loaded business scenarios
├── demo.py                # Interactive CLI demo
├── requirements.txt       # Python dependencies
└── tests/
    └── test_system.py     # Unit & integration tests
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
cd /path/to/project
pip install -r requirements.txt
```

The first run will download the `all-MiniLM-L6-v2` model (~80 MB) automatically.

---

## Step-by-Step Terminal Execution

Follow these steps to run the system and verify the output in your terminal:

### 1. Activate the Virtual Environment
Before running any scripts, ensure the virtual environment is active.
```bash
source venv/bin/activate
```
_You should see `(venv)` appearing at the beginning of your terminal prompt._

### 2. Run the Interactive Demo
Launch the main demonstration script to see the system's memory retrieval and scoring in action.
```bash
python3 demo.py
```
**Expected Terminal Output:**
- The system will load the embedding model.
- You will see "Memory Store Ready" and "Seed Data Loaded".
- A main menu will appear with several scenarios (Invoice Processing, Support Ticket, etc.).
- Select a choice (e.g., `1` for Invoice Processing) to see the context-aware retrieval.

### 3. Run the Unit Tests
To verify all components (scoring, lifecycle, storage) are working correctly:
```bash
pytest tests/test_system.py
```
**Expected Terminal Output:**
- You should see a list of tests passing (marked with `.`).
- A final summary will show the total number of passed tests.

### 4. Exit the Virtual Environment
Once finished, you can return to your normal terminal session:
```bash
deactivate
```

---

## Running the Demo

```bash
python demo.py
```

This launches an interactive CLI with a menu:

1. **Invoice Processing** — Supplier XYZ scenario (quality issues, logistics, payments)
2. **Support Ticket** — TechCorp Inc. scenario (churn risk, SLA, stakeholder preferences)
3. **Custom Query** — Free-form text query against all stored memories
4. **Lifecycle Sweep** — Demonstrates staleness transitions
5. **Store Statistics** — Memory counts by type and status

### Example Output (Invoice Scenario)

```
══════════════════════════════════════════════════════════════════════
  Decision: Invoice Payment Authorization
══════════════════════════════════════════════════════════════════════
  QUERY       : Should this ₹2,50,000 invoice be fast-tracked for payment?
  CANDIDATES  : 10
  RETURNED    : 5

--- Memory #1 (score: 0.812) ---
  Type       : experiential
  Content    : Lesson learned: Always request sample inspection before
               releasing full payment to Supplier XYZ for orders above ₹2,00,000.
  Scores     : semantic=0.654  temporal=0.368  relational=1.000
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Test coverage includes:
- Memory CRUD operations
- Vector embedding generation & cosine similarity
- Entity linking and retrieval
- Three-axis relevance scoring
- Lifecycle sweep (staleness, archival, evergreen)
- Conflict resolution (newer memory preferred)
- Overload prevention (top-k cap)
- Serialization round-trip

---

## Design Decisions & Trade-offs

### Why vector embeddings instead of keyword search?

Keyword search misses semantic relationships — e.g., "quality problems" wouldn't match "defective goods" or "broken products." Vector embeddings capture meaning, not just words.

### Why not use an LLM for scoring?

The assignment recommends deep thinking over LLM delegation. Rule-based scoring with explicit formulas is:
- **Transparent**: You can explain exactly why a memory was retrieved
- **Fast**: No API calls or GPU inference for scoring
- **Tunable**: Adjust weights in `config.py` to change behavior
- **Deterministic**: Same query → same results

### Why in-memory storage?

For a prototype, in-memory storage (with optional JSON persistence) avoids database setup complexity while demonstrating all concepts. For production, the `MemoryStore` could be backed by a vector database like Pinecone, Weaviate, or pgvector.

### Conflict Resolution Strategy

When contradictory memories exist (e.g., "supplier quality was terrible" vs "supplier quality improved"), the system:
1. Groups memories by entity + topic (first tag)
2. Boosts the most recent memory by 10%
3. Annotates older conflicting memories with a warning
4. Keeps both — because historical context about past problems is still valuable

### Scalability Considerations

| Concern | Current Approach | Production Path |
|---|---|---|
| Storage | In-memory dict | Vector DB (Pinecone, Weaviate) |
| Search | Brute-force cosine similarity | Approximate Nearest Neighbors (HNSW) |
| Embeddings | Computed on write | Batch embedding pipeline |
| Lifecycle | On-demand sweep | Scheduled background job |
| Multi-agent | Single process | Shared DB + cache layer |

---

## Addressing Assignment Questions

**How would the system scale?**
→ Replace in-memory storage with a vector database (e.g., Pinecone, pgvector) and use ANN indexes for sub-millisecond retrieval across millions of memories.

**Should emotional context be stored?**
→ Yes — sentiments like customer frustration are stored as tags and importance ratings. The semantic embedding naturally captures emotional tone.

**How is data privacy ensured?**
→ Entity-scoped access (queries are filtered by entity_id), memory status controls (archived = hidden), and the system can be extended with role-based access.

**Can the system explain its retrievals?**
→ Yes — `RetrievalEngine.explain()` provides a human-readable breakdown of semantic, temporal, and relational scores for any retrieved memory.

**Multi-agent shared context?**
→ The `MemoryStore` can be shared across agents. With a database backend, concurrent access is naturally supported.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| `sentence-transformers` | Vector embeddings (`all-MiniLM-L6-v2`) |
| `numpy` | Cosine similarity & vector math |
| `dataclasses` | Clean, typed data models |
| `pytest` | Test framework |

---

## Author

Built as a solution to the Context & Memory Management design challenge for AI agents in business environments.
