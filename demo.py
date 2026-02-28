from __future__ import annotations

from context_manager import ContextManager
from lifecycle_manager import LifecycleManager
from memory_store import MemoryStore
from retrieval_engine import RetrievalEngine
from seed_data import seed_all


def divider(title: str = "") -> None:
    print("\n" + "═" * 70)
    if title:
        print(f"  {title}")
        print("═" * 70)


def run_demo() -> None:

    print("Context & Memory Management System — Demo\n")

    store = MemoryStore()
    ctx_mgr = ContextManager()
    lc_mgr = LifecycleManager()
    engine = RetrievalEngine(store, ctx_mgr, lc_mgr)

    entity_ids = seed_all(store)

    divider("MEMORY STORE STATISTICS")
    stats = store.stats()
    print(f"  Total memories  : {stats['total_memories']}")
    print(f"  Total entities  : {stats['total_entities']}")
    print(f"  By type         : {stats['by_type']}")
    print(f"  By status       : {stats['by_status']}")

    while True:
        divider("MAIN MENU")
        print("""
  Choose a demo scenario:

    1. Invoice Processing — Supplier XYZ (₹2,50,000 invoice)
    2. Support Ticket — TechCorp Inc. (API integration issue)
    3. Custom query (any text + optional entity)
    4. Run lifecycle sweep (show staleness transitions)
    5. Show memory store statistics
    6. Exit
        """)

        choice = input("  Enter choice (1-6): ").strip()

        if choice == "1":
            run_invoice_demo(engine, entity_ids["supplier"])

        elif choice == "2":
            run_support_demo(engine, entity_ids["customer"])

        elif choice == "3":
            run_custom_query(engine, entity_ids)

        elif choice == "4":
            run_lifecycle_demo(store, lc_mgr)

        elif choice == "5":
            divider("MEMORY STORE STATISTICS")
            stats = store.stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")

        elif choice == "6":
            print("\n  Goodbye! 👋\n")
            break

        else:
            print("Invalid choice. Try again.")

def run_invoice_demo(engine: RetrievalEngine, entity_id: str) -> None:

    queries = [
        (
            "Should this ₹2,50,000 invoice from Supplier XYZ be fast-tracked "
            "for payment, or does it need additional quality inspection?",
            "Decision: Invoice Payment Authorization",
        ),
        (
            "What are the quality issues and defect history with this supplier?",
            "Query: Supplier Quality History",
        ),
        (
            "Are there any seasonal or logistics risks for deliveries right now?",
            "Query: Logistics & Seasonal Risks",
        ),
    ]

    for query_text, label in queries:
        divider(label)
        result = engine.query(text=query_text, entity_id=entity_id, top_k=5)
        print(result.display())

        if result.scored_memories:
            print("\n  📋 WHY was the top memory retrieved?")
            print("  " + engine.explain(result.scored_memories[0]).replace("\n", "\n  "))


def run_support_demo(engine: RetrievalEngine, entity_id: str) -> None:

    queries = [
        (
            "Should this support ticket from TechCorp Inc. be immediately "
            "escalated? What context is relevant for the decision?",
            "Decision: Ticket Escalation",
        ),
        (
            "What is TechCorp's relationship status, churn risk, and contract history?",
            "Query: Customer Relationship Context",
        ),
        (
            "How should we communicate with TechCorp about this issue? "
            "Who are the stakeholders and what is their preferred style?",
            "Query: Communication & Stakeholder Preferences",
        ),
    ]

    for query_text, label in queries:
        divider(label)
        result = engine.query(text=query_text, entity_id=entity_id, top_k=5)
        print(result.display())

        if result.scored_memories:
            print("\n  📋 WHY was the top memory retrieved?")
            print("  " + engine.explain(result.scored_memories[0]).replace("\n", "\n  "))

def run_custom_query(engine: RetrievalEngine, entity_ids: dict) -> None:
    print("\n  Available entities:")
    for label, eid in entity_ids.items():
        print(f"    {label}: {eid}")
    print()

    query = input("  Enter your query: ").strip()
    if not query:
        print("Empty query.")
        return

    eid = input("  Entity ID (leave blank for all): ").strip() or None
    top_k = input("  Top-K results (default 5): ").strip()
    top_k = int(top_k) if top_k.isdigit() else 5

    result = engine.query(text=query, entity_id=eid, top_k=top_k)
    print(result.display())


def run_lifecycle_demo(store: MemoryStore, lc_mgr: LifecycleManager) -> None:
    divider("LIFECYCLE SWEEP")

    all_memories = list(store.memories.values())
    print(f"\n  Running sweep on {len(all_memories)} memories ...\n")

    stats = lc_mgr.run_lifecycle_sweep(all_memories)
    for key, count in stats.items():
        print(f"    {key}: {count}")

    stale = [m for m in all_memories if m.status.value == "stale"]
    archived = [m for m in all_memories if m.status.value == "archived"]

    if stale:
        print(f"\n  ⚠️  Stale memories ({len(stale)}):")
        for m in stale:
            print(f"    - [{m.memory_type.value}] {m.content[:80]}...")

    if archived:
        print(f"\n  🗄️  Archived memories ({len(archived)}):")
        for m in archived:
            print(f"    - [{m.memory_type.value}] {m.content[:80]}...")

    if not stale and not archived:
        print("\nAll memories are still active.")

if __name__ == "__main__":
    run_demo()
