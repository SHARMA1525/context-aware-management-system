"""
Seed Data — Pre-loaded business scenarios from the assignment.

Creates entities and memories for:
  1. Invoice Processing (Supplier XYZ)
  2. Customer Support Ticket (TechCorp Inc.)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from memory_store import MemoryStore
from models import Entity, MemoryType


def _days_ago(n: int) -> datetime:
    """Helper to create a datetime N days in the past."""
    return datetime.now() - timedelta(days=n)


def seed_invoice_scenario(store: MemoryStore) -> str:
    """
    Seed Scenario 1: Invoice Processing with Historical Context.

    Returns the entity_id for Supplier XYZ.
    """
    # --- Entity ---
    supplier = Entity(
        id="supplier-xyz",
        name="Supplier XYZ",
        entity_type="supplier",
        metadata={"industry": "manufacturing", "region": "Maharashtra"},
    )
    store.add_entity(supplier)

    # --- Immediate context ---
    store.add_memory(
        content="Invoice #INV-2024-1587 received from Supplier XYZ for ₹2,50,000. "
                "Linked to Purchase Order PO-2024-892 for raw material steel rods.",
        memory_type=MemoryType.IMMEDIATE,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["invoice", "purchase-order", "payment"],
        importance=7,
        created_at=_days_ago(1),
        metadata={"amount": 250000, "currency": "INR", "po_number": "PO-2024-892"},
    )

    store.add_memory(
        content="Goods Receiving Note GRN-2024-445 confirms receipt of 500 steel rods "
                "at Warehouse-B on 20-Feb-2024. Inspection pending.",
        memory_type=MemoryType.IMMEDIATE,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["grn", "warehouse", "inspection"],
        importance=6,
        created_at=_days_ago(2),
        metadata={"grn_number": "GRN-2024-445", "warehouse": "Warehouse-B"},
    )

    store.add_memory(
        content="Contract agreement with Supplier XYZ specifies Net-30 payment terms, "
                "2% early payment discount if paid within 10 days, and quality requirements "
                "of less than 1% defect rate per shipment.",
        memory_type=MemoryType.IMMEDIATE,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["contract", "payment-terms", "quality"],
        importance=8,
        created_at=_days_ago(90),
        metadata={"payment_terms": "Net-30", "early_discount": "2%"},
    )

    # --- Historical context ---
    store.add_memory(
        content="Supplier XYZ delivered 30% broken products in batch B-2024-112 four months ago, "
                "leading to ₹50,000 in replacement costs and a 2-week delay in production line A.",
        memory_type=MemoryType.HISTORICAL,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["quality", "defects", "production-delay"],
        importance=9,
        created_at=_days_ago(120),
        metadata={"defect_rate": "30%", "cost_impact": 50000, "delay_weeks": 2},
    )

    store.add_memory(
        content="Supplier XYZ disputed invoice #INV-2024-0902 eight months ago claiming "
                "non-receipt of payment confirmation. Issue resolved after 3 weeks of follow-up.",
        memory_type=MemoryType.HISTORICAL,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["payment", "dispute", "invoice"],
        importance=6,
        created_at=_days_ago(240),
        metadata={"disputed_amount": 180000, "resolution_days": 21},
    )

    store.add_memory(
        content="Supplier XYZ quality has been satisfactory (less than 1% defects) for "
                "the last 3 months, with on-time delivery across all 5 shipments.",
        memory_type=MemoryType.HISTORICAL,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["quality", "delivery", "improvement"],
        importance=7,
        created_at=_days_ago(15),
        metadata={"defect_rate": "<1%", "shipments_on_time": 5},
    )

    # --- Temporal context ---
    store.add_memory(
        content="The 'Ship To' Warehouse-B location experienced severe road damage during "
                "last monsoon (July 2024), causing delivery delays of 9+ hours and additional "
                "handling costs of ₹10,000 per shipment.",
        memory_type=MemoryType.TEMPORAL,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["logistics", "warehouse", "monsoon", "seasonal"],
        importance=7,
        created_at=_days_ago(200),
        metadata={"season": "monsoon", "extra_cost": 10000},
    )

    store.add_memory(
        content="Delivery quality from Supplier XYZ degrades during summer months "
                "(March-May) due to heat-sensitive packaging of certain raw materials.",
        memory_type=MemoryType.TEMPORAL,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["quality", "seasonal", "packaging", "summer"],
        importance=7,
        created_at=_days_ago(300),
        metadata={"affected_months": ["March", "April", "May"]},
    )

    # --- Experiential context ---
    store.add_memory(
        content="Lesson learned: Always request sample inspection before releasing full "
                "payment to Supplier XYZ for orders above ₹2,00,000. Previous large orders "
                "without inspection led to undetected quality issues.",
        memory_type=MemoryType.EXPERIENTIAL,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["quality", "inspection", "payment", "lesson", "evergreen"],
        importance=9,
        created_at=_days_ago(100),
        metadata={"threshold_amount": 200000},
    )

    store.add_memory(
        content="Lesson learned: Early payment discount from Supplier XYZ is worth pursuing "
                "only when incoming quality is confirmed. Paying early on defective goods "
                "complicated the dispute resolution process in Q1-2024.",
        memory_type=MemoryType.EXPERIENTIAL,
        entity_id="supplier-xyz",
        entity_type="supplier",
        tags=["payment", "discount", "quality", "lesson"],
        importance=8,
        created_at=_days_ago(150),
        metadata={"quarter": "Q1-2024"},
    )

    return "supplier-xyz"


def seed_support_scenario(store: MemoryStore) -> str:
    """
    Seed Scenario 2: Customer Support Ticket Escalation.

    Returns the entity_id for TechCorp Inc.
    """
    # --- Entity ---
    customer = Entity(
        id="customer-techcorp",
        name="TechCorp Inc.",
        entity_type="customer",
        metadata={"tier": "enterprise", "contract_value": 5000000},
    )
    store.add_entity(customer)

    # --- Immediate context ---
    store.add_memory(
        content="Support ticket #TKT-2024-3891 submitted by TechCorp Inc.: "
                "'REST API integration returning 502 errors intermittently during peak "
                "hours (10am-2pm IST). Affecting their order processing pipeline.'",
        memory_type=MemoryType.IMMEDIATE,
        entity_id="customer-techcorp",
        entity_type="customer",
        tags=["support", "api", "integration", "error"],
        importance=8,
        created_at=_days_ago(0),
        metadata={"ticket_id": "TKT-2024-3891", "error_code": "502"},
    )

    store.add_memory(
        content="TechCorp Inc. SLA terms: Enterprise Platinum plan with 4-hour response time, "
                "99.9% uptime guarantee, and dedicated support engineer assigned.",
        memory_type=MemoryType.IMMEDIATE,
        entity_id="customer-techcorp",
        entity_type="customer",
        tags=["sla", "contract", "support"],
        importance=8,
        created_at=_days_ago(60),
        metadata={"sla_response_hours": 4, "uptime_guarantee": "99.9%"},
    )

    # --- Historical context ---
    store.add_memory(
        content="TechCorp Inc. renewed their ₹50 lakh annual contract 2 months ago but "
                "mentioned considering competitors (CloudAPI Pro, FastConnect) during "
                "renewal discussion. Final renewal included a 10% discount.",
        memory_type=MemoryType.HISTORICAL,
        entity_id="customer-techcorp",
        entity_type="customer",
        tags=["contract", "renewal", "competitors", "churn-risk"],
        importance=9,
        created_at=_days_ago(60),
        metadata={"contract_value": 5000000, "discount": "10%"},
    )

    store.add_memory(
        content="Similar API integration issue occurred with TechCorp Inc. 6 months ago "
                "(ticket #TKT-2024-1456). Root cause was rate limiting on their account. "
                "Resolved in 48 hours, but customer expressed frustration about response time.",
        memory_type=MemoryType.HISTORICAL,
        entity_id="customer-techcorp",
        entity_type="customer",
        tags=["support", "api", "integration", "resolution"],
        importance=8,
        created_at=_days_ago(180),
        metadata={"ticket_id": "TKT-2024-1456", "resolution_hours": 48},
    )

    store.add_memory(
        content="TechCorp Inc. API usage increased 300% in the last quarter, from 50K to "
                "200K daily requests. Suggests business-critical dependency on our platform.",
        memory_type=MemoryType.HISTORICAL,
        entity_id="customer-techcorp",
        entity_type="customer",
        tags=["usage", "api", "growth", "dependency"],
        importance=8,
        created_at=_days_ago(30),
        metadata={"daily_requests_before": 50000, "daily_requests_now": 200000},
    )

    # --- Temporal context ---
    store.add_memory(
        content="TechCorp Inc. has a product launch scheduled for next month. Their CTO "
                "mentioned during last check-in that API reliability is critical for the launch.",
        memory_type=MemoryType.TEMPORAL,
        entity_id="customer-techcorp",
        entity_type="customer",
        tags=["launch", "timeline", "critical"],
        importance=9,
        created_at=_days_ago(14),
        metadata={"launch_eta": "next_month"},
    )

    # --- Experiential context ---
    store.add_memory(
        content="Lesson learned: TechCorp Inc.'s CTO (Raj Mehta) is directly involved in "
                "escalations. He prefers technical deep-dives with raw data over summary "
                "reports. Communication should include error logs, RCA timelines, and "
                "concrete remediation steps.",
        memory_type=MemoryType.EXPERIENTIAL,
        entity_id="customer-techcorp",
        entity_type="customer",
        tags=["stakeholder", "communication", "escalation", "evergreen"],
        importance=9,
        created_at=_days_ago(180),
        metadata={"cto_name": "Raj Mehta", "preference": "technical_deep_dive"},
    )

    store.add_memory(
        content="Lesson learned: For enterprise customers like TechCorp with churn risk, "
                "involve Customer Success Manager (CSM) early in escalations. In the past, "
                "delayed CSM involvement led to unnecessary executive escalation.",
        memory_type=MemoryType.EXPERIENTIAL,
        entity_id="customer-techcorp",
        entity_type="customer",
        tags=["escalation", "csm", "churn-risk", "lesson"],
        importance=8,
        created_at=_days_ago(150),
        metadata={"best_practice": True},
    )

    return "customer-techcorp"


def seed_all(store: MemoryStore) -> dict:
    """
    Seed both business scenarios and return entity IDs.

    Returns
    -------
    {"supplier": "supplier-xyz", "customer": "customer-techcorp"}
    """
    supplier_id = seed_invoice_scenario(store)
    customer_id = seed_support_scenario(store)
    print(f"\n[SeedData] Loaded {len(store.memories)} memories for 2 scenarios.")
    return {"supplier": supplier_id, "customer": customer_id}
