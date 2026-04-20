"""
TCB-RCA Algorithm Validation Test

This script tests the TCB-RCA algorithm with the sample scenario:
    user-db connection pool exhaustion (T-30s)
    → auth-service timeout (T-20s)
    → api-gateway 502 errors (T-10s)

Expected result:
    Root cause = user-db (deepest anomalous node)
    Propagation path = user-db → auth-service → api-gateway
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.rca_agent.tcb_rca import (
    AnomalyEvidence,
    TCBRCAEngine,
    logs_to_anomaly_data,
)


def load_sample_logs():
    """Load sample logs from the JSONL file."""
    log_file = PROJECT_ROOT / "mcp_servers" / "observability_mcp" / "app" / "sample_logs.jsonl"
    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    print("=" * 70)
    print("  TCB-RCA Algorithm Validation Test")
    print("=" * 70)

    # 1. Load sample data
    raw_logs = load_sample_logs()
    print(f"\n[1] Loaded {len(raw_logs)} log records")

    anomaly_data = logs_to_anomaly_data(raw_logs)
    print(f"    Services with data: {list(anomaly_data.keys())}")

    for svc, anomalies in anomaly_data.items():
        error_count = sum(1 for a in anomalies if a.severity_score > 0)
        print(f"    - {svc}: {len(anomalies)} total, {error_count} anomalies")

    # 2. Set up topology
    topology = {
        "frontend-web": ["api-gateway"],
        "api-gateway": ["auth-service", "catalog-service", "order-service"],
        "auth-service": ["user-db"],
        "catalog-service": [],
        "order-service": ["message-queue", "order-db"],
        "message-queue": ["worker-service"],
        "worker-service": ["order-db"],
        "user-db": [],
        "order-db": [],
    }

    metadata = {
        "frontend-web": {"type": "frontend", "criticality": "high"},
        "api-gateway": {"type": "gateway", "criticality": "high"},
        "auth-service": {"type": "backend", "criticality": "high"},
        "catalog-service": {"type": "backend", "criticality": "medium"},
        "order-service": {"type": "backend", "criticality": "high"},
        "message-queue": {"type": "queue", "criticality": "high"},
        "worker-service": {"type": "worker", "criticality": "high"},
        "user-db": {"type": "database", "criticality": "high"},
        "order-db": {"type": "database", "criticality": "high"},
    }

    # 3. Initialize TCB-RCA engine
    engine = TCBRCAEngine(
        topology_graph=topology,
        service_metadata=metadata,
        delta_t_seconds=120,
        max_depth=10,
    )

    # 4. Execute algorithm
    symptom_service = "api-gateway"
    alert_time = datetime.fromisoformat("2026-03-24T13:03:15+09:00")

    print(f"\n[2] Executing TCB-RCA")
    print(f"    Symptom service: {symptom_service}")
    print(f"    Alert time: {alert_time.isoformat()}")
    print(f"    Delta-t: 120 seconds")

    result = engine.execute(
        incident_id="INC-TEST-001",
        symptom_service=symptom_service,
        alert_time=alert_time,
        anomaly_data=anomaly_data,
    )

    # 5. Print results
    print(f"\n[3] Traversal Summary")
    for key, value in result.traversal_summary.items():
        print(f"    {key}: {value}")

    print(f"\n[4] Root Cause Candidates (ranked)")
    print("-" * 70)

    for rc in result.root_cause_candidates:
        print(f"\n  Rank #{rc.rank}: {rc.cause_service}")
        print(f"    Confidence:    {rc.confidence:.3f}")
        print(f"    Depth:         {rc.depth} (hops from symptom)")
        print(f"    Temporal gap:  {rc.temporal_gap_seconds:.0f}s before alert")
        print(f"    Backtrack:     {' → '.join(rc.backtrack_path)}")
        print(f"    Description:   {rc.cause_description}")
        print(f"    Evidence chain ({len(rc.evidence_chain)} steps):")
        for step in rc.evidence_chain:
            print(
                f"      [{step['service']}] "
                f"severity={step['severity_score']:.3f}, "
                f"anomalies={step['anomaly_count']}, "
                f"earliest={step['earliest_anomaly']}"
            )

    print(f"\n[5] Inferred Propagation Path")
    print(f"    {' → '.join(result.propagation_path)}")

    print(f"\n[6] Blast Radius")
    print(f"    {result.blast_radius}")

    # 6. Validation
    print(f"\n{'=' * 70}")
    print("  VALIDATION")
    print(f"{'=' * 70}")

    top_cause = result.root_cause_candidates[0] if result.root_cause_candidates else None

    checks = [
        ("Root cause is user-db", top_cause and top_cause.cause_service == "user-db"),
        ("Confidence > 0.7", top_cause and top_cause.confidence > 0.7),
        ("Depth is 2", top_cause and top_cause.depth == 2),
        ("Propagation path starts at user-db", result.propagation_path[0] == "user-db"),
        ("Propagation path ends at api-gateway", result.propagation_path[-1] == "api-gateway"),
        ("Backtrack path: user-db → auth-service → api-gateway",
         top_cause and top_cause.backtrack_path == ["user-db", "auth-service", "api-gateway"]),
        ("Multiple candidates found", len(result.root_cause_candidates) >= 1),
        ("Blast radius includes api-gateway", "api-gateway" in result.blast_radius),
    ]

    all_passed = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {desc}")

    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
