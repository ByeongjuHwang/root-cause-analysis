#!/usr/bin/env python3
"""
Phase 1 Validation Test — tests all critical changes without network/FastAPI.

Tests:
  1. TCB-RCA algorithm correctness (original test_tcb_rca logic)
  2. Dynamic topology resolution (3-level fallback)
  3. Topology Agent returns full graph
  4. Verifier uses dynamic known services
  5. RCA Service cause_service field in output
  6. Case study topology loading
  7. Evaluation function correctness
"""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS_COUNT = 0
FAIL_COUNT = 0


def check(desc: str, condition: bool):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {desc}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {desc}")


def test_tcb_rca_algorithm():
    print("\n=== Test 1: TCB-RCA Algorithm ===")
    from agents.rca_agent.tcb_rca import TCBRCAEngine, logs_to_anomaly_data
    from datetime import datetime

    records = []
    with open(PROJECT_ROOT / "mcp_servers/observability_mcp/app/sample_logs.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    anomaly_data = logs_to_anomaly_data(records)

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
    metadata = {s: {"type": "backend", "criticality": "high"} for s in topology}

    engine = TCBRCAEngine(topology_graph=topology, service_metadata=metadata)
    result = engine.execute(
        incident_id="INC-T1",
        symptom_service="api-gateway",
        alert_time=datetime.fromisoformat("2026-03-24T13:03:15+09:00"),
        anomaly_data=anomaly_data,
    )

    top = result.root_cause_candidates[0] if result.root_cause_candidates else None
    check("Root cause is user-db", top and top.cause_service == "user-db")
    check("Confidence > 0.7", top and top.confidence > 0.7)
    check("Depth is 2", top and top.depth == 2)
    check("Propagation: user-db → auth-service → api-gateway",
          result.propagation_path == ["user-db", "auth-service", "api-gateway"])
    check("Blast radius includes api-gateway", "api-gateway" in result.blast_radius)


def test_dynamic_topology_resolution():
    print("\n=== Test 2: Dynamic Topology Resolution ===")
    from agents.rca_agent.service import (
        RCAService, _load_topology_from_mcp, _extract_topology_from_agent_result,
    )

    # Strategy 1: From agent result
    agent_topo = _extract_topology_from_agent_result({
        "topology_graph": {"a": ["b"], "b": []},
        "service_metadata": {"a": {"type": "svc"}, "b": {"type": "db"}},
    })
    check("Agent result extraction works", agent_topo is not None and len(agent_topo[0]) == 2)

    # Strategy 2: From MCP
    graph, meta = _load_topology_from_mcp()
    check("MCP default loads 9 nodes", len(graph) == 9)

    # Strategy 3: MCP with case study
    graph2, meta2 = _load_topology_from_mcp(topology_file="case_study_topology.json")
    check("Case study topology loads", len(graph2) > 0 and "app-deployer" in graph2)

    # None case
    check("Returns None for empty result",
          _extract_topology_from_agent_result({"summary": "x"}) is None)


def test_rca_service_dynamic():
    print("\n=== Test 3: RCA Service Dynamic Topology ===")

    async def _run():
        from agents.rca_agent.service import RCAService

        svc = RCAService()
        records = []
        with open(PROJECT_ROOT / "mcp_servers/observability_mcp/app/sample_logs.jsonl") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        evidence = [
            {
                "type": "log", "source": "mcp", "content": r.get("message", ""),
                "timestamp": r.get("timestamp"), "level": r.get("level", "INFO"),
                "metadata": {
                    "service": r.get("service"), "error_type": r.get("error_type"),
                    "status_code": r.get("status_code"), "latency_ms": r.get("latency_ms"),
                    "upstream": r.get("upstream"),
                },
            }
            for r in records
        ]

        # With topology_graph in result → source should be "topology-agent"
        topo_data = json.load(open(PROJECT_ROOT / "mcp_servers/architecture_mcp/app/system_topology.json"))
        edges = topo_data["diagram"]["content"]["edges"]
        g = {}
        for src, dst in edges:
            g.setdefault(src, []).append(dst)
            g.setdefault(dst, [])
        m = {n: {"type": i.get("type"), "criticality": i.get("criticality")}
             for n, i in topo_data["services"].items()}

        result = await svc.synthesize(
            incident_id="INC-DYN",
            service="api-gateway",
            log_result={"evidence": evidence},
            topology_result={"topology_graph": g, "service_metadata": m},
        )
        check("Source is topology-agent", result["topology_source"] == "topology-agent")
        check("cause_service in candidates",
              result["root_cause_candidates"] and "cause_service" in result["root_cause_candidates"][0])
        check("Top cause_service is user-db",
              result["root_cause_candidates"][0]["cause_service"] == "user-db")

        # Without topology_graph → fallback to architecture-mcp
        result2 = await svc.synthesize(
            incident_id="INC-FB",
            service="api-gateway",
            log_result={"evidence": evidence},
            topology_result={"summary": "no graph"},
        )
        check("Fallback source is architecture-mcp", result2["topology_source"] == "architecture-mcp")

    asyncio.run(_run())


def test_topology_agent_full_graph():
    print("\n=== Test 4: Topology Agent Returns Full Graph ===")

    async def _run():
        from agents.topology_agent.skills import TopologyAnalysisService

        svc = TopologyAnalysisService()
        result = await svc.analyze(service="api-gateway", suspected_downstream="auth-service")
        check("topology_graph present", "topology_graph" in result)
        check("service_metadata present", "service_metadata" in result)
        check("9 nodes in graph", len(result["topology_graph"]) == 9)
        check("propagation_path computed", len(result.get("propagation_path", [])) > 0)

    asyncio.run(_run())


def test_verifier_dynamic_services():
    print("\n=== Test 5: Verifier Dynamic Known Services ===")
    from agents.verifier_agent.service import _extract_known_services

    known = _extract_known_services({
        "topology_agent": {"related_services": ["custom-svc-x"], "blast_radius": ["custom-svc-y"]},
    })
    check("Includes custom services", "custom-svc-x" in known and "custom-svc-y" in known)
    check("Also includes defaults", "user-db" in known and "api-gateway" in known)

    known2 = _extract_known_services({})
    check("Empty input → defaults", len(known2) > 0 and "api-gateway" in known2)


def test_evaluation_function():
    print("\n=== Test 6: Evaluation Function ===")
    sys.path.insert(0, str(PROJECT_ROOT))

    # Import evaluate_result directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_experiment", PROJECT_ROOT / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = {
        "root_cause_candidates": [
            {"cause": "user-db (database): conn pool", "cause_service": "user-db", "confidence": 0.85}
        ],
        "verification": {"verdict": "accepted"},
        "impact_analysis": {"propagation_path": ["user-db", "auth-service", "api-gateway"]},
    }
    scenario = {
        "ground_truth_root_cause": "user-db",
        "ground_truth_path": ["user-db", "auth-service", "api-gateway"],
    }
    ev = mod.evaluate_result(result, scenario, elapsed=5.0)
    check("AC@1 correct", ev["ac_at_1"] is True)
    check("AC@3 correct", ev["ac_at_3"] is True)
    check("Path accuracy correct", ev["path_accuracy"] is True)

    # False positive scenario
    result_fp = {
        "root_cause_candidates": [{"cause": "minor spike", "confidence": 0.3}],
        "verification": {"verdict": "rejected"},
    }
    scenario_fp = {"ground_truth_root_cause": None, "ground_truth_path": []}
    ev_fp = mod.evaluate_result(result_fp, scenario_fp, elapsed=3.0)
    check("FP handled for null root cause", ev_fp["fp_handled"] is True)


def test_system_variants_defined():
    print("\n=== Test 7: System Variants in run_experiment ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_experiment", PROJECT_ROOT / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    check("ours defined", "ours" in mod.SYSTEMS)
    check("b1 defined", "b1" in mod.SYSTEMS)
    check("b2 defined", "b2" in mod.SYSTEMS)
    check("b3 defined", "b3" in mod.SYSTEMS)
    check("8 synthetic scenarios", all(f"s{i}" in mod.SCENARIOS for i in range(1, 9)))
    check("2 case studies", "case1" in mod.SCENARIOS and "case2" in mod.SCENARIOS)


if __name__ == "__main__":
    test_tcb_rca_algorithm()
    test_dynamic_topology_resolution()
    test_rca_service_dynamic()
    test_topology_agent_full_graph()
    test_verifier_dynamic_services()
    test_evaluation_function()
    test_system_variants_defined()

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {PASS_COUNT} PASS, {FAIL_COUNT} FAIL")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL_COUNT == 0 else 1)
