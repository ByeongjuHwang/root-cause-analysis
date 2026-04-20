"""
experiment_core.py — 실험 프레임워크 공통 코어.

단일 진실 원천 (Single Source of Truth):
    - SYSTEMS:     시스템 변형 정의 (ours/b1/b2/b3)
    - SCENARIOS:   합성 시나리오 정의 (s1~s8, case1, case2)
    - evaluate_prediction(): 통일된 평가 함수

이 모듈은 모든 실험 러너가 공유합니다:
    - run_experiment.py   (단일 합성 실험)
    - run_all_experiments.py (일괄 실행)
    - run_rcaeval.py      (RCAEval 공개 벤치마크)
    - run_case_study.py   (실세계 Case Study)

설계 원칙:
    1. 시나리오/시스템 추가 시 이 파일만 수정하면 됨
    2. 평가 지표(AC@1, AC@3, Path Accuracy, FP Handled)는 모든 실험에서 동일 정의
    3. 모든 반환값은 JSON 직렬화 가능
    4. 외부 I/O 없음 (순수 데이터 + 순수 함수)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent


# =============================================================================
# System variants (논문 Table 1)
# =============================================================================

SYSTEMS: Dict[str, Dict[str, Any]] = {
    "ours": {
        "name": "Ours (Multi-Agent + MCP + Verifier)",
        "orchestrator_module": "agents.orchestrator.main",
        "port_base": 18000,
        "uses_verifier": True,
        "uses_mcp": True,
        "is_monolithic": False,
    },
    "b1": {
        "name": "B1 (Monolithic Single-Process)",
        "orchestrator_module": "agents.monolithic.main",
        "port_base": 20000,
        "uses_verifier": False,
        "uses_mcp": False,
        "is_monolithic": True,
    },
    "b2": {
        "name": "B2 (No-MCP Parallel Pipeline)",
        "orchestrator_module": "agents.orchestrator.main_parallel",
        "port_base": 21000,
        "uses_verifier": True,
        "uses_mcp": False,
        "is_monolithic": False,
    },
    "b3": {
        "name": "B3 (No-Verifier Pipeline)",
        "orchestrator_module": "agents.orchestrator.main_no_verifier",
        "port_base": 19000,
        "uses_verifier": False,
        "uses_mcp": True,
        "is_monolithic": False,
    },
}


def resolve_services(system_key: str) -> List[Tuple[str, str, int]]:
    """시스템 변형별 서비스 목록을 반환.

    Returns:
        [(service_name, module_path, port), ...]

    Raises:
        KeyError: 알 수 없는 system_key
    """
    config = SYSTEMS[system_key]
    port_base = config["port_base"]

    if config["is_monolithic"]:
        return [("monolithic", config["orchestrator_module"], port_base)]

    services = [
        ("orchestrator", config["orchestrator_module"], port_base),
        ("log-agent", "agents.log_agent.main", port_base + 1),
        ("topology-agent", "agents.topology_agent.main", port_base + 2),
    ]
    if config["uses_verifier"]:
        services.append(("verifier-agent", "agents.verifier_agent.main", port_base + 3))
    services.append(("rca-agent", "agents.rca_agent.main", port_base + 4))
    return services


def build_env_overrides(system_key: str) -> Dict[str, str]:
    """시스템 변형에 맞는 환경변수 오버라이드.

    각 에이전트가 상대 URL을 통해 서로를 찾을 수 있도록 한다.
    """
    port_base = SYSTEMS[system_key]["port_base"]
    return {
        "LOG_AGENT_URL": f"http://127.0.0.1:{port_base + 1}",
        "TOPOLOGY_AGENT_URL": f"http://127.0.0.1:{port_base + 2}",
        "VERIFIER_AGENT_URL": f"http://127.0.0.1:{port_base + 3}",
        "RCA_AGENT_URL": f"http://127.0.0.1:{port_base + 4}",
    }


# =============================================================================
# Synthetic scenarios (논문 Section: Synthetic Fault Injection)
# =============================================================================

# 모든 시나리오의 공통 시간 윈도우 (로그와 일치)
_DEFAULT_TIME_RANGE = {
    "start": "2026-03-24T13:02:30+09:00",
    "end": "2026-03-24T13:03:20+09:00",
}


def _synthetic_incident(incident_id: str, symptom: str, trace_id: str = "trace-001") -> Dict[str, Any]:
    """Boilerplate reducer for synthetic scenarios."""
    return {
        "incident_id": incident_id,
        "service": "api-gateway",
        "time_range": _DEFAULT_TIME_RANGE,
        "symptom": symptom,
        "trace_id": trace_id,
        "attachments": {"diagram_uri": "arch://system/latest"},
    }


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "s1": {
        "kind": "synthetic",
        "log_file": None,  # uses default sample_logs.jsonl
        "incident_file": "demo_incident.json",
        "description": "S1: DB Connection Pool Exhaustion",
        "ground_truth_root_cause": "user-db",
        "ground_truth_path": ["user-db", "auth-service", "api-gateway"],
    },
    "s2": {
        "kind": "synthetic",
        "log_file": "scenarios/s2_queue_backlog.jsonl",
        "incident": _synthetic_incident(
            "INC-S2-QUEUE", "order placement requests failing with 504 timeout errors"
        ),
        "description": "S2: Worker Crash + Queue Backlog",
        "ground_truth_root_cause": "worker-service",
        "ground_truth_path": ["worker-service", "message-queue", "order-service", "api-gateway"],
    },
    "s3": {
        "kind": "synthetic",
        "log_file": "scenarios/s3_slow_query.jsonl",
        "incident": _synthetic_incident(
            "INC-S3-SLOWQUERY", "product search requests timing out with 504 errors",
            trace_id="trace-003",
        ),
        "description": "S3: Catalog Slow Query",
        "ground_truth_root_cause": "catalog-service",
        "ground_truth_path": ["catalog-service", "api-gateway"],
    },
    "s4": {
        "kind": "synthetic",
        "log_file": "scenarios/s4_disk_full.jsonl",
        "incident": _synthetic_incident(
            "INC-S4-DISKFULL", "checkout requests failing with 500 errors from order-service"
        ),
        "description": "S4: DB Disk Full (Fan-out)",
        "ground_truth_root_cause": "order-db",
        "ground_truth_path": ["order-db", "order-service", "api-gateway"],
    },
    "s5": {
        "kind": "synthetic",
        "log_file": "scenarios/s5_false_positive.jsonl",
        "incident": _synthetic_incident(
            "INC-S5-FALSEPOS", "brief latency spike observed on api-gateway, self-recovered"
        ),
        "description": "S5: GC Pause (False Positive)",
        "ground_truth_root_cause": None,  # no real root cause
        "ground_truth_path": [],
    },
    "s6": {
        "kind": "synthetic",
        "log_file": "scenarios/s6_noisy_db.jsonl",
        "incident": _synthetic_incident(
            "INC-S6-NOISY", "login requests failing with 502, multiple services reporting errors simultaneously"
        ),
        "description": "S6: Noisy DB Fault (misleading errors in unrelated services)",
        "ground_truth_root_cause": "user-db",
        "ground_truth_path": ["user-db", "auth-service", "api-gateway"],
    },
    "s7": {
        "kind": "synthetic",
        "log_file": "scenarios/s7_concurrent_faults.jsonl",
        "incident": _synthetic_incident(
            "INC-S7-CONCURRENT", "multiple services failing simultaneously: auth 502, orders 500"
        ),
        "description": "S7: Concurrent Dual Faults (user-db + worker-service)",
        "ground_truth_root_cause": "user-db",
        "ground_truth_root_causes_alt": ["worker-service"],
        "ground_truth_path": ["user-db", "auth-service", "api-gateway"],
        "ground_truth_paths_alt": [
            ["worker-service", "message-queue", "order-service", "api-gateway"],
        ],
    },
    "s8": {
        "kind": "synthetic",
        "log_file": "scenarios/s8_partial_observability.jsonl",
        "incident": _synthetic_incident(
            "INC-S8-PARTIAL", "login requests failing with 502, auth-service reporting database errors"
        ),
        "description": "S8: Partial Observability (root cause logs missing)",
        "ground_truth_root_cause": "user-db",
        "ground_truth_path": ["user-db", "auth-service", "api-gateway"],
    },
    "case1": {
        "kind": "case_study",
        "log_file": "case_study_logs.jsonl",
        "incident_file": "case_study_incident.json",
        "topology_file": "case_study_topology.json",
        "description": "Case Study 1: AppInstallTimeout (Real-world)",
        "ground_truth_root_cause": "device-state-gateway",
        "ground_truth_path": ["device-state-gateway", "app-deployer"],
    },
    "case2": {
        "kind": "case_study",
        "log_file": "case_study_logs.jsonl",
        "incident_file": "case_study_incident_case2.json",
        "topology_file": "case_study_topology.json",
        "description": "Case Study 2: AppUninstallTimeout (Real-world)",
        "ground_truth_root_cause": "device-state-gateway",
        "ground_truth_path": ["device-state-gateway", "app-deployer"],
    },
}


# =============================================================================
# Ground truth matching — single source of truth
# =============================================================================

def _normalize(s: Optional[str]) -> str:
    """Normalize a service name for matching.

    Removes hyphens and underscores and case-folds.
    RCAEval uses 'cartservice' while we might predict 'cart-service' → treat equal.
    """
    if not s:
        return ""
    return s.strip().lower().replace("-", "").replace("_", "")


def candidate_matches_truth(candidate: Dict[str, Any], ground_truth: str) -> bool:
    """Check if an RCA candidate matches a ground truth service name.

    Unified matching rule used by ALL evaluation pipelines
    (synthetic, case study, RCAEval).

    Preference order:
        1. cause_service exact match (LLM/deterministic outputs this field)
        2. cause_service normalized match (handles hyphen/underscore variants)
        3. cause description substring match (legacy text-only output)

    Args:
        candidate: RCA candidate dict with keys 'cause_service' and/or 'cause'
        ground_truth: ground truth service name (any casing/format)

    Returns:
        True if they refer to the same service
    """
    if not ground_truth or not isinstance(candidate, dict):
        return False

    gt_norm = _normalize(ground_truth)

    # 1. Exact-or-normalized cause_service match
    cs = candidate.get("cause_service")
    if cs:
        cs_norm = _normalize(str(cs))
        if cs_norm == gt_norm:
            return True
        # Substring both ways (handles 'user-db' vs 'user-db-primary')
        if gt_norm in cs_norm or cs_norm in gt_norm:
            return True

    # 2. Cause description substring (fallback for text-only baselines)
    cause_text = candidate.get("cause") or ""
    if cause_text and gt_norm in _normalize(cause_text):
        return True

    return False


# =============================================================================
# Unified evaluation function
# =============================================================================

def evaluate_prediction(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate an RCA prediction against ground truth.

    This is the SINGLE evaluation function used across:
        - run_experiment.py (synthetic scenarios)
        - run_rcaeval.py    (public benchmark)
        - run_case_study.py (real-world incidents)

    All metrics are JSON-serializable primitives (bool, float, int, str, None).

    Args:
        prediction: RCA result dict from the orchestrator (FinalRCAResult.dump())
                    Must contain 'root_cause_candidates' (List[Dict])
                    Should contain 'impact_analysis.propagation_path' (List[str])
                    Should contain 'verification.verdict' (str)
        ground_truth: dict with keys
                    - ground_truth_root_cause: str | None
                    - ground_truth_root_causes_alt: List[str] (optional)
                    - ground_truth_path: List[str]
                    - ground_truth_paths_alt: List[List[str]] (optional)
        elapsed_seconds: wall-clock time for the experiment (optional)

    Returns:
        Dict with metrics, all JSON-serializable:
            ac_at_1, ac_at_3, path_accuracy, fp_handled,
            top_confidence, verdict, predicted_top_cause_service, ...
    """
    # --- Extract prediction fields with safe defaults ---
    candidates = prediction.get("root_cause_candidates") or []
    if not isinstance(candidates, list):
        candidates = []

    top_candidate = candidates[0] if candidates else {}
    top_cause_service = top_candidate.get("cause_service") or ""
    top_cause_text = top_candidate.get("cause") or ""
    top_confidence = float(top_candidate.get("confidence") or 0.0)

    verdict = (
        prediction.get("verification", {}).get("verdict")
        or prediction.get("verdict")
        or "unknown"
    )

    # Propagation path — try multiple locations (impact_analysis, top level)
    prop_path = (
        prediction.get("impact_analysis", {}).get("propagation_path")
        or prediction.get("propagation_path")
        or []
    )
    if not isinstance(prop_path, list):
        prop_path = []
    prop_path = [str(p) for p in prop_path]

    # --- Extract ground truth ---
    gt_root = ground_truth.get("ground_truth_root_cause")
    gt_root_alts = ground_truth.get("ground_truth_root_causes_alt") or []
    gt_path = ground_truth.get("ground_truth_path") or []
    gt_paths_alt = ground_truth.get("ground_truth_paths_alt") or []

    all_gt_roots: List[str] = []
    if gt_root:
        all_gt_roots.append(gt_root)
    all_gt_roots.extend(gt_root_alts)

    # --- AC@1: Top-1 candidate matches any acceptable root cause ---
    ac_at_1 = False
    if all_gt_roots and candidates:
        ac_at_1 = any(
            candidate_matches_truth(top_candidate, gt) for gt in all_gt_roots
        )

    # --- AC@3: Any of top-3 candidates matches any acceptable root cause ---
    ac_at_3 = False
    if all_gt_roots:
        for c in candidates[:3]:
            if any(candidate_matches_truth(c, gt) for gt in all_gt_roots):
                ac_at_3 = True
                break

    # --- Path accuracy: propagation path matches any acceptable GT path ---
    path_accuracy: Optional[bool]
    if gt_path:
        all_gt_paths = [gt_path] + list(gt_paths_alt)
        path_accuracy = any(prop_path == list(p) for p in all_gt_paths)
    else:
        path_accuracy = None  # N/A when no GT path

    # --- FP handled: for false-positive scenarios (GT root cause is None) ---
    fp_handled: Optional[bool]
    if gt_root is None and not gt_root_alts:
        fp_handled = (
            top_confidence < 0.5
            or verdict in ("rejected", "weak-evidence", "weak_evidence")
            or not candidates
        )
    else:
        fp_handled = None  # N/A for real-fault scenarios

    # Display-friendly top cause (prefer structured field, fallback to text)
    display_top = top_cause_service or (top_cause_text[:100] if top_cause_text else "(none)")

    result: Dict[str, Any] = {
        "ac_at_1": bool(ac_at_1),
        "ac_at_3": bool(ac_at_3),
        "path_accuracy": path_accuracy,
        "fp_handled": fp_handled,
        "top_confidence": round(top_confidence, 3),
        "verdict": str(verdict),
        "ground_truth_root_cause": gt_root,
        "predicted_top_cause_service": top_cause_service or None,
        "predicted_top_cause_excerpt": display_top[:100] if display_top else None,
        "predicted_path": prop_path,
        "predicted_path_length": len(prop_path),
    }

    if elapsed_seconds is not None:
        result["elapsed_seconds"] = round(float(elapsed_seconds), 2)

    return result


# =============================================================================
# Scenario helpers
# =============================================================================

def scenario_ground_truth(scenario_key: str) -> Dict[str, Any]:
    """Extract ground truth fields from a scenario entry.

    Returns only the GT fields (strips operational fields like log_file).
    """
    if scenario_key not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_key}")
    sc = SCENARIOS[scenario_key]
    return {
        "ground_truth_root_cause": sc.get("ground_truth_root_cause"),
        "ground_truth_root_causes_alt": sc.get("ground_truth_root_causes_alt", []),
        "ground_truth_path": sc.get("ground_truth_path", []),
        "ground_truth_paths_alt": sc.get("ground_truth_paths_alt", []),
    }


def resolve_scenario_inputs(
    scenario_key: str,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve a scenario's input files and incident payload.

    Centralizes the logic for loading scenario inputs so that synthetic,
    case study, and any future scenario kind share the same resolution.

    Returns:
        Dict with keys:
            incident:       Dict — the incident payload
            log_file:       Optional[Path] — override for observability MCP
            topology_file:  Optional[Path] — override for architecture MCP
            kind:           str — 'synthetic' or 'case_study'
    """
    import json as _json

    root = project_root or PROJECT_ROOT
    sc = SCENARIOS[scenario_key]
    kind = sc.get("kind", "synthetic")

    # Incident payload
    if "incident" in sc:
        incident = dict(sc["incident"])  # copy
    else:
        inc_path = root / sc.get("incident_file", "demo_incident.json")
        incident = _json.loads(inc_path.read_text(encoding="utf-8"))

    # Log file
    log_file: Optional[Path] = None
    if sc.get("log_file"):
        log_file = root / sc["log_file"]

    # Topology file
    topology_file: Optional[Path] = None
    if sc.get("topology_file"):
        topology_file = root / sc["topology_file"]
    elif kind == "case_study":
        topology_file = root / "case_study_topology.json"

    # Propagate topology/log file into incident.attachments for orchestrator
    attachments = incident.get("attachments") or {}
    if log_file and "log_file" not in attachments:
        attachments["log_file"] = str(log_file)
    if topology_file and "topology_file" not in attachments:
        attachments["topology_file"] = str(topology_file)
    incident["attachments"] = attachments

    return {
        "incident": incident,
        "log_file": log_file,
        "topology_file": topology_file,
        "kind": kind,
    }
