"""
Evidence-Aware RCA — Response Builder (Phase 4b).

Helpers to convert each Agent's legacy result dict into a structured
AgentResponse. This is the foundation for Phase 4c (Verifier using
AgentResponse) and Phase 4d (Orchestrator adaptive execution).

Design principles
-----------------
1. Non-invasive: Agent business logic is unchanged. We only add a conversion
   step near the end of analyze() that packages existing data into the
   AgentResponse schema.

2. Best-effort: legacy dicts are ad-hoc and may lack fields. Builders extract
   what's available and mark gaps via missing_evidence so downstream
   components (Orchestrator) can see what's missing.

3. Dual output pattern: the legacy dict is returned unchanged, but we attach
   `_agent_response` as a JSON-serialisable dict inside it. Pydantic models
   cannot be embedded in a dict that downstream code sends over HTTP, so we
   always dump to dict form. Downstream can reconstitute with
   AgentResponse.model_validate(dict).

4. Feature flag: callers check os.getenv("A2A_CONTRACT_MODE"). These builder
   functions don't read env themselves — the caller decides whether to
   invoke them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .a2a_contract import (
    AgentResponse,
    Candidate,
    ConsistencyChecks,
    FailureMode,
)
from .evidence import EvidenceCollection, EvidenceUnit, completeness_score


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def _restore_evidence_collection(payload: Optional[Dict[str, Any]]) -> EvidenceCollection:
    """Reconstitute an EvidenceCollection from its JSON-shaped payload.

    Log Agent stores evidence_collection in its legacy_result as the
    serialised dict from get_evidence_collection_payload(). When we need a
    structured EvidenceCollection for downstream reasoning, rebuild it.
    Failures are tolerated — a missing payload just yields an empty
    collection, not an error.
    """
    if not payload or not payload.get("units"):
        return EvidenceCollection()
    units: List[EvidenceUnit] = []
    for u in payload["units"]:
        try:
            units.append(EvidenceUnit.model_validate(u))
        except Exception:
            # Corrupt unit — skip rather than fail the entire response
            continue
    return EvidenceCollection(units=units)


def _clip_confidence(x: Any, default: float = 0.0) -> float:
    """Defensively coerce any numeric-ish input to [0, 1]."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN guard
        return default
    return max(0.0, min(1.0, v))


def _infer_failure_mode(anomaly_type: Optional[str], hint: Optional[str] = None) -> Optional[FailureMode]:
    """Rough heuristic: observed anomaly_type → likely failure_mode.

    This is a best-effort default — LLM-driven analysis in Phase 4c will
    refine it. We keep the mapping conservative: only obvious pairs get a
    non-null failure_mode; ambiguous cases get None so we don't inject
    misleading labels.
    """
    if not anomaly_type:
        return None
    mapping: Dict[str, FailureMode] = {
        "resource_saturation": "resource_exhaustion",
        "latency_degradation": "dependency_timeout",
        "error_spike": "cascading_failure",
        "network_degradation": "network_partition",
        "volume_shift": "partial_outage",
        "keyword_distress": "retry_storm",
        "dependency_failure": "cascading_failure",
    }
    return mapping.get(anomaly_type)


# ---------------------------------------------------------------------------
# Log Agent → AgentResponse
# ---------------------------------------------------------------------------

def build_log_agent_response(
    legacy_result: Dict[str, Any],
    request_id: str,
) -> AgentResponse:
    """Package Log Agent's legacy dict into a structured AgentResponse.

    Extracted fields
    ----------------
    suspected_downstream   → main candidate service
    confidence             → Candidate.confidence
    anomalous_services     → extra candidates (rank 2+)
    evidence_collection    → AgentResponse.evidence_collection
    service_statistics     → assumptions ("dual_window baseline=480s")
    referenced_upstreams   → extra assumptions

    Populated fields
    ----------------
    completeness_score: computed from evidence_collection vs candidates.
    missing_evidence:   derived from which modalities are absent.

    The function is total: any field in the legacy dict that is missing or
    malformed just yields a weaker AgentResponse rather than raising.
    """
    incident_id = request_id or legacy_result.get("incident_id") or "UNKNOWN"

    # --- Reconstruct the EvidenceCollection (may be absent if Phase 3b flag off)
    evidence_payload = legacy_result.get("evidence_collection")
    collection = _restore_evidence_collection(evidence_payload)

    # --- Primary candidate = suspected_downstream
    primary_svc = (legacy_result.get("suspected_downstream") or "").strip()
    primary_conf = _clip_confidence(legacy_result.get("confidence"))

    # assumptions derived from the running config
    assumptions: List[str] = []
    stats = legacy_result.get("service_statistics") or {}
    if stats:
        win = stats.get("window") or {}
        b_sec = win.get("baseline_seconds")
        i_sec = win.get("incident_seconds")
        if b_sec or i_sec:
            assumptions.append(
                f"dual_window: baseline={b_sec}s, incident={i_sec}s"
            )
        mode = stats.get("mode")
        if mode:
            assumptions.append(f"stats_mode={mode}")

    referenced_upstreams = legacy_result.get("referenced_upstreams") or {}
    if referenced_upstreams:
        assumptions.append(
            f"upstream_ref_resolution: {sum(len(v) for v in referenced_upstreams.values())} references"
        )

    # missing_evidence derived from modality gaps
    modalities_present = set(collection.modalities_present())
    missing_evidence: List[str] = []
    if "log" not in modalities_present:
        missing_evidence.append("log_evidence_units")
    if "metric" not in modalities_present:
        missing_evidence.append("metric_evidence_units")
    if "trace" not in modalities_present:
        missing_evidence.append("trace_evidence_units")
    # Note: topology evidence typically requires upstream agent input

    # --- Primary Candidate ---
    candidates: List[Candidate] = []
    if primary_svc:
        # Supporting evidence = evidence IDs for this service
        supporting_ids = [u.evidence_id for u in collection.by_service(primary_svc)]

        # Failure mode: pick from strongest evidence on this service
        strongest = collection.strongest_by_service(primary_svc)
        failure_mode = _infer_failure_mode(
            strongest.anomaly_type if strongest else None
        )

        candidates.append(Candidate(
            service=primary_svc,
            confidence=primary_conf,
            supporting_evidence=supporting_ids,
            assumptions=list(assumptions),
            missing_evidence=list(missing_evidence),
            reasoning=(legacy_result.get("hypothesis") or "")[:300],
            failure_mode=failure_mode,
        ))

    # --- Secondary candidates from anomalous_services (if any) ---
    anomalous = legacy_result.get("anomalous_services") or []
    for svc in anomalous:
        if not svc or svc == primary_svc:
            continue
        if any(c.service == svc for c in candidates):
            continue
        # Derive a secondary confidence — generally lower than primary.
        # If we have evidence for this service, use its severity as a
        # proxy; otherwise halve the primary's confidence.
        strongest = collection.strongest_by_service(svc)
        if strongest:
            conf2 = strongest.severity
        else:
            conf2 = max(0.0, primary_conf * 0.5)
        supporting_ids = [u.evidence_id for u in collection.by_service(svc)]
        failure_mode = _infer_failure_mode(
            strongest.anomaly_type if strongest else None
        )
        candidates.append(Candidate(
            service=svc,
            confidence=conf2,
            supporting_evidence=supporting_ids,
            assumptions=list(assumptions),
            missing_evidence=list(missing_evidence),
            reasoning=f"secondary candidate (anomalous_services rank)",
            failure_mode=failure_mode,
        ))

    # --- Completeness score ---
    cand_services = [c.service for c in candidates]
    score = (
        completeness_score(collection, cand_services)
        if cand_services else 0.0
    )

    return AgentResponse(
        agent_name="log_agent",
        request_id=incident_id,
        candidates=candidates,
        evidence_collection=collection,
        completeness_score=score,
        consistency_checks=None,
        recommended_next_actions=[],
        reasoning=(legacy_result.get("llm_reasoning") or "")[:600],
    )


# ---------------------------------------------------------------------------
# Topology Agent → AgentResponse
# ---------------------------------------------------------------------------

def build_topology_agent_response(
    legacy_result: Dict[str, Any],
    request_id: str,
) -> AgentResponse:
    """Package Topology Agent's legacy dict into AgentResponse.

    Topology Agent typically returns:
      - candidates (list of services ranked by structural proximity)
      - related_services
      - propagation_path

    Topology is a "structural guess" source, not an observational one.
    We translate candidates directly but mark each with failure_mode=None
    (topology cannot determine HOW without observational evidence).
    """
    incident_id = request_id or legacy_result.get("incident_id") or "UNKNOWN"

    candidates_in = legacy_result.get("candidates") or []
    if not candidates_in:
        # Some topology agents use "suspected_downstream" as a singleton
        fallback = legacy_result.get("suspected_downstream")
        if fallback:
            candidates_in = [{"service": fallback, "confidence": 0.5}]

    candidates: List[Candidate] = []
    for idx, c in enumerate(candidates_in):
        if isinstance(c, dict):
            svc = c.get("service") or c.get("cause_service") or ""
            conf = _clip_confidence(c.get("confidence"), default=0.3)
        elif isinstance(c, str):
            svc, conf = c, max(0.1, 0.5 - idx * 0.1)
        else:
            continue
        svc = svc.strip()
        if not svc:
            continue
        candidates.append(Candidate(
            service=svc,
            confidence=conf,
            supporting_evidence=[],  # Topology doesn't produce evidence IDs directly
            assumptions=["topology_graph_available"],
            missing_evidence=["observational_evidence_log_or_metric"],
            reasoning=(
                c.get("reasoning", "") if isinstance(c, dict) else ""
            )[:200],
            failure_mode=None,  # topology cannot determine failure mechanism
        ))

    # Topology has no direct evidence collection; completeness is low by design
    return AgentResponse(
        agent_name="topology_agent",
        request_id=incident_id,
        candidates=candidates,
        evidence_collection=EvidenceCollection(),  # topology evidence produced elsewhere
        completeness_score=0.0,  # structural guess only
        consistency_checks=None,
        recommended_next_actions=[
            "fetch_log_evidence_for_top_candidates",
            "fetch_metric_evidence_for_top_candidates",
        ],
        reasoning=(legacy_result.get("reasoning") or "")[:400],
    )


# ---------------------------------------------------------------------------
# RCA Agent → AgentResponse
# ---------------------------------------------------------------------------

def build_rca_agent_response(
    legacy_result: Dict[str, Any],
    request_id: str,
    upstream_log_response: Optional[AgentResponse] = None,
) -> AgentResponse:
    """Package RCA Agent's legacy dict into AgentResponse.

    The RCA Agent synthesises Log + Topology findings. We:
      - Map root_cause_candidates → Candidate[]
      - Inherit evidence_collection from upstream Log Agent (if given)
      - For each candidate, match supporting evidence via service name
      - Mark failure_mode from LLM reasoning keywords (heuristic)

    upstream_log_response
    ---------------------
    If the caller can pass the Log Agent's AgentResponse (to inherit its
    evidence_collection), supporting_evidence IDs for RCA candidates can be
    resolved. Otherwise we fall back to empty supporting_evidence.
    """
    incident_id = request_id or legacy_result.get("incident_id") or "UNKNOWN"

    # Evidence collection: inherit from upstream if available; otherwise empty
    if upstream_log_response is not None:
        collection = upstream_log_response.evidence_collection
    else:
        collection = EvidenceCollection()

    raw_candidates = legacy_result.get("root_cause_candidates") or []
    overall_conf = _clip_confidence(legacy_result.get("overall_confidence"))

    # Assumptions extracted from the synthesis
    assumptions: List[str] = []
    convergence = legacy_result.get("evidence_convergence")
    if convergence:
        assumptions.append(f"evidence_convergence={convergence}")
    if legacy_result.get("algorithm"):
        assumptions.append(f"algorithm={legacy_result['algorithm']}")

    # Missing evidence: derived from modalities absent from collection
    modalities_present = set(collection.modalities_present())
    missing: List[str] = []
    if "log" not in modalities_present:
        missing.append("log_evidence_units")
    if "metric" not in modalities_present:
        missing.append("metric_evidence_units")
    if not legacy_result.get("propagation_path"):
        missing.append("topology_propagation_path")

    candidates: List[Candidate] = []
    for c in raw_candidates:
        if not isinstance(c, dict):
            continue
        svc = (c.get("cause_service") or "").strip()
        if not svc:
            continue
        conf = _clip_confidence(c.get("confidence"))
        reasoning = (c.get("reasoning") or "")[:300]
        propagation = c.get("propagation_path") or legacy_result.get("propagation_path")

        # supporting evidence: any evidence_id in collection that mentions this service
        supporting_ids = [u.evidence_id for u in collection.by_service(svc)]

        # Heuristic: pick failure mode from anomaly type of strongest evidence
        strongest = collection.strongest_by_service(svc)
        failure_mode = _infer_failure_mode(
            strongest.anomaly_type if strongest else None
        )
        # Override with LLM reasoning keywords if strongly hinted
        lower = reasoning.lower()
        if "resource" in lower or "cpu" in lower or "memory" in lower:
            failure_mode = failure_mode or "resource_exhaustion"
        elif "cascade" in lower or "propagate" in lower or "upstream" in lower:
            failure_mode = failure_mode or "cascading_failure"
        elif "timeout" in lower:
            failure_mode = failure_mode or "dependency_timeout"

        candidates.append(Candidate(
            service=svc,
            confidence=conf,
            supporting_evidence=supporting_ids,
            assumptions=list(assumptions),
            missing_evidence=list(missing),
            reasoning=reasoning,
            topology_path=propagation if isinstance(propagation, list) else None,
            failure_mode=failure_mode,
        ))

    # Completeness: use the strength of evidence behind the top candidate
    cand_services = [c.service for c in candidates[:3]]
    score = (
        completeness_score(collection, cand_services)
        if cand_services and len(collection) > 0
        else _clip_confidence(overall_conf)  # fallback to LLM's own confidence
    )

    return AgentResponse(
        agent_name="rca_agent",
        request_id=incident_id,
        candidates=candidates,
        evidence_collection=collection,
        completeness_score=score,
        consistency_checks=None,
        recommended_next_actions=[],
        reasoning=(legacy_result.get("llm_synthesis_reasoning") or "")[:600],
    )


# ---------------------------------------------------------------------------
# Verifier Agent → AgentResponse
# ---------------------------------------------------------------------------

def build_verifier_agent_response(
    legacy_result: Dict[str, Any],
    request_id: str,
    upstream_rca_response: Optional[AgentResponse] = None,
    consistency_by_service: Optional[Dict[str, ConsistencyChecks]] = None,
) -> AgentResponse:
    """Package Verifier's legacy dict into a structured AgentResponse.

    Unlike other agents, Verifier always produces consistency_checks at the
    per-candidate level. We aggregate into AgentResponse.consistency_checks
    using the *top candidate's* per-dimension result (common pattern for the
    schema's single-field consistency_checks).

    Parameters
    ----------
    legacy_result
        The dict returned by VerifierService.verify() — contains
        revised_root_cause_candidates, verification_notes, etc.
    request_id
        Incident ID.
    upstream_rca_response
        If provided, Verifier inherits evidence_collection from upstream
        RCA (which in turn inherited from Log Agent). This is how Verifier's
        AgentResponse gets evidence_ids on its candidates.
    consistency_by_service
        Per-service ConsistencyChecks computed by the Verifier during its
        own pipeline. Key is lowercase service name. This is the core
        evidence-aware result — we cannot recompute it here (needs all the
        per-candidate signals); Verifier computes and passes in.
    """
    incident_id = request_id or legacy_result.get("incident_id") or "UNKNOWN"

    # Evidence collection — inherit from upstream RCA if present
    if upstream_rca_response is not None:
        collection = upstream_rca_response.evidence_collection
    else:
        collection = EvidenceCollection()

    raw_candidates = legacy_result.get("revised_root_cause_candidates") or []
    consistency_by_service = consistency_by_service or {}

    # Assumptions — Verifier's own config
    assumptions: List[str] = [
        "verifier_mode=hybrid (5-signal + consistency_checks)",
    ]
    if legacy_result.get("verdict"):
        assumptions.append(f"verdict={legacy_result['verdict']}")

    # Rejection info: a meta-hint for downstream
    rejected_count = legacy_result.get("rejected_candidates_count", 0)
    if rejected_count:
        assumptions.append(f"rejected_during_verify={rejected_count}")

    # missing_evidence = modalities absent from the inherited collection
    modalities_present = set(collection.modalities_present())
    missing: List[str] = []
    if collection and len(collection) > 0:
        if "log" not in modalities_present:
            missing.append("log_evidence_units")
        if "metric" not in modalities_present:
            missing.append("metric_evidence_units")
        if "trace" not in modalities_present:
            missing.append("trace_evidence_units")

    candidates: List[Candidate] = []
    for c in raw_candidates:
        if not isinstance(c, dict):
            continue
        svc = (c.get("cause_service") or "").strip()
        if not svc:
            # Free-text candidate — skip for AgentResponse (it needs structured service)
            continue
        conf = _clip_confidence(c.get("confidence"))
        reasoning = (c.get("reasoning") or "")[:300]

        supporting_ids = [u.evidence_id for u in collection.by_service(svc)]

        # Failure mode — inherit from upstream if available
        failure_mode: Optional[FailureMode] = None
        if upstream_rca_response is not None:
            for uc in upstream_rca_response.candidates:
                if uc.service.lower() == svc.lower():
                    failure_mode = uc.failure_mode
                    break
        if failure_mode is None:
            strongest = collection.strongest_by_service(svc)
            failure_mode = _infer_failure_mode(
                strongest.anomaly_type if strongest else None
            )

        # Pull per-candidate ConsistencyChecks if the Verifier computed one
        per_cand_cc = consistency_by_service.get(svc.lower())

        # Propagation path if present (v8 signals field carries topology)
        topology_path = None
        if isinstance(c.get("propagation_path"), list):
            topology_path = c["propagation_path"]

        # Build per-candidate assumptions — include consistency status as hints
        cand_assumps = list(assumptions)
        if per_cand_cc is not None:
            dims = []
            for dim in ("temporal", "topological", "modality", "counter_evidence"):
                val = getattr(per_cand_cc, dim)
                if val is True:
                    dims.append(f"{dim}=OK")
                elif val is False:
                    dims.append(f"{dim}=FAIL")
                else:
                    dims.append(f"{dim}=NA")
            cand_assumps.append("consistency: " + ", ".join(dims))

        # Verifier-specific hint: rank after re-scoring
        rank = c.get("rank")
        if rank is not None:
            cand_assumps.append(f"verifier_rank={rank}")

        candidates.append(Candidate(
            service=svc,
            confidence=conf,
            supporting_evidence=supporting_ids,
            assumptions=cand_assumps,
            missing_evidence=list(missing),
            reasoning=reasoning,
            topology_path=topology_path,
            failure_mode=failure_mode,
        ))

    # Top-level ConsistencyChecks — use the top candidate's checks.
    # If Verifier didn't compute any (e.g. all candidates lacked cause_service),
    # leave as None (AgentResponse schema permits).
    top_cc: Optional[ConsistencyChecks] = None
    if candidates:
        top_svc = candidates[0].service.lower()
        top_cc = consistency_by_service.get(top_svc)

    # Completeness uses evidence_collection coverage of the accepted candidates
    cand_services = [c.service for c in candidates[:3]]
    score = (
        completeness_score(collection, cand_services)
        if cand_services and len(collection) > 0
        else _clip_confidence(legacy_result.get("final_confidence"))
    )

    return AgentResponse(
        agent_name="verifier_agent",
        request_id=incident_id,
        candidates=candidates,
        evidence_collection=collection,
        completeness_score=score,
        consistency_checks=top_cc,
        recommended_next_actions=[],
        reasoning=(legacy_result.get("explanation") or "")[:600],
    )


# ---------------------------------------------------------------------------
# Attach helper — write AgentResponse into a legacy dict
# ---------------------------------------------------------------------------

def attach_agent_response(
    legacy_result: Dict[str, Any],
    agent_response: AgentResponse,
    key: str = "_agent_response",
) -> Dict[str, Any]:
    """Embed a serialised AgentResponse into the legacy dict.

    We use `model_dump(mode="json")` (not `model_dump_json()` → string) so
    downstream JSON serialisers (FastAPI, Orchestrator) can still operate
    over the whole dict natively.

    Key prefix `_` signals "internal metadata" — tools that only look at
    business-logic fields won't trip over it.
    """
    try:
        legacy_result[key] = agent_response.model_dump(mode="json")
    except Exception:
        # If for any reason dumping fails, we still preserve the legacy dict
        # — this helper is defensive by design.
        pass
    return legacy_result


__all__ = [
    "attach_agent_response",
    "build_log_agent_response",
    "build_rca_agent_response",
    "build_topology_agent_response",
    "build_verifier_agent_response",
]
