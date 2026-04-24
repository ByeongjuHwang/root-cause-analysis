"""
Log Agent Shadow Mode — Phase 3a.

This module runs the evidence-aware collection **in parallel** with the
existing Log Agent pipeline, for observational purposes only. The shadow
result is written to llm_logs/ but is NOT returned to the orchestrator.

Purpose
-------
We want to verify, using real RCAEval traffic, that the evidence-aware
path produces reasonable results BEFORE we replace the production path
(Phase 3b). Shadow mode lets us:
  - compare the services/candidates the two paths point to
  - compare completeness_score vs the legacy confidence
  - catch evidence-aware crashes in realistic data without breaking live
    RCA pipelines

Guarantees
----------
1. Shadow failures MUST NOT propagate. If anything breaks here, the
   legacy Log Agent result is still returned to the orchestrator.
2. No mutation of shared state. The shadow call passes its own kwargs
   and reads only stateless functions.
3. Output lives under llm_logs/ with a distinguishable filename so it
   won't be confused with the real LLM trace logs.

Design decisions
----------------
- We write JSON (one file per incident), not append to a running log.
  Rationale: easier to diff/inspect individual incidents.
- We compute BOTH legacy-summary AND evidence-aware summary so a
  side-by-side comparison is possible without re-running experiments.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output location — llm_logs/ at the repo root
# ---------------------------------------------------------------------------

def _llm_logs_dir() -> Path:
    """Resolve the llm_logs/ directory, creating it if needed.

    The Log Agent runs from the repo root, so CWD is reliable. If the
    directory doesn't exist we create it (mirrors the existing LLM-trace
    logging pattern).
    """
    d = Path("llm_logs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _shadow_filename(incident_id: Optional[str]) -> Path:
    """Naming: {ts}_log_agent_shadow_{incident_short_or_uuid}.json"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short = (incident_id or uuid.uuid4().hex[:8])[:24].replace("/", "_").replace(":", "_")
    return _llm_logs_dir() / f"{ts}_log_agent_shadow_{short}.json"


# ---------------------------------------------------------------------------
# Summary helpers — compress the verbose collection for human reading
# ---------------------------------------------------------------------------

def _summarise_collection(col_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the large EvidenceCollection payload into a compact summary
    useful for side-by-side inspection.

    Returns a dict with:
      - total_units
      - per_service: {service: {"log_count": N, "metric_count": M,
                                "max_severity": 0.x, "top_anomaly": "..."}}
      - dominant_services: top 3 services by max severity, tie-broken
        by evidence count
    """
    units = col_payload.get("units") or []
    total = len(units)

    per_service: Dict[str, Dict[str, Any]] = {}
    for u in units:
        for svc in u.get("services") or []:
            entry = per_service.setdefault(svc, {
                "log_count": 0, "metric_count": 0, "trace_count": 0,
                "topology_count": 0, "max_severity": 0.0,
                "top_anomaly": None,
            })
            modality = u.get("modality")
            key = f"{modality}_count"
            if key in entry:
                entry[key] += 1
            sev = u.get("severity") or 0.0
            if sev > entry["max_severity"]:
                entry["max_severity"] = sev
                entry["top_anomaly"] = u.get("anomaly_type")

    # Rank by (max_severity desc, evidence_count desc)
    def _rank_key(kv: Tuple[str, Dict[str, Any]]) -> Tuple[float, int]:
        _, info = kv
        ev_count = (
            info["log_count"] + info["metric_count"]
            + info["trace_count"] + info["topology_count"]
        )
        return (-info["max_severity"], -ev_count)

    dominant = sorted(per_service.items(), key=_rank_key)[:3]
    dominant_services = [
        {"service": svc, **info} for svc, info in dominant
    ]

    return {
        "total_units": total,
        "services_covered": col_payload.get("services_covered") or [],
        "modalities_present": col_payload.get("modalities_present") or [],
        "per_service": per_service,
        "dominant_services": dominant_services,
    }


def _compare_legacy_vs_evidence(
    legacy_result: Dict[str, Any],
    evidence_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute structured agreement metrics between the two paths."""
    legacy_suspected = legacy_result.get("suspected_downstream")
    legacy_anomalous = legacy_result.get("anomalous_services") or []

    # Evidence-aware top pick: the dominant service with highest severity.
    evidence_top = None
    dom = evidence_summary.get("dominant_services") or []
    if dom:
        evidence_top = dom[0].get("service")

    evidence_top3 = [d.get("service") for d in dom[:3]]

    return {
        "legacy_top": legacy_suspected,
        "legacy_anomalous": legacy_anomalous[:5],
        "evidence_top": evidence_top,
        "evidence_top3": evidence_top3,
        "top_matches": (
            legacy_suspected is not None
            and evidence_top is not None
            and legacy_suspected == evidence_top
        ),
        "legacy_top_in_evidence_top3": (
            legacy_suspected is not None
            and legacy_suspected in evidence_top3
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point — called from the Log Agent after its regular analyze()
# ---------------------------------------------------------------------------

def run_shadow_evidence_collection(
    legacy_result: Dict[str, Any],
    *,
    incident_id: Optional[str],
    symptom_service: str,
    start: str,
    end: str,
    log_file: Optional[str] = None,
    metrics_file: Optional[str] = None,
    baseline_range: Optional[Tuple[str, str]] = None,
    incident_range: Optional[Tuple[str, str]] = None,
) -> Optional[Path]:
    """Run the evidence-aware path in shadow mode and persist the result.

    Parameters mirror the Log Agent's analyze() for ease of integration.
    The legacy_result is the dict the Log Agent is about to return to the
    orchestrator; we include a compact copy in the shadow JSON so later
    inspection can compare both outputs without cross-referencing files.

    Returns
    -------
    The path to the shadow log file if written, or None if shadow mode is
    disabled or an error occurred. The caller should ignore the return
    value; it exists only for tests.
    """
    # Feature flag — shadow is OFF by default to protect production performance.
    # Phase 3a empirical observation showed that running shadow in parallel
    # with production Log Agent reduces AC@1 by ~10%p (67% vs 78% on the
    # 30-case smoke). Likely cause: repository cache warm-up, timing shifts,
    # or LLM-client shared state. Until we understand and fix that, shadow
    # must be explicitly enabled to run.
    #
    # To enable shadow (for observational runs, paper figures, debugging):
    #   PowerShell: $env:LOG_AGENT_SHADOW_ENABLE = "1"
    #   bash:       export LOG_AGENT_SHADOW_ENABLE=1
    if os.getenv("LOG_AGENT_SHADOW_ENABLE") != "1":
        return None

    t0 = time.time()

    try:
        # Local import: keep skills_llm.py import graph unchanged when
        # shadow is disabled, and delay evidence_tools imports to actual
        # shadow invocation time.
        from mcp_servers.observability_mcp.app.evidence_tools import (
            get_evidence_collection_payload,
        )
    except Exception as exc:
        log.warning("shadow import failed: %s", exc)
        return None

    # Unpack optional ranges (they come as tuples from the existing pipeline)
    baseline_start = baseline_range[0] if baseline_range else None
    baseline_end = baseline_range[1] if baseline_range else None
    incident_start = incident_range[0] if incident_range else None
    incident_end = incident_range[1] if incident_range else None

    # Run evidence-aware collection. Any exception here stays contained.
    try:
        payload = get_evidence_collection_payload(
            start=start, end=end,
            log_file=log_file, metrics_file=metrics_file,
            baseline_start=baseline_start, baseline_end=baseline_end,
            incident_start=incident_start, incident_end=incident_end,
            focus_services=None,
            # Symptom/topology inputs are optional — the shadow pass does
            # not know the full topology path here, so we only pass the
            # symptom service and let topology evidence be skipped when
            # path/candidates are absent. Proper topology evidence will
            # be wired in Phase 3b.
            symptom_service=symptom_service,
            topology_path=None,
            candidate_services=None,
        )
    except Exception as exc:
        log.warning("shadow evidence collection failed: %s", exc)
        payload = {
            "count": 0, "services_covered": [],
            "modalities_present": [], "units": [],
            "error": str(exc),
        }

    evidence_summary = _summarise_collection(payload)

    # Legacy-side compact snapshot
    legacy_compact = {
        "suspected_downstream": legacy_result.get("suspected_downstream"),
        "anomalous_services": (legacy_result.get("anomalous_services") or [])[:10],
        "confidence": legacy_result.get("confidence"),
        "evidence_count": len(legacy_result.get("evidence") or []),
        "hypothesis": (legacy_result.get("hypothesis") or "")[:200],
    }

    comparison = _compare_legacy_vs_evidence(legacy_result, evidence_summary)

    elapsed_ms = int((time.time() - t0) * 1000)

    record = {
        "schema_version": "shadow.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "incident_id": incident_id,
        "symptom_service": symptom_service,
        "window": {"start": start, "end": end},
        "baseline_range": list(baseline_range) if baseline_range else None,
        "incident_range": list(incident_range) if incident_range else None,
        "elapsed_ms": elapsed_ms,

        "legacy": legacy_compact,

        "evidence_aware": {
            "summary": evidence_summary,
            # Full payload is bulky but useful for debugging. Keep it,
            # one file per incident is fine.
            "full_payload": payload,
        },

        "comparison": comparison,
    }

    # Write out
    try:
        out_path = _shadow_filename(incident_id)
        out_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(
            "shadow: wrote %s (units=%d, match=%s, elapsed=%dms)",
            out_path.name, evidence_summary.get("total_units", 0),
            comparison.get("top_matches"), elapsed_ms,
        )
        return out_path
    except Exception as exc:
        log.warning("shadow: failed to write record: %s", exc)
        return None


__all__ = ["run_shadow_evidence_collection"]
