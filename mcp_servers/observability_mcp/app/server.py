import sys
import logging
from typing import Optional, Tuple

from mcp.server.fastmcp import FastMCP

from .repository import search_logs as repo_search_logs
from .repository import get_error_summary as repo_get_error_summary
from .repository import get_trace_logs as repo_get_trace_logs
from .metric_repository import get_metric_summary as repo_get_metric_summary
from .metric_repository import get_latency_summary as repo_get_latency_summary
from .metric_repository import (
    get_retry_timeout_summary as repo_get_retry_timeout_summary,
)


# STDIO transport에서는 stdout에 임의 로그를 쓰면 JSON-RPC가 깨질 수 있으므로 stderr 로깅 사용
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

mcp = FastMCP("observability-mcp")


@mcp.resource("resource://incident-schema", mime_type="application/json")
def incident_schema() -> dict:
    """Return the incident request schema used by the RCA system."""
    return {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string"},
            "service": {"type": "string"},
            "time_range": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"}
                },
                "required": ["start", "end"]
            },
            "symptom": {"type": "string"},
            "trace_id": {"type": ["string", "null"]}
        },
        "required": ["incident_id", "service", "time_range", "symptom"]
    }


@mcp.resource("resource://sample-log-fields", mime_type="application/json")
def sample_log_fields() -> dict:
    """Return the supported log fields for analysis."""
    return {
        "fields": [
            "timestamp",
            "service",
            "level",
            "trace_id",
            "message",
            "upstream",
            "status_code",
            "latency_ms",
            "error_type"
        ]
    }


@mcp.tool()
def search_logs(service: str, start: str, end: str, keyword: Optional[str] = None) -> dict:
    """
    Search logs for a target service within a time range.

    Args:
        service: Target microservice name
        start: ISO-8601 start timestamp
        end: ISO-8601 end timestamp
        keyword: Optional keyword filter
    """
    logging.info("search_logs called: service=%s start=%s end=%s keyword=%s", service, start, end, keyword)

    rows = repo_search_logs(service=service, start=start, end=end, keyword=keyword)
    return {
        "service": service,
        "count": len(rows),
        "logs": [r.model_dump() for r in rows[:100]]
    }


@mcp.tool()
def get_error_summary(service: str, start: str, end: str) -> dict:
    """
    Summarize error patterns for a target service in a time window.

    Args:
        service: Target microservice name
        start: ISO-8601 start timestamp
        end: ISO-8601 end timestamp
    """
    logging.info("get_error_summary called: service=%s start=%s end=%s", service, start, end)

    summary = repo_get_error_summary(service=service, start=start, end=end)
    return summary.model_dump()


@mcp.tool()
def get_trace_logs(trace_id: str) -> dict:
    """
    Retrieve all logs associated with a trace_id.

    Args:
        trace_id: Distributed trace identifier
    """
    logging.info("get_trace_logs called: trace_id=%s", trace_id)

    rows = repo_get_trace_logs(trace_id=trace_id)
    return {
        "trace_id": trace_id,
        "count": len(rows),
        "logs": [r.model_dump() for r in rows[:200]]
    }


# =============================================================================
# v8: Metric tools (RCAEval wide-format metrics.csv)
# =============================================================================
# These three tools expose the metric feature extraction added in
# metric_repository.py. All three accept an optional baseline_start/baseline_end
# pair — when provided, statistics are compared between the named baseline
# window and the [start, end] incident window; when omitted, a 50/50 split is
# used as a fallback.
#
# Return schemas are designed to be LLM-friendly: small, flat, numeric.
# A `has_data` field lets the LLM/caller detect missing metrics gracefully.
#
# Note on metrics_file: in this deployment we pass the CSV path via the
# OBSERVABILITY_METRICS_FILE environment variable (set per-case by the
# orchestrator). Callers may also pass `metrics_file` explicitly.


def _parse_baseline_arg(
    baseline_start: Optional[str],
    baseline_end: Optional[str],
) -> Optional[Tuple[str, str]]:
    """Fold the separate start/end args into the tuple shape expected by
    metric_repository. Both must be present for the tuple to be built."""
    if baseline_start and baseline_end:
        return (baseline_start, baseline_end)
    return None


@mcp.tool()
def get_metric_summary(
    service: str,
    start: str,
    end: str,
    metrics_file: Optional[str] = None,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
) -> dict:
    """
    Summarise CPU and memory metrics for a service in the incident window.

    This is the primary signal for CPU/memory fault detection. The
    cpu_spike_zscore field is computed as
    (incident_max - baseline_mean) / baseline_stddev and is typically
    >20 for a real CPU fault (vs single-digit for noise).

    Args:
        service: Target microservice name (must match column prefix in CSV)
        start: ISO-8601 incident-window start
        end:   ISO-8601 incident-window end
        metrics_file: optional path; env OBSERVABILITY_METRICS_FILE is fallback
        baseline_start: optional ISO-8601 baseline-window start
        baseline_end:   optional ISO-8601 baseline-window end
    """
    logging.info(
        "get_metric_summary: service=%s start=%s end=%s baseline=%s~%s",
        service, start, end, baseline_start, baseline_end,
    )
    return repo_get_metric_summary(
        service=service, start=start, end=end,
        metrics_file=metrics_file,
        baseline_range=_parse_baseline_arg(baseline_start, baseline_end),
    )


@mcp.tool()
def get_latency_summary(
    service: str,
    start: str,
    end: str,
    metrics_file: Optional[str] = None,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
) -> dict:
    """
    Summarise Istio p50/p95/p99 latencies for a service in the incident window.

    Values are reported in MILLISECONDS. The `p95_delta_ms` and `p99_delta_ms`
    fields show the shift from baseline; large positive values (tens to
    hundreds of ms) indicate a delay fault.

    Args: see get_metric_summary.
    """
    logging.info(
        "get_latency_summary: service=%s start=%s end=%s baseline=%s~%s",
        service, start, end, baseline_start, baseline_end,
    )
    return repo_get_latency_summary(
        service=service, start=start, end=end,
        metrics_file=metrics_file,
        baseline_range=_parse_baseline_arg(baseline_start, baseline_end),
    )


@mcp.tool()
def get_retry_timeout_summary(
    service: str,
    start: str,
    end: str,
    metrics_file: Optional[str] = None,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
) -> dict:
    """
    Summarise error/timeout/drop indicators from Istio + network metrics.

    Returns Istio error_delta / request_delta / error_rate plus per-direction
    network packet drop and error totals (rx_drop_delta, tx_drop_delta,
    rx_errors_delta, tx_errors_delta) and the sockets gauge.

    Packet drops are the strongest signal for a 'loss' fault; non-zero
    sockets spike or network errors correlate with socket faults.

    Args: see get_metric_summary.
    """
    logging.info(
        "get_retry_timeout_summary: service=%s start=%s end=%s baseline=%s~%s",
        service, start, end, baseline_start, baseline_end,
    )
    return repo_get_retry_timeout_summary(
        service=service, start=start, end=end,
        metrics_file=metrics_file,
        baseline_range=_parse_baseline_arg(baseline_start, baseline_end),
    )


# =============================================================================
# Phase 2: Evidence-aware tools
# =============================================================================
# These four tools return structured EvidenceUnit collections rather than raw
# data. They are additive — all pre-existing tools remain available.
#
# Evidence tools let Agents (Phase 3+) receive already-interpreted signals:
# instead of parsing log stats or metric summaries themselves, they get a list
# of EvidenceUnits with explicit `modality`, `anomaly_type`, `severity`, and
# `observation` fields.
#
# The underlying conversion happens in evidence_tools.py which reuses the
# existing repository functions for raw data collection.

from .evidence_tools import (
    get_log_evidence_payload,
    get_metric_evidence_payload,
    get_topology_evidence_payload,
    get_evidence_collection_payload,
)


@mcp.tool()
def get_log_evidence(
    start: str,
    end: str,
    log_file: Optional[str] = None,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
    incident_start: Optional[str] = None,
    incident_end: Optional[str] = None,
    focus_services: Optional[list] = None,
) -> dict:
    """
    Return log-modality EvidenceUnits for all services within [start, end].

    Unlike search_logs (which returns raw rows) or get_service_statistics
    (which returns numeric aggregates), this tool returns a list of structured
    EvidenceUnit dicts — each carrying modality='log', an anomaly_type
    ('volume_shift' | 'error_spike' | 'keyword_distress' | 'dependency_failure'),
    a severity score in [0,1], and the numeric observation behind it.

    Dual-window usage: pass baseline_start/baseline_end for the comparison
    baseline and incident_start/incident_end for the incident window. The
    severity normalisation and low-volume artifact suppression match those
    in evidence_factory.

    Args:
        start, end: ISO-8601 outer window (used for search + activity filter)
        log_file: optional log path override
        baseline_start, baseline_end: optional dual-window baseline
        incident_start, incident_end: optional dual-window incident
        focus_services: if given, only produce evidence for these services
    """
    logging.info(
        "get_log_evidence: window=[%s, %s] baseline=%s~%s incident=%s~%s focus=%s",
        start, end, baseline_start, baseline_end,
        incident_start, incident_end, focus_services,
    )
    return get_log_evidence_payload(
        start=start, end=end, log_file=log_file,
        baseline_start=baseline_start, baseline_end=baseline_end,
        incident_start=incident_start, incident_end=incident_end,
        focus_services=focus_services,
    )


@mcp.tool()
def get_metric_evidence(
    start: str,
    end: str,
    metrics_file: Optional[str] = None,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
    incident_start: Optional[str] = None,
    incident_end: Optional[str] = None,
    focus_services: Optional[list] = None,
) -> dict:
    """
    Return metric-modality EvidenceUnits from Prometheus/Istio metrics.

    Produces up to three unit types per service:
      - resource_saturation: CPU spike z-score or mem_jump ratio
      - latency_degradation: p95 or p99 delta
      - network_degradation: packet drops or Istio error delta

    Severity uses the normalised [0,1] mapping from evidence_factory:
    e.g. CPU z=20 → sev≈0.46, z=289 → sev≈1.0.

    Args: see get_metric_summary plus focus_services.
    """
    logging.info(
        "get_metric_evidence: window=[%s, %s] baseline=%s~%s focus=%s",
        start, end, baseline_start, baseline_end, focus_services,
    )
    return get_metric_evidence_payload(
        start=start, end=end, metrics_file=metrics_file,
        baseline_start=baseline_start, baseline_end=baseline_end,
        incident_start=incident_start, incident_end=incident_end,
        focus_services=focus_services,
    )


@mcp.tool()
def get_topology_evidence(
    symptom_service: str,
    candidate_services: list,
    path: list,
    start: str,
    end: str,
) -> dict:
    """
    Return topology-modality EvidenceUnits representing structural proximity.

    Per v7 H4 principle, topology evidence has severity capped at 0.5 — this
    is structural guessing, not observation. A candidate supported only by
    topology evidence should not be chosen as final Top-1 by the RCA Agent.

    Args:
        symptom_service: the observed failing service
        candidate_services: services to evaluate for topology proximity
        path: the service-call path (e.g. ['api-gateway', 'auth-service', 'user-db'])
        start, end: the window for tagging the evidence (content itself is time-invariant)
    """
    logging.info(
        "get_topology_evidence: symptom=%s candidates=%d path_len=%d",
        symptom_service, len(candidate_services), len(path),
    )
    return get_topology_evidence_payload(
        symptom_service=symptom_service,
        candidate_services=candidate_services,
        path=path,
        start=start, end=end,
    )


@mcp.tool()
def get_evidence_collection(
    start: str,
    end: str,
    log_file: Optional[str] = None,
    metrics_file: Optional[str] = None,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
    incident_start: Optional[str] = None,
    incident_end: Optional[str] = None,
    focus_services: Optional[list] = None,
    symptom_service: Optional[str] = None,
    topology_path: Optional[list] = None,
    candidate_services: Optional[list] = None,
) -> dict:
    """
    Unified endpoint — return a merged EvidenceCollection across log + metric
    (+ topology if the relevant args are provided).

    This is the primary tool for evidence-aware Agents: one call returns the
    complete multi-modality evidence for an incident window, already
    deduplicated by evidence_id.

    The topology slice requires all three of symptom_service, topology_path,
    and candidate_services to be provided; otherwise it is skipped.

    Args: union of get_log_evidence and get_metric_evidence, plus the
    topology trio (symptom_service, topology_path, candidate_services).
    """
    logging.info(
        "get_evidence_collection: window=[%s, %s] baseline=%s~%s topology_given=%s",
        start, end, baseline_start, baseline_end,
        bool(symptom_service and topology_path and candidate_services),
    )
    return get_evidence_collection_payload(
        start=start, end=end,
        log_file=log_file, metrics_file=metrics_file,
        baseline_start=baseline_start, baseline_end=baseline_end,
        incident_start=incident_start, incident_end=incident_end,
        focus_services=focus_services,
        symptom_service=symptom_service,
        topology_path=topology_path,
        candidate_services=candidate_services,
    )


def main():
    # 논문 1차 PoC는 로컬 실행이 쉬운 stdio로 시작
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()