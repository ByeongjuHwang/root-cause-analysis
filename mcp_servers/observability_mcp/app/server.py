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


def main():
    # 논문 1차 PoC는 로컬 실행이 쉬운 stdio로 시작
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()