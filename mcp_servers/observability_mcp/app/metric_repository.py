"""
metric_repository.py — RCAEval-style metric CSV loader and summary functions.

This module provides deterministic metric feature extraction on top of
RCAEval's wide-format `metrics.csv` files:

    time, {service}_{metric_name}, {service}_{metric_name}, ...

Where `time` is a Unix timestamp (seconds) and each service has ~50 metric
columns covering CPU, memory, disk, network, and Istio (latency/errors).

The three public functions below map onto the three MCP tools exposed in
server.py:

    get_metric_summary(service, start, end, metrics_file, baseline_range)
        → CPU, memory, and general resource deltas
    get_latency_summary(service, start, end, metrics_file, baseline_range)
        → Istio p50/p95/p99 during incident vs baseline
    get_retry_timeout_summary(service, start, end, metrics_file, baseline_range)
        → Istio error totals + network drop/reset counts

All three accept an OPTIONAL `baseline_range`; if omitted they fall back to
`[start, mid]` vs `[mid, end]` 50/50 split (same pattern as v6 logs).

Design notes:

  * The wide CSV layout is memory-friendly for small windows (~1400 rows × 418
    cols = ~580K cells = a few MB). We lazily read only the columns we need.

  * Many counter-style metrics (cpu-*-seconds-total, network-*-bytes-total)
    are CUMULATIVE. We compute per-second rates via successive differences
    before summarising.

  * Gauge-style metrics (memory-usage-bytes, latency-95, sockets) are used
    as-is.

  * NaN / missing values are skipped defensively; services with no data in a
    window return `has_activity=False` and neutral values.
"""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Metric name patterns (RCAEval RE2-OB convention)
# =============================================================================

# These are SUFFIXES — the actual column is `{service}_{suffix}`.
# Chosen conservatively; we only read what we summarise.
CPU_USAGE_SUFFIX = "container-cpu-usage-seconds-total"
MEM_USAGE_SUFFIX = "container-memory-usage-bytes"
MEM_WSS_SUFFIX = "container-memory-working-set-bytes"

# Istio latency percentiles (in SECONDS per row — multiply by 1000 for ms)
ISTIO_LATENCY_50 = "istio-latency-50"
ISTIO_LATENCY_95 = "istio-latency-95"
ISTIO_LATENCY_99 = "istio-latency-99"

# Error / request totals (cumulative counters → use deltas)
ISTIO_ERROR_TOTAL = "istio-error-total"
ISTIO_REQUEST_TOTAL = "istio-request-total"

# Network error / drop (cumulative)
NET_RX_ERRORS = "container-network-receive-errors-total"
NET_TX_ERRORS = "container-network-transmit-errors-total"
NET_RX_DROP = "container-network-receive-packets-dropped-total"
NET_TX_DROP = "container-network-transmit-packets-dropped-total"

# Sockets (gauge)
SOCKETS_GAUGE = "container-sockets"


# =============================================================================
# CSV loader
# =============================================================================

def _parse_time_to_unix(t: str) -> Optional[float]:
    """Parse CSV time cell. RCAEval uses Unix seconds as bare number."""
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        # Fallback: ISO 8601
        try:
            return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None


def _resolve_metrics_file(metrics_file: Optional[str] = None) -> Optional[Path]:
    candidate = metrics_file or os.getenv("OBSERVABILITY_METRICS_FILE")
    if not candidate:
        return None
    p = Path(candidate).expanduser().resolve()
    return p if p.exists() else None


def _window_to_unix(
    start: str, end: str,
    baseline_range: Optional[Tuple[str, str]] = None,
) -> Tuple[float, float, float, float]:
    """Convert ISO/Unix timestamps to Unix floats for both windows.

    Returns (baseline_start, baseline_end, incident_start, incident_end).
    If baseline_range is None, splits [start, end] into 50/50.
    """
    def to_unix(s: str) -> float:
        if not s:
            return 0.0
        try:
            return float(s)
        except ValueError:
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0

    s_start = to_unix(start)
    s_end = to_unix(end)
    if baseline_range:
        b_start = to_unix(baseline_range[0])
        b_end = to_unix(baseline_range[1])
        return b_start, b_end, s_start, s_end
    # Legacy 50/50 fallback
    mid = (s_start + s_end) / 2.0
    return s_start, mid, mid, s_end


def load_metrics(
    metrics_file: Optional[str] = None,
    columns_prefix: Optional[str] = None,
) -> Tuple[List[str], List[List[Optional[float]]]]:
    """Load the metrics CSV into (headers, rows).

    Args:
        metrics_file: path override (env: OBSERVABILITY_METRICS_FILE)
        columns_prefix: if provided, restrict to columns starting with this
            prefix (e.g. "checkoutservice_") plus the time column. Saves
            memory when we know the target service up front.

    Returns:
        (headers, rows) where rows[i][0] is the time as Unix float and
        rows[i][j] is the j-th column value (None if empty/NaN).

    Returns ([], []) if no file.
    """
    path = _resolve_metrics_file(metrics_file)
    if path is None:
        return [], []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        if not headers:
            return [], []

        if columns_prefix:
            kept_idx = [0] + [
                i for i, h in enumerate(headers)
                if i > 0 and h.startswith(columns_prefix)
            ]
            headers = [headers[i] for i in kept_idx]
        else:
            kept_idx = list(range(len(headers)))

        out_rows: List[List[Optional[float]]] = []
        for r in reader:
            if not r:
                continue
            row: List[Optional[float]] = []
            for out_pos, src_pos in enumerate(kept_idx):
                if src_pos >= len(r):
                    row.append(None)
                    continue
                cell = r[src_pos]
                if out_pos == 0:
                    row.append(_parse_time_to_unix(cell))
                else:
                    try:
                        v = float(cell) if cell != "" else None
                        if v is not None and math.isnan(v):
                            v = None
                        row.append(v)
                    except ValueError:
                        row.append(None)
            out_rows.append(row)
    return headers, out_rows


# =============================================================================
# Helpers: slicing and statistics
# =============================================================================

def _rows_in_window(
    rows: List[List[Optional[float]]], start: float, end: float,
) -> List[List[Optional[float]]]:
    return [r for r in rows if r and r[0] is not None and start <= r[0] <= end]


def _column_values(
    rows: List[List[Optional[float]]], col_idx: int,
) -> List[float]:
    out: List[float] = []
    for r in rows:
        if col_idx < len(r):
            v = r[col_idx]
            if v is not None:
                out.append(float(v))
    return out


def _find_col(headers: List[str], service: str, suffix: str) -> Optional[int]:
    target = f"{service}_{suffix}"
    try:
        return headers.index(target)
    except ValueError:
        return None


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stddev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def _rate_series(values: List[float]) -> List[float]:
    """Convert a cumulative counter series into per-sample differences (rate).

    RCAEval samples at ~1Hz, so the difference equals approximately the rate
    in units/sec. Negative deltas (counter resets) are clamped to 0.
    """
    if len(values) < 2:
        return [0.0] * len(values)
    rates = [0.0]
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        rates.append(max(d, 0.0))
    return rates


def _counter_delta(values: List[float]) -> float:
    """Total increase of a cumulative counter over a window.

    Handles counter resets: if series dips, we sum the positive deltas only.
    """
    if len(values) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        if d > 0:
            total += d
    return total


# =============================================================================
# Public API — the three MCP tool functions
# =============================================================================

def get_metric_summary(
    service: str,
    start: str,
    end: str,
    metrics_file: Optional[str] = None,
    baseline_range: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Summarise CPU and memory for a service in the incident window.

    Returns a dict with:
        cpu_avg, cpu_max         : incident-window CPU rate (seconds/sec)
        cpu_baseline_avg, cpu_baseline_stddev
        cpu_spike_zscore         : (incident_max - baseline_avg) / baseline_std
        mem_avg, mem_max         : incident-window memory bytes (gauge)
        mem_baseline_avg
        mem_jump                 : (incident_max - baseline_avg) / baseline_avg
        has_data                 : False if service has no metrics
    """
    headers, rows = load_metrics(metrics_file, columns_prefix=f"{service}_")
    if not headers or not rows:
        return {"has_data": False, "service": service, "reason": "no_metrics_file"}

    cpu_col = _find_col(headers, service, CPU_USAGE_SUFFIX)
    mem_col = _find_col(headers, service, MEM_USAGE_SUFFIX)
    if cpu_col is None and mem_col is None:
        return {"has_data": False, "service": service, "reason": "service_not_in_metrics"}

    b_start, b_end, i_start, i_end = _window_to_unix(start, end, baseline_range)
    b_rows = _rows_in_window(rows, b_start, b_end)
    i_rows = _rows_in_window(rows, i_start, i_end)

    summary: Dict[str, Any] = {"service": service, "has_data": True}

    # CPU: counter → rate
    if cpu_col is not None:
        b_cpu_cum = _column_values(b_rows, cpu_col)
        i_cpu_cum = _column_values(i_rows, cpu_col)
        b_cpu_rates = _rate_series(b_cpu_cum)
        i_cpu_rates = _rate_series(i_cpu_cum)
        b_avg = _mean(b_cpu_rates) if b_cpu_rates else 0.0
        b_std = _stddev(b_cpu_rates) if b_cpu_rates else 0.0
        i_avg = _mean(i_cpu_rates) if i_cpu_rates else 0.0
        i_max = max(i_cpu_rates) if i_cpu_rates else 0.0
        spike_z = ((i_max - b_avg) / b_std) if b_std > 1e-9 else (
            10.0 if i_max > b_avg + 1e-6 else 0.0
        )
        summary.update({
            "cpu_avg": round(i_avg, 4),
            "cpu_max": round(i_max, 4),
            "cpu_baseline_avg": round(b_avg, 4),
            "cpu_baseline_stddev": round(b_std, 4),
            "cpu_spike_zscore": round(spike_z, 2),
        })

    # Memory: gauge
    if mem_col is not None:
        b_mem = _column_values(b_rows, mem_col)
        i_mem = _column_values(i_rows, mem_col)
        b_avg = _mean(b_mem)
        i_avg = _mean(i_mem)
        i_max = max(i_mem) if i_mem else 0.0
        mem_jump = ((i_max - b_avg) / b_avg) if b_avg > 1e-9 else 0.0
        summary.update({
            "mem_avg_bytes": int(i_avg),
            "mem_max_bytes": int(i_max),
            "mem_baseline_avg_bytes": int(b_avg),
            "mem_jump_ratio": round(mem_jump, 3),
        })

    return summary


def get_latency_summary(
    service: str,
    start: str,
    end: str,
    metrics_file: Optional[str] = None,
    baseline_range: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Summarise Istio p50/p95/p99 latencies for a service.

    Istio metrics in RCAEval are in SECONDS. This function returns
    milliseconds for human readability and paper-friendly values.

    Returns:
        p50_ms, p95_ms, p99_ms                  (incident window mean)
        p50_baseline_ms, p95_baseline_ms, p99_baseline_ms
        p95_delta_ms                            (incident − baseline)
        p99_delta_ms
        has_data                                (False if no latency cols)
    """
    headers, rows = load_metrics(metrics_file, columns_prefix=f"{service}_")
    if not headers or not rows:
        return {"has_data": False, "service": service, "reason": "no_metrics_file"}

    cols = {
        "50": _find_col(headers, service, ISTIO_LATENCY_50),
        "95": _find_col(headers, service, ISTIO_LATENCY_95),
        "99": _find_col(headers, service, ISTIO_LATENCY_99),
    }
    if all(v is None for v in cols.values()):
        return {"has_data": False, "service": service,
                "reason": "no_istio_latency_columns"}

    b_start, b_end, i_start, i_end = _window_to_unix(start, end, baseline_range)
    b_rows = _rows_in_window(rows, b_start, b_end)
    i_rows = _rows_in_window(rows, i_start, i_end)

    summary: Dict[str, Any] = {"service": service, "has_data": True}
    for pct, col in cols.items():
        if col is None:
            continue
        b_vals = _column_values(b_rows, col)
        i_vals = _column_values(i_rows, col)
        b_avg = _mean(b_vals)
        i_avg = _mean(i_vals)
        # Convert seconds → milliseconds
        summary[f"p{pct}_ms"] = round(i_avg * 1000, 2)
        summary[f"p{pct}_baseline_ms"] = round(b_avg * 1000, 2)
        summary[f"p{pct}_delta_ms"] = round((i_avg - b_avg) * 1000, 2)

    return summary


def get_retry_timeout_summary(
    service: str,
    start: str,
    end: str,
    metrics_file: Optional[str] = None,
    baseline_range: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Summarise error / drop / reset indicators from Istio + network metrics.

    Note: TRUE retry and timeout counts are not separately emitted by
    RCAEval's metric exporter. We proxy them via:
        timeout_like_count   = istio-error-total delta (incident)
        request_total_delta  = istio-request-total delta (for error rate)
        error_rate           = error_delta / request_delta
        rx_drop_delta        = network-receive-packets-dropped-total delta
        tx_drop_delta        = network-transmit-packets-dropped-total delta
        rx_errors_delta      = network-receive-errors-total delta
        tx_errors_delta      = network-transmit-errors-total delta

    These are STRONG proxies for 'loss' fault (packet drops) and 'socket'
    fault (network errors). They complement v6 log-keyword hits.
    """
    headers, rows = load_metrics(metrics_file, columns_prefix=f"{service}_")
    if not headers or not rows:
        return {"has_data": False, "service": service, "reason": "no_metrics_file"}

    b_start, b_end, i_start, i_end = _window_to_unix(start, end, baseline_range)
    i_rows = _rows_in_window(rows, i_start, i_end)

    summary: Dict[str, Any] = {"service": service, "has_data": True}

    # Istio errors / requests
    err_col = _find_col(headers, service, ISTIO_ERROR_TOTAL)
    req_col = _find_col(headers, service, ISTIO_REQUEST_TOTAL)
    if err_col is not None:
        summary["error_delta"] = round(_counter_delta(_column_values(i_rows, err_col)), 2)
    if req_col is not None:
        summary["request_delta"] = round(_counter_delta(_column_values(i_rows, req_col)), 2)
    if "error_delta" in summary and "request_delta" in summary:
        rd = summary["request_delta"]
        summary["error_rate"] = round(summary["error_delta"] / rd, 4) if rd > 0 else 0.0

    # Network packet drops (strong loss-fault signal)
    for key, suffix in (
        ("rx_drop_delta", NET_RX_DROP),
        ("tx_drop_delta", NET_TX_DROP),
        ("rx_errors_delta", NET_RX_ERRORS),
        ("tx_errors_delta", NET_TX_ERRORS),
    ):
        col = _find_col(headers, service, suffix)
        if col is not None:
            summary[key] = round(_counter_delta(_column_values(i_rows, col)), 2)

    # Sockets gauge (avg during incident) — strong socket-fault signal
    sock_col = _find_col(headers, service, SOCKETS_GAUGE)
    if sock_col is not None:
        i_socks = _column_values(i_rows, sock_col)
        summary["sockets_avg"] = round(_mean(i_socks), 1) if i_socks else 0.0
        summary["sockets_max"] = int(max(i_socks)) if i_socks else 0

    return summary


# =============================================================================
# Batch helper — compute all three summaries for all services at once.
# Used by Log Agent to populate the metric sections of its prompt without
# issuing 3×N separate MCP calls.
# =============================================================================

def get_all_service_metric_summaries(
    start: str,
    end: str,
    metrics_file: Optional[str] = None,
    baseline_range: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """Return {service: {metric, latency, retry_timeout}} for every service
    present in the metrics CSV. Runs the CSV scan once (no prefix filter) and
    bulk-computes per-service summaries.
    """
    headers, rows = load_metrics(metrics_file, columns_prefix=None)
    if not headers or not rows:
        return {"has_data": False, "services": {}, "reason": "no_metrics_file"}

    # Services are the unique prefixes (before first underscore) of all
    # non-time columns.
    services: set = set()
    for h in headers[1:]:
        if "_" in h:
            services.add(h.split("_", 1)[0])

    out: Dict[str, Any] = {"has_data": True, "services": {}}
    for svc in sorted(services):
        out["services"][svc] = {
            "metric":        get_metric_summary(svc, start, end, metrics_file, baseline_range),
            "latency":       get_latency_summary(svc, start, end, metrics_file, baseline_range),
            "retry_timeout": get_retry_timeout_summary(svc, start, end, metrics_file, baseline_range),
        }
    return out
