
import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .models import LogRecord, ErrorSummary


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_FILE = BASE_DIR / "sample_logs.jsonl"


def _resolve_log_file(log_file: Optional[str] = None) -> Path:
    candidate = log_file or os.getenv("OBSERVABILITY_LOG_FILE")
    return Path(candidate).expanduser().resolve() if candidate else DEFAULT_LOG_FILE


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_logs(log_file: Optional[str] = None) -> List[LogRecord]:
    results: List[LogRecord] = []
    selected = _resolve_log_file(log_file)
    if not selected.exists():
        return results

    with selected.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(LogRecord(**json.loads(line)))
    return results


def search_logs(
    service: str,
    start: str,
    end: str,
    keyword: Optional[str] = None,
    log_file: Optional[str] = None,
) -> List[LogRecord]:
    start_dt = _parse_ts(start)
    end_dt = _parse_ts(end)

    records = load_logs(log_file=log_file)
    filtered = []

    for r in records:
        ts = _parse_ts(r.timestamp)
        if r.service != service:
            continue
        if not (start_dt <= ts <= end_dt):
            continue
        if keyword and keyword.lower() not in r.message.lower():
            continue
        filtered.append(r)

    return filtered


def get_trace_logs(trace_id: str, log_file: Optional[str] = None) -> List[LogRecord]:
    return [r for r in load_logs(log_file=log_file) if r.trace_id == trace_id]


import re as _re

# =============================================================================
# Anomaly keyword dictionaries (used by get_service_statistics)
# =============================================================================
# These patterns target textual anomaly signals that don't require ERROR-level
# logs. Case-insensitive substring matching on r.message.
#
# References / rationale:
#   - latency: captures "slow", "latency spike", "took N ms", "delay"
#   - timeout: the single strongest non-error distress signal in microservices
#   - retry:   retry/back-off logs reveal upstream instability before errors
#   - reset:   "connection reset", "refused", "broken pipe" = network/socket issues
#
# Lists are deliberately conservative — high-precision keywords only.

_LATENCY_PATTERNS = _re.compile(
    r"\b(slow|latency|latent|delay(ed)?|lag(gy)?|took\s+\d+\s*(ms|s\b|sec))\b",
    _re.IGNORECASE,
)
_TIMEOUT_PATTERNS = _re.compile(
    r"\b(time[\s\-]?out|timed\s+out|deadline\s+exceeded|dl\s+exceeded)\b",
    _re.IGNORECASE,
)
_RETRY_PATTERNS = _re.compile(
    r"\b(retry(ing)?|retries|retry\s+attempt|back[\s\-]?off|resend)\b",
    _re.IGNORECASE,
)
_RESET_PATTERNS = _re.compile(
    r"\b(reset(\s+by\s+peer)?|connection\s+refused|conn\s+refused|"
    r"broken\s+pipe|epipe|econnrefused|econnreset)\b",
    _re.IGNORECASE,
)


def get_service_statistics(
    start: str,
    end: str,
    log_file: Optional[str] = None,
    baseline_window_ratio: float = 0.5,
    baseline_range: Optional[tuple] = None,
    incident_range: Optional[tuple] = None,
) -> dict:
    """서비스별 통계 시그널 추출 — ERROR 로그가 없어도 이상 탐지 가능.

    두 가지 모드:
    (1) Legacy: start/end 하나의 창을 받고 baseline_window_ratio로 반으로 쪼갬
        (pre-v6 동작과 동일. 호출자에서 baseline_range/incident_range를 안 주면
        자동으로 이 모드 사용).
    (2) Dual-window (v6 권장): baseline_range=(s,e)와 incident_range=(s,e)를
        별도로 받아, 사이에 gap이 있는 두 개의 이격된 창을 분석.
        RCAEval 같은 instant-fault 벤치마크에서 "주입 직전 정상 기준선 vs 주입 직후
        이상 구간"을 깨끗하게 분리할 수 있다.

    각 서비스에 대해 다음을 계산 (필드명 alias 포함 for backward compat):
      - total_logs, baseline_logs, incident_logs  (legacy names)
      - baseline_count, incident_count             (new aliases)
      - volume_delta: (incident - baseline_normalised) / max(baseline_normalised, 1)
                     두 창 길이가 다르면 단위 시간당 비율로 정규화하여 계산
      - error_ratio: (ERROR+WARN+5xx) / total
      - level_counts: {'ERROR', 'WARN', 'INFO', 'DEBUG', 'OTHER'}
      - latency_hits: latency 키워드 매칭 (incident 창 기준)
      - timeout_hits: timeout 키워드 매칭
      - retry_hits:   retry 키워드 매칭
      - reset_hits:   reset/refused 키워드 매칭
      - has_activity

    Returns:
        {
          "window": {...},                              # 실제 사용된 시간대
          "mode": "legacy" | "dual",
          "services": {svc_name: {...}, ...},
          "total_log_count": int
        }
    """
    # --- Resolve effective windows ---
    if baseline_range is not None and incident_range is not None:
        mode = "dual"
        b_start_dt = _parse_ts(baseline_range[0])
        b_end_dt = _parse_ts(baseline_range[1])
        i_start_dt = _parse_ts(incident_range[0])
        i_end_dt = _parse_ts(incident_range[1])
        # Outer envelope (for load_logs filtering)
        outer_start_dt = min(b_start_dt, i_start_dt)
        outer_end_dt = max(b_end_dt, i_end_dt)
    else:
        mode = "legacy"
        outer_start_dt = _parse_ts(start)
        outer_end_dt = _parse_ts(end)
        window_secs = (outer_end_dt - outer_start_dt).total_seconds()
        if window_secs <= 0:
            return {"window": {"start": start, "end": end, "baseline_end": end},
                    "mode": mode, "services": {}, "total_log_count": 0}
        mid_dt = outer_start_dt + (outer_end_dt - outer_start_dt) * baseline_window_ratio
        b_start_dt, b_end_dt = outer_start_dt, mid_dt
        i_start_dt, i_end_dt = mid_dt, outer_end_dt

    baseline_secs = max((b_end_dt - b_start_dt).total_seconds(), 1.0)
    incident_secs = max((i_end_dt - i_start_dt).total_seconds(), 1.0)

    # --- Scan logs once ---
    records = load_logs(log_file=log_file)
    per_service: dict = {}
    total_count = 0

    for r in records:
        try:
            ts = _parse_ts(r.timestamp)
        except Exception:
            continue
        in_baseline = (b_start_dt <= ts <= b_end_dt)
        in_incident = (i_start_dt <= ts <= i_end_dt)
        if not (in_baseline or in_incident):
            continue

        total_count += 1
        svc = r.service
        bucket = per_service.setdefault(svc, {
            "baseline_logs": 0,
            "incident_logs": 0,
            "level_counts": {"ERROR": 0, "WARN": 0, "INFO": 0, "DEBUG": 0, "OTHER": 0},
            "error_logs": 0,
            "latency_hits": 0,
            "timeout_hits": 0,
            "retry_hits": 0,
            "reset_hits": 0,
        })

        if in_baseline:
            bucket["baseline_logs"] += 1
        if in_incident:
            bucket["incident_logs"] += 1

        level = (r.level or "").upper()
        if level in bucket["level_counts"]:
            bucket["level_counts"][level] += 1
        else:
            bucket["level_counts"]["OTHER"] += 1

        if level in ("ERROR", "WARN") or (r.status_code and r.status_code >= 500):
            bucket["error_logs"] += 1

        # Keyword counts — only within incident window (where anomalies manifest)
        if in_incident:
            msg = r.message or ""
            if _LATENCY_PATTERNS.search(msg):
                bucket["latency_hits"] += 1
            if _TIMEOUT_PATTERNS.search(msg):
                bucket["timeout_hits"] += 1
            if _RETRY_PATTERNS.search(msg):
                bucket["retry_hits"] += 1
            if _RESET_PATTERNS.search(msg):
                bucket["reset_hits"] += 1

    # --- Build per-service summary with rate-normalised volume_delta ---
    services: dict = {}
    for svc, b in per_service.items():
        total = b["baseline_logs"] + b["incident_logs"]
        baseline = b["baseline_logs"]
        incident = b["incident_logs"]

        # Rate-normalise: logs/sec in each window. This is fair when the two
        # windows have different lengths (e.g. baseline 480s vs incident 150s).
        baseline_rate = baseline / baseline_secs
        incident_rate = incident / incident_secs
        if baseline_rate < 1e-9:
            volume_delta = 1.0 if incident_rate > 0 else 0.0
        else:
            volume_delta = (incident_rate - baseline_rate) / baseline_rate

        services[svc] = {
            # Legacy field names (kept for backward compat)
            "total_logs": total,
            "baseline_logs": baseline,
            "incident_logs": incident,
            # New aliases matching spec
            "baseline_count": baseline,
            "incident_count": incident,
            # Signals
            "volume_delta": round(volume_delta, 3),
            "error_ratio": round(b["error_logs"] / total, 3) if total else 0.0,
            "level_counts": b["level_counts"],
            # Keyword signals (incident window only)
            "latency_hits": b["latency_hits"],
            "timeout_hits": b["timeout_hits"],
            "retry_hits": b["retry_hits"],
            "reset_hits": b["reset_hits"],
            "has_activity": total > 0,
        }

    return {
        "window": {
            "baseline_start": b_start_dt.isoformat(),
            "baseline_end": b_end_dt.isoformat(),
            "incident_start": i_start_dt.isoformat(),
            "incident_end": i_end_dt.isoformat(),
            "baseline_seconds": baseline_secs,
            "incident_seconds": incident_secs,
            # Legacy-compat fields
            "start": outer_start_dt.isoformat(),
            "end": outer_end_dt.isoformat(),
        },
        "mode": mode,
        "services": services,
        "total_log_count": total_count,
    }


def get_error_summary(
    service: str,
    start: str,
    end: str,
    log_file: Optional[str] = None,
) -> ErrorSummary:
    records = search_logs(service=service, start=start, end=end, keyword=None, log_file=log_file)

    error_logs = [
        r for r in records
        if r.level.upper() in ("ERROR", "WARN") or (r.status_code and r.status_code >= 500)
    ]

    error_type_count = {}
    upstream_count = {}

    for r in error_logs:
        key = r.error_type or "UNKNOWN"
        error_type_count[key] = error_type_count.get(key, 0) + 1

        if r.upstream:
            upstream_count[r.upstream] = upstream_count.get(r.upstream, 0) + 1

    top_error_types = [
        {"name": k, "count": v}
        for k, v in sorted(error_type_count.items(), key=lambda x: x[1], reverse=True)
    ]

    top_upstreams = [
        {"name": k, "count": v}
        for k, v in sorted(upstream_count.items(), key=lambda x: x[1], reverse=True)
    ]

    return ErrorSummary(
        service=service,
        total_logs=len(records),
        error_logs=len(error_logs),
        top_error_types=top_error_types[:5],
        top_upstreams=top_upstreams[:5],
    )
