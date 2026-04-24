"""
Evidence-Aware RCA — Evidence Factory (Phase 1).

본 모듈은 기존 MCP / repository 계층의 출력(raw dict, stats, metric
summary 등)을 EvidenceUnit으로 변환하는 factory 함수들을 제공한다.

설계 원칙
---------
1. 비침습(non-intrusive): 기존 코드(agents/log_agent 등)를 건드리지 않고,
   factory 호출만으로 기존 출력을 스키마로 승격시킬 수 있다.
2. 느슨한 결합: factory는 입력 dict의 정확한 key를 가정하되, 누락되면
   안전하게 skip하거나 빈 EvidenceUnit을 만들지 않는다.
3. severity 정규화: 각 modality의 서로 다른 스케일 (CPU z-score vs
   p95 delta ms vs error_ratio)을 [0, 1]로 매핑하는 로직이 한 곳에 모인다.

Phase 1 범위
------------
Factory만 구현. 실제 호출은 Phase 2에서 MCP 서버가, Phase 3에서 Agent가
수행한다. 본 파일의 함수는 단독으로도 단위 테스트 가능하다.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .evidence import (
    AnomalyType,
    EvidenceCollection,
    EvidenceUnit,
    Modality,
    TimeRange,
    make_evidence_id,
)


# ---------------------------------------------------------------------------
# Severity normalisation helpers
# ---------------------------------------------------------------------------

def _sigmoid_severity(x: float, scale: float = 1.0) -> float:
    """Smooth [0, 1] mapping — 0 → 0, scale → ~0.46, 2*scale → ~0.76, ∞ → 1.

    수학적으로: 2 * logistic(|x|/scale) - 1

    - 정상 노이즈(x ≈ 0)는 severity ≈ 0 (증거로 불채택)
    - scale 값은 '중간 세기 신호' (scale=20 이면 z=20에서 sev≈0.46)
    - 2*scale 이상이면 '강한 신호' (z=40 이상이면 sev≈0.76)

    이 설계는 factory의 threshold(sev >= 0.3)와 맞물린다:
      - sev < 0.3 ↔ |x| < ~0.36*scale → noise, 증거로 불채택
      - 실데이터에서 CPU fault의 z=100~300은 sev≈1 (확실 채택)
    """
    try:
        abs_x = abs(float(x))
        # logistic: 1 / (1 + e^(-abs_x/scale)) ∈ [0.5, 1]
        # 2 * logistic - 1 ∈ [0, 1] with monotonic behavior
        val = 2.0 / (1.0 + math.exp(-abs_x / max(scale, 1e-9))) - 1.0
    except OverflowError:
        val = 1.0
    return max(0.0, min(1.0, val))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _severity_from_cpu_zscore(z: Optional[float]) -> float:
    """CPU spike z-score → severity.

    실데이터 관찰: CPU fault 시 z가 100~300 수준으로 튐. 정상 노이즈는
    z < 3 수준. scale=20으로 두면 z=20에서 sev≈0.73, z=50에서 sev≈0.92.

    방향성 (Phase 3b fix): **양의 z만** severity를 받는다. 음의 z는
    "CPU가 baseline보다 오히려 낮다"는 뜻이고, 이것은 fault가 아니다
    (해당 서비스가 downstream 문제로 덜 일하게 된 증상일 수 있음).
    """
    if z is None or z <= 0:
        return 0.0
    return _sigmoid_severity(z, scale=20.0)


def _severity_from_mem_jump(ratio: Optional[float]) -> float:
    """Memory jump ratio → severity. 오직 양의 ratio(증가)만 이상으로 본다.

    음의 ratio는 메모리 해제 / 서비스 trim의 정상 현상일 수 있다.
    """
    if ratio is None or ratio <= 0:
        return 0.0
    return _sigmoid_severity(ratio, scale=0.3)


def _severity_from_latency_delta(delta_ms: Optional[float]) -> float:
    """p95 / p99 latency delta (ms) → severity.

    **양의 delta만** severity를 받는다 (느려진 경우). 음의 delta는
    "latency가 오히려 빨라짐"이고, 이는 증상이 아니다.

    이 수정은 Phase 3a shadow 관찰에서 발견된 hub-service bias의
    직접적 해결이다: downstream이 느려져서 호출량이 줄면 hub의
    latency는 오히려 감소하는데, 이전에는 abs()로 severity를 줘서
    hub가 evidence 상위에 올라가는 잘못된 동작을 했다.
    """
    if delta_ms is None or delta_ms <= 0:
        return 0.0
    return _sigmoid_severity(delta_ms, scale=100.0)


def _severity_from_drop_count(cnt: Optional[float]) -> float:
    """packet drop 누적 수 → severity. 오직 양수(실제 drop 발생)만."""
    if cnt is None or cnt <= 0:
        return 0.0
    return _sigmoid_severity(float(cnt), scale=100.0)


def _severity_from_error_ratio(r: Optional[float]) -> float:
    """error_ratio [0, 1] → severity. 오직 양수만.

    원래 error_ratio는 0 이상만 가능하지만, 방어적으로 guard.
    """
    if r is None or r <= 0:
        return 0.0
    return _sigmoid_severity(r, scale=0.1)


def _severity_from_volume_delta(delta: Optional[float]) -> float:
    """log volume delta → severity.

    **volume_delta는 예외다** — 양수(급증)와 음수(급감) 모두 이상 신호다.
    Crash된 서비스는 로그가 급감하고, retry storm은 급증한다. 따라서
    abs()를 유지한다. 단 상한 0.6으로 noisy signal을 견제.
    """
    if delta is None:
        return 0.0
    raw = _sigmoid_severity(abs(delta), scale=0.5)
    return min(0.6, raw)


def _severity_from_keyword_hits(
    hits: Dict[str, int], total_logs: Optional[int] = None
) -> float:
    """timeout/retry/reset 키워드 hit count → severity.

    hits dict: {"timeout": 5, "retry": 2, "reset": 0} 형태.
    """
    if not hits:
        return 0.0
    total_hits = sum(v for v in hits.values() if isinstance(v, (int, float)))
    if total_hits <= 0:
        return 0.0
    # 로그 총량 대비 비율로 보정 가능
    if total_logs and total_logs > 0:
        rate = total_hits / total_logs
        return min(1.0, max(_sigmoid_severity(rate, scale=0.05), 0.3))
    return _sigmoid_severity(float(total_hits), scale=10.0)


# ---------------------------------------------------------------------------
# Log evidence factories
# ---------------------------------------------------------------------------

def evidence_from_service_statistics(
    service: str,
    stats: Dict[str, Any],
    time_range: TimeRange,
    source: str = "observability_mcp/log_repository",
) -> List[EvidenceUnit]:
    """v6 dual-window statistics → EvidenceUnits.

    stats 예시:
        {
            "service": "checkoutservice",
            "baseline_count": 120, "incident_count": 380,
            "volume_delta": 2.17, "error_ratio": 0.05,
            "timeout_hits": 12, "retry_hits": 3, "reset_hits": 1,
            "top_level": "ERROR",
        }

    이 함수는 하나의 dict에서 최대 3개의 EvidenceUnit을 생성할 수 있다:
      (1) volume_shift:    volume_delta가 의미 있을 때
      (2) error_spike:     error_ratio가 0 초과일 때
      (3) keyword_distress: timeout/retry/reset hit 합이 > 0일 때

    중복/무의미한 증거는 만들지 않는다.
    """
    out: List[EvidenceUnit] = []

    baseline_count = int(stats.get("baseline_count") or 0)
    incident_count = int(stats.get("incident_count") or 0)
    volume_delta = stats.get("volume_delta")
    error_ratio = stats.get("error_ratio")

    timeout_hits = int(stats.get("timeout_hits") or 0)
    retry_hits = int(stats.get("retry_hits") or 0)
    reset_hits = int(stats.get("reset_hits") or 0)
    latency_hits = int(stats.get("latency_hits") or 0)

    total_logs = baseline_count + incident_count

    # (1) volume_shift
    # 낮은 volume 서비스의 rate-norm artifact 방지: 최소 incident_count
    # 30건 이상일 때만 증거로 승격.
    if (
        volume_delta is not None
        and abs(volume_delta) >= 0.3
        and incident_count >= 30
    ):
        ev = EvidenceUnit(
            evidence_id=make_evidence_id(
                "log", [service], time_range, "volume_shift",
                extra=f"vd={volume_delta:.2f}",
            ),
            modality="log",
            time_range=time_range,
            services=[service],
            anomaly_type="volume_shift",
            severity=_severity_from_volume_delta(volume_delta),
            observation={
                "volume_delta": round(float(volume_delta), 4),
                "baseline_count": baseline_count,
                "incident_count": incident_count,
            },
            source=source,
        )
        out.append(ev)

    # (2) error_spike
    if error_ratio is not None and float(error_ratio) > 0.01:
        ev = EvidenceUnit(
            evidence_id=make_evidence_id(
                "log", [service], time_range, "error_spike",
                extra=f"er={error_ratio:.3f}",
            ),
            modality="log",
            time_range=time_range,
            services=[service],
            anomaly_type="error_spike",
            severity=_severity_from_error_ratio(error_ratio),
            observation={
                "error_ratio": round(float(error_ratio), 4),
                "incident_count": incident_count,
            },
            source=source,
        )
        out.append(ev)

    # (3) keyword_distress
    total_kw = timeout_hits + retry_hits + reset_hits + latency_hits
    if total_kw > 0:
        hits_dict = {
            "timeout": timeout_hits,
            "retry": retry_hits,
            "reset": reset_hits,
            "latency": latency_hits,
        }
        ev = EvidenceUnit(
            evidence_id=make_evidence_id(
                "log", [service], time_range, "keyword_distress",
                extra=f"kw={total_kw}",
            ),
            modality="log",
            time_range=time_range,
            services=[service],
            anomaly_type="keyword_distress",
            severity=_severity_from_keyword_hits(hits_dict, total_logs),
            observation={
                "keyword_hits": hits_dict,
                "total_logs": total_logs,
            },
            source=source,
        )
        out.append(ev)

    return out


def evidence_from_log_records(
    service: str,
    records: List[Dict[str, Any]],
    time_range: TimeRange,
    max_samples: int = 3,
    source: str = "observability_mcp/log_repository",
) -> Optional[EvidenceUnit]:
    """ERROR/WARN level 로그 레코드 목록 → dependency_failure evidence.

    records는 기존 search_logs()의 반환 형식:
        [{"timestamp": "...", "level": "ERROR", "message": "...",
          "upstream": "user-db", "status_code": 500, ...}, ...]

    upstream 필드가 있는 에러 레코드는 dependency_failure evidence로
    승격된다. 없으면 error_spike에 포함되므로 여기서는 만들지 않는다.
    """
    upstream_errors = [
        r for r in records
        if (r.get("level", "") or "").upper() in ("ERROR", "WARN")
        and r.get("upstream")
    ]
    if not upstream_errors:
        return None

    upstream_svcs = sorted({r["upstream"] for r in upstream_errors})
    samples = [
        (r.get("message") or "")[:200]
        for r in upstream_errors[:max_samples]
    ]

    return EvidenceUnit(
        evidence_id=make_evidence_id(
            "log",
            [service, *upstream_svcs],
            time_range,
            "dependency_failure",
            extra=f"n={len(upstream_errors)}",
        ),
        modality="log",
        time_range=time_range,
        services=[service, *upstream_svcs],
        anomaly_type="dependency_failure",
        severity=min(1.0, 0.3 + 0.05 * len(upstream_errors)),
        observation={
            "upstream_error_count": len(upstream_errors),
            "upstream_services": upstream_svcs,
            "status_codes": sorted({
                r.get("status_code") for r in upstream_errors
                if r.get("status_code")
            }),
        },
        source=source,
        raw_samples=samples,
    )


# ---------------------------------------------------------------------------
# Metric evidence factories
# ---------------------------------------------------------------------------

def evidence_from_metric_summary(
    service: str,
    summary: Dict[str, Any],
    time_range: TimeRange,
    source: str = "observability_mcp/metric_repository",
) -> List[EvidenceUnit]:
    """v8 metric summary → EvidenceUnits.

    summary 예시 (metric_repository.get_all_service_metric_summaries에서 온):
        {
            "metric":          {"has_data": True, "cpu_spike_zscore": 289.6,
                                "cpu_max": 5.16, "mem_jump_ratio": 0.40, ...},
            "latency":         {"has_data": True, "p95_delta_ms": 349.5,
                                "p99_delta_ms": 512.0, ...},
            "retry_timeout":   {"has_data": True, "rx_drop_delta": 0,
                                "error_delta": 12, "sockets_max": 980},
        }

    최대 3개의 EvidenceUnit 생성:
      (1) resource_saturation: CPU 또는 Memory 이상
      (2) latency_degradation: p95 또는 p99 이상
      (3) network_degradation: packet drops 또는 errors
    """
    out: List[EvidenceUnit] = []

    m = summary.get("metric") or {}
    l = summary.get("latency") or {}
    rt = summary.get("retry_timeout") or {}

    # (1) resource_saturation
    if m.get("has_data"):
        cpu_z = m.get("cpu_spike_zscore")
        mem_j = m.get("mem_jump_ratio")
        cpu_sev = _severity_from_cpu_zscore(cpu_z)
        mem_sev = _severity_from_mem_jump(mem_j)
        sev = max(cpu_sev, mem_sev)
        if sev >= 0.3:  # minimum threshold to avoid noise
            ev = EvidenceUnit(
                evidence_id=make_evidence_id(
                    "metric", [service], time_range, "resource_saturation",
                    extra=f"cpu_z={cpu_z} mem={mem_j}",
                ),
                modality="metric",
                time_range=time_range,
                services=[service],
                anomaly_type="resource_saturation",
                severity=sev,
                observation={
                    "cpu_spike_zscore": cpu_z,
                    "cpu_max": m.get("cpu_max"),
                    "mem_jump_ratio": mem_j,
                    "mem_max": m.get("mem_max"),
                },
                source=source,
            )
            out.append(ev)

    # (2) latency_degradation
    if l.get("has_data"):
        p95 = l.get("p95_delta_ms")
        p99 = l.get("p99_delta_ms")
        sev = max(
            _severity_from_latency_delta(p95),
            _severity_from_latency_delta(p99),
        )
        if sev >= 0.3:
            ev = EvidenceUnit(
                evidence_id=make_evidence_id(
                    "metric", [service], time_range, "latency_degradation",
                    extra=f"p95={p95} p99={p99}",
                ),
                modality="metric",
                time_range=time_range,
                services=[service],
                anomaly_type="latency_degradation",
                severity=sev,
                observation={
                    "p95_delta_ms": p95,
                    "p99_delta_ms": p99,
                    "p50_ms": l.get("p50_ms"),
                    "p95_ms": l.get("p95_ms"),
                },
                source=source,
            )
            out.append(ev)

    # (3) network_degradation
    if rt.get("has_data"):
        rx_drop = rt.get("rx_drop_delta") or 0
        tx_drop = rt.get("tx_drop_delta") or 0
        err_delta = rt.get("error_delta") or 0
        drop_sev = _severity_from_drop_count(rx_drop + tx_drop)
        # istio error도 포함
        if err_delta > 0:
            drop_sev = max(drop_sev, _sigmoid_severity(err_delta, scale=20))
        if drop_sev >= 0.3:
            ev = EvidenceUnit(
                evidence_id=make_evidence_id(
                    "metric", [service], time_range, "network_degradation",
                    extra=f"rx={rx_drop} tx={tx_drop} err={err_delta}",
                ),
                modality="metric",
                time_range=time_range,
                services=[service],
                anomaly_type="network_degradation",
                severity=drop_sev,
                observation={
                    "rx_drop_delta": rx_drop,
                    "tx_drop_delta": tx_drop,
                    "error_delta": err_delta,
                    "sockets_max": rt.get("sockets_max"),
                },
                source=source,
            )
            out.append(ev)

    return out


# ---------------------------------------------------------------------------
# Topology evidence factory
# ---------------------------------------------------------------------------

def evidence_from_topology_path(
    symptom_service: str,
    candidate_service: str,
    path: List[str],
    time_range: TimeRange,
    source: str = "topology_agent",
) -> Optional[EvidenceUnit]:
    """토폴로지 경로 기반 추측 evidence.

    주의: topology evidence는 '관측'이 아니라 '구조적 추측'이므로
    severity 상한을 0.5로 둔다. (v7 scoring rule H4의 정신)
    경로상 존재만으로는 정답이 될 수 없다는 원칙.
    """
    if not path or candidate_service not in path or symptom_service not in path:
        return None

    # path distance: symptom으로부터 얼마나 멀리 있는가
    try:
        sym_idx = path.index(symptom_service)
        cand_idx = path.index(candidate_service)
        distance = abs(sym_idx - cand_idx)
    except ValueError:
        return None

    if distance == 0:
        # 후보가 symptom 자체인 경우는 증거가 아니다
        return None

    # 가까울수록 severity 높음. 단 상한 0.5.
    severity = min(0.5, 0.5 / distance)

    return EvidenceUnit(
        evidence_id=make_evidence_id(
            "topology",
            [symptom_service, candidate_service],
            time_range,
            "topology_proximity",
            extra=f"d={distance}",
        ),
        modality="topology",
        time_range=time_range,
        services=[candidate_service],
        anomaly_type="topology_proximity",
        severity=severity,
        observation={
            "symptom_service": symptom_service,
            "path_distance": distance,
            "path_length": len(path),
        },
        source=source,
        topology_path=list(path),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def build_collection_from_mcp_outputs(
    service_stats: Dict[str, Dict[str, Any]],
    metric_summaries: Optional[Dict[str, Any]] = None,
    log_records_by_service: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    time_range: Optional[TimeRange] = None,
) -> EvidenceCollection:
    """여러 MCP 출력을 한 번에 EvidenceCollection으로 변환.

    이 함수는 Phase 2의 MCP 서버에서 호출할 것이다. Phase 1에서는 factory
    조합 예시이자 테스트 용도다.

    Parameters
    ----------
    service_stats
        {service_name: stats_dict} 형태 (get_service_statistics의 결과).
    metric_summaries
        metric_repository.get_all_service_metric_summaries()의 결과. 기대 구조:
            {"services": {service_name: {...metric summary dict...}}}
    log_records_by_service
        {service_name: [log_record_dict, ...]} 형태.
    time_range
        관측 구간. 없으면 service_stats 또는 metric_summaries에서 추론
        시도하나, 명시적으로 주는 것이 안전하다.
    """
    collection = EvidenceCollection()
    if time_range is None:
        # 최소한의 fallback — 실전에서는 명시적 주입 권장
        raise ValueError(
            "time_range is required. Factory cannot infer it safely from "
            "heterogeneous MCP outputs."
        )

    # Log stats
    for svc, stats in (service_stats or {}).items():
        units = evidence_from_service_statistics(svc, stats, time_range)
        collection.extend(units)

    # Upstream-errored log records
    for svc, records in (log_records_by_service or {}).items():
        unit = evidence_from_log_records(svc, records, time_range)
        if unit:
            collection.add(unit)

    # Metric summaries
    if metric_summaries:
        svc_map = metric_summaries.get("services") or {}
        for svc, summ in svc_map.items():
            units = evidence_from_metric_summary(svc, summ, time_range)
            collection.extend(units)

    return collection


__all__ = [
    "build_collection_from_mcp_outputs",
    "evidence_from_log_records",
    "evidence_from_metric_summary",
    "evidence_from_service_statistics",
    "evidence_from_topology_path",
]
