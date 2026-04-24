"""
Observability MCP — Evidence-aware tools (Phase 2).

본 모듈은 기존 raw-data MCP tool (search_logs, get_metric_summary, ...)
위에 얹는 Evidence-Aware layer를 제공한다. Agent가 이 tool을 호출하면
raw data가 아니라 이미 구조화된 EvidenceUnit 컬렉션을 받는다.

설계 원칙
---------
1. 기존 tool은 유지 (backward compatibility). Phase 2는 **추가만** 한다.
2. Evidence 생성 로직은 common/evidence_factory.py에 이미 있다. 본 모듈은
   repository를 호출하여 raw data를 얻고, factory에 넘겨 EvidenceUnit으로
   변환하는 얇은 레이어 역할을 한다.
3. 반환은 JSON-serialisable dict (pydantic model_dump). MCP는 JSON-RPC
   로 직렬화하여 전달하며, 수신자(Agent)는 EvidenceCollection으로 재구성한다.
4. 시간 창 해석은 기존 metric tool과 동일한 규칙을 따른다:
   baseline_start/baseline_end가 오면 dual-window, 없으면 legacy split.

Phase 2 범위
------------
본 파일만 신규로 추가하고, server.py에 4개의 @mcp.tool()을 wrapping한다.
Agent 코드는 변경하지 않는다. Phase 3에서 Log Agent가 이 tool을 쓰기
시작한다.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root must be on sys.path so that we can import common/*
# (This file lives at mcp_servers/observability_mcp/app/evidence_tools.py,
#  which is deep; climb 4 levels to the repo root.)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.evidence import (  # noqa: E402
    EvidenceCollection,
    EvidenceUnit,
    TimeRange,
)
from common.evidence_factory import (  # noqa: E402
    evidence_from_log_records,
    evidence_from_metric_summary,
    evidence_from_service_statistics,
    evidence_from_topology_path,
)

from .repository import get_service_statistics as repo_get_service_statistics  # noqa: E402
from .repository import search_logs as repo_search_logs  # noqa: E402
from .metric_repository import get_all_service_metric_summaries as repo_get_all_metric  # noqa: E402


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_baseline_tuple(
    baseline_start: Optional[str],
    baseline_end: Optional[str],
) -> Optional[Tuple[str, str]]:
    """기존 MCP tool과 동일한 규칙 — 둘 다 있어야 dual-window로 간주."""
    if baseline_start and baseline_end:
        return (baseline_start, baseline_end)
    return None


def _effective_time_range(
    start: str,
    end: str,
    incident_start: Optional[str] = None,
    incident_end: Optional[str] = None,
) -> TimeRange:
    """Evidence에 기록할 '관측 시간 구간'을 결정한다.

    incident window가 명시되면 그것을 사용 (더 좁고 정확). 없으면 [start, end].
    """
    s = incident_start or start
    e = incident_end or end
    return TimeRange(start=s, end=e)


def _collection_to_payload(collection: EvidenceCollection) -> Dict[str, Any]:
    """MCP tool 반환용 JSON-serialisable dict.

    pydantic의 model_dump는 datetime 등 내부 타입도 처리하므로 그대로 사용.
    추가로 편의 필드(count, services_covered, modalities_present)를 넣어
    Agent가 parse 전에 빠르게 요약 볼 수 있게 한다.
    """
    return {
        "count": len(collection),
        "services_covered": collection.services_covered(),
        "modalities_present": collection.modalities_present(),
        "units": [u.model_dump(mode="json") for u in collection.units],
    }


# ---------------------------------------------------------------------------
# build_log_evidence — log modality
# ---------------------------------------------------------------------------

def build_log_evidence(
    start: str,
    end: str,
    log_file: Optional[str] = None,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
    incident_start: Optional[str] = None,
    incident_end: Optional[str] = None,
    focus_services: Optional[List[str]] = None,
) -> EvidenceCollection:
    """로그 기반 Evidence 생성.

    다음 세 가지 evidence type이 서비스별로 최대 한 건씩 생성될 수 있다:
      - volume_shift:       rate-normalised log count 급변
      - error_spike:        error_ratio 기준선 이상
      - keyword_distress:   timeout/retry/reset/latency 키워드 히트
      - dependency_failure: upstream 필드가 있는 에러 로그 집합

    factory에서 low-volume rate-norm artifact(v6 regression)을 방지하는
    조건이 이미 적용되어 있으므로, 이 함수 자체에는 별도 guard를 두지 않는다.

    Parameters
    ----------
    focus_services
        특정 서비스만 조사할 때. None이면 모든 서비스.
        Phase 4의 adaptive execution에서 "service X에 집중해서 추가 조사"
        요청이 들어올 때 사용된다.
    """
    # 1. dual-window statistics 호출 (기존 repository 그대로)
    baseline_range = _resolve_baseline_tuple(baseline_start, baseline_end)
    incident_range = (
        (incident_start, incident_end)
        if (incident_start and incident_end) else None
    )

    stats_result = repo_get_service_statistics(
        start=start,
        end=end,
        log_file=log_file,
        baseline_range=baseline_range,
        incident_range=incident_range,
    )

    # stats_result structure: {"window": ..., "mode": ..., "services": {svc: {...}}, ...}
    services_stats: Dict[str, Dict[str, Any]] = stats_result.get("services") or {}

    # 2. effective time range for evidence tagging
    effective_tr = _effective_time_range(start, end, incident_start, incident_end)

    # 3. per-service factory
    collection = EvidenceCollection()

    for svc, stats in services_stats.items():
        # focus filter
        if focus_services and svc not in focus_services:
            continue

        # has_activity 없는 서비스는 skip (factory에서도 걸러지지만 빠른 path)
        if not stats.get("has_activity", True):
            continue

        units = evidence_from_service_statistics(
            service=svc,
            stats=stats,
            time_range=effective_tr,
        )
        collection.extend(units)

    # 4. Dependency-failure evidence는 실제 log record를 훑어서 upstream 필드를
    #    검사해야 한다. volume이 클 수 있으니 stats에서 error_ratio > 0인
    #    서비스에 대해서만 수행.
    for svc, stats in services_stats.items():
        if focus_services and svc not in focus_services:
            continue
        if (stats.get("error_ratio") or 0) <= 0:
            continue

        try:
            # repository.search_logs expects ISO strings; we reuse the outer
            # window since upstream-field presence doesn't care about the
            # baseline/incident split.
            records = repo_search_logs(
                service=svc, start=start, end=end, log_file=log_file,
            )
            record_dicts = [r.model_dump() for r in records]
            unit = evidence_from_log_records(
                service=svc,
                records=record_dicts,
                time_range=effective_tr,
            )
            if unit:
                collection.add(unit)
        except Exception as exc:
            log.warning(
                "dependency_failure evidence collection failed for %s: %s",
                svc, exc,
            )

    log.info(
        "build_log_evidence: window=[%s, %s] baseline=%s services=%d evidence=%d",
        start, end, baseline_range,
        len(services_stats), len(collection),
    )
    return collection


# ---------------------------------------------------------------------------
# build_metric_evidence — metric modality
# ---------------------------------------------------------------------------

def build_metric_evidence(
    start: str,
    end: str,
    metrics_file: Optional[str] = None,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
    incident_start: Optional[str] = None,
    incident_end: Optional[str] = None,
    focus_services: Optional[List[str]] = None,
) -> EvidenceCollection:
    """Metric (Prometheus / Istio) 기반 Evidence 생성.

    다음 세 가지 evidence type이 서비스별로 최대 한 건씩 생성:
      - resource_saturation:  CPU z-score 또는 mem_jump_ratio
      - latency_degradation:  p95 또는 p99 delta
      - network_degradation:  packet drops 또는 istio error_delta

    metrics_file 경로는 다음 순서로 결정된다:
      (1) 함수 인자 metrics_file
      (2) 환경변수 OBSERVABILITY_METRICS_FILE
      (3) None → metric_repository가 data not found 처리

    Severity 정규화는 factory에서 담당 (CPU z-score → sigmoid, 등등).
    """
    baseline_range = _resolve_baseline_tuple(baseline_start, baseline_end)
    effective_tr = _effective_time_range(start, end, incident_start, incident_end)

    # Metric repository 호출 — incident 창에 대해 per-service 요약
    m_start = incident_start or start
    m_end = incident_end or end

    try:
        metric_result = repo_get_all_metric(
            start=m_start, end=m_end,
            metrics_file=metrics_file,
            baseline_range=baseline_range,
        )
    except Exception as exc:
        # Metric이 없는 환경에서도 graceful하게 빈 collection 반환
        log.info("metric summary unavailable, returning empty collection: %s", exc)
        return EvidenceCollection()

    services_map: Dict[str, Any] = metric_result.get("services") or {}

    collection = EvidenceCollection()
    for svc, summary in services_map.items():
        if focus_services and svc not in focus_services:
            continue

        units = evidence_from_metric_summary(
            service=svc,
            summary=summary,
            time_range=effective_tr,
        )
        collection.extend(units)

    log.info(
        "build_metric_evidence: window=[%s, %s] services=%d evidence=%d",
        m_start, m_end, len(services_map), len(collection),
    )
    return collection


# ---------------------------------------------------------------------------
# build_topology_evidence — topology modality (structural guess)
# ---------------------------------------------------------------------------

def build_topology_evidence(
    symptom_service: str,
    candidate_services: List[str],
    path: List[str],
    start: str,
    end: str,
) -> EvidenceCollection:
    """토폴로지 경로 기반 추측 Evidence.

    주의: 이는 관측이 아닌 '구조적 추측'이다 (v7 H4 원칙). severity ≤ 0.5로
    상한되어 있어, 단독으로는 최종 후보가 될 수 없다. 관측 증거(log/metric)
    와 결합될 때만 의미 있는 증거가 된다.
    """
    tr = TimeRange(start=start, end=end)
    collection = EvidenceCollection()

    for candidate in candidate_services:
        unit = evidence_from_topology_path(
            symptom_service=symptom_service,
            candidate_service=candidate,
            path=path,
            time_range=tr,
        )
        if unit:
            collection.add(unit)

    log.info(
        "build_topology_evidence: symptom=%s candidates=%d path_len=%d evidence=%d",
        symptom_service, len(candidate_services), len(path), len(collection),
    )
    return collection


# ---------------------------------------------------------------------------
# build_evidence_collection — unified entry point
# ---------------------------------------------------------------------------

def build_evidence_collection(
    start: str,
    end: str,
    log_file: Optional[str] = None,
    metrics_file: Optional[str] = None,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
    incident_start: Optional[str] = None,
    incident_end: Optional[str] = None,
    focus_services: Optional[List[str]] = None,
    symptom_service: Optional[str] = None,
    topology_path: Optional[List[str]] = None,
    candidate_services: Optional[List[str]] = None,
) -> EvidenceCollection:
    """Log + Metric + (선택) Topology를 한 번에 수집한 통합 Collection.

    Phase 3에서 Log Agent (혹은 evidence-aware Log Agent)가 이 tool을
    호출하여 한 번의 요청으로 전체 증거를 받을 수 있게 한다.

    topology evidence는 `symptom_service`, `topology_path`, `candidate_services`
    세 인자가 모두 주어질 때만 생성된다.

    반환
    ----
    단일 EvidenceCollection. 증거 ID 기준으로 중복이 제거되어 있다.
    """
    combined = EvidenceCollection()

    # log
    log_col = build_log_evidence(
        start=start, end=end,
        log_file=log_file,
        baseline_start=baseline_start, baseline_end=baseline_end,
        incident_start=incident_start, incident_end=incident_end,
        focus_services=focus_services,
    )
    combined.extend(log_col.units)

    # metric
    metric_col = build_metric_evidence(
        start=start, end=end,
        metrics_file=metrics_file,
        baseline_start=baseline_start, baseline_end=baseline_end,
        incident_start=incident_start, incident_end=incident_end,
        focus_services=focus_services,
    )
    combined.extend(metric_col.units)

    # topology (optional)
    if symptom_service and topology_path and candidate_services:
        topo_col = build_topology_evidence(
            symptom_service=symptom_service,
            candidate_services=candidate_services,
            path=topology_path,
            start=start, end=end,
        )
        combined.extend(topo_col.units)

    log.info(
        "build_evidence_collection: total=%d (log+metric+topology)",
        len(combined),
    )
    return combined


# ---------------------------------------------------------------------------
# Public payload functions — wrapped by server.py @mcp.tool()
# ---------------------------------------------------------------------------
#
# server.py는 이 함수들을 호출해 dict로 직렬화하여 JSON-RPC로 반환한다.
# collection이 EvidenceUnit 리스트를 담고 있으며, 각 unit은 pydantic의
# model_dump(mode="json")로 JSON 친화적 dict로 바뀐다.

def get_log_evidence_payload(
    start: str, end: str, log_file: Optional[str] = None,
    baseline_start: Optional[str] = None, baseline_end: Optional[str] = None,
    incident_start: Optional[str] = None, incident_end: Optional[str] = None,
    focus_services: Optional[List[str]] = None,
) -> Dict[str, Any]:
    col = build_log_evidence(
        start=start, end=end, log_file=log_file,
        baseline_start=baseline_start, baseline_end=baseline_end,
        incident_start=incident_start, incident_end=incident_end,
        focus_services=focus_services,
    )
    return _collection_to_payload(col)


def get_metric_evidence_payload(
    start: str, end: str, metrics_file: Optional[str] = None,
    baseline_start: Optional[str] = None, baseline_end: Optional[str] = None,
    incident_start: Optional[str] = None, incident_end: Optional[str] = None,
    focus_services: Optional[List[str]] = None,
) -> Dict[str, Any]:
    col = build_metric_evidence(
        start=start, end=end, metrics_file=metrics_file,
        baseline_start=baseline_start, baseline_end=baseline_end,
        incident_start=incident_start, incident_end=incident_end,
        focus_services=focus_services,
    )
    return _collection_to_payload(col)


def get_topology_evidence_payload(
    symptom_service: str,
    candidate_services: List[str],
    path: List[str],
    start: str, end: str,
) -> Dict[str, Any]:
    col = build_topology_evidence(
        symptom_service=symptom_service,
        candidate_services=candidate_services,
        path=path,
        start=start, end=end,
    )
    return _collection_to_payload(col)


def get_evidence_collection_payload(
    start: str, end: str,
    log_file: Optional[str] = None,
    metrics_file: Optional[str] = None,
    baseline_start: Optional[str] = None, baseline_end: Optional[str] = None,
    incident_start: Optional[str] = None, incident_end: Optional[str] = None,
    focus_services: Optional[List[str]] = None,
    symptom_service: Optional[str] = None,
    topology_path: Optional[List[str]] = None,
    candidate_services: Optional[List[str]] = None,
) -> Dict[str, Any]:
    col = build_evidence_collection(
        start=start, end=end,
        log_file=log_file, metrics_file=metrics_file,
        baseline_start=baseline_start, baseline_end=baseline_end,
        incident_start=incident_start, incident_end=incident_end,
        focus_services=focus_services,
        symptom_service=symptom_service,
        topology_path=topology_path,
        candidate_services=candidate_services,
    )
    return _collection_to_payload(col)


__all__ = [
    "build_evidence_collection",
    "build_log_evidence",
    "build_metric_evidence",
    "build_topology_evidence",
    "get_evidence_collection_payload",
    "get_log_evidence_payload",
    "get_metric_evidence_payload",
    "get_topology_evidence_payload",
]
