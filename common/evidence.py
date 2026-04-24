"""
Evidence-Aware RCA — Evidence Unit schema (Phase 1).

이 모듈은 MCP / A2A 레이어를 단순 데이터 전송 프로토콜이 아닌
'RCA 증거의 구조적 계약'으로 확장하기 위한 기본 데이터 단위를 정의한다.

설계 원칙
---------
1. Modality 독립성: log / metric / trace / topology 네 가지 관측 원천
   각각이 동일한 스키마로 통합된다.
2. Provenance 추적: 모든 EvidenceUnit은 고유 ID를 가지며, 다른 증거와
   상호 참조(supporting_refs)를 통해 역추적 가능해야 한다.
3. LLM 친화성: EvidenceUnit은 그 자체로 LLM 프롬프트에 삽입 가능한
   compact한 자연어 요약(to_prompt_snippet)을 생성할 수 있다.
4. JSON 직렬화 보장: A2A 메시지 본문으로 그대로 전송 가능해야 한다.

Phase 1 범위
------------
본 파일은 스키마 정의와 helpers만 제공한다.
기존 Agent 코드는 건드리지 않는다. Phase 2에서 Observability MCP가
이 스키마를 실제로 반환하고, Phase 3에서 Agent들이 소비한다.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Type aliases — explicit enumerations keep the schema self-documenting
# ---------------------------------------------------------------------------

Modality = Literal["log", "metric", "trace", "topology"]

AnomalyType = Literal[
    # Log-centric
    "error_spike",          # ERROR/WARN 로그 급증
    "volume_shift",         # 로그 볼륨 급변 (rate-normalised)
    "keyword_distress",     # timeout / retry / reset / refused 키워드
    "dependency_failure",   # upstream 호출 실패 (5xx with upstream field)

    # Metric-centric
    "resource_saturation",  # CPU / Memory 포화
    "latency_degradation",  # p95 / p99 tail latency 증가
    "network_degradation",  # packet drops / socket 급증

    # Trace-centric
    "span_error",           # trace 내 error span
    "span_latency",         # 특정 span 구간의 지연

    # Topology-centric
    "topology_proximity",   # symptom으로부터의 path 거리
    "topology_dependency",  # 직접 dependency (구조적 추측)
]

Severity = float  # 0.0 ~ 1.0. 실제 값 범위는 field_validator에서 강제.


# ---------------------------------------------------------------------------
# TimeRange
# ---------------------------------------------------------------------------

class TimeRange(BaseModel):
    """Evidence의 관측 시간 구간.

    ISO-8601 문자열을 그대로 쓰는 이유는:
      (1) JSON 직렬화 시 type 변환이 불필요하고
      (2) 기존 MCP의 search_logs(start, end) API와 호환되며
      (3) human-readable이다.

    내부 비교 / 계산이 필요하면 .to_datetime()을 사용한다.
    """
    model_config = ConfigDict(frozen=True)

    start: str  # ISO-8601, e.g. "2026-04-23T13:02:30+09:00"
    end: str

    @field_validator("start", "end")
    @classmethod
    def _iso_parseable(cls, v: str) -> str:
        # datetime.fromisoformat raises ValueError if malformed.
        # We don't normalise here; we keep the original string so that
        # downstream tools that do their own parsing see exactly what
        # the upstream provided.
        datetime.fromisoformat(v)
        return v

    def to_datetime(self) -> Tuple[datetime, datetime]:
        return (
            datetime.fromisoformat(self.start),
            datetime.fromisoformat(self.end),
        )

    def duration_seconds(self) -> float:
        s, e = self.to_datetime()
        return (e - s).total_seconds()

    def contains(self, ts_iso: str) -> bool:
        """주어진 ISO 타임스탬프가 이 구간에 포함되는가."""
        try:
            ts = datetime.fromisoformat(ts_iso)
        except ValueError:
            return False
        s, e = self.to_datetime()
        return s <= ts <= e

    def overlaps(self, other: "TimeRange") -> bool:
        s1, e1 = self.to_datetime()
        s2, e2 = other.to_datetime()
        return s1 <= e2 and s2 <= e1


# ---------------------------------------------------------------------------
# EvidenceUnit
# ---------------------------------------------------------------------------

class EvidenceUnit(BaseModel):
    """RCA 증거의 기본 단위.

    핵심 필드
    --------
    evidence_id
        불변 식별자. Agent 간 메시지에서 증거를 참조할 때 사용.
        형식: "ev_{modality}_{short_hash}" (예: "ev_log_a7f3").
    modality
        증거 원천 종류 (log / metric / trace / topology).
    time_range
        증거가 관측된 시간 구간.
    services
        이 증거가 지목하는 서비스 목록. 보통 1개이나 dependency
        failure 같은 경우 여러 개가 될 수 있다.
    anomaly_type
        무엇에 대한 이상인가 (error_spike, latency_degradation, ...).
    severity
        [0, 1] 정규화된 심각도. 서로 다른 modality 간 비교 가능하도록
        설계됨 (예: CPU z=289 → severity=1.0, p95_delta=20ms → severity=0.3).
    observation
        증거의 정량적 근거 (숫자, 카운트, 통계량 등). modality에 따라
        구조는 자유롭되, 재현성을 위해 raw data가 아닌 집계 통계량을
        담는 것을 원칙으로 한다.
    source
        어느 MCP tool 또는 Agent가 이 증거를 생성했는가.
    topology_path
        해당되는 경우의 서비스 호출 경로. 주로 topology evidence가 사용.
    supporting_refs
        이 증거를 보강하는 다른 증거의 ID 목록.
    raw_samples
        LLM이 참고할 수 있는 최소한의 원시 샘플 (예: 에러 로그 2-3줄).
        길이 제한을 두어 프롬프트 폭증을 방지.

    불변성
    ------
    EvidenceUnit은 생성 이후 변경되지 않는다 (pydantic frozen). 추가
    정보가 생기면 새 EvidenceUnit을 만들어 supporting_refs에 걸어라.

    JSON 직렬화
    -----------
    pydantic v2의 model_dump_json() / model_validate_json() 로 왕복 가능.
    A2A 메시지에 그대로 실을 수 있다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(
        default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}",
        description="불변 식별자. _make_evidence_id를 통해 결정적 생성도 가능.",
    )
    modality: Modality
    time_range: TimeRange
    services: List[str]
    anomaly_type: AnomalyType
    severity: Severity

    observation: Dict[str, Any]
    source: str

    topology_path: Optional[List[str]] = None
    supporting_refs: List[str] = Field(default_factory=list)
    raw_samples: List[str] = Field(default_factory=list)

    # Phase 4a: 시간적 선후 관계 (논문 Section 3).
    # 이 증거보다 시간상 먼저 일어난 것으로 판단되는 다른 증거의 ID들.
    # 주의: time_range.start의 단순 비교로 자동 유도할 수도 있지만, 명시
    # 관계를 저장함으로써 Verifier의 temporal consistency check가 매번
    # time_range를 비교하지 않아도 되도록 한다. 비어있을 수 있다 (대부분).
    # 예: API error_spike(t=13:03)의 preceded_by = [db_timeout(t=13:02).id]
    preceded_by: List[str] = Field(
        default_factory=list,
        description="이 증거보다 시간상 먼저 발생한 증거 ID 목록",
    )

    # ---------- Validators -----------------------------------------------

    @field_validator("severity")
    @classmethod
    def _severity_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"severity must be in [0, 1], got {v}")
        return float(v)

    @field_validator("services")
    @classmethod
    def _services_nonempty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("services must contain at least one service name")
        cleaned = [s.strip() for s in v if s and s.strip()]
        if not cleaned:
            raise ValueError("services contains only empty strings")
        return cleaned

    @field_validator("raw_samples")
    @classmethod
    def _raw_samples_bounded(cls, v: List[str]) -> List[str]:
        # Prevent prompt-bombing: cap at 5 samples, 240 chars each.
        out = []
        for s in (v or [])[:5]:
            s = str(s)
            if len(s) > 240:
                s = s[:237] + "..."
            out.append(s)
        return out

    # ---------- Helpers --------------------------------------------------

    def to_prompt_snippet(self, max_len: int = 200) -> str:
        """LLM 프롬프트 한 줄 요약.

        예:
            "[ev_log_a7f3 | log | error_spike | sev=0.85 | services=checkoutservice
             | 2026-04-23T13:02:30/13:03:20 | error_count=142 baseline=3]"

        프롬프트 길이 폭증을 방지하기 위해 max_len로 잘라낸다.
        """
        obs_summary = ", ".join(
            f"{k}={_fmt_obs(v)}"
            for k, v in list(self.observation.items())[:3]
        )
        snippet = (
            f"[{self.evidence_id} | {self.modality} | {self.anomaly_type} "
            f"| sev={self.severity:.2f} "
            f"| services={','.join(self.services)} "
            f"| {self.time_range.start}..{self.time_range.end} "
            f"| {obs_summary}]"
        )
        if len(snippet) > max_len:
            snippet = snippet[: max_len - 3] + "..."
        return snippet

    def involves_service(self, service: str) -> bool:
        target = service.strip().lower()
        return any(s.strip().lower() == target for s in self.services)

    def with_supporting(self, *refs: str) -> "EvidenceUnit":
        """supporting_refs가 덧붙여진 새 사본을 반환 (immutable pattern)."""
        new_refs = list(dict.fromkeys([*self.supporting_refs, *refs]))
        return self.model_copy(update={"supporting_refs": new_refs})


def _fmt_obs(v: Any) -> str:
    """observation 값의 프롬프트 친화적 포맷."""
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"
    return str(v)


# ---------------------------------------------------------------------------
# ID generation — deterministic when you need reproducibility
# ---------------------------------------------------------------------------

def make_evidence_id(
    modality: Modality,
    services: List[str],
    time_range: TimeRange,
    anomaly_type: AnomalyType,
    extra: Optional[str] = None,
) -> str:
    """결정적 evidence_id 생성.

    같은 (modality, services, time_range, anomaly_type)에 대해 항상 같은
    ID를 만든다. 테스트 재현성 / 캐시 활용 / 중복 제거 목적.

    Hash 충돌 확률: 8 hex chars → ~4.3 billion 조합. 한 incident 당
    evidence 수가 수백 개를 넘지 않으므로 충돌 확률은 무시 가능 수준.
    """
    key_parts = [
        modality,
        ",".join(sorted(set(services))),
        time_range.start,
        time_range.end,
        anomaly_type,
        extra or "",
    ]
    digest = hashlib.sha1("|".join(key_parts).encode("utf-8")).hexdigest()[:8]
    return f"ev_{modality}_{digest}"


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

class EvidenceCollection(BaseModel):
    """여러 EvidenceUnit을 묶어 다루기 위한 컨테이너.

    중복 제거 (evidence_id 기준) 및 서비스/모달리티별 인덱싱을 제공한다.
    Phase 2의 MCP tool이 반환하는 단위이기도 하다.
    """

    model_config = ConfigDict(extra="forbid")

    units: List[EvidenceUnit] = Field(default_factory=list)

    def add(self, unit: EvidenceUnit) -> None:
        """중복 ID이면 무시한다."""
        existing = {u.evidence_id for u in self.units}
        if unit.evidence_id not in existing:
            self.units.append(unit)

    def extend(self, units: List[EvidenceUnit]) -> None:
        for u in units:
            self.add(u)

    def by_service(self, service: str) -> List[EvidenceUnit]:
        return [u for u in self.units if u.involves_service(service)]

    def by_modality(self, modality: Modality) -> List[EvidenceUnit]:
        return [u for u in self.units if u.modality == modality]

    def services_covered(self) -> List[str]:
        svcs: List[str] = []
        for u in self.units:
            for s in u.services:
                if s not in svcs:
                    svcs.append(s)
        return svcs

    def modalities_present(self) -> List[Modality]:
        return sorted({u.modality for u in self.units})

    def strongest_by_service(self, service: str) -> Optional[EvidenceUnit]:
        """해당 서비스의 증거 중 severity가 가장 높은 것."""
        candidates = self.by_service(service)
        if not candidates:
            return None
        return max(candidates, key=lambda u: u.severity)

    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self):
        return iter(self.units)


# ---------------------------------------------------------------------------
# Completeness / consistency scoring — used by A2A contract
# ---------------------------------------------------------------------------

def completeness_score(
    collection: EvidenceCollection,
    candidate_services: List[str],
    required_modalities: Optional[List[Modality]] = None,
) -> float:
    """후보 서비스들에 대한 증거의 '완전도' 점수 [0, 1].

    이 점수는 Orchestrator의 adaptive execution 루프에서 재호출 여부를
    결정할 때 사용된다 (Phase 4).

    점수 구성:
      (a) 각 후보 서비스마다 최소 1개의 log 또는 metric evidence가 있는가 (40%)
      (b) required_modalities가 주어졌다면 전부 커버되었는가 (40%)
      (c) 후보 간 증거 중복 제거 후 순 증거 수 / 기대 최소치 (20%)

    완벽한 점수(1.0)는 "추가 조회가 필요 없다"는 뜻이지, "정답이 맞다"는
    뜻이 아니다. 정확도는 RCA Agent가 판단한다.
    """
    if not candidate_services:
        return 0.0

    # (a) per-candidate coverage
    covered = 0
    for svc in candidate_services:
        evs = collection.by_service(svc)
        observed = any(u.modality in ("log", "metric", "trace") for u in evs)
        if observed:
            covered += 1
    a_ratio = covered / len(candidate_services)

    # (b) required modalities
    if required_modalities:
        present = set(collection.modalities_present())
        needed = set(required_modalities)
        b_ratio = len(needed & present) / len(needed) if needed else 1.0
    else:
        b_ratio = 1.0

    # (c) evidence density — 후보당 최소 2개 evidence를 이상적으로 본다
    expected = max(1, 2 * len(candidate_services))
    c_ratio = min(1.0, len(collection) / expected)

    return round(0.4 * a_ratio + 0.4 * b_ratio + 0.2 * c_ratio, 4)


__all__ = [
    "AnomalyType",
    "EvidenceCollection",
    "EvidenceUnit",
    "Modality",
    "Severity",
    "TimeRange",
    "completeness_score",
    "make_evidence_id",
]
