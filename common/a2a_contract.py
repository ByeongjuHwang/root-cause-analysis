"""
Evidence-Aware RCA — A2A Contract schema (Phase 1).

본 모듈은 Agent 간(A2A) 메시지를 '단순 응답'에서 'evidence-carrying
contract'로 승격시키기 위한 표준 데이터 구조를 정의한다.

기존 구조의 한계
---------------
지금까지 Agent 응답은 각자의 ad-hoc JSON이었다:
  - Log Agent:    {"anomalous_services": [...], "suspected_downstream": "X", "confidence": 0.8}
  - RCA Agent:    {"final_candidates": [{"cause_service": "X", ...}], ...}
  - Verifier:     {"adjusted_candidates": [...], "reasoning": "..."}

이 구조의 문제:
  1. 각 Agent의 응답이 서로 다른 key를 쓰므로 Orchestrator가 통일된
     재호출 판단을 할 수 없다.
  2. 응답에 'why'는 있어도 'what evidence'가 없어, 다른 Agent가 그
     판단을 역추적하기 어렵다.
  3. confidence만 있고 'how complete is the evidence'가 없어, 낮은
     confidence가 '증거 부족' 때문인지 '모순되는 증거' 때문인지
     구분 불가.

본 모듈의 해결책
----------------
모든 Agent의 응답을 AgentResponse 단일 스키마로 통일한다. 각 후보는
다음을 명시해야 한다:
  - supporting_evidence: 이 후보를 지지하는 Evidence ID 목록
  - assumptions:         이 판단에 깔린 전제 조건 (예: baseline_window=480s)
  - missing_evidence:    더 있으면 좋겠는 증거의 종류
  - confidence + completeness_score: 두 차원의 메타 평가

Orchestrator는 completeness_score와 missing_evidence를 보고 추가
MCP tool 호출이나 Agent 재호출을 결정한다 (Phase 4).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .evidence import EvidenceCollection, EvidenceUnit


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AgentName = Literal[
    "log_agent",
    "topology_agent",
    "rca_agent",
    "verifier_agent",
    "orchestrator",
]


# FailureMode — 논문 Section 3에 명시된 "failure mode" 필드.
# anomaly_type과는 다름: anomaly_type은 관측된 증상(error_spike 등)이고,
# failure_mode는 그 증상의 원인이 되는 장애 패턴이다. 즉:
#   anomaly_type = WHAT (observed symptom)
#   failure_mode  = HOW (causal mechanism)
# 예시: anomaly_type=latency_degradation + failure_mode=resource_exhaustion
#       anomaly_type=error_spike        + failure_mode=cascading_failure
FailureMode = Literal[
    "resource_exhaustion",      # CPU/Mem/Disk 고갈
    "cascading_failure",        # 하위 서비스 실패의 상위 전파
    "network_partition",        # 네트워크 분리/지연
    "dependency_timeout",       # 외부 의존성 호출 timeout
    "deadlock_or_saturation",   # 동시성 문제, 쓰레드/커넥션 풀 고갈
    "configuration_error",      # 잘못된 설정/배포
    "data_corruption",          # DB / cache 데이터 이상
    "retry_storm",              # 재시도 폭주
    "noisy_neighbor",           # 인접 서비스의 자원 점유
    "partial_outage",           # 일부 인스턴스 다운
    "unknown",                  # 판정 불가 / 미지 장애 패턴
]


# ---------------------------------------------------------------------------
# Candidate — one proposed root cause with its evidence trail
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    """Agent가 제시하는 후보 원인 1건.

    supporting_evidence / assumptions / missing_evidence 세 필드가 이
    스키마의 핵심이다:

      - supporting_evidence: 이 판단을 지지하는 EvidenceUnit ID 목록.
        EvidenceCollection과 대조하여 실제 존재하는지 검증 가능 (역추적성).
      - assumptions:         이 판단이 성립하려면 참이어야 하는 전제.
        예: "baseline window는 480s이며 그 안에 drift가 없다",
            "upstream 필드가 서비스 이름을 정확히 가리킨다".
      - missing_evidence:    이 판단을 더 확고히 하려면 필요한 증거의 종류.
        예: "trace_data_for_user-db", "metric_for_cache-layer".
        Orchestrator가 이를 읽고 해당 MCP tool을 추가 호출한다.

    confidence vs severity
    ----------------------
    Candidate.confidence는 "이 판단이 맞을 가능성" (Agent의 메타 평가).
    EvidenceUnit.severity는 "관측된 이상의 심각도" (객관적).
    둘은 다른 차원이다.
    """

    model_config = ConfigDict(extra="forbid")

    service: str = Field(..., description="후보 원인 서비스 이름")
    confidence: float = Field(..., description="[0, 1] Agent의 자기평가")

    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="이 후보를 지지하는 EvidenceUnit ID 목록",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="이 판단이 전제하는 가정 (사람이 읽는 문장)",
    )
    missing_evidence: List[str] = Field(
        default_factory=list,
        description="더 있으면 좋을 증거 종류 (Orchestrator가 해석)",
    )

    reasoning: str = Field(
        default="",
        description="자연어 이유. LLM이 사후 설명용으로 쓰는 필드.",
    )

    # 선택적: RCA Agent가 분석한 전파 경로
    topology_path: Optional[List[str]] = Field(
        default=None,
        description="해당 후보를 지목하는 토폴로지 경로 (있을 경우)",
    )

    # Phase 4a: failure mode classification.
    # anomaly_type과 구분되는 개념: anomaly_type은 observed symptom,
    # failure_mode는 causal mechanism. 논문 Section 3에 명시된 필드.
    failure_mode: Optional[FailureMode] = Field(
        default=None,
        description="장애 패턴 분류 (resource_exhaustion, cascading_failure, ...)",
    )

    @field_validator("confidence")
    @classmethod
    def _conf_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return float(v)

    @field_validator("service")
    @classmethod
    def _service_nonempty(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("service must be non-empty")
        return cleaned

    def evidence_count(self) -> int:
        return len(self.supporting_evidence)

    def is_evidence_backed(self) -> bool:
        """가장 기본적인 판정 규칙: 증거가 하나라도 있는가."""
        return len(self.supporting_evidence) > 0

    def has_missing_evidence(self) -> bool:
        return len(self.missing_evidence) > 0


# ---------------------------------------------------------------------------
# Consistency checks — Verifier Agent의 일관성 판정 결과
# ---------------------------------------------------------------------------

class ConsistencyChecks(BaseModel):
    """Verifier Agent가 후보에 대해 수행하는 4차원 일관성 검사.

    각 차원은 True(통과) / False(불일치) / None(판단 불가)의 3-value.
    '판단 불가'는 관련 증거 자체가 없을 때 쓴다. False와 None은 다르다:
      - False: 증거는 있으나 모순된다 → drop 정당화
      - None:  증거가 없음 → missing_evidence로 보고하여 재조회 유도

    4차원 (논문 Section 5에 명시)
    ---------------------------
    temporal
        시간적 타당성: 원인 evidence의 시각이 symptom보다 먼저인가.
        (원인이 결과보다 시간상 먼저라는 기본 물리 법칙)
    topological
        토폴로지 도달성: 후보 서비스가 symptom 서비스의 upstream /
        transitive dependency인가. (구조상 불가능한 경로는 배제)
    modality
        교차 맥락 일관성: 서로 다른 modality의 증거가 같은 서비스를
        지목하는가. (log + metric + trace 교차 검증)
        논문의 "교차 맥락 일관성"에 대응.
    counter_evidence
        반례 존재 여부: 이 후보를 반증하는 증거가 있는가.
        True = 반례 없음 (후보 유지 OK)
        False = 반례 존재 (후보 기각)
        None = 반례 검사 불가
        논문의 "반례 존재 여부"에 대응. 예: 후보 서비스에 "정상 처리 중"
        이라는 evidence가 있거나, CPU/Mem 모두 정상 범위인 경우.
    """

    model_config = ConfigDict(extra="forbid")

    temporal: Optional[bool] = None
    topological: Optional[bool] = None
    modality: Optional[bool] = None
    counter_evidence: Optional[bool] = None

    def passed(self) -> bool:
        """모든 수행된 검사가 True인가 (None 제외)."""
        results = [
            self.temporal, self.topological,
            self.modality, self.counter_evidence,
        ]
        performed = [r for r in results if r is not None]
        if not performed:
            return False
        return all(performed)

    def failed_dimensions(self) -> List[str]:
        """False로 판정된 차원 이름 목록."""
        out = []
        for dim in ("temporal", "topological", "modality", "counter_evidence"):
            if getattr(self, dim) is False:
                out.append(dim)
        return out

    def skipped_dimensions(self) -> List[str]:
        """None(판단 불가)인 차원 목록."""
        out = []
        for dim in ("temporal", "topological", "modality", "counter_evidence"):
            if getattr(self, dim) is None:
                out.append(dim)
        return out


# ---------------------------------------------------------------------------
# AgentResponse — the unified contract for all agents
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """모든 Agent가 따르는 표준 응답 스키마.

    이 스키마로 통일됨으로써:
      1. Orchestrator가 Agent 종류와 무관하게 같은 로직으로 후속 호출을
         결정할 수 있다.
      2. 하류 Agent가 상류 Agent의 판단을 역추적할 수 있다
         (supporting_evidence → EvidenceCollection 조회).
      3. 로그/메트릭에 기반하지 않는 추측성 판단이 구조적으로 구분된다
         (supporting_evidence가 비어있으면 evidence-backed가 아님).

    evidence_collection
    -------------------
    이 Agent가 생성했거나 참조한 EvidenceUnit들의 집합. Orchestrator는
    여러 Agent의 response를 merge하여 전역 EvidenceCollection을 유지한다.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName
    request_id: str = Field(..., description="incident_id 또는 대응 요청 ID")

    candidates: List[Candidate] = Field(default_factory=list)
    evidence_collection: EvidenceCollection = Field(
        default_factory=EvidenceCollection
    )

    completeness_score: float = Field(
        default=0.0,
        description="[0, 1] 현재까지 수집한 증거의 완전도",
    )

    # Verifier 전용 (다른 Agent는 None)
    consistency_checks: Optional[ConsistencyChecks] = Field(
        default=None,
        description="Verifier의 4-dim 일관성 판정 결과",
    )

    # Adaptive execution을 위한 힌트
    recommended_next_actions: List[str] = Field(
        default_factory=list,
        description="Agent가 제안하는 추가 수행 (예: 'query_trace_for_svcX')",
    )

    # Debug / explainability
    reasoning: str = Field(
        default="",
        description="자연어 요약. 디버그 및 논문 figure용.",
    )

    @field_validator("completeness_score")
    @classmethod
    def _score_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"completeness_score must be in [0, 1], got {v}")
        return float(v)

    # ---------- Query helpers -------------------------------------------

    def top_candidate(self) -> Optional[Candidate]:
        """confidence 기준 최상위 후보."""
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.confidence)

    def evidence_backed_candidates(self) -> List[Candidate]:
        return [c for c in self.candidates if c.is_evidence_backed()]

    def needs_more_evidence(self, threshold: float = 0.7) -> bool:
        """Orchestrator가 재호출 여부를 결정할 때 쓰는 primary gate."""
        return self.completeness_score < threshold

    def all_missing_evidence(self) -> List[str]:
        """모든 후보의 missing_evidence를 합쳐 중복 제거."""
        out: List[str] = []
        for c in self.candidates:
            for m in c.missing_evidence:
                if m not in out:
                    out.append(m)
        return out

    def confidence_gap(self) -> Optional[float]:
        """top-1과 top-2의 confidence 차이. 1개 이하면 None.

        작은 gap은 'Agent가 두 후보 사이에서 고민 중'이라는 신호 →
        Orchestrator가 차별적 증거를 추가 조회하도록 유도할 수 있다.
        """
        if len(self.candidates) < 2:
            return None
        sorted_c = sorted(
            self.candidates, key=lambda c: c.confidence, reverse=True
        )
        return sorted_c[0].confidence - sorted_c[1].confidence

    # ---------- Validation against evidence_collection -------------------

    def validate_evidence_refs(self) -> List[str]:
        """모든 후보의 supporting_evidence가 evidence_collection에 실제로
        존재하는지 검증. Dangling reference를 반환한다.

        이는 self-consistency check다: Agent가 존재하지 않는 증거 ID를
        언급하면 환각 가능성이 있다.
        """
        known_ids = {u.evidence_id for u in self.evidence_collection.units}
        dangling: List[str] = []
        for cand in self.candidates:
            for ref in cand.supporting_evidence:
                if ref not in known_ids:
                    dangling.append(ref)
        return dangling


# ---------------------------------------------------------------------------
# AgentRequest — optional but useful for logging / replay
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    """Orchestrator가 Agent를 호출할 때 보내는 표준 요청 포맷.

    Phase 1에서는 schema만 정의. Phase 3에서 실제 사용.

    include_evidence_refs
    ---------------------
    이전 Agent가 수집한 EvidenceUnit ID들. 현재 Agent가 이들을 참조하며
    새 증거를 수집할 수 있다 (cumulative evidence).

    focus_services
    --------------
    Orchestrator가 특정 서비스에 대한 추가 조사를 요청할 때 사용.
    재호출 루프에서 채워짐. 처음 호출 때는 None (= 전수조사).
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName
    request_id: str
    incident_summary: Dict[str, Any] = Field(default_factory=dict)

    # 증거 맥락
    include_evidence_refs: List[str] = Field(default_factory=list)
    focus_services: Optional[List[str]] = None

    # Adaptive execution metadata
    iteration: int = 0
    max_iterations: int = 3

    prior_response: Optional[AgentResponse] = None


# ---------------------------------------------------------------------------
# Convenience factory — create a minimal response from legacy agents
# ---------------------------------------------------------------------------

def make_legacy_response(
    agent_name: AgentName,
    request_id: str,
    predicted_service: str,
    confidence: float,
    reasoning: str = "",
) -> AgentResponse:
    """기존 Agent 코드를 AgentResponse 스키마로 래핑할 때 쓰는 헬퍼.

    Phase 2-3 진행 중 점진적 migration을 위해 제공한다. 기존 Agent가
    예전 포맷으로 응답하면 이 함수로 최소한의 AgentResponse를 만든다.
    evidence_collection은 비어있고, completeness_score는 0.0이다.

    주의: evidence-backed가 아니므로 Orchestrator 입장에서는 언제나
    'needs_more_evidence=True' 상태가 된다. 이는 의도된 동작이다
    — migration 중인 Agent는 재호출/보완 대상이 되어야 한다.
    """
    cand = Candidate(
        service=predicted_service,
        confidence=confidence,
        supporting_evidence=[],
        assumptions=["legacy_agent_no_explicit_evidence"],
        missing_evidence=["structured_evidence_units"],
        reasoning=reasoning,
    )
    return AgentResponse(
        agent_name=agent_name,
        request_id=request_id,
        candidates=[cand],
        evidence_collection=EvidenceCollection(),
        completeness_score=0.0,
        reasoning=reasoning,
    )


__all__ = [
    "AgentName",
    "AgentRequest",
    "AgentResponse",
    "Candidate",
    "ConsistencyChecks",
    "FailureMode",
    "make_legacy_response",
]
