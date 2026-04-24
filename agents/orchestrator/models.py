
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    start: str
    end: str


class IncidentRequest(BaseModel):
    incident_id: str
    service: str
    time_range: TimeRange
    symptom: str
    trace_id: Optional[str] = None
    attachments: Optional[Dict[str, Any]] = None


class Evidence(BaseModel):
    type: str
    source: str
    content: str
    timestamp: Optional[str] = None
    level: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentResult(BaseModel):
    agent: str
    summary: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RootCauseCandidate(BaseModel):
    """근본 원인 후보.
    
    기존 필드: rank, cause, confidence, evidence_refs (결정론적 버전 호환)
    LLM 버전 추가 필드: cause_service, reasoning, supporting_evidence
    """
    rank: int
    cause: str
    confidence: float
    evidence_refs: List[str] = Field(default_factory=list)
    # === LLM version 추가 필드 (optional) ===
    cause_service: Optional[str] = None
    reasoning: Optional[str] = None
    supporting_evidence: Optional[Dict[str, Any]] = None


class IncidentSummary(BaseModel):
    service: str
    symptom: str
    time_range: TimeRange


class FinalVerdict(BaseModel):
    cause: str
    confidence: float
    explanation: str
    # === LLM version: cause_service도 최종 판정에 기록 ===
    cause_service: Optional[str] = None


class ImpactAnalysis(BaseModel):
    affected_services: List[str]
    related_services: List[str]
    propagation_path: List[str]
    blast_radius: List[str]


class VerificationResult(BaseModel):
    verdict: str
    final_confidence: float
    notes: List[str]


class EvidenceSummary(BaseModel):
    log_evidence: List[str]
    topology_evidence: List[str]


class FinalRCAResult(BaseModel):
    incident_id: str
    incident_summary: IncidentSummary
    root_cause_candidates: List[RootCauseCandidate]
    final_verdict: FinalVerdict
    impact_analysis: ImpactAnalysis
    verification: VerificationResult
    evidence_summary: EvidenceSummary
    agent_results: List[AgentResult]
    # === LLM version 추가 필드 (optional) ===
    evidence_convergence: Optional[str] = None
    synthesis_reasoning: Optional[str] = None
    tcb_rca_reference: Optional[Dict[str, Any]] = None
    # === Graceful degradation tracking ===
    agent_errors: Optional[List[Dict[str, str]]] = None
    # === Phase 4d: A2A contract parsing ===
    # When A2A_PARSE_CONTRACTS env flag is on, Orchestrator collects each
    # agent's serialised AgentResponse (_agent_response field) into a dict
    # keyed by agent name. This makes the paper's contract-based architecture
    # directly observable in the final output — reviewers can point at this
    # field to see AgentResponses flowing end-to-end.
    #   Key     : agent_name ("log_agent", "topology_agent", "rca_agent", "verifier_agent")
    #   Value   : dict (serialised AgentResponse)
    agent_contracts: Optional[Dict[str, Dict[str, Any]]] = None
    # === Phase 4d-2: Adaptive execution tracking ===
    # Records any re-invocations triggered by low completeness_score.
    # Each entry describes an iteration's reason and outcome.
    adaptive_iterations: Optional[List[Dict[str, Any]]] = None
