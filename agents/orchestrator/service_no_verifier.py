"""
B3 Baseline: No-Verifier Pipeline.

본 프레임워크와 동일한 순차 파이프라인이지만, 교차 검증 에이전트를
제거하고 RCA 에이전트의 결과를 그대로 최종 결과로 사용한다.

이 비교군은 "교차 검증 에이전트가 최종 판단에 미치는 영향"을
분리 측정하기 위한 ablation 설계이다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .a2a_client import A2AClient
from .models import (
    AgentResult,
    Evidence,
    EvidenceSummary,
    FinalRCAResult,
    FinalVerdict,
    ImpactAnalysis,
    IncidentRequest,
    IncidentSummary,
    RootCauseCandidate,
    VerificationResult,
)


class NoVerifierOrchestratorService:
    """교차 검증 에이전트를 제거한 순차 파이프라인 오케스트레이터.

    원본 OrchestratorService와 동일하되, 4단계(Verifier) 호출을 완전히
    제거하고 RCA 에이전트의 결과를 그대로 최종 결과로 사용한다.
    """

    def __init__(
        self,
        log_agent_url: str,
        topology_agent_url: str,
        rca_agent_url: str,
    ):
        self.log_agent_url = log_agent_url
        self.topology_agent_url = topology_agent_url
        self.rca_agent_url = rca_agent_url
        self.a2a_client = A2AClient()

    async def analyze_incident(self, incident: IncidentRequest) -> FinalRCAResult:
        # 1) Log agent — 원본과 동일
        log_raw = await self.a2a_client.send_message(
            agent_base_url=self.log_agent_url,
            text=self._build_log_agent_prompt(incident),
            metadata={
                "incident_id": incident.incident_id,
                "service": incident.service,
                "start": incident.time_range.start,
                "end": incident.time_range.end,
                "trace_id": incident.trace_id,
                "symptom": incident.symptom,
                "desired_output": "application/json",
                "log_file": (incident.attachments or {}).get("log_file"),
            },
        )
        log_result = self._parse_agent_result(log_raw, "log_agent")

        # 2) Topology agent — 원본과 동일
        suspected_downstream = self._guess_downstream_from_log_result(
            root_service=incident.service,
            log_result=log_result,
        )

        topology_raw = await self.a2a_client.send_message(
            agent_base_url=self.topology_agent_url,
            text=self._build_topology_agent_prompt(incident, suspected_downstream),
            metadata={
                "incident_id": incident.incident_id,
                "service": incident.service,
                "diagram_uri": (incident.attachments or {}).get("diagram_uri", "arch://system/latest"),
                "topology_file": (incident.attachments or {}).get("topology_file"),
                "suspected_downstream": suspected_downstream,
                "desired_output": "application/json",
            },
        )
        topology_result = self._parse_agent_result(topology_raw, "topology_agent")

        # 3) RCA agent — 원본과 동일
        rca_raw = await self.a2a_client.send_message(
            agent_base_url=self.rca_agent_url,
            text="Synthesize root cause candidates from log and topology results.",
            metadata={
                "incident_id": incident.incident_id,
                "service": incident.service,
                "log_result": {
                    "service": incident.service,
                    "summary": log_result.summary,
                    "confidence": log_result.confidence,
                    "evidence": [e.model_dump() for e in log_result.evidence],
                    **log_result.metadata,
                },
                "topology_result": {
                    "service": incident.service,
                    "summary": topology_result.summary,
                    "confidence": topology_result.confidence,
                    "evidence": [e.model_dump() for e in topology_result.evidence],
                    **topology_result.metadata,
                },
                "desired_output": "application/json",
            },
        )
        rca_result = self._parse_rca_result(rca_raw)

        # ============================================================
        # 4) Verifier agent — 완전히 제거됨 (B3 핵심 차이점)
        # ============================================================
        # RCA 결과를 검증 없이 그대로 사용

        final_candidates = self._normalize_candidates(
            self._ensure_list(rca_result.get("root_cause_candidates", []))
        )

        top_cause = final_candidates[0].cause if final_candidates else "No root cause identified."
        top_confidence = final_candidates[0].confidence if final_candidates else 0.0
        top_cause_service = final_candidates[0].cause_service if final_candidates else None

        evidence_summary_dict = self._build_evidence_summary(log_result, topology_result)

        # Verification 결과는 "skipped"로 표시
        verification_result = VerificationResult(
            verdict="skipped",
            final_confidence=top_confidence,  # 검증 없이 RCA confidence 그대로
            notes=["B3 baseline: verification step was skipped."],
        )

        final_verdict = FinalVerdict(
            cause=top_cause,
            confidence=top_confidence,
            explanation=f"B3 (No-Verifier): RCA result used directly without cross-validation. "
                       f"Top cause: '{top_cause}' with unverified confidence {top_confidence:.2f}.",
            cause_service=top_cause_service,
        )

        impact_analysis = ImpactAnalysis(
            affected_services=self._string_list(rca_result.get("affected_services", [incident.service])),
            related_services=self._string_list(rca_result.get("related_services", [])),
            propagation_path=self._string_list(
                rca_result.get("propagation_path", topology_result.metadata.get("propagation_path", []))
            ),
            blast_radius=self._string_list(
                rca_result.get("blast_radius", topology_result.metadata.get("blast_radius", []))
            ),
        )

        evidence_summary = EvidenceSummary(
            log_evidence=evidence_summary_dict["log_evidence"],
            topology_evidence=evidence_summary_dict["topology_evidence"],
        )

        return FinalRCAResult(
            incident_id=incident.incident_id,
            incident_summary=IncidentSummary(
                service=incident.service,
                symptom=incident.symptom,
                time_range=incident.time_range,
            ),
            root_cause_candidates=final_candidates,
            final_verdict=final_verdict,
            impact_analysis=impact_analysis,
            verification=verification_result,
            evidence_summary=evidence_summary,
            agent_results=[log_result, topology_result],
            # LLM 메타데이터
            evidence_convergence=rca_result.get("evidence_convergence"),
            synthesis_reasoning=rca_result.get("llm_synthesis_reasoning"),
            tcb_rca_reference=rca_result.get("tcb_rca_reference"),
        )

    # ---- 아래는 원본 OrchestratorService에서 그대로 가져온 헬퍼 메서드들 ----

    def _build_log_agent_prompt(self, incident: IncidentRequest) -> str:
        return f"""
Analyze logs for the following incident.

Incident ID: {incident.incident_id}
Service: {incident.service}
Time range: {incident.time_range.start} ~ {incident.time_range.end}
Symptom: {incident.symptom}
Trace ID: {incident.trace_id}

Return JSON with:
1. summary
2. confidence
3. evidence
4. hypothesis
5. suspected downstream service if identifiable
""".strip()

    def _build_topology_agent_prompt(self, incident: IncidentRequest, suspected_downstream: Optional[str]) -> str:
        return f"""
Analyze topology for the following incident.

Incident ID: {incident.incident_id}
Target service: {incident.service}
Suspected downstream service: {suspected_downstream}
Diagram URI: {(incident.attachments or {}).get("diagram_uri", "arch://system/latest")}

Return JSON with:
1. summary
2. confidence
3. dependency_info
4. related_services
5. propagation_path
6. blast_radius
7. evidence
""".strip()

    def _parse_agent_result(self, raw: Dict[str, Any], agent_name: str) -> AgentResult:
        task = raw.get("result", {}).get("task", {})
        artifacts = task.get("artifacts", [])
        data: Dict[str, Any] = {}
        if artifacts and isinstance(artifacts, list):
            first_artifact = artifacts[0]
            if isinstance(first_artifact, dict):
                parts = first_artifact.get("parts", [])
                if parts and isinstance(parts, list):
                    first_part = parts[0]
                    if isinstance(first_part, dict):
                        data = first_part.get("data", {}) or {}

        evidence_items: List[Evidence] = []
        for item in self._ensure_list(data.get("evidence", [])):
            if not isinstance(item, dict):
                continue
            evidence_items.append(Evidence(
                type=str(item.get("type", "unknown")),
                source=str(item.get("source", "unknown")),
                content=str(item.get("content", "")),
                timestamp=item.get("timestamp"),
                level=item.get("level"),
                trace_id=item.get("trace_id"),
                metadata=item.get("metadata", {}) or {},
            ))

        metadata: Dict[str, Any] = {}
        for key in ("propagation_path", "blast_radius", "dependency_info",
                     "related_services", "hypothesis", "suspected_downstream",
                     # Topology Agent full graph
                     "topology_graph", "service_metadata", "topology_file",
                     # LLM 필드
                     "anomalous_services", "llm_reasoning",
                     "path_assessment", "alternative_paths",
                     "critical_services_in_blast", "topology_supports_hypothesis", "referenced_upstreams"):
            if key in data:
                metadata[key] = data[key]

        return AgentResult(
            agent=agent_name,
            summary=str(data.get("summary", "")),
            confidence=float(data.get("confidence", 0.5)),
            evidence=evidence_items,
            metadata=metadata,
        )

    def _parse_rca_result(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        task = raw.get("result", {}).get("task", {})
        artifacts = task.get("artifacts", [])
        if artifacts and isinstance(artifacts, list):
            first_artifact = artifacts[0]
            if isinstance(first_artifact, dict):
                parts = first_artifact.get("parts", [])
                if parts and isinstance(parts, list):
                    first_part = parts[0]
                    if isinstance(first_part, dict):
                        return first_part.get("data", {}) or {}
        return {"summary": "", "confidence": 0.0, "root_cause_candidates": []}

    def _normalize_candidates(self, raw_candidates: List[Dict[str, Any]]) -> List[RootCauseCandidate]:
        """후보 정규화. LLM 필드 (cause_service, reasoning, supporting_evidence) 보존."""
        normalized = []
        for idx, c in enumerate(raw_candidates, 1):
            if not isinstance(c, dict):
                continue
            normalized.append(RootCauseCandidate(
                rank=int(c.get("rank", idx)),
                cause=str(c.get("cause", "Unknown")),
                confidence=float(c.get("confidence", 0.5)),
                evidence_refs=self._string_list(c.get("evidence_refs", [])),
                # === LLM 필드 추가 ===
                cause_service=c.get("cause_service"),
                reasoning=c.get("reasoning"),
                supporting_evidence=c.get("supporting_evidence"),
            ))
        normalized.sort(key=lambda c: c.confidence, reverse=True)
        for idx, c in enumerate(normalized, 1):
            c.rank = idx
        return normalized

    def _guess_downstream_from_log_result(self, root_service: str, log_result: AgentResult) -> Optional[str]:
        suspected = log_result.metadata.get("suspected_downstream")
        if suspected and suspected != root_service:
            return str(suspected)
        for evidence in log_result.evidence:
            meta = evidence.metadata or {}
            upstream = meta.get("upstream")
            if upstream and upstream != root_service:
                return str(upstream)
        return None

    def _build_evidence_summary(self, log_result: AgentResult, topology_result: AgentResult) -> Dict[str, List[str]]:
        log_evidence = [e.content for e in log_result.evidence[:5]]
        topology_evidence = [e.content for e in topology_result.evidence[:5]]
        propagation_path = self._string_list(topology_result.metadata.get("propagation_path", []))
        if propagation_path:
            topology_evidence.append(f"propagation path: {' -> '.join(propagation_path)}")
        return {"log_evidence": log_evidence, "topology_evidence": topology_evidence}

    def _ensure_list(self, value: Any) -> List[Any]:
        return value if isinstance(value, list) else []

    def _string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(v) for v in value if v is not None]
