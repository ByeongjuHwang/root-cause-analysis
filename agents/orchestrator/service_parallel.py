"""
B2 Baseline: Parallel Independent Multi-Agent.

본 프레임워크와 동일한 4개 에이전트를 사용하되, 순차 정보 전달 없이
모든 에이전트를 독립적으로 병렬 실행한 뒤 결과를 투표/집계한다.

핵심 차이 (Sequential vs Parallel):
┌─────────────────────────────────────────────────────┐
│ Ours (Sequential):                                  │
│ Log ──(anomaly_map)──> Topology ──(both)──> RCA ──> Verifier │
│ 각 단계가 이전 단계의 출력을 입력으로 받음            │
├─────────────────────────────────────────────────────┤
│ B2 (Parallel):                                      │
│ Log Agent       ─┐                                  │
│ Topology Agent  ─┼──> Aggregator ──> Result         │
│ RCA Agent       ─┘                                  │
│ 각 에이전트가 원본 인시던트 정보만 받아 독립 분석    │
└─────────────────────────────────────────────────────┘

이 비교군은 "순차적 정보 전달이 제공하는 가치"를 분리 측정한다.
"""

from __future__ import annotations

import asyncio
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


class ParallelOrchestratorService:
    """병렬 독립 멀티 에이전트 오케스트레이터.

    모든 에이전트를 동시에 호출하며, 에이전트 간 정보 공유가 없다.
    각 에이전트는 원본 인시던트 정보만 받아 독립적으로 분석한다.
    """

    def __init__(
        self,
        log_agent_url: str,
        topology_agent_url: str,
        rca_agent_url: str,
        verifier_agent_url: str,
    ):
        self.log_agent_url = log_agent_url
        self.topology_agent_url = topology_agent_url
        self.rca_agent_url = rca_agent_url
        self.verifier_agent_url = verifier_agent_url
        self.a2a_client = A2AClient()

    async def analyze_incident(self, incident: IncidentRequest) -> FinalRCAResult:
        # ============================================================
        # 핵심 차이: 3개 에이전트를 동시에(parallel) 호출
        # 순차 파이프라인과 달리, 각 에이전트가 이전 단계의 결과를
        # 받지 않고 원본 인시던트 정보만으로 독립 분석한다.
        # ============================================================

        log_task = self._call_log_agent(incident)
        topology_task = self._call_topology_agent_independent(incident)
        rca_task = self._call_rca_agent_independent(incident)

        # 동시 실행
        log_raw, topology_raw, rca_raw = await asyncio.gather(
            log_task, topology_task, rca_task,
            return_exceptions=True,
        )

        # 실패한 에이전트 처리
        log_result = self._safe_parse_agent(log_raw, "log_agent")
        topology_result = self._safe_parse_agent(topology_raw, "topology_agent")
        rca_result = self._safe_parse_rca(rca_raw)

        # ============================================================
        # Aggregator: 독립적 결과들을 병합
        # Sequential에서는 Verifier가 RCA 결과를 교차 검증하지만,
        # Parallel에서는 각 에이전트의 독립적 판단을 투표/병합한다.
        # ============================================================
        aggregated = self._aggregate_results(
            incident, log_result, topology_result, rca_result
        )

        return aggregated

    # ---- 에이전트 호출 (각각 독립, 정보 전달 없음) ----

    async def _call_log_agent(self, incident: IncidentRequest) -> Dict[str, Any]:
        """로그 에이전트: 원본 인시던트 정보만 전달 (이전과 동일)."""
        return await self.a2a_client.send_message(
            agent_base_url=self.log_agent_url,
            text=f"Analyze logs for incident {incident.incident_id} at {incident.service}.",
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

    async def _call_topology_agent_independent(self, incident: IncidentRequest) -> Dict[str, Any]:
        """토폴로지 에이전트: suspected_downstream 힌트 없이 독립 호출.

        Sequential에서는 로그 에이전트의 suspected_downstream을 전달하지만,
        Parallel에서는 이 정보가 없다. 토폴로지 에이전트는 구조 정보만으로
        분석해야 한다.
        """
        return await self.a2a_client.send_message(
            agent_base_url=self.topology_agent_url,
            text=f"Analyze topology for incident at {incident.service}.",
            metadata={
                "incident_id": incident.incident_id,
                "service": incident.service,
                "diagram_uri": (incident.attachments or {}).get("diagram_uri", "arch://system/latest"),
                "topology_file": (incident.attachments or {}).get("topology_file"),
                # 핵심: suspected_downstream이 None — 로그 에이전트의 힌트 없음
                "suspected_downstream": None,
                "desired_output": "application/json",
            },
        )

    async def _call_rca_agent_independent(self, incident: IncidentRequest) -> Dict[str, Any]:
        """RCA 에이전트: 로그 분석 결과와 토폴로지 분석 결과 없이 독립 호출.

        Sequential에서는 log_result(이상 징후 맵)과 topology_result(구조 정보)를
        받지만, Parallel에서는 이 둘이 비어있다. RCA 에이전트는 원본 인시던트
        정보만으로 근본 원인을 추론해야 한다.
        """
        return await self.a2a_client.send_message(
            agent_base_url=self.rca_agent_url,
            text="Synthesize root cause candidates independently.",
            metadata={
                "incident_id": incident.incident_id,
                "service": incident.service,
                # 핵심: log_result가 비어있음 — 로그 에이전트의 전처리된 증거 없음
                "log_result": {
                    "service": incident.service,
                    "summary": "",
                    "confidence": 0.0,
                    "evidence": [],
                },
                # 핵심: topology_result가 비어있음 — 토폴로지 에이전트의 분석 없음
                "topology_result": {
                    "service": incident.service,
                    "summary": "",
                    "confidence": 0.0,
                    "evidence": [],
                },
                "desired_output": "application/json",
            },
        )

    # ---- 안전한 파싱 ----

    def _safe_parse_agent(self, raw: Any, agent_name: str) -> AgentResult:
        """예외가 발생한 에이전트도 graceful하게 처리."""
        if isinstance(raw, Exception):
            return AgentResult(
                agent=agent_name,
                summary=f"Agent failed: {raw}",
                confidence=0.0,
                evidence=[],
                metadata={},
            )
        return self._parse_agent_result(raw, agent_name)

    def _safe_parse_rca(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, Exception):
            return {"summary": "", "confidence": 0.0, "root_cause_candidates": []}
        return self._parse_rca_result(raw)

    # ---- Aggregator: 독립 결과 병합 ----

    def _aggregate_results(
        self,
        incident: IncidentRequest,
        log_result: AgentResult,
        topology_result: AgentResult,
        rca_result: Dict[str, Any],
    ) -> FinalRCAResult:
        """3개 독립 에이전트의 결과를 병합.

        Sequential과의 핵심 차이:
        - Sequential: 각 단계가 이전 단계를 기반으로 정제된 결과 생성
        - Parallel: 각 에이전트의 독립 결과를 사후적으로(post-hoc) 병합

        병합 전략:
        1. RCA 에이전트의 후보가 있으면 그것을 기본으로 사용
        2. 로그 에이전트의 hypothesis를 보조 후보로 추가
        3. 토폴로지 에이전트의 propagation 정보를 impact에 반영
        4. Confidence는 각 에이전트의 개별 confidence를 평균
        """

        # RCA 후보 수집
        rca_candidates = self._ensure_list(rca_result.get("root_cause_candidates", []))

        # 로그 에이전트의 hypothesis를 추가 후보로 변환
        log_hypothesis = log_result.metadata.get("hypothesis", "")
        log_suspected = log_result.metadata.get("suspected_downstream")

        # 로그 기반 후보 생성 (RCA 후보가 비었을 때 대안)
        if log_suspected and not rca_candidates:
            rca_candidates.append({
                "rank": 1,
                "cause": f"Log-based hypothesis: {log_hypothesis} (suspected: {log_suspected})",
                "confidence": log_result.confidence * 0.6,  # 로그만으로는 낮은 확신
                "evidence_refs": ["log-agent:hypothesis"],
            })

        # 후보가 여전히 비어있으면 증상 서비스 자체를 약한 후보로
        if not rca_candidates:
            rca_candidates.append({
                "rank": 1,
                "cause": f"No independent RCA result. Symptom observed at {incident.service}.",
                "confidence": max(log_result.confidence * 0.3, 0.1),
                "evidence_refs": ["aggregator:fallback"],
            })

        # 후보 정규화
        final_candidates = self._normalize_candidates(rca_candidates)

        # Confidence 병합: 각 에이전트의 독립적 confidence를 가중 평균
        agent_confidences = [
            log_result.confidence,
            topology_result.confidence,
            float(rca_result.get("confidence", 0.0)),
        ]
        # 0이 아닌 값들만 평균
        nonzero = [c for c in agent_confidences if c > 0]
        avg_confidence = sum(nonzero) / len(nonzero) if nonzero else 0.0

        # 최종 confidence: 후보의 confidence와 에이전트 평균의 가중 결합
        top_raw_conf = final_candidates[0].confidence if final_candidates else 0.0
        merged_confidence = round(0.6 * top_raw_conf + 0.4 * avg_confidence, 3)
        if final_candidates:
            final_candidates[0].confidence = merged_confidence

        top_cause = final_candidates[0].cause if final_candidates else "No cause identified."
        top_confidence = final_candidates[0].confidence if final_candidates else 0.0

        # Verdict 결정
        if top_confidence >= 0.80:
            verdict = "accepted"
        elif top_confidence >= 0.50:
            verdict = "revised"
        else:
            verdict = "weak-evidence"

        # 토폴로지 정보 (independent — suspected_downstream 없이 분석됨)
        propagation_path = self._string_list(
            rca_result.get("propagation_path",
                          topology_result.metadata.get("propagation_path", []))
        )
        blast_radius = self._string_list(
            rca_result.get("blast_radius",
                          topology_result.metadata.get("blast_radius", []))
        )

        return FinalRCAResult(
            incident_id=incident.incident_id,
            incident_summary=IncidentSummary(
                service=incident.service,
                symptom=incident.symptom,
                time_range=incident.time_range,
            ),
            root_cause_candidates=final_candidates,
            final_verdict=FinalVerdict(
                cause=top_cause,
                confidence=top_confidence,
                explanation=f"B2 (Parallel): Agents analyzed independently, results merged post-hoc. "
                           f"Log confidence: {log_result.confidence:.2f}, "
                           f"Topology confidence: {topology_result.confidence:.2f}, "
                           f"RCA confidence: {rca_result.get('confidence', 0.0):.2f}. "
                           f"Merged confidence: {top_confidence:.2f}.",
            ),
            impact_analysis=ImpactAnalysis(
                affected_services=self._string_list(
                    rca_result.get("affected_services", [incident.service])
                ),
                related_services=self._string_list(
                    rca_result.get("related_services",
                                  topology_result.metadata.get("related_services", []))
                ),
                propagation_path=propagation_path,
                blast_radius=blast_radius,
            ),
            verification=VerificationResult(
                verdict=verdict,
                final_confidence=top_confidence,
                notes=[
                    "B2 baseline: parallel independent analysis, no cross-validation.",
                    f"Agent confidences — log: {log_result.confidence:.2f}, "
                    f"topology: {topology_result.confidence:.2f}, "
                    f"rca: {rca_result.get('confidence', 0.0):.2f}.",
                ],
            ),
            evidence_summary=EvidenceSummary(
                log_evidence=[e.content for e in log_result.evidence[:5]],
                topology_evidence=[e.content for e in topology_result.evidence[:5]],
            ),
            agent_results=[log_result, topology_result],
        )

    # ---- 원본에서 가져온 헬퍼 메서드들 ----

    def _parse_agent_result(self, raw: Dict[str, Any], agent_name: str) -> AgentResult:
        task = raw.get("result", {}).get("task", {})
        artifacts = task.get("artifacts", [])
        data: Dict[str, Any] = {}
        if artifacts and isinstance(artifacts, list):
            first = artifacts[0]
            if isinstance(first, dict):
                parts = first.get("parts", [])
                if parts and isinstance(parts, list):
                    fp = parts[0]
                    if isinstance(fp, dict):
                        data = fp.get("data", {}) or {}

        evidence_items = []
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
                     "topology_graph", "service_metadata", "topology_file"):
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
            first = artifacts[0]
            if isinstance(first, dict):
                parts = first.get("parts", [])
                if parts and isinstance(parts, list):
                    fp = parts[0]
                    if isinstance(fp, dict):
                        return fp.get("data", {}) or {}
        return {"summary": "", "confidence": 0.0, "root_cause_candidates": []}

    def _normalize_candidates(self, raw: List[Dict[str, Any]]) -> List[RootCauseCandidate]:
        result = []
        for idx, c in enumerate(raw, 1):
            if not isinstance(c, dict):
                continue
            result.append(RootCauseCandidate(
                rank=int(c.get("rank", idx)),
                cause=str(c.get("cause", "Unknown")),
                confidence=float(c.get("confidence", 0.5)),
                evidence_refs=self._string_list(c.get("evidence_refs", [])),
            ))
        result.sort(key=lambda x: x.confidence, reverse=True)
        for idx, c in enumerate(result, 1):
            c.rank = idx
        return result

    def _ensure_list(self, v: Any) -> List[Any]:
        return v if isinstance(v, list) else []

    def _string_list(self, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        return [str(x) for x in v if x is not None]
