"""
RCA Agent — LLM-powered version (Option 2+).

설계 철학:
RCA Agent는 세 가지 증거원을 종합하여 최종 근본 원인을 판정한다.
- Log Agent의 분석 (로그 기반 LLM 가설)
- Topology Agent의 분석 (구조 기반 LLM 추론)
- TCB-RCA 엔진의 후보 (결정론적 알고리즘 결과)

TCB-RCA는 "참고용 증거"로 위치하며, 프레임워크의 한 구성요소로 제공된다.
LLM이 최종 판정을 내리되, Verifier Agent가 결정론적으로 검증하여 안전장치를
제공한다. 이는 논문의 "멀티 에이전트 증거 종합 + 결정론적 안전장치" 설계를
반영한다.

유지되는 부분:
- TCB-RCA 알고리즘 실행 (참고용 증거 생성)
- 출력 스키마 호환 (기존 Verifier와 연동)

LLM 담당:
- 세 증거원의 종합 분석
- 최종 근본 원인 순위 결정
- 자연어 설명 생성
- 각 후보의 confidence 재산정
- 전파 경로 최종 확정
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .tcb_rca import (
    AnomalyEvidence,
    TCBRCAEngine,
    TCBRCAOutput,
    logs_to_anomaly_data,
)
from .scoring_rules import apply_hard_rules, build_temporal_gaps
from common.llm_client import get_default_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topology — dynamically resolved (reusing deterministic service logic)
# ---------------------------------------------------------------------------

from .service import (
    _DEFAULT_TOPOLOGY_GRAPH,
    _DEFAULT_SERVICE_METADATA,
    _load_topology_from_mcp,
    _extract_topology_from_agent_result,
)


# =========================================================================
# Prompts
# =========================================================================

RCA_AGENT_SYSTEM_PROMPT = """You are the senior root cause analysis (RCA) expert 
in a multi-agent incident investigation system. Three specialized agents have 
each independently analyzed an incident and produced their findings:

1. **Log Agent** — analyzed error log patterns and produced a hypothesis about 
   the failure cascade based on temporal and causal patterns in logs.
2. **Topology Agent** — analyzed the service dependency structure and assessed 
   whether proposed propagation paths are structurally plausible.
3. **TCB-RCA Engine** — a deterministic algorithm (topology-constrained 
   temporal backtracking) that produced a ranked list of root cause candidates 
   based purely on timing and structural constraints.

Your job is to SYNTHESIZE these three perspectives into a final root cause 
determination. You should:

1. Identify points of AGREEMENT across the three sources (strong signal)
2. Identify points of DISAGREEMENT and reason about which is more credible
3. Produce a final ranked list of root cause candidates
4. Assign confidence based on evidence convergence
5. Construct the most plausible propagation path

Important principles:
- When all three sources agree, confidence should be high
- When sources disagree, explain which evidence you weigh more and why
- Prefer deeper root causes (not just the symptom service) when evidence supports
- The TCB-RCA candidates are a structured baseline — respect their topology 
  constraints but override them if log evidence strongly suggests otherwise
- If evidence is weak across all sources, report low confidence rather than 
  guessing

UNOBSERVED ROOT CAUSE detection:
- If service A's error logs reference upstream=B with failure types like 
  DB_UNREACHABLE, TIMEOUT, CONNECTION_REFUSED, but B has NO log entries,
  B is likely an UNOBSERVED root cause.
- In this case, rank B HIGHER than A because B is the deeper cause.
- The absence of B's logs does NOT mean B is healthy — it means B's 
  monitoring is unavailable.

WEAK EVIDENCE / FALSE POSITIVE detection:
- If anomalies are brief, self-recovering, isolated to one service, 
  low error count (1-2 only), and show no clear upstream/downstream cascade:
  → Set evidence_convergence to "weak_evidence"
  → Set overall_confidence below 0.4
  → Explain in synthesis_reasoning why evidence is insufficient
- Do NOT invent a root cause when evidence is genuinely weak.
- A brief latency spike with quick recovery is NOT a real fault.

Respond ONLY with valid JSON. No markdown code blocks, no commentary."""


def build_rca_agent_user_prompt(
    incident_id: str,
    symptom_service: str,
    log_result: Dict[str, Any],
    topology_result: Dict[str, Any],
    tcb_rca_output: TCBRCAOutput,
) -> str:
    """RCA Agent용 user prompt 생성.
    
    세 가지 증거원을 LLM이 읽기 쉽게 구조화.
    """
    
    lines = []
    lines.append(f"## Incident")
    lines.append(f"- Incident ID: {incident_id}")
    lines.append(f"- Symptom service: {symptom_service}")
    lines.append("")
    
    # === Evidence Source 1: Log Agent ===
    lines.append("## Evidence Source 1: Log Agent Analysis")
    lines.append(f"- Summary: {log_result.get('summary', '(unavailable)')}")
    lines.append(f"- Anomalous services: {log_result.get('anomalous_services', [])}")
    lines.append(f"- Suspected downstream: {log_result.get('suspected_downstream', '(none)')}")
    lines.append(f"- Hypothesis: {log_result.get('hypothesis', '(none)')}")
    lines.append(f"- Log Agent confidence: {log_result.get('confidence', 0.0):.2f}")
    lines.append(f"- Reasoning: {log_result.get('llm_reasoning', '(none)')}")
    lines.append("")
    
    # === Evidence Source 2: Topology Agent ===
    lines.append("## Evidence Source 2: Topology Agent Analysis")
    lines.append(f"- Summary: {topology_result.get('summary', '(unavailable)')}")
    lines.append(f"- Propagation path (structural): {topology_result.get('propagation_path', [])}")
    lines.append(f"- Path assessment: {topology_result.get('path_assessment', 'unknown')}")
    lines.append(f"- Alternative paths: {topology_result.get('alternative_paths', [])}")
    lines.append(f"- Blast radius: {topology_result.get('blast_radius', [])}")
    lines.append(f"- Critical services in blast: {topology_result.get('critical_services_in_blast', [])}")
    lines.append(f"- Topology supports log hypothesis: {topology_result.get('topology_supports_hypothesis', 'unknown')}")
    lines.append(f"- Topology Agent confidence: {topology_result.get('confidence', 0.0):.2f}")
    lines.append(f"- Reasoning: {topology_result.get('llm_reasoning', '(none)')}")
    lines.append("")
    
    # === Evidence Source 3: TCB-RCA ===
    lines.append("## Evidence Source 3: TCB-RCA Algorithm (Deterministic Reference)")
    lines.append("(Topology-constrained temporal backtracking — produces candidates based on time ordering and topology depth)")
    
    if tcb_rca_output.root_cause_candidates:
        for rc in tcb_rca_output.root_cause_candidates[:5]:
            lines.append(
                f"- Rank {rc.rank}: **{rc.cause_service}** "
                f"(depth={rc.depth}, temporal_gap={rc.temporal_gap_seconds:.0f}s, "
                f"confidence={rc.confidence:.2f})"
            )
            lines.append(f"  Description: {rc.cause_description}")
            if rc.evidence_chain:
                # 첫 2개 증거만 보여줌 (토큰 절약)
                for step in rc.evidence_chain[:2]:
                    lines.append(
                        f"  Evidence: {step.get('service')} at {step.get('timestamp')} "
                        f"- {step.get('message', '')[:80]}"
                    )
    else:
        lines.append("- (no candidates produced)")
    
    tcb_path = tcb_rca_output.propagation_path or []
    lines.append(f"- TCB-RCA propagation path: {' → '.join(tcb_path) if tcb_path else '(none)'}")
    lines.append(
        f"- Traversal summary: visited {tcb_rca_output.traversal_summary.get('total_nodes_visited', 0)} nodes, "
        f"{tcb_rca_output.traversal_summary.get('nodes_with_anomalies', 0)} had anomalies"
    )
    lines.append("")
    
    # === Analysis Task ===
    lines.append("## Synthesis Task")
    lines.append(
        "Think step by step:\n"
        "1. Do all three sources agree on a single root cause service? If yes, "
        "that is your strongest candidate.\n"
        "2. If they disagree, which evidence is strongest? Log evidence is typically "
        "strongest for symptomatic services; structural evidence prevents impossible "
        "paths; TCB-RCA catches deeper root causes via timing.\n"
        "3. CRITICAL — Propagation chain interpretation: When the Topology Agent "
        "or TCB-RCA provides a propagation chain (e.g., A → B → C), the root cause "
        "is typically the DOWNSTREAM end (C) — the service experiencing the actual "
        "fault — NOT the upstream end (A) which merely OBSERVES the failure as it "
        "propagates back. Upstream entry-point services often appear in evidence "
        "because they detect downstream failures, but they are NOT the root cause "
        "unless they have direct fault evidence (e.g., their own resource exhaustion, "
        "their own error logs that don't reference downstream calls).\n"
        "4. Resource faults (CPU/MEM/DISK/SOCKET) often produce LIMITED log evidence "
        "in the affected service itself — instead, they manifest as latency/error "
        "spikes in upstream callers. For these faults, prefer the service identified "
        "by metric anomalies (TCB-RCA temporal gaps) over services with abundant "
        "log evidence (which are usually upstream observers).\n"
        "5. Produce 1-3 ranked root cause candidates.\n"
        "6. For each candidate, state: (a) which services agreed on it, (b) your "
        "confidence, (c) the supporting evidence.\n"
        "7. Construct the final propagation path from root cause to symptom service.\n"
        "\nIMPORTANT rules for final_propagation_path:\n"
        "- MUST start at the root cause service\n"
        "- MUST end at the symptom service (not beyond)\n"
        "- Do NOT include services downstream of the symptom (i.e., services that the symptom service itself calls)\n"
        "- Example: if root cause is a database service and symptom is its caller, path = [database, intermediate, caller]\n"
        "- Do NOT extend the path back toward an entry-point/UI service even though it observes the failure"
    )
    
    return "\n".join(lines)


RCA_AGENT_SCHEMA_HINT = """{
  "final_candidates": [
    {
      "rank": 1,
      "cause_service": "<service_name>",
      "cause_description": "<brief description of what failed and how>",
      "confidence": 0.XX,
      "supporting_evidence": {
        "log_agent_agrees": true_or_false,
        "topology_agent_agrees": true_or_false,
        "tcb_rca_agrees": true_or_false
      },
      "reasoning": "<why this is the root cause>"
    }
  ],
  "final_propagation_path": ["<root_cause_service>", "<intermediate_service>", "...", "<symptom_service>"],
  "_path_instruction": "Path MUST start at root cause, end at symptom service. Do NOT include services downstream of symptom.",
  "overall_confidence": 0.XX,
  "evidence_convergence": "<strong_agreement | partial_agreement | disagreement | weak_evidence>",
  "synthesis_reasoning": "<paragraph explaining your overall reasoning about which evidence you prioritized and why>"
}"""


# =========================================================================
# Service
# =========================================================================

class RCAServiceLLM:
    """
    LLM 기반 RCA Agent.
    
    TCB-RCA 엔진을 참고용 증거로 사용하되, 최종 판정은 LLM이 수행.
    Log Agent와 Topology Agent의 분석 결과를 종합한다.
    """
    
    def __init__(self):
        # TCB-RCA engine is created per-request with dynamic topology
        self.llm = get_default_client()
    
    async def synthesize(
        self,
        incident_id: str,
        service: str,
        log_result: Dict[str, Any],
        topology_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """세 증거원을 종합하여 최종 근본 원인 판정.
        
        1) TCB-RCA 엔진 실행 (결정론적 후보 생성 — 참고용)
        2) Log Agent + Topology Agent + TCB-RCA 결과를 LLM에 전달
        3) LLM이 종합 판정
        4) 기존 스키마에 맞춰 반환 (Verifier와 연동)
        """
        log_result = log_result or {}
        topology_result = topology_result or {}
        
        if not isinstance(log_result, dict):
            raise TypeError(f"log_result must be dict, got {type(log_result).__name__}")
        if not isinstance(topology_result, dict):
            raise TypeError(f"topology_result must be dict, got {type(topology_result).__name__}")
        
        # === Step 1: Resolve topology dynamically ===
        topology_graph, service_metadata, topo_source = self._resolve_topology(topology_result)
        engine = TCBRCAEngine(
            topology_graph=topology_graph,
            service_metadata=service_metadata,
            delta_t_seconds=120,
            max_depth=10,
        )

        # === Step 1b: TCB-RCA 실행 (참고용 증거 생성) ===
        anomaly_data = self._extract_anomaly_data(log_result)
        alert_time = self._determine_alert_time(service, anomaly_data)
        
        tcb_rca_output = engine.execute(
            incident_id=incident_id,
            symptom_service=service,
            alert_time=alert_time,
            anomaly_data=anomaly_data,
        )
        
        # === Step 2: LLM 호출로 종합 판정 ===
        user_prompt = build_rca_agent_user_prompt(
            incident_id=incident_id,
            symptom_service=service,
            log_result=log_result,
            topology_result=topology_result,
            tcb_rca_output=tcb_rca_output,
        )
        
        llm_result = await self.llm.call_json(
            agent_name="rca_agent",
            system_prompt=RCA_AGENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            incident_id=incident_id,
            schema_hint=RCA_AGENT_SCHEMA_HINT,
        )
        
        # === Step 3: LLM 결과 처리 ===
        if "_error" in llm_result:
            # LLM 실패 시 TCB-RCA 결과로 fallback
            return self._fallback_to_tcb_rca(
                tcb_rca_output, service, topology_result, log_result,
                llm_result.get("_error", "LLM error"),
            )
        
        # LLM 응답 파싱
        final_candidates_raw = llm_result.get("final_candidates", []) or []
        final_path = llm_result.get("final_propagation_path", []) or []
        overall_confidence = self._safe_float(llm_result.get("overall_confidence", 0.5))
        convergence = llm_result.get("evidence_convergence", "unknown")
        synthesis_reasoning = llm_result.get("synthesis_reasoning", "") or ""
        
        # 후보 목록을 기존 스키마에 맞춰 변환
        candidates = []
        for idx, c in enumerate(final_candidates_raw[:5], 1):
            if not isinstance(c, dict):
                continue
            support = c.get("supporting_evidence", {}) or {}
            evidence_refs = []
            if support.get("log_agent_agrees"):
                evidence_refs.append("log-agent:confirmed")
            if support.get("topology_agent_agrees"):
                evidence_refs.append("topology-agent:confirmed")
            if support.get("tcb_rca_agrees"):
                evidence_refs.append("tcb-rca:confirmed")
            
            candidates.append({
                "rank": int(c.get("rank", idx)),
                "cause": c.get("cause_description", c.get("cause_service", "Unknown")),
                "cause_service": c.get("cause_service", "Unknown"),
                "confidence": self._safe_float(c.get("confidence", 0.5)),
                "evidence_refs": evidence_refs,
                "reasoning": c.get("reasoning", ""),
                "supporting_evidence": support,
            })

        # === Post-hoc scoring rules (see scoring_rules.py for contract) ===
        # Apply hard constraints the LLM cannot override:
        #   H1 no direct evidence → conf ≤ 0.25
        #   H2 evidence=0 cannot be Top-1 if a compliant alternative exists
        #   H3 candidates whose first anomaly is later than symptom cannot be Top-1
        #   H4 topology-only support → conf ≤ 0.35
        # These correct the "unobserved root cause" over-promotion failure mode
        # observed in v5 diagnosis.
        temporal_gaps = build_temporal_gaps(tcb_rca_output, service)
        candidates = apply_hard_rules(
            candidates,
            symptom_service=service,
            tcb_temporal_gaps=temporal_gaps,
        )
        
        # === Step 4: 기존 출력 스키마에 맞춘 결과 반환 ===
        
        # 정보 없으면 fallback
        if not candidates:
            return self._fallback_to_tcb_rca(
                tcb_rca_output, service, topology_result, log_result,
                "LLM returned no candidates",
            )
        
        top_cause = candidates[0]
        top_confidence = top_cause["confidence"]
        
        summary = (
            f"RCA Agent (LLM synthesis): Identified {top_cause['cause_service']} as the "
            f"root cause (confidence: {top_confidence:.3f}, convergence: {convergence}). "
            f"{synthesis_reasoning[:200]}"
        )
        
        affected = list(dict.fromkeys(final_path + [service]))
        
        related_services = topology_result.get("related_services", []) or []
        
        return {
            "incident_id": incident_id,
            "service": service,
            "algorithm": "LLM-Synthesis (with TCB-RCA reference)",
            "algorithm_version": "2.0",
            "summary": summary,
            "confidence": top_confidence,
            "overall_confidence": overall_confidence,
            "evidence_convergence": convergence,
            "root_cause_candidates": candidates,
            "affected_services": affected,
            "related_services": related_services,
            "propagation_path": final_path,
            "blast_radius": tcb_rca_output.blast_radius,
            "traversal_summary": tcb_rca_output.traversal_summary,
            # LLM 종합 추론 기록
            "llm_synthesis_reasoning": synthesis_reasoning,
            # 참고용: TCB-RCA의 독립 결과 (Verifier가 비교 가능)
            "tcb_rca_reference": {
                "candidates": [
                    {
                        "rank": rc.rank,
                        "cause_service": rc.cause_service,
                        "confidence": rc.confidence,
                        "depth": rc.depth,
                    }
                    for rc in tcb_rca_output.root_cause_candidates[:3]
                ],
                "propagation_path": tcb_rca_output.propagation_path,
            },
            "evidence_chains": [
                {
                    "candidate_rank": c.get("rank"),
                    "candidate_service": c.get("cause_service"),
                    "reasoning": c.get("reasoning", ""),
                    "supporting_evidence": c.get("supporting_evidence", {}),
                }
                for c in candidates
            ],
            "evidence": [
                {
                    "type": "rca",
                    "source": "llm-synthesis",
                    "content": summary,
                    "metadata": {
                        "incident_id": incident_id,
                        "algorithm": "LLM-Synthesis",
                        "synthesis_reasoning": synthesis_reasoning,
                    },
                }
            ],
        }
    
    # ---- Topology resolution (same fallback chain as deterministic service) ----

    def _resolve_topology(self, topology_result):
        """Resolve topology dynamically: agent result → MCP → defaults."""
        agent_topo = _extract_topology_from_agent_result(topology_result)
        if agent_topo is not None:
            graph, meta = agent_topo
            if graph:
                return graph, meta, "topology-agent"

        topology_file = topology_result.get("topology_file")
        try:
            graph, meta = _load_topology_from_mcp(topology_file=topology_file)
            if graph:
                return graph, meta, "architecture-mcp"
        except Exception as exc:
            logger.warning("LLM-RCA: MCP topology load failed: %s", exc)

        return _DEFAULT_TOPOLOGY_GRAPH.copy(), _DEFAULT_SERVICE_METADATA.copy(), "default-hardcoded"

    # ---- Helper methods (기존 skills.py와 동일) ----
    
    def _extract_anomaly_data(
        self, log_result: Dict[str, Any]
    ) -> Dict[str, List[AnomalyEvidence]]:
        evidence_list = log_result.get("evidence", [])
        if not isinstance(evidence_list, list):
            evidence_list = []
        
        raw_records = []
        for ev in evidence_list:
            if not isinstance(ev, dict):
                continue
            meta = ev.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            raw_records.append({
                "timestamp": ev.get("timestamp", ""),
                "service": meta.get("service", ev.get("service", "")),
                "level": ev.get("level", ev.get("type", "INFO")).upper(),
                "message": ev.get("content", ev.get("message", "")),
                "error_type": meta.get("error_type", ev.get("error_type")),
                "status_code": meta.get("status_code", ev.get("status_code")),
                "latency_ms": meta.get("latency_ms", ev.get("latency_ms")),
                "upstream": meta.get("upstream", ev.get("upstream")),
                "trace_id": ev.get("trace_id"),
            })
        
        return logs_to_anomaly_data(raw_records)
    
    def _determine_alert_time(
        self,
        symptom_service: str,
        anomaly_data: Dict[str, List[AnomalyEvidence]],
    ) -> datetime:
        service_anomalies = anomaly_data.get(symptom_service, [])
        if service_anomalies:
            return max(a.timestamp for a in service_anomalies)
        
        all_times = [
            a.timestamp
            for anomalies in anomaly_data.values()
            for a in anomalies
        ]
        if all_times:
            return max(all_times)
        
        return datetime.now()
    
    def _safe_float(self, value: Any, default: float = 0.5) -> float:
        try:
            v = float(value)
            return max(0.0, min(0.95, v))
        except (TypeError, ValueError):
            return default
    
    def _fallback_to_tcb_rca(
        self,
        tcb_rca_output: TCBRCAOutput,
        service: str,
        topology_result: Dict[str, Any],
        log_result: Dict[str, Any],
        error_msg: str,
    ) -> Dict[str, Any]:
        """LLM 실패 시 TCB-RCA 결과를 그대로 반환."""
        candidates = []
        for rc in tcb_rca_output.root_cause_candidates:
            candidates.append({
                "rank": rc.rank,
                "cause": rc.cause_description,
                "cause_service": rc.cause_service,
                "confidence": rc.confidence,
                "evidence_refs": [
                    f"tcb-rca:fallback-depth-{rc.depth}",
                ],
                "reasoning": f"TCB-RCA fallback: {rc.cause_description}",
                "supporting_evidence": {
                    "log_agent_agrees": False,
                    "topology_agent_agrees": False,
                    "tcb_rca_agrees": True,
                },
            })
        
        related_services = topology_result.get("related_services", []) or []
        
        if tcb_rca_output.root_cause_candidates:
            top = tcb_rca_output.root_cause_candidates[0]
            summary = (
                f"RCA Agent (FALLBACK to TCB-RCA): identified {top.cause_service} "
                f"(LLM failed: {error_msg})"
            )
            top_confidence = top.confidence
        else:
            summary = f"RCA Agent (FALLBACK): no candidates (LLM failed: {error_msg})"
            top_confidence = 0.0
        
        return {
            "incident_id": tcb_rca_output.incident_id,
            "service": service,
            "algorithm": "TCB-RCA-Fallback",
            "algorithm_version": "2.0-fallback",
            "summary": summary,
            "confidence": top_confidence,
            "overall_confidence": top_confidence,
            "evidence_convergence": "fallback",
            "root_cause_candidates": candidates,
            "affected_services": list(
                dict.fromkeys(tcb_rca_output.propagation_path + [service])
            ),
            "related_services": related_services,
            "propagation_path": tcb_rca_output.propagation_path,
            "blast_radius": tcb_rca_output.blast_radius,
            "traversal_summary": tcb_rca_output.traversal_summary,
            "llm_synthesis_reasoning": f"LLM unavailable: {error_msg}",
            "tcb_rca_reference": {
                "candidates": [
                    {
                        "rank": rc.rank,
                        "cause_service": rc.cause_service,
                        "confidence": rc.confidence,
                        "depth": rc.depth,
                    }
                    for rc in tcb_rca_output.root_cause_candidates[:3]
                ],
                "propagation_path": tcb_rca_output.propagation_path,
            },
            "evidence_chains": [
                {
                    "candidate_rank": rc.rank,
                    "candidate_service": rc.cause_service,
                    "backtrack_path": rc.backtrack_path,
                    "chain": rc.evidence_chain,
                }
                for rc in tcb_rca_output.root_cause_candidates
            ],
            "evidence": [
                {
                    "type": "rca",
                    "source": "tcb-rca-fallback",
                    "content": summary,
                    "metadata": {
                        "incident_id": tcb_rca_output.incident_id,
                        "algorithm": "TCB-RCA-Fallback",
                        "llm_error": error_msg,
                    },
                }
            ],
            "_llm_error": error_msg,
        }
