"""
Topology Analysis Agent — LLM-powered version.

기존 skills.py의 구조를 유지하되, 다음을 LLM으로 교체/보강:
- 정적 의존성 조회 결과에 대한 LLM 해석 추가
- 장애 전파 경로 후보들에 대한 LLM 우선순위 추론
- blast radius의 중요도 판단
- 자연어 형태의 토폴로지 분석 summary

유지되는 부분 (결정론적):
- MCP를 통한 토폴로지 조회 (get_service_dependencies, get_related_services, 
  find_path, infer_blast_radius)
- 그래프 기반 경로 탐색
- blast radius 계산

철학: "LLM은 구조 이해와 우선순위 추론, 규칙은 그래프 탐색"
"""

from typing import Any, Dict, List, Optional
import os

from mcp_servers.architecture_mcp.app.repository import (
    get_service_dependencies,
    get_related_services,
    find_path,
    infer_blast_radius,
)
from common.llm_client import get_default_client
from .skills import _build_full_topology


# =========================================================================
# Prompts
# =========================================================================

TOPOLOGY_AGENT_SYSTEM_PROMPT = """You are a microservice topology expert. 
Your role is to analyze the dependency structure of a microservice system during 
an incident and reason about fault propagation paths.

Your analysis approach:
1. Given the symptom service, identify which services are in its transitive 
   dependency chain (services it depends on, directly or indirectly)
2. Evaluate which propagation path from a suspected root cause to the symptom 
   is most plausible based on dependency structure
3. Assess the blast radius — which services would be affected if this fault 
   propagates further
4. Consider service criticality and type when ranking propagation likelihood

You are NOT analyzing log evidence — only the TOPOLOGY STRUCTURE and the hint 
provided by the log agent. Your job is to reason about which structural paths 
are consistent with the symptom.

Respond ONLY with valid JSON. No markdown code blocks, no commentary."""


def build_topology_agent_user_prompt(
    symptom_service: str,
    suspected_downstream: Optional[str],
    dependency_info: Dict[str, Any],
    related_services: List[str],
    blast_radius: List[str],
    computed_path: List[str],
) -> str:
    """Topology Agent용 user prompt 생성."""
    
    lines = []
    lines.append("## Incident Context")
    lines.append(f"- Symptom service: {symptom_service}")
    if suspected_downstream:
        lines.append(f"- Log agent's suspected downstream: {suspected_downstream}")
    else:
        lines.append("- Log agent did not identify a clear suspected downstream")
    lines.append("")
    
    # Dependency info
    lines.append("## Topology Structure (from architecture MCP)")
    service = dependency_info.get("service", symptom_service)
    depends_on = dependency_info.get("depends_on", []) or []
    upstream_of = dependency_info.get("upstream_of", []) or []
    svc_type = dependency_info.get("type", "unknown")
    criticality = dependency_info.get("criticality", "unknown")
    
    lines.append(f"- Symptom service '{service}' (type: {svc_type}, criticality: {criticality})")
    lines.append(f"  - Depends on (calls): {depends_on if depends_on else '(none)'}")
    lines.append(f"  - Called by: {upstream_of if upstream_of else '(none)'}")
    lines.append("")
    
    lines.append(f"- Related services in dependency subgraph: {related_services if related_services else '(none)'}")
    lines.append("")
    
    # Computed propagation path
    if computed_path:
        path_str = " → ".join(computed_path)
        lines.append(f"## Computed Propagation Path (graph search)")
        lines.append(f"- From {symptom_service} to {suspected_downstream or '?'}: {path_str}")
        lines.append("")
    
    # Blast radius
    if blast_radius:
        lines.append(f"## Blast Radius")
        lines.append(f"- Services potentially affected: {', '.join(blast_radius)}")
        lines.append("")
    
    lines.append("## Analysis Task")
    lines.append(
        "Think step by step:\n"
        "1. Is the computed propagation path structurally plausible?\n"
        "2. Are there alternative paths from the symptom to the suspected root cause?\n"
        "3. How confident are you in the propagation path given the topology alone?\n"
        "4. Within the blast radius, which services are MOST critical to monitor?\n"
        "5. Does the topology support the log agent's hypothesis about suspected_downstream?"
    )
    
    return "\n".join(lines)


TOPOLOGY_AGENT_SCHEMA_HINT = """{
  "propagation_path_assessment": "<plausible | alternative_exists | implausible | no_path>",
  "propagation_path_confidence": 0.XX,
  "alternative_paths": [["<service>", "<service>", ...]],
  "critical_services_in_blast": ["<service_name>", ...],
  "topology_supports_hypothesis": true_or_false,
  "reasoning": "<brief explanation of structural analysis>"
}"""


# =========================================================================
# Service
# =========================================================================

# Phase 4b: A2A contract dual-output helper (same pattern as RCA Agent).
def _maybe_attach_topology_response(
    result: dict,
    incident_id: str,
) -> dict:
    """Attach AgentResponse when A2A_CONTRACT_MODE is set; else return as-is.

    Topology Agent has no upstream (it runs independently of Log Agent), so
    we build a standalone AgentResponse with an empty evidence_collection.
    """
    if os.getenv("A2A_CONTRACT_MODE", "off") == "off":
        return result
    try:
        from common.response_builder import (
            build_topology_agent_response,
            attach_agent_response,
        )
        agent_resp = build_topology_agent_response(
            legacy_result=result,
            request_id=incident_id or "UNKNOWN",
        )
        attach_agent_response(result, agent_resp)
    except Exception:
        pass
    return result


class ArchitectureAdapter:
    """MCP 기반 아키텍처 조회 래퍼. 기존과 동일."""
    
    async def get_service_dependencies(
        self, service: str, topology_file: Optional[str] = None
    ) -> Dict[str, Any]:
        return get_service_dependencies(service, topology_file=topology_file)
    
    async def get_related_services(
        self, service: str, topology_file: Optional[str] = None
    ) -> Dict[str, Any]:
        return get_related_services(service, topology_file=topology_file)
    
    async def find_path(
        self, source: str, target: str, topology_file: Optional[str] = None
    ) -> Dict[str, Any]:
        return find_path(source, target, topology_file=topology_file)
    
    async def infer_blast_radius(
        self, service: str, depth: int = 2, topology_file: Optional[str] = None
    ) -> Dict[str, Any]:
        return infer_blast_radius(service, depth, topology_file=topology_file)


class TopologyAnalysisServiceLLM:
    """LLM 기반 Topology Analysis Agent."""
    
    def __init__(self):
        self.arch = ArchitectureAdapter()
        self.llm = get_default_client()
    
    async def analyze(
        self,
        service: str,
        suspected_downstream: Optional[str] = None,
        diagram_uri: Optional[str] = None,
        topology_file: Optional[str] = None,
        incident_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """토폴로지 분석 메인 엔트리포인트.
        
        1) MCP로 토폴로지 조회 (결정론적)
        2) 경로 탐색 및 blast radius 계산 (결정론적)
        3) LLM으로 구조적 추론 및 우선순위 판단
        4) 결과 통합 반환
        """
        
        # === Step 1: MCP 조회 (결정론적, 기존 로직 유지) ===
        deps = await self.arch.get_service_dependencies(service, topology_file=topology_file)
        related = await self.arch.get_related_services(service, topology_file=topology_file)
        blast = await self.arch.infer_blast_radius(
            service=service, depth=2, topology_file=topology_file
        )
        
        propagation = []
        if suspected_downstream:
            path_result = await self.arch.find_path(
                service, suspected_downstream, topology_file=topology_file
            )
            propagation = path_result.get("path", []) or []
        
        related_services = related.get("related_services", []) or []
        blast_radius = blast.get("blast_radius", []) or []
        
        # === Step 2: LLM 호출로 구조적 추론 ===
        user_prompt = build_topology_agent_user_prompt(
            symptom_service=service,
            suspected_downstream=suspected_downstream,
            dependency_info=deps,
            related_services=related_services,
            blast_radius=blast_radius,
            computed_path=propagation,
        )
        
        llm_result = await self.llm.call_json(
            agent_name="topology_agent",
            system_prompt=TOPOLOGY_AGENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            incident_id=incident_id,
            schema_hint=TOPOLOGY_AGENT_SCHEMA_HINT,
        )
        
        # === Step 3: LLM 결과 처리 ===
        if "_error" in llm_result:
            fallback = self._fallback_result(
                service, suspected_downstream, deps, related_services,
                blast_radius, propagation, diagram_uri, topology_file,
                llm_result.get("_error", "LLM error"),
            )
            return _maybe_attach_topology_response(fallback, incident_id or "UNKNOWN")
        
        path_assessment = llm_result.get("propagation_path_assessment", "unknown")
        try:
            path_conf = float(llm_result.get("propagation_path_confidence", 0.6))
        except (TypeError, ValueError):
            path_conf = 0.6
        path_conf = max(0.0, min(0.95, path_conf))
        
        alternative_paths = llm_result.get("alternative_paths", []) or []
        critical_services = llm_result.get("critical_services_in_blast", []) or []
        topology_supports = bool(llm_result.get("topology_supports_hypothesis", True))
        reasoning = llm_result.get("reasoning", "") or ""
        
        # Confidence 산출: 기본값 + LLM 추론 조정
        if path_assessment == "plausible":
            final_confidence = 0.76 + (path_conf - 0.6) * 0.3  # 상향 조정
        elif path_assessment == "alternative_exists":
            final_confidence = 0.65
        elif path_assessment == "implausible":
            final_confidence = 0.40
        else:
            final_confidence = 0.60
        final_confidence = max(0.30, min(0.95, final_confidence))
        
        # === Step 4: Summary 생성 (LLM 추론 반영) ===
        summary_parts = [
            f"{service} depends on {deps.get('depends_on', [])}.",
            f"Related services: {related_services}.",
        ]
        if propagation:
            summary_parts.append(
                f"Propagation path from {service} to {suspected_downstream}: "
                f"{' → '.join(propagation)}."
            )
        summary_parts.append(f"Blast radius: {blast_radius}.")
        summary_parts.append(f"LLM assessment: {path_assessment} (confidence {path_conf:.2f}).")
        
        # Build full topology for RCA Agent consumption
        full_topo = _build_full_topology(topology_file=topology_file)

        result = {
            "summary": " ".join(summary_parts),
            "confidence": round(final_confidence, 3),
            "dependency_info": deps,
            "related_services": related_services,
            "propagation_path": propagation,
            "blast_radius": blast_radius,
            "topology_file": topology_file,
            # Full topology for RCA engine
            "topology_graph": full_topo["topology_graph"],
            "service_metadata": full_topo["service_metadata"],
            # LLM-specific fields
            "path_assessment": path_assessment,
            "alternative_paths": alternative_paths,
            "critical_services_in_blast": critical_services,
            "topology_supports_hypothesis": topology_supports,
            "llm_reasoning": reasoning,
            "evidence": [
                {
                    "type": "topology",
                    "source": diagram_uri or "arch://system/latest",
                    "content": f"Dependency map confirms {service} relations: {deps}. "
                               f"Structural analysis: {path_assessment}.",
                    "metadata": {
                        "service": service,
                        "topology_file": topology_file,
                        "llm_reasoning": reasoning,
                    },
                }
            ],
        }
        return _maybe_attach_topology_response(result, incident_id or "UNKNOWN")
    
    def _fallback_result(
        self,
        service: str,
        suspected_downstream: Optional[str],
        deps: Dict[str, Any],
        related_services: List[str],
        blast_radius: List[str],
        propagation: List[str],
        diagram_uri: Optional[str],
        topology_file: Optional[str],
        error_msg: str,
    ) -> Dict[str, Any]:
        """LLM 호출 실패 시 결정론적 결과 반환."""
        summary_parts = [
            f"{service} depends on {deps.get('depends_on', [])}.",
            f"Related services are {related_services}.",
        ]
        if propagation:
            summary_parts.append(
                f"Likely propagation path from {service} to {suspected_downstream}: {propagation}."
            )
        summary_parts.append(f"Estimated blast radius: {blast_radius}.")
        summary_parts.append(f"(LLM failed: {error_msg})")
        
        # Build full topology even in fallback mode
        try:
            full_topo = _build_full_topology(topology_file=topology_file)
        except Exception:
            full_topo = {"topology_graph": {}, "service_metadata": {}}

        return {
            "summary": " ".join(summary_parts),
            "confidence": 0.76 if propagation else 0.68,
            "dependency_info": deps,
            "related_services": related_services,
            "propagation_path": propagation,
            "blast_radius": blast_radius,
            "topology_file": topology_file,
            "topology_graph": full_topo["topology_graph"],
            "service_metadata": full_topo["service_metadata"],
            "path_assessment": "fallback",
            "alternative_paths": [],
            "critical_services_in_blast": [],
            "topology_supports_hypothesis": True,
            "llm_reasoning": f"ERROR: {error_msg}",
            "evidence": [
                {
                    "type": "topology",
                    "source": diagram_uri or "arch://system/latest",
                    "content": f"Dependency map (fallback mode): {deps}",
                    "metadata": {"service": service, "topology_file": topology_file},
                }
            ],
            "_llm_error": error_msg,
        }
