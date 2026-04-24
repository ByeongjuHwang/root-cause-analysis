"""
Orchestrator Service — coordinates the multi-agent RCA pipeline.

Paper artifact enhancements:
    - Graceful degradation: if any agent fails, the pipeline continues
      with a degraded result rather than crashing the whole analysis.
    - topology_graph and service_metadata are forwarded from the
      Topology Agent to the RCA Agent so TCB-RCA uses dynamic topology.
    - agent_errors are tracked and included in the final result metadata.
"""

from __future__ import annotations

import logging
import os
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

logger = logging.getLogger(__name__)


def _empty_agent_result(agent_name: str, error_msg: str) -> AgentResult:
    """Create a minimal AgentResult for a failed agent."""
    return AgentResult(
        agent=agent_name,
        summary=f"[DEGRADED] Agent call failed: {error_msg}",
        confidence=0.0,
        evidence=[],
        metadata={"degraded": True, "error": error_msg},
    )


class OrchestratorService:
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
        agent_errors: List[Dict[str, str]] = []

        # ==============================================================
        # 1) Log Agent
        # ==============================================================
        try:
            _attachments = incident.attachments or {}
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
                    "log_file": _attachments.get("log_file"),
                    # v6: dual analysis windows — Log Agent uses these for
                    # precise baseline vs incident statistics.
                    "baseline_range": _attachments.get("baseline_range"),
                    "incident_range": _attachments.get("incident_range"),
                    # v8: metrics.csv path — Log Agent feeds this to the
                    # metric MCP tools (get_metric_summary / latency / retry).
                    "metrics_file": _attachments.get("metrics_file"),
                },
            )
            log_result = self._parse_agent_result(log_raw, "log_agent")
        except Exception as exc:
            err = f"Log Agent failed: {exc}"
            logger.error(err)
            agent_errors.append({"agent": "log_agent", "error": str(exc)})
            log_result = _empty_agent_result("log_agent", str(exc))

        # ==============================================================
        # 2) Topology Agent
        # ==============================================================
        try:
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
        except Exception as exc:
            err = f"Topology Agent failed: {exc}"
            logger.error(err)
            agent_errors.append({"agent": "topology_agent", "error": str(exc)})
            topology_result = _empty_agent_result("topology_agent", str(exc))

        # ==============================================================
        # 3) RCA Agent — pass topology_graph from Topology Agent
        # ==============================================================
        try:
            topo_meta_for_rca = {
                "service": incident.service,
                "summary": topology_result.summary,
                "confidence": topology_result.confidence,
                "evidence": [e.model_dump() for e in topology_result.evidence],
                **topology_result.metadata,
            }

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
                    "topology_result": topo_meta_for_rca,
                    "desired_output": "application/json",
                },
            )
            rca_result = self._parse_rca_result(rca_raw)
        except Exception as exc:
            err = f"RCA Agent failed: {exc}"
            logger.error(err)
            agent_errors.append({"agent": "rca_agent", "error": str(exc)})
            rca_result = {
                "summary": f"[DEGRADED] {exc}",
                "confidence": 0.0,
                "root_cause_candidates": [],
                "affected_services": [incident.service],
                "related_services": [],
                "propagation_path": [],
                "blast_radius": [],
            }

        # ==============================================================
        # 4) Verifier Agent (graceful: skip if unavailable)
        # ==============================================================
        try:
            draft_rca = {
                "root_cause_candidates": self._ensure_list(rca_result.get("root_cause_candidates", [])),
                "affected_services": self._ensure_list(rca_result.get("affected_services", [incident.service])),
                "related_services": self._ensure_list(rca_result.get("related_services", [])),
                "propagation_path": self._ensure_list(
                    rca_result.get(
                        "propagation_path",
                        topology_result.metadata.get("propagation_path", []),
                    )
                ),
            }

            verifier_raw = await self.a2a_client.send_message(
                agent_base_url=self.verifier_agent_url,
                text=self._build_verifier_prompt(incident),
                metadata={
                    "incident_id": incident.incident_id,
                    "service": incident.service,
                    "draft_rca": draft_rca,
                    "agent_results": {
                        "log_agent": {
                            "summary": log_result.summary,
                            "confidence": log_result.confidence,
                            "evidence": [e.model_dump() for e in log_result.evidence],
                        },
                        "topology_agent": {
                            "summary": topology_result.summary,
                            "confidence": topology_result.confidence,
                            "propagation_path": self._ensure_list(topology_result.metadata.get("propagation_path", [])),
                            "blast_radius": self._ensure_list(topology_result.metadata.get("blast_radius", [])),
                            "related_services": self._ensure_list(topology_result.metadata.get("related_services", [])),
                        },
                        "rca_agent": rca_result,
                    },
                    "desired_output": "application/json",
                },
            )
            verification = self._parse_verification(verifier_raw)
        except Exception as exc:
            err = f"Verifier Agent failed: {exc}"
            logger.error(err)
            agent_errors.append({"agent": "verifier_agent", "error": str(exc)})
            verification = {
                "verdict": "skipped",
                "verification_notes": [f"Verifier unavailable: {exc}"],
                "revised_root_cause_candidates": [],
                "final_confidence": 0.0,
                "explanation": "Verification skipped due to agent failure.",
            }

        # ==============================================================
        # Phase 4d-2: Adaptive Execution
        # ==============================================================
        # If Verifier's completeness_score is too low, re-invoke Log Agent
        # with focus_services hint (from Verifier's missing_evidence).
        # Recompute RCA and Verifier with the new Log Agent output.
        # Repeat up to ADAPTIVE_MAX_ITERATIONS.
        #
        # Requires A2A_PARSE_CONTRACTS=on (to read completeness_score from
        # verifier's _agent_response). If that flag is off, adaptive is a
        # no-op.
        adaptive_iterations: List[Dict[str, Any]] = []
        if (
            os.getenv("ADAPTIVE_EXECUTION", "off") != "off"
            and os.getenv("A2A_PARSE_CONTRACTS", "off") != "off"
        ):
            log_result, rca_result, verification, adaptive_iterations = (
                await self._run_adaptive_iterations(
                    incident=incident,
                    log_result=log_result,
                    topology_result=topology_result,
                    rca_result=rca_result,
                    verification=verification,
                    agent_errors=agent_errors,
                )
            )

        # ==============================================================
        # Assemble final result
        # ==============================================================
        final_candidates = self._normalize_verified_candidates(
            self._ensure_list(verification.get("revised_root_cause_candidates", []))
        )
        if not final_candidates:
            final_candidates = self._normalize_verified_candidates(
                self._ensure_list(rca_result.get("root_cause_candidates", []))
            )

        top_cause = final_candidates[0].cause if final_candidates else "No strong root cause identified."
        top_confidence = final_candidates[0].confidence if final_candidates else 0.0
        top_cause_service = final_candidates[0].cause_service if final_candidates else None

        evidence_summary_dict = self._build_evidence_summary(log_result, topology_result)

        final_verdict = FinalVerdict(
            cause=top_cause,
            confidence=top_confidence,
            explanation=self._build_final_explanation(final_candidates, verification),
            cause_service=top_cause_service,
        )

        impact_analysis = ImpactAnalysis(
            affected_services=self._string_list(rca_result.get("affected_services", [incident.service])),
            related_services=self._string_list(rca_result.get("related_services", [])),
            propagation_path=self._string_list(
                rca_result.get(
                    "propagation_path",
                    topology_result.metadata.get("propagation_path", []),
                )
            ),
            blast_radius=self._string_list(
                rca_result.get(
                    "blast_radius",
                    topology_result.metadata.get("blast_radius", []),
                )
            ),
        )

        verification_result = VerificationResult(
            verdict=str(verification.get("verdict", "rejected")),
            final_confidence=float(verification.get("final_confidence", 0.0)),
            notes=self._string_list(verification.get("verification_notes", [])),
        )

        evidence_summary = EvidenceSummary(
            log_evidence=evidence_summary_dict["log_evidence"],
            topology_evidence=evidence_summary_dict["topology_evidence"],
        )

        # ==============================================================
        # Phase 4d-1: Collect _agent_response from each agent (contracts)
        # ==============================================================
        # When A2A_PARSE_CONTRACTS is on, we assemble a dict of every agent's
        # structured AgentResponse so it appears as a first-class field in
        # FinalRCAResult. This makes the paper's A2A-as-contract claim
        # directly inspectable in the final output. No runtime behaviour
        # changes — the contract parsing is additive.
        agent_contracts: Optional[Dict[str, Dict[str, Any]]] = None
        if os.getenv("A2A_PARSE_CONTRACTS", "off") != "off":
            agent_contracts = {}
            log_contract = log_result.metadata.get("_agent_response")
            if isinstance(log_contract, dict):
                agent_contracts["log_agent"] = log_contract
            topo_contract = topology_result.metadata.get("_agent_response")
            if isinstance(topo_contract, dict):
                agent_contracts["topology_agent"] = topo_contract
            rca_contract = rca_result.get("_agent_response")
            if isinstance(rca_contract, dict):
                agent_contracts["rca_agent"] = rca_contract
            verifier_contract = verification.get("_agent_response")
            if isinstance(verifier_contract, dict):
                agent_contracts["verifier_agent"] = verifier_contract
            if not agent_contracts:
                # No contracts seen — likely A2A_CONTRACT_MODE is off
                agent_contracts = None

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
            # LLM 버전 메타데이터
            evidence_convergence=rca_result.get("evidence_convergence"),
            synthesis_reasoning=rca_result.get("llm_synthesis_reasoning"),
            tcb_rca_reference=rca_result.get("tcb_rca_reference"),
            # Degradation tracking
            agent_errors=agent_errors if agent_errors else None,
            # Phase 4d-1: contract-based architecture observability
            agent_contracts=agent_contracts,
            # Phase 4d-2: record any re-invocations triggered by low completeness
            adaptive_iterations=(
                adaptive_iterations if adaptive_iterations else None
            ),
        )

    # ------------------------------------------------------------------
    # Phase 4d-2: Adaptive Execution helper
    # ------------------------------------------------------------------

    async def _run_adaptive_iterations(
        self,
        incident: IncidentRequest,
        log_result: AgentResult,
        topology_result: AgentResult,
        rca_result: Dict[str, Any],
        verification: Dict[str, Any],
        agent_errors: List[Dict[str, str]],
    ) -> tuple:
        """Re-invoke Log Agent → RCA → Verifier if completeness_score is low.

        Returns the (possibly updated) (log_result, rca_result, verification,
        iterations_log) tuple. If the initial pass already has sufficient
        completeness, this is a no-op and returns an empty iterations_log.

        Topology Agent is NOT re-invoked — its output is purely structural
        (dependency graph) and doesn't benefit from re-asking.
        """
        threshold = float(os.getenv("ADAPTIVE_THRESHOLD", "0.5"))
        max_iter = int(os.getenv("ADAPTIVE_MAX_ITERATIONS", "3"))
        iterations: List[Dict[str, Any]] = []

        for iter_num in range(1, max_iter + 1):
            score = self._extract_completeness_score(verification)
            if score is None:
                # No contract available → cannot judge → stop
                iterations.append({
                    "iteration": iter_num,
                    "reason": "no_contract_available",
                    "completeness_score": None,
                    "action": "stop",
                })
                break
            if score >= threshold:
                # Good enough → stop
                iterations.append({
                    "iteration": iter_num,
                    "reason": "completeness_sufficient",
                    "completeness_score": score,
                    "threshold": threshold,
                    "action": "stop",
                })
                break

            # Below threshold → extract focus services and re-invoke
            focus = self._extract_focus_services(verification, top_k=5)
            if not focus:
                iterations.append({
                    "iteration": iter_num,
                    "reason": "no_focus_services_derivable",
                    "completeness_score": score,
                    "threshold": threshold,
                    "action": "stop",
                })
                break

            iter_record: Dict[str, Any] = {
                "iteration": iter_num,
                "reason": "completeness_below_threshold",
                "completeness_score": score,
                "threshold": threshold,
                "focus_services": focus,
                "action": "re_invoke_log_agent",
            }

            try:
                # Re-invoke Log Agent with focus_services hint
                _attachments = incident.attachments or {}
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
                        "log_file": _attachments.get("log_file"),
                        "baseline_range": _attachments.get("baseline_range"),
                        "incident_range": _attachments.get("incident_range"),
                        "metrics_file": _attachments.get("metrics_file"),
                        # Phase 4d-2 adaptive hint
                        "focus_services": focus,
                    },
                )
                log_result = self._parse_agent_result(log_raw, "log_agent")
            except Exception as exc:
                iter_record["error"] = f"log_agent_re_invoke_failed: {exc}"
                iterations.append(iter_record)
                agent_errors.append({
                    "agent": f"log_agent_iter_{iter_num}",
                    "error": str(exc),
                })
                break

            # Re-invoke RCA with new log_result
            try:
                topo_meta_for_rca = {
                    "service": incident.service,
                    "summary": topology_result.summary,
                    "confidence": topology_result.confidence,
                    "evidence": [e.model_dump() for e in topology_result.evidence],
                    **topology_result.metadata,
                }
                rca_raw = await self.a2a_client.send_message(
                    agent_base_url=self.rca_agent_url,
                    text="Synthesize root cause candidates (adaptive iteration).",
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
                        "topology_result": topo_meta_for_rca,
                        "desired_output": "application/json",
                    },
                )
                rca_result = self._parse_rca_result(rca_raw)
            except Exception as exc:
                iter_record["error"] = f"rca_agent_re_invoke_failed: {exc}"
                iterations.append(iter_record)
                agent_errors.append({
                    "agent": f"rca_agent_iter_{iter_num}",
                    "error": str(exc),
                })
                break

            # Re-invoke Verifier
            try:
                draft_rca = {
                    "root_cause_candidates": self._ensure_list(
                        rca_result.get("root_cause_candidates", [])),
                    "affected_services": self._ensure_list(
                        rca_result.get("affected_services", [incident.service])),
                    "related_services": self._ensure_list(
                        rca_result.get("related_services", [])),
                    "propagation_path": self._ensure_list(
                        rca_result.get(
                            "propagation_path",
                            topology_result.metadata.get("propagation_path", []),
                        )
                    ),
                }
                verifier_raw = await self.a2a_client.send_message(
                    agent_base_url=self.verifier_agent_url,
                    text=self._build_verifier_prompt(incident),
                    metadata={
                        "incident_id": incident.incident_id,
                        "service": incident.service,
                        "draft_rca": draft_rca,
                        "agent_results": {
                            "log_agent": {
                                "summary": log_result.summary,
                                "confidence": log_result.confidence,
                                "evidence": [e.model_dump() for e in log_result.evidence],
                            },
                            "topology_agent": {
                                "summary": topology_result.summary,
                                "confidence": topology_result.confidence,
                                "propagation_path": self._ensure_list(
                                    topology_result.metadata.get("propagation_path", [])),
                                "blast_radius": self._ensure_list(
                                    topology_result.metadata.get("blast_radius", [])),
                                "related_services": self._ensure_list(
                                    topology_result.metadata.get("related_services", [])),
                            },
                            "rca_agent": rca_result,
                        },
                        "desired_output": "application/json",
                    },
                )
                verification = self._parse_verification(verifier_raw)
                iter_record["new_completeness_score"] = (
                    self._extract_completeness_score(verification)
                )
            except Exception as exc:
                iter_record["error"] = f"verifier_re_invoke_failed: {exc}"
                iterations.append(iter_record)
                agent_errors.append({
                    "agent": f"verifier_agent_iter_{iter_num}",
                    "error": str(exc),
                })
                break

            iterations.append(iter_record)

        return log_result, rca_result, verification, iterations

    def _extract_completeness_score(
        self, verification: Dict[str, Any]
    ) -> Optional[float]:
        """Read Verifier's completeness_score from the embedded AgentResponse.

        Returns None if no contract was emitted (A2A_CONTRACT_MODE off)
        or if parsing fails.
        """
        contract = verification.get("_agent_response")
        if not isinstance(contract, dict):
            return None
        score = contract.get("completeness_score")
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    def _extract_focus_services(
        self, verification: Dict[str, Any], top_k: int = 5,
    ) -> List[str]:
        """Pick services that Verifier deems needing more evidence.

        Heuristic:
          1. Top candidates from Verifier's AgentResponse.candidates (by confidence)
          2. Plus any services in rejected_candidates list (need more investigation)

        If AgentResponse missing, return [].
        """
        contract = verification.get("_agent_response")
        if not isinstance(contract, dict):
            return []

        services: List[str] = []
        seen: set = set()

        # 1. Top candidates — likely suspects that need evidence shoring up
        for cand in contract.get("candidates", []) or []:
            if not isinstance(cand, dict):
                continue
            svc = cand.get("service")
            if not svc or svc in seen:
                continue
            # Only focus on candidates that have missing_evidence
            missing = cand.get("missing_evidence", [])
            if missing:
                services.append(svc)
                seen.add(svc)
            if len(services) >= top_k:
                break

        return services

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_log_agent_prompt(self, incident: IncidentRequest) -> str:
        return f"""\
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
5. suspected downstream service if identifiable"""

    def _build_topology_agent_prompt(
        self,
        incident: IncidentRequest,
        suspected_downstream: Optional[str],
    ) -> str:
        return f"""\
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
7. topology_graph (full)
8. evidence"""

    def _build_verifier_prompt(self, incident: IncidentRequest) -> str:
        return f"""\
Verify the draft RCA for incident {incident.incident_id}.

Target service: {incident.service}
Symptom: {incident.symptom}

Validate:
1. evidence consistency
2. topology consistency
3. confidence recalibration
4. revised ranking of root cause candidates

Return JSON with:
1. verdict
2. verification_notes
3. revised_root_cause_candidates
4. final_confidence
5. explanation"""

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

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
            evidence_items.append(
                Evidence(
                    type=str(item.get("type", "unknown")),
                    source=str(item.get("source", "unknown")),
                    content=str(item.get("content", "")),
                    timestamp=item.get("timestamp"),
                    level=item.get("level"),
                    trace_id=item.get("trace_id"),
                    metadata=item.get("metadata", {}) or {},
                )
            )

        metadata: Dict[str, Any] = {}
        for key in (
            "propagation_path",
            "blast_radius",
            "dependency_info",
            "related_services",
            "hypothesis",
            "suspected_downstream",
            # Topology Agent full graph (핵심 추가)
            "topology_graph",
            "service_metadata",
            "topology_file",
            # LLM 버전에서 추가로 제공하는 필드들
            "anomalous_services",
            "llm_reasoning",
            "path_assessment",
            "alternative_paths",
            "critical_services_in_blast",
            "topology_supports_hypothesis",
            "referenced_upstreams",
            # Phase 3b: evidence_collection for downstream agents
            "evidence_collection",
            # Phase 4b: structured AgentResponse (when A2A_CONTRACT_MODE is on)
            "_agent_response",
        ):
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

        return {
            "summary": "",
            "confidence": 0.0,
            "root_cause_candidates": [],
            "affected_services": [],
            "related_services": [],
            "propagation_path": [],
            "blast_radius": [],
        }

    def _parse_verification(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        task = raw.get("result", {}).get("task", {})
        artifacts = task.get("artifacts", [])

        if artifacts and isinstance(artifacts, list):
            first_artifact = artifacts[0]
            if isinstance(first_artifact, dict):
                parts = first_artifact.get("parts", [])
                if parts and isinstance(parts, list):
                    first_part = parts[0]
                    if isinstance(first_part, dict):
                        data = first_part.get("data", {}) or {}
                        return {
                            "verdict": data.get("verdict", "rejected"),
                            "verification_notes": data.get("verification_notes", []),
                            "revised_root_cause_candidates": data.get("revised_root_cause_candidates", []),
                            "final_confidence": data.get("final_confidence", 0.0),
                            "explanation": data.get("explanation", ""),
                            # Phase 4b: carry through AgentResponse when present
                            "_agent_response": data.get("_agent_response"),
                        }

        return {
            "verdict": "rejected",
            "verification_notes": ["No verification artifact returned."],
            "revised_root_cause_candidates": [],
            "final_confidence": 0.0,
            "explanation": "No verification artifact returned.",
        }

    # ------------------------------------------------------------------
    # Candidate normalization
    # ------------------------------------------------------------------

    def _normalize_verified_candidates(
        self,
        raw_candidates: List[Dict[str, Any]],
    ) -> List[RootCauseCandidate]:
        normalized: List[RootCauseCandidate] = []

        for idx, candidate in enumerate(raw_candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            normalized.append(
                RootCauseCandidate(
                    rank=int(candidate.get("rank", idx)),
                    cause=str(candidate.get("cause", "Unknown cause")),
                    confidence=float(candidate.get("confidence", 0.5)),
                    evidence_refs=self._string_list(candidate.get("evidence_refs", [])),
                    cause_service=candidate.get("cause_service"),
                    reasoning=candidate.get("reasoning"),
                    supporting_evidence=candidate.get("supporting_evidence"),
                )
            )

        normalized.sort(key=lambda c: c.confidence, reverse=True)
        for idx, candidate in enumerate(normalized, start=1):
            candidate.rank = idx

        return normalized

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _guess_downstream_from_log_result(
        self,
        root_service: str,
        log_result: AgentResult,
    ) -> Optional[str]:
        suspected = log_result.metadata.get("suspected_downstream")
        if suspected and suspected != root_service:
            return str(suspected)

        summary_guess = self._guess_downstream_from_text(root_service, log_result.summary)
        if summary_guess:
            return summary_guess

        for evidence in log_result.evidence:
            metadata = evidence.metadata or {}
            upstream = metadata.get("upstream")
            if upstream and upstream != root_service:
                return str(upstream)
            service_name = metadata.get("service")
            if service_name and service_name != root_service:
                return str(service_name)
            guessed = self._guess_downstream_from_text(root_service, evidence.content)
            if guessed:
                return guessed

        return None

    def _guess_downstream_from_text(
        self,
        root_service: str,
        text: Optional[str],
    ) -> Optional[str]:
        if not text:
            return None

        text_lower = text.lower()
        known_services = [
            "auth-service", "catalog-service", "order-service",
            "worker-service", "message-queue", "user-db", "order-db",
            "config-service",
        ]

        for service in known_services:
            if service in text_lower and service != root_service:
                return service

        return None

    def _build_evidence_summary(
        self,
        log_result: AgentResult,
        topology_result: AgentResult,
    ) -> Dict[str, List[str]]:
        log_evidence = [e.content for e in log_result.evidence[:5]]
        topology_evidence = [e.content for e in topology_result.evidence[:5]]

        propagation_path = self._string_list(topology_result.metadata.get("propagation_path", []))
        if propagation_path:
            topology_evidence.append(f"propagation path: {' -> '.join(propagation_path)}")

        return {
            "log_evidence": log_evidence,
            "topology_evidence": topology_evidence,
        }

    def _build_final_explanation(
        self,
        final_candidates: List[RootCauseCandidate],
        verification: Dict[str, Any],
    ) -> str:
        explanation = str(verification.get("explanation") or "").strip()
        if explanation:
            return explanation

        if final_candidates:
            top = final_candidates[0]
            return f"The top-ranked root cause is '{top.cause}' with confidence {top.confidence:.2f}."

        return "No strong root cause could be identified."

    def _ensure_list(self, value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        return []

    def _string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(v) for v in value if v is not None]
