"""
Verifier Agent Service — LLM-aware cross-validation.

Option B 설계 (from thesis framework design):
  - LLM이 제공하는 최종 판정을 결정론적 규칙으로 재검증
  - LLM hallucination (존재하지 않는 서비스 등) 차단
  - 세 증거원 (Log / Topology / TCB-RCA)의 합의 여부 검증
  - 결정론적 baseline (cause_service 없음)과도 호환

검증 단계:
  Stage 1: cause_service 환각 검증 (실제 토폴로지 노드에 존재하는가?)
  Stage 2: supporting_evidence 합의 카운트 (LLM 필드 활용)
  Stage 3: evidence_convergence 기반 confidence 재보정
  Stage 4: 전통적 키워드 매칭 (fallback, 결정론적 버전 호환)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


# 기본 서비스 목록 (fallback, Topology Agent 결과가 없을 때 사용)
_DEFAULT_KNOWN_SERVICES: Set[str] = {
    "frontend-web",
    "api-gateway",
    "auth-service",
    "catalog-service",
    "order-service",
    "worker-service",
    "message-queue",
    "user-db",
    "order-db",
    # Case study 서비스들
    "device-state-gateway",
    "app-deployer",
    "event-api",
    "config-service",
}


def _extract_known_services(agent_results: Dict[str, Any]) -> Set[str]:
    """Extract known services from topology agent results dynamically.

    Falls back to the default set if topology data is unavailable.
    """
    topo = agent_results.get("topology_agent", {})
    if not isinstance(topo, dict):
        return _DEFAULT_KNOWN_SERVICES.copy()

    # Topology Agent now includes related_services and blast_radius
    services: Set[str] = set()

    for key in ("related_services", "blast_radius"):
        vals = topo.get(key, [])
        if isinstance(vals, list):
            services.update(str(v) for v in vals if v)

    # Also extract from propagation_path
    pp = topo.get("propagation_path", [])
    if isinstance(pp, list):
        services.update(str(v) for v in pp if v)

    # RCA agent may have full service lists too
    rca = agent_results.get("rca_agent", {})
    if isinstance(rca, dict):
        for key in ("affected_services", "blast_radius"):
            vals = rca.get(key, [])
            if isinstance(vals, list):
                services.update(str(v) for v in vals if v)

    if services:
        return services | _DEFAULT_KNOWN_SERVICES
    return _DEFAULT_KNOWN_SERVICES.copy()


class VerifierService:
    """LLM 출력과 결정론적 baseline을 모두 검증할 수 있는 하이브리드 Verifier."""
    
    async def verify(
        self,
        incident_id: str,
        service: str,
        draft_rca: Dict[str, Any],
        agent_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        draft_rca = draft_rca or {}
        agent_results = agent_results or {}

        # Dynamically resolve known services from topology data
        known_services = _extract_known_services(agent_results)

        candidates = draft_rca.get("root_cause_candidates", []) or []
        if not isinstance(candidates, list):
            candidates = []

        propagation_path = draft_rca.get("propagation_path", []) or []
        if not isinstance(propagation_path, list):
            propagation_path = []

        log_result = agent_results.get("log_agent", {}) or {}
        topo_result = agent_results.get("topology_agent", {}) or {}
        rca_result = agent_results.get("rca_agent", {}) or {}

        if not isinstance(log_result, dict):
            log_result = {}
        if not isinstance(topo_result, dict):
            topo_result = {}
        if not isinstance(rca_result, dict):
            rca_result = {}

        log_summary = str(log_result.get("summary") or "").lower()
        topo_summary = str(topo_result.get("summary") or "").lower()
        rca_summary = str(rca_result.get("summary") or "").lower()

        topo_path = topo_result.get("propagation_path", []) or propagation_path
        if not isinstance(topo_path, list):
            topo_path = []
        
        # LLM이 제공한 evidence_convergence (전역 메타데이터)
        evidence_convergence = rca_result.get("evidence_convergence")
        
        # LLM이 제공한 anomalous_services (로그 에이전트 결과)
        anomalous_services = log_result.get("anomalous_services", []) or []
        if not isinstance(anomalous_services, list):
            anomalous_services = []

        verification_notes: List[str] = []
        revised_candidates: List[Dict[str, Any]] = []
        rejected_candidates: List[Dict[str, Any]] = []

        for c in candidates:
            if not isinstance(c, dict):
                continue

            cause = str(c.get("cause", ""))
            confidence = float(c.get("confidence", 0.5))
            cause_lower = cause.lower()
            cause_service = c.get("cause_service")  # LLM이 제공하면 있고, 아니면 None
            supporting_evidence = c.get("supporting_evidence") or {}
            
            # =====================================================
            # Stage 1: Hallucination 검증 (LLM 버전 전용)
            # =====================================================
            if cause_service:
                cause_service_normalized = str(cause_service).strip().lower()
                # 토폴로지 노드에 실제로 존재하는지 확인
                known_lower = {s.lower() for s in known_services}
                
                if cause_service_normalized not in known_lower:
                    # Hallucination 감지 → 후보 거부
                    verification_notes.append(
                        f"[REJECTED] cause_service '{cause_service}' does not exist "
                        f"in known topology. Possible LLM hallucination."
                    )
                    rejected_candidates.append({**c, "_verdict": "rejected-hallucination"})
                    continue
            
            # =====================================================
            # Stage 2: Supporting evidence 합의 카운트 (LLM 버전)
            # =====================================================
            agreement_count = 0
            agreement_sources = []
            
            if isinstance(supporting_evidence, dict):
                if supporting_evidence.get("log_agent_agrees"):
                    agreement_count += 1
                    agreement_sources.append("log")
                if supporting_evidence.get("topology_agent_agrees"):
                    agreement_count += 1
                    agreement_sources.append("topology")
                if supporting_evidence.get("tcb_rca_agrees"):
                    agreement_count += 1
                    agreement_sources.append("tcb-rca")
            
            has_llm_metadata = bool(supporting_evidence) and isinstance(supporting_evidence, dict)
            
            # =====================================================
            # Stage 3: Fallback — 전통적 키워드 매칭
            # =====================================================
            matched_log = self._is_supported_by_logs(cause_lower, log_summary)
            matched_topology = self._is_supported_by_topology(
                cause_lower, topo_path, topo_summary
            )
            matched_rca = self._is_supported_by_rca(cause_lower, rca_summary)
            
            # cause_service 있으면 토폴로지 path에서도 확인
            if cause_service:
                cs_lower = cause_service.lower()
                path_lower = [str(p).lower() for p in topo_path]
                if cs_lower in path_lower:
                    matched_topology = True
                # Log summary나 anomalous_services에서도 확인
                if cs_lower in log_summary or cause_service in anomalous_services:
                    matched_log = True
            
            # =====================================================
            # Stage 4: Confidence 재보정
            # =====================================================
            verdict_for_candidate = "weak-evidence"
            
            if has_llm_metadata:
                # LLM 버전: supporting_evidence 합의 수로 주로 판정
                if agreement_count >= 2:
                    confidence = min(0.95, confidence + 0.08)
                    verification_notes.append(
                        f"Candidate '{cause_service or cause}' confirmed by "
                        f"{agreement_count}/3 sources ({', '.join(agreement_sources)})."
                    )
                    verdict_for_candidate = "accepted"
                elif agreement_count == 1:
                    # 1개만 지지 → 키워드 매칭으로 보강 시도
                    if matched_log and matched_topology:
                        confidence = min(0.90, confidence + 0.03)
                        verification_notes.append(
                            f"Candidate '{cause_service or cause}' has 1 LLM source "
                            f"agreement but supported by keyword matching "
                            f"(log + topology)."
                        )
                        verdict_for_candidate = "revised"
                    else:
                        confidence = max(0.40, confidence - 0.05)
                        verification_notes.append(
                            f"Candidate '{cause_service or cause}' supported by only "
                            f"1/3 LLM sources ({agreement_sources[0] if agreement_sources else 'none'})."
                        )
                        verdict_for_candidate = "revised"
                else:
                    # 0개 합의 → 약한 증거로 강등
                    confidence = max(0.20, confidence - 0.15)
                    verification_notes.append(
                        f"Candidate '{cause_service or cause}' has no LLM source agreement."
                    )
                    verdict_for_candidate = "weak-evidence"
            else:
                # 결정론적 버전: 기존 키워드 매칭 로직
                if matched_log and matched_topology:
                    confidence = min(0.95, confidence + 0.08)
                    verification_notes.append(
                        f"Candidate '{cause}' is supported by both log evidence and topology path."
                    )
                    verdict_for_candidate = "accepted"
                elif matched_log and matched_rca:
                    confidence = min(0.92, confidence + 0.05)
                    verification_notes.append(
                        f"Candidate '{cause}' is supported by logs and RCA synthesis."
                    )
                    verdict_for_candidate = "revised"
                elif matched_log:
                    confidence = min(0.90, confidence + 0.03)
                    verification_notes.append(
                        f"Candidate '{cause}' is supported by logs only."
                    )
                    verdict_for_candidate = "revised"
                elif matched_topology:
                    confidence = max(0.40, confidence - 0.02)
                    verification_notes.append(
                        f"Candidate '{cause}' is structurally plausible but log support is limited."
                    )
                    verdict_for_candidate = "revised"
                else:
                    confidence = max(0.20, confidence - 0.15)
                    verification_notes.append(
                        f"Candidate '{cause}' lacks strong support from both logs and topology."
                    )
                    verdict_for_candidate = "weak-evidence"
            
            # =====================================================
            # Stage 5: evidence_convergence 전역 조정
            # =====================================================
            if evidence_convergence == "weak_evidence":
                # LLM 자체가 증거가 약하다고 판정 → confidence 추가 감쇠
                confidence = max(0.15, confidence - 0.10)
                verification_notes.append(
                    f"LLM reported 'weak_evidence' convergence; confidence further reduced."
                )
                if verdict_for_candidate == "accepted":
                    verdict_for_candidate = "revised"

            revised_candidates.append({
                **c,
                "confidence": round(confidence, 3),
                "_verdict": verdict_for_candidate,
                "_agreement_count": agreement_count,
            })

        revised_candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        for idx, c in enumerate(revised_candidates, start=1):
            c["rank"] = idx

        verdict = self._derive_final_verdict(revised_candidates, evidence_convergence)

        # 임시 필드 제거
        for c in revised_candidates:
            c.pop("_verdict", None)
            c.pop("_agreement_count", None)

        if not revised_candidates:
            if rejected_candidates:
                verdict = "rejected"
                verification_notes.append(
                    f"All {len(rejected_candidates)} candidates rejected due to hallucination."
                )
            else:
                verdict = "rejected"
                verification_notes.append("No candidate was provided for verification.")

        final_confidence = revised_candidates[0]["confidence"] if revised_candidates else 0.0
        explanation = (
            verification_notes[0]
            if verification_notes
            else "No verification note available."
        )

        return {
            "incident_id": incident_id,
            "service": service,
            "verdict": verdict,
            "verification_notes": verification_notes,
            "revised_root_cause_candidates": revised_candidates,
            "final_confidence": final_confidence,
            "explanation": explanation,
            "rejected_candidates_count": len(rejected_candidates),
        }

    def _derive_final_verdict(
        self,
        candidates: List[Dict[str, Any]],
        evidence_convergence: Optional[str],
    ) -> str:
        if not candidates:
            return "rejected"

        top = candidates[0]
        top_conf = float(top.get("confidence", 0.0))
        verdict_for_candidate = top.get("_verdict", "weak-evidence")
        agreement = top.get("_agreement_count", 0)
        
        # evidence_convergence가 weak_evidence면 강하게 제한
        if evidence_convergence == "weak_evidence":
            return "weak-evidence"
        
        # LLM supporting_evidence가 3개 모두 동의 + 높은 confidence → accepted
        if agreement == 3 and top_conf >= 0.75:
            return "accepted"
        
        # 전통적 accepted 기준
        if verdict_for_candidate == "accepted" and top_conf >= 0.80:
            return "accepted"
        
        if verdict_for_candidate in {"accepted", "revised"} and top_conf >= 0.55:
            return "revised"
        
        return "weak-evidence"

    # ======================================================
    # Fallback: 결정론적 버전 호환용 키워드 매칭
    # ======================================================
    
    def _is_supported_by_logs(self, cause_lower: str, log_summary: str) -> bool:
        checks = [
            ("auth-service", "auth-service"),
            ("catalog-service", "catalog-service"),
            ("order-service", "order-service"),
            ("worker-service", "worker"),
            ("message-queue", "queue"),
            ("timeout", "timeout"),
            ("latency", "latency"),
            ("database", "database"),
            ("db", "db"),
            ("pool", "pool"),
            ("connection", "connection"),
            ("stale", "stale"),
            ("retry", "retry"),
        ]
        for cause_token, log_token in checks:
            if cause_token in cause_lower and log_token in log_summary:
                return True
        return False

    def _is_supported_by_topology(
        self,
        cause_lower: str,
        topo_path: List[str],
        topo_summary: str,
    ) -> bool:
        joined_path = " ".join(str(p) for p in topo_path).lower()

        service_tokens = [
            "auth-service",
            "catalog-service",
            "order-service",
            "worker-service",
            "message-queue",
            "user-db",
            "order-db",
            "frontend-web",
            "api-gateway",
            "device-state-gateway",
            "app-deployer",
            "event-api",
        ]
        for token in service_tokens:
            if token in cause_lower and (token in joined_path or token in topo_summary):
                return True

        if "database" in cause_lower and ("db" in joined_path or "db" in topo_summary):
            return True
        if "queue" in cause_lower and ("queue" in joined_path or "queue" in topo_summary):
            return True

        return False

    def _is_supported_by_rca(self, cause_lower: str, rca_summary: str) -> bool:
        if not rca_summary:
            return False

        keywords = [
            "auth-service",
            "catalog-service",
            "order-service",
            "worker-service",
            "message-queue",
            "database",
            "db",
            "timeout",
            "latency",
            "connection",
            "queue",
        ]
        for keyword in keywords:
            if keyword in cause_lower and keyword in rca_summary:
                return True

        return False
