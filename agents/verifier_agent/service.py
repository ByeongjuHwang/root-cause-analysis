"""
Verifier Agent Service — LLM-aware cross-validation.

Option B 설계 (from thesis framework design):
  - LLM이 제공하는 최종 판정을 결정론적 규칙으로 재검증
  - LLM hallucination (존재하지 않는 서비스 등) 차단
  - 세 증거원 (Log / Topology / TCB-RCA)의 합의 여부 검증
  - 결정론적 baseline (cause_service 없음)과도 호환

검증 단계 (v8+):
  Stage 1: cause_service 환각 검증 (실제 토폴로지 노드에 존재하는가?)
  Stage 1b: 5-signal evidence extraction (per candidate):
            has_log_evidence, has_trace_support, has_metric_shift,
            has_topology_path, is_temporally_prior
  Stage 2: HARD drop rules (R1: topology path 없음 / R2: 증거 modality 0개)
  Stage 3: SOFT rules (modality count 기반 confidence cap, temporal penalty)
  Stage 4: supporting_evidence 합의 카운트 (legacy LLM 호환)
  Stage 5: evidence_convergence 기반 confidence 재보정
  Stage 6: 전통적 키워드 매칭 (fallback, 결정론적 버전 호환)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# v8: 5-signal evidence extraction
# =============================================================================

# Confidence caps based on modality support count (v8 R3):
MODALITY_CAP_SINGLE = 0.60  # Only one of log/trace/metric supports candidate
MODALITY_CAP_DOUBLE = 0.80  # Two modalities support
# Three modalities → no cap (LLM's value preserved)

TEMPORAL_VIOLATION_PENALTY = 0.15  # v8 R4: candidate is later than symptom

# Metric shift thresholds (uses log-derived statistics as metric proxy)
METRIC_SHIFT_VOLUME_DELTA = 0.3  # abs(volume_delta) > 0.3 counts as shift
METRIC_SHIFT_ERROR_RATIO = 0.05  # error_ratio > 0.05 counts


def _candidate_evidence_signals(
    cause_service: Optional[str],
    log_evidence: List[Dict[str, Any]],
    service_statistics: Dict[str, Any],
    topo_path: List[str],
    topo_dependencies: List[str],
    rca_metadata: Dict[str, Any],
) -> Dict[str, bool]:
    """Compute the 5 evidence signals for a candidate service.

    Args:
        cause_service: the candidate's cause_service name (may be None)
        log_evidence: list of log evidence dicts (from log_agent.evidence)
        service_statistics: dict from get_service_statistics() (v6 output,
            nested under service_statistics.services[svc])
        topo_path: propagation path from topology agent
        topo_dependencies: direct/transitive dependency list of symptom service
        rca_metadata: synthesis result metadata including scoring_rules output

    Returns:
        Dict with 5 boolean signals:
          has_log_evidence:   candidate has at least one log entry of its own
          has_trace_support:  candidate's logs contain at least one trace_id
          has_metric_shift:   candidate's log-stats show volume/error/keyword anomaly
          has_topology_path:  candidate appears in propagation path or is a dependency
          is_temporally_prior: candidate is not flagged as temporally later than symptom
    """
    result = {
        "has_log_evidence":    False,
        "has_trace_support":   False,
        "has_metric_shift":    False,
        "has_topology_path":   False,
        "is_temporally_prior": True,  # default assumption (benefit of doubt)
    }

    if not cause_service:
        # No structured service name — can only rely on keyword fallback
        return result

    cs_lower = str(cause_service).strip().lower()

    # --- 1. has_log_evidence: candidate owns at least one log line ---
    # --- 2. has_trace_support: any of those logs have trace_id ---
    for ev in log_evidence:
        if not isinstance(ev, dict):
            continue
        meta = ev.get("metadata") or {}
        svc = str(meta.get("service") or "").strip().lower()
        if svc == cs_lower:
            result["has_log_evidence"] = True
            if ev.get("trace_id"):
                result["has_trace_support"] = True
                # Can stop early since both are True
                break

    # --- 3. has_metric_shift: volume delta / error ratio / keyword hits ---
    # We use v6's service_statistics as the "metric" proxy (we don't have
    # real Prometheus metrics in this system — log-rate shifts play that role).
    svcs = (service_statistics or {}).get("services") or {}
    # Try case-insensitive lookup
    stats = None
    for k, v in svcs.items():
        if str(k).strip().lower() == cs_lower:
            stats = v
            break
    # --- 3a. has_metric_shift (PRIMARY — v8): real metric evidence ---
    # If the Log Agent emitted explicit modality=metric evidence for this
    # service (from Prometheus/Istio), that is the strongest signal. We check
    # this FIRST and only fall back to the log-rate proxy if there's no
    # dedicated metric entry.
    for ev in log_evidence:
        if not isinstance(ev, dict):
            continue
        if ev.get("modality") != "metric":
            continue
        meta = ev.get("metadata") or {}
        svc = str(meta.get("service") or "").strip().lower()
        if svc == cs_lower:
            result["has_metric_shift"] = True
            break

    # --- 3b. has_metric_shift (FALLBACK): log-rate proxy ---
    # Older flow or cases without metrics.csv — use v6 log statistics.
    # Only run if the real metric check above didn't already flip the flag.
    if not result["has_metric_shift"] and stats:
        vd = abs(float(stats.get("volume_delta", 0) or 0))
        er = float(stats.get("error_ratio", 0) or 0)
        kw_hits = (
            int(stats.get("timeout_hits", 0) or 0)
            + int(stats.get("retry_hits", 0) or 0)
            + int(stats.get("reset_hits", 0) or 0)
        )
        if vd > METRIC_SHIFT_VOLUME_DELTA or er > METRIC_SHIFT_ERROR_RATIO or kw_hits > 0:
            result["has_metric_shift"] = True

    # --- 4. has_topology_path: candidate in propagation path OR dependency ---
    path_lower = {str(p).strip().lower() for p in topo_path or []}
    deps_lower = {str(d).strip().lower() for d in topo_dependencies or []}
    if cs_lower in path_lower or cs_lower in deps_lower:
        result["has_topology_path"] = True

    # --- 5. is_temporally_prior: RCA Agent's v7 scoring_rules flag ---
    # The RCA agent annotates candidates with _temporal_violation when a
    # candidate's first anomaly is later than the symptom. If that flag is
    # True, the candidate CANNOT precede the symptom → not temporally_prior.
    temporal_violation = rca_metadata.get("_temporal_violation_by_service", {}).get(cs_lower)
    if temporal_violation is True:
        result["is_temporally_prior"] = False

    return result


def _compute_modality_count(signals: Dict[str, bool]) -> int:
    """Count supporting modalities (log / trace / metric). Max 3."""
    return (
        int(signals.get("has_log_evidence", False))
        + int(signals.get("has_trace_support", False))
        + int(signals.get("has_metric_shift", False))
    )


# =============================================================================
# Legacy helpers (below)
# =============================================================================


# 기본 서비스 목록 (fallback, Topology Agent 결과가 없을 때 사용)
# [TT-PATCH] Cleared. With multi-system support, hardcoded OB/case-study
# services would silently mismatch on TT or other datasets. Verifier now
# relies entirely on dynamic extraction from topology_agent results
# (related_services / blast_radius / propagation_path).
_DEFAULT_KNOWN_SERVICES: Set[str] = set()


def _extract_known_services(agent_results: Dict[str, Any]) -> Set[str]:
    """Extract known services from topology agent results dynamically.

    [TT-PATCH-2] Now includes the FULL topology (topology_graph keys +
    service_metadata keys) in addition to the incident-relevant subset
    (related_services, blast_radius, propagation_path). This is critical
    for orphan services (e.g., entry-point auth services with no edges)
    which would otherwise be rejected as hallucinations even though they
    are valid services in the system.

    Falls back to the default set if topology data is unavailable.
    """
    topo = agent_results.get("topology_agent", {})
    if not isinstance(topo, dict):
        return _DEFAULT_KNOWN_SERVICES.copy()

    services: Set[str] = set()

    # [TT-PATCH-2] Full topology — covers ALL services including orphans
    topology_graph = topo.get("topology_graph", {})
    if isinstance(topology_graph, dict):
        services.update(str(k) for k in topology_graph.keys() if k)
        # Also include callees (graph values) in case some services only
        # appear as targets, not sources
        for callees in topology_graph.values():
            if isinstance(callees, list):
                services.update(str(c) for c in callees if c)

    service_metadata = topo.get("service_metadata", {})
    if isinstance(service_metadata, dict):
        services.update(str(k) for k in service_metadata.keys() if k)

    # Incident-relevant subset (kept for completeness)
    for key in ("related_services", "blast_radius"):
        vals = topo.get(key, [])
        if isinstance(vals, list):
            services.update(str(v) for v in vals if v)

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

        # --- v8: sources for 5-signal evidence extraction ---
        log_evidence_list: List[Dict[str, Any]] = []
        for e in (log_result.get("evidence") or []):
            if isinstance(e, dict):
                log_evidence_list.append(e)

        # v6 Log Agent publishes service_statistics; fall back gracefully if absent.
        service_statistics = log_result.get("service_statistics") or {}
        if not isinstance(service_statistics, dict):
            service_statistics = {}

        # Dependencies of the symptom service (for has_topology_path when the
        # candidate isn't on the path yet but is a direct downstream).
        topo_deps: List[str] = []
        for key in ("related_services", "blast_radius"):
            vals = topo_result.get(key) or []
            if isinstance(vals, list):
                topo_deps.extend(str(v) for v in vals if v)

        # v7 scoring_rules emits _temporal_violation on each candidate; build a
        # service→bool lookup so we can reference it in the 5-signal extractor.
        temporal_violation_by_service: Dict[str, bool] = {}
        for _c in candidates:
            if not isinstance(_c, dict):
                continue
            _cs = str(_c.get("cause_service") or "").strip().lower()
            if _cs:
                temporal_violation_by_service[_cs] = bool(_c.get("_temporal_violation", False))
        rca_metadata_for_signals = {
            "_temporal_violation_by_service": temporal_violation_by_service,
        }

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
            # Stage 1b (v8): 5-signal evidence extraction + HARD drops
            # =====================================================
            signals = _candidate_evidence_signals(
                cause_service=cause_service,
                log_evidence=log_evidence_list,
                service_statistics=service_statistics,
                topo_path=topo_path,
                topo_dependencies=topo_deps,
                rca_metadata=rca_metadata_for_signals,
            )
            modality_count = _compute_modality_count(signals)

            # R1: candidate must be on a topology path (or a dependency).
            # Structurally impossible candidates are dropped, not just
            # demoted — this is the Verifier's core role per thesis 3장.
            # Exception: skip this check for candidates without cause_service
            # (deterministic baseline uses free-text cause strings).
            if cause_service and not signals["has_topology_path"]:
                verification_notes.append(
                    f"[DROPPED] '{cause_service}' is not on the topology "
                    f"propagation path nor a dependency of the symptom "
                    f"service '{service}'. Structurally impossible."
                )
                rejected_candidates.append({**c, "_verdict": "dropped-no-topology-path",
                                            "_signals": signals})
                continue

            # R2: zero evidence modality → drop.
            # If NONE of log/trace/metric supports this candidate, it is a
            # pure structural guess. The v5 adservice-at-0.68 failure falls
            # here: topology_agrees but zero real evidence.
            if cause_service and modality_count == 0:
                verification_notes.append(
                    f"[DROPPED] '{cause_service}' has no supporting evidence "
                    f"(log=0, trace=0, metric=0). Topology alone is insufficient."
                )
                rejected_candidates.append({**c, "_verdict": "dropped-no-evidence",
                                            "_signals": signals})
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
            # Stage 4b (v8): modality-count cap + temporal penalty
            # =====================================================
            # These apply to candidates that SURVIVED the hard drops in Stage 1b.
            # Guard: only run if we have signals (cause_service was present).
            if cause_service:
                # R3: single-modality candidates capped at 0.60; double at 0.80.
                if modality_count == 1:
                    before = confidence
                    confidence = min(confidence, MODALITY_CAP_SINGLE)
                    if confidence < before:
                        which = [k for k in ("has_log_evidence",
                                              "has_trace_support",
                                              "has_metric_shift")
                                 if signals.get(k)]
                        verification_notes.append(
                            f"Candidate '{cause_service}' supported by 1 modality "
                            f"({which[0] if which else 'unknown'}); "
                            f"confidence capped at {MODALITY_CAP_SINGLE}."
                        )
                elif modality_count == 2:
                    before = confidence
                    confidence = min(confidence, MODALITY_CAP_DOUBLE)
                    if confidence < before:
                        verification_notes.append(
                            f"Candidate '{cause_service}' supported by 2 modalities; "
                            f"confidence capped at {MODALITY_CAP_DOUBLE}."
                        )

                # R4: temporal causality penalty
                if not signals["is_temporally_prior"]:
                    confidence = max(0.10, confidence - TEMPORAL_VIOLATION_PENALTY)
                    verification_notes.append(
                        f"Candidate '{cause_service}' flagged as temporally posterior "
                        f"to the symptom; confidence penalised by "
                        f"{TEMPORAL_VIOLATION_PENALTY}."
                    )
                    if verdict_for_candidate == "accepted":
                        verdict_for_candidate = "revised"

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
                "_signals": signals if cause_service else None,
                "_modality_count": modality_count if cause_service else None,
            })

        revised_candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        for idx, c in enumerate(revised_candidates, start=1):
            c["rank"] = idx

        verdict = self._derive_final_verdict(revised_candidates, evidence_convergence)

        # 임시 필드 제거, diagnostic signals는 public 필드로 승격
        for c in revised_candidates:
            c.pop("_verdict", None)
            c.pop("_agreement_count", None)
            # v8: keep signals/modality_count as proper fields for downstream
            # analysis (diagnose_failures.py, paper tables). Rename with no
            # underscore prefix to signal they are intentional outputs.
            sig = c.pop("_signals", None)
            mc = c.pop("_modality_count", None)
            if sig is not None:
                c["verification_signals"] = sig
            if mc is not None:
                c["modality_count"] = mc

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

        result = {
            "incident_id": incident_id,
            "service": service,
            "verdict": verdict,
            "verification_notes": verification_notes,
            "revised_root_cause_candidates": revised_candidates,
            "final_confidence": final_confidence,
            "explanation": explanation,
            "rejected_candidates_count": len(rejected_candidates),
        }

        # >>> ADAPTIVE-FIX: attach AgentResponse contract when A2A_CONTRACT_MODE=on
        # Without this, Orchestrator._run_adaptive_iterations cannot read
        # completeness_score and the adaptive loop never executes.
        _maybe_attach_verifier_response(
            result=result,
            incident_id=incident_id,
            agent_results=agent_results,
        )
        # <<< ADAPTIVE-FIX

        return result

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

        # [TT-PATCH] Generic check: any service token from the actual topology
        # path or summary is considered. We also keep the legacy keyword
        # heuristics (database/queue) for cause descriptions that don't name
        # the service explicitly.
        topo_summary_lower = (topo_summary or "").lower()
        # Extract service-like tokens from topology path itself
        for token in topo_path:
            t = str(token).lower().strip()
            if t and t in cause_lower:
                return True

        if "database" in cause_lower and ("db" in joined_path or "db" in topo_summary_lower
                                          or "mongo" in joined_path or "mongo" in topo_summary_lower
                                          or "redis" in joined_path or "redis" in topo_summary_lower):
            return True
        if "queue" in cause_lower and ("queue" in joined_path or "queue" in topo_summary_lower
                                       or "mq" in joined_path or "mq" in topo_summary_lower):
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


# =============================================================================
# ADAPTIVE-FIX: AgentResponse contract attachment for Adaptive Re-invocation
# =============================================================================
# Added to make the Orchestrator's _run_adaptive_iterations() functional.
# The Orchestrator reads verifier's _agent_response.completeness_score to decide
# whether to re-invoke. Without this helper, completeness_score is always None
# and the adaptive loop stops at the first iteration with action="stop"
# (reason="no_contract_available"). This was confirmed in the original RCAEval
# results where adaptive_iterations was always None.

def _maybe_attach_verifier_response(
    result: Dict[str, Any],
    incident_id: str,
    agent_results: Dict[str, Any],
) -> None:
    """Attach AgentResponse to verifier's result dict when A2A_CONTRACT_MODE is on.

    Inherits evidence_collection from the upstream RCA AgentResponse if present,
    so the verifier contract has populated evidence references. ConsistencyChecks
    are intentionally left as None — the thesis adopts the 5-signal scheme, and
    the 4D consistency schema is retained for forward compatibility only.

    This function is defensive: if anything fails, we record the error on the
    result dict but never raise, so the verifier's primary output is unaffected.
    """
    if os.getenv("A2A_CONTRACT_MODE", "off") == "off":
        return

    try:
        from common.response_builder import (
            build_verifier_agent_response,
            attach_agent_response,
        )
        from common.a2a_contract import AgentResponse

        upstream_rca_response: Optional[AgentResponse] = None
        rca_result = agent_results.get("rca_agent") if isinstance(agent_results, dict) else None
        if isinstance(rca_result, dict):
            rca_contract = rca_result.get("_agent_response")
            if isinstance(rca_contract, dict):
                try:
                    upstream_rca_response = AgentResponse.model_validate(rca_contract)
                except Exception:
                    upstream_rca_response = None

        agent_resp = build_verifier_agent_response(
            legacy_result=result,
            request_id=incident_id or "UNKNOWN",
            upstream_rca_response=upstream_rca_response,
            consistency_by_service={},
        )
        attach_agent_response(result, agent_resp)
    except Exception as exc:
        result["_agent_response_error"] = str(exc)
