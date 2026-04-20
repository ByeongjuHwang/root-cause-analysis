"""
Log Analysis Agent — LLM-powered version (v3: topology-aware).

v2 대비 개선:
1. Topology 제약 주입: suspected_downstream은 실제 의존 서비스만 제안
2. Upstream 참조 추적: "auth-service가 upstream=user-db 에러" → user-db도 후보
3. 로그 없는 서비스도 "referenced root cause"로 식별 가능
"""

from typing import Any, Dict, List, Optional

from mcp_servers.observability_mcp.app.repository import (
    load_logs,
    search_logs,
    get_error_summary,
    get_trace_logs,
    get_service_statistics,
)
from mcp_servers.observability_mcp.app.models import LogRecord
from common.llm_client import get_default_client

# Architecture MCP에서 의존성 정보 가져오기 (topology 제약용)
try:
    from mcp_servers.architecture_mcp.app.repository import (
        get_service_dependencies,
    )
    _HAS_ARCH_MCP = True
except ImportError:
    _HAS_ARCH_MCP = False


_KNOWN_SERVICES = [
    "frontend-web",
    "api-gateway",
    "auth-service",
    "catalog-service",
    "order-service",
    "worker-service",
    "message-queue",
    "user-db",
    "order-db",
]


# =========================================================================
# Prompts
# =========================================================================

LOG_AGENT_SYSTEM_PROMPT = """You are a log analysis expert for a microservice system. 
Your role is to analyze log evidence from multiple services during an incident and 
identify anomaly patterns.

Your analysis should follow this reasoning approach:
1. Which services show clear signs of failure (errors, timeouts, resource issues)?
2. What is the chronological order of anomalies across services?
3. Is there a service whose errors likely CAUSE errors in another service 
   (i.e., a "downstream" dependency that failed first)?
4. CRITICAL: Pay close attention to the "upstream" field in log entries. 
   If service A logs errors with upstream=B, it means A depends on B 
   and B may be the deeper root cause — even if B has NO log entries of its own.
5. How confident are you in the anomaly pattern?

Respond ONLY with valid JSON. No markdown code blocks, no commentary."""


def _format_error_entry(e: Any) -> str:
    if isinstance(e, dict):
        et = e.get("error_type") or e.get("type") or e.get("name") or "UNKNOWN"
        cnt = e.get("count") or e.get("n") or e.get("frequency") or 0
        return f"{et}({cnt})"
    if isinstance(e, (tuple, list)) and len(e) >= 2:
        return f"{e[0]}({e[1]})"
    if isinstance(e, (tuple, list)) and len(e) == 1:
        return str(e[0])
    return str(e)


def _extract_referenced_upstreams(error_evidence: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """에러 로그에서 upstream 필드로 참조된 서비스 추출.
    
    Returns: {referenced_service: [referencing_service1, ...]}
    예: auth-service 로그에 upstream=user-db → {"user-db": ["auth-service"]}
    """
    refs: Dict[str, List[str]] = {}
    for ev in error_evidence:
        meta = ev.get("metadata", {}) or {}
        upstream = meta.get("upstream")
        svc = meta.get("service", "")
        if upstream and upstream != svc:
            if upstream not in refs:
                refs[upstream] = []
            if svc not in refs[upstream]:
                refs[upstream].append(svc)
    return refs


def build_log_agent_user_prompt(
    symptom_service: str,
    symptom: str,
    time_range_start: str,
    time_range_end: str,
    service_error_summary: Dict[str, Dict[str, Any]],
    error_evidence_samples: List[Dict[str, Any]],
    known_topology_hint: Optional[List[str]] = None,
    symptom_depends_on: Optional[List[str]] = None,
    referenced_upstreams: Optional[Dict[str, List[str]]] = None,
    service_statistics: Optional[Dict[str, Any]] = None,
) -> str:
    """Log Agent용 user prompt 생성.

    Args:
        service_statistics: get_service_statistics() 결과 (선택).
            제공되면 프롬프트에 volume delta / error ratio 섹션이 추가되어
            ERROR 로그 없는 resource fault(CPU/Memory/Loss/Delay)도 LLM이
            감지할 수 있게 된다. None이면 기존 동작(ERROR 중심) 그대로.
    """
    
    lines = []
    lines.append(f"## Incident Context")
    lines.append(f"- Symptom service: {symptom_service}")
    lines.append(f"- Reported symptom: {symptom}")
    lines.append(f"- Time range: {time_range_start} to {time_range_end}")
    lines.append("")
    
    # === 토폴로지 제약 (신규) ===
    if symptom_depends_on:
        lines.append(f"## Topology Constraint (from architecture)")
        lines.append(
            f"- {symptom_service} DIRECTLY depends on: {symptom_depends_on}"
        )
        lines.append(
            f"- suspected_downstream MUST be one of these services "
            f"or a transitive dependency of them."
        )
        lines.append("")
    
    # === Upstream 참조 분석 (신규 - S8 핵심) ===
    if referenced_upstreams:
        lines.append(f"## Upstream References Found in Logs")
        lines.append(
            "These services are referenced as 'upstream' in error logs, "
            "meaning they may be root causes even if they have NO log entries:"
        )
        for ref_svc, callers in referenced_upstreams.items():
            has_own_logs = ref_svc in service_error_summary
            status = "HAS logs" if has_own_logs else "NO logs (unobserved!)"
            lines.append(
                f"- **{ref_svc}** ({status}) — referenced by: {', '.join(callers)}"
            )
        lines.append("")
    
    # === Service-level statistics (volume/rate signals) ===
    # 이 섹션은 ERROR 로그가 없는 결함(CPU/Memory/Loss/Delay)에서 특히 중요하다.
    # volume_delta 급감 → 서비스 다운/느려짐, 급증 → 재시도 폭주.
    # 키워드 히트(latency/timeout/retry/reset)는 level-agnostic anomaly 신호.
    if service_statistics and service_statistics.get("services"):
        win = service_statistics.get("window", {}) or {}
        b_sec = int(win.get("baseline_seconds", 0))
        i_sec = int(win.get("incident_seconds", 0))
        mode = service_statistics.get("mode", "legacy")

        lines.append(f"## Service-level Log Statistics")
        if mode == "dual" and b_sec and i_sec:
            lines.append(
                f"- Baseline window: {win.get('baseline_start', '?')} → "
                f"{win.get('baseline_end', '?')} ({b_sec}s of normal period)"
            )
            lines.append(
                f"- Incident window: {win.get('incident_start', '?')} → "
                f"{win.get('incident_end', '?')} ({i_sec}s around fault injection)"
            )
            lines.append(
                f"- volume_delta is RATE-NORMALISED (logs/sec ratio), so the two"
                f" windows being different lengths is correctly handled."
            )
        else:
            lines.append(
                f"- Legacy 50/50 split mode (single window). "
                f"volume_delta compares raw counts."
            )
        lines.append(
            f"- Strong anomaly signals: abs(volume_delta) > 0.5, elevated "
            f"error_ratio, or non-zero timeout/retry/reset hits."
        )
        lines.append(
            f"- Services with 0 logs are still SUSPECT if they are transitive "
            f"dependencies of {symptom_service} (see Topology Constraint above)."
        )
        lines.append("")
        lines.append(
            f"| service | base | incid | Δvol | err% | lat | to | retry | reset | top_lvl |"
        )
        lines.append(
            f"|---------|------|-------|------|------|-----|----|-------|-------|---------|"
        )

        # 정렬: anomaly 신호 총합이 큰 서비스 우선
        def _score(stats):
            return (
                abs(stats.get("volume_delta", 0)) * 2
                + stats.get("error_ratio", 0) * 3
                + min(stats.get("timeout_hits", 0), 10) * 0.2
                + min(stats.get("retry_hits", 0), 10) * 0.15
                + min(stats.get("reset_hits", 0), 10) * 0.2
                + min(stats.get("latency_hits", 0), 10) * 0.1
            )
        svc_items = sorted(
            service_statistics["services"].items(),
            key=lambda kv: _score(kv[1]),
            reverse=True,
        )
        for svc, stats in svc_items[:25]:
            lvl_counts = stats.get("level_counts", {})
            top_lvl = max(lvl_counts.items(), key=lambda x: x[1])[0] if lvl_counts else "N/A"
            lines.append(
                f"| {svc} | {stats['baseline_count']} | {stats['incident_count']} | "
                f"{stats['volume_delta']:+.2f} | {stats['error_ratio']:.2f} | "
                f"{stats.get('latency_hits', 0)} | {stats.get('timeout_hits', 0)} | "
                f"{stats.get('retry_hits', 0)} | {stats.get('reset_hits', 0)} | "
                f"{top_lvl} |"
            )
        lines.append("")
        lines.append(
            "Column legend: base=baseline_count, incid=incident_count, "
            "Δvol=volume_delta (rate-normalised), err%=error_ratio, "
            "lat=latency-keyword hits, to=timeout hits, retry=retry hits, "
            "reset=reset/refused hits, top_lvl=most frequent log level."
        )
        lines.append("")

    lines.append(f"## Service-level Error Summary")
    if not service_error_summary:
        lines.append("- No services show error logs in this time range.")
    else:
        for svc, info in sorted(service_error_summary.items(), 
                                key=lambda x: -x[1].get("error_logs", 0)):
            top_errors = info.get("top_error_types", []) or []
            top_errors_str = ", ".join(_format_error_entry(e) for e in top_errors[:3]) if top_errors else "N/A"
            lines.append(
                f"- **{svc}**: {info.get('error_logs', 0)} errors out of {info.get('total_logs', 0)} logs. "
                f"Top error types: {top_errors_str}"
            )
    lines.append("")
    
    lines.append(f"## Sample Error/Warning Log Evidence (chronological)")
    if not error_evidence_samples:
        lines.append("- No error-level evidence available.")
    else:
        for ev in error_evidence_samples[:30]:
            meta = ev.get("metadata", {}) or {}
            svc = meta.get("service", "?")
            ts = ev.get("timestamp", "?")
            level = ev.get("level", "?")
            msg = (ev.get("content", "") or "")[:120]
            upstream = meta.get("upstream")
            upstream_str = f" [upstream={upstream}]" if upstream else ""
            status = meta.get("status_code")
            status_str = f" [status={status}]" if status else ""
            error_type = meta.get("error_type")
            error_type_str = f" [type={error_type}]" if error_type else ""
            lines.append(
                f"- {ts} | {svc} | {level}{upstream_str}{status_str}{error_type_str}\n"
                f"  {msg}"
            )
    lines.append("")
    
    if known_topology_hint:
        lines.append(f"## Known Services in System")
        lines.append(f"Services: {', '.join(known_topology_hint)}")
        lines.append("")
    
    lines.append("## Analysis Task")
    lines.append(
        "Think step by step:\n"
        "1. FIRST, examine the 'Service-level Log Statistics' table if present. "
        "Services with extreme volume_delta (e.g. -0.8 = 80% volume drop, likely "
        "service slowdown/crash; +0.5 = retry storm) are strong candidates. "
        "Also check keyword signals: non-zero timeout (to) or reset hits on a "
        "service are high-precision distress indicators even without ERROR logs.\n"
        "2. Identify anomalous services by COMBINING signals: a service with "
        "(a) large Δvol, (b) elevated err%, or (c) non-trivial to/retry/reset hits "
        "is a strong candidate. Don't require ERROR-level logs.\n"
        "3. Determine the earliest/most-affected service.\n"
        "4. CRITICAL: Look at 'upstream' fields in error logs. If service A "
        "reports errors with upstream=B, then B is likely a deeper root cause. "
        "Follow this chain to find the deepest cause.\n"
        "5. If a service is referenced as 'upstream' in errors but has NO logs "
        "of its own, it is likely an UNOBSERVED root cause — still report it "
        "as suspected_downstream.\n"
        "6. suspected_downstream should be the DEEPEST root cause you can identify, "
        "not just the immediate dependency of the symptom service. It MAY be a "
        "service that has no errors but shows abnormal volume_delta (e.g. CPU-bound "
        "service that became slow but still logs normally), or a service whose "
        "downstream shows many timeout/retry hits pointing back to it.\n"
        "7. State your hypothesis about what likely happened.\n"
        "8. Assess confidence (0.0 to 1.0) based on evidence quality. "
        "A confidence of 0 is ONLY appropriate when truly no service shows any "
        "statistical anomaly — volume_delta in [-0.1, +0.1] AND zero keyword "
        "hits AND error_ratio ≈ 0 for ALL services."
    )
    
    return "\n".join(lines)


LOG_AGENT_SCHEMA_HINT = """{
  "anomalous_services": ["<service_name>", ...],
  "earliest_anomalous_service": "<service_name or null>",
  "suspected_downstream": "<the DEEPEST root cause service, even if it has no logs. Follow upstream references to find it.>",
  "hypothesis": "<1-2 sentence natural language hypothesis about root cause direction>",
  "confidence": 0.XX,
  "reasoning": "<brief explanation of how you arrived at these conclusions>"
}"""


# =========================================================================
# Service
# =========================================================================

class LogAnalysisServiceLLM:
    """LLM 기반 Log Analysis Agent."""
    
    def __init__(self):
        self.llm = get_default_client()
    
    def _candidate_services(self, log_file: Optional[str]) -> List[str]:
        observed = sorted({record.service for record in load_logs(log_file=log_file)})
        return list(dict.fromkeys(_KNOWN_SERVICES + observed))
    
    def _get_symptom_dependencies(
        self, symptom_service: str, topology_file: Optional[str]
    ) -> Optional[List[str]]:
        """증상 서비스의 직접 의존성을 Architecture MCP에서 조회."""
        if not _HAS_ARCH_MCP:
            return None
        import os
        topo = topology_file or os.getenv("ARCHITECTURE_TOPOLOGY_FILE")
        try:
            deps = get_service_dependencies(symptom_service, topology_file=topo)
            return deps.get("depends_on", []) or []
        except Exception:
            return None
    
    async def analyze(
        self,
        service: str,
        start: str,
        end: str,
        trace_id: Optional[str],
        symptom: str,
        log_file: Optional[str] = None,
        incident_id: Optional[str] = None,
        topology_file: Optional[str] = None,
        baseline_range: Optional[tuple] = None,
        incident_range: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        # === Step 1: 로그 수집 (결정론적) ===
        all_evidence: List[Dict[str, Any]] = []
        service_summaries: Dict[str, Dict[str, Any]] = {}
        
        candidates = self._candidate_services(log_file)
        
        for svc in candidates:
            logs = search_logs(service=svc, start=start, end=end, log_file=log_file)
            if not logs:
                continue
            
            error_logs = [
                r for r in logs
                if r.level.upper() in ("ERROR", "WARN") or (r.status_code and r.status_code >= 500)
            ]
            
            for log_record in logs:
                all_evidence.append(self._log_to_evidence(log_record))
            
            # NOTE: v3 (pre-stats) only added to service_summaries when error_logs > 0.
            # That caused CPU/Memory/Loss faults — which produce no ERROR logs —
            # to yield an empty summary, driving the LLM to "no anomaly" conclusions.
            # We now include EVERY service with activity so the LLM can see
            # volume/rate changes via the statistics section below.
            error_summary = get_error_summary(
                service=svc, start=start, end=end, log_file=log_file
            )
            top_types = getattr(error_summary, 'top_error_types', []) or []
            service_summaries[svc] = {
                "total_logs": len(logs),
                "error_logs": len(error_logs),
                "top_error_types": list(top_types[:3]),
            }
        
        if trace_id:
            trace_logs = get_trace_logs(trace_id=trace_id, log_file=log_file)
            existing_keys = {
                (e["timestamp"], e["metadata"].get("service")) for e in all_evidence
            }
            for log_record in trace_logs:
                ev = self._log_to_evidence(log_record)
                ev["metadata"]["from_trace"] = True
                key = (ev["timestamp"], ev["metadata"].get("service"))
                if key not in existing_keys:
                    all_evidence.append(ev)
        
        # === Step 2: 에러/경고 증거 추출 ===
        error_evidence = sorted(
            [
                e for e in all_evidence
                if (e.get("level", "") or "").upper() in ("ERROR", "WARN")
            ],
            key=lambda e: e.get("timestamp", ""),
        )
        
        # === Step 2.5: Topology 제약 및 upstream 참조 분석 (신규) ===
        symptom_depends_on = self._get_symptom_dependencies(service, topology_file)
        referenced_upstreams = _extract_referenced_upstreams(error_evidence)

        # === Step 2.6: Service-level statistics (v4 / v6 dual-window) ===
        # ERROR 로그가 없는 결함(CPU/Memory/Loss/Delay)을 감지하기 위한 통계.
        # v6: baseline_range와 incident_range가 주어지면 이격된 두 창으로 분석,
        # 없으면 legacy single-window 모드로 fallback.
        try:
            if baseline_range and incident_range:
                service_statistics = get_service_statistics(
                    start=start, end=end, log_file=log_file,
                    baseline_range=baseline_range,
                    incident_range=incident_range,
                )
            else:
                service_statistics = get_service_statistics(
                    start=start, end=end, log_file=log_file,
                )
        except Exception as exc:
            # Defensive: stats는 optional이므로 실패해도 파이프라인을 막지 않음
            service_statistics = None
            _ = exc

        # === Step 3: LLM 호출 ===
        user_prompt = build_log_agent_user_prompt(
            symptom_service=service,
            symptom=symptom,
            time_range_start=start,
            time_range_end=end,
            service_error_summary=service_summaries,
            error_evidence_samples=error_evidence,
            known_topology_hint=candidates,
            symptom_depends_on=symptom_depends_on,
            referenced_upstreams=referenced_upstreams if referenced_upstreams else None,
            service_statistics=service_statistics,
        )
        
        llm_result = await self.llm.call_json(
            agent_name="log_agent",
            system_prompt=LOG_AGENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            incident_id=incident_id,
            schema_hint=LOG_AGENT_SCHEMA_HINT,
        )
        
        # === Step 4: 결과 통합 ===
        if "_error" in llm_result:
            return self._fallback_result(
                service, symptom, all_evidence, service_summaries, 
                log_file, llm_result.get("_error", "LLM error")
            )
        
        anomalous_services = llm_result.get("anomalous_services", []) or []
        suspected_downstream = llm_result.get("suspected_downstream")
        hypothesis = llm_result.get("hypothesis", "") or ""
        
        try:
            confidence = float(llm_result.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        
        reasoning = llm_result.get("reasoning", "") or ""
        confidence = max(0.0, min(0.95, confidence))
        
        summary = self._build_summary_from_llm(
            service, symptom, service_summaries, hypothesis
        )

        # === v4: evidence 반환 크기 제한 ===
        # 이전엔 all_evidence 전체(171K+ 가능)를 반환했다가 HTTP/직렬화 단계에서
        # 통째로 drop되는 현상이 있었음. 의미 있는 evidence만 남긴다:
        #   (1) ERROR/WARN level 전부
        #   (2) upstream 필드가 있는 로그 전부
        #   (3) suspected_downstream / anomalous_services 서비스 로그 상위 N개
        #   (4) incident window의 대표 샘플 (그 외 서비스별 최대 5개)
        # 총 상한 200개로 캡.
        evidence_for_return = self._select_evidence_for_return(
            all_evidence=all_evidence,
            anomalous_services=anomalous_services,
            suspected_downstream=suspected_downstream,
            max_total=200,
        )

        return {
            "summary": summary,
            "confidence": confidence,
            "evidence": evidence_for_return,
            "hypothesis": hypothesis,
            "suspected_downstream": suspected_downstream,
            "anomalous_services": anomalous_services,
            "service_error_summary": service_summaries,
            "service_statistics": service_statistics,  # v4: RCA Agent가 참고 가능
            "llm_reasoning": reasoning,
            "log_file": log_file,
            # 신규: upstream 참조 정보 (RCA Agent가 참고 가능)
            "referenced_upstreams": referenced_upstreams if referenced_upstreams else {},
        }
    
    def _log_to_evidence(self, record: LogRecord) -> Dict[str, Any]:
        return {
            "type": "log",
            "source": "observability-mcp",
            "timestamp": record.timestamp,
            "content": record.message,
            "level": record.level,
            "trace_id": record.trace_id,
            "metadata": {
                "service": record.service,
                "status_code": record.status_code,
                "latency_ms": record.latency_ms,
                "upstream": record.upstream,
                "error_type": record.error_type,
            },
        }

    def _select_evidence_for_return(
        self,
        all_evidence: List[Dict[str, Any]],
        anomalous_services: List[str],
        suspected_downstream: Optional[str],
        max_total: int = 200,
        per_service_cap: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return a bounded, representative subset of evidence.

        Priority order (higher priority kept first):
            1. ERROR / WARN level — always keep
            2. Logs with upstream field set — always keep
            3. Logs from anomalous_services / suspected_downstream — up to per_service_cap each
            4. Other services — up to per_service_cap each

        Total is hard-capped at max_total, chronologically sorted.
        """
        if not all_evidence:
            return []

        priority_services = set(anomalous_services or [])
        if suspected_downstream:
            priority_services.add(suspected_downstream)

        def _level(ev: Dict[str, Any]) -> str:
            return (ev.get("level") or "").upper()

        def _svc(ev: Dict[str, Any]) -> str:
            return (ev.get("metadata") or {}).get("service") or ""

        def _has_upstream(ev: Dict[str, Any]) -> bool:
            return bool((ev.get("metadata") or {}).get("upstream"))

        # Tier 1: always keep
        tier1 = [ev for ev in all_evidence if _level(ev) in ("ERROR", "WARN") or _has_upstream(ev)]
        kept_ids = {id(ev) for ev in tier1}

        # Tier 2: priority services (cap per service)
        tier2: List[Dict[str, Any]] = []
        per_svc_count: Dict[str, int] = {}
        for ev in all_evidence:
            if id(ev) in kept_ids:
                continue
            svc = _svc(ev)
            if svc in priority_services and per_svc_count.get(svc, 0) < per_service_cap:
                tier2.append(ev)
                per_svc_count[svc] = per_svc_count.get(svc, 0) + 1
                kept_ids.add(id(ev))

        # Tier 3: other services (lower cap)
        tier3: List[Dict[str, Any]] = []
        other_cap = max(1, per_service_cap // 2)
        other_count: Dict[str, int] = {}
        for ev in all_evidence:
            if id(ev) in kept_ids:
                continue
            svc = _svc(ev)
            if other_count.get(svc, 0) < other_cap:
                tier3.append(ev)
                other_count[svc] = other_count.get(svc, 0) + 1
                kept_ids.add(id(ev))

        combined = tier1 + tier2 + tier3
        # Hard cap
        if len(combined) > max_total:
            combined = tier1[:max_total]  # prioritize error/warn if over budget
            if len(combined) < max_total:
                remaining = max_total - len(combined)
                combined = combined + tier2[:remaining]

        return sorted(
            combined,
            key=lambda item: (item.get("timestamp") or "", item.get("source") or ""),
        )
    
    def _build_summary_from_llm(
        self, service, symptom, service_summaries, hypothesis,
    ) -> str:
        affected = [
            f"{svc} ({info.get('error_logs', 0)} errors)"
            for svc, info in service_summaries.items()
            if info.get("error_logs", 0) > 0
        ]
        affected_str = (
            f"Errors detected in {len(affected)} service(s): {', '.join(affected)}. "
            if affected else "No error logs found. "
        )
        hypothesis_str = f"LLM hypothesis: {hypothesis}" if hypothesis else ""
        return f"Incident at {service}: '{symptom}'. {affected_str}{hypothesis_str}"
    
    def _fallback_result(
        self, service, symptom, all_evidence, service_summaries, log_file, error_msg,
    ) -> Dict[str, Any]:
        # Apply the same evidence cap as the happy path
        evidence_bounded = self._select_evidence_for_return(
            all_evidence=all_evidence,
            anomalous_services=list(service_summaries.keys()),
            suspected_downstream=None,
            max_total=200,
        )
        return {
            "summary": f"LLM analysis failed: {error_msg}. Returning raw evidence.",
            "confidence": 0.3,
            "evidence": evidence_bounded,
            "hypothesis": "LLM unavailable; manual analysis required.",
            "suspected_downstream": None,
            "anomalous_services": list(service_summaries.keys()),
            "service_error_summary": service_summaries,
            "service_statistics": None,
            "llm_reasoning": f"ERROR: {error_msg}",
            "log_file": log_file,
            "referenced_upstreams": {},
            "_llm_error": error_msg,
        }
