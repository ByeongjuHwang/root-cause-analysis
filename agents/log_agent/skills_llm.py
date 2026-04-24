"""
Log Analysis Agent — LLM-powered version (v3: topology-aware).

v2 대비 개선:
1. Topology 제약 주입: suspected_downstream은 실제 의존 서비스만 제안
2. Upstream 참조 추적: "auth-service가 upstream=user-db 에러" → user-db도 후보
3. 로그 없는 서비스도 "referenced root cause"로 식별 가능
"""

from typing import Any, Dict, List, Optional
import os

from mcp_servers.observability_mcp.app.repository import (
    load_logs,
    search_logs,
    get_error_summary,
    get_trace_logs,
    get_service_statistics,
)
from mcp_servers.observability_mcp.app.models import LogRecord
# v8: metric repository for CPU/memory/latency/retry summaries
try:
    from mcp_servers.observability_mcp.app.metric_repository import (
        get_all_service_metric_summaries,
    )
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False
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
    metric_summaries: Optional[Dict[str, Any]] = None,
    evidence_collection: Optional[Dict[str, Any]] = None,
) -> str:
    """Log Agent용 user prompt 생성.

    Args:
        service_statistics: get_service_statistics() 결과 (선택).
            제공되면 프롬프트에 volume delta / error ratio 섹션이 추가되어
            ERROR 로그 없는 resource fault(CPU/Memory/Loss/Delay)도 LLM이
            감지할 수 있게 된다. None이면 기존 동작(ERROR 중심) 그대로.
        metric_summaries: v8 — get_all_service_metric_summaries() 결과 (선택).
            제공되면 CPU z-score, memory jump, p95/p99 latency, network drops
            를 별도 섹션으로 출력. 이 modality가 있으면 CPU/MEM/DELAY/LOSS
            fault 감지가 크게 좋아진다 (log-rate보다 훨씬 강한 신호).
        evidence_collection: Phase 3b — get_evidence_collection() 결과 (선택).
            구조화된 EvidenceUnit 목록 (modality/anomaly_type/severity 등).
            제공되면 LLM에게 "evidence layer가 짚은 dominant 서비스" 정보를
            주어 multi-modality 정보를 한 번에 종합 판단할 수 있게 한다.
            기존 service_statistics + metric_summaries와 보완 관계
            (직교 정보가 아니라 같은 raw data의 다른 표현이므로 LLM이
            상호 검증하여 더 안정된 판단 가능).
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

    # === v8: Metric-based signals (Prometheus / Istio metrics) ===
    # This is an INDEPENDENT modality from log rates. Strong metric signals
    # (cpu_z > 20, mem_jump > 0.3, p95_delta > 100ms, rx_drop > 0) point at
    # the actual resource-affected service for CPU/MEM/DELAY/LOSS faults,
    # which are often invisible in logs alone.
    if metric_summaries and metric_summaries.get("services"):
        svc_metrics = metric_summaries["services"]

        lines.append("## Metric Summary (CPU / Memory / Latency / Network)")
        lines.append(
            "- Values are from the actual metrics time-series (Prometheus / Istio), "
            "a DIFFERENT signal source than the log rate table above."
        )
        lines.append(
            "- STRONG signals (any one of these is a candidate):"
        )
        lines.append(
            "  * cpu_spike_z > 20       → CPU fault (z-score of peak vs baseline)"
        )
        lines.append(
            "  * mem_jump > 0.30        → memory fault (ratio above baseline avg)"
        )
        lines.append(
            "  * p95_Δ_ms or p99_Δ_ms > 100  → delay fault (tail latency)"
        )
        lines.append(
            "  * rx_drop > 0 or tx_drop > 0  → loss fault (packet drops)"
        )
        lines.append(
            "  * sockets_max unusually high   → socket fault"
        )
        lines.append("")
        lines.append(
            "| service | cpu_z | cpu_max | mem_jump | p95_Δms | p99_Δms | err_Δ | rx_drop | sockets |"
        )
        lines.append(
            "|---------|-------|---------|----------|---------|---------|-------|---------|---------|"
        )

        # Rank by "worst" signal: max of cpu_z, scaled mem_jump, scaled p95_delta, drops.
        def _sig_score(entry):
            m = entry.get("metric") or {}
            l = entry.get("latency") or {}
            rt = entry.get("retry_timeout") or {}
            if not (m.get("has_data") or l.get("has_data") or rt.get("has_data")):
                return 0.0
            cpu_z = abs(m.get("cpu_spike_zscore", 0) or 0)
            mem_j = abs(m.get("mem_jump_ratio", 0) or 0) * 20  # bring into similar range
            p95_d = abs(l.get("p95_delta_ms", 0) or 0) / 5     # 100ms ≈ 20
            drop  = min((rt.get("rx_drop_delta", 0) or 0) + (rt.get("tx_drop_delta", 0) or 0), 1000) / 10
            return max(cpu_z, mem_j, p95_d, drop)

        ranked = sorted(
            [(svc, info) for svc, info in svc_metrics.items()],
            key=lambda kv: _sig_score(kv[1]),
            reverse=True,
        )

        # Only show services with actual data — hide GKE nodes / loadgen etc.
        shown = 0
        for svc, info in ranked:
            if shown >= 15:
                break
            m = info.get("metric") or {}
            l = info.get("latency") or {}
            rt = info.get("retry_timeout") or {}
            if not (m.get("has_data") or l.get("has_data") or rt.get("has_data")):
                continue

            cpu_z = m.get("cpu_spike_zscore")
            cpu_max = m.get("cpu_max")
            mem_j = m.get("mem_jump_ratio")
            p95_d = l.get("p95_delta_ms")
            p99_d = l.get("p99_delta_ms")
            err_d = rt.get("error_delta")
            rx_drop = rt.get("rx_drop_delta")
            sock_max = rt.get("sockets_max")

            def _fmt(v, fmt="{:.2f}", missing="-"):
                return fmt.format(v) if v is not None else missing

            lines.append(
                f"| {svc} | {_fmt(cpu_z, '{:+.1f}')} | {_fmt(cpu_max, '{:.2f}')} | "
                f"{_fmt(mem_j, '{:+.2f}')} | {_fmt(p95_d, '{:+.1f}')} | "
                f"{_fmt(p99_d, '{:+.1f}')} | {_fmt(err_d, '{:.0f}')} | "
                f"{_fmt(rx_drop, '{:.0f}')} | {_fmt(sock_max, '{:d}', '-')} |"
            )
            shown += 1

        lines.append("")
        lines.append(
            "IMPORTANT: If any service shows cpu_z > 30 or mem_jump > 0.3 or "
            "p95_Δms > 100 or rx_drop > 0, that service is a STRONG root-cause "
            "candidate regardless of what the log rate table says. Metric shifts "
            "are causally upstream of log-rate changes in most faults."
        )
        lines.append("")

    # === Phase 3b: Evidence-Aware Summary ===
    # 위의 service_statistics + metric_summaries 섹션은 raw 신호 (개별 카운트 / z-score)다.
    # 이 섹션은 같은 raw 데이터를 evidence_factory가 정규화한 결과다:
    #   - severity는 modality 간 비교 가능한 [0,1] 점수 (degradation only — directional)
    #   - anomaly_type은 fault 분류 힌트
    #   - dominant_services는 evidence-aware layer가 종합 판단한 상위 후보
    # LLM은 두 표현을 교차 검증하여 더 안정된 root cause를 결정해야 한다.
    # 두 표현이 일치하면 강한 신호, 충돌하면 evidence 한계(예: hub bias)를
    # legacy stats + topology constraint로 교정해야 한다.
    if evidence_collection and evidence_collection.get("units"):
        units = evidence_collection["units"]
        modalities_present = evidence_collection.get("modalities_present") or []

        # per-service severity aggregation (max severity per service)
        # plus per-modality count for evidence diversity scoring
        per_service: Dict[str, Dict[str, Any]] = {}
        for u in units:
            sev = float(u.get("severity") or 0.0)
            anom = u.get("anomaly_type") or "unknown"
            mod = u.get("modality") or "unknown"
            for svc in (u.get("services") or []):
                entry = per_service.setdefault(svc, {
                    "max_severity": 0.0,
                    "top_anomaly": None,
                    "modalities": set(),
                    "evidence_count": 0,
                })
                entry["evidence_count"] += 1
                entry["modalities"].add(mod)
                if sev > entry["max_severity"]:
                    entry["max_severity"] = sev
                    entry["top_anomaly"] = anom

        # Rank by (max_severity desc, evidence_count desc)
        ranked = sorted(
            per_service.items(),
            key=lambda kv: (-kv[1]["max_severity"], -kv[1]["evidence_count"]),
        )

        if ranked:
            lines.append(f"## Evidence-Aware Summary "
                         f"(modalities: {','.join(modalities_present)})")
            lines.append(
                "evidence_factory가 raw signal을 [0,1] severity로 정규화하고 "
                "modality 간 비교 가능한 형태로 통합한 결과. severity는 degradation만 "
                "반영 (개선 신호는 0). 위 service_statistics/metric_summaries와 "
                "교차 검증용으로 사용하라:"
            )
            for svc, info in ranked[:5]:
                mods_str = ",".join(sorted(info["modalities"]))
                lines.append(
                    f"- **{svc}**: max_severity={info['max_severity']:.2f} "
                    f"({info['top_anomaly']}), "
                    f"modalities={{{mods_str}}}, "
                    f"evidence_count={info['evidence_count']}"
                )
            lines.append("")
            lines.append(
                "주의: Evidence dominant 서비스가 legacy 분석과 일치하면 강한 신호이나, "
                "충돌하는 경우 다음을 고려하라:"
            )
            lines.append(
                "  (a) Hub service는 downstream 부하로 metric이 흔들릴 수 있음 — "
                "topology constraint를 우선시"
            )
            lines.append(
                "  (b) Symptom service는 측정 위치이므로 자연히 latency degradation을 "
                "보이며, 그것이 root cause라는 뜻은 아님"
            )
            lines.append(
                "  (c) 단일 modality(metric만)로 지목된 후보보다 multi-modality(log+metric) "
                "지지를 받는 후보가 우선"
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
        # v9: re-order samples so distress-keyword logs (timeout/retry/reset)
        # appear FIRST — they are the highest-precision distress signals even
        # when buried among dozens of regular errors. Within each priority
        # group we keep chronological order.
        def _has_distress_kw(ev):
            msg = (ev.get("content") or "").lower()
            return any(kw in msg for kw in (
                "timeout", "timed out", "deadline exceeded",
                "retry", "retries", "back-off", "backoff",
                "reset", "refused", "broken pipe", "econnrefused", "econnreset",
            ))

        prioritised = sorted(
            error_evidence_samples,
            key=lambda ev: (not _has_distress_kw(ev), ev.get("timestamp", "")),
        )

        for ev in prioritised[:30]:
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
            # v9: annotate distress lines so the LLM can see them at a glance
            distress_marker = " [DISTRESS]" if _has_distress_kw(ev) else ""
            lines.append(
                f"- {ts} | {svc} | {level}{upstream_str}{status_str}{error_type_str}{distress_marker}\n"
                f"  {msg}"
            )
    lines.append("")
    
    if known_topology_hint:
        lines.append(f"## Known Services in System")
        lines.append(f"Services: {', '.join(known_topology_hint)}")
        lines.append("")
    
    lines.append("## Analysis Task")
    lines.append(
        "Think step by step, USING METRIC SIGNALS FIRST:\n"
        "1. FIRST, examine the 'Metric Summary' table if present. This is the "
        "strongest signal. Services with cpu_z > 30, mem_jump > 0.3, p95_Δms > 100, "
        "or rx_drop > 0 are PRIME root-cause candidates. The service with the "
        "MOST EXTREME metric anomaly is usually the actual root cause for "
        "cpu/memory/delay/loss faults.\n"
        "2. If no metric table is provided (or has_data=False for all services), "
        "fall back to 'Service-level Log Statistics' — the log rate table. Focus on "
        "non-zero timeout/retry/reset keyword hits; these are high-precision signals.\n"
        "3. Identify anomalous services by COMBINING signals across both modalities. "
        "Priority: METRIC > KEYWORDS > VOLUME_DELTA > ERROR_RATIO. A service with "
        "strong metric anomaly beats a service with only volume_delta spike.\n"
        "4. CRITICAL CAVEAT: volume_delta can be misleading for low-volume services "
        "(e.g. a service with 5 logs in baseline and 5 in incident may show +2.20 Δvol "
        "just because the windows differ in length). If a service has total logs < 50, "
        "treat its volume_delta as low-confidence.\n"
        "5. Determine the earliest/most-affected service based on metric onset times "
        "(if visible) or log timestamps.\n"
        "6. Follow upstream references in error logs. If service A reports errors with "
        "upstream=B and B has elevated metrics, B is a deeper root cause.\n"
        "7. suspected_downstream should be the DEEPEST cause you can identify from "
        "evidence. Prefer services with direct metric evidence over topology guesses.\n"
        "8. State your hypothesis about what likely happened.\n"
        "9. Assess confidence (0.0 to 1.0). Confidence of 0 is ONLY appropriate when "
        "NO service shows any metric anomaly AND no log keyword hits AND no "
        "error_ratio shift for ANY service."
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
        metrics_file: Optional[str] = None,
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

            # === v9 (7순위): per-service evidence reduction at COLLECTION time ===
            # Previously we appended EVERY log of EVERY service to all_evidence
            # (often 10,000+ entries for large cases). This exploded memory and
            # slowed _select_evidence_for_return downstream.
            #
            # New strategy: we only keep the evidence that can plausibly carry
            # a signal:
            #   1) ALL error/warn logs (already identified above)
            #   2) ALL logs whose message contains timeout / retry / reset /
            #      latency keywords (from v6 patterns)
            #   3) ALL logs with upstream != None (upstream-reference chain)
            #   4) UP TO SAMPLE_CAP INFO/DEBUG samples per service (representative)
            # Everything else is discarded. This preserves the LLM-relevant
            # signal while bounding memory. service_statistics (computed
            # separately in Step 2.6) still sees EVERY log because it reads
            # directly from the cached load_logs() output.
            from mcp_servers.observability_mcp.app.repository import (
                _LATENCY_PATTERNS, _TIMEOUT_PATTERNS,
                _RETRY_PATTERNS, _RESET_PATTERNS,
            )

            SAMPLE_CAP = 30  # normal INFO/DEBUG samples per service (upper bound)
            normal_sampled = 0

            for log_record in logs:
                lvl = (log_record.level or "").upper()
                msg = log_record.message or ""
                is_error_warn = (
                    lvl in ("ERROR", "WARN")
                    or (log_record.status_code and log_record.status_code >= 500)
                )
                has_upstream = bool(log_record.upstream)
                has_keyword = (
                    _LATENCY_PATTERNS.search(msg) is not None
                    or _TIMEOUT_PATTERNS.search(msg) is not None
                    or _RETRY_PATTERNS.search(msg) is not None
                    or _RESET_PATTERNS.search(msg) is not None
                )

                if is_error_warn or has_upstream or has_keyword:
                    # Signal-bearing: always keep
                    all_evidence.append(self._log_to_evidence(log_record))
                elif normal_sampled < SAMPLE_CAP:
                    # Representative sample cap for normal traffic
                    all_evidence.append(self._log_to_evidence(log_record))
                    normal_sampled += 1
                # else: skip (noise)
            
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
                ev["modality"] = "trace"  # v8: reclassify trace-sourced evidence
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

        # === Step 2.7 (v8): Metric summaries from RCAEval metrics.csv ===
        # CPU/memory/latency/network-drops metrics are far stronger signals than
        # log-rate volume deltas for CPU, MEM, DELAY, and LOSS faults. We pull
        # all three summary types in one batch scan (one CSV parse, N services).
        # This runs AFTER log statistics so the LLM gets both modalities in the
        # prompt — log rate + real metrics — and can cross-validate.
        metric_summaries: Optional[Dict[str, Any]] = None
        if _HAS_METRICS and metrics_file:
            try:
                # Prefer the incident_range (narrow, post-injection) for metric
                # summaries because that's where the real delta manifests.
                # Fall back to outer [start, end] otherwise.
                if incident_range:
                    m_start, m_end = incident_range[0], incident_range[1]
                else:
                    m_start, m_end = start, end
                metric_summaries = get_all_service_metric_summaries(
                    start=m_start, end=m_end,
                    metrics_file=metrics_file,
                    baseline_range=baseline_range,
                )
            except Exception as exc:
                # Metrics are optional — do not block the pipeline on failure
                metric_summaries = None
                _ = exc

        # === Step 2.8 (Phase 3b): Evidence-Aware Collection ===
        # Build a structured EvidenceCollection from log + metric data using
        # evidence_factory. This is the same raw data already gathered above,
        # but normalised to severity [0,1] with directional semantics
        # (degradation only). Provides modality-comparable signals to the LLM.
        #
        # Feature flag: opt-in via env var EVIDENCE_AWARE_PROMPT=1.
        # Phase 3b finding: keep this off by default until evaluated on
        # multiple smoke runs and the full 90-case run.
        # Set EVIDENCE_AWARE_PROMPT=1 to enable evidence injection into prompt.
        evidence_collection: Optional[Dict[str, Any]] = None
        if os.getenv("EVIDENCE_AWARE_PROMPT") == "1":
            try:
                from mcp_servers.observability_mcp.app.evidence_tools import (
                    get_evidence_collection_payload,
                )
                # Use same window resolution as the rest of the pipeline.
                ev_baseline_start = baseline_range[0] if baseline_range else None
                ev_baseline_end = baseline_range[1] if baseline_range else None
                ev_incident_start = incident_range[0] if incident_range else None
                ev_incident_end = incident_range[1] if incident_range else None

                evidence_collection = get_evidence_collection_payload(
                    start=start, end=end,
                    log_file=log_file, metrics_file=metrics_file,
                    baseline_start=ev_baseline_start, baseline_end=ev_baseline_end,
                    incident_start=ev_incident_start, incident_end=ev_incident_end,
                    focus_services=None,
                    symptom_service=service,
                    topology_path=None,        # not yet plumbed into log agent
                    candidate_services=None,
                )
            except Exception as exc:
                # Evidence collection is supplementary — never block the pipeline
                evidence_collection = None
                _ = exc

        # v8: materialise metric evidence entries for services with a detectable
        # signal. Each entry has modality=metric so downstream agents (RCA /
        # Verifier) can count it as a supporting modality independent of logs.
        # Only emit entries when the signal is meaningful — otherwise we'd flood
        # the evidence list with noise for all 10+ services.
        if metric_summaries and metric_summaries.get("services"):
            for svc, info in metric_summaries["services"].items():
                m = info.get("metric") or {}
                l = info.get("latency") or {}
                rt = info.get("retry_timeout") or {}
                if not (m.get("has_data") or l.get("has_data") or rt.get("has_data")):
                    continue

                # Signal detection — at least one of these must be above threshold
                cpu_z = abs(m.get("cpu_spike_zscore") or 0)
                mem_j = abs(m.get("mem_jump_ratio") or 0)
                p95_d = abs(l.get("p95_delta_ms") or 0)
                p99_d = abs(l.get("p99_delta_ms") or 0)
                rx_drop = rt.get("rx_drop_delta") or 0
                tx_drop = rt.get("tx_drop_delta") or 0
                err_d = rt.get("error_delta") or 0

                # Thresholds aligned with prompt guidance (cpu_z > 20, mem_jump > 0.3,
                # p95_delta > 100ms, any packet drop, any istio error)
                has_signal = (
                    cpu_z > 20 or mem_j > 0.15 or p95_d > 50 or p99_d > 50
                    or rx_drop > 0 or tx_drop > 0 or err_d > 0
                )
                if not has_signal:
                    continue

                # Build a compact human-readable content string summarising the
                # dominant signal(s) for this service.
                parts = []
                if cpu_z > 20:
                    parts.append(f"CPU spike z={cpu_z:.1f} (max={m.get('cpu_max', 0):.2f})")
                if mem_j > 0.15:
                    parts.append(f"Memory jump {mem_j:+.2f}")
                if p95_d > 50:
                    parts.append(f"p95 latency Δ={p95_d:+.0f}ms")
                if p99_d > 50:
                    parts.append(f"p99 latency Δ={p99_d:+.0f}ms")
                if rx_drop > 0 or tx_drop > 0:
                    parts.append(f"packet drops rx={rx_drop:.0f} tx={tx_drop:.0f}")
                if err_d > 0:
                    parts.append(f"istio errors +{err_d:.0f}")
                content = "; ".join(parts)

                # Use incident_range start as the canonical "when" for the signal
                evidence_ts = incident_range[0] if incident_range else start

                all_evidence.append({
                    "type": "metric",
                    "modality": "metric",     # v8 discriminator
                    "source": "observability-mcp/metric_repository",
                    "timestamp": evidence_ts,
                    "content": content,
                    "level": "INFO",           # metric entries aren't log-level
                    "trace_id": None,
                    "metadata": {
                        "service": svc,
                        "cpu_spike_zscore": m.get("cpu_spike_zscore"),
                        "cpu_max": m.get("cpu_max"),
                        "mem_jump_ratio": m.get("mem_jump_ratio"),
                        "p95_delta_ms": l.get("p95_delta_ms"),
                        "p99_delta_ms": l.get("p99_delta_ms"),
                        "rx_drop_delta": rt.get("rx_drop_delta"),
                        "tx_drop_delta": rt.get("tx_drop_delta"),
                        "error_delta": rt.get("error_delta"),
                        "sockets_max": rt.get("sockets_max"),
                    },
                })

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
            metric_summaries=metric_summaries,
            evidence_collection=evidence_collection,
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

        # === Phase 3a: Shadow mode — evidence-aware 경로를 병렬 실행 ===
        # 전적으로 관측 목적. 실패해도 legacy 경로에 전혀 영향 없다.
        # 결과는 llm_logs/*_log_agent_shadow_*.json 에 저장된다.
        legacy_result = {
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
            "referenced_upstreams": referenced_upstreams if referenced_upstreams else {},
            # Phase 3b: structured evidence for downstream agents (RCA / Verifier).
            # None when EVIDENCE_AWARE_PROMPT is off; otherwise a serialised
            # EvidenceCollection payload (JSON-friendly dict from MCP tool).
            "evidence_collection": evidence_collection,
        }

        try:
            from agents.log_agent.shadow import run_shadow_evidence_collection
            run_shadow_evidence_collection(
                legacy_result=legacy_result,
                incident_id=incident_id,
                symptom_service=service,
                start=start, end=end,
                log_file=log_file, metrics_file=metrics_file,
                baseline_range=baseline_range,
                incident_range=incident_range,
            )
        except Exception:
            # Shadow는 절대로 production 경로에 영향 주지 않아야 한다.
            # 어떤 예외도 여기서 무시한다 (log 기록은 shadow module 내부에서 처리).
            pass

        # === Phase 4b: A2A Contract dual output ===
        # Attach a structured AgentResponse to legacy_result so downstream
        # agents (Phase 4c Verifier, Phase 4d Orchestrator) can consume it
        # without parsing ad-hoc fields. Legacy dict fields remain unchanged
        # — this is additive only.
        # Feature flag A2A_CONTRACT_MODE: when "off" (default) we skip the
        # attach entirely so the dict is byte-identical to pre-Phase-4b.
        if os.getenv("A2A_CONTRACT_MODE", "off") != "off":
            try:
                from common.response_builder import (
                    build_log_agent_response,
                    attach_agent_response,
                )
                agent_resp = build_log_agent_response(
                    legacy_result=legacy_result,
                    request_id=incident_id or "UNKNOWN",
                )
                attach_agent_response(legacy_result, agent_resp)
            except Exception:
                # Contract building must never break the Agent.
                pass

        return legacy_result
    
    def _log_to_evidence(self, record: LogRecord) -> Dict[str, Any]:
        # v8: modality discriminator — RCA Agent / Verifier use this to
        # count supporting modalities per candidate (has_log / trace / metric).
        return {
            "type": "log",
            "modality": "log",  # v8
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
            3. v9: Logs whose message contains timeout/retry/reset/latency
               keywords — always keep (these are high-precision distress signals
               even when level=INFO)
            4. Logs from anomalous_services / suspected_downstream — up to
               per_service_cap each
            5. Other services — up to per_service_cap / 2 each

        Total is hard-capped at max_total, chronologically sorted.
        """
        if not all_evidence:
            return []

        # v9: import regex patterns for keyword-based tier-1 promotion
        from mcp_servers.observability_mcp.app.repository import (
            _LATENCY_PATTERNS, _TIMEOUT_PATTERNS,
            _RETRY_PATTERNS, _RESET_PATTERNS,
        )

        priority_services = set(anomalous_services or [])
        if suspected_downstream:
            priority_services.add(suspected_downstream)

        def _level(ev: Dict[str, Any]) -> str:
            return (ev.get("level") or "").upper()

        def _svc(ev: Dict[str, Any]) -> str:
            return (ev.get("metadata") or {}).get("service") or ""

        def _has_upstream(ev: Dict[str, Any]) -> bool:
            return bool((ev.get("metadata") or {}).get("upstream"))

        def _has_distress_keyword(ev: Dict[str, Any]) -> bool:
            """v9: distress keyword in message promotes log to tier-1."""
            msg = ev.get("content") or ""
            if not msg:
                return False
            return (
                _TIMEOUT_PATTERNS.search(msg) is not None
                or _RETRY_PATTERNS.search(msg) is not None
                or _RESET_PATTERNS.search(msg) is not None
                or _LATENCY_PATTERNS.search(msg) is not None
            )

        # Tier 1: always keep (error/warn, upstream, or distress keywords)
        tier1 = [
            ev for ev in all_evidence
            if _level(ev) in ("ERROR", "WARN")
            or _has_upstream(ev)
            or _has_distress_keyword(ev)
        ]
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
            combined = tier1[:max_total]  # prioritize error/warn/keyword if over budget
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
