"""
B1 Baseline: Monolithic Agent.

4개 전문 에이전트의 로직을 하나의 서비스에 통합한 모놀리식 구현.
- A2A 프로토콜 미사용 (직접 함수 호출)
- MCP 서버 미경유 (직접 파일 접근)
- HTTP 서버 1개만 기동

이 비교군은 "멀티 에이전트 분업이 제공하는 가치"를 분리 측정하기
위한 설계이다. 같은 알고리즘을 사용하되 아키텍처만 다르다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 직접 임포트 — MCP/A2A 없이 로직 모듈을 직접 사용
from mcp_servers.observability_mcp.app.repository import (
    load_logs,
    search_logs,
    get_error_summary,
    get_trace_logs,
)
from mcp_servers.observability_mcp.app.models import LogRecord
from mcp_servers.architecture_mcp.app.repository import (
    load_topology_data,
    get_service_dependencies,
    get_related_services,
    find_path,
    infer_blast_radius,
)
from agents.rca_agent.tcb_rca import (
    AnomalyEvidence,
    TCBRCAEngine,
    logs_to_anomaly_data,
)


class MonolithicRCAService:
    """모든 RCA 로직을 하나의 클래스에 통합한 모놀리식 서비스.

    에이전트 간 통신 없이, 로그 분석 → 토폴로지 분석 → RCA 추론 →
    교차 검증을 하나의 함수 안에서 순차적으로 수행한다.
    """

    def __init__(self, log_file: Optional[str] = None, topology_file: Optional[str] = None):
        self.log_file = log_file or os.getenv("OBSERVABILITY_LOG_FILE")
        self.topology_file = topology_file or os.getenv("ARCHITECTURE_TOPOLOGY_FILE")

    async def analyze_incident(self, incident: dict) -> dict:
        """하나의 함수 안에서 전체 RCA 파이프라인 실행."""

        service = incident["service"]
        start = incident["time_range"]["start"]
        end = incident["time_range"]["end"]
        symptom = incident.get("symptom", "")
        trace_id = incident.get("trace_id")
        incident_id = incident.get("incident_id", "unknown")

        # ============================================================
        # Phase 1: 로그 분석 (원래 Log Agent의 역할)
        # MCP 서버 없이 repository를 직접 호출
        # ============================================================
        all_evidence = []
        service_summaries = {}

        # 관측된 모든 서비스에 대해 로그 수집
        observed_services = sorted({r.service for r in load_logs(log_file=self.log_file)})

        for svc in observed_services:
            logs = search_logs(service=svc, start=start, end=end, log_file=self.log_file)
            if not logs:
                continue

            error_logs = [
                r for r in logs
                if r.level.upper() in ("ERROR", "WARN") or (r.status_code and r.status_code >= 500)
            ]

            for log_record in logs:
                all_evidence.append(self._log_to_evidence(log_record))

            if error_logs:
                error_summary = get_error_summary(service=svc, start=start, end=end, log_file=self.log_file)
                service_summaries[svc] = {
                    "total_logs": len(logs),
                    "error_logs": len(error_logs),
                    "top_error_types": error_summary.top_error_types[:3],
                }

        if trace_id:
            trace_logs = get_trace_logs(trace_id=trace_id, log_file=self.log_file)
            existing_keys = {(e["timestamp"], e["metadata"].get("service")) for e in all_evidence}
            for log_record in trace_logs:
                ev = self._log_to_evidence(log_record)
                key = (ev["timestamp"], ev["metadata"].get("service"))
                if key not in existing_keys:
                    all_evidence.append(ev)

        # 로그 기반 downstream 추정
        suspected_downstream = self._detect_downstream(service, all_evidence)

        log_confidence = self._compute_log_confidence(service, service_summaries)

        # ============================================================
        # Phase 2: 토폴로지 분석 (원래 Topology Agent의 역할)
        # MCP 서버 없이 repository를 직접 호출
        # ============================================================
        try:
            deps = get_service_dependencies(service, topology_file=self.topology_file)
            related = get_related_services(service, topology_file=self.topology_file)
            blast = infer_blast_radius(service, depth=2, topology_file=self.topology_file)
        except KeyError:
            deps = {"service": service, "depends_on": [], "upstream_of": [], "criticality": "unknown", "type": "unknown"}
            related = {"service": service, "related_services": []}
            blast = {"service": service, "blast_radius": [service]}

        propagation_path = []
        if suspected_downstream:
            path_result = find_path(service, suspected_downstream, topology_file=self.topology_file)
            propagation_path = path_result.get("path", [])

        # ============================================================
        # Phase 3: RCA 추론 (원래 RCA Agent의 역할)
        # 직접 TCB-RCA 엔진 호출
        # ============================================================
        topology_graph, service_metadata = self._load_topology_graph()

        engine = TCBRCAEngine(
            topology_graph=topology_graph,
            service_metadata=service_metadata,
            delta_t_seconds=120,
            max_depth=10,
        )

        # 로그 증거를 anomaly 데이터로 변환
        raw_records = []
        for ev in all_evidence:
            meta = ev.get("metadata", {})
            raw_records.append({
                "timestamp": ev.get("timestamp", ""),
                "service": meta.get("service", ""),
                "level": ev.get("level", "INFO"),
                "message": ev.get("content", ""),
                "error_type": meta.get("error_type"),
                "status_code": meta.get("status_code"),
                "latency_ms": meta.get("latency_ms"),
                "upstream": meta.get("upstream"),
                "trace_id": ev.get("trace_id"),
            })

        anomaly_data = logs_to_anomaly_data(raw_records)

        # alert time 결정
        service_anomalies = anomaly_data.get(service, [])
        if service_anomalies:
            alert_time = max(a.timestamp for a in service_anomalies)
        else:
            all_times = [a.timestamp for anoms in anomaly_data.values() for a in anoms]
            alert_time = max(all_times) if all_times else datetime.now()

        rca_output = engine.execute(
            incident_id=incident_id,
            symptom_service=service,
            alert_time=alert_time,
            anomaly_data=anomaly_data,
        )

        # ============================================================
        # Phase 4: 교차 검증 (원래 Verifier Agent의 역할)
        # 직접 로직 실행
        # ============================================================
        rca_candidates = []
        for rc in rca_output.root_cause_candidates:
            rca_candidates.append({
                "rank": rc.rank,
                "cause": rc.cause_description,
                "confidence": rc.confidence,
                "evidence_refs": [
                    f"tcb-rca:backtrack-depth-{rc.depth}",
                    f"tcb-rca:temporal-gap-{rc.temporal_gap_seconds:.0f}s",
                ] + [f"evidence:{step['service']}" for step in rc.evidence_chain],
            })

        # 간이 교차 검증
        verified_candidates = self._verify_candidates(
            rca_candidates, service, service_summaries, propagation_path
        )

        # ============================================================
        # 결과 조합
        # ============================================================
        top_cause = verified_candidates[0]["cause"] if verified_candidates else "No root cause identified."
        top_confidence = verified_candidates[0]["confidence"] if verified_candidates else 0.0

        affected = list(dict.fromkeys(rca_output.propagation_path + [service]))

        return {
            "incident_id": incident_id,
            "incident_summary": {
                "service": service,
                "symptom": symptom,
                "time_range": incident["time_range"],
            },
            "root_cause_candidates": verified_candidates,
            "final_verdict": {
                "cause": top_cause,
                "confidence": top_confidence,
                "explanation": f"B1 (Monolithic): All analysis performed in single process. "
                              f"Top cause confidence: {top_confidence:.2f}.",
            },
            "impact_analysis": {
                "affected_services": affected,
                "related_services": related.get("related_services", []),
                "propagation_path": rca_output.propagation_path,
                "blast_radius": rca_output.blast_radius,
            },
            "verification": {
                "verdict": self._derive_verdict(verified_candidates),
                "final_confidence": top_confidence,
                "notes": ["B1 baseline: monolithic single-process analysis."],
            },
            "evidence_summary": {
                "log_evidence": [e["content"] for e in all_evidence[:5] if e.get("level") in ("ERROR", "WARN")],
                "topology_evidence": [
                    f"Dependency: {service} -> {deps.get('depends_on', [])}",
                    f"Propagation: {' -> '.join(propagation_path)}" if propagation_path else "No propagation path",
                ],
            },
            "agent_results": [],  # 모놀리식이라 개별 에이전트 결과 없음
        }

    # ---- Helper Methods ----

    def _log_to_evidence(self, record: LogRecord) -> dict:
        return {
            "type": "log",
            "source": "direct-file-access",
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

    def _detect_downstream(self, service: str, evidence: list) -> Optional[str]:
        upstream_counts = {}
        for ev in evidence:
            meta = ev.get("metadata", {})
            if meta.get("service") == service and meta.get("upstream"):
                up = meta["upstream"]
                upstream_counts[up] = upstream_counts.get(up, 0) + 1
        if upstream_counts:
            return max(upstream_counts, key=upstream_counts.get)
        return None

    def _compute_log_confidence(self, service: str, summaries: dict) -> float:
        if service not in summaries:
            return 0.3
        info = summaries[service]
        ratio = info["error_logs"] / max(info["total_logs"], 1)
        cascade = min(0.15, sum(1 for s in summaries.values() if s["error_logs"] > 0) * 0.05)
        return min(0.95, 0.5 + ratio * 0.3 + cascade)

    def _load_topology_graph(self):
        """토폴로지 파일에서 그래프와 메타데이터 로드."""
        try:
            data = load_topology_data(topology_file=self.topology_file)
        except Exception:
            # 기본 토폴로지
            return self._default_topology()

        services_data = data.get("services", {})
        graph = {}
        metadata = {}

        for svc_name, svc_info in services_data.items():
            graph[svc_name] = svc_info.get("depends_on", [])
            metadata[svc_name] = {
                "type": svc_info.get("type", "unknown"),
                "criticality": svc_info.get("criticality", "medium"),
            }

        return graph, metadata

    def _default_topology(self):
        graph = {
            "frontend-web": ["api-gateway"],
            "api-gateway": ["auth-service", "catalog-service", "order-service"],
            "auth-service": ["user-db"],
            "catalog-service": [],
            "order-service": ["message-queue", "order-db"],
            "message-queue": ["worker-service"],
            "worker-service": ["order-db"],
            "user-db": [],
            "order-db": [],
        }
        metadata = {svc: {"type": "unknown", "criticality": "medium"} for svc in graph}
        return graph, metadata

    def _verify_candidates(self, candidates: list, service: str,
                           summaries: dict, propagation_path: list) -> list:
        """간이 교차 검증. Verifier Agent의 로직을 축약."""
        verified = []
        log_summary = " ".join(summaries.keys()).lower()
        path_str = " ".join(propagation_path).lower()

        for c in candidates:
            cause_lower = c["cause"].lower()
            confidence = c["confidence"]

            matched_log = any(svc in cause_lower and svc in log_summary
                            for svc in summaries.keys())
            matched_topo = any(svc in cause_lower and svc in path_str
                             for svc in propagation_path) if propagation_path else False

            if matched_log and matched_topo:
                confidence = min(0.95, confidence + 0.08)
            elif matched_log:
                confidence = min(0.90, confidence + 0.03)
            elif matched_topo:
                confidence = max(0.40, confidence - 0.02)
            else:
                confidence = max(0.20, confidence - 0.15)

            verified.append({
                **c,
                "confidence": round(confidence, 3),
            })

        verified.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        for idx, c in enumerate(verified, 1):
            c["rank"] = idx

        return verified

    def _derive_verdict(self, candidates: list) -> str:
        if not candidates:
            return "rejected"
        top_conf = candidates[0].get("confidence", 0)
        if top_conf >= 0.80:
            return "accepted"
        if top_conf >= 0.55:
            return "revised"
        return "weak-evidence"
