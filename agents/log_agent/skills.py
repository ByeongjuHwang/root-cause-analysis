
"""
Log Analysis Agent Skills.

This module queries observability data and returns structured log evidence
for the TCB-RCA algorithm. It supports an external JSONL log file so the
thesis demo can be executed against real incident logs instead of only the
bundled sample file.

[TT-PATCH] _KNOWN_SERVICES is now driven by the KNOWN_SERVICES_OVERRIDE env var
so the same code works for OB / TT / SS without hardcoded service lists.
"""

import os
from typing import Any, Dict, List, Optional

from mcp_servers.observability_mcp.app.repository import (
    load_logs,
    search_logs,
    get_error_summary,
    get_trace_logs,
)
from mcp_servers.observability_mcp.app.models import LogRecord


# [TT-PATCH] System-agnostic: services come from observed logs + optional env override.
# To use a hand-curated service list, set: KNOWN_SERVICES_OVERRIDE="svc1,svc2,svc3"
_KNOWN_SERVICES_ENV = os.getenv("KNOWN_SERVICES_OVERRIDE", "")
_KNOWN_SERVICES: List[str] = (
    [s.strip() for s in _KNOWN_SERVICES_ENV.split(",") if s.strip()]
    if _KNOWN_SERVICES_ENV
    else []
)


class LogAnalysisService:
    def _candidate_services(self, log_file: Optional[str]) -> List[str]:
        observed = sorted({record.service for record in load_logs(log_file=log_file)})
        # [TT-PATCH] observed first (primary), env override second (supplementary)
        return list(dict.fromkeys(observed + _KNOWN_SERVICES))

    async def analyze(
        self,
        service: str,
        start: str,
        end: str,
        trace_id: Optional[str],
        symptom: str,
        log_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        all_evidence: List[Dict[str, Any]] = []
        service_summaries: Dict[str, Dict[str, Any]] = {}

        for svc in self._candidate_services(log_file):
            logs = search_logs(service=svc, start=start, end=end, log_file=log_file)
            if not logs:
                continue

            error_logs = [
                r for r in logs
                if r.level.upper() in ("ERROR", "WARN") or (r.status_code and r.status_code >= 500)
            ]

            for log_record in logs:
                all_evidence.append(self._log_to_evidence(log_record))

            if error_logs:
                error_summary = get_error_summary(service=svc, start=start, end=end, log_file=log_file)
                service_summaries[svc] = {
                    "total_logs": len(logs),
                    "error_logs": len(error_logs),
                    "top_error_types": error_summary.top_error_types[:3],
                }

        if trace_id:
            trace_logs = get_trace_logs(trace_id=trace_id, log_file=log_file)
            existing_keys = {(e["timestamp"], e["metadata"].get("service")) for e in all_evidence}
            for log_record in trace_logs:
                ev = self._log_to_evidence(log_record)
                ev["metadata"]["from_trace"] = True
                key = (ev["timestamp"], ev["metadata"].get("service"))
                if key not in existing_keys:
                    all_evidence.append(ev)

        summary = self._build_summary(service, symptom, service_summaries)
        confidence = self._compute_confidence(service, service_summaries)
        suspected_downstream = self._detect_downstream(service, all_evidence)

        return {
            "summary": summary,
            "confidence": confidence,
            "evidence": sorted(all_evidence, key=lambda item: (item.get("timestamp") or "", item.get("source") or "")),
            "hypothesis": self._build_hypothesis(service, suspected_downstream, service_summaries),
            "suspected_downstream": suspected_downstream,
            "service_error_summary": service_summaries,
            "log_file": log_file,
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

    def _build_summary(self, service: str, symptom: str, service_summaries: Dict[str, Dict[str, Any]]) -> str:
        affected = [
            f"{svc} ({info['error_logs']} errors)"
            for svc, info in service_summaries.items()
            if info["error_logs"] > 0
        ]
        if affected:
            return (
                f"Incident at {service}: '{symptom}'. "
                f"Errors detected in {len(affected)} service(s): {', '.join(affected)}."
            )
        return f"Incident at {service}: '{symptom}'. No significant errors found in logs."

    def _compute_confidence(self, service: str, service_summaries: Dict[str, Dict[str, Any]]) -> float:
        if service not in service_summaries:
            return 0.3
        info = service_summaries[service]
        error_ratio = info["error_logs"] / max(info["total_logs"], 1)
        services_with_errors = sum(1 for s in service_summaries.values() if s["error_logs"] > 0)
        cascade_bonus = min(0.15, services_with_errors * 0.05)
        return min(0.95, 0.5 + error_ratio * 0.3 + cascade_bonus)

    def _detect_downstream(self, service: str, evidence: List[Dict[str, Any]]) -> Optional[str]:
        upstream_counts: Dict[str, int] = {}
        for ev in evidence:
            meta = ev.get("metadata", {})
            if meta.get("service") == service and meta.get("upstream"):
                upstream = meta["upstream"]
                upstream_counts[upstream] = upstream_counts.get(upstream, 0) + 1
        if upstream_counts:
            return max(upstream_counts, key=upstream_counts.get)
        return None

    def _build_hypothesis(
        self,
        service: str,
        suspected_downstream: Optional[str],
        service_summaries: Dict[str, Dict[str, Any]],
    ) -> str:
        if suspected_downstream and suspected_downstream in service_summaries:
            return (
                f"{suspected_downstream} anomalies likely propagated upstream "
                f"to {service}, causing the observed symptoms."
            )
        return f"Anomaly detected at {service}; further investigation needed."
