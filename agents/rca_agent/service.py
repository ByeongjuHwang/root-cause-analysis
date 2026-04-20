"""
RCA Agent Service — powered by the TCB-RCA algorithm.

This service receives log analysis results and topology analysis results
from the orchestrator (via A2A protocol), converts them into the format
expected by the TCB-RCA engine, executes the algorithm, and returns
structured root cause analysis results.

IMPORTANT (논문 artifact 요구사항):
    The TCB-RCA engine's topology_graph is built DYNAMICALLY from the
    Topology Agent's result (which in turn queries the Architecture MCP).
    This ensures that topology changes are reflected in the RCA without
    any hardcoded assumptions.

    Fallback chain:
        1. topology_result["topology_graph"]  (from Topology Agent)
        2. Architecture MCP repository        (direct load if topology_file given)
        3. Default hardcoded graph            (last resort, for backward compat)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .tcb_rca import (
    AnomalyEvidence,
    TCBRCAEngine,
    TCBRCAOutput,
    logs_to_anomaly_data,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default topology — ONLY used when Topology Agent result is unavailable.
# In normal operation, the topology comes from the Topology Agent / MCP.
# ---------------------------------------------------------------------------

_DEFAULT_TOPOLOGY_GRAPH: Dict[str, List[str]] = {
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

_DEFAULT_SERVICE_METADATA: Dict[str, Dict[str, Any]] = {
    "frontend-web": {"type": "frontend", "criticality": "high"},
    "api-gateway": {"type": "gateway", "criticality": "high"},
    "auth-service": {"type": "backend", "criticality": "high"},
    "catalog-service": {"type": "backend", "criticality": "medium"},
    "order-service": {"type": "backend", "criticality": "high"},
    "message-queue": {"type": "queue", "criticality": "high"},
    "worker-service": {"type": "worker", "criticality": "high"},
    "user-db": {"type": "database", "criticality": "high"},
    "order-db": {"type": "database", "criticality": "high"},
}


def _load_topology_from_mcp(
    topology_file: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """Load full topology from the Architecture MCP repository.

    Returns (topology_graph, service_metadata) or raises on failure.
    """
    from mcp_servers.architecture_mcp.app.repository import load_topology_data

    data = load_topology_data(topology_file=topology_file)

    # Build adjacency list from diagram edges
    edges = data.get("diagram", {}).get("content", {}).get("edges", [])
    graph: Dict[str, List[str]] = {}
    for src, dst in edges:
        graph.setdefault(src, []).append(dst)
        graph.setdefault(dst, [])  # ensure leaf nodes exist

    # Build metadata from services catalog
    services = data.get("services", {})
    metadata: Dict[str, Dict[str, Any]] = {}
    for svc_name, svc_info in services.items():
        metadata[svc_name] = {
            "type": svc_info.get("type", "unknown"),
            "criticality": svc_info.get("criticality", "medium"),
        }
        # Ensure the service exists in the graph
        graph.setdefault(svc_name, [])

    return graph, metadata


def _extract_topology_from_agent_result(
    topology_result: Dict[str, Any],
) -> Optional[Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]]:
    """Try to extract a full topology graph from the Topology Agent's result.

    The Topology Agent includes a ``topology_graph`` key in its response.
    If present, we use it directly so the RCA engine uses exactly the same
    topology that the Topology Agent analyzed.

    Returns (topology_graph, service_metadata) or None if not available.
    """
    topo_graph = topology_result.get("topology_graph")
    if not topo_graph or not isinstance(topo_graph, dict):
        return None

    svc_meta = topology_result.get("service_metadata")
    if not svc_meta or not isinstance(svc_meta, dict):
        svc_meta = {svc: {"type": "unknown", "criticality": "medium"} for svc in topo_graph}

    return topo_graph, svc_meta


class RCAService:
    """
    RCA Agent service that orchestrates the TCB-RCA algorithm execution.

    The topology graph is resolved dynamically per-request from:
        1. topology_result (from Topology Agent)
        2. Architecture MCP repository (fallback)
        3. Default constants (last resort)
    """

    async def synthesize(
        self,
        incident_id: str,
        service: str,
        log_result: Dict[str, Any],
        topology_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        log_result = log_result or {}
        topology_result = topology_result or {}

        if not isinstance(log_result, dict):
            raise TypeError(f"log_result must be dict, got {type(log_result).__name__}")
        if not isinstance(topology_result, dict):
            raise TypeError(f"topology_result must be dict, got {type(topology_result).__name__}")

        # --- Resolve topology dynamically ---
        topology_graph, service_metadata, topo_source = self._resolve_topology(
            topology_result
        )

        logger.info(
            "TCB-RCA using topology source=%s, nodes=%d",
            topo_source, len(topology_graph),
        )

        # --- Build engine per-request with resolved topology ---
        engine = TCBRCAEngine(
            topology_graph=topology_graph,
            service_metadata=service_metadata,
            delta_t_seconds=120,
            max_depth=10,
        )

        anomaly_data = self._extract_anomaly_data(log_result)
        alert_time = self._determine_alert_time(service, anomaly_data)

        result = engine.execute(
            incident_id=incident_id,
            symptom_service=service,
            alert_time=alert_time,
            anomaly_data=anomaly_data,
        )

        return self._format_output(result, service, topology_result, topo_source)

    def _resolve_topology(
        self,
        topology_result: Dict[str, Any],
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]], str]:
        """Resolve topology graph with fallback chain.

        Returns:
            (topology_graph, service_metadata, source_description)
        """
        # Strategy 1: Extract from Topology Agent result
        agent_topo = _extract_topology_from_agent_result(topology_result)
        if agent_topo is not None:
            graph, meta = agent_topo
            if graph:
                return graph, meta, "topology-agent"

        # Strategy 2: Load from Architecture MCP (respects topology_file env)
        topology_file = topology_result.get("topology_file")
        try:
            graph, meta = _load_topology_from_mcp(topology_file=topology_file)
            if graph:
                return graph, meta, "architecture-mcp"
        except Exception as exc:
            logger.warning("Failed to load topology from MCP: %s", exc)

        # Strategy 3: Default hardcoded (last resort)
        logger.warning("Using default hardcoded topology (fallback)")
        return (
            _DEFAULT_TOPOLOGY_GRAPH.copy(),
            _DEFAULT_SERVICE_METADATA.copy(),
            "default-hardcoded",
        )

    def _extract_anomaly_data(
        self, log_result: Dict[str, Any],
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

    def _format_output(
        self,
        result: TCBRCAOutput,
        symptom_service: str,
        topology_result: Dict[str, Any],
        topology_source: str,
    ) -> Dict[str, Any]:
        candidates = []
        for rc in result.root_cause_candidates:
            candidates.append({
                "rank": rc.rank,
                "cause": rc.cause_description,
                "cause_service": rc.cause_service,
                "confidence": rc.confidence,
                "evidence_refs": [
                    f"tcb-rca:backtrack-depth-{rc.depth}",
                    f"tcb-rca:temporal-gap-{rc.temporal_gap_seconds:.0f}s",
                ] + [
                    f"evidence:{step['service']}"
                    for step in rc.evidence_chain
                ],
            })

        related_services = topology_result.get("related_services", [])
        if not isinstance(related_services, list):
            related_services = []

        if result.root_cause_candidates:
            top = result.root_cause_candidates[0]
            summary = (
                f"TCB-RCA identified {top.cause_service} as the most likely root cause "
                f"(confidence: {top.confidence:.3f}, depth: {top.depth}, "
                f"temporal gap: {top.temporal_gap_seconds:.0f}s). "
                f"Traversal visited {result.traversal_summary['total_nodes_visited']} nodes, "
                f"{result.traversal_summary['nodes_with_anomalies']} had anomalies."
            )
        else:
            summary = "TCB-RCA found no root cause candidates."

        return {
            "incident_id": result.incident_id,
            "service": symptom_service,
            "algorithm": "TCB-RCA",
            "algorithm_version": "1.0",
            "topology_source": topology_source,
            "summary": summary,
            "confidence": (
                result.root_cause_candidates[0].confidence
                if result.root_cause_candidates
                else 0.0
            ),
            "root_cause_candidates": candidates,
            "affected_services": list(
                dict.fromkeys(result.propagation_path + [symptom_service])
            ),
            "related_services": related_services,
            "propagation_path": result.propagation_path,
            "blast_radius": result.blast_radius,
            "traversal_summary": result.traversal_summary,
            "evidence_chains": [
                {
                    "candidate_rank": rc.rank,
                    "candidate_service": rc.cause_service,
                    "backtrack_path": rc.backtrack_path,
                    "chain": rc.evidence_chain,
                }
                for rc in result.root_cause_candidates
            ],
            "evidence": [
                {
                    "type": "rca",
                    "source": "tcb-rca-engine",
                    "content": summary,
                    "metadata": {
                        "incident_id": result.incident_id,
                        "algorithm": "TCB-RCA",
                        "topology_source": topology_source,
                    },
                }
            ],
        }
