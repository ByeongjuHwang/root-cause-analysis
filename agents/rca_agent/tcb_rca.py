"""
TCB-RCA: Topology-Constrained Temporal Backtracking Root Cause Analysis

This module implements the novel TCB-RCA algorithm proposed in the thesis.
Unlike existing RCA methods that learn causal graphs from data and apply
probabilistic random walks, TCB-RCA uses the *known* service topology as
a hard constraint and performs deterministic backward-in-time traversal
to locate the root cause of a fault.

Algorithm Overview:
    1. Start at the symptom service (s_alert) at time t_alert
    2. Query anomaly evidence for s_alert in [t_alert - Δt, t_alert]
    3. Identify downstream dependencies via the topology graph (constraint)
    4. For each dependency, check if anomalies occurred *before* the current
       service's earliest anomaly (temporal backtracking)
    5. Recursively traverse deeper until no earlier anomaly is found
    6. The deepest node with the earliest anomaly is the root cause candidate
    7. Score candidates using temporal precedence, topology distance, and
       anomaly severity

Key Differentiators from Existing Work:
    - Deterministic traversal (vs. stochastic random walk in MicroCause)
    - Known topology as hard constraint (vs. learned causal graph)
    - Backward temporal ordering enforced at each step
    - Integrated with MCP/A2A multi-agent architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnomalyEvidence:
    """A single anomaly observation at a service node."""
    service: str
    timestamp: datetime
    level: str
    message: str
    error_type: Optional[str] = None
    status_code: Optional[int] = None
    latency_ms: Optional[int] = None
    upstream: Optional[str] = None
    trace_id: Optional[str] = None

    @property
    def severity_score(self) -> float:
        """
        Compute a severity score in [0, 1] based on observable signals.

        Scoring rules (논문 Section 4.2에 해당):
            - ERROR level       → base 0.7
            - WARN level        → base 0.4
            - status_code ≥ 500 → +0.15
            - latency ≥ 3000ms  → +0.10
            - CIRCUIT_OPEN      → +0.05  (cascading failure indicator)
        """
        score = 0.0
        if self.level == "ERROR":
            score = 0.7
        elif self.level == "WARN":
            score = 0.4
        else:
            return 0.0

        if self.status_code and self.status_code >= 500:
            score += 0.15
        if self.latency_ms and self.latency_ms >= 3000:
            score += 0.10
        if self.error_type and self.error_type in (
            "CIRCUIT_OPEN", "CONNECTION_REFUSED", "CONNECTION_LIMIT",
        ):
            score += 0.05

        return min(score, 1.0)


@dataclass
class BacktrackNode:
    """
    A node in the backtracking traversal tree.

    Each node represents a service visited during the temporal backtracking,
    along with the anomalies found there and a reference to its parent node
    (the service that led us here via the topology graph).
    """
    service: str
    depth: int
    earliest_anomaly_time: Optional[datetime] = None
    anomalies: List[AnomalyEvidence] = field(default_factory=list)
    severity_score: float = 0.0
    parent_service: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class RootCauseResult:
    """The output of the TCB-RCA algorithm for a single root cause candidate."""
    rank: int
    cause_service: str
    cause_description: str
    confidence: float
    depth: int  # topology distance from symptom
    temporal_gap_seconds: float  # time between root cause and symptom
    evidence_chain: List[Dict[str, Any]]  # ordered list of traversal steps
    backtrack_path: List[str]  # service path from symptom to root cause


@dataclass
class TCBRCAOutput:
    """Complete output of the TCB-RCA algorithm."""
    incident_id: str
    symptom_service: str
    root_cause_candidates: List[RootCauseResult]
    traversal_summary: Dict[str, Any]
    propagation_path: List[str]  # inferred fault propagation (root → symptom)
    blast_radius: List[str]


# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------

class TCBRCAEngine:
    """
    Topology-Constrained Temporal Backtracking RCA Engine.

    This class implements the core algorithm. It requires:
        - A topology graph (adjacency list of service dependencies)
        - An anomaly data accessor (function that returns anomalies for a
          given service and time window)
    """

    def __init__(
        self,
        topology_graph: Dict[str, List[str]],
        service_metadata: Dict[str, Dict[str, Any]],
        delta_t_seconds: int = 120,
        max_depth: int = 10,
    ):
        """
        Args:
            topology_graph: Directed edges {service: [dependencies]}.
                            e.g. {"api-gateway": ["auth-service", "order-service"]}
            service_metadata: Service info {service: {type, criticality, ...}}
            delta_t_seconds: Time window (Δt) for anomaly lookup at each step
            max_depth: Maximum backtracking depth to prevent infinite loops
        """
        self.topology = topology_graph
        self.metadata = service_metadata
        self.delta_t = timedelta(seconds=delta_t_seconds)
        self.max_depth = max_depth

        # Build reverse graph for blast radius computation
        self._reverse_graph: Dict[str, List[str]] = {}
        for src, deps in self.topology.items():
            for dep in deps:
                self._reverse_graph.setdefault(dep, []).append(src)

    def execute(
        self,
        incident_id: str,
        symptom_service: str,
        alert_time: datetime,
        anomaly_data: Dict[str, List[AnomalyEvidence]],
    ) -> TCBRCAOutput:
        """
        Execute the TCB-RCA algorithm.

        Args:
            incident_id: Unique incident identifier
            symptom_service: The service where the symptom was observed
            alert_time: The time the alert was triggered
            anomaly_data: Pre-fetched anomaly data keyed by service name.
                          {service_name: [AnomalyEvidence, ...]}

        Returns:
            TCBRCAOutput with ranked root cause candidates and evidence chains

        논문 Algorithm 1 (TCB-RCA Main Procedure):
            Input:  G(V,E), s_alert, t_alert, Δt, anomaly_data
            Output: Ranked list of root cause candidates with evidence chains
        """
        logger.info(
            "TCB-RCA execute: incident=%s symptom=%s alert_time=%s",
            incident_id, symptom_service, alert_time,
        )

        # Phase 1: Temporal Backtracking Traversal
        visited: Dict[str, BacktrackNode] = {}
        self._backtrack(
            service=symptom_service,
            current_time=alert_time,
            depth=0,
            parent=None,
            anomaly_data=anomaly_data,
            visited=visited,
        )

        # Phase 2: Identify Leaf Nodes (Root Cause Candidates)
        candidates = self._identify_root_causes(visited, symptom_service, alert_time)

        # Phase 3: Score and Rank Candidates
        ranked = self._score_and_rank(candidates, visited, symptom_service, alert_time)

        # Phase 4: Infer Propagation Path
        if ranked:
            top_cause = ranked[0].cause_service
            propagation = self._trace_propagation_path(top_cause, symptom_service, visited)
        else:
            propagation = [symptom_service]

        # Phase 5: Compute Blast Radius
        blast = self._compute_blast_radius(
            ranked[0].cause_service if ranked else symptom_service,
            depth=3,
        )

        return TCBRCAOutput(
            incident_id=incident_id,
            symptom_service=symptom_service,
            root_cause_candidates=ranked,
            traversal_summary={
                "total_nodes_visited": len(visited),
                "max_depth_reached": max((n.depth for n in visited.values()), default=0),
                "nodes_with_anomalies": sum(
                    1 for n in visited.values() if n.anomalies
                ),
                "nodes_without_anomalies": sum(
                    1 for n in visited.values() if not n.anomalies
                ),
            },
            propagation_path=propagation,
            blast_radius=blast,
        )

    # ------------------------------------------------------------------
    # Phase 1: Recursive Temporal Backtracking
    # ------------------------------------------------------------------

    def _backtrack(
        self,
        service: str,
        current_time: datetime,
        depth: int,
        parent: Optional[str],
        anomaly_data: Dict[str, List[AnomalyEvidence]],
        visited: Dict[str, BacktrackNode],
    ) -> None:
        """
        Recursively traverse the topology graph backward in time.

        논문 Algorithm 2 (Temporal Backtracking Step):
            1. If service already visited or depth > max_depth: return
            2. Query anomalies for service in [current_time - Δt, current_time]
            3. If anomalies found:
               a. Record the node with its earliest anomaly timestamp
               b. For each downstream dependency in topology:
                  - Recursively backtrack with t = earliest_anomaly_time
            4. If no anomalies found: record as a clean node (boundary)
        """
        if service in visited or depth > self.max_depth:
            return

        # Query anomalies within the time window
        window_start = current_time - self.delta_t
        window_end = current_time

        service_anomalies = self._query_anomalies_in_window(
            service, window_start, window_end, anomaly_data,
        )

        # Create the backtrack node
        node = BacktrackNode(
            service=service,
            depth=depth,
            parent_service=parent,
        )

        if service_anomalies:
            node.anomalies = service_anomalies
            node.earliest_anomaly_time = min(a.timestamp for a in service_anomalies)
            node.severity_score = max(a.severity_score for a in service_anomalies)

        visited[service] = node

        # If anomalies found, backtrack deeper through dependencies
        if service_anomalies and depth < self.max_depth:
            dependencies = self.topology.get(service, [])

            for dep in dependencies:
                if dep not in visited:
                    # KEY INSIGHT: Use the earliest anomaly time of the current
                    # service as the new "current_time" for the dependency.
                    # This enforces temporal ordering: the dependency's anomaly
                    # must have occurred BEFORE the current service's anomaly.
                    backtrack_time = node.earliest_anomaly_time

                    node.children.append(dep)
                    self._backtrack(
                        service=dep,
                        current_time=backtrack_time,
                        depth=depth + 1,
                        parent=service,
                        anomaly_data=anomaly_data,
                        visited=visited,
                    )

    def _query_anomalies_in_window(
        self,
        service: str,
        window_start: datetime,
        window_end: datetime,
        anomaly_data: Dict[str, List[AnomalyEvidence]],
    ) -> List[AnomalyEvidence]:
        """
        Retrieve anomaly evidence for a service within a time window.

        Only returns entries with severity_score > 0 (i.e., WARN or ERROR level).
        """
        all_anomalies = anomaly_data.get(service, [])
        filtered = []

        for a in all_anomalies:
            if a.severity_score <= 0:
                continue
            if window_start <= a.timestamp <= window_end:
                filtered.append(a)

        # Sort by timestamp (earliest first)
        filtered.sort(key=lambda x: x.timestamp)

        logger.debug(
            "Anomalies for %s in [%s, %s]: %d found",
            service, window_start, window_end, len(filtered),
        )
        return filtered

    # ------------------------------------------------------------------
    # Phase 2: Identify Root Cause Candidates
    # ------------------------------------------------------------------

    def _identify_root_causes(
        self,
        visited: Dict[str, BacktrackNode],
        symptom_service: str,
        alert_time: datetime,
    ) -> List[BacktrackNode]:
        """
        Identify root cause candidates from the traversal tree.

        논문 Definition 1 (Root Cause Candidate):
            A node N is a root cause candidate if:
            (a) N has anomalies (severity_score > 0), AND
            (b) None of N's topology-constrained children have anomalies
                that temporally precede N's anomalies (temporal leaf), OR
            (c) N is at the maximum depth of traversal

        In other words: the "deepest" anomalous node in the backtracking
        tree where the fault trail ends.
        """
        candidates = []

        for service, node in visited.items():
            if not node.anomalies:
                continue

            # Check if any child has earlier anomalies
            has_earlier_child = False
            for child_name in node.children:
                child_node = visited.get(child_name)
                if (
                    child_node
                    and child_node.anomalies
                    and child_node.earliest_anomaly_time
                    and node.earliest_anomaly_time
                    and child_node.earliest_anomaly_time < node.earliest_anomaly_time
                ):
                    has_earlier_child = True
                    break

            # Root cause candidate: anomalous node with no earlier-anomalous children
            if not has_earlier_child:
                candidates.append(node)

        # If no candidates found (shouldn't happen if symptom service has anomalies),
        # fall back to the symptom service
        if not candidates and symptom_service in visited:
            candidates.append(visited[symptom_service])

        return candidates

    # ------------------------------------------------------------------
    # Phase 3: Score and Rank
    # ------------------------------------------------------------------

    def _score_and_rank(
        self,
        candidates: List[BacktrackNode],
        visited: Dict[str, BacktrackNode],
        symptom_service: str,
        alert_time: datetime,
    ) -> List[RootCauseResult]:
        """
        Score each root cause candidate and produce a ranked list.

        논문 Equation 1 (TCB-RCA Confidence Score):
            Score(c) = α · S_severity(c) + β · S_temporal(c) + γ · S_topology(c)

        Where:
            S_severity = normalized max anomaly severity at the candidate node
            S_temporal = temporal precedence score (earlier = higher)
            S_topology = topology depth bonus (deeper in dependency chain = higher)
            α, β, γ = weighting coefficients (α=0.4, β=0.35, γ=0.25)
        """
        ALPHA = 0.40  # severity weight
        BETA = 0.35   # temporal precedence weight
        GAMMA = 0.25  # topology depth weight

        results: List[RootCauseResult] = []

        # Compute temporal range for normalization
        all_times = [
            n.earliest_anomaly_time
            for n in visited.values()
            if n.earliest_anomaly_time
        ]
        if all_times:
            earliest_global = min(all_times)
            latest_global = max(all_times)
            time_range_seconds = max(
                (latest_global - earliest_global).total_seconds(), 1.0
            )
        else:
            earliest_global = alert_time
            time_range_seconds = 1.0

        max_depth = max((n.depth for n in visited.values()), default=1)
        max_depth = max(max_depth, 1)

        for node in candidates:
            # S_severity: already in [0, 1]
            s_severity = node.severity_score

            # S_temporal: how early is this anomaly relative to the full range?
            # Earlier anomalies get higher scores (they are more likely the cause)
            if node.earliest_anomaly_time:
                seconds_from_latest = (
                    latest_global - node.earliest_anomaly_time
                ).total_seconds()
                s_temporal = seconds_from_latest / time_range_seconds
            else:
                s_temporal = 0.0

            # S_topology: deeper nodes get higher scores
            # (farther from symptom = closer to root cause)
            s_topology = node.depth / max_depth

            # Combined confidence score
            confidence = ALPHA * s_severity + BETA * s_temporal + GAMMA * s_topology
            confidence = round(min(confidence, 0.99), 3)

            # Build the evidence chain (backtrack path from this candidate to symptom)
            evidence_chain = self._build_evidence_chain(node, visited)
            backtrack_path = self._build_backtrack_path(node, visited)

            # Compute temporal gap between root cause and symptom
            temporal_gap = 0.0
            if node.earliest_anomaly_time:
                temporal_gap = (alert_time - node.earliest_anomaly_time).total_seconds()

            # Build cause description
            cause_desc = self._build_cause_description(node, backtrack_path)

            results.append(RootCauseResult(
                rank=0,  # will be set after sorting
                cause_service=node.service,
                cause_description=cause_desc,
                confidence=confidence,
                depth=node.depth,
                temporal_gap_seconds=temporal_gap,
                evidence_chain=evidence_chain,
                backtrack_path=backtrack_path,
            ))

        # Sort by confidence descending
        results.sort(key=lambda r: r.confidence, reverse=True)
        for idx, r in enumerate(results, start=1):
            r.rank = idx

        return results

    def _build_cause_description(
        self, node: BacktrackNode, backtrack_path: List[str],
    ) -> str:
        """Generate a human-readable root cause description."""
        svc = node.service
        meta = self.metadata.get(svc, {})
        svc_type = meta.get("type", "unknown")

        # Summarize the most severe anomaly
        if node.anomalies:
            worst = max(node.anomalies, key=lambda a: a.severity_score)
            anomaly_desc = worst.message
        else:
            anomaly_desc = "unknown anomaly"

        path_str = " → ".join(reversed(backtrack_path))

        return (
            f"{svc} ({svc_type}): {anomaly_desc}. "
            f"Fault propagation path: {path_str}."
        )

    def _build_evidence_chain(
        self, candidate: BacktrackNode, visited: Dict[str, BacktrackNode],
    ) -> List[Dict[str, Any]]:
        """Build an ordered evidence chain from the root cause to the symptom."""
        chain = []
        current = candidate

        while current:
            step = {
                "service": current.service,
                "depth": current.depth,
                "anomaly_count": len(current.anomalies),
                "severity_score": round(current.severity_score, 3),
                "earliest_anomaly": (
                    current.earliest_anomaly_time.isoformat()
                    if current.earliest_anomaly_time
                    else None
                ),
                "anomaly_details": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "level": a.level,
                        "message": a.message,
                        "error_type": a.error_type,
                        "severity": round(a.severity_score, 3),
                    }
                    for a in sorted(current.anomalies, key=lambda x: x.timestamp)
                ],
            }
            chain.append(step)

            # Walk up to parent
            parent_name = current.parent_service
            current = visited.get(parent_name) if parent_name else None

        # Reverse so it reads root_cause → ... → symptom
        chain.reverse()
        return chain

    def _build_backtrack_path(
        self, candidate: BacktrackNode, visited: Dict[str, BacktrackNode],
    ) -> List[str]:
        """Build the service path from symptom to root cause."""
        path = []
        current = candidate

        while current:
            path.append(current.service)
            parent_name = current.parent_service
            current = visited.get(parent_name) if parent_name else None

        # path is [root_cause, ..., symptom] — keep this order for backtrack semantics
        return path

    # ------------------------------------------------------------------
    # Phase 4: Propagation Path Inference
    # ------------------------------------------------------------------

    def _trace_propagation_path(
        self,
        root_cause: str,
        symptom: str,
        visited: Dict[str, BacktrackNode],
    ) -> List[str]:
        """
        Trace the inferred fault propagation path from root cause to symptom.

        This is the *reverse* of the backtrack path: it shows how the fault
        actually propagated through the system.
        """
        # Walk from root_cause upward through parent pointers
        path = []
        node = visited.get(root_cause)

        while node:
            path.append(node.service)
            parent_name = node.parent_service
            node = visited.get(parent_name) if parent_name else None

        # path is [root_cause, ..., symptom] — this IS the propagation direction
        return path

    # ------------------------------------------------------------------
    # Phase 5: Blast Radius
    # ------------------------------------------------------------------

    def _compute_blast_radius(self, service: str, depth: int = 3) -> List[str]:
        """Compute upstream services impacted by a fault in the given service."""
        impacted = set()
        frontier = [(service, 0)]

        while frontier:
            node, d = frontier.pop(0)
            if d > depth:
                continue
            impacted.add(node)

            for upstream in self._reverse_graph.get(node, []):
                if upstream not in impacted:
                    frontier.append((upstream, d + 1))

        return sorted(impacted)


# ---------------------------------------------------------------------------
# Helper: Convert raw log data to AnomalyEvidence
# ---------------------------------------------------------------------------

def logs_to_anomaly_data(
    log_records: List[Dict[str, Any]],
) -> Dict[str, List[AnomalyEvidence]]:
    """
    Convert a list of raw log record dicts to grouped AnomalyEvidence.

    Returns:
        {service_name: [AnomalyEvidence, ...]}
    """
    grouped: Dict[str, List[AnomalyEvidence]] = {}

    for rec in log_records:
        ts_str = rec.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        evidence = AnomalyEvidence(
            service=rec.get("service", "unknown"),
            timestamp=ts,
            level=rec.get("level", "INFO"),
            message=rec.get("message", ""),
            error_type=rec.get("error_type"),
            status_code=rec.get("status_code"),
            latency_ms=rec.get("latency_ms"),
            upstream=rec.get("upstream"),
            trace_id=rec.get("trace_id"),
        )

        grouped.setdefault(evidence.service, []).append(evidence)

    return grouped
