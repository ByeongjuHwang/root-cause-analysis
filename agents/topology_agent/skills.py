"""
Topology Analysis Agent Skills — deterministic mode.

Queries the Architecture MCP repository for service topology data.

Enhanced for paper artifact:
    - Returns the FULL topology_graph and service_metadata in every response
      so the RCA Agent can use the exact same topology for TCB-RCA.
    - topology_file parameter is propagated through all calls.
"""

from typing import Dict, Any, List, Optional

from mcp_servers.architecture_mcp.app.repository import (
    load_topology_data,
    get_service_dependencies,
    get_related_services,
    find_path,
    infer_blast_radius,
)


def _build_full_topology(
    topology_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the full topology graph and service metadata from the MCP repository.

    Returns a dict with:
        topology_graph:    {service: [dependencies]}
        service_metadata:  {service: {type, criticality}}
    """
    data = load_topology_data(topology_file=topology_file)

    edges = data.get("diagram", {}).get("content", {}).get("edges", [])
    graph: Dict[str, List[str]] = {}
    for src, dst in edges:
        graph.setdefault(src, []).append(dst)
        graph.setdefault(dst, [])

    services = data.get("services", {})
    metadata: Dict[str, Dict[str, Any]] = {}
    for svc_name, svc_info in services.items():
        metadata[svc_name] = {
            "type": svc_info.get("type", "unknown"),
            "criticality": svc_info.get("criticality", "medium"),
        }
        graph.setdefault(svc_name, [])

    return {"topology_graph": graph, "service_metadata": metadata}


class ArchitectureAdapter:
    async def get_service_dependencies(self, service: str, topology_file: Optional[str] = None) -> Dict[str, Any]:
        return get_service_dependencies(service, topology_file=topology_file)

    async def get_related_services(self, service: str, topology_file: Optional[str] = None) -> Dict[str, Any]:
        return get_related_services(service, topology_file=topology_file)

    async def find_path(self, source: str, target: str, topology_file: Optional[str] = None) -> Dict[str, Any]:
        return find_path(source, target, topology_file=topology_file)

    async def infer_blast_radius(self, service: str, depth: int = 2, topology_file: Optional[str] = None) -> Dict[str, Any]:
        return infer_blast_radius(service, depth, topology_file=topology_file)


class TopologyAnalysisService:
    def __init__(self):
        self.arch = ArchitectureAdapter()

    async def analyze(
        self,
        service: str,
        suspected_downstream: Optional[str] = None,
        diagram_uri: Optional[str] = None,
        topology_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        deps = await self.arch.get_service_dependencies(service, topology_file=topology_file)
        related = await self.arch.get_related_services(service, topology_file=topology_file)
        blast = await self.arch.infer_blast_radius(service=service, depth=2, topology_file=topology_file)

        propagation = []
        if suspected_downstream:
            path_result = await self.arch.find_path(service, suspected_downstream, topology_file=topology_file)
            propagation = path_result.get("path", [])

        summary_parts = [
            f"{service} depends on {deps.get('depends_on', [])}.",
            f"Related services are {related.get('related_services', [])}.",
        ]

        if propagation:
            summary_parts.append(
                f"Likely propagation path from {service} to {suspected_downstream}: {propagation}."
            )

        summary_parts.append(f"Estimated blast radius: {blast.get('blast_radius', [])}.")

        # --- Build full topology graph for RCA Agent consumption ---
        full_topo = _build_full_topology(topology_file=topology_file)

        return {
            "summary": " ".join(summary_parts),
            "confidence": 0.76 if propagation else 0.68,
            "dependency_info": deps,
            "related_services": related.get("related_services", []),
            "propagation_path": propagation,
            "blast_radius": blast.get("blast_radius", []),
            "topology_file": topology_file,
            # --- Full topology for RCA engine (핵심 추가) ---
            "topology_graph": full_topo["topology_graph"],
            "service_metadata": full_topo["service_metadata"],
            "evidence": [
                {
                    "type": "topology",
                    "source": diagram_uri or "arch://system/latest",
                    "content": f"Dependency map confirms {service} relations: {deps}",
                    "metadata": {"service": service, "topology_file": topology_file},
                }
            ],
        }
