
import json
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Any, Optional


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY_FILE = BASE_DIR / "system_topology.json"


def _resolve_topology_file(topology_file: Optional[str] = None) -> Path:
    candidate = topology_file or os.getenv("ARCHITECTURE_TOPOLOGY_FILE")
    return Path(candidate).expanduser().resolve() if candidate else DEFAULT_TOPOLOGY_FILE


def load_topology_data(topology_file: Optional[str] = None) -> Dict[str, Any]:
    selected = _resolve_topology_file(topology_file)
    with selected.open('r', encoding='utf-8') as f:
        return json.load(f)


def get_system_diagram(topology_file: Optional[str] = None) -> Dict[str, Any]:
    return load_topology_data(topology_file)["diagram"]


def get_service_catalog(topology_file: Optional[str] = None) -> Dict[str, Any]:
    return load_topology_data(topology_file)["services"]


def get_service_metadata(service: str, topology_file: Optional[str] = None) -> Dict[str, Any]:
    services = get_service_catalog(topology_file)
    if service not in services:
        raise KeyError(f"Unknown service: {service}")
    return {"service": service, **services[service]}


def get_service_dependencies(service: str, topology_file: Optional[str] = None) -> Dict[str, Any]:
    meta = get_service_metadata(service, topology_file)
    return {
        "service": service,
        "depends_on": meta["depends_on"],
        "upstream_of": meta["upstream_of"],
        "criticality": meta["criticality"],
        "type": meta["type"],
    }


def get_related_services(service: str, topology_file: Optional[str] = None) -> Dict[str, Any]:
    meta = get_service_metadata(service, topology_file)
    related = list(dict.fromkeys(meta["depends_on"] + meta["upstream_of"]))
    return {"service": service, "related_services": related}


def _build_graph(topology_file: Optional[str] = None) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {}
    edges = get_system_diagram(topology_file)["content"]["edges"]
    for src, dst in edges:
        graph.setdefault(src, []).append(dst)
        graph.setdefault(dst, [])
    return graph


def find_path(source: str, target: str, topology_file: Optional[str] = None) -> Dict[str, Any]:
    graph = _build_graph(topology_file)

    if source not in graph or target not in graph:
        return {"source": source, "target": target, "path": [], "found": False}

    q = deque([(source, [source])])
    visited = set()

    while q:
        node, path = q.popleft()
        if node == target:
            return {"source": source, "target": target, "path": path, "found": True}

        if node in visited:
            continue
        visited.add(node)

        for nxt in graph.get(node, []):
            if nxt not in visited:
                q.append((nxt, path + [nxt]))

    return {"source": source, "target": target, "path": [], "found": False}


def infer_blast_radius(service: str, depth: int = 2, topology_file: Optional[str] = None) -> Dict[str, Any]:
    graph = _build_graph(topology_file)
    reverse_graph: Dict[str, List[str]] = {}

    for src, dsts in graph.items():
        for dst in dsts:
            reverse_graph.setdefault(dst, []).append(src)

    impacted = set()
    frontier = [(service, 0)]

    while frontier:
        node, d = frontier.pop(0)
        if d > depth:
            continue
        impacted.add(node)

        for upstream in reverse_graph.get(node, []):
            if upstream not in impacted:
                frontier.append((upstream, d + 1))

    return {"service": service, "depth": depth, "blast_radius": sorted(list(impacted))}
