#!/usr/bin/env python3
"""
build_trainticket_topology.py — RE2-TT 트레이스 데이터에서 자동으로 토폴로지 추출.

v2 변경점:
    - "Orphan service" 자동 보강: 트레이스에 등장하지만 부모/자식 엣지가
      없는 서비스도 토폴로지에 포함 (예: ts-auth-service 같은 entry-point)
    - 셀프 루프 제외, 출현 빈도 추적

사용법 (Windows PowerShell):
    cd C:\\Users\\hwang\\Documents\\GitHub\\rca_work
    python build_trainticket_topology.py `
        --data-dir "C:\\Users\\hwang\\Documents\\RE2-TT" `
        --output trainticket_topology.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# RE2-TT traces.csv 컬럼명
COL_SPAN_ID     = 'spanID'
COL_PARENT_SPAN = 'parentSpanID'
COL_SERVICE     = 'serviceName'

# Fault injection target services (RE2-TT 폴더명에서 자동 발견)
# 토폴로지에 반드시 포함되어야 함 — orphan이라도 강제 추가
FAULT_TARGET_SERVICES_DEFAULT = {
    'ts-auth-service',
    'ts-order-service',
    'ts-route-service',
    'ts-train-service',
    'ts-travel-service',
}


def find_fault_targets(data_dir: Path) -> Set[str]:
    """data_dir의 폴더명에서 fault target 서비스 추출."""
    targets = set()
    for sub in data_dir.iterdir():
        if not sub.is_dir():
            continue
        # ts-auth-service_cpu → ts-auth-service
        for fault in ('cpu', 'mem', 'disk', 'delay', 'loss', 'socket'):
            suffix = '_' + fault
            if sub.name.endswith(suffix):
                targets.add(sub.name[:-len(suffix)])
                break
    return targets if targets else FAULT_TARGET_SERVICES_DEFAULT


def find_trace_files(data_dir: Path, all_reps: bool = False) -> List[Path]:
    candidates: List[Path] = []
    fault_folders = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    for ff in fault_folders:
        rep_folders = sorted([p for p in ff.iterdir() if p.is_dir()])
        if not all_reps:
            rep_folders = rep_folders[:1]
        for rep in rep_folders:
            tf = rep / 'traces.csv'
            if tf.exists():
                candidates.append(tf)
    return candidates


def extract_from_trace(
    trace_file: Path,
    verbose: bool = False,
) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """단일 traces.csv → (edges, all_observed_services).

    Returns:
        edges:   set of (parent_service, child_service)
        observed: set of all serviceName seen (포함 entry-point + orphan)
    """
    edges: Set[Tuple[str, str]] = set()
    observed: Set[str] = set()
    span_to_service: Dict[str, str] = {}
    rows_buffer: List[Tuple[str, str]] = []

    file_size_mb = trace_file.stat().st_size / 1024 / 1024
    if verbose:
        case_id = f"{trace_file.parent.parent.name}/{trace_file.parent.name}"
        print(f"  Reading {case_id}/traces.csv ({file_size_mb:.1f} MB)...", flush=True)

    try:
        with trace_file.open('r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                service = (row.get(COL_SERVICE) or '').strip()
                span_id = (row.get(COL_SPAN_ID) or '').strip()
                parent_span = (row.get(COL_PARENT_SPAN) or '').strip()

                if not service:
                    continue
                observed.add(service)
                if span_id:
                    span_to_service[span_id] = service
                if parent_span:
                    rows_buffer.append((parent_span, service))

            if verbose:
                print(f"    {row_count:,} rows, {len(observed)} observed services", flush=True)
    except Exception as exc:
        print(f"  [warn] {trace_file}: {exc}", file=sys.stderr)
        return edges, observed

    for parent_span, child_service in rows_buffer:
        parent_service = span_to_service.get(parent_span)
        if parent_service and parent_service != child_service:
            edges.add((parent_service, child_service))

    if verbose:
        edge_services = set(s for e in edges for s in e)
        orphans = observed - edge_services
        print(f"    -> {len(edges)} edges, {len(orphans)} orphan services", flush=True)
        if orphans:
            print(f"       orphans: {sorted(orphans)}", flush=True)

    return edges, observed


def classify_service(name: str, in_deg: int, out_deg: int, is_orphan: bool) -> Tuple[str, str]:
    lname = name.lower()

    if any(k in lname for k in ('ui-dashboard', 'frontend', 'gateway', 'ts-ui')):
        svc_type = 'frontend' if 'ui' in lname or 'frontend' in lname else 'gateway'
    elif any(k in lname for k in ('mongo', 'redis', 'mysql', 'db', 'database')):
        svc_type = 'database'
    elif any(k in lname for k in ('mq', 'queue', 'rabbit', 'kafka')):
        svc_type = 'queue'
    elif 'auth' in lname:
        svc_type = 'auth'   # entry-point auth services
    else:
        svc_type = 'backend'

    if svc_type in ('frontend', 'gateway', 'auth'):
        crit = 'high'
    elif svc_type == 'database':
        crit = 'high'
    elif in_deg >= 5:
        crit = 'high'
    elif in_deg >= 2:
        crit = 'medium'
    else:
        crit = 'low'

    return svc_type, crit


def build_topology(
    edges: Set[Tuple[str, str]],
    observed_services: Set[str],
    fault_targets: Set[str],
) -> Dict:
    """엣지 + 관측 서비스 → topology JSON.

    포함 규칙:
        1. 엣지에 등장한 모든 서비스 (caller/callee)
        2. 트레이스에 등장한 orphan 서비스 (관측됨 but 엣지 없음)
        3. fault_targets 중 위 둘에 안 잡힌 것 (강제 추가, 경고 표시)
    """
    services_in_edges: Set[str] = set(s for e in edges for s in e)
    orphan_services = observed_services - services_in_edges

    # fault targets 중 트레이스에 아예 없는 서비스 식별
    missing_targets = fault_targets - observed_services - services_in_edges

    all_services = services_in_edges | orphan_services | missing_targets

    depends_on: Dict[str, Set[str]] = defaultdict(set)
    upstream_of: Dict[str, Set[str]] = defaultdict(set)
    for caller, callee in edges:
        depends_on[caller].add(callee)
        upstream_of[callee].add(caller)

    in_degree = {s: len(upstream_of.get(s, set())) for s in all_services}
    out_degree = {s: len(depends_on.get(s, set())) for s in all_services}

    services_dict: Dict[str, Dict] = {}
    for s in sorted(all_services):
        is_orphan = s in orphan_services
        is_missing = s in missing_targets
        svc_type, crit = classify_service(s, in_degree[s], out_degree[s], is_orphan)
        entry = {
            "type": svc_type,
            "criticality": crit,
            "depends_on": sorted(depends_on.get(s, set())),
            "upstream_of": sorted(upstream_of.get(s, set())),
        }
        if is_orphan:
            entry["_note"] = "orphan: observed in traces but no parent/child edges (entry-point or self-only calls)"
        if is_missing:
            entry["_note"] = "missing: fault-target but absent from traces — added defensively"
        services_dict[s] = entry

    return {
        "_comment": (
            "Train Ticket topology auto-extracted from RCAEval RE2-TT trace data. "
            "Edges = observed (parent->child) span pairs unioned across cases. "
            "Orphan services (entry-points like ts-auth-service) are included with empty edges."
        ),
        "_meta": {
            "edge_count": len(edges),
            "service_count": len(all_services),
            "orphan_services": sorted(orphan_services),
            "missing_fault_targets": sorted(missing_targets),
        },
        "diagram": {
            "uri": "arch://train-ticket/latest",
            "name": "train-ticket-topology",
            "description": (
                "FudanSELab Train Ticket benchmark used in RCAEval RE2-TT. "
                "Topology extracted from observed traces."
            ),
            "mimeType": "application/json",
            "content": {
                "services": sorted(all_services),
                "edges": sorted([list(e) for e in edges]),
            },
        },
        "services": services_dict,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--all-reps', action='store_true')
    ap.add_argument('--max-folders', type=int, default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Data dir not found: {args.data_dir}")

    fault_targets = find_fault_targets(args.data_dir)
    print(f"Fault target services (auto-detected from folder names): {sorted(fault_targets)}\n")

    print(f"Scanning trace files under {args.data_dir} ...")
    trace_files = find_trace_files(args.data_dir, all_reps=args.all_reps)
    if args.max_folders:
        trace_files = trace_files[:args.max_folders]

    if not trace_files:
        raise SystemExit("No traces.csv files found.")

    print(f"  Found {len(trace_files)} trace files (all_reps={args.all_reps})")
    total_size_mb = sum(tf.stat().st_size for tf in trace_files) / 1024 / 1024
    print(f"  Total size: {total_size_mb:.1f} MB\n")

    all_edges: Set[Tuple[str, str]] = set()
    all_observed: Set[str] = set()

    for i, tf in enumerate(trace_files, 1):
        print(f"[{i}/{len(trace_files)}]", flush=True)
        edges, observed = extract_from_trace(tf, verbose=not args.quiet)
        all_edges.update(edges)
        all_observed.update(observed)
        services_so_far = set(s for e in all_edges for s in e) | all_observed
        print(f"  cumulative: {len(all_edges)} edges, {len(services_so_far)} services\n", flush=True)

    if not all_edges and not all_observed:
        raise SystemExit("Nothing extracted. Check trace file format.")

    topology = build_topology(all_edges, all_observed, fault_targets)
    services_in_edges = set(s for e in all_edges for s in e)
    orphan_services = all_observed - services_in_edges
    missing_targets = fault_targets - all_observed - services_in_edges

    print(f"\n=== Final topology ===")
    print(f"  Services in edges:    {len(services_in_edges)}")
    print(f"  Orphan services:      {len(orphan_services)}")
    if orphan_services:
        for s in sorted(orphan_services):
            print(f"    - {s}")
    print(f"  Missing fault targets: {len(missing_targets)}")
    if missing_targets:
        for s in sorted(missing_targets):
            print(f"    ! {s}  (added defensively)")
    print(f"  Total services:       {len(topology['services'])}")
    print(f"  Total edges:          {len(all_edges)}")

    print(f"\n  Service list:")
    for s in sorted(topology['services'].keys()):
        info = topology['services'][s]
        note = f"  [{info['_note']}]" if '_note' in info else ""
        print(f"    {s:40s} type={info['type']:10s} crit={info['criticality']:6s} "
              f"in={len(info['upstream_of'])} out={len(info['depends_on'])}{note}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(topology, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")
    print(f"  ({args.output.stat().st_size / 1024:.1f} KB)")

    # Sanity check
    fault_target_coverage = fault_targets & set(topology['services'].keys())
    print(f"\nFault target coverage: {len(fault_target_coverage)}/{len(fault_targets)}")
    for t in sorted(fault_targets):
        in_topo = t in topology['services']
        in_edges = t in services_in_edges
        marker = '✓' if in_edges else ('?' if in_topo else '✗')
        print(f"  {marker} {t}")


if __name__ == '__main__':
    main()
