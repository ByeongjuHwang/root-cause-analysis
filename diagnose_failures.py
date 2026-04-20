#!/usr/bin/env python3
"""
diagnose_failures.py — RCAEval 결과의 실패 원인을 정량 태깅.

목적:
    "후보 생성 문제"와 "랭킹 문제"를 분리해서,
    어느 에이전트를 강화해야 하는지 데이터로 결정한다.

각 케이스를 아래 4가지 실패 유형 중 하나로 분류:

    [A] RANKING       — 정답이 Top-3 안에 있는데 Top-1은 아님    → RCA Agent 재랭킹 강화
    [B] CANDIDATE     — 정답이 Top-3 밖에 있음                    → Log/Metric/Trace 수집 강화
    [C] PATH          — 원인 서비스는 맞았으나 전파 경로 틀림     → Topology Agent 강화
    [D] OVERCONFIDENT — 근거 없이 잘못된 후보에 high confidence → Verifier 강화

추가로 각 케이스별로 다음을 기록:
    - log_evidence_count:    로그 증거가 수집되었는가
    - topology_evidence_cnt: 토폴로지 증거가 수집되었는가
    - has_cause_service:     RCA Agent가 구조화 필드를 채웠는가
    - top1_service:          실제 Top-1 예측
    - top3_services:         Top-3 예측 목록
    - confidence_top1:       Top-1의 confidence
    - verifier_verdict:      accepted/rejected/weak
    - agent_errors:          어떤 에이전트가 실패했는가

사용법:
    python diagnose_failures.py --results-dir experiments/rcaeval
    python diagnose_failures.py --results-dir experiments/rcaeval --csv diagnosis.csv
    python diagnose_failures.py --results-dir experiments/rcaeval --only ours
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Unified matching
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_core import candidate_matches_truth  # noqa: E402


FAILURE_TYPES = {
    "CORRECT":        "Top-1 is correct",
    "RANKING":        "Answer in Top-3 but not Top-1 (re-ranking problem)",
    "CANDIDATE":      "Answer NOT in Top-3 (candidate generation problem)",
    "PATH":           "Top-1 service correct but propagation path wrong",
    "OVERCONFIDENT":  "Wrong candidate with confidence >= 0.7 and verdict=accepted",
    "AGENT_ERROR":    "One or more agents failed mid-pipeline",
}


def _extract_path(result: Dict[str, Any]) -> List[str]:
    prop = (
        result.get("impact_analysis", {}).get("propagation_path")
        or result.get("propagation_path")
        or []
    )
    return [str(p) for p in (prop or [])]


def _extract_evidence_counts(result: Dict[str, Any]) -> Dict[str, int]:
    """Count evidence from each agent's output."""
    counts: Dict[str, int] = defaultdict(int)
    for ar in result.get("agent_results") or []:
        agent = ar.get("agent") or "unknown"
        evidence_list = ar.get("evidence") or []
        counts[agent] = len(evidence_list)
    return dict(counts)


def _has_topology_evidence(result: Dict[str, Any]) -> bool:
    topo = result.get("evidence_summary", {}).get("topology_evidence") or []
    return len(topo) > 0


def _has_log_evidence(result: Dict[str, Any]) -> bool:
    logs = result.get("evidence_summary", {}).get("log_evidence") or []
    return len(logs) > 0


def _agent_errors(result: Dict[str, Any]) -> List[str]:
    errs = result.get("agent_errors") or []
    return [e.get("agent", "?") for e in errs if isinstance(e, dict)]


def diagnose_case(record: Dict[str, Any]) -> Dict[str, Any]:
    """Classify one experiment record into a failure category + metadata.

    Accepts both run_experiment.py records (synthetic) and run_rcaeval.py
    records (benchmark). Dispatches based on presence of 'benchmark' key.
    """
    # Hard error during pipeline — no result to diagnose
    if "error" in record and record.get("error"):
        return {
            "failure_type": "AGENT_ERROR",
            "failure_reason": f"Pipeline error: {str(record['error'])[:200]}",
            "case": record.get("case") or f"{record.get('system')}_{record.get('scenario')}",
            "system": record.get("system"),
            "gt_service": record.get("gt_service") or record.get("evaluation", {}).get("ground_truth_root_cause"),
            "gt_fault": record.get("gt_fault"),
            "elapsed_seconds": record.get("elapsed_seconds"),
        }

    # Extract prediction
    result = record.get("result", {}) or {}
    evaluation = record.get("evaluation") or {}
    candidates = result.get("root_cause_candidates") or []
    top3 = candidates[:3]
    top1 = candidates[0] if candidates else {}

    # Ground truth (both schemas)
    gt_service = (
        record.get("gt_service")
        or evaluation.get("ground_truth_root_cause")
    )
    gt_path = (
        evaluation.get("ground_truth_path")  # may not exist for RCAEval
        or []
    )

    # Match checks
    ac_at_1 = bool(evaluation.get("ac_at_1"))
    ac_at_3 = bool(evaluation.get("ac_at_3"))
    path_accuracy = evaluation.get("path_accuracy")  # bool|None

    # If we don't have ac_at_X flags, recompute from candidates
    if gt_service and not ac_at_1:
        ac_at_1 = candidate_matches_truth(top1, gt_service) if top1 else False
        ac_at_3 = any(candidate_matches_truth(c, gt_service) for c in top3)

    # Failure classification
    top1_conf = float(top1.get("confidence") or 0.0)
    verdict = (
        result.get("verification", {}).get("verdict")
        or evaluation.get("verdict")
        or "unknown"
    )
    agent_errs = _agent_errors(result)

    if agent_errs:
        failure_type = "AGENT_ERROR"
        failure_reason = f"Agents failed: {agent_errs}"
    elif ac_at_1 and path_accuracy is False:
        failure_type = "PATH"
        failure_reason = "Top-1 service correct but propagation path diverges from GT"
    elif ac_at_1:
        failure_type = "CORRECT"
        failure_reason = "Top-1 matches ground truth"
    elif ac_at_3:
        failure_type = "RANKING"
        failure_reason = "GT found in Top-3 but not Top-1"
    else:
        # Not in Top-3 — further split into CANDIDATE vs OVERCONFIDENT
        if top1_conf >= 0.7 and verdict.lower().startswith("accept"):
            failure_type = "OVERCONFIDENT"
            failure_reason = (
                f"Wrong top-1 '{top1.get('cause_service') or top1.get('cause')}' "
                f"accepted with conf={top1_conf:.2f}"
            )
        else:
            failure_type = "CANDIDATE"
            failure_reason = "GT not present among top-3 candidates"

    # Per-agent evidence counts
    evidence_counts = _extract_evidence_counts(result)

    return {
        # Identity
        "case": record.get("case") or f"{record.get('system')}_{record.get('scenario')}",
        "system": record.get("system"),
        "gt_service": gt_service,
        "gt_fault": record.get("gt_fault"),
        # Classification
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        # Metrics already computed
        "ac_at_1": ac_at_1,
        "ac_at_3": ac_at_3,
        "path_accuracy": path_accuracy,
        # Prediction details
        "top1_service": top1.get("cause_service") or (top1.get("cause") or "")[:60],
        "top1_confidence": round(top1_conf, 3),
        "top3_services": [
            c.get("cause_service") or (c.get("cause") or "")[:40] for c in top3
        ],
        # Diagnostic signals (what inputs did the pipeline have?)
        "verifier_verdict": verdict,
        "log_evidence_count": evidence_counts.get("log-agent", 0),
        "topology_evidence_count": evidence_counts.get("topology-agent", 0),
        "rca_evidence_count": evidence_counts.get("rca-agent", 0),
        "has_log_evidence": _has_log_evidence(result),
        "has_topology_evidence": _has_topology_evidence(result),
        # Path
        "predicted_path": _extract_path(result),
        "predicted_path_length": len(_extract_path(result)),
        "gt_path": gt_path,
        "gt_path_length": len(gt_path),
        # Misc
        "elapsed_seconds": record.get("elapsed_seconds"),
        "agent_errors": agent_errs,
    }


# =============================================================================
# Loading results
# =============================================================================

def load_results(results_dir: Path, only_system: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load all experiment JSON files from a results directory.

    Filenames follow the pattern:
        {system}_{case}.json   (e.g. ours_cartservice_cpu_1.json)
    or
        {system}_{scenario}.json  (e.g. ours_s1.json)

    Skips summary files and retry/backup files.
    """
    records = []
    skip_prefixes = ("rcaeval_summary", "rcaeval_results", "summary_", "results_")

    for p in sorted(results_dir.glob("*.json")):
        if any(p.name.startswith(pref) for pref in skip_prefixes):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [!] Skipping {p.name}: {e}", file=sys.stderr)
            continue

        # Filter by system if requested
        if only_system and data.get("system") != only_system:
            continue

        records.append(data)
    return records


# =============================================================================
# Reporting
# =============================================================================

def summarize(diagnoses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate diagnoses into a summary report."""
    if not diagnoses:
        return {"total": 0}

    total = len(diagnoses)
    by_type = Counter(d["failure_type"] for d in diagnoses)

    # Break down by fault type (RCAEval) or scenario (synthetic)
    by_fault_failure: Dict[str, Counter] = defaultdict(Counter)
    for d in diagnoses:
        key = d.get("gt_fault") or "N/A"
        by_fault_failure[key][d["failure_type"]] += 1

    # Break down by service
    by_service_failure: Dict[str, Counter] = defaultdict(Counter)
    for d in diagnoses:
        key = d.get("gt_service") or "unknown"
        by_service_failure[key][d["failure_type"]] += 1

    # Evidence signal for failures
    failures = [d for d in diagnoses if d["failure_type"] != "CORRECT"]
    candidate_fails = [d for d in diagnoses if d["failure_type"] == "CANDIDATE"]
    ranking_fails = [d for d in diagnoses if d["failure_type"] == "RANKING"]

    def _frac(predicate, items):
        if not items:
            return 0.0
        return sum(1 for x in items if predicate(x)) / len(items)

    diagnostic_signals = {
        # Of CANDIDATE failures, how many had evidence at all?
        "candidate_failures_had_log_evidence": _frac(
            lambda d: d["has_log_evidence"], candidate_fails
        ),
        "candidate_failures_had_topology_evidence": _frac(
            lambda d: d["has_topology_evidence"], candidate_fails
        ),
        "candidate_failures_log_avg": (
            sum(d["log_evidence_count"] for d in candidate_fails) / len(candidate_fails)
            if candidate_fails else 0.0
        ),
        # RANKING failures: do they have comparable evidence to CORRECT?
        "ranking_failures_log_avg": (
            sum(d["log_evidence_count"] for d in ranking_fails) / len(ranking_fails)
            if ranking_fails else 0.0
        ),
        # Low-confidence correct vs high-confidence incorrect?
        "correct_avg_confidence": (
            sum(d["top1_confidence"] for d in diagnoses if d["failure_type"] == "CORRECT")
            / max(by_type.get("CORRECT", 1), 1)
        ),
        "incorrect_avg_confidence": (
            sum(d["top1_confidence"] for d in failures)
            / max(len(failures), 1)
        ),
    }

    return {
        "total": total,
        "by_type": dict(by_type),
        "by_type_rate": {k: round(v / total, 3) for k, v in by_type.items()},
        "by_fault": {k: dict(v) for k, v in by_fault_failure.items()},
        "by_service": {k: dict(v) for k, v in by_service_failure.items()},
        "diagnostic_signals": {k: round(v, 3) for k, v in diagnostic_signals.items()},
    }


def print_report(summary: Dict[str, Any]) -> None:
    total = summary["total"]
    if total == 0:
        print("No records found.")
        return

    print("=" * 78)
    print("  FAILURE DIAGNOSIS REPORT")
    print("=" * 78)
    print(f"  Total cases: {total}")
    print()

    # 1. Headline distribution
    print("  [1] Failure Type Distribution")
    print("  " + "-" * 76)
    order = ["CORRECT", "RANKING", "CANDIDATE", "PATH", "OVERCONFIDENT", "AGENT_ERROR"]
    for ft in order:
        count = summary["by_type"].get(ft, 0)
        rate = summary["by_type_rate"].get(ft, 0.0)
        bar = "█" * int(rate * 50)
        label = f"{ft:<15}"
        print(f"  {label} {count:>3}/{total} ({rate:.1%}) {bar}")
        desc = FAILURE_TYPES.get(ft, "")
        print(f"  {'':<15} → {desc}")
    print()

    # 2. Diagnostic interpretation
    print("  [2] Actionable Interpretation")
    print("  " + "-" * 76)
    by_type = summary["by_type"]
    n_correct = by_type.get("CORRECT", 0)
    n_ranking = by_type.get("RANKING", 0)
    n_candidate = by_type.get("CANDIDATE", 0)
    n_path = by_type.get("PATH", 0)
    n_overconfident = by_type.get("OVERCONFIDENT", 0)
    n_agent_err = by_type.get("AGENT_ERROR", 0)

    lines = []

    if n_ranking >= n_candidate and n_ranking > n_correct // 2:
        lines.append(
            f"  ▲ RANKING dominant ({n_ranking}). Top-3에는 정답이 들어오지만 Top-1을 못 잡음.\n"
            f"    → RCA Agent의 재랭킹/confidence scoring 개선이 1순위"
        )
    if n_candidate >= n_ranking and n_candidate > total // 4:
        lines.append(
            f"  ▲ CANDIDATE dominant ({n_candidate}). 정답 서비스가 후보군에도 없음.\n"
            f"    → Log/Metric/Trace 수집 단계 강화 필요 (증거 부족)"
        )
    if n_path > 0:
        lines.append(
            f"  ▲ PATH failures: {n_path}. 원인은 맞히지만 전파 경로가 틀림.\n"
            f"    → Topology Agent의 그래프 탐색/정렬 강화"
        )
    if n_overconfident > 0:
        lines.append(
            f"  ▲ OVERCONFIDENT: {n_overconfident}. 근거 없이 확신을 가짐.\n"
            f"    → Verifier Agent의 거부 임계값/증거 요구 조건 강화"
        )
    if n_agent_err > 0:
        lines.append(
            f"  ▲ AGENT_ERROR: {n_agent_err}. 파이프라인 중간 실패.\n"
            f"    → 로깅/재시도 메커니즘 검토"
        )
    if not lines:
        lines.append("  시스템이 전반적으로 잘 작동 중.")

    for line in lines:
        print(line)
    print()

    # 3. Evidence signals
    print("  [3] Evidence & Confidence Signals")
    print("  " + "-" * 76)
    sig = summary["diagnostic_signals"]
    print(f"  CANDIDATE 실패 중 로그 증거 있었음:     {sig['candidate_failures_had_log_evidence']:.1%}")
    print(f"  CANDIDATE 실패 중 토폴로지 증거 있었음: {sig['candidate_failures_had_topology_evidence']:.1%}")
    print(f"  CANDIDATE 실패 평균 로그 수:            {sig['candidate_failures_log_avg']:.1f}")
    print(f"  RANKING 실패 평균 로그 수:              {sig['ranking_failures_log_avg']:.1f}")
    print(f"  정답 케이스 평균 Top-1 confidence:      {sig['correct_avg_confidence']:.3f}")
    print(f"  오답 케이스 평균 Top-1 confidence:      {sig['incorrect_avg_confidence']:.3f}")
    conf_gap = sig['correct_avg_confidence'] - sig['incorrect_avg_confidence']
    if conf_gap < 0.1:
        print(f"  ⚠ Confidence gap {conf_gap:+.3f} → confidence가 정답 신호로 안 작동함 (Verifier 필요)")
    else:
        print(f"  ✓ Confidence gap {conf_gap:+.3f} → 정답일 때 더 높은 confidence 보임")
    print()

    # 4. Per-fault breakdown (RCAEval-specific)
    if summary["by_fault"] and set(summary["by_fault"].keys()) != {"N/A"}:
        print("  [4] By Fault Type")
        print("  " + "-" * 76)
        header = f"  {'Fault':<10} {'Total':>5}  {'OK':>4} {'Rank':>4} {'Cand':>4} {'Path':>4} {'OC':>3} {'Err':>3}"
        print(header)
        for fault in sorted(summary["by_fault"].keys()):
            counts = summary["by_fault"][fault]
            total_f = sum(counts.values())
            row = (
                f"  {fault:<10} {total_f:>5}  "
                f"{counts.get('CORRECT', 0):>4} "
                f"{counts.get('RANKING', 0):>4} "
                f"{counts.get('CANDIDATE', 0):>4} "
                f"{counts.get('PATH', 0):>4} "
                f"{counts.get('OVERCONFIDENT', 0):>3} "
                f"{counts.get('AGENT_ERROR', 0):>3}"
            )
            print(row)
        print()

    # 5. Per-service breakdown
    if summary["by_service"]:
        print("  [5] By Service")
        print("  " + "-" * 76)
        for service in sorted(summary["by_service"].keys()):
            counts = summary["by_service"][service]
            total_s = sum(counts.values())
            n_ok = counts.get("CORRECT", 0)
            pct = n_ok / total_s if total_s else 0
            print(
                f"  {service:<30} {n_ok:>3}/{total_s:<3} correct ({pct:.1%})  "
                f"[rank={counts.get('RANKING', 0)}, cand={counts.get('CANDIDATE', 0)}, "
                f"path={counts.get('PATH', 0)}]"
            )
        print()

    print("=" * 78)


# =============================================================================
# CSV export
# =============================================================================

def export_csv(diagnoses: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "system", "case", "gt_service", "gt_fault",
        "failure_type", "failure_reason",
        "ac_at_1", "ac_at_3", "path_accuracy",
        "top1_service", "top1_confidence", "top3_services",
        "verifier_verdict",
        "log_evidence_count", "topology_evidence_count", "rca_evidence_count",
        "has_log_evidence", "has_topology_evidence",
        "predicted_path", "predicted_path_length",
        "gt_path_length",
        "elapsed_seconds", "agent_errors",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for d in diagnoses:
            row = {k: d.get(k) for k in fields}
            # Serialize list fields
            for list_field in ("top3_services", "predicted_path", "agent_errors"):
                if row.get(list_field) is not None:
                    row[list_field] = json.dumps(row[list_field], ensure_ascii=False)
            w.writerow(row)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/rcaeval"),
                        help="Directory containing experiment JSON files")
    parser.add_argument("--only", default=None,
                        help="Filter by system key (e.g. 'ours', 'b3')")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Export per-case diagnoses to CSV")
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="Export summary dict as JSON")
    parser.add_argument("--show-cases", type=str, default=None,
                        help="Show cases of a specific failure type (RANKING, CANDIDATE, PATH, OVERCONFIDENT)")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: results dir does not exist: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from {args.results_dir}...", file=sys.stderr)
    records = load_results(args.results_dir, only_system=args.only)
    print(f"  Loaded {len(records)} records", file=sys.stderr)

    if not records:
        print("No records to diagnose. Ensure experiment JSON files are in the directory.", file=sys.stderr)
        sys.exit(1)

    diagnoses = [diagnose_case(r) for r in records]
    summary = summarize(diagnoses)

    print_report(summary)

    # Optional: show specific failure cases
    if args.show_cases:
        target = args.show_cases.upper()
        print(f"\n--- {target} cases ---")
        for d in diagnoses:
            if d["failure_type"] == target:
                print(f"  {d['case']}")
                print(f"    GT:      {d['gt_service']} ({d.get('gt_fault', '')})")
                print(f"    Top-1:   {d['top1_service']} (conf={d['top1_confidence']})")
                print(f"    Top-3:   {d['top3_services']}")
                print(f"    Reason:  {d['failure_reason']}")
                print(f"    Evidence: log={d['log_evidence_count']}, "
                      f"topo={d['topology_evidence_count']}")

    if args.csv:
        export_csv(diagnoses, args.csv)
        print(f"\n[OK] CSV exported: {args.csv}", file=sys.stderr)

    if args.summary_json:
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[OK] Summary JSON: {args.summary_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
