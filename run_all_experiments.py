#!/usr/bin/env python3
"""
run_all_experiments.py — 전체 시나리오 일괄 실행 + 결과 요약 + CSV export.

사용법:
    python run_all_experiments.py                         # 전체 실행
    python run_all_experiments.py --systems ours b3       # 특정 시스템만
    python run_all_experiments.py --scenarios s1 s2       # 특정 시나리오만
    python run_all_experiments.py --dry-run               # 실행 계획만 표시
    python run_all_experiments.py --csv                   # CSV도 생성

전체 실행 (논문 Table 2, 3):
    - ours × S1~S8 + case1, case2   (10개)
    - b1   × S1~S8                  (8개)
    - b2   × S1~S8                  (8개)
    - b3   × S1~S8                  (8개)
    합계 34개 실험
"""

from dotenv import load_dotenv
load_dotenv()

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SUMMARY_DIR = EXPERIMENTS_DIR / "summaries"

# ---- 실행 계획 ----

DEFAULT_PLAN = [
    # (system, scenario)
    *[("ours", f"s{i}") for i in range(1, 9)],
    ("ours", "case1"),
    ("ours", "case2"),
    *[("b1", f"s{i}") for i in range(1, 9)],
    *[("b2", f"s{i}") for i in range(1, 9)],
    *[("b3", f"s{i}") for i in range(1, 9)],
]


def run_one(system: str, scenario: str) -> dict:
    cmd = [PYTHON, "run_experiment.py", "--system", system, "--scenario", scenario, "--quiet"]

    start = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180,
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            return {
                "system": system, "scenario": scenario, "success": False,
                "error": f"Exit code {result.returncode}",
                "stderr_tail": (result.stderr or "")[-500:],
                "elapsed": round(elapsed, 2),
            }

        result_file = EXPERIMENTS_DIR / f"{system}_{scenario}.json"
        if not result_file.exists():
            return {
                "system": system, "scenario": scenario, "success": False,
                "error": "Result file not found", "elapsed": round(elapsed, 2),
            }

        data = json.loads(result_file.read_text(encoding="utf-8"))
        ev = data.get("evaluation", {})

        return {
            "system": system, "scenario": scenario, "success": True,
            "ac_at_1": ev.get("ac_at_1"),
            "ac_at_3": ev.get("ac_at_3"),
            "path_accuracy": ev.get("path_accuracy"),
            "fp_handled": ev.get("fp_handled"),
            "top_confidence": ev.get("top_confidence"),
            "verdict": ev.get("verdict"),
            "ground_truth": ev.get("ground_truth_root_cause"),
            "predicted_service": ev.get("predicted_top_cause_service"),
            "predicted_excerpt": ev.get("predicted_top_cause_excerpt"),
            "elapsed": round(elapsed, 2),
        }
    except subprocess.TimeoutExpired:
        return {"system": system, "scenario": scenario, "success": False,
                "error": "Timeout (3min)", "elapsed": 180.0}
    except Exception as e:
        return {"system": system, "scenario": scenario, "success": False,
                "error": str(e), "elapsed": round(time.time() - start, 2)}


def format_result(r: dict) -> str:
    if not r["success"]:
        return f"  [FAIL] {r['system']:5s} {r['scenario']:6s} | {r.get('error', 'unknown')}"
    ac1 = "✓" if r.get("ac_at_1") else "✗"
    ac3 = "✓" if r.get("ac_at_3") else "✗"
    path = "✓" if r.get("path_accuracy") else ("✗" if r.get("path_accuracy") is False else "-")
    fp = "✓" if r.get("fp_handled") else ("✗" if r.get("fp_handled") is False else "-")
    pred = r.get("predicted_service") or (r.get("predicted_excerpt") or "")[:25]
    gt = r.get("ground_truth") or "(none)"
    return (
        f"  {r['system']:5s} {r['scenario']:6s} | "
        f"AC@1:{ac1} AC@3:{ac3} Path:{path} FP:{fp} | "
        f"conf={r.get('top_confidence', 0):.2f} | "
        f"{r.get('elapsed', 0):5.1f}s | "
        f"GT:{gt:20s} Pred:{pred}"
    )


def compute_summary(results: list) -> dict:
    by_system = {}
    for r in results:
        if not r["success"]:
            continue
        sk = r["system"]
        if sk not in by_system:
            by_system[sk] = {
                "total": 0, "ac_at_1_count": 0, "ac_at_3_count": 0,
                "path_accuracy_count": 0, "path_accuracy_applicable": 0,
                "fp_handled_count": 0, "fp_applicable": 0, "total_elapsed": 0.0,
            }
        s = by_system[sk]
        s["total"] += 1
        s["total_elapsed"] += r.get("elapsed", 0)
        if r.get("ac_at_1"): s["ac_at_1_count"] += 1
        if r.get("ac_at_3"): s["ac_at_3_count"] += 1
        if r.get("path_accuracy") is not None:
            s["path_accuracy_applicable"] += 1
            if r.get("path_accuracy"): s["path_accuracy_count"] += 1
        if r.get("fp_handled") is not None:
            s["fp_applicable"] += 1
            if r.get("fp_handled"): s["fp_handled_count"] += 1

    for sk, s in by_system.items():
        t = max(s["total"], 1)
        s["ac_at_1_rate"] = round(s["ac_at_1_count"] / t, 3)
        s["ac_at_3_rate"] = round(s["ac_at_3_count"] / t, 3)
        pa = max(s["path_accuracy_applicable"], 1)
        s["path_accuracy_rate"] = round(s["path_accuracy_count"] / pa, 3)
        fp = max(s["fp_applicable"], 1)
        s["fp_handled_rate"] = round(s["fp_handled_count"] / fp, 3)
        s["avg_elapsed"] = round(s["total_elapsed"] / t, 2)

    return by_system


def export_csv(results: list, filepath: Path):
    """Export per-experiment results to CSV for paper tables."""
    fields = [
        "system", "scenario", "success", "ac_at_1", "ac_at_3",
        "path_accuracy", "fp_handled", "top_confidence", "verdict",
        "ground_truth", "predicted_service", "elapsed",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def export_summary_csv(summary: dict, filepath: Path):
    """Export per-system summary to CSV for paper Table 2."""
    fields = [
        "system", "total", "ac_at_1_count", "ac_at_1_rate",
        "ac_at_3_count", "ac_at_3_rate",
        "path_accuracy_count", "path_accuracy_rate",
        "fp_handled_count", "fp_handled_rate", "avg_elapsed",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for sk, s in summary.items():
            writer.writerow({"system": sk, **s})


def print_summary_table(results: list, summary: dict):
    print("\n" + "=" * 100)
    print("  EXPERIMENT SUMMARY")
    print("=" * 100)

    for sys_key in ("ours", "b1", "b2", "b3"):
        s = summary.get(sys_key)
        if not s:
            continue
        print(f"\n  System: {sys_key}")
        print(f"    Total experiments: {s['total']}")
        print(f"    AC@1:  {s['ac_at_1_count']}/{s['total']} ({s['ac_at_1_rate']:.1%})")
        print(f"    AC@3:  {s['ac_at_3_count']}/{s['total']} ({s['ac_at_3_rate']:.1%})")
        print(f"    Path:  {s['path_accuracy_count']}/{s['path_accuracy_applicable']} ({s['path_accuracy_rate']:.1%})")
        if s["fp_applicable"] > 0:
            print(f"    FP:    {s['fp_handled_count']}/{s['fp_applicable']} ({s['fp_handled_rate']:.1%})")
        print(f"    Avg elapsed: {s['avg_elapsed']:.1f}s")

    print("\n" + "=" * 100)
    print("  PER-SCENARIO RESULTS")
    print("=" * 100 + "\n")

    for r in results:
        print(format_result(r))

    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"\n  [!] {len(failures)} experiments FAILED:")
        for r in failures:
            print(f"      - {r['system']}/{r['scenario']}: {r.get('error')}")
    print("=" * 100)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--systems", nargs="+", choices=["ours", "b1", "b2", "b3"])
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--csv", action="store_true", help="Also export CSV files")
    args = parser.parse_args()

    plan = DEFAULT_PLAN[:]
    if args.systems:
        plan = [p for p in plan if p[0] in args.systems]
    if args.scenarios:
        plan = [p for p in plan if p[1] in args.scenarios]

    if not plan:
        print("  [!] No experiments match the filter.")
        return

    print(f"\n  Plan: {len(plan)} experiments")
    for i, (sys_key, sc) in enumerate(plan, 1):
        print(f"    {i:2d}. {sys_key} x {sc}")

    if args.dry_run:
        print("\n  [DRY RUN] Not executing.")
        return

    print(f"\n  Starting execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Expected duration: ~{len(plan) * 45 / 60:.1f} minutes\n")

    results = []
    overall_start = time.time()

    for i, (sys_key, sc) in enumerate(plan, 1):
        print(f"\n  [{i}/{len(plan)}] Running {sys_key} x {sc}...")
        r = run_one(sys_key, sc)
        results.append(r)
        print(format_result(r))

        if args.stop_on_fail and not r["success"]:
            print(f"\n  [!] Stopping on first failure ({sys_key}/{sc})")
            break

    total_elapsed = time.time() - overall_start

    summary = compute_summary(results)
    print_summary_table(results, summary)

    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # Save JSON
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "summary_by_system": summary,
        "results": results,
    }

    out_file = SUMMARY_DIR / f"summary_{timestamp}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_file = SUMMARY_DIR / "summary_latest.json"
    latest_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Summary saved: {out_file}")

    # CSV export
    if args.csv:
        csv_results = SUMMARY_DIR / f"results_{timestamp}.csv"
        csv_summary = SUMMARY_DIR / f"summary_table_{timestamp}.csv"
        export_csv(results, csv_results)
        export_summary_csv(summary, csv_summary)
        print(f"  CSV results: {csv_results}")
        print(f"  CSV summary: {csv_summary}")


if __name__ == "__main__":
    main()
