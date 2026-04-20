#!/usr/bin/env python3
"""
show_results.py — 기존 실험 결과에서 논문용 요약 표를 생성.

실험을 다시 실행하지 않고 experiments/ 폴더의 기존 JSON 결과를 읽어
콘솔 표, CSV, LaTeX 형태로 출력합니다.

사용법:
    python show_results.py              # 콘솔 요약
    python show_results.py --csv        # CSV도 생성
    python show_results.py --latex      # LaTeX 표 출력
    python show_results.py --per-scenario  # 시나리오별 상세
"""

import argparse
import csv
import json
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"

SYSTEM_NAMES = {
    "ours": "Ours (Proposed)",
    "b1": "B1 (Monolithic)",
    "b2": "B2 (No-MCP)",
    "b3": "B3 (No-Verifier)",
}

SYSTEM_ORDER = ["ours", "b1", "b2", "b3"]


def load_results():
    """Load all experiment result JSONs."""
    results = []
    for f in sorted(EXPERIMENTS_DIR.glob("*.json")):
        if f.name.startswith("_") or f.parent.name == "summaries":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "evaluation" in data and "system" in data:
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def aggregate(results):
    by_system = {}
    for r in results:
        sk = r["system"]
        ev = r.get("evaluation", {})
        if sk not in by_system:
            by_system[sk] = {
                "total": 0, "ac1": 0, "ac3": 0,
                "path_ok": 0, "path_total": 0,
                "fp_ok": 0, "fp_total": 0,
                "elapsed": [],
            }
        s = by_system[sk]
        s["total"] += 1
        if ev.get("ac_at_1"):
            s["ac1"] += 1
        if ev.get("ac_at_3"):
            s["ac3"] += 1
        if ev.get("path_accuracy") is not None:
            s["path_total"] += 1
            if ev["path_accuracy"]:
                s["path_ok"] += 1
        if ev.get("fp_handled") is not None:
            s["fp_total"] += 1
            if ev["fp_handled"]:
                s["fp_ok"] += 1
        s["elapsed"].append(ev.get("elapsed_seconds", 0))
    return by_system


def print_console(results, summary):
    print("\n" + "=" * 75)
    print("  논문 실험 결과 요약 (Paper Experiment Summary)")
    print("=" * 75)

    header = f"{'System':<20} {'AC@1':>8} {'AC@3':>8} {'Path':>8} {'FP':>6} {'Avg(s)':>8}"
    print(f"\n{header}")
    print("-" * 60)

    for sk in SYSTEM_ORDER:
        s = summary.get(sk)
        if not s:
            continue
        t = max(s["total"], 1)
        pt = max(s["path_total"], 1)
        ft = max(s["fp_total"], 1)
        ac1 = s["ac1"] / t * 100
        ac3 = s["ac3"] / t * 100
        path = s["path_ok"] / pt * 100
        fp_str = f"{s['fp_ok']}/{s['fp_total']}" if s["fp_total"] > 0 else "-"
        avg = sum(s["elapsed"]) / t
        name = SYSTEM_NAMES.get(sk, sk)
        print(f"{name:<20} {ac1:>7.1f}% {ac3:>7.1f}% {path:>7.1f}% {fp_str:>6} {avg:>7.1f}")

    print()


def print_per_scenario(results):
    print("\n" + "=" * 95)
    print("  시나리오별 상세 결과 (Per-Scenario Detail)")
    print("=" * 95)

    header = (
        f"{'System':<6} {'Scenario':<8} {'AC@1':>5} {'AC@3':>5} {'Path':>5} "
        f"{'Conf':>6} {'Verdict':<10} {'GT':<18} {'Predicted':<18}"
    )
    print(f"\n{header}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: (SYSTEM_ORDER.index(x["system"]) if x["system"] in SYSTEM_ORDER else 99, x["scenario"])):
        ev = r.get("evaluation", {})
        ac1 = "✓" if ev.get("ac_at_1") else "✗"
        ac3 = "✓" if ev.get("ac_at_3") else "✗"
        path = "✓" if ev.get("path_accuracy") else ("✗" if ev.get("path_accuracy") is False else "-")
        conf = ev.get("top_confidence", 0)
        verdict = ev.get("verdict", "?")
        gt = ev.get("ground_truth_root_cause") or "(none)"
        pred = ev.get("predicted_top_cause_service") or ev.get("predicted_top_cause_excerpt", "")[:18]
        print(
            f"{r['system']:<6} {r['scenario']:<8} {ac1:>5} {ac3:>5} {path:>5} "
            f"{conf:>6.2f} {verdict:<10} {gt:<18} {pred:<18}"
        )


def print_latex(summary):
    print("\n% === LaTeX Table (copy to paper) ===")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Comparison of system variants on synthetic scenarios (S1--S8).}")
    print(r"\label{tab:main-results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"System & AC@1 (\%) & AC@3 (\%) & Path Acc. (\%) & Avg. Time (s) \\")
    print(r"\midrule")

    latex_names = {
        "ours": r"\textbf{Proposed}",
        "b1": "B1 (Monolithic)",
        "b2": "B2 (No-MCP)",
        "b3": "B3 (No-Verifier)",
    }

    for sk in SYSTEM_ORDER:
        s = summary.get(sk)
        if not s:
            continue
        t = max(s["total"], 1)
        pt = max(s["path_total"], 1)
        ac1 = s["ac1"] / t * 100
        ac3 = s["ac3"] / t * 100
        path = s["path_ok"] / pt * 100
        avg = sum(s["elapsed"]) / t
        name = latex_names.get(sk, sk)
        print(f"{name} & {ac1:.1f} & {ac3:.1f} & {path:.1f} & {avg:.1f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def export_csv(results, summary):
    ts = "latest"

    # Per-experiment CSV
    csv_path = EXPERIMENTS_DIR / "summaries" / f"results_{ts}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["system", "scenario", "ac_at_1", "ac_at_3", "path_accuracy",
              "fp_handled", "top_confidence", "verdict", "ground_truth_root_cause",
              "predicted_top_cause_service", "elapsed_seconds"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {"system": r["system"], "scenario": r["scenario"]}
            row.update(r.get("evaluation", {}))
            w.writerow(row)
    print(f"  CSV (per-experiment): {csv_path}")

    # Summary CSV
    sum_path = EXPERIMENTS_DIR / "summaries" / f"summary_table_{ts}.csv"
    fields2 = ["system", "total", "ac1", "ac1_rate", "ac3", "ac3_rate",
               "path_ok", "path_total", "path_rate", "avg_elapsed"]
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields2, extrasaction="ignore")
        w.writeheader()
        for sk in SYSTEM_ORDER:
            s = summary.get(sk)
            if not s:
                continue
            t = max(s["total"], 1)
            pt = max(s["path_total"], 1)
            w.writerow({
                "system": sk, "total": s["total"],
                "ac1": s["ac1"], "ac1_rate": round(s["ac1"] / t, 3),
                "ac3": s["ac3"], "ac3_rate": round(s["ac3"] / t, 3),
                "path_ok": s["path_ok"], "path_total": s["path_total"],
                "path_rate": round(s["path_ok"] / pt, 3),
                "avg_elapsed": round(sum(s["elapsed"]) / t, 2),
            })
    print(f"  CSV (summary table): {sum_path}")


def main():
    parser = argparse.ArgumentParser(description="Show experiment results")
    parser.add_argument("--csv", action="store_true", help="Export CSV files")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table")
    parser.add_argument("--per-scenario", action="store_true", help="Show per-scenario detail")
    args = parser.parse_args()

    results = load_results()
    if not results:
        print("No experiment results found in experiments/")
        print("Run experiments first: python run_all_experiments.py")
        return

    print(f"Loaded {len(results)} experiment results")

    summary = aggregate(results)
    print_console(results, summary)

    if args.per_scenario:
        print_per_scenario(results)

    if args.latex:
        print_latex(summary)

    if args.csv:
        export_csv(results, summary)


if __name__ == "__main__":
    main()
