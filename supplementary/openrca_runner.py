"""
OpenRCA Bank Runner.

OpenRCA Bank 데이터셋의 136개 쿼리를 우리 멀티에이전트 프레임워크로 처리.

파이프라인:
1. Metric Agent (LLM): KPI 이상 탐지 → 의심 component + reason 추론
2. Log Agent (LLM): GC/앱 로그 분석 → JVM 관련 이상 탐지
3. RCA Agent (LLM): Metric + Log + Topology 종합 → 최종 component + reason + reasoning

사용법:
    # .env에 OPENAI_API_KEY 설정 필요
    uv run openrca_runner.py                           # 전체 136개
    uv run openrca_runner.py --start 0 --end 10        # 첫 10개만
    uv run openrca_runner.py --task task_3              # component만 묻는 쿼리
    uv run openrca_runner.py --task task_6              # component+reason 쿼리
    uv run openrca_runner.py --dry-run                  # 데이터 확인만
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 프로젝트 루트를 path에 추가 (common.llm_client 사용을 위해)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.llm_client import LLMClient
from openrca_adapter import (
    OpenRCABankAdapter,
    OpenRCAQuery,
    OpenRCAGroundTruth,
    MetricAnomaly,
    BANK_COMPONENTS,
    BANK_REASONS,
)

# NOTE: openrca_adapter.py should be the v2 version with golden KPI rules


# =========================================================================
# Agent Prompts
# =========================================================================

METRIC_AGENT_SYSTEM = """You are a metrics analysis expert for a banking microservice system.
You analyze KPI (Key Performance Indicator) anomalies to identify which component 
is experiencing problems and what type of resource issue is occurring.

The system has these components: {components}
Possible failure reasons: {reasons}

Given a list of metric anomalies (KPI values that deviate significantly from baseline),
identify the most likely root cause component and reason.

Respond ONLY with valid JSON."""

METRIC_AGENT_SCHEMA = """{
  "anomalous_components": ["<component>", ...],
  "primary_suspect": "<component with strongest anomaly signal>",
  "suspected_reason": "<one of the possible failure reasons>",
  "metric_evidence": "<brief description of which KPIs are anomalous and why>",
  "confidence": 0.XX
}"""

LOG_AGENT_SYSTEM = """You are a log analysis expert for a banking microservice system.
You analyze Java GC logs and application logs to identify JVM-related issues 
(memory leaks, GC storms, OOM errors).

The system has these components: {components}

If the logs show GC-related issues (frequent full GC, long GC pauses, heap exhaustion),
identify which component is affected.

If logs show no significant issues, say so clearly.

Respond ONLY with valid JSON."""

LOG_AGENT_SCHEMA = """{
  "jvm_issues_found": true_or_false,
  "affected_component": "<component or null>",
  "issue_type": "<JVM Out of Memory (OOM) Heap | high JVM CPU load | normal GC activity | null>",
  "log_evidence": "<brief description>",
  "confidence": 0.XX
}"""

RCA_AGENT_SYSTEM = """You are the senior root cause analysis expert for a banking microservice system.
Three analysis sources have examined the system:
1. Metric Agent — analyzed KPI anomalies (CPU, memory, disk, network metrics)
2. Log Agent — analyzed GC/application logs for JVM issues
3. Topology — the known service dependency structure

Your job is to SYNTHESIZE these into a final root cause determination.

CRITICAL RULES:
- "root cause component" MUST be EXACTLY one of: {components}
- "root cause reason" MUST be EXACTLY one of: {reasons}
- Match the EXACT strings above (case-sensitive, no modifications)
- If you're unsure, pick the component with the STRONGEST metric anomaly signal
- For the occurrence datetime, use the timestamp of the earliest anomaly for the root cause component

Respond ONLY with valid JSON."""

RCA_AGENT_SCHEMA = """{
  "root cause component": "<EXACT component name from the list>",
  "root cause reason": "<EXACT reason from the list>",
  "root cause occurrence datetime": "<YYYY-MM-DD HH:MM:SS format, UTC+8>",
  "confidence": 0.XX,
  "reasoning": "<paragraph explaining which evidence you used and why>"
}"""


# =========================================================================
# Runner
# =========================================================================

class OpenRCABankRunner:
    """OpenRCA Bank 데이터셋 실행기."""
    
    def __init__(self, dataset_path: str, output_dir: str = "openrca_results"):
        self.adapter = OpenRCABankAdapter(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm = LLMClient(log_dir="llm_logs/openrca")
    
    async def run_single(self, row_id: int) -> Dict[str, Any]:
        """단일 쿼리 처리."""
        query, gt = self.adapter.get_query_and_gt(row_id)
        
        start_time = time.time()
        
        # === Stage 1: Metric Agent ===
        metric_anomalies = self.adapter.load_metrics_for_window(
            query.date_str, query.start_ts, query.end_ts
        )
        
        # Network latency from traces (추가)
        try:
            latency_anomalies = self.adapter.detect_network_latency(
                query.date_str, query.start_ts, query.end_ts
            )
            metric_anomalies = metric_anomalies + latency_anomalies
        except Exception:
            latency_anomalies = []
        
        metric_result = await self._run_metric_agent(query, metric_anomalies)
        
        # === Stage 2: Log Agent ===
        logs = self.adapter.load_logs_for_window(
            query.date_str, query.start_ts, query.end_ts
        )
        log_result = await self._run_log_agent(query, logs)
        
        # === Stage 3: Topology ===
        topology = self.adapter.get_topology()
        
        # === Stage 4: RCA Agent (synthesis) ===
        rca_result = await self._run_rca_agent(
            query, metric_result, log_result, topology
        )
        
        elapsed = time.time() - start_time
        
        # OpenRCA 예측 형식으로 변환
        component = rca_result.get("root cause component", "")
        reason = rca_result.get("root cause reason", "")
        occurrence_dt = rca_result.get("root cause occurrence datetime", "")
        
        prediction = self.adapter.format_prediction(component, reason, occurrence_dt)
        
        return {
            "row_id": row_id,
            "task_index": query.task_index,
            "instruction": query.instruction,
            "prediction": prediction,
            "scoring_points": query.scoring_points,
            "ground_truth": {
                "component": gt.component,
                "reason": gt.reason,
                "datetime": gt.datetime_str,
            },
            "our_result": {
                "component": component,
                "reason": reason,
                "datetime": occurrence_dt,
                "confidence": rca_result.get("confidence", 0),
                "reasoning": rca_result.get("reasoning", ""),
            },
            "agent_outputs": {
                "metric_agent": metric_result,
                "log_agent": log_result,
                "rca_agent": rca_result,
            },
            "metric_anomalies_count": len(metric_anomalies),
            "logs_count": len(logs),
            "elapsed_seconds": round(elapsed, 2),
        }
    
    async def _run_metric_agent(
        self, query: OpenRCAQuery, anomalies: List[MetricAnomaly]
    ) -> Dict[str, Any]:
        """Metric Agent: KPI 이상 분석."""
        if not anomalies:
            return {
                "anomalous_components": [],
                "primary_suspect": None,
                "suspected_reason": None,
                "metric_evidence": "No metric anomalies detected in time window.",
                "confidence": 0.1,
            }
        
        # 이상 데이터를 프롬프트용 텍스트로 변환
        lines = [f"## Metric Anomalies in {query.start_dt} to {query.end_dt}"]
        lines.append(f"(Total {len(anomalies)} anomalies detected by Z-score analysis)\n")
        
        # Component별 그룹핑
        by_component: Dict[str, List[MetricAnomaly]] = {}
        for a in anomalies:
            by_component.setdefault(a.component, []).append(a)
        
        for comp, comp_anomalies in sorted(
            by_component.items(), key=lambda x: -max(a.severity for a in x[1])
        ):
            lines.append(f"### {comp}")
            for a in comp_anomalies[:5]:  # 컴포넌트당 최대 5개
                ts_str = datetime.fromtimestamp(a.timestamp).strftime("%H:%M:%S")
                lines.append(
                    f"  - {ts_str} | {a.kpi_name.split('_')[-1]}: {a.value:.2f} "
                    f"(threshold: {a.threshold}, severity: {a.severity:.2f}) "
                    f"→ suggests: {a.suspected_reason}"
                )
            lines.append("")
        
        user_prompt = "\n".join(lines)
        
        system = METRIC_AGENT_SYSTEM.format(
            components=", ".join(BANK_COMPONENTS),
            reasons=", ".join(BANK_REASONS),
        )
        
        result = await self.llm.call_json(
            agent_name="openrca_metric_agent",
            system_prompt=system,
            user_prompt=user_prompt,
            incident_id=f"OPENRCA-{query.row_id:03d}",
            schema_hint=METRIC_AGENT_SCHEMA,
        )
        
        return result if "_error" not in result else {
            "anomalous_components": list(by_component.keys())[:3],
            "primary_suspect": anomalies[0].component if anomalies else None,
            "suspected_reason": anomalies[0].suspected_reason if anomalies else None,
            "metric_evidence": f"LLM failed, using top Z-score anomaly: {anomalies[0].component}",
            "confidence": 0.3,
        }
    
    async def _run_log_agent(
        self, query: OpenRCAQuery, logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Log Agent: GC/앱 로그 분석."""
        if not logs:
            return {
                "jvm_issues_found": False,
                "affected_component": None,
                "issue_type": None,
                "log_evidence": "No logs found in time window.",
                "confidence": 0.1,
            }
        
        # GC 관련 로그만 필터 (토큰 절약)
        gc_logs = [l for l in logs if l.get("log_name") == "gc"]
        other_logs = [l for l in logs if l.get("log_name") != "gc"]
        
        lines = [f"## Log Evidence in {query.start_dt} to {query.end_dt}"]
        
        if gc_logs:
            lines.append(f"\n### GC Logs ({len(gc_logs)} entries, showing first 20)")
            for l in gc_logs[:20]:
                ts_str = datetime.fromtimestamp(l["timestamp"]).strftime("%H:%M:%S")
                lines.append(f"  - {ts_str} | {l['component']} | {l['value'][:150]}")
        
        if other_logs:
            lines.append(f"\n### Application Logs ({len(other_logs)} entries, showing first 10)")
            for l in other_logs[:10]:
                ts_str = datetime.fromtimestamp(l["timestamp"]).strftime("%H:%M:%S")
                lines.append(f"  - {ts_str} | {l['component']} | {l['log_name']} | {l['value'][:150]}")
        
        user_prompt = "\n".join(lines)
        
        system = LOG_AGENT_SYSTEM.format(components=", ".join(BANK_COMPONENTS))
        
        result = await self.llm.call_json(
            agent_name="openrca_log_agent",
            system_prompt=system,
            user_prompt=user_prompt,
            incident_id=f"OPENRCA-{query.row_id:03d}",
            schema_hint=LOG_AGENT_SCHEMA,
        )
        
        return result if "_error" not in result else {
            "jvm_issues_found": False,
            "affected_component": None,
            "issue_type": None,
            "log_evidence": f"LLM failed: {result.get('_error')}",
            "confidence": 0.1,
        }
    
    async def _run_rca_agent(
        self,
        query: OpenRCAQuery,
        metric_result: Dict[str, Any],
        log_result: Dict[str, Any],
        topology: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """RCA Agent: 종합 판정."""
        lines = []
        lines.append(f"## Incident: {query.instruction[:200]}")
        lines.append(f"- Time range: {query.start_dt} to {query.end_dt}")
        lines.append(f"- Number of failures: {query.num_failures}")
        lines.append("")
        
        # Metric Agent 결과
        lines.append("## Evidence 1: Metric Agent Analysis")
        lines.append(f"- Primary suspect: {metric_result.get('primary_suspect', 'unknown')}")
        lines.append(f"- Suspected reason: {metric_result.get('suspected_reason', 'unknown')}")
        lines.append(f"- Anomalous components: {metric_result.get('anomalous_components', [])}")
        lines.append(f"- Evidence: {metric_result.get('metric_evidence', 'N/A')}")
        lines.append(f"- Confidence: {metric_result.get('confidence', 0)}")
        lines.append("")
        
        # Log Agent 결과
        lines.append("## Evidence 2: Log Agent Analysis")
        lines.append(f"- JVM issues found: {log_result.get('jvm_issues_found', False)}")
        lines.append(f"- Affected component: {log_result.get('affected_component', 'none')}")
        lines.append(f"- Issue type: {log_result.get('issue_type', 'none')}")
        lines.append(f"- Evidence: {log_result.get('log_evidence', 'N/A')}")
        lines.append("")
        
        # Topology
        lines.append("## Evidence 3: System Topology")
        for comp, deps in topology.items():
            if deps:
                lines.append(f"  {comp} → {deps}")
        lines.append("")
        
        lines.append("## Task")
        lines.append("Synthesize all evidence to identify:")
        lines.append("1. The root cause component (MUST be an exact match from the component list)")
        lines.append("2. The root cause reason (MUST be an exact match from the reason list)")
        lines.append("3. The occurrence datetime (when the anomaly started, in YYYY-MM-DD HH:MM:SS UTC+8)")
        lines.append("4. Your reasoning explaining the evidence chain")
        
        user_prompt = "\n".join(lines)
        
        system = RCA_AGENT_SYSTEM.format(
            components=", ".join(BANK_COMPONENTS),
            reasons=" | ".join(BANK_REASONS),
        )
        
        result = await self.llm.call_json(
            agent_name="openrca_rca_agent",
            system_prompt=system,
            user_prompt=user_prompt,
            incident_id=f"OPENRCA-{query.row_id:03d}",
            schema_hint=RCA_AGENT_SCHEMA,
        )
        
        if "_error" in result:
            # Fallback: Metric Agent 결과 직접 사용
            return {
                "root cause component": metric_result.get("primary_suspect", ""),
                "root cause reason": metric_result.get("suspected_reason", ""),
                "root cause occurrence datetime": "",
                "confidence": 0.2,
                "reasoning": f"LLM failed: {result.get('_error')}. Using metric agent fallback.",
            }
        
        return result
    
    async def run_batch(
        self,
        start: int = 0,
        end: Optional[int] = None,
        task_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """배치 실행."""
        queries = self.adapter.queries
        
        if task_filter:
            indices = [q.row_id for q in queries if q.task_index == task_filter]
        else:
            indices = list(range(len(queries)))
        
        if end is not None:
            indices = [i for i in indices if start <= i < end]
        else:
            indices = [i for i in indices if i >= start]
        
        print(f"\n  OpenRCA Bank Runner")
        print(f"  Queries to process: {len(indices)}")
        print(f"  Estimated time: ~{len(indices) * 15 / 60:.1f} minutes")
        print(f"  Estimated cost: ~${len(indices) * 0.002:.2f}")
        print()
        
        results = []
        overall_start = time.time()
        
        for i, idx in enumerate(indices):
            print(f"  [{i+1}/{len(indices)}] Processing query {idx} ({queries[idx].task_index})...", end=" ")
            
            try:
                result = await self.run_single(idx)
                results.append(result)
                
                # 간단한 결과 표시
                gt_comp = result["ground_truth"]["component"]
                pred_comp = result["our_result"]["component"]
                gt_reason = result["ground_truth"]["reason"]
                pred_reason = result["our_result"]["reason"]
                
                comp_ok = "✓" if pred_comp == gt_comp else "✗"
                reason_ok = "✓" if pred_reason == gt_reason else "✗"
                
                print(f"Comp:{comp_ok} Reason:{reason_ok} | {result['elapsed_seconds']:.1f}s | GT:{gt_comp}/{gt_reason[:20]} Pred:{pred_comp}/{pred_reason[:20]}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "row_id": idx,
                    "error": str(e),
                })
        
        total_elapsed = time.time() - overall_start
        
        # 결과 저장
        self._save_results(results, total_elapsed)
        self._save_prediction_csv(results)
        self._print_summary(results, total_elapsed)
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]], elapsed: float):
        """전체 결과 JSON 저장."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = self.output_dir / f"openrca_bank_{timestamp}.json"
        
        # LLM 통계
        stats = self.llm.get_stats()
        
        out_file.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "total_elapsed": round(elapsed, 2),
            "total_queries": len(results),
            "llm_stats": stats,
            "results": results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        
        print(f"\n  Results saved: {out_file}")
    
    def _save_prediction_csv(self, results: List[Dict[str, Any]]):
        """OpenRCA 평가 호환 CSV 저장."""
        import csv
        
        csv_file = self.output_dir / "prediction_bank.csv"
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["instruction", "prediction", "row_id", "task_index"])
            writer.writeheader()
            
            for r in results:
                if "error" in r:
                    continue
                writer.writerow({
                    "instruction": r.get("instruction", ""),
                    "prediction": r.get("prediction", ""),
                    "row_id": r.get("row_id", ""),
                    "task_index": r.get("task_index", ""),
                })
        
        print(f"  Prediction CSV: {csv_file}")
        print(f"  (Can evaluate with: python -m main.evaluate -p {csv_file} -q dataset/Bank/query.csv -r openrca_results/report.csv)")
    
    def _print_summary(self, results: List[Dict[str, Any]], elapsed: float):
        """결과 요약 출력."""
        valid = [r for r in results if "error" not in r]
        
        if not valid:
            print("\n  No valid results.")
            return
        
        # Component accuracy
        comp_correct = sum(
            1 for r in valid 
            if r["our_result"]["component"] == r["ground_truth"]["component"]
        )
        
        # Reason accuracy
        reason_correct = sum(
            1 for r in valid 
            if r["our_result"]["reason"] == r["ground_truth"]["reason"]
        )
        
        # Both correct
        both_correct = sum(
            1 for r in valid
            if r["our_result"]["component"] == r["ground_truth"]["component"]
            and r["our_result"]["reason"] == r["ground_truth"]["reason"]
        )
        
        total = len(valid)
        
        print(f"\n  {'='*60}")
        print(f"  OpenRCA Bank Results Summary")
        print(f"  {'='*60}")
        print(f"  Total queries: {total}")
        print(f"  Component accuracy: {comp_correct}/{total} ({comp_correct/total:.1%})")
        print(f"  Reason accuracy:    {reason_correct}/{total} ({reason_correct/total:.1%})")
        print(f"  Both correct:       {both_correct}/{total} ({both_correct/total:.1%})")
        print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  LLM stats: {self.llm.get_stats()}")
        print(f"  {'='*60}")
        
        # Task별 분류
        by_task: Dict[str, List] = {}
        for r in valid:
            task = r["task_index"]
            by_task.setdefault(task, []).append(r)
        
        print(f"\n  Per-task breakdown:")
        for task in sorted(by_task.keys()):
            task_results = by_task[task]
            task_comp = sum(1 for r in task_results if r["our_result"]["component"] == r["ground_truth"]["component"])
            task_reason = sum(1 for r in task_results if r["our_result"]["reason"] == r["ground_truth"]["reason"])
            print(f"    {task}: {len(task_results)} queries | Comp: {task_comp}/{len(task_results)} | Reason: {task_reason}/{len(task_results)}")


async def main():
    parser = argparse.ArgumentParser(description="OpenRCA Bank Runner")
    parser.add_argument("--dataset", default="dataset/Bank",
                       help="Path to OpenRCA Bank dataset (relative to openrca repo)")
    parser.add_argument("--start", type=int, default=0, help="Start query index")
    parser.add_argument("--end", type=int, default=None, help="End query index (exclusive)")
    parser.add_argument("--task", type=str, default=None,
                       help="Filter by task type (task_1 ~ task_7)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    parser.add_argument("--output", default="openrca_results", help="Output directory")
    args = parser.parse_args()
    
    # OpenRCA 데이터 경로 탐색
    dataset_path = args.dataset
    if not Path(dataset_path).exists():
        # openrca 레포 기준 상대경로 시도
        alt = Path(os.getenv("OPENRCA_PATH", "")) / "dataset" / "Bank"
        if alt.exists():
            dataset_path = str(alt)
        else:
            print(f"  ERROR: Dataset not found at {dataset_path}")
            print(f"  Set OPENRCA_PATH env var or use --dataset")
            return
    
    runner = OpenRCABankRunner(dataset_path, args.output)
    
    print(f"  Dataset: {dataset_path}")
    print(f"  Queries: {len(runner.adapter.queries)}")
    print(f"  Ground truths: {len(runner.adapter.ground_truths)}")
    
    if args.dry_run:
        print("\n  [DRY RUN] Not executing.")
        # 쿼리 분포 표시
        from collections import Counter
        tasks = Counter(q.task_index for q in runner.adapter.queries)
        for task, count in sorted(tasks.items()):
            print(f"    {task}: {count} queries")
        return
    
    await runner.run_batch(
        start=args.start,
        end=args.end,
        task_filter=args.task,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
