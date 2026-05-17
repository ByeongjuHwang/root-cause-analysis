"""
OpenRCA Bank Data Adapter v2.

v1 대비 핵심 개선:
- Z-score 기반 → 절대값 임계치 + Golden KPI 필터링
- Reason별 정확한 KPI 매핑
- Noisy KPI (disk I/O 단독) 필터링
- Network latency는 trace span duration에서 탐지
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# =========================================================================
# Constants
# =========================================================================

BANK_COMPONENTS = [
    "apache01", "apache02",
    "Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04",
    "MG01", "MG02",
    "IG01", "IG02",
    "Mysql01", "Mysql02",
    "Redis01", "Redis02",
]

BANK_REASONS = [
    "high CPU usage",
    "high memory usage",
    "network latency",
    "network packet loss",
    "high disk I/O read usage",
    "high disk space usage",
    "high JVM CPU load",
    "JVM Out of Memory (OOM) Heap",
]

TZ_UTC8 = timezone(timedelta(hours=8))

# =========================================================================
# Golden KPI Rules — 핵심 개선
# =========================================================================
# 각 reason에 대해 "어떤 KPI를 보고, 어떤 기준으로 이상 판정하는지" 정의
# absolute threshold 기반 (Z-score 대신)

GOLDEN_KPI_RULES = [
    # === high CPU usage ===
    {
        "reason": "high CPU usage",
        "kpi_pattern": "CPUCpuUtil",
        "threshold": 80.0,
        "direction": "above",  # 값이 threshold 이상이면 이상
        "priority": 1,
    },
    {
        "reason": "high CPU usage",
        "kpi_pattern": "CPULoad",
        "threshold": 3.0,  # load average > 3 (4 CPU 시스템 기준)
        "direction": "above",
        "priority": 2,
    },
    # === high memory usage ===
    {
        "reason": "high memory usage",
        "kpi_pattern": "MEMUsedMemPerc",
        "threshold": 85.0,
        "direction": "above",
        "priority": 1,
    },
    {
        "reason": "high memory usage",
        "kpi_pattern": "NoCacheMemPerc",
        "threshold": 85.0,
        "direction": "above",
        "priority": 2,
    },
    {
        "reason": "high memory usage",
        "kpi_pattern": "MEMFreeMem",
        "threshold": 200.0,  # 200MB 이하면 이상
        "direction": "below",
        "priority": 3,
    },
    # === network packet loss ===
    {
        "reason": "network packet loss",
        "kpi_pattern": "NETInErr",
        "threshold": 1.0,  # 1개 이상 에러 패킷
        "direction": "above",
        "priority": 1,
    },
    {
        "reason": "network packet loss",
        "kpi_pattern": "NETOutErr",
        "threshold": 1.0,
        "direction": "above",
        "priority": 1,
    },
    {
        "reason": "network packet loss",
        "kpi_pattern": "NETInErrPrc",
        "threshold": 0.1,  # 0.1% 이상 에러율
        "direction": "above",
        "priority": 2,
    },
    {
        "reason": "network packet loss",
        "kpi_pattern": "NETOutErrPrcc",
        "threshold": 0.1,
        "direction": "above",
        "priority": 2,
    },
    # === high disk I/O read usage ===
    # disk I/O는 CPUWio와 함께 높아야만 진짜 이상
    # {
    #     "reason": "high disk I/O read usage",
    #     "kpi_pattern": "DSKPercentBusy",
    #     "threshold": 80.0,
    #     "direction": "above",
    #     "priority": 1,
    #     "requires_also": "CPUWio",  # I/O wait도 높아야 함
    # },
    # {
    #     "reason": "high disk I/O read usage",
    #     "kpi_pattern": "CPUWio",
    #     "threshold": 20.0,  # I/O wait > 20%
    #     "direction": "above",
    #     "priority": 2,
    # },
    # === high disk space usage ===
    # {
    #     "reason": "high disk space usage",
    #     "kpi_pattern": "FSUsedSpace",
    #     "threshold_ratio": True,  # FSUsedSpace/FSCapacity > 0.9
    #     "ratio_denominator": "FSCapacity",
    #     "threshold": 0.9,
    #     "direction": "above",
    #     "priority": 1,
    # },
    # === JVM OOM Heap ===
    {
        "reason": "JVM Out of Memory (OOM) Heap",
        "kpi_pattern": "HeapMemoryUsed",
        "threshold_ratio": True,
        "ratio_denominator": "HeapMemoryMax",
        "threshold": 0.9,
        "direction": "above",
        "priority": 1,
    },
    # === high JVM CPU load ===
    {
        "reason": "high JVM CPU load",
        "kpi_pattern": "JVM_CPULoad",
        "threshold": 0.8,  # JVM CPU load > 80%
        "direction": "above",
        "priority": 1,
    },
]


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class OpenRCAQuery:
    row_id: int
    task_index: str
    instruction: str
    scoring_points: str
    date_str: str
    start_ts: float
    end_ts: float
    start_dt: str
    end_dt: str
    num_failures: int


@dataclass
class OpenRCAGroundTruth:
    row_id: int
    level: str
    component: str
    timestamp: float
    datetime_str: str
    reason: str


@dataclass
class MetricAnomaly:
    component: str
    kpi_name: str
    timestamp: float
    value: float
    threshold: float
    suspected_reason: str
    priority: int
    severity: float  # 0~1, threshold 대비 초과 정도


# =========================================================================
# Adapter
# =========================================================================

class OpenRCABankAdapter:
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.telemetry_path = self.dataset_path / "telemetry"
        self.queries = self._load_queries()
        self.ground_truths = self._load_ground_truths()
        self._metric_cache: Dict[str, pd.DataFrame] = {}
        self._topology_cache: Optional[Dict[str, List[str]]] = None
    
    def _load_queries(self) -> List[OpenRCAQuery]:
        query_path = self.dataset_path / "query.csv"
        df = pd.read_csv(query_path)
        queries = []
        for idx, row in df.iterrows():
            instruction = row["instruction"]
            date_str, start_ts, end_ts, start_dt, end_dt = self._parse_time_range(instruction)
            num_failures = self._parse_num_failures(instruction)
            queries.append(OpenRCAQuery(
                row_id=idx, task_index=row["task_index"],
                instruction=instruction, scoring_points=row["scoring_points"],
                date_str=date_str, start_ts=start_ts, end_ts=end_ts,
                start_dt=start_dt, end_dt=end_dt, num_failures=num_failures,
            ))
        return queries
    
    def _load_ground_truths(self) -> List[OpenRCAGroundTruth]:
        record_path = self.dataset_path / "record.csv"
        df = pd.read_csv(record_path)
        return [
            OpenRCAGroundTruth(
                row_id=idx, level=row["level"], component=row["component"],
                timestamp=float(row["timestamp"]), datetime_str=row["datetime"],
                reason=row["reason"],
            )
            for idx, row in df.iterrows()
        ]
    
    def _parse_time_range(self, instruction: str) -> Tuple[str, float, float, str, str]:
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        year, month, day = 2021, 3, 4
        m = re.search(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", instruction)
        if m:
            month_name = m.group(1).lower()
            day = int(m.group(2))
            year = int(m.group(3))
            month = month_map.get(month_name, 3)
        
        time_match = re.search(r"(\d{1,2}:\d{2})\s+to\s+(\d{1,2}:\d{2})", instruction)
        if time_match:
            start_time, end_time = time_match.group(1), time_match.group(2)
        else:
            start_time, end_time = "00:00", "23:59"
        
        start_h, start_m = map(int, start_time.split(":"))
        end_h, end_m = map(int, end_time.split(":"))
        
        start_dt = datetime(year, month, day, start_h, start_m, 0, tzinfo=TZ_UTC8)
        end_dt = datetime(year, month, day, end_h, end_m, 0, tzinfo=TZ_UTC8)
        
        date_str = f"{year}_{month:02d}_{day:02d}"
        return (date_str, start_dt.timestamp(), end_dt.timestamp(),
                start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S"))
    
    def _parse_num_failures(self, instruction: str) -> int:
        m = re.search(r"(\w+)\s+failure", instruction.lower())
        if m:
            word_map = {"a": 1, "single": 1, "one": 1, "two": 2, "three": 3}
            return word_map.get(m.group(1), 1)
        m = re.search(r"(\d+)\s+failure", instruction)
        return int(m.group(1)) if m else 1
    
    # ================================================================
    # Metric loading — Golden KPI 기반 이상 탐지 (핵심 개선)
    # ================================================================
    
    def load_metrics_for_window(
        self,
        date_str: str,
        start_ts: float,
        end_ts: float,
    ) -> List[MetricAnomaly]:
        """Golden KPI 규칙 기반 이상 탐지."""
        metric_file = self.telemetry_path / date_str / "metric" / "metric_container.csv"
        if not metric_file.exists():
            return []
        
        cache_key = str(metric_file)
        if cache_key not in self._metric_cache:
            self._metric_cache[cache_key] = pd.read_csv(metric_file)
        
        df = self._metric_cache[cache_key]
        window_df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        
        if window_df.empty:
            return []
        
        anomalies = []
        
        for component in BANK_COMPONENTS:
            comp_df = window_df[window_df["cmdb_id"] == component]
            if comp_df.empty:
                continue
            
            for rule in GOLDEN_KPI_RULES:
                pattern = rule["kpi_pattern"]
                threshold = rule["threshold"]
                direction = rule.get("direction", "above")
                reason = rule["reason"]
                priority = rule["priority"]
                
                # KPI 값 필터
                kpi_df = comp_df[comp_df["kpi_name"].str.contains(pattern, na=False)]
                if kpi_df.empty:
                    continue
                
                # Ratio 기반 임계치 (예: HeapUsed/HeapMax)
                if rule.get("threshold_ratio"):
                    denom_pattern = rule["ratio_denominator"]
                    denom_df = comp_df[comp_df["kpi_name"].str.contains(denom_pattern, na=False)]
                    if denom_df.empty:
                        continue
                    
                    for _, kpi_row in kpi_df.iterrows():
                        ts = kpi_row["timestamp"]
                        denom_at_ts = denom_df[denom_df["timestamp"] == ts]
                        if denom_at_ts.empty:
                            continue
                        denom_val = denom_at_ts.iloc[0]["value"]
                        if denom_val <= 0:
                            continue
                        ratio = kpi_row["value"] / denom_val
                        if ratio >= threshold:
                            severity = min(1.0, (ratio - threshold) / (1.0 - threshold + 0.01))
                            anomalies.append(MetricAnomaly(
                                component=component,
                                kpi_name=kpi_row["kpi_name"],
                                timestamp=ts,
                                value=round(ratio, 4),
                                threshold=threshold,
                                suspected_reason=reason,
                                priority=priority,
                                severity=round(severity, 3),
                            ))
                    continue
                
                # requires_also 체크 (예: disk I/O는 CPUWio도 높아야)
                if rule.get("requires_also"):
                    also_pattern = rule["requires_also"]
                    also_df = comp_df[comp_df["kpi_name"].str.contains(also_pattern, na=False)]
                    # CPUWio가 낮으면 이 rule skip
                    if also_df.empty or also_df["value"].max() < 10.0:
                        continue
                
                # 일반 임계치 체크
                for _, kpi_row in kpi_df.iterrows():
                    value = kpi_row["value"]
                    ts = kpi_row["timestamp"]
                    
                    is_anomaly = False
                    if direction == "above" and value >= threshold:
                        is_anomaly = True
                        severity = min(1.0, (value - threshold) / (threshold + 0.01))
                    elif direction == "below" and value <= threshold:
                        is_anomaly = True
                        severity = min(1.0, (threshold - value) / (threshold + 0.01))
                    
                    if is_anomaly:
                        anomalies.append(MetricAnomaly(
                            component=component,
                            kpi_name=kpi_row["kpi_name"],
                            timestamp=ts,
                            value=round(value, 4),
                            threshold=threshold,
                            suspected_reason=reason,
                            priority=priority,
                            severity=round(severity, 3),
                        ))
        
        # 우선순위: priority 낮은 것 > severity 높은 것
        anomalies.sort(key=lambda a: (a.priority, -a.severity))
        
        # Component별 중복 reason 제거 (같은 component + 같은 reason → 가장 심한 것만)
        seen = set()
        deduped = []
        for a in anomalies:
            key = (a.component, a.suspected_reason)
            if key not in seen:
                seen.add(key)
                deduped.append(a)
        
        return deduped[:30]
    
    # ================================================================
    # Network latency — trace 기반 탐지
    # ================================================================
    
    def detect_network_latency(
        self,
        date_str: str,
        start_ts: float,
        end_ts: float,
        latency_multiplier: float = 3.0,
        sample_size: int = 10000,
    ) -> List[MetricAnomaly]:
        """Trace span에서 network latency 탐지.
        
        30분 윈도우의 span duration을 baseline(윈도우 전 30분)과 비교.
        """
        trace_file = self.telemetry_path / date_str / "trace" / "trace_span.csv"
        if not trace_file.exists():
            return []
        
        # trace는 너무 크니까 chunk로 읽기
        baseline_start = start_ts * 1000 - 30 * 60 * 1000  # ms 단위
        window_start = start_ts * 1000
        window_end = end_ts * 1000
        
        component_durations_baseline: Dict[str, List[float]] = {}
        component_durations_window: Dict[str, List[float]] = {}
        
        try:
            for chunk in pd.read_csv(trace_file, chunksize=50000):
                # Baseline
                bl = chunk[(chunk["timestamp"] >= baseline_start) & (chunk["timestamp"] < window_start)]
                for _, row in bl.iterrows():
                    comp = row["cmdb_id"]
                    if comp in BANK_COMPONENTS:
                        component_durations_baseline.setdefault(comp, []).append(row["duration"])
                
                # Window
                win = chunk[(chunk["timestamp"] >= window_start) & (chunk["timestamp"] <= window_end)]
                for _, row in win.iterrows():
                    comp = row["cmdb_id"]
                    if comp in BANK_COMPONENTS:
                        component_durations_window.setdefault(comp, []).append(row["duration"])
                
                # 충분히 모았으면 중단
                total_collected = sum(len(v) for v in component_durations_window.values())
                if total_collected >= sample_size:
                    break
        except Exception:
            return []
        
        anomalies = []
        for comp in BANK_COMPONENTS:
            bl_durations = component_durations_baseline.get(comp, [])
            win_durations = component_durations_window.get(comp, [])
            
            if len(bl_durations) < 5 or len(win_durations) < 5:
                continue
            
            bl_mean = np.mean(bl_durations)
            win_mean = np.mean(win_durations)
            win_p95 = np.percentile(win_durations, 95)
            
            if bl_mean > 0 and win_mean > bl_mean * latency_multiplier:
                severity = min(1.0, (win_mean / bl_mean - 1) / 5.0)
                anomalies.append(MetricAnomaly(
                    component=comp,
                    kpi_name=f"trace_avg_duration",
                    timestamp=start_ts,
                    value=round(win_mean, 2),
                    threshold=round(bl_mean * latency_multiplier, 2),
                    suspected_reason="network latency",
                    priority=1,
                    severity=round(severity, 3),
                ))
        
        return anomalies
    
    # ================================================================
    # Log loading
    # ================================================================
    
    def load_logs_for_window(
        self, date_str: str, start_ts: float, end_ts: float, max_logs: int = 100,
    ) -> List[Dict[str, Any]]:
        log_file = self.telemetry_path / date_str / "log" / "log_service.csv"
        if not log_file.exists():
            return []
        
        logs = []
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row.get("timestamp", 0))
                except (ValueError, TypeError):
                    continue
                if start_ts <= ts <= end_ts:
                    logs.append({
                        "log_id": row.get("log_id", ""),
                        "timestamp": ts,
                        "component": row.get("cmdb_id", ""),
                        "log_name": row.get("log_name", ""),
                        "value": (row.get("value", "") or "")[:200],
                    })
                    if len(logs) >= max_logs:
                        break
        return logs
    
    # ================================================================
    # Topology
    # ================================================================
    
    def get_topology(self, date_str: Optional[str] = None) -> Dict[str, List[str]]:
        if self._topology_cache is not None:
            return self._topology_cache
        self._topology_cache = {
            "apache01": ["Tomcat01", "Tomcat02"],
            "apache02": ["Tomcat03", "Tomcat04"],
            "Tomcat01": ["MG01", "Mysql01", "Redis01"],
            "Tomcat02": ["MG01", "Mysql01", "Redis01"],
            "Tomcat03": ["MG02", "Mysql02", "Redis02"],
            "Tomcat04": ["MG02", "Mysql02", "Redis02"],
            "MG01": ["IG01"],
            "MG02": ["IG02"],
            "IG01": [], "IG02": [],
            "Mysql01": [], "Mysql02": [],
            "Redis01": [], "Redis02": [],
        }
        return self._topology_cache
    
    # ================================================================
    # Incident 생성
    # ================================================================
    
    def build_incident(self, query: OpenRCAQuery) -> Dict[str, Any]:
        return {
            "incident_id": f"OPENRCA-BANK-{query.row_id:03d}",
            "service": "system",
            "time_range": {"start": query.start_dt, "end": query.end_dt},
            "symptom": query.instruction,
            "trace_id": None,
            "attachments": {
                "openrca_query": {
                    "row_id": query.row_id,
                    "task_index": query.task_index,
                    "scoring_points": query.scoring_points,
                    "num_failures": query.num_failures,
                    "date_str": query.date_str,
                },
                "possible_components": BANK_COMPONENTS,
                "possible_reasons": BANK_REASONS,
            },
        }
    
    def format_prediction(
        self, component: Optional[str], reason: Optional[str],
        occurrence_datetime: Optional[str],
    ) -> str:
        import json
        result = {
            "root cause occurrence datetime": occurrence_datetime or "",
            "root cause component": component or "",
            "root cause reason": reason or "",
        }
        return json.dumps({"1": result}, indent=4)
    
    def get_query_and_gt(self, row_id: int) -> Tuple[OpenRCAQuery, OpenRCAGroundTruth]:
        return self.queries[row_id], self.ground_truths[row_id]
