#!/usr/bin/env python3
"""
convert_case_study.py — Case Study 익명화 데이터를 프레임워크 LogRecord 포맷으로 변환.

사용법:
    python convert_case_study.py <input_jsonl> <output_jsonl> [--window-center TIMESTAMP] [--window-minutes N]

입력 포맷  (anonymize.py 출력):
    {"timestamp", "service", "host", "content", "fields": {...}}

출력 포맷  (프레임워크 sample_logs.jsonl과 동일):
    {"timestamp", "service", "level", "trace_id", "message",
     "upstream", "status_code", "latency_ms", "error_type"}
"""

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# === 서비스 간 알려진 관계 (Case Study topology 기반) ===
# app-deployer가 device-state-gateway를 호출 — 이것만 알고 있음
KNOWN_UPSTREAM_MAP = {
    "app-deployer": "device-state-gateway",
}


def detect_level(record: dict) -> str:
    """로그 레벨 추론. content 키워드 + fields 기반."""
    content = record.get("content", "").lower()
    fields = record.get("fields", {})

    # 명시적 실패 상태
    deploy_state = (fields.get("deployState") or "").upper()
    exec_status = (fields.get("executionStatus") or "").upper()
    deploy_sub = (fields.get("deploySubState") or "").lower()

    if deploy_state == "FAILED" or exec_status in ("FAILED", "FAILURE"):
        return "ERROR"
    if "timeout" in deploy_sub or "failed" in deploy_sub:
        return "ERROR"
    if exec_status == "IGNORED":
        return "WARN"

    # content 키워드 기반
    if any(kw in content for kw in ["error", "exception", "failed", "failure", "timeout", "refused"]):
        return "ERROR"
    if any(kw in content for kw in ["warn", "retry", "slow", "degraded"]):
        return "WARN"

    return "INFO"


def detect_error_type(record: dict) -> Optional[str]:
    """에러 유형 추론."""
    fields = record.get("fields", {})
    content = record.get("content", "").lower()

    deploy_sub = fields.get("deploySubState") or ""
    error_code = fields.get("errorCode")
    exec_status = (fields.get("executionStatus") or "").upper()

    # deploySubState 기반 (app-deployer 전용)
    if "Timeout" in deploy_sub:
        return deploy_sub  # AppInstallTimeout, AppUninstallTimeout
    if "Failed" in deploy_sub:
        return deploy_sub

    # errorCode 기반 (device-job-worker-svc)
    if error_code and error_code != "null":
        return error_code

    # executionStatus 기반
    if exec_status in ("FAILED", "FAILURE"):
        return "EXECUTION_FAILED"
    if exec_status == "IGNORED":
        return "EXECUTION_IGNORED"

    # content 키워드
    if "websocket" in content:
        return "WEBSOCKET_ERROR"
    if "tunnel" in content:
        return "TUNNEL_ERROR"
    if "timeout" in content:
        return "TIMEOUT"
    if "connection" in content and ("refused" in content or "error" in content):
        return "CONNECTION_ERROR"

    return None


def detect_status_code(record: dict) -> Optional[int]:
    """HTTP 상태 코드 추출."""
    content = record.get("content", "")

    # status=NNN 패턴
    m = re.search(r'status[=:](\d{3})', content)
    if m:
        code = int(m.group(1))
        if 100 <= code <= 599:
            return code

    return None


def detect_latency(record: dict) -> Optional[int]:
    """응답 지연 시간(ms) 추출."""
    content = record.get("content", "")

    # elapsed=NNN 패턴
    m = re.search(r'elapsed[=:](\d+)', content)
    if m:
        return int(m.group(1))

    # timeInMillis=NNN 패턴
    m = re.search(r'timeInMillis[=:](\d+)', content)
    if m:
        return int(m.group(1))

    return None


def detect_upstream(record: dict) -> Optional[str]:
    """upstream 서비스 추론."""
    service = record.get("service", "")
    return KNOWN_UPSTREAM_MAP.get(service)


def truncate_message(content: str, max_len: int = 300) -> str:
    """content를 LogRecord.message 수준으로 축약."""
    # 첫 줄 또는 첫 300자
    first_line = content.split("\n")[0].strip()
    if len(first_line) > max_len:
        return first_line[:max_len] + "..."
    return first_line


def convert_record(record: dict) -> dict:
    """한 레코드를 LogRecord 포맷으로 변환."""
    fields = record.get("fields", {})

    return {
        "timestamp": record["timestamp"],
        "service": record["service"],
        "level": detect_level(record),
        "trace_id": fields.get("traceId"),
        "message": truncate_message(record.get("content", "")),
        "upstream": detect_upstream(record),
        "status_code": detect_status_code(record),
        "latency_ms": detect_latency(record),
        "error_type": detect_error_type(record),
    }


def parse_ts(ts_str: str) -> datetime:
    s = ts_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if "+" not in s and "-" not in s[10:]:
        s = s + "+00:00"
    return datetime.fromisoformat(s)


def main():
    if len(sys.argv) < 3:
        print("사용법: python convert_case_study.py <input.jsonl> <output.jsonl> [--window-center TS] [--window-minutes N]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # 선택적 시간 윈도우 필터
    center = None
    window_minutes = None
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--window-center" and i + 1 < len(args):
            center = parse_ts(args[i + 1])
            i += 2
        elif args[i] == "--window-minutes" and i + 1 < len(args):
            window_minutes = int(args[i + 1])
            i += 2
        else:
            i += 1

    # 읽기
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"입력: {len(records)}건")

    # 시간 윈도우 필터 (선택적)
    if center and window_minutes:
        delta = timedelta(minutes=window_minutes)
        start = center - delta
        end = center + delta
        records = [r for r in records if start <= parse_ts(r["timestamp"]) <= end]
        print(f"윈도우 필터 (±{window_minutes}분): {len(records)}건")

    # 변환
    converted = []
    level_counts = {"ERROR": 0, "WARN": 0, "INFO": 0}
    error_type_counts = {}

    for r in records:
        c = convert_record(r)
        converted.append(c)
        level_counts[c["level"]] = level_counts.get(c["level"], 0) + 1
        if c["error_type"]:
            error_type_counts[c["error_type"]] = error_type_counts.get(c["error_type"], 0) + 1

    # 시간순 정렬
    converted.sort(key=lambda r: r["timestamp"])

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n출력: {len(converted)}건 -> {output_path}")
    print(f"레벨 분포: {level_counts}")
    if error_type_counts:
        print(f"에러 유형 (상위 10개):")
        for et, cnt in sorted(error_type_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {et}: {cnt}건")

    # 서비스별 카운트
    svc_counts = {}
    for r in converted:
        svc_counts[r["service"]] = svc_counts.get(r["service"], 0) + 1
    print(f"서비스별:")
    for svc, cnt in sorted(svc_counts.items(), key=lambda x: -x[1]):
        print(f"  {svc}: {cnt}건")


if __name__ == "__main__":
    main()