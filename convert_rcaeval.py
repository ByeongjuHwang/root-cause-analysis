#!/usr/bin/env python3
"""
convert_rcaeval.py — RCAEval RE2-OB logs.csv를 프레임워크 LogRecord JSONL로 변환.

RCAEval RE2-OB 실제 logs.csv 포맷:
    time, timestamp, container_name, message, level, req_path, error, cluster_id, log_template
    
    - timestamp: 나노초 Unix timestamp (19자리)
    - container_name: 서비스명
    - level: info, debug, warn, error (이미 존재)

폴더 구조:
    RE2-OB/
    ├── checkoutservice_cpu/      ← {service}_{fault}
    │   ├── 1/  (instance)
    │   │   ├── logs.csv
    │   │   ├── inject_time.txt   ← 초 단위 Unix timestamp
    │   │   └── ...
    │   ├── 2/
    │   └── 3/

사용법:
    python convert_rcaeval.py <instance_folder> <output.jsonl>
    
    예: python convert_rcaeval.py RE2-OB/checkoutservice_cpu/1 output.jsonl
"""

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ============ 타임스탬프 변환 ============

def ts_to_iso(ts) -> str:
    """Unix timestamp(초/밀리초/마이크로초/나노초 자동 감지) → ISO 8601 UTC."""
    try:
        ts = float(ts)
    except (ValueError, TypeError):
        return str(ts)
    
    abs_ts = abs(ts)
    if abs_ts > 1e17:        # 나노초 (19자리)
        seconds = ts / 1e9
    elif abs_ts > 1e14:      # 마이크로초 (16자리)
        seconds = ts / 1e6
    elif abs_ts > 1e11:      # 밀리초 (13자리)
        seconds = ts / 1e3
    else:                     # 초 (10자리)
        seconds = ts
    
    try:
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    except (ValueError, OSError):
        return str(ts)


# ============ 레벨 정규화 ============

def normalize_level(level_str: str) -> str:
    """RCAEval의 level을 프레임워크 레벨로 정규화."""
    if not level_str:
        return "INFO"
    level_upper = level_str.strip().upper()
    if level_upper in ("ERROR", "ERR", "FATAL", "PANIC"):
        return "ERROR"
    if level_upper in ("WARN", "WARNING"):
        return "WARN"
    if level_upper in ("DEBUG", "TRACE"):
        return "DEBUG"
    return "INFO"


# ============ 필드 추출 ============

def extract_status_code(message: str, error: str, req_path: str) -> Optional[int]:
    text = f"{message} {error} {req_path}"
    
    m = re.search(r'\b(?:status|code)[=:\s]+(\d{3})\b', text, re.IGNORECASE)
    if m:
        code = int(m.group(1))
        if 100 <= code <= 599:
            return code
    
    m = re.search(r'\bHTTP\s+(\d{3})\b', text)
    if m:
        return int(m.group(1))
    
    m = re.search(r'(?:^|\s)(5\d{2})(?:\s|$)', text)
    if m:
        return int(m.group(1))
    
    return None


def extract_latency(message: str, error: str) -> Optional[int]:
    text = f"{message} {error}"
    
    m = re.search(r'\b(\d+)\s*ms\b', text)
    if m:
        val = int(m.group(1))
        if 0 < val < 600000:
            return val
    
    m = re.search(r'latency[=:\s]+(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    
    m = re.search(r'took\s+(\d+)\s*ms', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    
    return None


def extract_error_type(message: str, error: str, level: str) -> Optional[str]:
    if level not in ("ERROR", "WARN"):
        return None
    
    text = f"{message} {error}".lower()
    
    patterns = [
        (r'connection\s+refused', 'CONNECTION_REFUSED'),
        (r'connection\s+reset', 'CONNECTION_RESET'),
        (r'deadline\s+exceeded', 'DEADLINE_EXCEEDED'),
        (r'timeout|timed\s+out', 'TIMEOUT'),
        (r'out\s+of\s+memory|\boom\b', 'OOM_ERROR'),
        (r'memory\s+(limit|exceeded)', 'MEMORY_LIMIT'),
        (r'cpu\s+(limit|throttl)', 'CPU_THROTTLE'),
        (r'disk\s+(full|space|quota)', 'DISK_FULL'),
        (r'unavailable', 'SERVICE_UNAVAILABLE'),
        (r'unreachable', 'UNREACHABLE'),
        (r'circuit\s+break', 'CIRCUIT_OPEN'),
        (r'retry|retries', 'RETRY_FAILURE'),
        (r'unauthoriz(ed|e)|authentication', 'AUTH_ERROR'),
        (r'permission|forbidden', 'PERMISSION_DENIED'),
        (r'not\s+found|404', 'NOT_FOUND'),
        (r'bad\s+request|400', 'BAD_REQUEST'),
        (r'internal\s+(server\s+)?error|500', 'INTERNAL_ERROR'),
        (r'502|bad\s+gateway', 'BAD_GATEWAY'),
        (r'503|service\s+unavailable', 'SERVICE_UNAVAILABLE'),
        (r'504|gateway\s+timeout', 'GATEWAY_TIMEOUT'),
    ]
    
    for pattern, etype in patterns:
        if re.search(pattern, text):
            return etype
    
    m = re.search(r'code\s*=\s*(\w+)', message)
    if m:
        return f"GRPC_{m.group(1).upper()}"
    
    return "UNKNOWN_ERROR" if level == "ERROR" else None


KNOWN_SERVICES = [
    "frontend", "cartservice", "productcatalogservice", "currencyservice",
    "paymentservice", "shippingservice", "emailservice", "checkoutservice",
    "recommendationservice", "adservice", "redis-cart", "loadgenerator"
]


def extract_upstream(message: str, error: str, service: str) -> Optional[str]:
    text = f"{message} {error}".lower()
    service_lower = service.lower()
    
    for svc in KNOWN_SERVICES:
        if svc.lower() == service_lower:
            continue
        if svc.lower() in text:
            return svc
    return None


# ============ 메인 변환 ============

def convert_instance(instance_folder: Path, output_path: Path,
                     case_folder_name: str) -> dict:
    """단일 instance 폴더를 JSONL로 변환."""
    logs_csv = instance_folder / "logs.csv"
    inject_time_file = instance_folder / "inject_time.txt"
    
    if not logs_csv.exists():
        raise FileNotFoundError(f"logs.csv not found in {instance_folder}")
    
    # inject_time 읽기 (초 단위 Unix timestamp, 10자리)
    inject_time_unix = None
    inject_time_iso = None
    if inject_time_file.exists():
        content = inject_time_file.read_text().strip()
        try:
            inject_time_unix = float(content)
            inject_time_iso = ts_to_iso(inject_time_unix)
        except ValueError:
            print(f"Warning: could not parse inject_time: {content}")
    
    # Ground truth: case_folder_name = "checkoutservice_cpu"
    parts = case_folder_name.rsplit("_", 1)
    if len(parts) == 2:
        gt_service = parts[0]
        gt_fault = parts[1]
    else:
        gt_service = case_folder_name
        gt_fault = "unknown"
    
    instance_num = instance_folder.name
    
    # logs.csv 변환
    converted = []
    level_counts = {}
    service_counts = {}
    
    with open(logs_csv, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # RCAEval 실제 컬럼명
            ts = row.get("timestamp") or row.get("time")
            service = row.get("container_name") or row.get("service")
            message = row.get("message") or ""
            raw_level = row.get("level") or "info"
            req_path = row.get("req_path") or ""
            error = row.get("error") or ""
            
            if not ts or not service:
                continue
            
            level = normalize_level(raw_level)
            
            # message에 error와 req_path 통합
            full_message = message
            if error and error.strip():
                full_message = f"{message} [error: {error}]"
            if req_path and req_path.strip():
                full_message = f"{full_message} [path: {req_path}]"
            
            record = {
                "timestamp": ts_to_iso(ts),
                "service": service,
                "level": level,
                "trace_id": None,
                "message": full_message[:500],
                "upstream": extract_upstream(message, error, service),
                "status_code": extract_status_code(message, error, req_path),
                "latency_ms": extract_latency(message, error),
                "error_type": extract_error_type(message, error, level),
            }
            
            converted.append(record)
            level_counts[level] = level_counts.get(level, 0) + 1
            service_counts[service] = service_counts.get(service, 0) + 1
    
    # 시간순 정렬
    converted.sort(key=lambda r: r["timestamp"])
    
    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # 메타
    meta_path = output_path.with_suffix('.meta.json')
    meta = {
        "case_folder": case_folder_name,
        "instance": instance_num,
        "case_name": f"{case_folder_name}_{instance_num}",
        "ground_truth_service": gt_service,
        "ground_truth_fault": gt_fault,
        "inject_time_unix": inject_time_unix,
        "inject_time_iso": inject_time_iso,
        "total_logs": len(converted),
        "level_counts": level_counts,
        "service_counts": service_counts,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    
    return meta


def main():
    if len(sys.argv) < 3:
        print("사용법: python convert_rcaeval.py <instance_folder> <output.jsonl>")
        print("예: python convert_rcaeval.py RE2-OB/checkoutservice_cpu/1 output.jsonl")
        sys.exit(1)
    
    instance_folder = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2])
    
    if not instance_folder.is_dir():
        print(f"ERROR: {instance_folder} is not a directory")
        sys.exit(1)
    
    case_folder_name = instance_folder.parent.name
    
    meta = convert_instance(instance_folder, output_path, case_folder_name)
    
    print(f"=== 변환 완료: {meta['case_name']} ===")
    print(f"  Ground Truth: {meta['ground_truth_service']} ({meta['ground_truth_fault']})")
    print(f"  Instance: {meta['instance']}")
    print(f"  Inject Time: {meta['inject_time_iso']} (Unix: {meta['inject_time_unix']})")
    print(f"  총 로그: {meta['total_logs']}건")
    print(f"  레벨 분포: {meta['level_counts']}")
    print(f"  서비스별 (상위 5개):")
    svcs = sorted(meta['service_counts'].items(), key=lambda x: -x[1])[:5]
    for svc, cnt in svcs:
        print(f"    {svc}: {cnt}건")
    print(f"  출력: {output_path}")
    print(f"  메타: {output_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
