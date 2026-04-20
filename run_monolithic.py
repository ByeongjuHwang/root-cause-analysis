"""
run_monolithic.py — B1 (Monolithic Agent) legacy standalone runner.

⚠️ DEPRECATED in favor of the unified experiment pipeline:

    python run_experiment.py --system b1 --scenario s1

This script is retained for backward compatibility and for invoking the
monolithic agent outside the main experiment harness (e.g. manual probing).
For all paper experiments, use run_experiment.py / run_all_experiments.py,
which share SYSTEMS, SCENARIOS and evaluate_prediction from experiment_core.

단일 HTTP 서버만 기동. A2A 통신 없음, MCP 서버 없음.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
STARTUP_TIMEOUT = 30.0
PORT = 20000  # B1 전용 포트 (Ours 18000, B3 19000과 분리)

# 시나리오 정의 (run_experiment.py와 동일)
SCENARIO_CONFIGS = {
    "s1": {
        "log_file": None,
        "incident_file": "demo_incident.json",
        "topology_file": None,
    },
    "s2": {
        "log_file": "scenarios/s2_queue_backlog.jsonl",
        "incident": {
            "incident_id": "INC-S2-QUEUE",
            "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "order placement requests failing with 504 timeout errors",
            "trace_id": "trace-001",
        },
        "topology_file": None,
    },
    "s3": {
        "log_file": "scenarios/s3_slow_query.jsonl",
        "incident": {
            "incident_id": "INC-S3-SLOWQUERY",
            "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "product search requests timing out with 504 errors",
            "trace_id": "trace-003",
        },
        "topology_file": None,
    },
    "s4": {
        "log_file": "scenarios/s4_disk_full.jsonl",
        "incident": {
            "incident_id": "INC-S4-DISKFULL",
            "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "checkout requests failing with 500 errors from order-service",
            "trace_id": "trace-001",
        },
        "topology_file": None,
    },
    "s5": {
        "log_file": "scenarios/s5_false_positive.jsonl",
        "incident": {
            "incident_id": "INC-S5-FALSEPOS",
            "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "brief latency spike observed on api-gateway, self-recovered",
            "trace_id": "trace-001",
        },
        "topology_file": None,
    },
    "s6": {
        "log_file": "scenarios/s6_noisy_db.jsonl",
        "incident": {
            "incident_id": "INC-S6-NOISY",
            "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "login requests failing with 502, multiple services reporting errors simultaneously",
            "trace_id": "trace-001",
        },
        "description": "S6: Noisy DB Fault (misleading errors in unrelated services)",
        "ground_truth_root_cause": "user-db",
        "ground_truth_path": ["user-db", "auth-service", "api-gateway"],
    },
    "s7": {
        "log_file": "scenarios/s7_concurrent_faults.jsonl",
        "incident": {
            "incident_id": "INC-S7-CONCURRENT",
            "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "multiple services failing simultaneously: auth 502, orders 500",
            "trace_id": "trace-001",
        },
        "description": "S7: Concurrent Dual Faults (user-db + worker-service)",
        "ground_truth_root_cause": "user-db",
        "ground_truth_path": ["user-db", "auth-service", "api-gateway"],
    },
    "s8": {
        "log_file": "scenarios/s8_partial_observability.jsonl",
        "incident": {
            "incident_id": "INC-S8-PARTIAL",
            "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "login requests failing with 502, auth-service reporting database errors",
            "trace_id": "trace-001",
        },
        "description": "S8: Partial Observability (root cause logs missing)",
        "ground_truth_root_cause": "user-db",
        "ground_truth_path": ["user-db", "auth-service", "api-gateway"],
    },
    "case1": {
        "log_file": "case_study_logs.jsonl",
        "incident_file": "case_study_incident.json",
        "topology_file": "case_study_topology.json",
        "case_study": True,
    },
    "case2": {
        "log_file": "case_study_logs.jsonl",
        "incident_file": "case_study_incident_case2.json",
        "topology_file": "case_study_topology.json",
        "case_study": True,
    },
}


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    # Parse args
    scenario_key = "s1"
    is_case_study = False

    for i, arg in enumerate(sys.argv):
        if arg == "--scenario" and i + 1 < len(sys.argv):
            scenario_key = sys.argv[i + 1]
        if arg == "--case-study":
            is_case_study = True
        if arg == "--incident" and i + 1 < len(sys.argv):
            # 직접 incident file 지정 (s1 기본 대체)
            SCENARIO_CONFIGS["s1"]["incident_file"] = sys.argv[i + 1]

    if scenario_key not in SCENARIO_CONFIGS:
        print(f"Unknown scenario: {scenario_key}. Available: {list(SCENARIO_CONFIGS.keys())}")
        return 1

    config = SCENARIO_CONFIGS[scenario_key]

    print(f"\n{'=' * 60}")
    print(f"  B1 Monolithic Agent — Scenario: {scenario_key}")
    print(f"{'=' * 60}\n")

    # Environment
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
    env["PORT"] = str(PORT)
    env["PYTHONUNBUFFERED"] = "1"

    if config.get("log_file"):
        env["OBSERVABILITY_LOG_FILE"] = str(PROJECT_ROOT / config["log_file"])
    if config.get("topology_file"):
        env["ARCHITECTURE_TOPOLOGY_FILE"] = str(PROJECT_ROOT / config["topology_file"])

    # Build incident
    if "incident" in config:
        incident = config["incident"]
    else:
        incident_path = PROJECT_ROOT / config.get("incident_file", "demo_incident.json")
        incident = json.loads(incident_path.read_text(encoding="utf-8"))

    # Start single process (no A2A, no multiple agents)
    if is_port_in_use(PORT):
        print(f"ERROR: Port {PORT} already in use.")
        return 1

    proc = subprocess.Popen(
        [PYTHON, "-m", "agents.monolithic.main"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=None, stderr=None, text=True,
    )

    try:
        # Wait for health
        deadline = time.time() + STARTUP_TIMEOUT
        url = f"http://127.0.0.1:{PORT}/health"
        started = False
        while time.time() < deadline:
            if proc.poll() is not None:
                print(f"ERROR: B1 monolithic server exited unexpectedly.")
                return 1
            try:
                if httpx.get(url, timeout=1.0).status_code == 200:
                    started = True
                    break
            except Exception:
                pass
            time.sleep(0.4)

        if not started:
            print(f"ERROR: B1 monolithic server did not start within {STARTUP_TIMEOUT}s")
            return 1

        print(f"[OK] B1 monolithic server on port {PORT}")
        print(f"Sending: {incident.get('incident_id', 'unknown')} -> {incident.get('service', 'unknown')}")

        start_time = time.time()
        response = httpx.post(f"http://127.0.0.1:{PORT}/analyze", json=incident, timeout=120.0)
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"  B1 RESULT (Monolithic)")
        print(f"{'=' * 60}")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Save
        experiments_dir = PROJECT_ROOT / "experiments"
        experiments_dir.mkdir(exist_ok=True)
        out_file = experiments_dir / f"b1_{scenario_key}.json"

        output = {
            "system": "b1",
            "system_name": "B1 (Monolithic Agent)",
            "scenario": scenario_key,
            "elapsed_seconds": round(elapsed, 2),
            "result": result,
        }
        out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n결과 저장: {out_file}")
        print(f"소요 시간: {elapsed:.1f}초")

        return 0

    finally:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
