"""
run_parallel.py — B2 (No-MCP Parallel Pipeline) legacy standalone runner.

⚠️ DEPRECATED in favor of the unified experiment pipeline:

    python run_experiment.py --system b2 --scenario s1

This script is retained for backward compatibility. For all paper
experiments, use run_experiment.py / run_all_experiments.py.

본 프레임워크와 동일한 에이전트 프로세스를 기동하되,
B2 오케스트레이터(병렬 독립 호출)를 사용한다.
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
PORT_BASE = 21000  # B2 전용 포트 (Ours=18000, B3=19000, B1=20000)

# B2는 verifier 포함 — 하지만 오케스트레이터가 호출하지 않음
# (에이전트 프로세스는 기동하되 오케스트레이터가 사용 여부 결정)
SERVICES = [
    ("orchestrator-b2", "agents.orchestrator.main_parallel", PORT_BASE),
    ("log-agent", "agents.log_agent.main", PORT_BASE + 1),
    ("topology-agent", "agents.topology_agent.main", PORT_BASE + 2),
    ("rca-agent", "agents.rca_agent.main", PORT_BASE + 4),
]

SCENARIO_CONFIGS = {
    "s1": {"log_file": None, "incident_file": "demo_incident.json", "topology_file": None},
    "s2": {
        "log_file": "scenarios/s2_queue_backlog.jsonl",
        "incident": {
            "incident_id": "INC-S2-QUEUE", "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "order placement requests failing with 504 timeout errors",
            "trace_id": "trace-001",
        },
    },
    "s3": {
        "log_file": "scenarios/s3_slow_query.jsonl",
        "incident": {
            "incident_id": "INC-S3-SLOWQUERY", "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "product search requests timing out with 504 errors",
            "trace_id": "trace-003",
        },
    },
    "s4": {
        "log_file": "scenarios/s4_disk_full.jsonl",
        "incident": {
            "incident_id": "INC-S4-DISKFULL", "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "checkout requests failing with 500 errors from order-service",
            "trace_id": "trace-001",
        },
    },
    "s5": {
        "log_file": "scenarios/s5_false_positive.jsonl",
        "incident": {
            "incident_id": "INC-S5-FALSEPOS", "service": "api-gateway",
            "time_range": {"start": "2026-03-24T13:02:30+09:00", "end": "2026-03-24T13:03:20+09:00"},
            "symptom": "brief latency spike observed on api-gateway, self-recovered",
            "trace_id": "trace-001",
        },
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


def wait_for_health(name: str, port: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited before healthy")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.4)
    raise RuntimeError(f"{name} timed out")


def main() -> int:
    scenario_key = "s1"
    is_case_study = False
    for i, arg in enumerate(sys.argv):
        if arg == "--scenario" and i + 1 < len(sys.argv):
            scenario_key = sys.argv[i + 1]
        if arg == "--case-study":
            is_case_study = True

    if scenario_key not in SCENARIO_CONFIGS:
        print(f"Unknown scenario: {scenario_key}")
        return 1

    config = SCENARIO_CONFIGS[scenario_key]

    print(f"\n{'=' * 60}")
    print(f"  B2 Parallel Independent Multi-Agent — {scenario_key}")
    print(f"{'=' * 60}\n")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
    env["LOG_AGENT_URL"] = f"http://127.0.0.1:{PORT_BASE + 1}"
    env["TOPOLOGY_AGENT_URL"] = f"http://127.0.0.1:{PORT_BASE + 2}"
    env["VERIFIER_AGENT_URL"] = f"http://127.0.0.1:{PORT_BASE + 3}"
    env["RCA_AGENT_URL"] = f"http://127.0.0.1:{PORT_BASE + 4}"

    if config.get("log_file"):
        env["OBSERVABILITY_LOG_FILE"] = str(PROJECT_ROOT / config["log_file"])
    if config.get("topology_file"):
        env["ARCHITECTURE_TOPOLOGY_FILE"] = str(PROJECT_ROOT / config["topology_file"])
    elif is_case_study:
        env["ARCHITECTURE_TOPOLOGY_FILE"] = str(PROJECT_ROOT / "case_study_topology.json")

    if "incident" in config:
        incident = config["incident"]
    else:
        incident_path = PROJECT_ROOT / config.get("incident_file", "demo_incident.json")
        incident = json.loads(incident_path.read_text(encoding="utf-8"))

    if (is_case_study or config.get("case_study")) and incident.get("attachments"):
        incident["attachments"]["topology_file"] = str(PROJECT_ROOT / "case_study_topology.json")

    processes = []
    start_time = time.time()

    try:
        for name, module, port in SERVICES:
            if is_port_in_use(port):
                raise RuntimeError(f"Port {port} in use")
            proc = subprocess.Popen(
                [PYTHON, "-m", module], cwd=PROJECT_ROOT,
                env={**env, "PORT": str(port), "PYTHONUNBUFFERED": "1"},
                stdout=None, stderr=None, text=True,
            )
            wait_for_health(name, port, proc)
            print(f"  [OK] {name} on :{port}")
            processes.append(proc)

        response = httpx.post(f"http://127.0.0.1:{PORT_BASE}/analyze", json=incident, timeout=120.0)
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"  B2 RESULT (Parallel)")
        print(f"{'=' * 60}")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 결과 저장
        experiments_dir = PROJECT_ROOT / "experiments"
        experiments_dir.mkdir(exist_ok=True)
        out_file = experiments_dir / f"b2_{scenario_key}.json"
        output = {
            "system": "b2",
            "system_name": "B2 (Parallel Independent Multi-Agent)",
            "scenario": scenario_key,
            "elapsed_seconds": round(elapsed, 2),
            "result": result,
        }
        out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n결과 저장: {out_file}")
        print(f"소요 시간: {elapsed:.1f}초")
        return 0

    finally:
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except Exception:
                    proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
