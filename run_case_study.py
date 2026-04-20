#!/usr/bin/env python3
"""
run_case_study.py — Case Study legacy standalone runner.

⚠️ DEPRECATED in favor of the unified experiment pipeline:

    python run_experiment.py --system ours --scenario case1
    python run_experiment.py --system ours --scenario case2

Retained for backward compatibility.

기존 run_demo.py와 동일하게 5개 에이전트를 띄우되,
환경변수를 Case Study 데이터로 설정하여 실행한다.

사용법:
    uv run run_case_study.py --case 1       (Case 1: AppInstallTimeout)
    uv run run_case_study.py --case 2       (Case 2: AppUninstallTimeout)

사전 준비:
    1. convert_case_study.py로 JSONL 변환 완료
    2. case_study_logs.jsonl이 프로젝트 루트에 존재
    3. case_study_topology.json이 프로젝트 루트에 존재
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

SERVICES = [
    ("orchestrator", "agents.orchestrator.main", 18000),
    ("log-agent", "agents.log_agent.main", 18001),
    ("topology-agent", "agents.topology_agent.main", 18002),
    ("verifier-agent", "agents.verifier_agent.main", 18003),
    ("rca-agent", "agents.rca_agent.main", 18004),
]

CASE_CONFIGS = {
    1: {
        "incident_file": "case_study_incident.json",
        "description": "Case 1: AppInstallTimeout",
    },
    2: {
        "incident_file": "case_study_incident_case2.json",
        "description": "Case 2: AppUninstallTimeout",
    },
}


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def ensure_port_available(name: str, port: int) -> None:
    if is_port_in_use(port):
        raise RuntimeError(
            f"Cannot start {name}: port {port} is already in use. "
            f"Stop existing process or wait a moment."
        )


def wait_for_health(name: str, port: int, proc: subprocess.Popen, timeout: float = STARTUP_TIMEOUT) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"

    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"{name} exited unexpectedly.\n"
                f"stdout:\n{stdout or '(empty)'}\n\n"
                f"stderr:\n{stderr or '(empty)'}"
            )
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.4)

    raise RuntimeError(f"{name} did not start within {timeout:.0f}s")


def start_service(name: str, module: str, port: int, env: dict) -> subprocess.Popen:
    ensure_port_available(name, port)
    proc = subprocess.Popen(
        [PYTHON, "-m", module],
        cwd=PROJECT_ROOT,
        env={**env, "PORT": str(port), "PYTHONUNBUFFERED": "1"},
        stdout=None,
        stderr=None,
        text=True,
    )
    wait_for_health(name, port, proc)
    print(f"[OK] {name} started on port {port}")
    return proc


def main() -> int:
    # Parse --case argument
    case_num = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--case" and i + 1 < len(sys.argv):
            case_num = int(sys.argv[i + 1])

    if case_num not in CASE_CONFIGS:
        print(f"Unknown case: {case_num}. Available: {list(CASE_CONFIGS.keys())}")
        return 1

    config = CASE_CONFIGS[case_num]
    print(f"\n{'=' * 60}")
    print(f"  Running {config['description']}")
    print(f"{'=' * 60}\n")

    # Verify required files exist
    log_file = PROJECT_ROOT / "case_study_logs.jsonl"
    topo_file = PROJECT_ROOT / "case_study_topology.json"
    incident_file = PROJECT_ROOT / config["incident_file"]

    for f in [log_file, topo_file, incident_file]:
        if not f.exists():
            print(f"ERROR: Required file not found: {f}")
            print(f"Run convert_case_study.py first to generate case_study_logs.jsonl")
            return 1

    processes: list[subprocess.Popen] = []
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    # Agent URLs
    env["LOG_AGENT_URL"] = "http://127.0.0.1:18001"
    env["TOPOLOGY_AGENT_URL"] = "http://127.0.0.1:18002"
    env["VERIFIER_AGENT_URL"] = "http://127.0.0.1:18003"
    env["RCA_AGENT_URL"] = "http://127.0.0.1:18004"

    # === Case Study 데이터 설정 ===
    # 이 두 환경변수가 핵심 — MCP 서버가 Case Study 파일을 읽게 함
    env["OBSERVABILITY_LOG_FILE"] = str(log_file)
    env["ARCHITECTURE_TOPOLOGY_FILE"] = str(topo_file)

    try:
        for name, module, port in SERVICES:
            proc = start_service(name, module, port, env)
            processes.append(proc)

        incident = json.loads(incident_file.read_text(encoding="utf-8"))
        # topology_file 경로를 절대경로로 설정
        if incident.get("attachments"):
            incident["attachments"]["topology_file"] = str(topo_file)

        print(f"\nSending incident: {incident['incident_id']}")
        print(f"Service: {incident['service']}")
        print(f"Symptom: {incident['symptom']}")
        print(f"Time range: {incident['time_range']['start']} ~ {incident['time_range']['end']}")
        print(f"Trace ID: {incident.get('trace_id', 'N/A')}")
        print()

        response = httpx.post(
            "http://127.0.0.1:18000/analyze",
            json=incident,
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()

        # 결과 출력
        print(f"\n{'=' * 60}")
        print(f"  CASE STUDY RESULT — {config['description']}")
        print(f"{'=' * 60}")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 결과 파일 저장
        result_file = PROJECT_ROOT / f"case_study_result_case{case_num}.json"
        result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n결과 저장: {result_file}")

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