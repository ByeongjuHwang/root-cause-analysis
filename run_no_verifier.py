"""
run_no_verifier.py — B3 (No-Verifier Pipeline) legacy standalone runner.

⚠️ DEPRECATED in favor of the unified experiment pipeline:

    python run_experiment.py --system b3 --scenario s1

Retained for backward compatibility and for manual probing outside
the main experiment harness.

본 프레임워크와 동일하되 교차 검증 에이전트를 제거한 변형체.
Verifier 에이전트 프로세스를 아예 기동하지 않음.

사용법:
    uv run run_no_verifier.py                           (기존 합성 데모)
    uv run run_no_verifier.py --incident case_study_incident.json --case-study
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

# Verifier 에이전트를 제외한 4개 서비스만 기동
SERVICES = [
    ("orchestrator-b3", "agents.orchestrator.main_no_verifier", 19000),
    ("log-agent", "agents.log_agent.main", 19001),
    ("topology-agent", "agents.topology_agent.main", 19002),
    ("rca-agent", "agents.rca_agent.main", 19004),
    # verifier-agent 제외됨
]


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def ensure_port_available(name: str, port: int) -> None:
    if is_port_in_use(port):
        raise RuntimeError(f"Cannot start {name}: port {port} already in use.")


def wait_for_health(name: str, port: int, proc: subprocess.Popen, timeout: float = STARTUP_TIMEOUT) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited before becoming healthy on port {port}.")
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
        stdout=None, stderr=None, text=True,
    )
    wait_for_health(name, port, proc)
    print(f"[OK] {name} on port {port}")
    return proc


def main() -> int:
    # Parse arguments
    incident_file = PROJECT_ROOT / "demo_incident.json"
    is_case_study = False

    for i, arg in enumerate(sys.argv):
        if arg == "--incident" and i + 1 < len(sys.argv):
            incident_file = PROJECT_ROOT / sys.argv[i + 1]
        if arg == "--case-study":
            is_case_study = True

    if not incident_file.exists():
        print(f"ERROR: Incident file not found: {incident_file}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"  B3 Baseline: No-Verifier Pipeline")
    print(f"  Incident: {incident_file.name}")
    print(f"{'=' * 60}\n")

    processes = []
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")

    # Agent URLs (19000번대 포트 사용 — 원본과 충돌 방지)
    env["LOG_AGENT_URL"] = "http://127.0.0.1:19001"
    env["TOPOLOGY_AGENT_URL"] = "http://127.0.0.1:19002"
    env["RCA_AGENT_URL"] = "http://127.0.0.1:19004"

    # Case Study 모드
    if is_case_study:
        log_file = PROJECT_ROOT / "case_study_logs.jsonl"
        topo_file = PROJECT_ROOT / "case_study_topology.json"
        if not log_file.exists() or not topo_file.exists():
            print(f"ERROR: Case study files not found")
            return 1
        env["OBSERVABILITY_LOG_FILE"] = str(log_file)
        env["ARCHITECTURE_TOPOLOGY_FILE"] = str(topo_file)

    try:
        for name, module, port in SERVICES:
            proc = start_service(name, module, port, env)
            processes.append(proc)

        incident = json.loads(incident_file.read_text(encoding="utf-8"))
        if is_case_study and incident.get("attachments"):
            topo_file = PROJECT_ROOT / "case_study_topology.json"
            incident["attachments"]["topology_file"] = str(topo_file)

        print(f"Sending: {incident['incident_id']} -> {incident['service']}")

        response = httpx.post(
            "http://127.0.0.1:19000/analyze",
            json=incident,
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()

        print(f"\n{'=' * 60}")
        print(f"  B3 RESULT (No-Verifier)")
        print(f"{'=' * 60}")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 결과 저장
        result_file = PROJECT_ROOT / f"b3_result_{incident_file.stem}.json"
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
