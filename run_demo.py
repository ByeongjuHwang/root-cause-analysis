#!/usr/bin/env python3
"""
run_demo.py — Quick demonstration of the multi-agent RCA framework.

Usage:
    python run_demo.py             # human-readable output
    python run_demo.py --json      # pure JSON to stdout (for E2E tests)
    python run_demo.py --quiet     # suppress agent logs
"""

from dotenv import load_dotenv
load_dotenv()

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
PORT_BASE = 18000


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_health(name: str, port: int, proc: subprocess.Popen, quiet: bool = False) -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited before becoming healthy")
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                if not quiet:
                    print(f"  [OK] {name} on :{port}", file=sys.stderr)
                return
        except Exception:
            pass
        time.sleep(0.4)
    raise RuntimeError(f"{name} timed out on startup")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", dest="json_mode",
                        help="Output pure JSON to stdout")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress agent subprocess output")
    args = parser.parse_args()

    json_mode = args.json_mode
    quiet = args.quiet or json_mode

    services = [
        ("orchestrator", "agents.orchestrator.main", PORT_BASE),
        ("log-agent", "agents.log_agent.main", PORT_BASE + 1),
        ("topology-agent", "agents.topology_agent.main", PORT_BASE + 2),
        ("verifier-agent", "agents.verifier_agent.main", PORT_BASE + 3),
        ("rca-agent", "agents.rca_agent.main", PORT_BASE + 4),
    ]

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
    env["LOG_AGENT_URL"] = f"http://127.0.0.1:{PORT_BASE + 1}"
    env["TOPOLOGY_AGENT_URL"] = f"http://127.0.0.1:{PORT_BASE + 2}"
    env["VERIFIER_AGENT_URL"] = f"http://127.0.0.1:{PORT_BASE + 3}"
    env["RCA_AGENT_URL"] = f"http://127.0.0.1:{PORT_BASE + 4}"

    incident_path = PROJECT_ROOT / "demo_incident.json"
    incident = json.loads(incident_path.read_text(encoding="utf-8"))

    processes = []
    stdout_redir = subprocess.DEVNULL if quiet else None
    stderr_redir = subprocess.DEVNULL if quiet else None

    try:
        if not json_mode:
            print("Starting agents...", file=sys.stderr)

        for name, module, port in services:
            if is_port_in_use(port):
                raise RuntimeError(f"Port {port} already in use for {name}")
            proc = subprocess.Popen(
                [PYTHON, "-m", module],
                cwd=PROJECT_ROOT,
                env={**env, "PORT": str(port), "PYTHONUNBUFFERED": "1"},
                stdout=stdout_redir, stderr=stderr_redir, text=True,
            )
            wait_for_health(name, port, proc, quiet=quiet)
            processes.append(proc)

        if not json_mode:
            print("\nSending incident to orchestrator...", file=sys.stderr)

        response = httpx.post(
            f"http://127.0.0.1:{PORT_BASE}/analyze",
            json=incident,
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()

        # Output
        print(json.dumps(result, indent=2, ensure_ascii=False))

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
    main()
