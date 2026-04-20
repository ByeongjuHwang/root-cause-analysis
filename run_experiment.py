#!/usr/bin/env python3
"""
run_experiment.py — 단일 시나리오를 단일 시스템 변형으로 실행.

본 스크립트는 실험 프레임워크의 CLI 진입점이며, 실제 데이터 정의
(SYSTEMS, SCENARIOS)와 평가 로직(evaluate_prediction)은 experiment_core.py에
통합되어 있다. run_rcaeval.py, run_case_study.py도 동일 코어를 사용한다.

사용법:
    python run_experiment.py --system ours --scenario s1
    python run_experiment.py --system b1   --scenario s3
    python run_experiment.py --system ours --scenario case1
    python run_experiment.py --system ours --scenario s1 --json --quiet

시스템 변형:   ours, b1, b2, b3 (experiment_core.SYSTEMS 참조)
시나리오:       s1~s8, case1, case2 (experiment_core.SCENARIOS 참조)
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
from typing import Any, Dict

import httpx

from experiment_core import (
    SYSTEMS,
    SCENARIOS,
    resolve_services,
    build_env_overrides,
    resolve_scenario_inputs,
    scenario_ground_truth,
    evaluate_prediction,
    PROJECT_ROOT,
)

PYTHON = sys.executable
STARTUP_TIMEOUT = 30.0
ANALYZE_TIMEOUT = 120.0
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# =============================================================================
# Service lifecycle
# =============================================================================

def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_health(name: str, port: int, proc: subprocess.Popen, quiet: bool) -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited before becoming healthy (exit={proc.returncode})")
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                if not quiet:
                    print(f"  [OK] {name} on :{port}", file=sys.stderr)
                return
        except Exception:
            pass
        time.sleep(0.4)
    raise RuntimeError(f"{name} timed out on startup (>{STARTUP_TIMEOUT}s)")


def _start_services(system_key: str, env: Dict[str, str], quiet: bool):
    """Start all services for a system. Returns list of (popen, name, port)."""
    services = resolve_services(system_key)
    stdout_target = subprocess.DEVNULL if quiet else None
    stderr_target = subprocess.DEVNULL if quiet else None

    processes = []
    for name, module, port in services:
        if _is_port_in_use(port):
            raise RuntimeError(f"Port {port} already in use for {name}")
        proc = subprocess.Popen(
            [PYTHON, "-m", module],
            cwd=PROJECT_ROOT,
            env={**env, "PORT": str(port), "PYTHONUNBUFFERED": "1"},
            stdout=stdout_target, stderr=stderr_target, text=True,
        )
        _wait_for_health(name, port, proc, quiet=quiet)
        processes.append((proc, name, port))
    return processes


def _stop_services(processes) -> None:
    for proc, _, _ in processes:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                proc.terminate()
    for proc, _, _ in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# =============================================================================
# Main execution
# =============================================================================

def run_single(
    system_key: str,
    scenario_key: str,
    json_mode: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run a single (system, scenario) experiment.

    Signature preserved for backward compatibility with external callers
    (run_all_experiments.py, external scripts).

    Args:
        system_key:   one of 'ours', 'b1', 'b2', 'b3'
        scenario_key: one of 's1'~'s8', 'case1', 'case2'
        json_mode:    if True, emit pure JSON to stdout (suitable for piping)
        quiet:        if True, suppress all agent subprocess output

    Returns:
        JSON-serializable experiment record dict.
    """
    system = SYSTEMS[system_key]
    scenario = SCENARIOS[scenario_key]

    if not json_mode:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  System: {system['name']}", file=sys.stderr)
        print(f"  Scenario: {scenario['description']}", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)

    # Resolve scenario inputs (incident, log file, topology file)
    inputs = resolve_scenario_inputs(scenario_key, project_root=PROJECT_ROOT)
    incident = inputs["incident"]

    # Environment
    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + pypath if pypath else "")
    env.update(build_env_overrides(system_key))

    if inputs["log_file"]:
        env["OBSERVABILITY_LOG_FILE"] = str(inputs["log_file"])
    if inputs["topology_file"]:
        env["ARCHITECTURE_TOPOLOGY_FILE"] = str(inputs["topology_file"])

    port_base = system["port_base"]
    start_time = time.time()
    processes = []

    try:
        processes = _start_services(system_key, env, quiet=(quiet or json_mode))

        analyze_url = f"http://127.0.0.1:{port_base}/analyze"
        response = httpx.post(analyze_url, json=incident, timeout=ANALYZE_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time

        # Unified evaluation (same function as run_rcaeval.py and case study)
        gt = scenario_ground_truth(scenario_key)
        evaluation = evaluate_prediction(result, gt, elapsed_seconds=elapsed)

        output = {
            "system": system_key,
            "system_name": system["name"],
            "scenario": scenario_key,
            "scenario_description": scenario["description"],
            "scenario_kind": scenario.get("kind", "synthetic"),
            "elapsed_seconds": round(elapsed, 2),
            "evaluation": evaluation,
            "result": result,
        }

        # Persist
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = EXPERIMENTS_DIR / f"{system_key}_{scenario_key}.json"
        out_file.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if json_mode:
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(f"\n  결과 저장: {out_file}", file=sys.stderr)
            print(f"  소요 시간: {elapsed:.1f}초", file=sys.stderr)
            print(
                "  평가: " + json.dumps(evaluation, indent=2, ensure_ascii=False),
                file=sys.stderr,
            )

        return output

    finally:
        _stop_services(processes)


# =============================================================================
# Backward-compat wrapper (external scripts may import this symbol)
# =============================================================================

def evaluate_result(
    result: Dict[str, Any],
    scenario: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    """Backward-compatible wrapper around experiment_core.evaluate_prediction.

    The original signature (result, scenario, elapsed) is preserved.
    Scenario dict is expected to contain the ground-truth fields
    (ground_truth_root_cause, ground_truth_path, etc.) — same shape as
    entries in SCENARIOS.

    New code should call experiment_core.evaluate_prediction directly.
    """
    return evaluate_prediction(result, scenario, elapsed_seconds=elapsed)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="RCA Experiment Runner (single system × single scenario)"
    )
    parser.add_argument("--system", required=True, choices=list(SYSTEMS.keys()))
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    parser.add_argument(
        "--json", action="store_true", dest="json_mode",
        help="Emit JSON to stdout (stderr is still used for diagnostics)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress agent subprocess stdout/stderr",
    )
    # Deprecated (scenario kind is auto-detected)
    parser.add_argument("--case-study", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    run_single(
        system_key=args.system,
        scenario_key=args.scenario,
        json_mode=args.json_mode,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
