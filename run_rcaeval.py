#!/usr/bin/env python3
"""
run_rcaeval.py — RCAEval RE2-OB/TT/SS benchmark runner.
"""

# [TT-PATCH] Force UTF-8 stdout/stderr — Windows cp949 console fails on '+' symbols etc.
import sys as _sys, os as _os
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from experiment_core import (
    SYSTEMS,
    resolve_services,
    build_env_overrides,
    evaluate_prediction,
    PROJECT_ROOT,
)

PYTHON = sys.executable
STARTUP_TIMEOUT = 30.0
ANALYZE_TIMEOUT = 120.0
RESULTS_DIR = PROJECT_ROOT / "experiments" / "rcaeval"

# [TT-PATCH] Dataset-aware topology file mapping.
# Add new datasets here as they're supported.
TOPOLOGY_FILES = {
    "ob": PROJECT_ROOT / "onlineboutique_topology.json",
    "tt": PROJECT_ROOT / "trainticket_topology.json",
    "ss": PROJECT_ROOT / "sockshop_topology.json",  # placeholder for future
}

# Dataset → (entry_point_service, diagram_uri)
# entry_point_service: the user-visible "front door" of the system used in
# convert_case() as the incident.service. For OB: 'frontend'. For TT we use
# 'ts-preserve-service' which is a primary booking entry per RE2-TT traces.
DATASET_DEFAULTS = {
    "ob": {
        "entry_point": "frontend",
        "diagram_uri": "arch://online-boutique/latest",
        "benchmark_label": "rcaeval-re2-ob",
    },
    "tt": {
        "entry_point": "ts-preserve-service",
        "diagram_uri": "arch://train-ticket/latest",
        "benchmark_label": "rcaeval-re2-tt",
    },
    "ss": {
        "entry_point": "front-end",
        "diagram_uri": "arch://sock-shop/latest",
        "benchmark_label": "rcaeval-re2-ss",
    },
}

# Backward-compat alias (some old code paths may reference this name)
ONLINEBOUTIQUE_TOPOLOGY = TOPOLOGY_FILES["ob"]


def get_topology_file(dataset: str) -> Path:
    """Resolve dataset key to the topology JSON path. Raises if missing."""
    if dataset not in TOPOLOGY_FILES:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(TOPOLOGY_FILES)}")
    path = TOPOLOGY_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"Topology file missing for dataset '{dataset}': {path}\n"
            f"Generate it with build_trainticket_topology.py (TT) or place it at the project root."
        )
    return path

# Analysis window: RCAEval inject_time은 순간(instant)이므로 그 주위로 윈도우를
# 잡아야 서비스별 통계를 산출할 수 있다. v6부터는 비대칭 두 창 설계로 전환:
#
#   baseline window: [t - BASELINE_START_SEC, t - BASELINE_END_SEC]
#                    e.g. [t-600s, t-120s] = 주입 10분 전부터 2분 전까지의 정상 기준선
#   incident window: [t - INCIDENT_PRE_SEC, t + INCIDENT_POST_SEC]
#                    e.g. [t-120s, t+30s]  = 주입 2분 전부터 30초 후까지의 이상 구간
#
# baseline과 incident 사이에 2분 간격을 두는 이유는 주입 직전에 이미 이상
# 증상이 로그로 흘러나올 수 있어 baseline을 오염시키기 때문. incident를
# 주입 2분 전부터 시작하는 이유는 실제 장애 관찰이 주입 지연으로 약간 늦게
# 나타나는 경우를 커버하고, 주입 직전에 이미 시작된 리트라이 로그 등을 잡기 위함.
INCIDENT_PRE_SEC = int(os.getenv("INCIDENT_PRE_SEC", "120"))     # t-120s
INCIDENT_POST_SEC = int(os.getenv("INCIDENT_POST_SEC", "30"))    # t+30s
BASELINE_START_SEC = int(os.getenv("BASELINE_START_SEC", "600"))  # t-10min
BASELINE_END_SEC = int(os.getenv("BASELINE_END_SEC", "120"))      # t-2min

# Legacy symmetric window (kept for backward compat with older call sites)
ANALYSIS_WINDOW_SECONDS = int(os.getenv("RCAEVAL_ANALYSIS_WINDOW", "300"))


def _fmt_iso(dt, use_z: bool) -> str:
    """Format datetime to ISO 8601, preserving Z suffix style if input used it."""
    from datetime import timezone
    if use_z:
        return (dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
                + f"{dt.microsecond // 1000:03d}Z")
    return dt.isoformat()


def _asymmetric_windows(inject_iso: str) -> Optional[Dict[str, Tuple[str, str]]]:
    """Compute baseline and incident windows around inject_time.

    Returns:
        {
          "baseline": (start_iso, end_iso),   # [t - BASELINE_START_SEC, t - BASELINE_END_SEC]
          "incident": (start_iso, end_iso),   # [t - INCIDENT_PRE_SEC,   t + INCIDENT_POST_SEC]
          "outer":    (start_iso, end_iso),   # envelope covering both windows
        }
        or None if inject_iso is unparsable.
    """
    if not inject_iso:
        return None
    try:
        from datetime import datetime, timedelta
        dt = datetime.fromisoformat(inject_iso.replace("Z", "+00:00"))
        use_z = inject_iso.endswith("Z")

        baseline_s = dt - timedelta(seconds=BASELINE_START_SEC)
        baseline_e = dt - timedelta(seconds=BASELINE_END_SEC)
        incident_s = dt - timedelta(seconds=INCIDENT_PRE_SEC)
        incident_e = dt + timedelta(seconds=INCIDENT_POST_SEC)
        outer_s = min(baseline_s, incident_s)
        outer_e = max(baseline_e, incident_e)

        return {
            "baseline": (_fmt_iso(baseline_s, use_z), _fmt_iso(baseline_e, use_z)),
            "incident": (_fmt_iso(incident_s, use_z), _fmt_iso(incident_e, use_z)),
            "outer":    (_fmt_iso(outer_s, use_z),    _fmt_iso(outer_e, use_z)),
        }
    except Exception:
        return None


def _analysis_window(inject_iso: str, window_seconds: int = 300) -> Tuple[str, str]:
    """Legacy symmetric window. Kept for any other caller that imports this.

    For RCAEval benchmark use, prefer _asymmetric_windows().
    """
    if not inject_iso:
        return inject_iso, inject_iso
    try:
        from datetime import datetime, timedelta, timezone
        dt = datetime.fromisoformat(inject_iso.replace("Z", "+00:00"))
        start_dt = dt - timedelta(seconds=window_seconds)
        end_dt = dt + timedelta(seconds=window_seconds)
        use_z = inject_iso.endswith("Z")
        return _fmt_iso(start_dt, use_z), _fmt_iso(end_dt, use_z)
    except Exception:
        return inject_iso, inject_iso


# =============================================================================
# Case discovery and conversion
# =============================================================================

def parse_case_name(case_name: str) -> Optional[Dict[str, Any]]:
    """Parse 'cartservice_cpu_1' → {service, fault, instance}."""
    parts = case_name.split("_")
    if len(parts) < 3:
        return None
    try:
        instance = int(parts[-1])
    except ValueError:
        return None
    return {
        "service": "_".join(parts[:-2]),
        "fault": parts[-2],
        "instance": instance,
        "name": case_name,
    }


def discover_cases(data_dir: Path) -> List[Dict[str, Any]]:
    """Discover all valid cases from the RCAEval data directory.

    Handles BOTH layouts:

    (A) Original RCAEval RE2-OB layout (as downloaded):
            <data-dir>/
                checkoutservice_cpu/
                    1/logs.csv
                    2/logs.csv
                    3/logs.csv
                cartservice_mem/
                    1/logs.csv
                    ...

    (B) Flattened layout (post-processing convenience):
            <data-dir>/
                checkoutservice_cpu_1/logs.csv
                checkoutservice_cpu_2/logs.csv
                cartservice_mem_1/logs.csv
                ...

    Returns cases in sorted order (service, fault, instance).
    """
    cases: List[Dict[str, Any]] = []

    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue

        # Layout A: nested — sub is '{service}_{fault}', children are instance dirs
        parts_outer = sub.name.split("_")
        if len(parts_outer) >= 2:
            service_outer = "_".join(parts_outer[:-1])
            fault_outer = parts_outer[-1]
            instance_dirs = [
                d for d in sorted(sub.iterdir())
                if d.is_dir() and (d / "logs.csv").exists() and d.name.isdigit()
            ]
            if instance_dirs:
                for inst_dir in instance_dirs:
                    cases.append({
                        "service": service_outer,
                        "fault": fault_outer,
                        "instance": int(inst_dir.name),
                        "name": f"{service_outer}_{fault_outer}_{inst_dir.name}",
                        "path": inst_dir,
                    })
                continue  # this sub was a nested case folder — don't try flat parse

        # Layout B: flattened — sub IS the instance folder
        if (sub / "logs.csv").exists():
            parsed = parse_case_name(sub.name)
            if parsed is not None:
                parsed["path"] = sub
                cases.append(parsed)

    # Stable sort: by (service, fault, instance)
    cases.sort(key=lambda c: (c["service"], c["fault"], c["instance"]))
    return cases


def filter_cases(
    cases: List[Dict[str, Any]],
    fault_types: Optional[List[str]] = None,
    services: Optional[List[str]] = None,
    first_only: bool = False,
) -> List[Dict[str, Any]]:
    if fault_types:
        cases = [c for c in cases if c["fault"] in fault_types]
    if services:
        cases = [c for c in cases if c["service"] in services]
    if first_only:
        seen = set()
        out = []
        for c in sorted(cases, key=lambda x: (x["service"], x["fault"], x["instance"])):
            key = (c["service"], c["fault"])
            if key not in seen:
                seen.add(key)
                out.append(c)
        cases = out
    return cases


def convert_case(case: Dict[str, Any], work_dir: Path,
                 dataset: str = "ob") -> Optional[Dict[str, Any]]:
    """Convert RCAEval case to framework input format.

    Delegates to convert_rcaeval.py which reads logs.csv / inject_time.txt
    and emits logs.jsonl + meta.json.

    [TT-PATCH] dataset parameter selects entry_point / topology / diagram_uri.
    """
    logs_path = work_dir / f"{case['name']}.jsonl"
    completed = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "convert_rcaeval.py"),
         str(case["path"]), str(logs_path)],
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        print(f"  [!] convert failed for {case['name']}: {completed.stderr[:200]}",
              file=sys.stderr)
        return None

    meta_path = logs_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # RCAEval convention (v6): use asymmetric dual-window around inject_time.
    #   baseline: [t-10min, t-2min] — clean normal period (not polluted by pre-fault signals)
    #   incident: [t-2min, t+30s]   — anomaly period (covers early-onset retries too)
    # `time_range` is the outer envelope so any other agent that treats it as
    # a single window still gets a valid range. Log Agent reads the precise
    # windows from `attachments.baseline_range` and `attachments.incident_range`.
    inject_iso = meta.get("inject_time_iso", "")
    windows = _asymmetric_windows(inject_iso)
    if windows is None:
        # Fallback to legacy symmetric window so the pipeline never crashes
        win_start, win_end = _analysis_window(inject_iso, ANALYSIS_WINDOW_SECONDS)
        baseline_range = None
        incident_range = None
    else:
        win_start, win_end = windows["outer"]
        baseline_range = list(windows["baseline"])
        incident_range = list(windows["incident"])

    # v8: RCAEval provides a metrics.csv alongside logs.csv in each case folder.
    # We pass the path through attachments so Log Agent / Verifier can call
    # the metric MCP tools without having to re-derive the path. If absent
    # (e.g. for bench runs that don't ship metrics), downstream tools
    # gracefully return has_data=False.
    metrics_csv = case["path"] / "metrics.csv"
    metrics_file = str(metrics_csv) if metrics_csv.exists() else None

    # [TT-PATCH] Dataset-aware entry point and topology.
    ds_config = DATASET_DEFAULTS.get(dataset, DATASET_DEFAULTS["ob"])
    topology_path = get_topology_file(dataset)

    incident = {
        "incident_id": f"INC-{case['name'].upper()}",
        "service": ds_config["entry_point"],   # entry-point service (varies by dataset)
        "time_range": {"start": win_start, "end": win_end},
        "symptom": f"Service {case['service']} experiencing {case['fault']} fault",
        "trace_id": None,
        "attachments": {
            "log_file": str(logs_path),
            "topology_file": str(topology_path),
            "diagram_uri": ds_config["diagram_uri"],
            "inject_time": inject_iso,
            # Dual windows (v6) — Log Agent uses these for baseline vs incident stats
            "baseline_range": baseline_range,
            "incident_range": incident_range,
            # v8: metric source (RCAEval metrics.csv) — None if not shipped
            "metrics_file": metrics_file,
        },
    }
    return {"logs_path": logs_path, "incident": incident, "meta": meta}


# =============================================================================
# Service lifecycle (shares resolve_services with run_experiment.py)
# =============================================================================

def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_health(name: str, port: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited early")
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.4)
    raise RuntimeError(f"{name} timed out on startup")


def _start_system(system_key: str, log_file: Path, topology_file: Path):
    """Start all services for a system with appropriate env vars."""
    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + pypath if pypath else "")
    env["OBSERVABILITY_LOG_FILE"] = str(log_file)
    env["ARCHITECTURE_TOPOLOGY_FILE"] = str(topology_file)
    env.update(build_env_overrides(system_key))

    # >>> ADAPTIVE-FIX: enable A2A contracts and adaptive re-invocation.
    # Without these, Orchestrator._run_adaptive_iterations stops at the first
    # iteration with action="stop" (reason="no_contract_available").
    # External overrides win (e.g. ADAPTIVE_MAX_ITERATIONS=0 for ablation).
    env["A2A_CONTRACT_MODE"] = os.environ.get("A2A_CONTRACT_MODE", "on")
    env["ADAPTIVE_THRESHOLD"] = os.environ.get("ADAPTIVE_THRESHOLD", "0.5")
    env["ADAPTIVE_MAX_ITERATIONS"] = os.environ.get("ADAPTIVE_MAX_ITERATIONS", "3")
    # <<< ADAPTIVE-FIX

    processes = []
    for name, module, port in resolve_services(system_key):
        if _is_port_in_use(port):
            raise RuntimeError(f"Port {port} in use")
        proc = subprocess.Popen(
            [PYTHON, "-m", module], cwd=PROJECT_ROOT,
            env={**env, "PORT": str(port), "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _wait_for_health(name, port, proc)
        processes.append((proc, name, port))
    return processes


def _stop_system(processes) -> None:
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
# Single experiment (uses pre-started persistent agents)
# =============================================================================

def run_one_persistent(
    system_key: str,
    case: Dict[str, Any],
    converted: Dict[str, Any],
    results_dir: Path,
    dataset: str = "ob",
) -> Dict[str, Any]:
    """Run a single case against ALREADY-RUNNING persistent agents.

    Unlike the legacy run_one() which starts/stops 5 agents per case (causing
    Windows TIME_WAIT port exhaustion on consecutive runs), this version sends
    the request to the orchestrator with log_file embedded in attachments.
    The orchestrator forwards log_file to the Log Agent, which resolves it via
    _resolve_log_file() — so no environment variable or agent restart needed.

    Required: agents must be started with _start_system_persistent() before
    the first call and stopped with _stop_system() after the last call.
    """
    benchmark_label = DATASET_DEFAULTS.get(dataset, DATASET_DEFAULTS["ob"])["benchmark_label"]
    port_base = SYSTEMS[system_key]["port_base"]
    start = time.time()

    try:
        # Send analyze request — log_file is already in incident.attachments
        resp = httpx.post(
            f"http://127.0.0.1:{port_base}/analyze",
            json=converted["incident"],
            timeout=ANALYZE_TIMEOUT,
        )
        resp.raise_for_status()
        rca_result = resp.json()
        elapsed = time.time() - start

        ground_truth = {
            "ground_truth_root_cause": case["service"],
            "ground_truth_path": [],
        }
        evaluation = evaluate_prediction(
            rca_result, ground_truth, elapsed_seconds=elapsed,
        )
        evaluation["gt_fault"] = case["fault"]

        output = {
            "system": system_key,
            "benchmark": benchmark_label,
            "case": case["name"],
            "gt_service": case["service"],
            "gt_fault": case["fault"],
            "elapsed_seconds": round(elapsed, 2),
            "evaluation": evaluation,
            "result": rca_result,
        }

        out_file = results_dir / f"{system_key}_{case['name']}.json"
        out_file.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output

    except Exception as exc:
        return {
            "system": system_key,
            "benchmark": benchmark_label,
            "case": case["name"],
            "gt_service": case["service"],
            "gt_fault": case["fault"],
            "elapsed_seconds": round(time.time() - start, 2),
            "error": str(exc),
        }


def _start_system_persistent(system_key: str, topology_file: Path):
    """Start all agents ONCE for the entire benchmark run.

    Unlike _start_system(), this does NOT set OBSERVABILITY_LOG_FILE — each
    case's log_file is passed via the HTTP request body (incident.attachments).
    Agents use _resolve_log_file() which checks the function parameter first,
    so the per-case log_file wins over any absent env var.
    """
    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + pypath if pypath else "")
    env["ARCHITECTURE_TOPOLOGY_FILE"] = str(topology_file)
    env.update(build_env_overrides(system_key))

    # >>> ADAPTIVE-FIX: enable A2A contracts and adaptive re-invocation.
    env["A2A_CONTRACT_MODE"] = os.environ.get("A2A_CONTRACT_MODE", "on")
    env["ADAPTIVE_THRESHOLD"] = os.environ.get("ADAPTIVE_THRESHOLD", "0.5")
    env["ADAPTIVE_MAX_ITERATIONS"] = os.environ.get("ADAPTIVE_MAX_ITERATIONS", "3")
    # <<< ADAPTIVE-FIX

    processes = []
    for name, module, port in resolve_services(system_key):
        if _is_port_in_use(port):
            raise RuntimeError(f"Port {port} in use at startup")
        proc = subprocess.Popen(
            [PYTHON, "-m", module], cwd=PROJECT_ROOT,
            env={**env, "PORT": str(port), "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _wait_for_health(name, port, proc)
        processes.append((proc, name, port))
    return processes


# =============================================================================
# Legacy single-case runner (kept for backward compat; restarts agents each call)
# =============================================================================

def run_one(
    system_key: str,
    case: Dict[str, Any],
    converted: Dict[str, Any],
    results_dir: Path,
    dataset: str = "ob",
) -> Dict[str, Any]:
    """Legacy: start agents, run one case, stop agents. Use run_one_persistent()
    for batch runs to avoid Windows TIME_WAIT port exhaustion.
    """
    benchmark_label = DATASET_DEFAULTS.get(dataset, DATASET_DEFAULTS["ob"])["benchmark_label"]
    topology_path = get_topology_file(dataset)
    port_base = SYSTEMS[system_key]["port_base"]
    start = time.time()
    processes = []

    try:
        processes = _start_system(
            system_key, converted["logs_path"], topology_path,
        )

        resp = httpx.post(
            f"http://127.0.0.1:{port_base}/analyze",
            json=converted["incident"],
            timeout=ANALYZE_TIMEOUT,
        )
        resp.raise_for_status()
        rca_result = resp.json()
        elapsed = time.time() - start

        # Build ground truth in the experiment_core format
        ground_truth = {
            "ground_truth_root_cause": case["service"],
            "ground_truth_path": [],  # RCAEval doesn't ship path ground truth
        }

        evaluation = evaluate_prediction(
            rca_result, ground_truth, elapsed_seconds=elapsed,
        )
        # Add RCAEval-specific context
        evaluation["gt_fault"] = case["fault"]

        output = {
            "system": system_key,
            "benchmark": benchmark_label,
            "case": case["name"],
            "gt_service": case["service"],
            "gt_fault": case["fault"],
            "elapsed_seconds": round(elapsed, 2),
            "evaluation": evaluation,
            "result": rca_result,
        }

        out_file = results_dir / f"{system_key}_{case['name']}.json"
        out_file.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output

    except Exception as exc:
        return {
            "system": system_key,
            "benchmark": benchmark_label,
            "case": case["name"],
            "gt_service": case["service"],
            "gt_fault": case["fault"],
            "elapsed_seconds": round(time.time() - start, 2),
            "error": str(exc),
        }
    finally:
        _stop_system(processes)


# =============================================================================
# Aggregation and export
# =============================================================================

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_system: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if "error" in r:
            continue
        sk = r["system"]
        ev = r.get("evaluation", {})
        s = by_system.setdefault(sk, {
            "total": 0, "ac1": 0, "ac3": 0, "elapsed": [],
            "by_fault": {},
        })
        s["total"] += 1
        if ev.get("ac_at_1"):
            s["ac1"] += 1
        if ev.get("ac_at_3"):
            s["ac3"] += 1
        s["elapsed"].append(r.get("elapsed_seconds", 0))

        fault = r.get("gt_fault", "unknown")
        fs = s["by_fault"].setdefault(fault, {"total": 0, "ac1": 0, "ac3": 0})
        fs["total"] += 1
        if ev.get("ac_at_1"):
            fs["ac1"] += 1
        if ev.get("ac_at_3"):
            fs["ac3"] += 1

    for sk, s in by_system.items():
        t = max(s["total"], 1)
        s["ac1_rate"] = round(s["ac1"] / t, 3)
        s["ac3_rate"] = round(s["ac3"] / t, 3)
        s["avg_elapsed"] = round(sum(s["elapsed"]) / t, 2)
        for f, fs in s["by_fault"].items():
            fs["ac1_rate"] = round(fs["ac1"] / max(fs["total"], 1), 3)
            fs["ac3_rate"] = round(fs["ac3"] / max(fs["total"], 1), 3)

    return by_system


def export_csv(results: List[Dict[str, Any]], path: Path) -> None:
    """Export per-case results as CSV for downstream analysis."""
    fields = [
        "system", "case", "gt_service", "gt_fault",
        "ac_at_1", "ac_at_3", "top_cause_service", "top_confidence",
        "elapsed_seconds", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            ev = r.get("evaluation", {})
            w.writerow({
                "system": r.get("system"),
                "case": r.get("case"),
                "gt_service": r.get("gt_service"),
                "gt_fault": r.get("gt_fault"),
                "ac_at_1": ev.get("ac_at_1"),
                "ac_at_3": ev.get("ac_at_3"),
                "top_cause_service": ev.get("predicted_top_cause_service"),
                "top_confidence": ev.get("top_confidence"),
                "elapsed_seconds": r.get("elapsed_seconds"),
                "error": r.get("error"),
            })


def print_summary(summary: Dict[str, Dict[str, Any]], n_results: int, n_errors: int,
                  dataset: str = "ob") -> None:
    label = DATASET_DEFAULTS.get(dataset, DATASET_DEFAULTS["ob"])["benchmark_label"].upper()
    print("\n" + "=" * 80)
    print(f"  {label} SUMMARY")
    print("=" * 80)
    for sk in ("ours", "b1", "b2", "b3"):
        s = summary.get(sk)
        if not s:
            continue
        print(f"\n  {sk}: AC@1={s['ac1']}/{s['total']} ({s['ac1_rate']:.1%})  "
              f"AC@3={s['ac3']}/{s['total']} ({s['ac3_rate']:.1%})  "
              f"Avg={s['avg_elapsed']:.1f}s")
        for f, fs in sorted(s["by_fault"].items()):
            print(f"    {f:<10} AC@1={fs['ac1']:>2}/{fs['total']:<2} ({fs['ac1_rate']:.1%})  "
                  f"AC@3={fs['ac3']:>2}/{fs['total']:<2} ({fs['ac3_rate']:.1%})")
    if n_errors:
        print(f"\n  ERRORS: {n_errors}/{n_results}")
    print("=" * 80)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RCAEval RE2 benchmark runner (OB / TT / SS)")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory containing RCAEval case folders")
    parser.add_argument("--dataset", default="ob",
                        choices=sorted(TOPOLOGY_FILES.keys()),
                        help="Dataset key: ob (Online Boutique) | tt (Train Ticket) | ss (Sock Shop)")
    parser.add_argument("--systems", default="ours,b1,b2,b3",
                        help="Comma-separated system keys")
    parser.add_argument("--fault-types", default=None,
                        help="Comma-separated fault types (cpu,mem,delay,...)")
    parser.add_argument("--first-only", action="store_true",
                        help="Use only first instance per (service, fault) pair")
    parser.add_argument("--csv", action="store_true", help="Export CSV")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    args = parser.parse_args()

    # [TT-PATCH] Validate topology file exists for the chosen dataset
    try:
        topology_path = get_topology_file(args.dataset)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    print(f"Dataset: {args.dataset}  |  Topology: {topology_path.name}")

    systems = [s.strip() for s in args.systems.split(",")]
    invalid = [s for s in systems if s not in SYSTEMS]
    if invalid:
        parser.error(f"Unknown system(s): {invalid}")

    fault_types = (
        [f.strip() for f in args.fault_types.split(",")] if args.fault_types else None
    )

    # Discover
    cases = discover_cases(args.data_dir)
    cases = filter_cases(cases, fault_types=fault_types, first_only=args.first_only)
    print(f"Discovered {len(cases)} cases, {len(systems)} systems → "
          f"{len(cases) * len(systems)} experiments")

    if args.dry_run:
        for s in systems:
            for c in cases[:5]:
                print(f"  {s} × {c['name']}")
            if len(cases) > 5:
                print(f"  ... and {len(cases) - 5} more for {s}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = RESULTS_DIR / "_work"
    work_dir.mkdir(exist_ok=True)

    # Convert
    print("\nConverting cases...")
    converted: Dict[str, Dict[str, Any]] = {}
    for c in cases:
        cv = convert_case(c, work_dir, dataset=args.dataset)   # [TT-PATCH]
        if cv:
            converted[c["name"]] = cv
    print(f"  {len(converted)}/{len(cases)} cases converted")

    # Run
    all_results: List[Dict[str, Any]] = []
    overall_start = time.time()

    # Group plan by system so we start each system's agents once.
    plan: Dict[str, List[Dict[str, Any]]] = {}
    for sk in systems:
        plan[sk] = [c for c in cases if c["name"] in converted]
    total = sum(len(v) for v in plan.values())

    completed = 0
    for sk, case_list in plan.items():
        if not case_list:
            continue
        print(f"\n=== Starting agents for system: {sk} ===")
        processes = _start_system_persistent(sk, topology_path)   # [TT-PATCH]
        print(f"  [OK] {len(processes)} agents up for {sk}")

        try:
            for c in case_list:
                completed += 1
                cv = converted[c["name"]]
                print(f"  [{completed}/{total}] {sk:<5} × {c['name']}...",
                      end=" ", flush=True)
                r = run_one_persistent(sk, c, cv, RESULTS_DIR, dataset=args.dataset)   # [TT-PATCH]
                ev = r.get("evaluation", {})
                status = "OK" if ev.get("ac_at_1") else ("FAIL" if "error" not in r else "ERR")
                print(f"{status} ({r.get('elapsed_seconds', 0):.1f}s)")
                all_results.append(r)
        finally:
            print(f"\n=== Stopping agents for system: {sk} ===")
            _stop_system(processes)

    total_elapsed = time.time() - overall_start
    summary = aggregate(all_results)
    n_errors = sum(1 for r in all_results if "error" in r)
    print_summary(summary, len(all_results), n_errors, dataset=args.dataset)   # [TT-PATCH]
    print(f"\nTotal: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_label = DATASET_DEFAULTS[args.dataset]["benchmark_label"]
    summary_file = RESULTS_DIR / f"rcaeval_summary_{args.dataset}_{ts}.json"
    summary_file.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "benchmark": benchmark_label,
        "dataset": args.dataset,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "summary": summary,
        "results": [
            {k: v for k, v in r.items() if k != "result"} for r in all_results
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary: {summary_file}")

    if args.csv:
        csv_file = RESULTS_DIR / f"rcaeval_results_{args.dataset}_{ts}.csv"
        export_csv(all_results, csv_file)
        print(f"CSV: {csv_file}")


if __name__ == "__main__":
    main()
